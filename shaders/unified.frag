#version 140

//#undef CLUSTER_DECALS
//#define DECALS_DEBUG
#if defined(DECALS_DEBUG) || defined(LIGHTS_DEBUG)
#define CLUSTER_DEBUG
#endif

////////////////////////////////////////
// vertex shader output
in mediump vec4					pxWSPosition;		// world space position
in mediump vec2					pxTexCoord1;		// ifdef VERTEX_TEXCOORD1
in mediump vec2					pxTexCoord2;		// ifdef VERTEX_TEXCOORD2
in mediump vec3					pxNormal;			// ifdef VERTEX_NORMAL
in lowp vec4					pxColor;			// ifdef VERTEX_COLOR
in highp vec4					pxLightSpacePos;	// ifdef DYNAMIC_SHADOWED
in mediump vec4					pxMapTexCoords;		// map coords in x,y=[0..1] or z,w=[0..map size]
in highp float					pxVSDepth;			// ifdef VERTEX_OUTPUT_DEPTH
flat in int						pxInstanceID;

out vec4						outFragColor0;

////////////////////////////////////////
// textures

uniform sampler2D				inTex0;				// diffuse (or the material mask when TERRAIN is defined. When defined, the RGB are the layers blend amount and Alpha is the height)
uniform sampler2D				inTex1;				// specular mask

#ifdef TERRAIN
	uniform usampler2D			inTex2;				// terrain materials mask
	uniform sampler2DArray		inTex3;				// terrain layers array (4 layers per material)
	#ifdef TERRAIN_TEX_NORMALS
	uniform sampler2DArray		inTex4;				// terrain layer normals array (4 layers per material)
	#endif
#else
	uniform sampler2D			inTex2;				// detail texture
	uniform sampler2D			inTex3;				// scene depth (water only)
	uniform sampler2D			inTex4;				// ground bounce light texture
#endif

// always bound, for all shaders
uniform sampler2D				inTex5;				// fog of war mask
uniform sampler2D				inTex6;				// floor shadows
uniform sampler2DShadow			inTex7;				// directional shadowmap
uniform samplerCube				inTex8;				// outdoor reflection cubemap
uniform sampler2D				inTex9;				// decal atlas
uniform usampler2D				inTex10;			// cluster counts
uniform usampler2D				inTex11;			// cluster list
uniform sampler2D				inTex12;			// decal data
uniform sampler2D				inTex13;			// light data
uniform sampler2D				inTex14;			// light 2D shadows
uniform sampler2D				inTex15;			// clouds texture
uniform sampler2D				inTex16;			// for mipmaps debugging

// uniforms - global
#pragma include <data/shaders/unified_uniforms.h>

#if defined(FOG_OF_WAR) || defined(DARKEN_MAP_EDGES)
#pragma include <data/shaders/fow_common.h>
#endif

// uniforms - per instance
// x/y = ifdef PIXEL_SPECULAR, intensity in .r, power in .g
// z = ifdef LIGHTING, the ambient light color index (0 == outside, 2 == inside)
// w = nothing for now
uniform vec4	inFragInstanceData[MAX_INSTANCES];
#define 		inSpecularIntensity		inFragInstanceData[pxInstanceID].x
#define 		inSpecularPower			inFragInstanceData[pxInstanceID].y
#define 		inAmbientColorIdx		int(inFragInstanceData[pxInstanceID].z)

////////////////////////////////////////
// terrain-only
uniform mediump float			inTerrainNormalWeight;
uniform mediump float			inTerrainSharpness;

#ifdef CLUSTER_DECALS
vec3 quatTransform( vec4 q, vec3 v ){
	return v + 2.0*cross(cross(v, q.xyz ) + q.w*v, q.xyz);
}

void doDecal(inout vec4 diffuseColor, vec4 scaleOffset, vec4 invPosScale, vec4 orientation, vec3 wsPosition, vec3 wsNormal)
{
	vec3 localPos = (wsPosition + invPosScale.xyz);
	//mult with orientation
	localPos = quatTransform(orientation, localPos);
	localPos.xz *= invPosScale.w;
	localPos.y *= 1.0; // 1/ (2 * vertical extent)
	//scale with aspect ratio
	localPos.z *= -(scaleOffset.x /scaleOffset.y ) * inDecalAtlasAspectRatio;
	//don't apply on backfaces
	vec3 localNormal = quatTransform(orientation, wsNormal);
	float normalDot = localNormal.y;
	//
	if ( any(lessThan( localPos, vec3(-1.0))) ||
		any(greaterThan( localPos, vec3(1.0))) ||
		normalDot < 0.0)
	{
		//view cluster fit
		//diffuseColor.x = 1.0; // oversampled
		//if ( normalDot < 0.0){ // View "backfacing" geom
		//	diffuseColor.x = 1.0;
		//}
	}
	else
	{
		//box fade, centered around origin
		vec3 bla = abs(localPos);
		float centerDistance = max( max(bla.x, bla.y), bla.z);
		float alphaFactor = 1.0;
		alphaFactor = saturate( (1.0 - centerDistance)*20.0);
		//skew projection based on decal-space normal
		lowp vec2 texCoords = localPos.xz;
		localNormal.z *= -1.0;
		texCoords -= 0.1 * localPos.y * localNormal.xz;
		//bring in texture space
		texCoords *= 0.5;
		texCoords += 0.5;
		texCoords = saturate(texCoords);
		//FIX manual LOD selection to prevent cluster edge errors
		//TODO: why not use frag coord depth ?
		lowp vec4 decalColor = textureLod(inTex9, texCoords * scaleOffset.xy + scaleOffset.zw, ( (pxVSDepth - 15.0) * inInvFarPlane * 2.0));
		//lowp vec4 decalColor = texture(inTex9, texCoords * scaleOffset.xy + scaleOffset.zw);
		diffuseColor.rgb = mix(diffuseColor.rgb, decalColor.rgb, decalColor.w * alphaFactor);
	}
}
#endif // CLUSTER_DECALS

////////////////////////////////////////
// Terrain

mediump vec4 lerp(mediump vec4 a, mediump vec4 b, mediump float ratio)
{
	return (a + ratio * (b - a));
}

#ifdef TERRAIN
lowp vec4 lerpHeight(lowp vec4 tex1, lowp vec4 tex2, lowp float ratio)
{
#if 1
	if (ratio < 0.001)
		return tex1;
	if (ratio > 0.999)
		return tex2;

	float a1 = tex1.a + (1.0 - ratio);
	float a2 = tex2.a + ratio;

	float ma = max(a1, a2) - mix(0.3, 0.1, inTerrainSharpness);

	float b1 = max(a1 - ma, 0.0);
	float b2 = max(a2 - ma, 0.0);

	return (tex1 * b1 + tex2 * b2) / (b1 + b2);
#else
	//CRB: Something I tried. It has an even sharper blend between layers and almost no color blending between layer colors
	// it's also slightly cheaper
	float a1 = tex1.a + (ratio * -2.2 + 1.1);
	float a2 = tex2.a + (ratio * 2.2 - 1.1);

	float sharpness = mix(1.0 / 0.9, 1.0 / 0.05, inTerrainSharpness);//  inTerrainSharpness; // values between 1.0/0.1 and 1.0/0.5
	float colorFactor = clamp((a2 - a1) * sharpness, -1.0, 1.0) * 0.5 + 0.5;

	return mix(tex1, tex2, colorFactor);
#endif
}

lowp vec3 heightNormal( float h_mid)
{
	float p1 = textureOffset(inTex0, pxTexCoord1, ivec2( 1, 0)).a;
	float p2 = textureOffset(inTex0, pxTexCoord1, ivec2( 0,-1)).a;
	vec2 dH = vec2( h_mid - p1, h_mid - p2);
	return normalize(vec3(-dH.x * inTerrainNormalWeight, 1.0, -dH.y * inTerrainNormalWeight));
}

void doTerrain(lowp inout vec4 diffuseColor, inout vec3 wsNormal)
{
#ifndef SHADOWS_LOW_QUALITY
	// create normal from height map
	wsNormal = heightNormal( diffuseColor.a);
#endif

	//
	// blend layers. first fetch material index.
	uint matIdx = texture(inTex2, pxTexCoord1).r * 4u; //texelFetch(inTex2, ivec2(pxTexCoord1 * textureSize(inTex0, 0)), 0).r * 4u;

	lowp vec4 layers[4];
	layers[0] = texture(inTex3, vec3(pxTexCoord2, matIdx));
	layers[1] = texture(inTex3, vec3(pxTexCoord2, matIdx + 1u));
	layers[2] = texture(inTex3, vec3(pxTexCoord2, matIdx + 2u));
	layers[3] = texture(inTex3, vec3(pxTexCoord2, matIdx + 3u));

	lowp vec4 finalColor;
	finalColor = lerpHeight(layers[0], layers[1], diffuseColor.r);
	finalColor = lerpHeight(finalColor, layers[2], diffuseColor.g);
	finalColor = lerpHeight(finalColor, layers[3], diffuseColor.b);
	diffuseColor = finalColor;

#ifdef SHADOWS_LOW_QUALITY
	// when using low quality settings, we could just multiply the height with the diffuse color, if we find a proper ramp
	//diffuseColor.rgb *= saturate(pow(1.0 - diffuseColor.a, 0.4) + 0.7);
#endif
}
#endif

////////////////////////////////////////////////////////////////////////
// Lighting  o
//          /|\
//         / | \

mediump float sampleShadowMap(vec2 base_uv, float u,float v, float depth)
{
	vec3 shadowPos;
    shadowPos.xy = base_uv + vec2(u, v) * inShadowSize.zw;
    shadowPos.z = depth;
	return texture(inTex7, shadowPos);
}

vec2 computeReceiverPlaneDepthBias(vec3 texCoordDX, vec3 texCoordDY)
{
    vec2 biasUV;
    biasUV.x = texCoordDY.y * texCoordDX.z - texCoordDX.y * texCoordDY.z;
    biasUV.y = texCoordDX.x * texCoordDY.z - texCoordDY.x * texCoordDX.z;
    biasUV *= 1.0f / ((texCoordDX.x * texCoordDY.y) - (texCoordDX.y * texCoordDY.x));
    return biasUV;
}

#ifndef DYNAMIC_SHADOWED
mediump float getShadow()
{
	return 1.0;
}
#else
mediump float getShadow()
{
	mediump vec3 lpos = pxLightSpacePos.xyz / pxLightSpacePos.w;

	float lightDepth = lpos.z;
	lightDepth -= 0.0005;
	vec2 uv = lpos.xy * inShadowSize.xy; // 1 unit - 1 texel

	#ifndef TERRAIN
		#define RECEIVER_BIAS
	#endif
	#ifdef RECEIVER_BIAS
		vec2 receiverPlaneBias = computeReceiverPlaneDepthBias( dFdx(lpos), dFdy(lpos));
		float fractionalSamplingError = dot(vec2(1.0f, 1.0f) * inShadowSize.xy, abs(receiverPlaneBias));
		lightDepth += -min(fractionalSamplingError, 0.002f);
	#endif

    vec2 base_uv;
    base_uv.x = floor(uv.x + 0.5);
    base_uv.y = floor(uv.y + 0.5);
    float s = (uv.x + 0.5 - base_uv.x);
    float t = (uv.y + 0.5 - base_uv.y);
    base_uv -= vec2(0.5, 0.5);
    base_uv *= inShadowSize.zw;

	//https://mynameismjp.wordpress.com/2013/09/10/shadow-maps
	//http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1

#ifdef SHADOWS_LOW_QUALITY
	#define FilterSize_ 2
#else
	#define FilterSize_ 3
#endif

	float sum = 0.0;
	#if FilterSize_ == 1
		return sampleShadowMap( lpos.xy + (vec2(0.5) * inShadowSize.zw), 0.0,0.0, lightDepth);
    #elif FilterSize_ == 2
        float uw0 = (3 - 2 * s);
        float uw1 = (1 + 2 * s);

        float u0 = (2 - s) / uw0 - 1;
        float u1 = s / uw1 + 1;

        float vw0 = (3 - 2 * t);
        float vw1 = (1 + 2 * t);

        float v0 = (2 - t) / vw0 - 1;
        float v1 = t / vw1 + 1;

        sum += uw0 * vw0 * sampleShadowMap(base_uv, u0, v0, lightDepth);
        sum += uw1 * vw0 * sampleShadowMap(base_uv, u1, v0, lightDepth);
        sum += uw0 * vw1 * sampleShadowMap(base_uv, u0, v1, lightDepth);
        sum += uw1 * vw1 * sampleShadowMap(base_uv, u1, v1, lightDepth);

        return sum * (1.0 / 16.0);
    #elif FilterSize_ == 3
        float uw0 = (4 - 3 * s);
        float uw1 = 7;
        float uw2 = (1 + 3 * s);

        float u0 = (3 - 2 * s) / uw0 - 2;
        float u1 = (3 + s) / uw1;
        float u2 = s / uw2 + 2;

        float vw0 = (4 - 3 * t);
        float vw1 = 7;
        float vw2 = (1 + 3 * t);

        float v0 = (3 - 2 * t) / vw0 - 2;
        float v1 = (3 + t) / vw1;
        float v2 = t / vw2 + 2;

        sum += uw0 * vw0 * sampleShadowMap(base_uv, u0, v0, lightDepth);
        sum += uw1 * vw0 * sampleShadowMap(base_uv, u1, v0, lightDepth);
        sum += uw2 * vw0 * sampleShadowMap(base_uv, u2, v0, lightDepth);

        sum += uw0 * vw1 * sampleShadowMap(base_uv, u0, v1, lightDepth);
        sum += uw1 * vw1 * sampleShadowMap(base_uv, u1, v1, lightDepth);
        sum += uw2 * vw1 * sampleShadowMap(base_uv, u2, v1, lightDepth);

        sum += uw0 * vw2 * sampleShadowMap(base_uv, u0, v2, lightDepth);
        sum += uw1 * vw2 * sampleShadowMap(base_uv, u1, v2, lightDepth);
        sum += uw2 * vw2 * sampleShadowMap(base_uv, u2, v2, lightDepth);

        return sum * (1.0 / 144.0);
	#endif

	// Debug light frustum
	//if ( (lpos.z < 0 || lpos.x < 0 || lpos.y < 0) || (lpos.x >= 1 || lpos.y >= 1|| lpos.z >= 1)) {
	//	shadowFactor = 0.0;
	//}
}
#endif // #ifndef DYNAMIC_SHADOWED

#ifndef TERRAIN
vec3 getFloorShadowsColor(float wsHeight, vec3 wsNormal)
{
	float heightNorm = saturate( wsHeight / 2.0);
	vec3 floorBounce = textureLod(inTex4, wsNormal.xz * inMapSize.zw * (heightNorm * 0.5) + pxMapTexCoords.xy, 1.0 + 2.0 * heightNorm).rgb;
	floorBounce *= (1.0 - saturate( wsNormal.y));
	return floorBounce;
}
#endif

#ifdef LIGHTING_ENABLED
mediump vec3 getAmbientalLighting(mediump vec3 wsNormal)
{
	float upness = wsNormal.y * wsNormal.y;
	int offset = inAmbientColorIdx;
	return mix(inAmbientColors[offset + 1], inAmbientColors[offset], upness);
}

mediump float getPhongLighting(mediump vec3 pixelToLightDir, mediump vec3 viewDir, mediump vec3 wsNormal, mediump float specularMask)
{
	mediump	float diffuse = dot(pixelToLightDir, wsNormal) * 0.5 + 0.5;
	diffuse *= diffuse;

#ifdef PIXEL_SPECULAR
	mediump vec3 refl = normalize(reflect(-pixelToLightDir, wsNormal)); // normalize((2.0 * attenDot) * wsNormal - pixelToLightDir);
	float scaledIntensity = inSpecularIntensity;
	float scaledPower = inSpecularPower;
	#ifdef PIXEL_TEXTURE_SPECULAR
		scaledIntensity *= specularMask;
		scaledPower *= specularMask;
	#endif
	mediump float specular = pow(saturate(dot(refl, viewDir)), 1.0 + scaledPower) * scaledIntensity;

	diffuse *= saturate(1.0 - specular);
	diffuse += specular;
#endif

	return diffuse;
}

float getHorizontalShadow(vec2 hL, float lightShadowT, float invLightRadiusSq)
{
	float shadow = 0.0;
	float shadowAngle = atan(hL.x, hL.y) * 0.15915494309 + 0.5;
	float hNormalisedDistance = saturate(dot(hL, hL) * invLightRadiusSq);

	float s = shadowAngle;
	float lightRange = texture(inTex14, vec2(s, lightShadowT)).r;
	return saturate(1.0 - (hNormalisedDistance - (lightRange * lightRange )) / (0.015));
}

mediump vec3 doLight(vec4 lightPos, vec4 lightColor, vec4 lightDir, vec4 lightLimits, mediump vec3 wsView, mediump vec3 wsPosition, mediump vec3 wsNormal, mediump float specularMask)
{
	// distance attenuation
	mediump	vec3 pixelToLight = lightPos.xyz - wsPosition;
	float lightDist2 = dot(pixelToLight, pixelToLight);
	mediump	float attenDist = max(0.0, 1.0 - lightDist2 * lightPos.w);
	attenDist *= attenDist;

	if (lightColor.w > 0.0)
	{
		vec4 limits = pixelToLight.xxzz * lightLimits; //NOTE: it's important that pixelToLight is non-normalized here
		float limAtten = max( limits.x, max(limits.y, max(limits.z,limits.w)));
		limAtten = 1.0 - limAtten*limAtten*limAtten;
		attenDist = min(attenDist, max( limAtten , 0));
	}

	if (lightColor.w < 0.0)
	{
		float shadow = getHorizontalShadow( pixelToLight.xz, lightLimits.x, lightPos.w);
		attenDist *= shadow;
	}

	mediump vec3 pixelToLightDir = pixelToLight * inversesqrt(lightDist2);

	// spotlight attenuation
	if (lightDir.w > 0.0) // && attenAngle > 0.0)
	{
		mediump	float spotAngle = dot(lightDir.xyz, pixelToLightDir);
		attenDist *= smoothstep(lightDir.w, lightDir.w * 0.5 + 0.5, spotAngle);
		//attenDist *= max(0.0, 1.0 - (1.0 - spotAngle) * 1.0 / (1.0 - lightDir.w)); // looks worse than the smoothstep version, but faster
	}

	// phong light
	mediump float lightIntensity = getPhongLighting(pixelToLightDir, wsView, wsNormal, specularMask);
	return lightColor.rgb * (lightIntensity * attenDist);
}

void doClusters(lowp inout vec4 diffuseColor, lowp inout vec3 lightContributions, mediump vec3 wsView, mediump vec3 wsPosition, mediump vec3 wsNormal, mediump float specularMask)
{
	//uvec4 clusterData = texelFetch(inTex10, ivec2(pxMapTexCoords.zw), 0); // this fails (for unknown reasons) on some older AMD video drivers
	uvec4 clusterData = texture(inTex10, pxMapTexCoords.xy);
	float decalCountFactor = 0.0;

#ifdef DECALS_DEBUG
	decalCountFactor = float(clusterData.x) / 12; // decal count
#elif defined(LIGHTS_DEBUG)
	decalCountFactor = float(clusterData.y) / 6; // light count
#endif

	uint listIdx = clusterData.w * 256u + clusterData.z;
	uint listY = listIdx / 2048u;
	uint listStart = listIdx - (listY*2048u);

#ifdef CLUSTER_DECALS
	for (uint i = 0U; i < clusterData.x; ++i)
	{
		uvec4 elemIdx = texelFetch(inTex11, ivec2(listStart + i, listY), 0);

		lowp vec4 atlasScaleOffset = 	texelFetch(inTex12, ivec2(0, elemIdx.x), 0);
		lowp vec4 positionSize = 		texelFetch(inTex12, ivec2(1, elemIdx.x), 0);
		lowp vec4 quatOrientation = 	texelFetch(inTex12, ivec2(2, elemIdx.x), 0);
		doDecal(diffuseColor, atlasScaleOffset, positionSize, quatOrientation, wsPosition, wsNormal);
	}
#endif

	for (uint i = 0U; i < clusterData.y; ++i)
	{
		uvec4 elemIdx = texelFetch(inTex11, ivec2(listStart + i, listY), 0);

		lowp vec4 lightPos 	 = texelFetch(inTex13, ivec2(0, elemIdx.y), 0); //.w contains light falloff factor ( 1/r*r )
		lowp vec4 lightColor = texelFetch(inTex13, ivec2(1, elemIdx.y), 0);
		lowp vec4 lightDir 	 = texelFetch(inTex13, ivec2(2, elemIdx.y), 0);
		lowp vec4 lightLimits = texelFetch(inTex13, ivec2(3, elemIdx.y), 0);

		lightContributions += doLight(lightPos, lightColor, lightDir, lightLimits, wsView, wsPosition, wsNormal, specularMask);
	}

//view decal clusters
#ifdef CLUSTER_DEBUG
	if (decalCountFactor > 0)
	{
		decalCountFactor = saturate(decalCountFactor);
		vec3 overdrawColor = pow(vec3( decalCountFactor, 1.0 - decalCountFactor,0.0), vec3(2.2));
		diffuseColor.rgb = decalCountFactor > 0.0 ? overdrawColor: diffuseColor.rgb;
	}
#endif
}

float getCloudOpacity(vec3 wsPosition)
{
	float speedNorm = saturate(inWindSpeed * 3.33);
	vec2 cloudSpeed = -mix(0.005, 0.01, speedNorm) * inWindDirection;
	float cloudScale = 0.0075;
	float cloudOpacity = inCloudOpacity;
	return (1.0 - cloudOpacity) + texture(inTex15, wsPosition.xz * cloudScale + inTime * cloudSpeed).r * cloudOpacity;
}

float getFloorShadows(float wsHeight, float directionalShadowFactor)
{
	lowp float shadow = pow(texture(inTex6, pxMapTexCoords.xy + inFloorShadowsParams.zw).r, inFloorShadowsStrength.x * (1.0 - directionalShadowFactor * inFloorShadowsParams.y));
	lowp float add = abs(wsHeight) / inFloorShadowsParams.x;
	shadow += add + inFloorShadowsStrength.z;
	return saturate(shadow);
}

float getTerrainFloorShadows(float directionalShadowFactor)
{
	lowp float shadow = pow(texture(inTex6, pxMapTexCoords.xy + inFloorShadowsParams.zw).r, inFloorShadowsStrength.y * (1.0 - directionalShadowFactor * inFloorShadowsParams.y));
	shadow += inFloorShadowsStrength.z;
	return min(1.0, shadow);
}

void doLighting(lowp inout vec4 diffuseColor, mediump vec3 wsPosition, mediump vec3 wsNormal)
{
	mediump	vec3 lightContributions = getAmbientalLighting(wsNormal);

	mediump	vec3 viewDir = normalize(inEyePos - wsPosition);

	#ifdef PIXEL_TEXTURE_SPECULAR
		float specularMask = texture(inTex1, pxTexCoord1).r;
	#else
		float specularMask = 0.0;
	#endif

	#ifdef LIGHTING_DIRECTIONAL
		mediump float shadowFactor = getShadow();
		mediump float lightIntensity = getPhongLighting(inLightDir, viewDir, wsNormal, specularMask);
		mediump vec3 dirLight = inLightDirColor.rgb * lightIntensity;

		shadowFactor *= getCloudOpacity(wsPosition);

		// this allows to change the color of the shadow, but it only works correctly when dirLight > ambient light
		//   therefore, for night scenes where directional shines into interiors (and the directional is darker than the interior ambient), it will cast dark shadows inside :()
		lightContributions = mix( lightContributions, dirLight, shadowFactor);
		//lightContributions += dirLight * shadowFactor;
	#else
		mediump float shadowFactor = 0.0;
	#endif

	#ifndef TERRAIN
		lightContributions += getFloorShadowsColor( wsPosition.y, wsNormal);
	#endif

	// lights + decals
	doClusters( diffuseColor, lightContributions, viewDir, wsPosition, wsNormal, specularMask);

	#ifdef PIXEL_TEXTURE_REFLECTION
		mediump vec3 reflDirWS = normalize(reflect( -viewDir, wsNormal));
		vec2 pxSSCoords = gl_FragCoord.xy * inScreenResolution.zw;
		reflDirWS.xz += (pxSSCoords * 2 - 1) * (1.0 - gl_FragCoord.z);
		//float roughness = 0;
		//roughness = saturate(roughness);
		//vec3 reflectionColor = textureLod(inTex8, reflDirWS, roughness * 6).rgb;
		vec3 reflectionColor = textureLod(inTex8, reflDirWS, 3).rgb;
		// inversed fresnel ! So only the top of objects has reflections
		float r0 = 0.0203; // Water R0, picked randomly
		float fresnel = r0 + (1-r0)*pow(1 - abs(dot(wsNormal, viewDir)), 5);
		reflectionColor *= 1 - fresnel;
		// apply as if everything is metal
		lightContributions += reflectionColor;
	#endif

	// floor shadows
	#ifdef TERRAIN
		lightContributions *= getTerrainFloorShadows(shadowFactor);
	#else
		if (wsPosition.y < inFloorShadowsParams.x)
			lightContributions *= getFloorShadows(wsPosition.y, shadowFactor);
	#endif

#ifdef IS_HUMAN
	// amplify lighting to highlight humans a little bit
	float upness = saturate(pow(wsNormal.y, 8.0));
	lightContributions *= 1.0 + upness;
#endif

// apply light
#ifndef CLUSTER_DEBUG
	#ifdef LIGHTING_ONLY
		diffuseColor.rgb = lightContributions;
	#else
		diffuseColor.rgb *= lightContributions;
	#endif
#endif

	// fresnel highlight test
//#ifdef IS_HUMAN
#if FALSE
	//float highlight = 1.0 - pow( 1.0 - abs(wsNormal.y), 128.0);
	float highlight = 1 - abs(dot(wsNormal, viewDir)); // 0 Up -> 1 Horizontal
	//float dir = 1.0 - saturate(wsNormal.x - wsNormal.z);
	float dir = 1.0 - saturate(-wsNormal.z);
	dir = 1.0 - (dir*dir*dir);
	highlight *= dir;
	//float highlight = 1.0 - abs(wsNormal.y);//
	//highlight = pow( highlight, 2.0);
	//highlight = 1.0 - smoothstep( 0.05,0.1, highlight); // shine
	float highlightAlpha = 1.0;
	//zoom in:
	//highlight = smoothstep( 0.5, 0.8, highlight); // highlight
	//highlightAlpha = 0.3;
	// zoom med:
	//highlight = smoothstep( 0.2, 0.5, highlight); // highlight
	//highlightAlpha = 0.85;
	// zoom far:
	highlight = smoothstep( 0.0, 0.3, highlight); // highlight
	highlightAlpha = 0.5;

	float timeScaled = fract(inTime * 0.33);
	timeScaled *= timeScaled;
	highlightAlpha *= max( timeScaled, 1.0 - timeScaled);

	//diffuseColor.rgb = mix(diffuseColor.rgb, vec3(1.0), highlight * 0.5);
	diffuseColor.rgb += highlight * highlightAlpha * vec3(1, 0, 0);
#endif
}
#endif // ifdef LIGHTING_ENABLED

//http://www.thetenthplanet.de/archives/1180
// non-tangent normal mapping
mat3 cotangentFrame( vec3 N, vec3 V, vec2 uv ) {
	// get edge vectors of the pixel triangle
	vec3 dp1 = dFdx( V );
	vec3 dp2 = dFdy( V );
	vec2 duv1 = dFdx( uv );
	vec2 duv2 = dFdy( uv );

	// solve the linear system
	vec3 dp2perp = cross( dp2, N );
	vec3 dp1perp = cross( N, dp1 );
	vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
	vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

	// construct a scale-invariant frame
	float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
	return mat3( T * invmax, N, B * invmax );
}

vec3 perturbNormal( vec3 mN, vec3 vN, vec3 V, vec2 uv)
{
    // assume N, the interpolated vertex normal and V, the view vector (vertex to eye)
    mat3 TBN = cotangentFrame( vN, -V, uv );
    return normalize( TBN * mN );
}

#ifdef WATER
vec4 doWater(lowp in vec3 wsPosition, lowp inout vec4 diffuseColor, inout vec3 wsNormal)
{
	// settings
	float clarity = inWaterClarity;
	vec3 colorClean = vec3(0.2, 0.5, 0.5);
	vec3 colorMurky = vec3(0.07, 0.07, 0.03);
	vec3 colorFog = mix(colorMurky, colorClean, saturate(clarity * clarity));
	clarity = 0.2 + clarity * 0.8;
	float invMaxFogDepth = 1.0 / (0.1 + clarity * 1.5);

	vec2 screenUV = ( gl_FragCoord.xy * inScreenResolution.zw);
	float sceneEyeZ = texture(inTex3, screenUV).r; // = eyeZ/far, normalised [0,1]
	sceneEyeZ *= inFarPlane; // [0 .. far]
	float pixelEyeZ = pxVSDepth; // eyeZ [0 .. far]
	float depthDiff = saturate( (sceneEyeZ - pixelEyeZ) * invMaxFogDepth);

	float murk = mix(6.0, 0.15, clarity);
	float fogFactor = 1.0 - exp2(-murk * depthDiff)*0.9;
	//float fogFactor = 0.1 + pow(depthDiff, clarity)*0.9; // alternate depth falloff, too milky
	vec4 waterColor = vec4( colorFog.rgb, fogFactor);


	float timeScaled = mod(inTime,1000.0) * 0.075;
	vec2 coord1 = wsPosition.xz * 0.23 + vec2(0.07, 0.33) * timeScaled;
	vec2 coord2 = wsPosition.xz * 0.17 + vec2(-0.43, 0.01) * timeScaled;

	#define WATER_METHOD 2

	#if WATER_METHOD == 1
		//Height based, caustics-looking thing
		// diffuse tex is wave height map
		//float waveHeight = texture(inTex0, coord1).r + texture(inTex0, coord2).r;
		//waveHeight = saturate( waveHeight * (clarity * 0.75 + 0.5));
		float waveHeight = texture(inTex0, coord1).r * texture(inTex0, coord2).r;
		waveHeight = saturate( waveHeight * (clarity * 1.5 + 0.5));
		//float nScale = 100.0; //TODO: incomplete
		//vec3 normalShift = vec3(-dFdx(waveHeight)*nScale, 0.0, dFdy(waveHeight)* nScale);
		//wsNormal = normalize(wsNormal + normalShift);
		//waveHeight = pow(waveHeight, 32);
		waveHeight *= waveHeight * 2.0;
		diffuseColor = vec4( vec3(waveHeight), 1.0);
	#endif

	#if WATER_METHOD == 2
		// diffuse tex is wave normal map, blue is up
		vec3 n1 = readNormalMap(inTex0, coord1);
		vec3 n2 = readNormalMap(inTex0, coord2);
		vec3 n12 = blendNormalMaps(n1, n2, 1.0);
		n12.z *= -1.0;
		//float shore = 5.0 * saturate( depthDiff*depthDiff*depthDiff);
		float shore = saturate( (sceneEyeZ - pixelEyeZ) * 2.0);
		shore = 2.0 * shore*shore*shore;
		float distFudge = saturate( (inEyePos.y - 20.0) * 0.01); // ~0 when close, 1 when 5x further
		float upWeight = ((1.0 - distFudge)*0.2 + 0.2) * shore; // this is a sort of wave smoothness value
		wsNormal = blendNormalMaps(wsNormal, n12, upWeight);
		// correct, but way more expensive and worse-looking
		//mediump	vec3 viewDir = normalize(inEyePos - wsPosition);
		//wsNormal = perturbNormal( n12, wsNormal, -viewDir, pxTexCoord1);
		//diffuseColor = vec4( waterColor.rgb * 0.1, 1.0);
		diffuseColor = vec4( vec3(0.1), 1.0);
	#endif

	return waterColor;
}
#endif // WATER

#ifdef SNOW_ENABLED
void doSnow(lowp inout vec4 diffuseColor, mediump vec3 wsPosition, lowp vec3 wsNormal)
{
	//old, super expensive
	//float noise = texture(inTex15, wsPosition.xz * 0.15).r * 0.75 + 0.25;
	//noise *= texture(inTex15, wsPosition.xz * 0.5).r * 0.5 + 0.5;
	//noise *= texture(inTex15, wsPosition.xz * vec2(1.97, 2.33)).r * 0.10 + 0.9;

	float noise = texture(inTex15, wsPosition.xz * 0.5).r * 0.5 + 0.5;

	float snowyness = wsNormal.y * (noise*0.5 + 0.5);
	snowyness = saturate( -0.8* inWindSpeed * dot( inWindDirection, wsNormal.xz) + snowyness);
	float invSnowAmt = 0.5;
	snowyness = smoothstep(invSnowAmt, invSnowAmt + 0.2, snowyness);
	//float outdoors = inAmbientColorIdx == 0 ? 1.0 : 0.0;
	diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.7), snowyness);
}
#endif // SNOW_ENABLED

void main (void)
{
	vec3 wsPosition;
	vec3 wsNormal;

#if defined(LIGHTING_ENABLED) || defined(CLUSTER_DECALS)
	wsPosition = pxWSPosition.xyz;
	wsNormal = normalize(pxNormal);
#endif

#if defined(VERTEX_TEXCOORD1) || defined(PIXEL_TEXTURE_DIFFUSE) // second define not really cool, but for cases where we want to pass in custom tex coords, instead of per-vertex
	vec2 texCoord = pxTexCoord1;
#endif

#ifdef PIXEL_TEXTURE_DIFFUSE
	lowp vec4 diffuseColor;

	#ifdef WATER
		vec4 waterColor = doWater(wsPosition, diffuseColor, wsNormal);
	#else
		diffuseColor = texture(inTex0, texCoord);
	#endif

	#ifdef PIXEL_ALPHA_TEST
		if (diffuseColor.a < 0.5)
			discard;

		// when using alpha-to-coverage
		//diffuseColor.a = (diffuseColor.a - 0.5) / max(fwidth(diffuseColor.a), 0.0001) + 0.5;
	#endif

	#ifdef DETAIL_TEXTURE
		lowp vec4 detailColor = texture(inTex2, texCoord);
		diffuseColor.rgb = mix( diffuseColor.rgb, detailColor.rgb, detailColor.a);
	#endif

	#ifdef EMISSIVE_MASKED
		lowp vec4 emissiveColor;
		emissiveColor.a = diffuseColor.a;
		emissiveColor.rgb = diffuseColor.rgb * diffuseColor.a;
		diffuseColor.rgb -= emissiveColor.rgb;
	#endif


#else
	lowp vec4 diffuseColor = vec4(1.0);
#endif

#if defined(VERTEX_MULTIPLY_COLOR) || (defined(VERTEX_COLOR) && !defined(WIND_ANIM))
	diffuseColor *= pxColor;
#endif

#ifdef TERRAIN
	doTerrain(diffuseColor, wsNormal);
#endif

#ifdef SNOW_ENABLED
	doSnow(diffuseColor, wsPosition.xyz, wsNormal);
#endif

#ifdef LIGHTING_ENABLED
	doLighting(diffuseColor, wsPosition.xyz, wsNormal);
#endif

#ifdef EMISSIVE_MASKED
	diffuseColor.rgb += emissiveColor.rgb;
#endif

#ifdef WATER
	diffuseColor.rgb *= diffuseColor.a; // premultiplied blending
	diffuseColor.rgb += waterColor.rgb * waterColor.a;
	diffuseColor.a = waterColor.a; // this is just bg occlusion, based on water fog
#endif

#ifdef FOG_OF_WAR
float lum = dot(vec3(0.299, 0.587, 0.114), diffuseColor.rgb);
vec3 dk1color = pow( vec3(0.3,0.6,0.8), vec3(2.2)) * lum;
	diffuseColor = doFogOfWar(diffuseColor, inTex5, pxMapTexCoords, dk1color); //  Note: this color is later multiplied by 140/255, which is the gray FOV color applied as a post-process.  this is the DK1 fog color:	vec3 color = pow( vec3(0.3,0.6,0.8), vec3(2.2)) * lum;
#endif

// the map edges are darkened as a post-process, together with the height fog. But when that is turned off, we need to do it in the unified shader.
#ifdef DARKEN_MAP_EDGES
	diffuseColor = doMapEdgeDarkening(diffuseColor, wsPosition, inMapSize.xy);
#endif

#if defined(VISUALIZE_MIP_LEVELS) && defined(VERTEX_TEXCOORD1)
	#ifdef TERRAIN
		vec2 texSize = inMapSize.xy * 128; // == TERRAIN_TILE_PIXELS_PER_METER
	#else
		vec2 texSize = textureSize(inTex0, 0);
	#endif
	mediump	vec2 mipTexCoord = texCoord * texSize / 16.0;
	mediump	vec4 mip = texture(inTex16, mipTexCoord);
	diffuseColor.rgb = mix(diffuseColor.rgb, mip.rgb, mip.a);
#endif

	outFragColor0 = diffuseColor;
}
