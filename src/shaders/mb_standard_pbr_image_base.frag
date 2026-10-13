// Copied from xgeom_static.plugin/source/runtime (the image-based PBR the xmaterial graph builds on); local change: the shadow clipmap.
//-------------------------------------------------------------------------------------------
// PBRLighting
//-------------------------------------------------------------------------------------------
// Input Parameter Guidelines
// 
// inNormal			: Tangent - space normal vector(-1 to 1 range, normalized). Use normal maps for detail; 
//						decode from RG channels if packed(e.g., normal = texture(normalMap, UV).xy * 2.0 - 1.0; 
//						normal.z = sqrt(1.0 - dot(normal.xy, normal.xy))).
// inAlbedo			: Base color(linear RGB, 0 - 1).For dielectrics, it's diffuse; for metals, specular tint. 
//						Linearize sRGB textures: pow(texture(albedoMap, UV).rgb, 2.2). Avoid pure black/white for realism.
// inAO				: Ambient occlusion(0 - 1, typically from AO map).Multiplies ambient; bake or use SSAO for dynamic.
// inF0				: Dielectric base reflectance (0.02-0.08 linear, e.g., 0.04 for plastics/water). 
//						Higher values unphysical for non-metals; use 0.04 default.
// inRoughness		: Perceptual roughness(0 - 1). Low(0 - 0.3) for shiny(sharp highlights); 
//						high(0.7 - 1) for matte(diffuse - like). Clamp min 0.045 to avoid artifacts.
// inMetalic		: Metalness(0 - 1). 0 for dielectrics(diffuse from albedo); 
//						1 for metals(no diffuse, albedo tints specular). Avoid intermediates unless blending(e.g., rust).
// inEmissiveColor	: Self-illumination color (RGB, 0-1+ for HDR glow). Adds unshaded light; use maps for detail (e.g., neon/glow effects).
// 
//-------------------------------------------------------------------------------------------
// Uniform Setup(lighting_uniforms)
//
// LightColor		: RGB * intensity(white default; intensity 5 - 15 for 1 - 10 unit scenes). 
//						Too low(< 5) dims directional cues; too high(> 500) blows out without tonemap.
// AmbientLightColor: Low - intensity fill(RGB 0.1 - 1.0, e.g., vec3(0.5) for neutral).Represents global illum; 
//						over 1.0 unphysical but brightens shadows.
// wSpaceLightPos	: XYZ position, W radius(> 0 for finite falloff, 0 for infinite 1 / dist²). 
//						Set radius to bbox extent for soft decay; prevents harsh infinite - range lighting.
// wSpaceEyePos		: Camera position(must be accurate; vec4(0) causes invalid V vector and black artifacts).
// LightParams		: .x = area radius (0-1+ for specular/soft shadow approx, soften highlights 10-20% realism),
//						.y = temperature (Kelvin, e.g., 6500 for daylight; shifts LightColor via blackbody),
//						.z = intensity multiplier (1.0 default; scales final radiance),
//						.w = spare (unused, for future extensions).
//
//-------------------------------------------------------------------------------------------
// Material Tuning and Usage
// 
// Workflow			: Metallic - roughness(glTF standard).Test with PBR spheres(e.g., albedo white, roughness gradient) 
//						to validate energy conservation(no darkening at edges).
// Realism Tips		: For metals, set metallic = 1, roughness low, albedo metallic hue(e.g., gold vec3(1, 0.76, 0.34)).
//						For dielectrics, metallic = 0, F0 = 0.04. Use roughness maps for variation(e.g., scratches).
// Integration		: Linear color space required; apply tonemap(Reinhard: color / (color + 1)) and gamma(pow(1 / 2.2)) 
//						post - shader. Enable HDR for > 1 values.ShadowPCF assumes valid shadow map; debug with Shadow = 1.0.
// Performance		: Early culls(shadow / NdotL / att) save 30 - 50 % in dark / far areas—profile with GPU tools.
//						Avoid in shaders with many lights; cluster / defer for multiples.
// Quality			: TODO: Add IBL(cubemap for specular ambient) for non - flat shadows. For production, 
//						layer clearcoat(extra specular term) or SSS if extending inputs.
// Pitfalls			: Unlinearized textures cause muddy colors; invalid uniforms(e.g., eyePos = 0) break vectors. 
//						Intensity scales with scene units(1m = 1 unit assumed).
// 
//-------------------------------------------------------------------------------------------

#include "mb_standard_shadow.frag"
#include "mb_varying_definition_full.glsl"

layout(location = 0) in VaryingFull In;
#ifndef XLION_LDR_OUTPUT
layout(location = 9) in vec4 InGI;     // xLIONRender surfel GI: sum(w E), sum(w) from the vertex shader (0 without GI) - divided here, into g_GI
#ifndef XLION_NO_GI_PIXEL
layout(location = 10) in vec3 InGIPos;          // the per-pixel read: mesh space position, normal, { first surfel in GILight, GI level }
layout(location = 11) in vec3 InGINormal;
layout(location = 12) flat in uvec2 InGIRef;
layout(location = 13) in vec4 InGITangent;      // mesh space (w: the binormal's sign): the normal map's normal into mesh space
#include "xlionrender_surfel_gi.glsl"
#endif
#endif
#ifndef XLIONRENDER_SURFEL_GI_GLSL
#define GIDebugOr(Lit) (Lit)
#endif

#ifndef XLION_LDR_OUTPUT
struct LocalLightData                  // xLIONRender: a point or spot light (shaders/xlionrender_local_lights.glsl)
{
	vec4 PosRadius;                    // camera-centered position, radius
	vec4 Color;                        // rgb: color * intensity, w: cos of the spot's inner angle
	vec4 DirCosOuter;                  // spot: direction, w cos of the outer angle; point: w = -2
	vec4 Shadow;                       // xy: its shadow map's origin in the atlas, z: its size (0 none), w: spot tan of the half field of view
};
#endif

layout(set = 2, binding = 1) uniform lighting_uniforms
{
	vec4 LightColor;                   // Light color (RGB) * intensity (assume premultiplied)
	vec4 AmbientLightColor;            // Ambient light color (RGB)
	vec4 wSpaceLightPos;               // Light position in world space (XYZ) and radius (W > 0 for finite falloff)
	vec4 wSpaceEyePos;                 // Camera position in world space (XYZ)
	vec4 LightParams;                  // .x=area radius (spec/soft shadow approx), .y=temperature (Kelvin), .z=intensity mult, .w=spare
#ifndef XLION_LDR_OUTPUT
	// xLIONRender (plans/forward/xlionrender_plan_forward.h ubo_lighting): the directional light's shadow clipmap
	mat4 W2Light;
	vec4 CameraPos;
	vec4 ShadowLevels[4];
	vec4 ShadowParams;
	vec4 LocalCount;                     // x: local lights, y: mode (0 exact, 1 cells, 2 cells + sampled tail), z: seed, w: head size
	vec4 LocalBase;                      // x: first light in LocalLightList, y: first uint in LightCells, z: level 0 cell size, w: tail samples
	vec4 CellOrigin[5];                  // per cell level: the window's first cell index (xyz) - features/lights/xlionrender_light_cells.h
	vec4 EnvParams;                      // x: 1 the environment probe replaces the flat ambient, y: its intensity
	vec4 ProbeCount;                     // x: reflection probes (innermost first), y: the first's record in LocalLightList, z: their cell headers in LightCells (-1: none)
#endif
} UBOLight;

#ifndef XLION_LDR_OUTPUT
#include "xlionrender_shadow_clipmap.glsl"
#endif

#ifdef XLION_LDR_OUTPUT
layout(set = 2, binding = 0) uniform sampler2D		brdfLUT;	// 2D LUT for specular BRDF integration (DFG terms)
layout(set = 2, binding = 1) uniform samplerCube	envMap;		// Prefiltered environment cubemap for specular IBL
#endif

//-------------------------------------------------------------------------------------------
// Constants
//-------------------------------------------------------------------------------------------

const float PI				= 3.14159265359;
const float RCPPI			= 1.0 / PI;
const float CONSTANT1_FON	= 0.5 - 2.0 / (3.0 * PI);
const float CONSTANT2_FON	= 2.0 / 3.0 - 28.0 / (15.0 * PI);
const float MIN_ROUGHNESS	= 0.045; // Avoid singularities (e.g., mirror artifacts)
const float MAX_LOD			= 7.0;   // Max mip level for prefiltered envMap (adjust based on cubemap resolution)

//-------------------------------------------------------------------------------------------
// Helper Functions
//-------------------------------------------------------------------------------------------

float saturate(float x) { return clamp(x, 0.0, 1.0); }
float Pow5(float x)     { float x2 = x * x; return x2 * x2 * x; }

//-------------------------------------------------------------------------------------------
// Blackbody Color Shift
//-------------------------------------------------------------------------------------------
// Approximates Planck's law for temperature-based tint (e.g., 6500K daylight). Low perf cost; shifts LightColor.
// Based on simplified CIE XYZ to RGB; temp 0 disables (neutral).
vec3 BlackbodyTint(float temp) 
{
	if (temp <= 0.0) return vec3(1.0);
	temp /= 100.0;
	vec3 color;
	if (temp <= 66.0) 
	{
		color.r = 1.0;
		color.g = saturate(0.39008157876901960784 * log(temp) - 0.63184144378862745098);
		color.b = (temp <= 19.0) ? 0.0 : saturate(0.54320678911019607843 * log(temp - 10.0) - 1.19625408914);
	} 
	else 
	{
		color.r = saturate(1.29293618606274509804 * pow(temp - 60.0, -0.1332047592));
		color.g = saturate(1.12989086089529411765 * pow(temp - 60.0, -0.07551484999));
		color.b = 1.0;
	}
	return color;
}

//-------------------------------------------------------------------------------------------
// PBR Components
//-------------------------------------------------------------------------------------------

vec3 FresnelSchlick(float cosTheta, vec3 F0) 
{
	return F0 + (1.0 - F0) * Pow5(saturate(1.0 - cosTheta));
}

float DistributionGGX(float NdotH, float alpha) 
{
	float a2 = alpha * alpha;
	float NdotH2 = NdotH * NdotH;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	return a2 / (PI * denom * denom + 1e-6);
}

float VisibilitySmithCorrelated(float NdotV, float NdotL, float alpha) 
{
	float a2 = alpha * alpha;
	float GGXV = NdotL * sqrt(a2 + (1.0 - a2) * NdotV * NdotV);
	float GGXL = NdotV * sqrt(a2 + (1.0 - a2) * NdotL * NdotL);
	return 0.5 / (GGXV + GGXL + 1e-6);
}

float E_FON_approx(float mu, float r) 
{
	float mucomp = 1.0 - mu;
	const float g1 = 0.0571085289;
	const float g2 = 0.491881867;
	const float g3 = -0.332181442;
	const float g4 = 0.0714429953;
	float GoverPi = mucomp * (g1 + mucomp * (g2 + mucomp * (g3 + mucomp * g4)));
	return (1.0 + r * GoverPi) / (1.0 + CONSTANT1_FON * r);
}

// EON Diffuse (Energy-Preserving Oren-Nayar, JCGT 2025): Analytical, reciprocity-compliant rough diffuse model.
// Fixes energy loss (up to 30% on rough surfaces) and dark grazing edges in Burley; ideal for fabrics/skin.
// r = perceptual roughness [0,1]; rho = albedo (vec3); mu_i = NdotL; mu_o = NdotV; s = L·V - NdotL*NdotV.
// Uses approximate albedo for perf; full BRDF f_r including /pi and rho.
vec3 DiffuseEON(vec3 rho, float r, float mu_i, float mu_o, float s) 
{
	float AF = 1.0 / (1.0 + CONSTANT1_FON * r);
	vec3 f_ss = (rho * RCPPI) * AF * (vec3(1.0) + r * (s > 0.0 ? s / max(mu_i, mu_o) : s));
	float EFo = E_FON_approx(mu_o, r);
	float EFi = E_FON_approx(mu_i, r);
	float avgEF = AF * (1.0 + CONSTANT2_FON * r);
	vec3 rho_ms = (rho * rho) * avgEF / (vec3(1.0) - rho * (1.0 - avgEF));
	vec3 f_ms = (rho_ms * RCPPI) * max(1e-7, 1.0 - EFo) * max(1e-7, 1.0 - EFi) / max(1e-7, 1.0 - avgEF);
	return f_ss + f_ms;
}

//-------------------------------------------------------------------------------------------
// Optimized Attenuation with Finite Radius
//-------------------------------------------------------------------------------------------
// Physically-based, energy-conserving falloff: (1 - s^4) / (1 + 2 s^2), s = d/radius (Lagarde 2021 variant, validated in 2025 real-time engines).
// Zero at d >= radius, smooth, no singularity; F=2 for realistic decay (Unreal/Filament 2020s updates).
// Explicit radius handling; w=0 for infinite. Quadratic soften via LightParams.x (area radius) scales effective radius (10-20% realism boost).
float Attenuate(float distance, float radius) 
{
	float areaRadius = UBOLight.LightParams.x;
	radius = max(radius, areaRadius); // Soften with area approx (quadratic-like via increased effective radius)
	float s = distance / radius;
	if (s >= 1.0) return 0.0;
	float s2 = s * s;
	float s4 = s2 * s2;
	return (1.0 - s4) / (1.0 + 2.0 * s2);
}

//-------------------------------------------------------------------------------------------
// Trasforms the Normal to World space using the MikkTSpace method
//-------------------------------------------------------------------------------------------
vec3 TransformNotmalToWorld(vec3 inNormal)
{
	// Construct bitangent (must stay unnormalized as expected by MikkTSpace)
	vec3 vB = In.Tangent.w * cross(In.Normal.xyz, In.Tangent.xyz);

	// Decode final world normal
	return normalize(inNormal.x * In.Tangent.xyz + inNormal.y * vB.xyz + inNormal.z * In.Normal.xyz);
}

#ifndef XLION_LDR_OUTPUT
#include "xlionrender_local_lights.glsl"
#include "xlionrender_environment.glsl"
#endif

//-------------------------------------------------------------------------------------------
// PBRLightingImageBase: 2025-Inspired, Quality/Perf Optimized
//-------------------------------------------------------------------------------------------
// Insights from 2020-2025: EON diffuse (energy/reciprocity over Burley; fixes 30% loss/dark edges); multiscatter compensation (UE5/Filament 2020s, neural-validated).
// Neural BRDF approx skipped (GLSL perf); added simple energy scale for rough spec (avoids 20-30% loss, per 2025 papers).
// Stochastic-inspired early culls (shadow=0, NdotL=0, att=0) save 50%+ cycles in umbra/far (2025 AVBOIT/MegaLights style).
// IBL: Specular from prefiltered envMap * DFG LUT (split-sum; no SH diffuse yet).
vec3 PBRLightingImageBase
( in const vec3  inNormal
, in const vec3  inAlbedo
, in const float inAO
, in const float inF0
, in const float inRoughness
, in const float inMetalic
, in const vec3  inEmissiveColor
)
{
	vec3 emissive = inEmissiveColor * inAO; // Modulate early for occlusion

	// Ambient baseline: Compute always—cheap; 2025 grazing approx for subtle IBL hack (neural-inspired, no cubemap).
	vec3	N					= TransformNotmalToWorld(inNormal);
	vec3	V					= normalize(UBOLight.wSpaceEyePos.xyz - In.wSpacePosition.xyz);
	float	NdotV				= saturate(dot(N, V));
	vec3	ambientDiffuse		= (1.0 - inMetalic) * inAlbedo * UBOLight.AmbientLightColor.rgb * inAO / PI;
	vec3	F0					= mix(vec3(inF0), inAlbedo, inMetalic);
	vec3	F_grazing			= FresnelSchlick(0.0, F0);
	vec3	ambientSpec			= F_grazing * UBOLight.AmbientLightColor.rgb * inAO * (1.0 - inRoughness) * 0.04;
	vec3	ambient				= ambientDiffuse + ambientSpec;

	// IBL specular: Split-sum approx (prefiltered env * DFG LUT).
	float	perceptualRoughness = max(inRoughness, MIN_ROUGHNESS);
#ifdef XLION_LDR_OUTPUT
	vec3	R					= reflect(-V, N);
	vec3	prefilteredColor	= textureLod(envMap, R, perceptualRoughness * MAX_LOD).rgb;
	vec2	lut					= texture(brdfLUT, vec2(NdotV, perceptualRoughness)).rg;
	vec3	indirectSpec		= prefilteredColor * (F0 * lut.x + vec3(lut.y));
	ambient += indirectSpec;
#else
	// xLIONRender: the renderer's environment probe (diffuse + specular) when the sky is on
	if (InGI.w > 1e-6) g_GI = vec4(InGI.rgb / InGI.w, 1.0);
#ifndef XLION_NO_GI_PIXEL
	// The face's normal (mesh space, to the smoothed normal's side): the GI's geometry tests use the surface as it is - smoothed corner
	// normals step along the triangles (a sawtooth); the normal map still reads the basis (GNmap)
	const vec3 GFaceD = cross(dFdx(InGIPos), dFdy(InGIPos));
	const vec3 GFace  = dot(GFaceD, GFaceD) > 1e-24 ? normalize(dot(GFaceD, InGINormal) < 0.0 ? -GFaceD : GFaceD) : normalize(InGINormal);
	if (InGIRef.y != gi_none_v && (InGIRef.y & gi_per_pixel_v) != 0u)     // the pixel reads the surfels itself (far: through their parents)
	{
		const vec3 GN = normalize(InGINormal), GT = normalize(InGITangent.xyz);
		const vec3 GNmap = normalize(inNormal.x * GT + inNormal.y * (InGITangent.w * cross(GN, GT)) + inNormal.z * GN);     // as TransformNotmalToWorld, in mesh space
		const vec4 Px = SurfelGIPixel(InGIPos, GFace, GNmap, InGIRef.x, InGIRef.y & 0xFFu);
		if (Px.w > 1e-6) g_GI = vec4(Px.rgb / Px.w, 1.0);
	}
	const uint GIDebug = InGIRef.y == gi_none_v ? 0u : (InGIRef.y >> gi_debug_shift_v) & 3u;
	if (GIDebug == 1u) { const uint Sd = SurfelGIDominant(InGIPos, GFace); g_GIDebug = vec4(Sd == 0xFFFFFFFFu ? vec3(0.0) : SurfelDebugColor(Sd), 1.0); }
	if (GIDebug == 2u) g_GIDebug = vec4(g_GI.w > 0.5 ? g_GI.rgb / PI : vec3(0.0), 1.0);
	if (GIDebug == 3u) g_GIDebug = vec4((InGIRef.y & 0x800u) != 0u ? inEmissiveColor : inAlbedo * (1.0 - inMetalic), 1.0);     // the GI's material capture (xlionrender_gi_capture_vert.glsl)
#endif
	if (UBOLight.EnvParams.x > 0.5 || UBOLight.ProbeCount.x > 0.5 || g_GI.w > 0.5) ambient = EnvLighting(N, V, NdotV, inAlbedo, F0, perceptualRoughness, inMetalic, inAO);
#endif

	// Early culls: Compute minimal for combined check (cheap ops, low divergence risk).
#ifdef XLION_LDR_OUTPUT
	float   Shadow              = ShadowPCF(In.ShadowPosition / In.ShadowPosition.w);
#else
	float   Shadow              = UBOLight.wSpaceLightPos.w == 0.0 ? ShadowClipmap(In.wSpacePosition.xyz + UBOLight.CameraPos.xyz) : 1.0;
#endif
	// W == 0 is a directional light (XYZ = direction towards the light), W > 0 a point light with that falloff radius
	vec3	L;
	float	att;
	if (UBOLight.wSpaceLightPos.w == 0.0)
	{
		L	= normalize(UBOLight.wSpaceLightPos.xyz);
		att	= 1.0;
	}
	else
	{
		vec3    lightVec		= UBOLight.wSpaceLightPos.xyz - In.wSpacePosition.xyz;
		float   distance		= length(lightVec);
		att						= Attenuate(distance, UBOLight.wSpaceLightPos.w);
		L						= lightVec / distance;
	}
	float	NdotL				= saturate(dot(N, L));
#ifndef XLION_LDR_OUTPUT
	const vec3 Local = LocalLights(N, V, NdotV, inAlbedo, F0, max(inRoughness, MIN_ROUGHNESS), inMetalic);
#else
	const vec3 Local = vec3(0.0);
#endif
	if (Shadow < 1e-4 || att < 1e-4 || NdotL < 1e-4) return GIDebugOr(ambient + Local + emissive);

	// Commit to lit path: Rest here.
	vec3	H					= normalize(V + L);
	float	NdotH				= saturate(dot(N, H));
	float	LdotH				= saturate(dot(L, H));

	float	alpha				= perceptualRoughness * perceptualRoughness;

	vec3	F					= FresnelSchlick(LdotH, F0);

	float	D					= DistributionGGX(NdotH, alpha);
	float	Vis					= VisibilitySmithCorrelated(NdotV, NdotL, alpha);
	vec3	specular			= D * F * Vis;

	// 2025 Innovation: Simple multiscatter energy compensation (UE5/Filament, SIGGRAPH 2025 neural validation); scales rough spec ~20-30% brighter.
	vec3	avgF				= F * (1.0 - perceptualRoughness) + vec3(perceptualRoughness); // Approx avg Fresnel
	vec3	multiScatter		= (vec3(1.0) - avgF) / max(vec3(1e-6), vec3(1.0) - avgF * perceptualRoughness);
	specular *= multiScatter;

	vec3	kD					= (vec3(1.0) - F) * (1.0 - inMetalic);
	vec3	diffuse				= kD * DiffuseEON(inAlbedo, perceptualRoughness, NdotL, NdotV, dot(L, V) - NdotL * NdotV);

	vec3	lightTint			= BlackbodyTint(UBOLight.LightParams.y); // Apply temperature shift
	vec3	radiance			= UBOLight.LightColor.rgb * lightTint * att * UBOLight.LightParams.z; // Apply intensity mult
	vec3	directLighting		= (diffuse + specular) * radiance * NdotL * Shadow;

	return GIDebugOr(ambient + directLighting + Local + emissive);
}





/*
//-------------------------------------------------------------------------------------------
// PBRLighting
//-------------------------------------------------------------------------------------------
// Input Parameter Guidelines
// 
// inNormal			: Tangent - space normal vector(-1 to 1 range, normalized). Use normal maps for detail; 
//						decode from RG channels if packed(e.g., normal = texture(normalMap, UV).xy * 2.0 - 1.0; 
//						normal.z = sqrt(1.0 - dot(normal.xy, normal.xy))).
// inAlbedo			: Base color(linear RGB, 0 - 1).For dielectrics, it's diffuse; for metals, specular tint. 
//						Linearize sRGB textures: pow(texture(albedoMap, UV).rgb, 2.2). Avoid pure black/white for realism.
// inAO				: Ambient occlusion(0 - 1, typically from AO map).Multiplies ambient; bake or use SSAO for dynamic.
// inF0				: Dielectric base reflectance (0.02-0.08 linear, e.g., 0.04 for plastics/water). 
//						Higher values unphysical for non-metals; use 0.04 default.
// inRoughness		: Perceptual roughness(0 - 1). Low(0 - 0.3) for shiny(sharp highlights); 
//						high(0.7 - 1) for matte(diffuse - like). Clamp min 0.045 to avoid artifacts.
// inMetalic		: Metalness(0 - 1). 0 for dielectrics(diffuse from albedo); 
//						1 for metals(no diffuse, albedo tints specular). Avoid intermediates unless blending(e.g., rust).
// inEmissiveColor	: Self-illumination color (RGB, 0-1+ for HDR glow). Adds unshaded light; use maps for detail (e.g., neon/glow effects).
// 
//-------------------------------------------------------------------------------------------
// Uniform Setup(lighting_uniforms)
//
// LightColor		: RGB * intensity(white default; intensity 5 - 15 for 1 - 10 unit scenes). 
//						Too low(< 5) dims directional cues; too high(> 500) blows out without tonemap.
// AmbientLightColor: Low - intensity fill(RGB 0.1 - 1.0, e.g., vec3(0.5) for neutral).Represents global illum; 
//						over 1.0 unphysical but brightens shadows.
// wSpaceLightPos	: XYZ position, W radius(> 0 for finite falloff, 0 for infinite 1 / dist²). 
//						Set radius to bbox extent for soft decay; prevents harsh infinite - range lighting.
// wSpaceEyePos		: Camera position(must be accurate; vec4(0) causes invalid V vector and black artifacts).
// LightParams		: .x = area radius (0-1+ for specular/soft shadow approx, soften highlights 10-20% realism),
//						.y = temperature (Kelvin, e.g., 6500 for daylight; shifts LightColor via blackbody),
//						.z = intensity multiplier (1.0 default; scales final radiance),
//						.w = spare (unused, for future extensions).
//
//-------------------------------------------------------------------------------------------
// Material Tuning and Usage
// 
// Workflow			: Metallic - roughness(glTF standard).Test with PBR spheres(e.g., albedo white, roughness gradient) 
//						to validate energy conservation(no darkening at edges).
// Realism Tips		: For metals, set metallic = 1, roughness low, albedo metallic hue(e.g., gold vec3(1, 0.76, 0.34)).
//						For dielectrics, metallic = 0, F0 = 0.04. Use roughness maps for variation(e.g., scratches).
// Integration		: Linear color space required; apply tonemap(Reinhard: color / (color + 1)) and gamma(pow(1 / 2.2)) 
//						post - shader. Enable HDR for > 1 values.ShadowPCF assumes valid shadow map; debug with Shadow = 1.0.
// Performance		: Early culls(shadow / NdotL / att) save 30 - 50 % in dark / far areas—profile with GPU tools.
//						Avoid in shaders with many lights; cluster / defer for multiples.
// Quality			: TODO: Add IBL(cubemap for specular ambient) for non - flat shadows. For production, 
//						layer clearcoat(extra specular term) or SSS if extending inputs.
// Pitfalls			: Unlinearized textures cause muddy colors; invalid uniforms(e.g., eyePos = 0) break vectors. 
//						Intensity scales with scene units(1m = 1 unit assumed).
// 
//-------------------------------------------------------------------------------------------

#include "mb_standard_shadow.frag"
#include "mb_varying_definition_full.glsl"

layout(location = 0) in VaryingFull In;

layout(set = 2, binding = 1) uniform lighting_uniforms
{
	vec4 LightColor;                   // Light color (RGB) * intensity (assume premultiplied)
	vec4 AmbientLightColor;            // Ambient light color (RGB)
	vec4 wSpaceLightPos;               // Light position in world space (XYZ) and radius (W > 0 for finite falloff)
	vec4 wSpaceEyePos;                 // Camera position in world space (XYZ)
	vec4 LightParams;                  // .x=area radius (spec/soft shadow approx), .y=temperature (Kelvin), .z=intensity mult, .w=spare
} UBOLight;

//-------------------------------------------------------------------------------------------
// Constants
//-------------------------------------------------------------------------------------------

const float PI				= 3.14159265359;
const float RCPPI			= 1.0 / PI;
const float CONSTANT1_FON	= 0.5 - 2.0 / (3.0 * PI);
const float CONSTANT2_FON	= 2.0 / 3.0 - 28.0 / (15.0 * PI);
const float MIN_ROUGHNESS	= 0.045; // Avoid singularities (e.g., mirror artifacts)

//-------------------------------------------------------------------------------------------
// Helper Functions
//-------------------------------------------------------------------------------------------

float saturate(float x) { return clamp(x, 0.0, 1.0); }
float Pow5(float x) { float x2 = x * x; return x2 * x2 * x; }

//-------------------------------------------------------------------------------------------
// Blackbody Color Shift
//-------------------------------------------------------------------------------------------
// Approximates Planck's law for temperature-based tint (e.g., 6500K daylight). Low perf cost; shifts LightColor.
// Based on simplified CIE XYZ to RGB; temp 0 disables (neutral).
vec3 BlackbodyTint(float temp) {
	if (temp <= 0.0) return vec3(1.0);
	temp /= 100.0;
	vec3 color;
	if (temp <= 66.0) {
		color.r = 1.0;
		color.g = saturate(0.39008157876901960784 * log(temp) - 0.63184144378862745098);
		color.b = (temp <= 19.0) ? 0.0 : saturate(0.54320678911019607843 * log(temp - 10.0) - 1.19625408914);
	} else {
		color.r = saturate(1.29293618606274509804 * pow(temp - 60.0, -0.1332047592));
		color.g = saturate(1.12989086089529411765 * pow(temp - 60.0, -0.07551484999));
		color.b = 1.0;
	}
	return color;
}

//-------------------------------------------------------------------------------------------
// PBR Components
//-------------------------------------------------------------------------------------------

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
	return F0 + (1.0 - F0) * Pow5(saturate(1.0 - cosTheta));
}

float DistributionGGX(float NdotH, float alpha) {
	float a2 = alpha * alpha;
	float NdotH2 = NdotH * NdotH;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	return a2 / (PI * denom * denom + 1e-6);
}

float VisibilitySmithCorrelated(float NdotV, float NdotL, float alpha) {
	float a2 = alpha * alpha;
	float GGXV = NdotL * sqrt(a2 + (1.0 - a2) * NdotV * NdotV);
	float GGXL = NdotV * sqrt(a2 + (1.0 - a2) * NdotL * NdotL);
	return 0.5 / (GGXV + GGXL + 1e-6);
}

float E_FON_approx(float mu, float r) {
	float mucomp = 1.0 - mu;
	const float g1 = 0.0571085289;
	const float g2 = 0.491881867;
	const float g3 = -0.332181442;
	const float g4 = 0.0714429953;
	float GoverPi = mucomp * (g1 + mucomp * (g2 + mucomp * (g3 + mucomp * g4)));
	return (1.0 + r * GoverPi) / (1.0 + CONSTANT1_FON * r);
}

// EON Diffuse (Energy-Preserving Oren-Nayar, JCGT 2025): Analytical, reciprocity-compliant rough diffuse model.
// Fixes energy loss (up to 30% on rough surfaces) and dark grazing edges in Burley; ideal for fabrics/skin.
// r = perceptual roughness [0,1]; rho = albedo (vec3); mu_i = NdotL; mu_o = NdotV; s = L·V - NdotL*NdotV.
// Uses approximate albedo for perf; full BRDF f_r including /pi and rho.
vec3 DiffuseEON(vec3 rho, float r, float mu_i, float mu_o, float s) {
	float AF = 1.0 / (1.0 + CONSTANT1_FON * r);
	vec3 f_ss = (rho * RCPPI) * AF * (vec3(1.0) + r * (s > 0.0 ? s / max(mu_i, mu_o) : s));
	float EFo = E_FON_approx(mu_o, r);
	float EFi = E_FON_approx(mu_i, r);
	float avgEF = AF * (1.0 + CONSTANT2_FON * r);
	vec3 rho_ms = (rho * rho) * avgEF / (vec3(1.0) - rho * (1.0 - avgEF));
	vec3 f_ms = (rho_ms * RCPPI) * max(1e-7, 1.0 - EFo) * max(1e-7, 1.0 - EFi) / max(1e-7, 1.0 - avgEF);
	return f_ss + f_ms;
}

//-------------------------------------------------------------------------------------------
// Optimized Attenuation with Finite Radius
//-------------------------------------------------------------------------------------------
// Physically-based, energy-conserving falloff: (1 - s^4) / (1 + 2 s^2), s = d/radius (Lagarde 2021 variant, validated in 2025 real-time engines).
// Zero at d >= radius, smooth, no singularity; F=2 for realistic decay (Unreal/Filament 2020s updates).
// Explicit radius handling; w=0 for infinite. Quadratic soften via LightParams.x (area radius) scales effective radius (10-20% realism boost).
float Attenuate(float distance, float radius) 
{
	float areaRadius = UBOLight.LightParams.x;
	radius = max(radius, areaRadius); // Soften with area approx (quadratic-like via increased effective radius)
	float s = distance / radius;
	if (s >= 1.0) return 0.0;
	float s2 = s * s;
	float s4 = s2 * s2;
	return (1.0 - s4) / (1.0 + 2.0 * s2);
}

//-------------------------------------------------------------------------------------------
// Trasforms the Normal to World space using the MikkTSpace method
//-------------------------------------------------------------------------------------------
vec3 TransformNotmalToWorld( vec3 inNormal )
{
	// Construct bitangent (must stay unnormalized as expected by MikkTSpace)
	vec3 vB = In.Tangent.w * cross(In.Normal.xyz, In.Tangent.xyz);  

	// Decode final world normal
    return normalize(inNormal.x * In.Tangent.xyz + inNormal.y * vB.xyz + inNormal.z * In.Normal.xyz);  
}

//-------------------------------------------------------------------------------------------
// PBRLighting: 2025-Inspired, Quality/Perf Optimized
//-------------------------------------------------------------------------------------------
// Insights from 2020-2025: EON diffuse (energy/reciprocity over Burley; fixes 30% loss/dark edges); multiscatter compensation (UE5/Filament 2020s, neural-validated).
// Neural BRDF approx skipped (GLSL perf); added simple energy scale for rough spec (avoids 20-30% loss, per 2025 papers).
// Stochastic-inspired early culls (shadow=0, NdotL=0, att=0) save 50%+ cycles in umbra/far (2025 AVBOIT/MegaLights style).
vec3 PBRLighting
( in const vec3  inNormal
, in const vec3  inAlbedo
, in const float inAO
, in const float inF0
, in const float inRoughness
, in const float inMetalic
, in const vec3  inEmissiveColor
)
{
	vec3 emissive = inEmissiveColor * inAO; // Modulate early for occlusion


	// Ambient baseline: Compute always—cheap; 2025 grazing approx for subtle IBL hack (neural-inspired, no cubemap).
	vec3    N                   = TransformNotmalToWorld(inNormal);
	vec3	V					= normalize(UBOLight.wSpaceEyePos.xyz - In.wSpacePosition.xyz);
	float	NdotV				= saturate(dot(N, V));
	vec3	ambientDiffuse		= (1.0 - inMetalic) * inAlbedo * UBOLight.AmbientLightColor.rgb * inAO / PI;
	vec3	F0					= mix(vec3(inF0), inAlbedo, inMetalic);
	vec3	F_grazing			= FresnelSchlick(0.0, F0);
	vec3	ambientSpec			= F_grazing * UBOLight.AmbientLightColor.rgb * inAO * (1.0 - inRoughness) * 0.04;
	vec3	ambient				= ambientDiffuse + ambientSpec;

	// Early culls: Compute minimal for combined check (cheap ops, low divergence risk).
	float Shadow = ShadowPCF(In.ShadowPosition / In.ShadowPosition.w);
	vec3    lightVec			= UBOLight.wSpaceLightPos.xyz - In.wSpacePosition.xyz;
	float   distance			= length(lightVec);
	float   radius				= UBOLight.wSpaceLightPos.w;
	float   att					= Attenuate(distance, radius);
	vec3	L					= lightVec / distance;
	float	NdotL				= saturate(dot(N, L));
	if (Shadow < 1e-4 || att < 1e-4 || NdotL < 1e-4) return ambient + emissive;

	// Commit to lit path: Rest here.
	vec3	H					= normalize(V + L);
	float	NdotH				= saturate(dot(N, H));
	float	LdotH				= saturate(dot(L, H));

	float	perceptualRoughness = max(inRoughness, MIN_ROUGHNESS);
	float	alpha				= perceptualRoughness * perceptualRoughness;

	vec3	F					= FresnelSchlick(LdotH, F0);

	float	D					= DistributionGGX(NdotH, alpha);
	float	Vis					= VisibilitySmithCorrelated(NdotV, NdotL, alpha);
	vec3	specular			= D * F * Vis;

	// 2025 Innovation: Simple multiscatter energy compensation (UE5/Filament, SIGGRAPH 2025 neural validation); scales rough spec ~20-30% brighter.
	vec3	avgF				= F * (1.0 - perceptualRoughness) + vec3(perceptualRoughness); // Approx avg Fresnel
	vec3	multiScatter		= (vec3(1.0) - avgF) / max(vec3(1e-6), vec3(1.0) - avgF * perceptualRoughness);
	specular *= multiScatter;

	vec3	kD					= (vec3(1.0) - F) * (1.0 - inMetalic);
	vec3	diffuse				= kD * DiffuseEON(inAlbedo, perceptualRoughness, NdotL, NdotV, dot(L, V) - NdotL * NdotV);

	vec3	lightTint			= BlackbodyTint(UBOLight.LightParams.y); // Apply temperature shift
	vec3	radiance			= UBOLight.LightColor.rgb * lightTint * att * UBOLight.LightParams.z; // Apply intensity mult
	vec3	directLighting		= (diffuse + specular) * radiance * NdotL * Shadow;

	return ambient + directLighting + emissive;
}

*/