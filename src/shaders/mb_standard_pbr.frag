// Copied from xgeom_static.plugin/source/runtime (the PBR the xmaterial graph builds on) - seed of the shared
// contract shaders, see xLIONRender/documentation/xlion_renderer_research.md. Local change: directional lights.
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
// wSpaceLightPos	: XYZ position, W radius(> 0 for finite falloff, 0 for infinite 1 / dist�). 
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
// Performance		: Early culls(shadow / NdotL / att) save 30 - 50 % in dark / far areas�profile with GPU tools.
//						Avoid in shaders with many lights; cluster / defer for multiples.
// Quality			: TODO: Add IBL(cubemap for specular ambient) for non - flat shadows. For production, 
//						layer clearcoat(extra specular term) or SSS if extending inputs.
// Pitfalls			: Unlinearized textures cause muddy colors; invalid uniforms(e.g., eyePos = 0) break vectors. 
//						Intensity scales with scene units(1m = 1 unit assumed).
// 
//-------------------------------------------------------------------------------------------

// The xmaterial graph's "Output: PBR" includes this file: XMAT_VARYING_IN when its "Input: Vertex Variants" step already declared In,
// XLION_NO_SHADOW_MAP in its LDR variant (the editors bind the graph's textures from sampler 0 there: no shadow map, unshadowed)
#ifndef XLION_NO_SHADOW_MAP
#include "mb_standard_shadow.frag"
#endif
#include "mb_varying_definition_full.glsl"

#ifndef XMAT_VARYING_IN
layout(location = 0) in VaryingFull In;
#endif
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
	vec4 Shadow;                       // x: its first face's entry in the local pages' table, z: 1 shadowed, w: spot tan of the half field of view
};
#endif

layout(set = 2, binding = 1) uniform lighting_uniforms
{
	vec4 LightColor;                   // Light color (RGB) * intensity (assume premultiplied)
	vec4 AmbientLightColor;            // Ambient light color (RGB)
	vec4 wSpaceLightPos;               // Light position in world space (XYZ) and radius (W > 0 for finite falloff)
	vec4 wSpaceEyePos;                 // Camera position in world space (XYZ)
	vec4 LightParams;                  // .x=area radius (spec/soft shadow approx), .y=temperature (Kelvin), .z=intensity mult, .w=1 a shadow caster pass (a masked material stops after its cut)
#ifndef XLION_LDR_OUTPUT
	// xLIONRender (plans/forward/xlionrender_plan_forward.h ubo_lighting): the directional light's shadow clipmap
	mat4 W2Light;                      // world -> light space (rotation)
	vec4 CameraPos;                    // world position of the camera (wSpacePosition is camera-centered)
	vec4 ShadowLevels[4];              // reserved (the sun's clipmap, gone 2026-10-27): kept so the compiled materials' layout holds
	vec4 ShadowParams;                 // reserved (as ShadowLevels)
	vec4 LocalCount;                     // x: local lights, y: mode (0 exact, 1 cells, 2 cells + sampled tail), z: seed, w: the tail's resampling candidates (> 1: RIS)
	vec4 LocalBase;                      // x: first light in LocalLightList, y: first uint in LightCells, z: level 0 cell size, w: tail samples
	vec4 CellOrigin[5];                  // per cell level: the window's first cell index (xyz) - features/lights/xlionrender_light_cells.h
	vec4 EnvParams;                      // x: 1 the environment probe replaces the flat ambient, y: its intensity
	vec4 ProbeCount;                     // x: reflection probes (innermost first), y: the first's record in LocalLightList, z: their cell headers in LightCells (-1: none)
	// the sun's virtual shadow pages (xlionrender_shadow_vsm.glsl; appended: materials compiled before them read the fields above only)
	vec4 VSMParams;                      // x: 1 the virtual pages (0: the clipmap), y: level 0 texel, z: levels, w: the page table's first uint in LightCells
	vec4 VSMParams2;                     // x: window (pages per level side), y: page texels, z: pool slots per row, w: 1 every tap through the table (tests)
	vec4 VSMLevels[12];                  // per level: xy the window's first page, z page size, w depth centre
	vec4 VSMParams3;                     // x: the resolution bias (levels added to the footprint's)
	mat4 VSMPrevW2Light;                 // the previous epoch (P1e): its world -> light, its levels, x: there is one, y: the current epoch's
	vec4 VSMPrevLevels[12];              // weight (the cross-fade), z: its level 0 texel, w: its table's first uint
	vec4 VSMPrevParams;
#endif
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
// r = perceptual roughness [0,1]; rho = albedo (vec3); mu_i = NdotL; mu_o = NdotV; s = L�V - NdotL*NdotV.
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
    const vec3 N = normalize(inNormal.x * In.Tangent.xyz + inNormal.y * vB.xyz + inNormal.z * In.Normal.xyz);
#ifdef XMAT_BACK_FACE_AS_FRONT
	return N;                           // Back Face Normal: None (foliage cards lit as the front)
#else
	return gl_FrontFacing ? N : -N;     // a back face (double sided: the material's or an instance's setting) is lit by its own side; culled, never taken
#endif  
}

#ifndef XLION_LDR_OUTPUT
#include "xlionrender_local_lights.glsl"
#include "xlionrender_environment.glsl"
#include "xlionrender_shadow_vsm.glsl"
#endif
#ifndef XLIONRENDER_SHADOW_VSM_GLSL
#define VSMDebugOr(Lit) (Lit)
#endif

#ifndef XLION_LDR_OUTPUT
#ifndef XLION_NO_GI_PIXEL
// The GI read at half resolution (xlionrender_gi_screen_frag.glsl's buffer, bound as GIMesh), upsampled to this pixel: its 4 half
// resolution neighbors, bilinear, each weighed by how close its distance to the eye is to this pixel's and how much its normal agrees
// (an edge keeps its side's light). g_GI and g_GIRefl, as the full read gives them; none of the 4 alike: the most alike
void SurfelGIScreen(in const vec3 Ng, in const float Dist)
{
    const uint W = GIMesh[0], H = GIMesh[1];
    if (W == 0u || H == 0u) return;
    const vec2  hp = gl_FragCoord.xy * 0.5 - 0.5;
    const vec2  i0 = floor(hp), f = hp - i0;
    vec4 Se = vec4(0.0), Sr = vec4(0.0);
    float Best = -1.0; uvec4 BestQ = uvec4(0u);
    for (int k = 0; k < 4; ++k)
    {
        const ivec2 P = clamp(ivec2(i0) + ivec2(k & 1, k >> 1), ivec2(0), ivec2(W - 1u, H - 1u));
        const uint  a = 4u * (8u + uint(P.y) * W + uint(P.x));     // (after the header: xlionrender_gi_screen_frag.glsl)
        const uvec4 Q = uvec4(GIMesh[a], GIMesh[a + 1u], GIMesh[a + 2u], GIMesh[a + 3u]);
        if ((Q.w & (1u << 30)) == 0u) continue;
        const float d = unpackHalf2x16(Q.y).y;
        const vec2  o = vec2(float((Q.w >> 16) & 63u), float((Q.w >> 22) & 63u)) / 63.0 * 2.0 - 1.0;     // (6 + 6 bits: xlionrender_gi_screen_frag.glsl)
        vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y));
        if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n = normalize(n);
        const float wd = max(0.0, 1.0 - abs(d - Dist) / max(0.05 * Dist, 1e-3)), wn = pow(max(dot(n, Ng), 0.0), 8.0);
        const float wb = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        const float w = wb * wd * wd * wn;
        if (wd * wn > Best) { Best = wd * wn; BestQ = Q; }
        const vec3 E = vec3(unpackHalf2x16(Q.x), unpackHalf2x16(Q.y).x);
        Se += vec4(E * w, w);
        if ((Q.w & (1u << 31)) != 0u) Sr += vec4(vec3(unpackHalf2x16(Q.z), unpackHalf2x16(Q.w & 0xFFFFu).x) * w, w);
    }
    if (Se.w > 1e-4) { g_GI = vec4(Se.rgb / Se.w, 1.0); g_GISource = 2u; }
    else if (Best >= 0.0) { g_GI = vec4(unpackHalf2x16(BestQ.x), unpackHalf2x16(BestQ.y).x, 1.0); g_GISource = 2u; }
    if (Sr.w > 1e-4) g_GIRefl = vec4(Sr.rgb / Sr.w, 1.0);
}

#endif

//-------------------------------------------------------------------------------------------
// The GI at this pixel (g_GI, g_GIRefl, g_GIDebug, g_GISource): the vertex read, the probe grid, the pixel read of the surfels, the
// local mean. Its own function: the half-resolution GI pass (xlionrender_gi_screen_frag.glsl) runs the same read
//-------------------------------------------------------------------------------------------
// The face normal's derivative product, taken by a caller where its whole quad still runs (xlionrender_gi_screen_frag.glsl: most of its
// lanes return before the read; a derivative after that is undefined)
bool g_bGIFaceD = false;
vec3 g_GIFaceD  = vec3(0.0);
void GIPixelGI(in const vec3 inNormal, in const vec3 N, in const vec3 V, in const vec3 inAlbedo, in const float inMetalic, in const vec3 inEmissiveColor)
{
#ifndef XLION_NO_GI_PIXEL
	if (InGIRef.y == gi_screen_v) { SurfelGIScreen(normalize(In.Normal.xyz) * (gl_FrontFacing ? 1.0 : -1.0), length(In.wSpacePosition.xyz)); return; }     // read at half resolution: upsampled
#endif
	if (InGI.w > 1e-6) { g_GI = vec4(InGI.rgb / InGI.w, 1.0); g_GISource = 1u; }
#ifndef XLION_NO_GI_PIXEL
	if (InGIRef.y == gi_probes_v) { g_GI = SurfelGIProbes(InGIPos, N); g_GISource = g_GI.w > 0.5 ? 1u : 0u; }     // no surfels of its own (a skinned mesh): the probe grid
	const uvec2 GIRef = InGIRef.y == gi_probes_v ? uvec2(0u, gi_none_v) : InGIRef;          // (the surfel reads below: none)
	// The face's normal (mesh space, to the smoothed normal's side): the GI's geometry tests use the surface as it is - smoothed corner
	// normals step along the triangles (a sawtooth); the normal map still reads the basis (GNmap)
	vec3 GFaceD;
	if (g_bGIFaceD) GFaceD = g_GIFaceD; else GFaceD = cross(dFdx(InGIPos), dFdy(InGIPos));
	const vec3 GN0    = normalize(InGINormal);
	vec3       GFace  = dot(GFaceD, GFaceD) > 1e-24 ? normalize(dot(GFaceD, GN0) < 0.0 ? -GFaceD : GFaceD) : GN0;
	if (dot(GFace, GN0) < 0.7) GFace = GN0;          // a pixel quad across a crease (a floor's edge at a wall's foot): its derivatives mix both faces
	if (GIRef.y != gi_none_v && (GIRef.y & gi_per_pixel_v) != 0u)     // the pixel reads the surfels itself (far: through their parents)
	{
		const vec3 GN = normalize(InGINormal), GT = normalize(InGITangent.xyz);
		const vec3 GNmap = normalize(inNormal.x * GT + inNormal.y * (InGITangent.w * cross(GN, GT)) + inNormal.z * GN);     // as TransformNotmalToWorld, in mesh space
		g_GIMoments = (GIRef.y & gi_no_moments_v) == 0u;
		g_GIPixelCap = GIPixelCap(GIRef.y);
		g_GINoCorners = (GIRef.y & 0x01000000u) != 0u;
		// The reflection direction in mesh space (through the two tangent frames: the world's and the mesh's), for the rough reflection's read
		{
			const vec3 Rw = reflect(-V, N), Tw = normalize(In.Tangent.xyz - In.Normal.xyz * dot(In.Normal.xyz, In.Tangent.xyz)), Nw = normalize(In.Normal.xyz), Bw = In.Tangent.w * cross(Nw, Tw);
			const vec3 Tm = normalize(GT - GN * dot(GN, GT)), Bm = InGITangent.w * cross(GN, Tm);
			g_GIReflDir = normalize(dot(Rw, Tw) * Tm + dot(Rw, Bw) * Bm + dot(Rw, Nw) * GN);
		}
		// The policy for a read (GIQueryResult): the pixel read where it is supported - a low confidence (the surfels near are mostly out of
		// sight) blends into the vertex read where that one is supported (no edge); unsupported, the vertex read where it is; else no GI
		const GIQueryResult Q = SurfelGIQuery(InGIPos, GFace, GNmap, GIRef.x, GIRef.y & 0xFFu);
		if (Q.bSupported) { g_GI = vec4(g_GI.w > 0.5 ? mix(g_GI.rgb, Q.Irradiance, smoothstep(0.0, 0.25, Q.Confidence)) : Q.Irradiance, 1.0); g_GISource = 2u; }
		if (!Q.bSupported || Q.Confidence < 0.25) g_GIRefl = vec4(0.0);      // (a weak read: the reflection stays the probes')
	}
	// Neither read supported (no surfel reaches the point, or only invalid ones): its cell's mean light (SurfelGILocalMean), not the no-GI
	// ambient (a scene lit by its GI is black without it). COVERAGE still shows it unsupported
	if (GIRef.y != gi_none_v && g_GISource == 0u && (GIRef.y & 0x8000u) == 0u) g_GI = vec4(SurfelGILocalMean(InGIPos, GFace, GIRef.x, GIRef.y & 0xFFu), 1.0);
	const uint GIDebug = GIDebugCode(GIRef.y);
	if (GIDebug == 1u) { const uint Sd = SurfelGIDominant(InGIPos, GFace); g_GIDebug = vec4(Sd == 0xFFFFFFFFu ? vec3(0.0) : SurfelDebugColor(Sd), 1.0); }
	if (GIDebug == 2u) g_GIDebug = vec4(g_GI.w > 0.5 ? g_GI.rgb / PI : vec3(0.0), 1.0);
	if (GIDebug == 3u) g_GIDebug = vec4((GIRef.y & gi_capture_emission_v) != 0u ? inEmissiveColor : inAlbedo * (1.0 - inMetalic), 1.0);
	if (GIDebug == 5u) g_GIDebug = vec4(g_GISource == 2u ? (SurfelGIIsPatch(SurfelGIDominant(InGIPos, GFace)) ? vec3(1.0, 0.85, 0.0) : vec3(0.0, 0.6, 0.0)) : g_GISource == 1u ? vec3(0.0, 0.0, 0.8) : (GIRef.y & gi_per_pixel_v) != 0u ? SurfelGIWhyUnsupported(InGIPos, GFace, GIRef.x, GIRef.y & 0xFFu) : vec3(1.0, 0.0, 1.0), 1.0);     // yellow: a patch surfel (PatchDefects: a defect of the model); unsupported: why (SurfelGIWhyUnsupported)     // COVERAGE: green the pixel read, blue the vertex read, magenta unsupported (the mean light)
	if (GIDebug == 4u) g_GIDebug = vec4(SurfelGITriangleColor(GIRef.x, uint(gl_PrimitiveID)), 1.0);        // the pixel's triangle's captured color     // the GI's material capture (xlionrender_gi_capture_vert.glsl)
#endif
}
#endif

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


	// Ambient baseline: Compute always�cheap; 2025 grazing approx for subtle IBL hack (neural-inspired, no cubemap).
	vec3    N                   = TransformNotmalToWorld(inNormal);
	vec3	V					= normalize(UBOLight.wSpaceEyePos.xyz - In.wSpacePosition.xyz);
	float	NdotV				= saturate(dot(N, V));
	vec3	ambientDiffuse		= (1.0 - inMetalic) * inAlbedo * UBOLight.AmbientLightColor.rgb * inAO / PI;
	vec3	F0					= mix(vec3(inF0), inAlbedo, inMetalic);
	vec3	F_grazing			= FresnelSchlick(0.0, F0);
	vec3	ambientSpec			= F_grazing * UBOLight.AmbientLightColor.rgb * inAO * (1.0 - inRoughness) * 0.04;
	vec3	ambient				= ambientDiffuse + ambientSpec;
#ifndef XLION_LDR_OUTPUT
	GIPixelGI(inNormal, N, V, inAlbedo, inMetalic, inEmissiveColor);
	if (UBOLight.EnvParams.x > 0.5 || UBOLight.ProbeCount.x > 0.5 || g_GI.w > 0.5) ambient = EnvLighting(N, V, NdotV, inAlbedo, F0, max(inRoughness, MIN_ROUGHNESS), inMetalic, inAO);   // the sky probe
#endif

	// Early culls: Compute minimal for combined check (cheap ops, low divergence risk).
#if defined(XLION_LDR_OUTPUT) && defined(XLION_NO_SHADOW_MAP)
	float Shadow = 1.0;
#elif defined(XLION_LDR_OUTPUT)
	float Shadow = ShadowPCF(In.ShadowPosition / In.ShadowPosition.w);       // editors that draw materials themselves: their own shadow map
#else
	// The sun's shadow: offset along, and biased by the slope of, the triangle's own plane (the screen derivatives: Unreal's
	// EstimatedGeoWorldNormal) - the interpolated normal of a faceted curved surface is not the plane its depth lies in (acne at the terminator)
	vec3 FaceN = cross(dFdx(In.wSpacePosition.xyz), dFdy(In.wSpacePosition.xyz));
	const vec3 VtxN = normalize(In.Normal.xyz) * (gl_FrontFacing ? 1.0 : -1.0);
	FaceN = dot(FaceN, FaceN) > 1e-20 ? normalize(FaceN) * sign(dot(FaceN, VtxN) + 1e-9) : VtxN;
	g_LocalFaceN = FaceN;                                                                  // (the local lights' shadows: LocalShadow)
	g_LocalFoot  = max(length(dFdx(In.wSpacePosition.xyz)), length(dFdy(In.wSpacePosition.xyz)));
	// (a surface facing away from the sun returns before the shadow is used - the early cull below, the same NdotL -: no lookups for
	// it, unless a debug view shows the pages; the footprint's derivatives stay outside the branch)
	VSMFootprint(In.wSpacePosition.xyz + UBOLight.CameraPos.xyz);
	const bool bSunFacing = saturate(dot(N, normalize(UBOLight.wSpaceLightPos.xyz))) >= 1e-4 || UBOLight.VSMParams2.w >= 2.0;
	float Shadow = UBOLight.wSpaceLightPos.w == 0.0 ? (bSunFacing ? SunShadow(In.wSpacePosition.xyz + UBOLight.CameraPos.xyz, FaceN) : 0.0) : 1.0;
#endif
	// xLIONRender: W == 0 is a directional light (XYZ = direction towards the light, no falloff); W > 0 a point
	// light with that falloff radius. (The original treated W == 0 as a zero-radius point light: always black.)
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
	if (Shadow < 1e-4 || att < 1e-4 || NdotL < 1e-4) return VSMDebugOr(GIDebugOr(ambient + Local + emissive));

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

	// NOTE: a multiscatter energy-compensation term used to sit here, intended to brighten rough
	// specular ~20-30%. Its formula did the opposite - multiScatter = (1-avgF)/(1-avgF*roughness)
	// is <1 for every roughness>0 and degenerates to exactly 0 at roughness=1 (denominator ->
	// 1-avgF, numerator -> 1-avgF, but the max(1e-6,...) floor wins first), so it was silently
	// killing specular instead of boosting it. Removed rather than "corrected" in place - a real
	// multiscatter compensation term needs validating against a reference render, which a comment
	// fix alone can't provide. Plain Cook-Torrance (D*F*Vis) below is the safe fallback.

	vec3	kD					= (vec3(1.0) - F) * (1.0 - inMetalic);
	vec3	diffuse				= kD * DiffuseEON(inAlbedo, perceptualRoughness, NdotL, NdotV, dot(L, V) - NdotL * NdotV);

	vec3	lightTint			= BlackbodyTint(UBOLight.LightParams.y); // Apply temperature shift
	vec3	radiance			= UBOLight.LightColor.rgb * lightTint * att * UBOLight.LightParams.z; // Apply intensity mult
	vec3	directLighting		= (diffuse + specular) * radiance * NdotL * Shadow;

	return VSMDebugOr(GIDebugOr(ambient + directLighting + Local + emissive));
}