// Image-based light: the reflection probes (scene::m_Probes, captured by the renderer) blended by their boxes, then the sky probe
// (UBOLight.EnvParams.x; scene::m_Sky) or the flat ambient for what the probes leave. Specular: the prefiltered level for the
// roughness along the reflection (box-projected for a probe: parallax), times the split-sum DFG (Karis); diffuse: the roughest
// level along the normal (an approximation of the irradiance - ponytail: SH9 when it shows).
// Samplers 8 (sky atlas), 9 (DFG), 10 (probe atlas) in every lit pipeline (slots 0-7: the shadow atlas and the material's textures).
// The probes' records share the light buffer (64 bytes, as a LocalLightData: PosRadius = center + blend, Color = box min,
// DirCosOuter = box max, Shadow = atlas slot); the world cell of a point lists the probes whose box reaches it (innermost first).
#include "xlionrender_environment_atlas.glsl"

layout(binding = 8)  uniform sampler2D SamplerEnvironment;
layout(binding = 9)  uniform sampler2D SamplerEnvDFG;
layout(binding = 10) uniform sampler2D SamplerProbes;

// Surfel GI (features/gi/xlionrender_surfel_gi.h): the diffuse irradiance at this pixel, set by the fragment shaders that have it
// (xlionrender_geom_standard_frag.glsl); w > 0.5: it replaces the probes' and the sky's diffuse (it holds the sky and the bounces)
vec4 g_GI = vec4(0.0);
#ifndef XLIONRENDER_SURFEL_GI_GLSL
vec4 g_GIRefl = vec4(0.0);  // (xlionrender_surfel_gi.glsl: the surfels' light around the reflection direction; w > 0.5: read)
#endif
uint g_GISource = 0u;      // which read set g_GI (the COVERAGE debug view): 0 none (the ambient, as without GI), 1 the vertex read, 2 the pixel read

// The reflection debug view (scene::m_bDebugReflection, UBOLight.EnvParams.z): the pixel shows only the receiver's environment / probe /
// reflection specular - R^ of documentation/reflections_strategy.md sec. 5 (the BRDF weighting included); black where none was evaluated
vec3 g_EnvSpecular = vec3(0.0);
#define ReflDebugOr(Lit) (UBOLight.EnvParams.z > 0.5 ? g_EnvSpecular : (Lit))

const vec2 kProbeAtlas = vec2(3072.0, 1024.0);          // 8 x 4 slots of the sky atlas' layout (384 x 256) - instance::probe_slots_v

vec3 EnvSample(in const vec3 Dir, in const float PerceptualRoughness)
{
    return AtlasSample(SamplerEnvironment, vec2(0.0), kEnvAtlas, Dir, PerceptualRoughness);
}

// How much a probe covers a camera-centered point: 1 inside its box, fading to 0 over the blend distance at its faces
float ProbeWeight(in const LocalLightData Pr, in const vec3 P)
{
    const vec3 Lo = Pr.Color.xyz, Hi = Pr.DirCosOuter.xyz;
    if (any(lessThan(P, Lo)) || any(greaterThan(P, Hi))) return 0.0;
    const vec3 Inside = min(P - Lo, Hi - P);
    return clamp(min(Inside.x, min(Inside.y, Inside.z)) / max(Pr.PosRadius.w, 1e-4), 0.0, 1.0);
}

// The reflection ray from P meets the probe's box: the direction from the probe's center to that point (parallax correction)
vec3 ProbeBoxProject(in const LocalLightData Pr, in const vec3 P, in const vec3 R)
{
    const vec3 Lo = Pr.Color.xyz, Hi = Pr.DirCosOuter.xyz;
    float t = 1e30;
    for (int a = 0; a < 3; ++a)                                 // the exit face per axis; an axis the ray runs along has none
        if (abs(R[a]) > 1e-6) t = min(t, ((R[a] > 0.0 ? Hi[a] : Lo[a]) - P[a]) / R[a]);
    const vec3 D = P + R * t - Pr.PosRadius.xyz;
    return t > 0.0 && t < 1e29 && dot(D, D) > 1e-12 ? normalize(D) : R;     // (outside the box, degenerate: the plain reflection)
}

// One probe into the running sums (its weight times what the earlier, inner probes left)
void ProbeAdd(in const int Record, in const vec3 P, in const vec3 N, in const vec3 R, in const float PerceptualRoughness, inout vec3 Spec, inout vec3 Diff, inout float Covered)
{
    const LocalLightData Pr = LocalLightList[Record];
    const float w = ProbeWeight(Pr, P) * (1.0 - Covered);
    if (w <= 0.0) return;
    Spec    += w * AtlasSample(SamplerProbes, Pr.Shadow.xy, kProbeAtlas, ProbeBoxProject(Pr, P, R), PerceptualRoughness);
    Diff    += w * AtlasSample(SamplerProbes, Pr.Shadow.xy, kProbeAtlas, g_GIRefl.w > 0.5 ? ProbeBoxProject(Pr, P, R) : N, 1.0);
    Covered += w;
}

#ifndef XLION_TAIL_SCREEN
// The traced reflections (documentation/reflections_strategy.md sec. 5; xlionrender_refl_trace_comp.glsl): the records' second region,
// a half-size texel each (U: the lobe-mean radiance, its confidence, the view depth, the octahedral normal). The 4 texels around the
// pixel, bilinear, each weighed by how close its view depth is to this pixel's and how much its normal agrees (the tail's weights,
// LocalTailScreen) and by its confidence: rgb U, a the confidence here (0: none of the 4 is this surface - the probes stay)
vec4 ReflScreen(in const vec3 N)
{
    const int  W = int(UBOLight.TailParams.y), H = int(UBOLight.TailParams.z);
    const uint R0 = uint(UBOLight.TailParams.w) + uint(W * H);
    if (W <= 0 || R0 + uint(W * H) > uint(TailScreen.length())) return vec4(0.0);
    const int Wf = int(UBOLight.TailParams.x);      // the sharp receivers' full-size records (2 pixels a record), this pixel's when it is this surface
    if (Wf > 0)
    {
        const ivec2 q  = ivec2(gl_FragCoord.xy);
        const uint  k  = uint(q.y * ((Wf + 1) & ~1) + q.x);
        const uint  Rf = R0 + uint(W * H) + k / 2u;
        if (Rf < uint(TailScreen.length()))
        {
            const uvec4 Q = TailScreen[Rf];
            const uvec2 r = (k & 1u) != 0u ? Q.zw : Q.xy;
            const vec2  bd = unpackHalf2x16(r.y);
            const float Dist = 1.0 / gl_FragCoord.w;
            if (bd.y > 0.0 && abs(bd.y - Dist) < 0.05 * Dist) return vec4(unpackHalf2x16(r.x), bd.x, 1.0);
        }
    }
    const vec2  hp = gl_FragCoord.xy * 0.5 - 0.5;
    const vec2  i0 = floor(hp), f = hp - i0;
    const float Dist = 1.0 / gl_FragCoord.w;
    vec3  Su = vec3(0.0);
    float Sc = 0.0, Sw = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        const ivec2 P = clamp(ivec2(i0) + ivec2(k & 1, k >> 1), ivec2(0), ivec2(W - 1, H - 1));
        const uvec4 Q = TailScreen[R0 + uint(P.y * W + P.x)];
        const float d = unpackHalf2x16(Q.w & 0xFFFFu).x;
        if (d <= 0.0) continue;
        const vec3  n  = OctDecode(vec2(float((Q.w >> 16) & 15u), float((Q.w >> 20) & 15u)) / 15.0 * 2.0 - 1.0);
        const float wd = max(0.0, 1.0 - abs(d - Dist) / max(0.05 * Dist, 1e-3)), wn = pow(max(dot(n, N), 0.0), 8.0);
        const float wb = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        const float w  = wb * wd * wd * wn;
        const float c  = unpackHalf2x16(Q.y).y;
        Su += w * c * vec3(unpackHalf2x16(Q.x), unpackHalf2x16(Q.y).x);
        Sc += w * c; Sw += w;
    }
    return Sw > 1e-4 && Sc > 1e-4 ? vec4(Su / Sc, Sc / Sw) : vec4(0.0);
}
#endif

vec3 EnvLighting(in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic, in const float AO)
{
    const vec3 R = reflect(-V, N);
    const vec3 P = In.wSpacePosition.xyz;                   // camera-centered, as the probes' boxes

    vec3  Spec = vec3(0.0), Diff = vec3(0.0);
    float Covered = 0.0;
    const int nProbes     = int(UBOLight.ProbeCount.x);
    const int FirstRecord = int(UBOLight.ProbeCount.y);
    const int Cell        = (nProbes > 0 && UBOLight.ProbeCount.z >= 0.0) ? LocalLightCell(P + UBOLight.CameraPos.xyz) : -1;
    if (Cell >= 0)
    {
        const uint Base   = uint(UBOLight.LocalBase.y);
        const uint Header = LightCells[Base + uint(UBOLight.ProbeCount.z) + uint(Cell)];     // bits 0-23 first entry, 24-31 count
        const uint First  = Header & 0xFFFFFFu, Count = Header >> 24;
        for (uint k = 0u; k < Count && Covered < 0.999; ++k)
            ProbeAdd(FirstRecord + int(LightCells[Base + First + k]), P, N, R, PerceptualRoughness, Spec, Diff, Covered);
    }
    else
    {
        for (int i = 0; i < nProbes && Covered < 0.999; ++i) ProbeAdd(FirstRecord + i, P, N, R, PerceptualRoughness, Spec, Diff, Covered);
    }

    const float Rest = 1.0 - Covered;
    if (Rest > 0.0)
    {
        if (UBOLight.EnvParams.x > 0.5)
        {
            Spec += Rest * EnvSample(R, PerceptualRoughness) * UBOLight.EnvParams.y;
            Diff += Rest * EnvSample(g_GIRefl.w > 0.5 ? R : N, 1.0) * UBOLight.EnvParams.y;
        }
        else
        {
            Spec += Rest * UBOLight.AmbientLightColor.rgb / 3.14159265359;     // the flat ambient's radiance
            Diff += Rest * UBOLight.AmbientLightColor.rgb / 3.14159265359;
        }
    }

    if (g_GI.w > 0.5 && g_GIRefl.w > 0.5)
    {
        // Reflections layer 1 (gi_dynamic_master_plan.md): the surfels' light around the reflection direction. Against the probes' and the
        // sky's roughest level along it (Diff, sampled along R above), it says how much of what they show there reaches this point: their
        // reflection scaled down by it (a capture's normalization, as Unreal 4 scales its captures by the local GI - here per direction:
        // a dark corner facing a bright doorway keeps the doorway). Rough surfaces (their lobe about the basis' width) take the surfels'
        // light itself: the bounce colors the cache holds (a red wall in a rough floor), not the probes' view
        const vec3  GIRefl = g_GIRefl.rgb / 3.14159265359;        // irradiance around R -> its radiance
        const float EnvLum = dot(Diff, vec3(0.2126, 0.7152, 0.0722));
        const float GILum  = dot(GIRefl, vec3(0.2126, 0.7152, 0.0722));
        Spec *= EnvLum > 1e-4 ? clamp(GILum / EnvLum, 0.0, 1.0) : 1.0;
        Spec  = mix(Spec, GIRefl, smoothstep(0.3, 0.7, PerceptualRoughness));
        Diff  = g_GI.rgb / 3.14159265359;
    }
    else if (g_GI.w > 0.5)
    {
        // (no read along R - the vertex read: the heuristic below. Not a visibility: the diffuse light's ratio, all directions alike)
        // The GI knows how much light reaches this point: where it gets a small part of what the sky alone would give (an arcade, a corner) the
        // sky and probe reflection (Spec) is mostly occluded too. Without this, the Fresnel bias of the split-sum (large at grazing angles) times an
        // unoccluded sky is a veil that washes out a wall the GI shows as dark (specular occlusion, from the ratio of the GI light to the open sky's)
        const vec3  GIDiff = g_GI.rgb / 3.14159265359;            // irradiance -> the diffuse term's radiance
        const float SkyLum = dot(Diff, vec3(0.2126, 0.7152, 0.0722));
        const float GILum  = dot(GIDiff, vec3(0.2126, 0.7152, 0.0722));
        Spec *= SkyLum > 1e-4 ? clamp(GILum / SkyLum, 0.0, 1.0) : 1.0;
        Diff  = GIDiff;
    }
#ifndef XLION_TAIL_SCREEN
    if (UBOLight.EnvParams.w > 0.0) { const vec4 U = ReflScreen(N); Spec = mix(Spec, U.rgb, U.a); }      // the traced reflections replace the probes where they cover
#endif
    const vec2 DFG = textureLod(SamplerEnvDFG, vec2(NdotV, PerceptualRoughness), 0.0).rg;
    g_EnvSpecular = Spec * (F0 * DFG.x + DFG.y) * AO;
    return (Diff * Albedo * (1.0 - Metallic) + Spec * (F0 * DFG.x + DFG.y)) * AO;     // (the lit sum as before, bitwise)
}
