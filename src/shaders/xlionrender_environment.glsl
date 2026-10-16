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
uint g_GISource = 0u;      // which read set g_GI (the COVERAGE debug view): 0 none (the ambient, as without GI), 1 the vertex read, 2 the pixel read

const vec2 kProbeAtlas = vec2(3072.0, 1024.0);          // 8 x 4 slots of the sky atlas' layout (384 x 256) - instance::probe_slots_v

vec3 AtlasSample(in const sampler2D Atlas, in const vec2 Origin, in const vec2 AtlasSize, in const vec3 Dir, in const float PerceptualRoughness)
{
    const float L  = clamp(PerceptualRoughness, 0.0, 1.0) * float(kEnvLevels - 1);
    const int   L0 = int(floor(L));
    const int   L1 = min(L0 + 1, kEnvLevels - 1);
    const vec2  S  = kEnvAtlas / AtlasSize;                  // a level's uv in the sky layout -> the atlas' uv
    const vec3  A  = textureLod(Atlas, Origin / AtlasSize + EnvLevelUV(L0, Dir) * S, 0.0).rgb;
    const vec3  B  = textureLod(Atlas, Origin / AtlasSize + EnvLevelUV(L1, Dir) * S, 0.0).rgb;
    return mix(A, B, L - float(L0));
}

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
    const vec3 Inv = 1.0 / max(abs(R), vec3(1e-6)) * sign(R + 1e-12);
    const vec3 T  = max((Lo - P) * Inv, (Hi - P) * Inv);
    const float t = min(T.x, min(T.y, T.z));
    return normalize(P + R * t - Pr.PosRadius.xyz);
}

// One probe into the running sums (its weight times what the earlier, inner probes left)
void ProbeAdd(in const int Record, in const vec3 P, in const vec3 N, in const vec3 R, in const float PerceptualRoughness, inout vec3 Spec, inout vec3 Diff, inout float Covered)
{
    const LocalLightData Pr = LocalLightList[Record];
    const float w = ProbeWeight(Pr, P) * (1.0 - Covered);
    if (w <= 0.0) return;
    Spec    += w * AtlasSample(SamplerProbes, Pr.Shadow.xy, kProbeAtlas, ProbeBoxProject(Pr, P, R), PerceptualRoughness);
    Diff    += w * AtlasSample(SamplerProbes, Pr.Shadow.xy, kProbeAtlas, N, 1.0);
    Covered += w;
}

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
            Diff += Rest * EnvSample(N, 1.0) * UBOLight.EnvParams.y;
        }
        else
        {
            Spec += Rest * UBOLight.AmbientLightColor.rgb / 3.14159265359;     // the flat ambient's radiance
            Diff += Rest * UBOLight.AmbientLightColor.rgb / 3.14159265359;
        }
    }

    if (g_GI.w > 0.5)
    {
        // The GI knows how much light reaches this point: where it gets a small part of what the sky alone would give (an arcade, a corner) the
        // sky and probe reflection (Spec) is mostly occluded too. Without this, the Fresnel bias of the split-sum (large at grazing angles) times an
        // unoccluded sky is a veil that washes out a wall the GI shows as dark (specular occlusion, from the ratio of the GI light to the open sky's)
        const vec3  GIDiff = g_GI.rgb / 3.14159265359;            // irradiance -> the diffuse term's radiance
        const float SkyLum = dot(Diff, vec3(0.2126, 0.7152, 0.0722));
        const float GILum  = dot(GIDiff, vec3(0.2126, 0.7152, 0.0722));
        Spec *= SkyLum > 1e-4 ? clamp(GILum / SkyLum, 0.0, 1.0) : 1.0;
        Diff  = GIDiff;
    }
    const vec2 DFG = textureLod(SamplerEnvDFG, vec2(NdotV, PerceptualRoughness), 0.0).rg;
    return (Diff * Albedo * (1.0 - Metallic) + Spec * (F0 * DFG.x + DFG.y)) * AO;
}
