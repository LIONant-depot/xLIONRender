// Image-based light: the reflection probes (scene::m_Probes, captured by the renderer) blended by their boxes, then the sky probe
// (UBOLight.EnvParams.x; scene::m_Sky) or the flat ambient for what the probes leave. Specular: the prefiltered level for the
// roughness along the reflection (box-projected for a probe: parallax), times the split-sum DFG (Karis); diffuse: the roughest
// level along the normal (an approximation of the irradiance - ponytail: SH9 when it shows).
// Samplers 8 (sky atlas), 9 (DFG), 10 (probe atlas) in every lit pipeline (slots 0-7: the shadow atlas and the material's textures).
// ponytail: probes are a loop over the lighting block's list (up to 16, innermost first); the world cells list them when there are many.
#include "xlionrender_environment_atlas.glsl"

layout(binding = 8)  uniform sampler2D SamplerEnvironment;
layout(binding = 9)  uniform sampler2D SamplerEnvDFG;
layout(binding = 10) uniform sampler2D SamplerProbes;

const vec2 kProbeAtlas = vec2(1536.0, 1024.0);          // 4 x 4 slots of the sky atlas' layout (384 x 256)

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

// How much probe i covers a camera-centered point: 1 inside its box, fading to 0 over the blend distance at its faces
float ProbeWeight(in const int i, in const vec3 P)
{
    const vec3 Lo = UBOLight.Probes[i].BoxMin.xyz, Hi = UBOLight.Probes[i].BoxMax.xyz;
    if (any(lessThan(P, Lo)) || any(greaterThan(P, Hi))) return 0.0;
    const vec3 Inside = min(P - Lo, Hi - P);
    return clamp(min(Inside.x, min(Inside.y, Inside.z)) / max(UBOLight.Probes[i].PosBlend.w, 1e-4), 0.0, 1.0);
}

// The reflection ray from P meets probe i's box: the direction from the probe's center to that point (parallax correction)
vec3 ProbeBoxProject(in const int i, in const vec3 P, in const vec3 R)
{
    const vec3 Lo = UBOLight.Probes[i].BoxMin.xyz, Hi = UBOLight.Probes[i].BoxMax.xyz;
    const vec3 Inv = 1.0 / max(abs(R), vec3(1e-6)) * sign(R + 1e-12);
    const vec3 T  = max((Lo - P) * Inv, (Hi - P) * Inv);
    const float t = min(T.x, min(T.y, T.z));
    return normalize(P + R * t - UBOLight.Probes[i].PosBlend.xyz);
}

vec3 EnvLighting(in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic, in const float AO)
{
    const vec3 R = reflect(-V, N);
    const vec3 P = In.wSpacePosition.xyz;                   // camera-centered, as the probes' boxes

    vec3  Spec = vec3(0.0), Diff = vec3(0.0);
    float Covered = 0.0;
    const int nProbes = int(UBOLight.ProbeCount.x);
    for (int i = 0; i < nProbes && Covered < 0.999; ++i)
    {
        const float w = ProbeWeight(i, P) * (1.0 - Covered);
        if (w <= 0.0) continue;
        const vec2 Origin = UBOLight.Probes[i].Slot.xy;
        Spec    += w * AtlasSample(SamplerProbes, Origin, kProbeAtlas, ProbeBoxProject(i, P, R), PerceptualRoughness);
        Diff    += w * AtlasSample(SamplerProbes, Origin, kProbeAtlas, N, 1.0);
        Covered += w;
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

    const vec2 DFG = textureLod(SamplerEnvDFG, vec2(NdotV, PerceptualRoughness), 0.0).rg;
    return (Diff * Albedo * (1.0 - Metallic) + Spec * (F0 * DFG.x + DFG.y)) * AO;
}
