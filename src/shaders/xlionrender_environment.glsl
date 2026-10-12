// Image-based light from the environment probe (features/environment): replaces the flat ambient when the scene's sky is on
// (UBOLight.EnvParams.x). Specular: the prefiltered level for the roughness along the reflection, times the split-sum DFG
// (Karis); diffuse: the roughest level along the normal (an approximation of the irradiance - ponytail: SH9 when it shows).
// Samplers 8 and 9 in every lit pipeline (slots 0-7: the shadow atlas and the material's textures).
#include "xlionrender_environment_atlas.glsl"

layout(binding = 8) uniform sampler2D SamplerEnvironment;
layout(binding = 9) uniform sampler2D SamplerEnvDFG;

vec3 EnvSample(in const vec3 Dir, in const float PerceptualRoughness)
{
    const float L  = clamp(PerceptualRoughness, 0.0, 1.0) * float(kEnvLevels - 1);
    const int   L0 = int(floor(L));
    const int   L1 = min(L0 + 1, kEnvLevels - 1);
    const vec3  A  = textureLod(SamplerEnvironment, EnvLevelUV(L0, Dir), 0.0).rgb;
    const vec3  B  = textureLod(SamplerEnvironment, EnvLevelUV(L1, Dir), 0.0).rgb;
    return mix(A, B, L - float(L0));
}

vec3 EnvLighting(in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic, in const float AO)
{
    const vec2 DFG      = textureLod(SamplerEnvDFG, vec2(NdotV, PerceptualRoughness), 0.0).rg;
    const vec3 Specular = EnvSample(reflect(-V, N), PerceptualRoughness) * (F0 * DFG.x + DFG.y);
    const vec3 Diffuse  = EnvSample(N, 1.0) * Albedo * (1.0 - Metallic);
    return (Diffuse + Specular) * AO * UBOLight.EnvParams.y;
}
