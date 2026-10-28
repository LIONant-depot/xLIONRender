#version 450
// Prefilters the environment's source into one level of the atlas (features/environment): GGX importance sampling with N = V = R
// (Karis' split sum), drawn over the level's rect (viewport). Roughness 0 copies. The border texels fold over the octahedral edges
// like the source's, so bilinear sampling of a level stays inside it.
// Filtered importance sampling (Krivanek & Colbert 2008): each sample reads the source pyramid's level whose texels cover the
// sample's solid angle (Karis' +1 bias), so 64 samples do what 256+ point samples did, without aliasing on bright details.
#include "xlionrender_environment_atlas.glsl"

layout(binding = 0) uniform sampler2D SamplerSource0;       // the source pyramid: 256, 128, 64, 32, 16, 8 (each with its border)
layout(binding = 1) uniform sampler2D SamplerSource1;
layout(binding = 2) uniform sampler2D SamplerSource2;
layout(binding = 3) uniform sampler2D SamplerSource3;
layout(binding = 4) uniform sampler2D SamplerSource4;
layout(binding = 5) uniform sampler2D SamplerSource5;

layout(push_constant) uniform prefilter
{
    vec4 Rect;          // x, y: the level's origin in the atlas, z: its size
    vec4 Params;        // x: perceptual roughness, y: the source's size, z: samples
} P;

layout(location = 0) out vec4 outColor;

vec2 Hammersley(in const uint i, in const uint n)
{
    uint b = i;
    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return vec2(float(i) / float(n), float(b) * 2.3283064365386963e-10);
}

vec3 SourceLevel(in const int Level, in const vec3 Dir)
{
    const float Size  = P.Params.y / float(1 << Level);
    const float Inner = Size - 2.0 * kEnvBorder;
    const vec2  UV    = (kEnvBorder + (OctEncode(Dir) * 0.5 + 0.5) * Inner) / Size;
    switch (Level)
    {
    case 0:  return textureLod(SamplerSource0, UV, 0.0).rgb;
    case 1:  return textureLod(SamplerSource1, UV, 0.0).rgb;
    case 2:  return textureLod(SamplerSource2, UV, 0.0).rgb;
    case 3:  return textureLod(SamplerSource3, UV, 0.0).rgb;
    case 4:  return textureLod(SamplerSource4, UV, 0.0).rgb;
    default: return textureLod(SamplerSource5, UV, 0.0).rgb;
    }
}

// The source at a fractional pyramid level (a blend of the two around it)
vec3 Source(in const vec3 Dir, in const float Level)
{
    const float L  = clamp(Level, 0.0, 5.0);
    const int   L0 = int(floor(L));
    const int   L1 = min(L0 + 1, 5);
    return mix(SourceLevel(L0, Dir), SourceLevel(L1, Dir), L - float(L0));
}

void main()
{
    const float Inner = P.Rect.z - 2.0 * kEnvBorder;
    const vec2  UV    = (gl_FragCoord.xy - P.Rect.xy - kEnvBorder) / Inner * 2.0 - 1.0;
    const vec3  N     = OctDecode(OctFold(UV));

    const float Rough = P.Params.x;
    // (level 0: its alpha too - a probe's distances, xlionrender_env_cube_to_oct_frag.glsl; the sky's 1)
    if (Rough <= 0.0) { const vec2 UV0 = (kEnvBorder + (OctEncode(N) * 0.5 + 0.5) * (P.Params.y - 2.0 * kEnvBorder)) / P.Params.y; outColor = vec4(Source(N, 0.0), textureLod(SamplerSource0, UV0, 0.0).a); return; }

    const vec3  Up  = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    const vec3  T   = normalize(cross(Up, N));
    const vec3  B   = cross(N, T);
    const float a   = Rough * Rough;
    const uint  n   = uint(P.Params.z);
    const float SrcInner     = P.Params.y - 2.0 * kEnvBorder;
    const float TexelSolid   = 4.0 * 3.14159265359 / (SrcInner * SrcInner);       // the source's level 0 texels share the sphere

    vec3  Sum    = vec3(0.0);
    float Weight = 0.0;
    for (uint i = 0u; i < n; ++i)
    {
        const vec2  Xi       = Hammersley(i, n);
        const float Phi      = 6.28318530718 * Xi.x;
        const float CosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
        const float SinTheta = sqrt(1.0 - CosTheta * CosTheta);
        const vec3  H        = normalize(T * (cos(Phi) * SinTheta) + B * (sin(Phi) * SinTheta) + N * CosTheta);
        const vec3  L        = reflect(-N, H);
        const float NdotL    = dot(N, L);
        if (NdotL <= 0.0) continue;
        // pdf of L for N = V: D(NdotH) / 4; the sample covers 1 / (n pdf) steradians -> the pyramid level that matches it
        const float a2         = a * a;
        const float d          = CosTheta * CosTheta * (a2 - 1.0) + 1.0;
        const float D          = a2 / (3.14159265359 * d * d);
        const float SampleSolid = 4.0 / (float(n) * D + 1e-6);
        const float Level      = max(0.5 * log2(SampleSolid / TexelSolid) + 1.0, 0.0);
        Sum    += Source(L, Level) * NdotL;
        Weight += NdotL;
    }
    outColor = vec4(Sum / max(Weight, 1e-6), 1.0);
}
