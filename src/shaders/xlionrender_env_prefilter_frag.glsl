#version 450
// Prefilters the environment's source into one level of the atlas (features/environment): GGX importance sampling with N = V = R
// (Karis' split sum), drawn over the level's rect (viewport). Roughness 0 copies. The border texels fold over the octahedral edges
// like the source's, so bilinear sampling of a level stays inside it.
#include "xlionrender_environment_atlas.glsl"

layout(binding = 0) uniform sampler2D SamplerSource;

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

vec3 Source(in const vec3 Dir)
{
    const float Size  = P.Params.y;
    const float Inner = Size - 2.0 * kEnvBorder;
    return textureLod(SamplerSource, (kEnvBorder + (OctEncode(Dir) * 0.5 + 0.5) * Inner) / Size, 0.0).rgb;
}

void main()
{
    const float Inner = P.Rect.z - 2.0 * kEnvBorder;
    const vec2  UV    = (gl_FragCoord.xy - P.Rect.xy - kEnvBorder) / Inner * 2.0 - 1.0;
    const vec3  N     = OctDecode(OctFold(UV));

    const float Rough = P.Params.x;
    if (Rough <= 0.0) { outColor = vec4(Source(N), 1.0); return; }

    const vec3  Up  = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    const vec3  T   = normalize(cross(Up, N));
    const vec3  B   = cross(N, T);
    const float a   = Rough * Rough;
    const uint  n   = uint(P.Params.z);

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
        Sum    += Source(L) * NdotL;
        Weight += NdotL;
    }
    outColor = vec4(Sum / max(Weight, 1e-6), 1.0);
}
