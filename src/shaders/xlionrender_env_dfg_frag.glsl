#version 450
// The split-sum DFG table (Karis 2013): x = N.V, y = perceptual roughness -> (scale, bias) of F0 for the prefiltered specular.
// Computed once (features/environment) with the same correlated Smith visibility as the direct lighting.
layout(push_constant) uniform dfg
{
    vec4 Params;        // x: the table's size, y: samples
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

float VisSmithCorrelated(in const float NdotV, in const float NdotL, in const float a)
{
    const float a2 = a * a;
    const float V  = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    const float L  = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(V + L, 1e-6);
}

void main()
{
    const float NdotV = max((gl_FragCoord.x) / P.Params.x, 1e-3);
    const float Rough = max((gl_FragCoord.y) / P.Params.x, 0.02);
    const float a     = Rough * Rough;
    const vec3  V     = vec3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    const uint  n     = uint(P.Params.y);

    float A = 0.0, B = 0.0;
    for (uint i = 0u; i < n; ++i)
    {
        const vec2  Xi       = Hammersley(i, n);
        const float Phi      = 6.28318530718 * Xi.x;
        const float CosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
        const float SinTheta = sqrt(1.0 - CosTheta * CosTheta);
        const vec3  H        = vec3(cos(Phi) * SinTheta, sin(Phi) * SinTheta, CosTheta);
        const vec3  L        = 2.0 * dot(V, H) * H - V;
        const float NdotL    = max(L.z, 0.0);
        const float NdotH    = max(H.z, 0.0);
        const float VdotH    = max(dot(V, H), 0.0);
        if (NdotL <= 0.0) continue;
        // pdf(L) = D NdotH / (4 VdotH): the estimator of D Vis F NdotL / pdf = Vis 4 VdotH NdotL / NdotH (times F)
        const float G  = VisSmithCorrelated(NdotV, NdotL, a) * 4.0 * VdotH * NdotL / NdotH;
        const float Fc = pow(1.0 - VdotH, 5.0);
        A += (1.0 - Fc) * G;
        B += Fc * G;
    }
    outColor = vec4(A / float(n), B / float(n), 0.0, 1.0);
}
