#version 450
// The frame's bloom and auto exposure, before the tone map (instance::Render, scene::post_settings).
// Bloom: only what is too bright for the eye (Tomas: where it hurts; the rest stays sharp): the exposed light past a threshold, with a soft knee, added on top. The HDR image down a chain of half sizes (Jimenez's 13 taps, the first step a Karis average: a lone bright pixel doesn't sparkle), then back up, each level the mix of itself and
// the coarser one's tent: a flat image stays exactly itself. Exposure: the log average of a small level, the center counting more, toward key / average at a rate; a 1 x 1 image keeps it.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 1, binding = 0) uniform sampler2D Src;
layout(set = 1, binding = 1) uniform sampler2D ExposureTex;
layout(set = 2, binding = 0, rgba16f) uniform image2D Dst;
layout(push_constant) uniform push_t { int m_W, m_H, m_Mode; float m_A, m_B, m_C, m_D, m_E; } Push;   // Mode 0 down (A: 1 the Karis step), 1 up, 2 exposure (A: seconds, B: rate, C: key, D, E: least, most), 3 fill (A)

vec3 Tap(in const vec2 uv) { const vec2 Half = 0.5 / vec2(textureSize(Src, 0)); const vec3 c = textureLod(Src, clamp(uv, Half, 1.0 - Half), 0.0).rgb; return all(lessThan(c, vec3(1e5))) ? c : vec3(0.0); }
float Luma(in const vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 Down(in const vec2 uv)
{
    const vec2 t = 1.0 / vec2(textureSize(Src, 0));
    const vec3 a = Tap(uv + t * vec2(-2, -2)), b = Tap(uv + t * vec2(0, -2)), c = Tap(uv + t * vec2(2, -2));
    const vec3 d = Tap(uv + t * vec2(-2, 0)),  e = Tap(uv),                   f = Tap(uv + t * vec2(2, 0));
    const vec3 g = Tap(uv + t * vec2(-2, 2)),  h = Tap(uv + t * vec2(0, 2)),  i = Tap(uv + t * vec2(2, 2));
    const vec3 j = Tap(uv + t * vec2(-1, -1)), k = Tap(uv + t * vec2(1, -1)), l = Tap(uv + t * vec2(-1, 1)), m = Tap(uv + t * vec2(1, 1));
    if (Push.m_A > 0.5)
    {
        const vec3 G0 = (a + b + d + e) * 0.25, G1 = (b + c + e + f) * 0.25, G2 = (d + e + g + h) * 0.25, G3 = (e + f + h + i) * 0.25, G4 = (j + k + l + m) * 0.25;
        const float W0 = 0.125 / (1.0 + Luma(G0)), W1 = 0.125 / (1.0 + Luma(G1)), W2 = 0.125 / (1.0 + Luma(G2)), W3 = 0.125 / (1.0 + Luma(G3)), W4 = 0.5 / (1.0 + Luma(G4));
        const vec3  c  = (G0 * W0 + G1 * W1 + G2 * W2 + G3 * W3 + G4 * W4) / (W0 + W1 + W2 + W3 + W4);
        // (Push: B the threshold, C the knee, D the exposure, E 1: times the adapted one; in exposed light: 1 is white, 0.18 middle grey)
        const float Ex = Push.m_D * (Push.m_E > 0.5 ? texelFetch(ExposureTex, ivec2(0), 0).r : 1.0);
        const float l  = Luma(c) * Ex, q = clamp(l - Push.m_B + Push.m_C, 0.0, 2.0 * Push.m_C);
        return c * (max(q * q / (4.0 * Push.m_C + 1e-5), l - Push.m_B) / max(l, 1e-4));
    }
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}
vec3 Tent(in const vec2 uv)
{
    const vec2 t = 1.0 / vec2(textureSize(Src, 0));
    return (Tap(uv) * 4.0 + (Tap(uv + t * vec2(-1, 0)) + Tap(uv + t * vec2(1, 0)) + Tap(uv + t * vec2(0, -1)) + Tap(uv + t * vec2(0, 1))) * 2.0
          + Tap(uv + t * vec2(-1, -1)) + Tap(uv + t * vec2(1, -1)) + Tap(uv + t * vec2(-1, 1)) + Tap(uv + t * vec2(1, 1))) / 16.0;
}
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Push.m_W || p.y >= Push.m_H) return;
    if (Push.m_Mode == 3) { imageStore(Dst, p, vec4(Push.m_A)); return; }
    if (Push.m_Mode == 2)
    {
        // (the image itself, a sparse grid of it: the bloom's chain holds only what is past its threshold)
        const ivec2 Sz = textureSize(Src, 0);
        float Sum = 0.0, SumW = 0.0;
        for (int y = 0; y < 36; ++y) for (int x = 0; x < 64; ++x)
        {
            const vec2  u = (vec2(x, y) + 0.5) / vec2(64.0, 36.0), o = u * 2.0 - 1.0;
            const float w = 1.0 - 0.75 * dot(o, o) * 0.5;
            Sum += log(max(Luma(texelFetch(Src, ivec2(u * vec2(Sz)), 0).rgb), 1e-4)) * w; SumW += w;
        }
        const float Target = clamp(Push.m_C / exp(Sum / SumW), Push.m_D, Push.m_E);
        const float Prev   = imageLoad(Dst, p).r;
        imageStore(Dst, p, vec4(Prev <= 0.0 ? Target : mix(Target, Prev, exp(-Push.m_A * Push.m_B))));
        return;
    }
    const vec2 uv = (vec2(p) + 0.5) / vec2(Push.m_W, Push.m_H);
    if (Push.m_Mode == 0) imageStore(Dst, p, vec4(Down(uv), 1.0));
    else imageStore(Dst, p, vec4(mix(imageLoad(Dst, p).rgb, Tent(uv), Push.m_A), 1.0));     // (A: the spread: more of the wider levels)
}
