#version 450
// Temporal antialiasing (instance::Render, scene::post_settings::m_bTAA): the jittered frame over its history. The history comes from where the pixel was, by depth and the camera's move
// (ponytail: movers ghost until the clamp catches them; the G-buffer's motion when it shows), Catmull-Rom (5 taps) so it stays sharp while moving, kept inside this frame's neighbours in YCoCg.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 1, binding = 0) uniform sampler2D HDR;
layout(set = 1, binding = 1) uniform sampler2D Depth;
layout(set = 1, binding = 2) uniform sampler2D History;
layout(set = 2, binding = 0, rgba16f) writeonly uniform image2D Dst;
layout(push_constant) uniform push_t { mat4 m_Reproject; int m_W, m_H; float m_Blend, m_bHistory; } Push;      // Reproject: this frame's clip to last frame's; Blend: this frame's share

vec3 ToYCoCg(in const vec3 c) { return vec3(dot(c, vec3(0.25, 0.5, 0.25)), dot(c, vec3(0.5, 0.0, -0.5)), dot(c, vec3(-0.25, 0.5, -0.25))); }
vec3 FromYCoCg(in const vec3 c) { return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }
float Luma(in const vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 Fetch(in const ivec2 p) { const vec3 c = texelFetch(HDR, clamp(p, ivec2(0), ivec2(Push.m_W, Push.m_H) - 1), 0).rgb; return all(lessThan(c, vec3(1e5))) ? max(c, vec3(0.0)) : vec3(0.0); }
vec3 SampleHistory(in const vec2 uv)
{
    const vec2 Sz = vec2(Push.m_W, Push.m_H), Pos = uv * Sz, C = floor(Pos - 0.5) + 0.5, f = Pos - C;
    const vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f), w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
    const vec2 w12 = w1 + w2, o12 = w2 / w12;
    const vec2 t0 = (C - 1.0) / Sz, t12 = (C + o12) / Sz, t3 = (C + 2.0) / Sz;
    vec3 Sum = textureLod(History, vec2(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y)
             + textureLod(History, vec2(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y)
             + textureLod(History, t12, 0.0).rgb * (w12.x * w12.y)
             + textureLod(History, vec2(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y)
             + textureLod(History, vec2(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    const float W = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(Sum / W, vec3(0.0));
}
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Push.m_W || p.y >= Push.m_H) return;
    const vec4 Here = texelFetch(HDR, p, 0);
    vec3 C = Fetch(p);
    if (Push.m_bHistory > 0.5)
    {
        const vec2 uv = (vec2(p) + 0.5) / vec2(Push.m_W, Push.m_H);
        const vec4 Prev = Push.m_Reproject * vec4(uv * 2.0 - 1.0, texelFetch(Depth, p, 0).r, 1.0);
        const vec2 puv = Prev.xy / Prev.w * 0.5 + 0.5;
        if (Prev.w > 0.0 && all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0))))
        {
            // (the history kept inside this frame's 3 x 3, mean +- 1.25 standard deviations in YCoCg - variance clipping, Salvi; the cross's min / max: noise by
            // lag 1 frame .90, over 4 .26 %; now .87, .20 %)
            vec3 M1 = vec3(0.0), M2 = vec3(0.0);
            for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) { const vec3 s = ToYCoCg(Fetch(p + ivec2(x, y))); M1 += s; M2 += s * s; }
            M1 /= 9.0; const vec3 Sd = sqrt(max(M2 / 9.0 - M1 * M1, vec3(0.0)));
            const vec3 Lo = M1 - 1.25 * Sd, Hi = M1 + 1.25 * Sd;
            const vec3 H = FromYCoCg(clamp(ToYCoCg(SampleHistory(puv)), Lo, Hi));
            // (this frame's share by how far the pixel moved: still, half of it - the jitter's difference times the share was a shimmer with its 8-frame cycle,
            // noise by lag 1 frame .87 at .1, .77 at .05; moving, the full share and a little more: no added ghosting)
            const float Moved = length(puv * vec2(Push.m_W, Push.m_H) - (vec2(p) + 0.5));
            const float Share = Push.m_Blend * mix(0.5, 1.5, clamp(Moved / 2.0, 0.0, 1.0));
            const float Wc = Share / (1.0 + Luma(C)), Wh = (1.0 - Share) / (1.0 + Luma(max(H, vec3(0.0))));      // (by the displayed brightness: a bright pixel doesn't flash)
            C = (C * Wc + max(H, vec3(0.0)) * Wh) / (Wc + Wh);
        }
    }
    imageStore(Dst, p, vec4(C, Here.a));
}
