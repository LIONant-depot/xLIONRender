#version 450
// Temporal antialiasing (instance::Render, scene::post_settings::m_bTAA): the jittered frame over its history. The history comes from where the pixel was, by depth and the camera's move
// (or, where something moved on its own, by its vertices' motion), Catmull-Rom (5 bilinear taps - the compute's sampler is bilinear: m_bLinear; with nearest the
// history snapped to whole pixels and lost the camera's sub-pixel motion, C22's floor "waves"), kept inside its 2 x 2 texels and inside this frame's neighbours in YCoCg.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 1, binding = 0) uniform sampler2D HDR;
layout(set = 1, binding = 1) uniform sampler2D Depth;
layout(set = 1, binding = 2) uniform sampler2D History;
layout(set = 1, binding = 3) uniform sampler2D Motion;      // (the full-size G-buffer's: last frame's NDC xy, clip w, w > 0.5 where a vertex gave its last position)
layout(set = 2, binding = 0, rgba16f) writeonly uniform image2D Dst;
layout(set = 2, binding = 1, rgba16f) writeonly uniform image2D Dbg;      // (XLION_TAA_DEBUG: the diagnostic, beside the output - Dst stays the history)
// Reproject: this frame's clip (jittered) to last frame's (un-jittered); Blend: this frame's share; Jitter / PrevJitter: this / last frame's, in uv; Gamma: the clip box's
// width in standard deviations, still / moving 8 px or more; Debug (XLION_TAA_DEBUG): 1 the motion, px (|m| x 0.25 red, x / y signed in green / blue around 0.5),
// 3 this frame's share x 4, 11 the camera's reprojection against the vertices' motion (|difference| x 4 red, x / y signed)
layout(push_constant) uniform push_t { mat4 m_Reproject; int m_W, m_H; float m_Blend, m_bHistory; vec2 m_Jitter; float m_GammaSlow, m_GammaFast; vec2 m_PrevJitter; float m_bMotion; int m_Debug; float m_MovingShare, m_Sharpen; } Push;

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
    const vec3 Sum = textureLod(History, vec2(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y)
                   + textureLod(History, vec2(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y)
                   + textureLod(History, t12, 0.0).rgb * (w12.x * w12.y)
                   + textureLod(History, vec2(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y)
                   + textureLod(History, vec2(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    const float W = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    // Kept inside the 2 x 2 history texels around the point (C22): Catmull-Rom's negative lobes overshoot, and resampled every frame the overshoot piles up (without
    // the clamp the detector's floor detail went to 1.13 x TAA off - ringing - and its waves tripled)
    const ivec2 q = ivec2(floor(Pos - 0.5)), Mx = ivec2(Push.m_W, Push.m_H) - 1;
    const vec3 a = texelFetch(History, clamp(q, ivec2(0), Mx), 0).rgb, b = texelFetch(History, clamp(q + ivec2(1, 0), ivec2(0), Mx), 0).rgb;
    const vec3 c = texelFetch(History, clamp(q + ivec2(0, 1), ivec2(0), Mx), 0).rgb, d = texelFetch(History, clamp(q + ivec2(1, 1), ivec2(0), Mx), 0).rgb;
    return clamp(Sum / W, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
}
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Push.m_W || p.y >= Push.m_H) return;
    const vec2 Sz = vec2(Push.m_W, Push.m_H);
    // This frame's 3 x 3: the pixel and the clip box's statistics (variance clipping, Salvi)
    vec3 C = vec3(0.0), M1 = vec3(0.0), M2 = vec3(0.0);
    float Sharp = 0.0;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x)
    {
        const vec3 c = Fetch(p + ivec2(x, y));
        if (x == 0 && y == 0) C = c;
        const vec3 s = ToYCoCg(c);
        M1 += s; M2 += s * s;
    }
    if (Push.m_bHistory > 0.5)
    {
        // Where the pixel was: its depth is where its jittered sample landed, unprojected with the jittered camera into last frame's un-jittered one, plus this
        // frame's jitter - the history at the pixel plus its motion, the jitter cancelled (C22)
        const vec4 Prev = Push.m_Reproject * vec4((vec2(p) + 0.5) / Sz * 2.0 - 1.0, texelFetch(Depth, p, 0).r, 1.0);
        vec2 puv = Prev.xy / Prev.w * 0.5 + 0.5 + Push.m_Jitter;
        if (Push.m_bMotion > 0.5)
        {   // What moved on its own (characters, moved meshes): its vertices' motion - last frame's NDC carries last frame's jitter, taken off. Only where it differs
            // from the camera's by over 0.75 px: the target keeps absolute NDC in half floats, up to ~0.5 px off at the screen's sides (Fable)
            const vec4 M = texelFetch(Motion, p, 0);
            const vec2 Vtx = M.xy * 0.5 + 0.5 - Push.m_PrevJitter + Push.m_Jitter;
            if (Push.m_Debug == 11) { const vec2 d = (puv - Vtx) * Sz; imageStore(Dbg, p, M.w > 0.5 ? vec4(length(d) * 4.0, 0.5 + 2.0 * d.x, 0.5 + 2.0 * d.y, 1.0) : vec4(0.0)); }
            if (M.w > 0.5 && length((Vtx - puv) * Sz) > 0.75) puv = Vtx;
        }
        if (Prev.w > 0.0 && all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0))))
        {
            const vec2 m = puv * Sz - (vec2(p) + 0.5);
            const float Moved = length(m);
            if (Push.m_Debug == 1) imageStore(Dbg, p, vec4(Moved * 0.25, 0.5 + 0.25 * m.x, 0.5 + 0.25 * m.y, 1.0));
            M1 /= 9.0; const vec3 Sd = sqrt(max(M2 / 9.0 - M1 * M1, vec3(0.0)));
            const float Gamma = mix(Push.m_GammaSlow, Push.m_GammaFast, smoothstep(1.0, 8.0, Moved));      // (wide while slow: the jitter's own change isn't clipped; narrow when fast)
            const vec3 H = max(FromYCoCg(clamp(ToYCoCg(SampleHistory(puv)), M1 - Gamma * Sd, M1 + Gamma * Sd)), vec3(0.0));
            // This frame's share by how far the pixel moved: still, half of it (the jitter's difference times the share was a shimmer with its 8-frame cycle); moving, a little more
            const float Share = mix(Push.m_Blend * 0.5, Push.m_MovingShare, clamp(Moved / 2.0, 0.0, 1.0));
            // The display sharpened where the history blurs - moving (TFAA's idea): an amount for the tone map, in the alpha above 1 - never in the history's colour
            Sharp = Push.m_Sharpen * clamp(Moved / 2.0, 0.0, 1.0);
            if (Push.m_Debug == 3) imageStore(Dbg, p, vec4(vec3(Share * 4.0), 1.0));
            const float Wc = Share / (1.0 + Luma(C)), Wh = (1.0 - Share) / (1.0 + Luma(H));      // (by the displayed brightness: a bright pixel doesn't flash)
            C = (C * Wc + H * Wh) / (Wc + Wh);
        }
    }
    const float A = texelFetch(HDR, p, 0).a;      // (coverage: 0 nothing drawn, 1 geometry)
    imageStore(Dst, p, vec4(C, A > 0.5 ? 1.0 + 0.99 * clamp(Sharp, 0.0, 1.0) : A));
}
