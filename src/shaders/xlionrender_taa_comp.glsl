#version 450
// Temporal antialiasing (instance::Render, scene::post_settings::m_bTAA): the jittered frame over its history. The history comes from where the pixel was, by depth and the camera's move
// (ponytail: movers ghost until the clamp catches them; the G-buffer's motion when it shows), Catmull-Rom (5 taps) so it stays sharp while moving, kept inside this frame's neighbours in YCoCg.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 1, binding = 0) uniform sampler2D HDR;
layout(set = 1, binding = 1) uniform sampler2D Depth;
layout(set = 1, binding = 2) uniform sampler2D History;
layout(set = 1, binding = 3) uniform sampler2D Motion;      // (the full-size G-buffer's: last frame's NDC xy, clip w, w > 0.5 where a vertex gave its last position)
layout(set = 2, binding = 0, rgba16f) writeonly uniform image2D Dst;
layout(set = 2, binding = 1, rgba16f) writeonly uniform image2D Dbg;      // (XLION_TAA_DEBUG: the diagnostic, beside the output - Dst stays the history)
layout(push_constant) uniform push_t { mat4 m_Reproject; int m_W, m_H; float m_Blend, m_bHistory; vec2 m_Jitter; float m_GammaSlow, m_GammaFast; int m_Debug, m_Pad0, m_Pad1, m_Pad2; vec2 m_PrevJitter; float m_bMotion, m_Pad3; } Push;
// m_Debug (XLION_TAA_DEBUG, C22 diagnostics - written instead of the colour, values scaled to show): 1 the jitter-free motion, pixels: |motion| x 0.25 in red,
// its x and y signed in green / blue (0.5 + 0.25 x px); 2 the clip distance |history - clipped| (YCoCg luma) x 4; 3 this frame's share x 4 (blend weight);
// 4 the history before the clip; 5 the reprojected uv's distance from the pixel, minus |motion|: 0 (consistency of Moved)      // Jitter: this frame's, in uv (the jittered sample's offset: added back after the reprojection)      // Reproject: this frame's clip to last frame's; Blend: this frame's share

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
    // Kept inside the 2 x 2 history texels around the point (C22, 2026-11-10): Catmull-Rom's negative lobes overshoot, and resampled again every
    // frame at a fraction of a pixel - a slowing camera, the history 95 % of the result - the overshoot piled up: jagged, crawling cobbles
    // and stepped highlights until the camera stopped. The clamp removes the overshoot, not the sharpness
    const ivec2 q = ivec2(floor(Pos - 0.5)), Mx = ivec2(Push.m_W, Push.m_H) - 1;
    const vec3 a = texelFetch(History, clamp(q, ivec2(0), Mx), 0).rgb, b = texelFetch(History, clamp(q + ivec2(1, 0), ivec2(0), Mx), 0).rgb;
    const vec3 c = texelFetch(History, clamp(q + ivec2(0, 1), ivec2(0), Mx), 0).rgb, d = texelFetch(History, clamp(q + ivec2(1, 1), ivec2(0), Mx), 0).rgb;
    if ((Push.m_Pad2 & 1) != 0) return Sum / W;      // (C22, measurement: the 2 x 2 clamp off - XLION_TAA_NO_CR_CLAMP)
    return clamp(Sum / W, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
}
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Push.m_W || p.y >= Push.m_H) return;
    const vec4 Here = texelFetch(HDR, p, 0);
    vec3 C = Fetch(p);
    if (Push.m_Pad1 == 1)
    {   // (C22, XLION_TAA_CURRENT=filtered: this frame's samples filtered back to the pixel's centre - each sits at its texel's centre minus the
        // jitter; Unreal's TemporalAA weights, a Gaussian of sigma 0.47 px. Unfiltered, the jittered sample counts as the pixel's own)
        vec3 Sum = vec3(0.0); float Wt = 0.0;
        for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x)
        {
            const vec2 o = vec2(x, y) - Push.m_Jitter * vec2(Push.m_W, Push.m_H);
            const float w = exp(-2.29 * dot(o, o));
            Sum += Fetch(p + ivec2(x, y)) * w; Wt += w;
        }
        C = Sum / Wt;
    }
    if (Push.m_bHistory > 0.5)
    {
        const vec2 uv = (vec2(p) + 0.5) / vec2(Push.m_W, Push.m_H);
        const vec4 Prev = Push.m_Reproject * vec4(uv * 2.0 - 1.0, texelFetch(Depth, p, 0).r, 1.0);
        vec2 puv = Prev.xy / Prev.w * 0.5 + 0.5 + Push.m_Jitter;      // (the jittered sample's point, last frame, plus the jitter: the history at the pixel plus its motion)
        if (Push.m_bMotion > 0.5)
        {   // (C22: what moves on its own - characters, skinned and moved meshes - by its own motion: the vertex's last position, as rasterised; last
            // frame's NDC carries last frame's jitter (taken off), this sample's point is the pixel minus this frame's (put back), as above)
            const vec4 M = texelFetch(Motion, p, 0);
            // (only where it moved on its own - over half a pixel from the camera's motion: the target keeps last frame's absolute NDC in half
            // floats, up to ~0.5 px off at the screen's sides (Fable, 2026-11-10); what only the camera moved keeps the exact depth reprojection)
            const vec2 Vtx = M.xy * 0.5 + 0.5 - Push.m_PrevJitter + Push.m_Jitter;
            if (M.w > 0.5 && length((Vtx - puv) * vec2(Push.m_W, Push.m_H)) > 0.75) puv = Vtx;
        }
        if (Push.m_Debug == 11)
        {   // (11: the camera reprojection (depth) against the vertices' motion, pixels: red |difference| x 4, green / blue the signed x / y around 0.5 - waves here = the depth's precision)
            const vec4 M = texelFetch(Motion, p, 0);
            const vec2 Cam = Prev.xy / Prev.w * 0.5 + 0.5 + Push.m_Jitter;
            const vec2 Vtx = M.xy * 0.5 + 0.5 - Push.m_PrevJitter + Push.m_Jitter;
            const vec2 d = (Cam - Vtx) * vec2(Push.m_W, Push.m_H);
            imageStore(Dbg, p, M.w > 0.5 ? vec4(length(d) * 4.0, 0.5 + 2.0 * d.x, 0.5 + 2.0 * d.y, 1.0) : vec4(0.0));
        }
        if (Prev.w > 0.0 && all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0))))
        {
            // (the history kept inside this frame's 3 x 3, mean +- 1.25 standard deviations in YCoCg - variance clipping, Salvi; the cross's min / max: noise by
            // lag 1 frame .90, over 4 .26 %; now .87, .20 %)
            vec3 M1 = vec3(0.0), M2 = vec3(0.0);
            // (C22: Push.m_Pad0 1 (XLION_TAA_BOX=centred): the statistics weighted by each sample's distance from the pixel's centre - a sample sits at
            // its texel's centre minus the jitter - so the box stays put when only the jitter moves; 0: the plain 3 x 3, as before)
            float Wsum = 0.0;
            for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x)
            {
                const vec3 s = ToYCoCg(Fetch(p + ivec2(x, y)));
                const vec2 o = vec2(x, y) - Push.m_Jitter * vec2(Push.m_W, Push.m_H);
                const float w = Push.m_Pad0 == 1 ? exp(-0.5 * dot(o, o)) : 1.0;
                M1 += s * w; M2 += s * s * w; Wsum += w;
            }
            M1 /= Wsum; const vec3 Sd = sqrt(max(M2 / Wsum - M1 * M1, vec3(0.0)));
            const float MovedPx = length(puv * vec2(Push.m_W, Push.m_H) - (vec2(p) + 0.5));
            if (Push.m_Debug == 1) { const vec2 m = puv * vec2(Push.m_W, Push.m_H) - (vec2(p) + 0.5); imageStore(Dbg, p, vec4(length(m) * 0.25, 0.5 + 0.25 * m.x, 0.5 + 0.25 * m.y, 1.0)); }
            const float Gamma = mix(Push.m_GammaSlow, Push.m_GammaFast, smoothstep(1.0, 8.0, MovedPx));      // (C22: the box's width - XLION_TAA_GAMMA "slow fast")
            const vec3 Lo = M1 - Gamma * Sd, Hi = M1 + Gamma * Sd;
            const vec3 Hs = Push.m_bHistory > 4.5 ? texelFetch(History, clamp(ivec2(floor(puv * vec2(Push.m_W, Push.m_H))), ivec2(0), ivec2(Push.m_W, Push.m_H) - 1), 0).rgb      // (5: the nearest texel - no resampling, XLION_TAA_FILTER=nearest)
                          : Push.m_bHistory > 1.5 && Push.m_bHistory < 2.5 ? textureLod(History, puv, 0.0).rgb : SampleHistory(puv);      // (C22, measurement: m_bHistory 2 - bilinear, XLION_TAA_FILTER=bilinear)
            const vec3 H = Push.m_bHistory > 3.5 && Push.m_bHistory < 4.5 ? Hs : FromYCoCg(clamp(ToYCoCg(Hs), Lo, Hi));      // (C22, measurement: m_bHistory 4 - no clip, XLION_TAA_NO_CLIP)
            // (this frame's share by how far the pixel moved: still, half of it - the jitter's difference times the share was a shimmer with its 8-frame cycle,
            // noise by lag 1 frame .87 at .1, .77 at .05; moving, the full share and a little more: no added ghosting)
            const float Moved = length(puv * vec2(Push.m_W, Push.m_H) - (vec2(p) + 0.5));
            float Share = Push.m_Blend * mix(0.5, 1.5, clamp(Moved / 2.0, 0.0, 1.0));
            if (Push.m_Pad3 > 0.0)
            {   // (C22, Fable's lab: the history's lookup between texels blurs it by its fraction - most at half a texel - and the fraction varies across
                // the floor and with every camera step: moving bands, "waves". This frame counts more where the lookup blurs: XLION_TAA_FRAC_SHARE=k)
                const vec2 f = fract(puv * vec2(Push.m_W, Push.m_H) - 0.5);
                const float Blur = 4.0 * max(f.x * (1.0 - f.x), f.y * (1.0 - f.y));
                Share = max(Share, Push.m_Pad3 * Blur);
            }
            // (C22, Fable's test b - XLION_TAA_MOTION_SHARE: the share by motion only, 0.05 still rising to 0.35 from 1 px a frame)
            if ((Push.m_Pad2 & 2) != 0) Share = mix(0.05, 0.35, clamp(Moved, 0.0, 1.0));
            if (Push.m_Pad3 < 0.0) Share = -Push.m_Pad3;      // (C22, measurement: the share fixed - XLION_TAA_FRAC_SHARE=-k; -1: this frame only)
            if (Push.m_Debug == 2) { imageStore(Dbg, p, vec4(vec3(length(ToYCoCg(H) - ToYCoCg(Hs)) * 4.0), 1.0)); }
            if (Push.m_Debug == 3) { imageStore(Dbg, p, vec4(vec3(Share * 4.0), 1.0)); }
            if (Push.m_Debug == 4) { imageStore(Dbg, p, vec4(Hs, 1.0)); }
            if (Push.m_Debug == 6) { imageStore(Dbg, p, vec4(vec3(length(ToYCoCg(H) - ToYCoCg(Hs)) / max(M1.x, 1e-4) * 0.25), 1.0)); }      // (6: the clip distance relative to the neighbourhood's luma, x 0.25)
            if (Push.m_Debug == 9) { imageStore(Dbg, p, vec4(vec3(abs(ToYCoCg(Hs).x - M1.x) / max(Sd.x, 1e-5) * 0.25), 1.0)); }      // (9: the history's distance from the neighbourhood's mean, in its sigmas, x 0.25 - the clip starts at Gamma)
            if (Push.m_Debug == 7) { imageStore(Dbg, p, vec4(vec3(Sd.x / max(M1.x, 1e-4) * 0.25), 1.0)); }      // (7: the neighbourhood's relative sigma, x 0.25)
            if (Push.m_Debug == 8) { imageStore(Dbg, p, vec4(vec3(Luma(Hs) / max(Luma(C), 1e-4) * 0.25), 1.0)); }      // (8: history luma / this frame's, x 0.25 - 1 shows 0.25)
            const float Wc = Share / (1.0 + Luma(C)), Wh = (1.0 - Share) / (1.0 + Luma(max(H, vec3(0.0))));      // (by the displayed brightness: a bright pixel doesn't flash)
            C = (C * Wc + max(H, vec3(0.0)) * Wh) / (Wc + Wh);
        }
    }
    imageStore(Dst, p, vec4(C, Here.a));
}
