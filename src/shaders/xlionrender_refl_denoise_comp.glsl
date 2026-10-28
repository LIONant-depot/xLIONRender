#version 450
// Reflections' reconstruction (documentation/reflections_strategy.md milestone D): at half size, over the reflection records the trace
// wrote (xlionrender_refl_trace_comp.glsl: the records' second region - x: U.rg, y: U.b and conf, z: the hit distance and G2/G1, w: the
// view depth (half), the octahedral normal (4 + 4 bits), the hit class (bits 24-27)). The screen tail's denoiser's shape
// (xlionrender_tail_denoise_comp.glsl), for a reflection:
//   pass 0, temporal: History (last frame's result) -> Temporal. The history is reprojected by the reflection's virtual motion - the hit
//            seen behind the mirror, the receiver's point pushed along the view ray by the hit's distance - on sharp receivers, by the
//            surface's motion on rough ones (their lobe averages what the virtual image would carry: it blurs with the surface), between
//            them blended by roughness; a mover's texel by its own motion, its history short. Its 2 x 2 texels are kept where they were a
//            receiver of this kind (normals within ~25 degrees, roughness within 0.1; the surface's motion also its depth within 3%),
//            clamped to this frame's 5 x 5 neighbourhood (mean +- (1 + roughness) standard deviations; the new sample's luminance at
//            most 4 x the neighbourhood's mean: fireflies), accumulated up to 6 frames on
//            mirrors (their one ray barely moves) to 24 on rough ones.
//   pass 1, spatial: Temporal -> the records (what the materials read: xlionrender_environment.glsl ReflScreen) and History. The texel and
//            8 taps on a disk of the lobe's footprint (roughness x 12 texels, at most 6; wider while the history is short), weighed by
//            normal, depth and roughness.
// The confidence (the coverage) is filtered with the light: the color premultiplied by it, and it - a screen trace's misses (class 4,
// coverage 0: xlionrender_refl_ssr_comp.glsl) average into a fraction the materials blend the probes by, not a hit / miss speckle.
// The class and the full-size sharp records are left as the trace wrote them (those are mirrors: not reconstructed)
layout(local_size_x = 8, local_size_y = 8) in;

#include "xlionrender_octahedral.glsl"

struct frame_t
{
    mat4 C2W;               // this frame's clip -> camera-centred world
    mat4 ToPrev;            // this frame's camera-centred world -> last frame's clip (the eye's move between)
    vec4 Params;            // x, y: size; z: 1 there is a last frame; w: the frame's number (the spatial taps' rotation)
};
layout(std430, set = 0, binding = 0) readonly buffer frame_b    { frame_t F[]; };
layout(std430, set = 0, binding = 1)          buffer records_b  { uvec4 Records[]; };    // the tail's W x H, then the reflections' W x H
layout(std430, set = 0, binding = 2)          buffer temporal_b { uvec4 Temporal[]; };
layout(std430, set = 0, binding = 3)          buffer history_b  { uvec4 History[]; };    // last frame's result: x: rg, y: b and conf, z: roughness and frames, w: the record's w
layout(set = 1, binding = 0) uniform sampler2D GColor;          // xy: the octahedral normal (x 0.5 + 0.5), z: roughness
layout(set = 1, binding = 1) uniform sampler2D GDepth;
layout(set = 1, binding = 2) uniform sampler2D GMotion;         // last frame's ndc xy, clip w, 1 (0: the camera's reprojection)
layout(push_constant) uniform push_t { uint m_Pass, m_Slot, m_Pad0, m_Pad1; } Push;

ivec2 Size() { return ivec2(F[Push.m_Slot].Params.xy); }
uint  Base() { return uint(Size().x * Size().y); }              // (the reflections' records after the tail's)
vec3  RecU(uvec4 R)     { return vec3(unpackHalf2x16(R.x), unpackHalf2x16(R.y).x); }
float RecConf(uvec4 R)  { return unpackHalf2x16(R.y).y; }
uint  RecClass(uvec4 R) { return (R.w >> 24) & 15u; }
float MetaDist(uint w)  { return unpackHalf2x16(w & 0xFFFFu).x; }
vec3  MetaNormal(uint w){ return OctDecode(vec2(float((w >> 16) & 15u), float((w >> 20) & 15u)) / 15.0 * 2.0 - 1.0); }
vec4  RecQ(uvec4 R)     { return vec4(RecU(R) * RecConf(R), RecConf(R)); }     // (the light premultiplied by the coverage, the coverage)
uvec4 WithQ(uvec4 R, vec4 Q) { const vec3 U = Q.a > 1e-4 ? Q.rgb / Q.a : vec3(0.0); return uvec4(packHalf2x16(U.rg), packHalf2x16(vec2(U.b, clamp(Q.a, 0.0, 1.0))), R.z, R.w); }

// This frame's records around the workgroup (its 8 x 8 texels and kNb more each side): their U and view depth (-1: none traced)
const int kNb   = 2;                        // the neighbourhood: 5 x 5 texels
const int kTile = 8 + 2 * kNb;
shared vec4 s_U[kTile * kTile];
shared float s_C[kTile * kTile];      // (their coverage)

void LoadTile()
{
    const ivec2 Org = ivec2(gl_WorkGroupID.xy) * 8 - kNb;
    for (uint t = gl_LocalInvocationIndex; t < uint(kTile * kTile); t += 64u)
    {
        const ivec2 q = clamp(Org + ivec2(int(t) % kTile, int(t) / kTile), ivec2(0), Size() - 1);
        const uvec4 R = Records[Base() + uint(q.y * Size().x + q.x)];
        s_U[t] = vec4(RecU(R) * RecConf(R), RecClass(R) == 0u ? -1.0 : MetaDist(R.w));
        s_C[t] = RecConf(R);
    }
    barrier();
}

void TemporalPass(ivec2 p)
{
    const uint  i = uint(p.y * Size().x + p.x);
    const uvec4 R = Records[Base() + i];
    const float z = texelFetch(GDepth, p, 0).r;
    if (z >= 1.0 || RecClass(R) == 0u) { Temporal[i] = R; return; }      // (no surface, or nothing traced: the probes - left alone)
    const vec4  G     = texelFetch(GColor, p, 0);
    const vec3  N     = OctDecode(G.xy * 2.0 - 1.0);
    const float Rough = G.z;
    const vec2  Ndc   = (vec2(p) + 0.5) / vec2(Size()) * 2.0 - 1.0;
    const vec4  Pc    = F[Push.m_Slot].C2W * vec4(Ndc, z, 1.0);
    const vec3  P     = Pc.xyz / Pc.w;                                  // camera-centred
    const float Dist  = length(P);

    // this frame's neighbourhood (the same surface's traced texels, 5 x 5): mean and deviation, the history's clamp
    const ivec2 l = ivec2(gl_LocalInvocationID.xy) + kNb;
    vec4 m1 = vec4(0.0), m2 = vec4(0.0);
    float k = 0.0;
    for (int y = -kNb; y <= kNb; ++y)
    for (int x = -kNb; x <= kNb; ++x)
    {
        const vec4 u = s_U[(l.y + y) * kTile + l.x + x];
        if (u.w < 0.0 || abs(u.w - MetaDist(R.w)) > 0.05 * MetaDist(R.w)) continue;
        const vec4 q = vec4(u.rgb, s_C[(l.y + y) * kTile + l.x + x]);
        m1 += q; m2 += q * q; k += 1.0;
    }
    m1 /= max(k, 1.0);
    const vec4 sd = sqrt(max(m2 / max(k, 1.0) - m1 * m1, 0.0));
    // a firefly (a highlight one ray in many finds: the sun's at a hit) taken at most 4 times the neighbourhood's mean luminance - it
    // would reset the history around it and sparkle; the energy it carries is the bias this clamp trades for stability (labelled)
    const vec4  Q0 = RecQ(R);
    const float lm = dot(m1.rgb, vec3(0.2126, 0.7152, 0.0722)) / max(m1.a, 1e-3), lu = dot(Q0.rgb, vec3(0.2126, 0.7152, 0.0722)) / max(Q0.a, 1e-3);     // (per coverage: sparse hits are not fireflies)
    const vec4  U  = lu > 4.0 * lm + 1e-4 ? vec4(Q0.rgb * ((4.0 * lm + 1e-4) / lu), Q0.a) : Q0;

    vec4  H = U;
    float n = 0.0, MaxMove = 1e3;
    if (F[Push.m_Slot].Params.z > 0.5)
    {
        // where the reflection was last frame: its virtual point (the hit behind the mirror; a miss: far along the view ray) on sharp
        // receivers, the surface's point on rough ones; a mover by its own motion
        const float Hit = unpackHalf2x16(R.z).x;
        const vec3  Xv  = P + (P / Dist) * (Hit > 0.0 ? Hit : 1e4);
        const float s   = smoothstep(0.1, 0.5, Rough);
        const vec4  Mo  = texelFetch(GMotion, p, 0);
        vec4 C = F[Push.m_Slot].ToPrev * vec4(mix(Xv, P, s), 1.0);
        vec2 Pn = C.xy / C.w;
        if (Mo.w > 0.5) { Pn = Mo.xy; MaxMove = 4.0; }
        const vec4  Cs  = F[Push.m_Slot].ToPrev * vec4(P, 1.0);
        const float dP  = Mo.w > 0.5 ? Mo.z : Cs.w;                     // (the surface's view depth last frame)
        const vec2  hp  = (Pn * 0.5 + 0.5) * vec2(Size()) - 0.5;
        const vec2  i0  = floor(hp), f = hp - i0;
        vec4 Hs = vec4(0.0); float Ws = 0.0, Ns = 0.0;
        for (int t = 0; t < 4; ++t)
        {
            const ivec2 q = ivec2(i0) + ivec2(t & 1, t >> 1);
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, Size()))) continue;
            const uvec4 Hq = History[uint(q.y * Size().x + q.x)];
            if (RecClass(Hq) == 0u) continue;
            if (dot(MetaNormal(Hq.w), N) < 0.9) continue;
            const vec2 rf = unpackHalf2x16(Hq.z);                       // (roughness, frames)
            if (abs(rf.x - Rough) > 0.1) continue;
            const float dq = MetaDist(Hq.w);
            if (s > 0.5 && abs(dq - dP) > 0.03 * dP + 0.02) continue;   // (by the surface's motion: the same surface)
            const float w = ((t & 1) != 0 ? f.x : 1.0 - f.x) * ((t >> 1) != 0 ? f.y : 1.0 - f.y);
            Hs += w * RecQ(Hq); Ns += w * rf.y; Ws += w;
        }
        if (Ws > 0.25) { H = Hs / Ws; n = Ns / Ws; }
    }
    // far out of the window (a light went off, a mover uncovered something): a short history; then the clamp
    const float g = 1.0 + Rough;
    if (any(greaterThan(abs(H - m1), 3.0 * g * sd + 1e-6))) n = min(n, 2.0);
    H = clamp(H, m1 - g * sd, m1 + g * sd);
    const float MaxN = min(mix(6.0, 24.0, smoothstep(0.05, 0.4, Rough)), MaxMove);
    n = min(floor(n + 0.5) + 1.0, MaxN);
    const uvec4 O = WithQ(R, mix(H, U, 1.0 / n));
    Temporal[i] = uvec4(O.x, O.y, packHalf2x16(vec2(Rough, n)), O.w);
}

void SpatialPass(ivec2 p)
{
    const uint  i = uint(p.y * Size().x + p.x);
    const uvec4 T = Temporal[i];
    if (RecClass(T) == 0u) { History[i] = uvec4(0u); return; }
    const vec2  rf = unpackHalf2x16(T.z);                               // (roughness, frames)
    const vec3  N  = MetaNormal(T.w);
    const float d  = MetaDist(T.w);
    const float r  = min(rf.x * 12.0, 6.0) * (rf.y < 4.0 ? 1.5 : 1.0);
    vec4  S = RecQ(T);
    float W = 1.0;
    if (r >= 0.75)
    {
        const float a0 = F[Push.m_Slot].Params.w * 2.39996323;         // (the golden angle: a new rotation each frame)
        for (int t = 0; t < 8; ++t)
        {
            const float a = a0 + float(t) * 2.39996323, rr = r * sqrt((float(t) + 0.5) / 8.0);
            const ivec2 q = p + ivec2(round(vec2(cos(a), sin(a)) * rr));
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, Size()))) continue;
            const uvec4 Tq = Temporal[uint(q.y * Size().x + q.x)];
            if (RecClass(Tq) == 0u) continue;
            const float dq = MetaDist(Tq.w);
            if (abs(dq - d) > 0.05 * d) continue;
            const float w = pow(max(dot(MetaNormal(Tq.w), N), 0.0), 32.0) * exp(-abs(unpackHalf2x16(Tq.z).x - rf.x) * 20.0);
            S += w * RecQ(Tq); W += w;
        }
        S /= W;
    }
    else
    {
        // a sharp receiver (no lobe to filter): its coverage feathered over its 3 x 3 neighbours of the same surface - the screen trace's
        // hit / miss edge (a ray passing behind a sphere: the probe beyond) was a hard, aliased cut in a mirror (Tomas). Its light stays its
        // own where it hit, its neighbours' hits' where it missed
        float C = 0.0, Cw = 0.0; vec3 Hs = vec3(0.0); float Hw = 0.0;
        for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            const ivec2 q = p + ivec2(x, y);
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, Size()))) continue;
            const uvec4 Tq = Temporal[uint(q.y * Size().x + q.x)];
            if (RecClass(Tq) == 0u || abs(MetaDist(Tq.w) - d) > 0.05 * d || dot(MetaNormal(Tq.w), N) < 0.8) continue;
            const float w = (x == 0 && y == 0) ? 2.0 : 1.0;
            const vec4  Q = RecQ(Tq);
            C += w * Q.a; Cw += w;
            Hs += w * Q.rgb; Hw += w * Q.a;
        }
        const float Cf = Cw > 0.0 ? C / Cw : S.a;
        const vec3  U  = S.a > 1e-3 ? S.rgb / S.a : (Hw > 1e-4 ? Hs / Hw : vec3(0.0));
        S = vec4(U * Cf, Cf);
    }
    const uvec4 O = WithQ(T, S);
    Records[Base() + i] = uvec4(O.x, O.y, Records[Base() + i].z, O.w);
    History[i] = uvec4(O.x, O.y, T.z, O.w);
}

void main()
{
    const ivec2 p  = ivec2(gl_GlobalInvocationID.xy);
    const bool  bIn = p.x < Size().x && p.y < Size().y;
    if (Push.m_Pass == 0u)
    {
        LoadTile();                                                     // (every lane: the barrier)
        if (bIn) TemporalPass(p);
    }
    else if (bIn) SpatialPass(p);
}
