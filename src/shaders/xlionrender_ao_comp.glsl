#version 450
// Ambient occlusion (xLION_Docs ao_design.md): GTAO with visibility bitmasks (Jimenez 2016; Therrien 2023 - HTraceAO's GTAO path), at
// half size on the GI's grid, from the G-buffer prepass's depth and normal (the gather's checkerboard sample), right after it. Short range
// only (Push.Radius): the occlusion below the GI's resolution. Each frame good on its own (Unreal's GTAO: a 4 x 4 interleaved pattern of
// slice rotations and a 4 x 4 edge-aware blur that sums them - a still and a moving camera alike; HypeHype's GTAO (Aaltonen, SIGGRAPH
// 2024) the same with no history at all). The history (Push.bPrev) is off: a moving frame is a still one
//  Mode 0: the raw AO a pixel, its slices rotated by its place in the 4 x 4 pattern -> AORaw (AO, distance)
//  Mode 1: AORaw's 4 x 4 around the pixel weighed by distance (the pattern's 16 rotations), then a third of the way from last frame's
//          (reprojected, the same surface by its distance) -> AOOut, and an "AO only" entry in the GI screen buffer's pixel (x: the AO
//          half, y.y: the distance, w: the normal guide, bits 28-29 3, bits 30-31 0) - what the materials' upsample reads (mb_standard_pbr.frag
//          SurfelGIScreen); the gather replaces it where it has GI (its irradiance by the AO)
layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, set = 0, binding = 0) buffer screen_b { uvec4 GIScreen[]; };             // its header: last frame's world to clip (1-4), the move (5); its pixels
layout(std430, set = 0, binding = 1) readonly buffer table_b  { uint GITable[]; };      // the gather's table: clip -> camera-centred world (0-15)
layout(std430, set = 0, binding = 2) readonly buffer prev_b   { vec2 AOPrev[]; };       // last frame's AOOut
layout(std430, set = 0, binding = 3) buffer out_b             { vec2 AOOut[]; };
layout(std430, set = 0, binding = 4) buffer raw_b             { vec2 AORaw[]; };
layout(set = 1, binding = 0) uniform sampler2D GPreDepth;
layout(set = 1, binding = 1) uniform sampler2D GPreNormal;
layout(set = 1, binding = 2) uniform sampler2D GPreId;
layout(set = 1, binding = 3) uniform sampler2D LastHDR;   // (XLION_SSGI, Mode bit 4: last frame's lit image - the screen GI's light. ponytail: read at this frame's texel, not reprojected: a moving camera lags a frame)      // (its y < 0: a back face - the normal turned to its visible side)

layout(push_constant) uniform push_t
{
    uint  W, H, Div, Mode;
    float Radius;           // world units
    float Thickness;        // world units: a sample's assumed depth (the bitmask's back face)
    float PixelsPerUnit;    // half-size pixels a world unit spans at distance 1
    uint  TableBase;
    uint  bPrev;            // last frame's AO is there (same size, no camera cut)
    float Blend;            // the new frame's weight in the history
} Push;

const float PI = 3.14159265, HALF_PI = 1.57079633;
const int   slices_v = 2, steps_v = 4;      // (two orthogonal slices: HypeHype's "+" cross; 4 steps a side - theirs 3, fp16 on phones)

mat4  g_C2W;
ivec2 g_Size;

ivec2 FullTexel(in const ivec2 p)
{
    const int D = int(Push.Div);
    return min(p * D + ivec2(((p.x ^ p.y) & 1) == 0 ? D / 2 - 1 : D / 2), g_Size - 1);     // (xlionrender_gi_gather_common.glsl's sample)
}
vec3 WorldAt(in const ivec2 q, in const float z)
{
    const vec4 W = g_C2W * vec4((vec2(q) + 0.5) / vec2(g_Size) * 2.0 - 1.0, z, 1.0);
    const float d = texelFetch(GPreId, q, 0).z;       // (the G-buffer's exact distance, as the gather's)
    return d > 0.0 ? normalize(W.xyz / W.w) * d : W.xyz / W.w;
}
vec3 RayAt(in const ivec2 q, in const float z)         // (the depth buffer's point on the texel's ray: the slice's direction)
{
    const vec4 W = g_C2W * vec4((vec2(q) + 0.5) / vec2(g_Size) * 2.0 - 1.0, z, 1.0);
    return W.xyz / W.w;
}
vec3 NormalAt(in const ivec2 Q)
{
    const vec2 o = texelFetch(GPreNormal, Q, 0).xy;
    vec3 N = vec3(o, 1.0 - abs(o.x) - abs(o.y));
    if (N.z < 0.0) N.xy = (1.0 - abs(N.yx)) * vec2(N.x >= 0.0 ? 1.0 : -1.0, N.y >= 0.0 ? 1.0 : -1.0);
    return normalize(N);
}

// The raw AO at half-size pixel p (Mode 0)
// (XLION_SSGI: Gi, the near field's irradiance - each sample's light times the cosine-weighted share of the sectors it newly occludes, SSRT3's)
float RawAO(in const ivec2 p, in const ivec2 Q, in const float z, in const vec3 P, in const float Dist, out vec3 Gi)
{
    Gi = vec3(0.0);
    const bool bGi = (Push.Mode & 4u) != 0u;
    vec3 GiSum = vec3(0.0);
    const vec3 V = -P / Dist, N0 = NormalAt(Q), N = texelFetch(GPreId, Q, 0).y >= 0.0 ? N0 : -N0;      // (a double-sided back face: its visible side)
    // the 4 x 4 pattern: 16 rotations a tile (a stratified order), the steps' offsets from the same index
    const uint  k16 = uint((p.x & 3) | ((p.y & 3) << 2));
    const uint  Order[16] = uint[16](0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    const float n0 = (float(Order[k16]) + 0.5) / 16.0, n1 = fract(float(Order[(k16 + 5u) & 15u]) / 16.0 + 0.03125);
    const float RadiusPx = min(Push.Radius * Push.PixelsPerUnit / Dist, 64.0);
    if (RadiusPx < 1.0) return 1.0;     // (Gi stays 0)
    float Sum = 0.0, Wsum = 0.0;
    for (int s = 0; s < slices_v; ++s)
    {
        const float Phi = (float(s) + n0) / float(slices_v) * PI;
        const vec2  Dir = vec2(cos(Phi), sin(Phi));
        // the slice's direction in the world: the step of a few pixels along Dir at this depth
        const vec3  Dw    = RayAt(FullTexel(p) + ivec2(round(Dir * 8.0 * float(Push.Div))), z) - RayAt(FullTexel(p), z);
        const vec3  Ortho = normalize(Dw - V * dot(Dw, V));
        const vec3  Axis  = normalize(cross(Ortho, V));
        const vec3  Np    = N - Axis * dot(N, Axis);
        const float NpLen = length(Np);
        if (NpLen < 1e-4) continue;
        // (the projected normal within the visible hemisphere, as GTAO's saturate: a grazing smooth normal that leans away from the eye
        // put the hemisphere behind the surface - every sample an occluder, a black wedge along Sponza's arch, Tomas)
        const float n = sign(dot(Ortho, Np)) * acos(clamp(dot(Np, V) / NpLen, 0.0, 1.0));
        uint Mask = 0u;
        vec3 GiS = vec3(0.0);
        for (int k = 0; k < steps_v; ++k)
        {
            const float t   = pow((float(k) + n1) / float(steps_v), 2.0);
            const vec2  Off = Dir * max(t * RadiusPx, 1.0 + float(k));
            for (int side = 0; side < 2; ++side)
            {
                const ivec2 q = ivec2(round(vec2(p) + (side == 0 ? -Off : Off)));
                if (q.x < 0 || q.y < 0 || uint(q.x) >= Push.W || uint(q.y) >= Push.H) continue;
                const ivec2 Qs = FullTexel(q);
                const float zs = texelFetch(GPreDepth, Qs, 0).r;
                if (zs >= 1.0) continue;
                const vec3  d  = RayAt(Qs, zs) - P;      // (the depth's point, not GPreId's exact distance: 4 bytes a sample, not 20)
                const float L  = length(d);
                if (L < 1e-5 || L > Push.Radius) continue;          // (short range: beyond it the GI's own visibility holds)
                const vec3  db = d - V * Push.Thickness;
                // the front and back horizons' cosines from the view direction, faded to the hemisphere's edge over the range's last 60 %
                // (HTraceAO's falloff: a sample at the range's end occluded fully and one past it not at all drew a sharp arc that moved
                // with the camera - Tomas's Sponza arch), then as fractions of the hemisphere around n (HTraceAO's)
                const float Fade = clamp((Push.Radius - L) / (0.6 * Push.Radius), 0.0, 1.0);
                const float MinCos = cos(side == 0 ? n - HALF_PI : n + HALF_PI);
                const vec2  c = mix(vec2(MinCos), vec2(dot(d, V) / L, dot(db, V) / max(length(db), 1e-5)), Fade);
                vec2 h = vec2(acos(clamp(c.x, -1.0, 1.0)), acos(clamp(c.y, -1.0, 1.0)));
                h = side == 0 ? clamp((n + h + HALF_PI) / PI, 0.0, 1.0) : clamp((n - h.yx + HALF_PI) / PI, 0.0, 1.0);
                h = h * h * (3.0 - 2.0 * h);
                const uvec2 b = uvec2(floor(vec2(min(h.x, h.y), max(h.x, h.y)) * 32.0));
                const uint lo = b.x < 32u ? 0xFFFFFFFFu << b.x : 0u;
                const uint hi = b.y != 0u ? 0xFFFFFFFFu >> (32u - b.y) : 0u;
                if (bGi) { const uint New = (lo & hi) & ~Mask; if (New != 0u) GiS += texelFetch(LastHDR, Qs, 0).rgb * (float(bitCount(New)) / 32.0); }
                Mask |= lo & hi;
            }
        }
        Sum  += (1.0 - float(bitCount(Mask)) / 32.0) * NpLen;
        GiSum += GiS * NpLen;
        Wsum += NpLen;
    }
    if (Wsum > 0.0 && bGi) { Gi = PI * GiSum / Wsum; if (any(isnan(Gi)) || any(isinf(Gi))) Gi = vec3(0.0); }
    return Wsum > 0.0 ? clamp(Sum / Wsum, 0.0, 1.0) : 1.0;
}

// XLION_AO_CORTAO (Mode bit 3, a prototype - xLION_Docs research/cortao_review_2026-11-14.md): CORTAO's idea (Isheden 2026, MIT,
// E:\source_code\GI\CORTAO), in our camera-centred world space: each of a few depth samples around the pixel is a sphere (its radius from the
// sample's share of the disc, slope-compensated by its normal against the view), and the occlusion is the share of a fixed set of rays in the
// normal's hemisphere that any sphere blocks - a bitmask, as brute force ray tracing has it, not two horizon slices that skew on surfaces not
// facing the camera. No lookup table (the analytic per-ray test) and no depth mips: the screen offsets at the pixel's radius instead
const int cortao_samples_v = 8, cortao_rays_v = 16;
const float cortao_golden_v = 2.3999632297;
vec3 CortaoRay(in const int k)
{
    const float a = float(k) * cortao_golden_v, r = sqrt((float(k) + 0.5) / float(cortao_rays_v));
    return vec3(cos(a) * r, sin(a) * r, sqrt(1.0 - r * r));
}
float CortaoAO(in const ivec2 p, in const vec3 P, in const float Dist, in const vec3 N0)
{
    const vec3  V = -P / Dist;
    const vec3  N = dot(N0, V) < 0.0 ? N0 - V * (dot(N0, V) - 1e-3) : N0;      // (a normal facing away: into the visible hemisphere)
    const float RadiusPx = min(Push.Radius * Push.PixelsPerUnit / Dist, 64.0);
    if (RadiusPx < 1.0) return 1.0;
    // the per-pixel randoms of the 4 x 4 pattern (as RawAO's: the filter's 4 x 4 sums the rotations)
    const uint  k16 = uint((p.x & 3) | ((p.y & 3) << 2));
    const uint  Order[16] = uint[16](0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    const float Rot = (float(Order[k16]) + 0.5) / 16.0 * 6.2831853, Dither = fract(float(Order[(k16 + 5u) & 15u]) / 16.0 + 0.03125);
    // the rays' frame around N, turned by the pixel's rotation (Frisvad's basis)
    vec3 T, B;
    { const vec3 n = normalize(N); const float s = n.z >= 0.0 ? 1.0 : -1.0, a = -1.0 / (s + n.z), b = n.x * n.y * a;
      T = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x); B = vec3(b, s + n.y * n.y * a, -n.y); }
    const float cr = cos(Rot), sr = sin(Rot);
    const mat3 TBN = mat3(cr * T + sr * B, cr * B - sr * T, normalize(N));
    const vec3 Pc = P + N * (Dist / max(Push.PixelsPerUnit, 1.0));      // (the centre pushed off its own surface by a half-size pixel's size there - CORTAO's normalBias x depth)
    uint Bits = 0u;
    for (int i = 0; i < cortao_samples_v; ++i)
    {
        const float a = float(i) * cortao_golden_v + Rot;
        const float r0 = pow(float(i) / float(cortao_samples_v), 1.75), r1 = pow(float(i + 1) / float(cortao_samples_v), 1.75);
        const float w = sqrt(r1 * r1 - r0 * r0);      // (the share of the disc this sample stands for, as a radius)
        const vec2  Off = vec2(cos(a), sin(a)) * mix(r0, r1, Dither) * RadiusPx;
        const ivec2 q = ivec2(round(vec2(p) + Off));
        if (q.x < 0 || q.y < 0 || uint(q.x) >= Push.W || uint(q.y) >= Push.H) continue;
        const ivec2 Qs = FullTexel(q);
        const float zs = texelFetch(GPreDepth, Qs, 0).r;
        if (zs >= 1.0) continue;
        const vec3  d = RayAt(Qs, zs) - Pc;
        const float L2 = dot(d, d);
        if (L2 < 1e-10 || L2 > Push.Radius * Push.Radius || dot(d, N) <= 0.0) continue;
        const float iL = inversesqrt(L2);
        vec3 Dir = d * iL;
        const vec3  Ns = NormalAt(Qs);
        const float Slope = inversesqrt(max(dot(Ns, V), 1.0 / 16.0));      // (CORTAO's slope compensation: a sample seen at a grazing angle stands for more surface)
        float Rad = min(Push.Radius * w * Slope * iL, 1.0);      // (the occluder's normalized angular radius; at most a quarter sphere - one close sample covered the whole hemisphere)
        if (dot(Ns, Dir) < 0.5) Dir = normalize(Dir - Ns * (Rad / max(1.0 - Rad * Rad * 0.5, 1e-6)));     // (a front face: moved into its object by its radius - no self-occlusion)
        else Rad *= mix(1.0, Slope, 0.5);      // (a back face: boosted, no leak)
        const vec3  H = transpose(TBN) * Dir;
        const float CosA = max(0.0, 1.0 - Rad * Rad * 0.5);
        for (int k = 0; k < cortao_rays_v; ++k) if (dot(CortaoRay(k), H) > CosA) Bits |= 1u << uint(k);
    }
    return 1.0 - float(bitCount(Bits)) / float(cortao_rays_v);
}

void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (uint(p.x) >= Push.W || uint(p.y) >= Push.H) return;
    const uint i = uint(p.y) * Push.W + uint(p.x);
    g_Size = textureSize(GPreDepth, 0);
    const uint T = Push.TableBase;
    g_C2W = mat4(uintBitsToFloat(uvec4(GITable[T + 0u], GITable[T + 1u], GITable[T + 2u], GITable[T + 3u])),
                 uintBitsToFloat(uvec4(GITable[T + 4u], GITable[T + 5u], GITable[T + 6u], GITable[T + 7u])),
                 uintBitsToFloat(uvec4(GITable[T + 8u], GITable[T + 9u], GITable[T + 10u], GITable[T + 11u])),
                 uintBitsToFloat(uvec4(GITable[T + 12u], GITable[T + 13u], GITable[T + 14u], GITable[T + 15u])));
    const ivec2 Q = FullTexel(p);
    const float z = texelFetch(GPreDepth, Q, 0).r;
    const bool bGi = (Push.Mode & 4u) != 0u;      // (XLION_SSGI off: its two planes neither read nor written - nothing reads them; the filter's 16 taps: a third of the loads)
    const uint NP = Push.W * Push.H;      // (the buffers: AO, distance at i; the screen GI's rg at N + i, b at 2N + i)
    if ((Push.Mode & 3u) == 0u)
    {
        if (z >= 1.0) { AORaw[i] = vec2(1.0, 0.0); AORaw[NP + i] = vec2(0.0); AORaw[2u * NP + i] = vec2(0.0); return; }
        const vec3 P = WorldAt(Q, z);
        const float Dist = length(P);
        vec3 Gi;
        if ((Push.Mode & 8u) != 0u) { const vec3 N0 = NormalAt(Q); Gi = vec3(0.0); AORaw[i] = vec2(CortaoAO(p, RayAt(Q, z), Dist, texelFetch(GPreId, Q, 0).y >= 0.0 ? N0 : -N0), Dist); }
        else AORaw[i] = vec2(RawAO(p, Q, z, RayAt(Q, z), Dist, Gi), Dist);
        if (bGi) { AORaw[NP + i] = Gi.rg; AORaw[2u * NP + i] = vec2(Gi.b, 0.0); }
        return;
    }

    // Mode 1: the 4 x 4 tile's rotations summed (edge aware: the distance's match), then the history
    const vec2 C = AORaw[i];
    if (C.y <= 0.0) { AOOut[i] = vec2(1.0, 0.0); AOOut[NP + i] = vec2(0.0); AOOut[2u * NP + i] = vec2(0.0); return; }
    const float Dist = C.y;
    float S = 0.0, Wt = 0.0;
    vec3  G = vec3(0.0);
    for (int y = -1; y <= 2; ++y) for (int x = -1; x <= 2; ++x)
    {
        const ivec2 q = clamp(p + ivec2(x, y), ivec2(0), ivec2(int(Push.W) - 1, int(Push.H) - 1));
        const uint  j = uint(q.y) * Push.W + uint(q.x);
        const vec2  A = AORaw[j];
        if (A.y <= 0.0) continue;
        const float w = max(0.0, 1.0 - abs(A.y - Dist) / (0.03 * Dist + 0.01));
        S += A.x * w; Wt += w;
        if (bGi) G += vec3(AORaw[NP + j], AORaw[2u * NP + j].x) * w;
    }
    if (bGi)
    {
        G = Wt > 1e-4 ? G / Wt : vec3(AORaw[NP + i], AORaw[2u * NP + i].x);
        AOOut[NP + i] = G.rg; AOOut[2u * NP + i] = vec2(G.b, 0.0);
    }
    float AO = Wt > 1e-4 ? S / Wt : C.x;

    // Temporal: last frame's where this point was, the same surface (its distance, 1 %)
    if (Push.bPrev != 0u)
    {
        const vec3 P = WorldAt(Q, z);
        const mat4 PrevW2C = mat4(uintBitsToFloat(GIScreen[1]), uintBitsToFloat(GIScreen[2]), uintBitsToFloat(GIScreen[3]), uintBitsToFloat(GIScreen[4]));
        const vec3 Prev = P + uintBitsToFloat(GIScreen[5].xyz);
        const vec4 Cp = PrevW2C * vec4(Prev, 1.0);
        if (Cp.w > 1e-6)
        {
            const ivec2 R = ivec2(floor((Cp.xy / Cp.w * 0.5 + 0.5) * vec2(float(Push.W), float(Push.H))));
            if (R.x >= 0 && R.y >= 0 && uint(R.x) < Push.W && uint(R.y) < Push.H)
            {
                const vec2 Hh = AOPrev[uint(R.y) * Push.W + uint(R.x)];
                const float Pd = length(Prev);
                if (Hh.y > 0.0 && abs(Hh.y - Pd) < 0.01 * Pd + 0.005) AO = mix(Hh.x, AO, Push.Blend);
            }
        }
    }
    AOOut[i] = vec2(AO, Dist);
    const vec3 N0 = NormalAt(Q), N = texelFetch(GPreId, Q, 0).y >= 0.0 ? N0 : -N0;     // (the upsample's guide: the visible side)
    vec2 on = N.xy / (abs(N.x) + abs(N.y) + abs(N.z));
    if (N.z < 0.0) on = (1.0 - abs(on.yx)) * vec2(on.x >= 0.0 ? 1.0 : -1.0, on.y >= 0.0 ? 1.0 : -1.0);
    const uvec2 Oq = uvec2(clamp(on * 0.5 + 0.5, 0.0, 1.0) * 63.0 + 0.5);
    GIScreen[8u + i] = uvec4(packHalf2x16(vec2(AO, 0.0)), packHalf2x16(vec2(0.0, Dist)), 0u, (Oq.x << 16) | (Oq.y << 22) | (3u << 28));
}
