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
float RawAO(in const ivec2 p, in const ivec2 Q, in const float z, in const vec3 P, in const float Dist)
{
    const vec3 V = -P / Dist, N = NormalAt(Q);
    // the 4 x 4 pattern: 16 rotations a tile (a stratified order), the steps' offsets from the same index
    const uint  k16 = uint((p.x & 3) | ((p.y & 3) << 2));
    const uint  Order[16] = uint[16](0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    const float n0 = (float(Order[k16]) + 0.5) / 16.0, n1 = fract(float(Order[(k16 + 5u) & 15u]) / 16.0 + 0.03125);
    const float RadiusPx = min(Push.Radius * Push.PixelsPerUnit / Dist, 64.0);
    if (RadiusPx < 1.0) return 1.0;
    float Sum = 0.0, Wsum = 0.0;
    for (int s = 0; s < slices_v; ++s)
    {
        const float Phi = (float(s) + n0) / float(slices_v) * PI;
        const vec2  Dir = vec2(cos(Phi), sin(Phi));
        // the slice's direction in the world: the step of a few pixels along Dir at this depth
        const vec3  Dw    = WorldAt(FullTexel(p) + ivec2(round(Dir * 8.0 * float(Push.Div))), z) - P;
        const vec3  Ortho = normalize(Dw - V * dot(Dw, V));
        const vec3  Axis  = normalize(cross(Ortho, V));
        const vec3  Np    = N - Axis * dot(N, Axis);
        const float NpLen = length(Np);
        if (NpLen < 1e-4) continue;
        const float n = sign(dot(Ortho, Np)) * acos(clamp(dot(Np, V) / NpLen, -1.0, 1.0));
        uint Mask = 0u;
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
                const vec3  d  = WorldAt(Qs, zs) - P;
                const float L  = length(d);
                if (L < 1e-5 || L > Push.Radius) continue;          // (short range: beyond it the GI's own visibility holds)
                const vec3  db = d - V * Push.Thickness;
                // the front and back horizons' angles from the view direction, as fractions of the hemisphere around n (HTraceAO's)
                vec2 h = vec2(acos(clamp(dot(d, V) / L, -1.0, 1.0)), acos(clamp(dot(db, V) / max(length(db), 1e-5), -1.0, 1.0)));
                h = side == 0 ? clamp((n + h + HALF_PI) / PI, 0.0, 1.0) : clamp((n - h.yx + HALF_PI) / PI, 0.0, 1.0);
                h = h * h * (3.0 - 2.0 * h);
                const uvec2 b = uvec2(floor(vec2(min(h.x, h.y), max(h.x, h.y)) * 32.0));
                const uint lo = b.x < 32u ? 0xFFFFFFFFu << b.x : 0u;
                const uint hi = b.y != 0u ? 0xFFFFFFFFu >> (32u - b.y) : 0u;
                Mask |= lo & hi;
            }
        }
        Sum  += (1.0 - float(bitCount(Mask)) / 32.0) * NpLen;
        Wsum += NpLen;
    }
    return Wsum > 0.0 ? clamp(Sum / Wsum, 0.0, 1.0) : 1.0;
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
    if (Push.Mode == 0u)
    {
        if (z >= 1.0) { AORaw[i] = vec2(1.0, 0.0); return; }
        const vec3 P = WorldAt(Q, z);
        const float Dist = length(P);
        AORaw[i] = vec2(RawAO(p, Q, z, P, Dist), Dist);
        return;
    }

    // Mode 1: the 4 x 4 tile's rotations summed (edge aware: the distance's match), then the history
    const vec2 C = AORaw[i];
    if (C.y <= 0.0) { AOOut[i] = vec2(1.0, 0.0); return; }
    const float Dist = C.y;
    float S = 0.0, Wt = 0.0;
    for (int y = -1; y <= 2; ++y) for (int x = -1; x <= 2; ++x)
    {
        const ivec2 q = clamp(p + ivec2(x, y), ivec2(0), ivec2(int(Push.W) - 1, int(Push.H) - 1));
        const vec2  A = AORaw[uint(q.y) * Push.W + uint(q.x)];
        if (A.y <= 0.0) continue;
        const float w = max(0.0, 1.0 - abs(A.y - Dist) / (0.03 * Dist + 0.01));
        S += A.x * w; Wt += w;
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
    const vec3 N = NormalAt(Q);
    vec2 on = N.xy / (abs(N.x) + abs(N.y) + abs(N.z));
    if (N.z < 0.0) on = (1.0 - abs(on.yx)) * vec2(on.x >= 0.0 ? 1.0 : -1.0, on.y >= 0.0 ? 1.0 : -1.0);
    const uvec2 Oq = uvec2(clamp(on * 0.5 + 0.5, 0.0, 1.0) * 63.0 + 0.5);
    GIScreen[8u + i] = uvec4(packHalf2x16(vec2(AO, 0.0)), packHalf2x16(vec2(0.0, Dist)), 0u, (Oq.x << 16) | (Oq.y << 22) | (3u << 28));
}
