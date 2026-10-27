#version 450
// The world-cell light lists built on the GPU (features/lights/xlionrender_light_cells.h: the same layout and lists as its CPU Build;
// documentation/lighting_strategy.md P2). Three dispatches over this frame's region of the cell ring:
//   pass 0, count: a thread a cell (a workgroup a 4 x 4 x 4 block of one level's window) - each light whose range at the level covers
//            the cell, whose sphere reaches the cell's box and whose cone (a spot's) does not miss it; the count into the header's
//            second uint.
//   pass 1, scan: one workgroup - the cells' first entries (the counts' exclusive prefix sum) into the headers' first uint.
//   pass 2, fill: a thread a cell again - its lights in light order; with a head (SAMPLED) the head/tail split of SplitCell: the
//            HeadCount most important (ties: the first in light order), back in light order, then the tail in light order with
//            its cumulative probabilities (floored at 1% of the tail's largest). Then the header: the count and the head count.
// Light order is kept (the shading sums the same lights in the same order as the CPU's lists). The input, at m_LightBase: the 5
// levels' window origins (ints) and the level-0 cell size (float bits), 16 uints; then per light 12 uints - world position, radius,
// spot direction, outer half angle (< 0: a point light), power.
layout(local_size_x = 64) in;

#include "xlionrender_light_eval.glsl"

const int  kLevels     = 5;         // light_cells::levels_v
const int  kCells      = 16;        // light_cells::cells_v
const int  kLevelCells = kCells * kCells * kCells;
const uint kMaxHead    = 15u;       // light_cells::max_head_v
const uint kLightUints = 12u;

layout(std430, set = 0, binding = 0) coherent buffer cell_b { uint Cells[]; };
layout(push_constant) uniform push_t { uint m_Pass, m_LightBase, m_nLights, m_CellBase, m_HeadCount, m_Bound, m_bSkipHidden, m_Pad1; } Push;

shared ivec3 s_Lo[64], s_Hi[64];
shared uint  s_Hit[64], s_List[64], s_nList;
shared vec4  s_PosR[64], s_DirOuter[64];     // the chunk's lights (their reads from the buffer are slow: a thread tests thousands)
shared float s_Power[64];

// Pass 2, with a head: the cell's most important entries so far, (importance desc, list order asc) - the head and, after it, the
// tail's largest (its floor); kept by Walk as it fills
float g_TopI[kMaxHead + 1u];
uint  g_TopK[kMaxHead + 1u], g_TopL[kMaxHead + 1u];
uint  g_nTop = 0u;
shared uint  s_Partial[64];

ivec3 Origin(int L) { const uint B = Push.m_LightBase + uint(3 * L); return ivec3(int(Cells[B]), int(Cells[B + 1u]), int(Cells[B + 2u])); }
float CellSize()     { return uintBitsToFloat(Cells[Push.m_LightBase + 15u]); }
uint  LightAt(uint i) { return Push.m_LightBase + 16u + kLightUints * i; }
vec3  LightPos(uint i)   { const uint B = LightAt(i); return vec3(uintBitsToFloat(Cells[B]), uintBitsToFloat(Cells[B + 1u]), uintBitsToFloat(Cells[B + 2u])); }
float LightRadius(uint i) { return uintBitsToFloat(Cells[LightAt(i) + 3u]); }

int Slot(ivec3 C) { const ivec3 M = ((C % kCells) + kCells) % kCells; return (M.z * kCells + M.y) * kCells + M.x; }

// The chunk's light j's squared distance to cell C's box at level size S (the CPU's order of operations, no fused multiply-adds), or
// -1 when the light does not reach it: its sphere misses the box, or its cone (a spot's) misses the cell's bounding sphere (OutsideCone)
float Reach(uint j, ivec3 C, float S)
{
    const vec3  P = s_PosR[j].xyz;
    const float R = s_PosR[j].w;
    precise float D2 = 0.0;
    for (int a = 0; a < 3; ++a)
    {
        precise const float Min = float(C[a]) * S, Max = Min + S, V = P[a];
        if (V < Min) D2 += (Min - V) * (Min - V); else if (V > Max) D2 += (V - Max) * (V - Max);
    }
    precise const float R2 = R * R;
    if (D2 > R2) return -1.0;
    const float Outer = s_DirOuter[j].w;
    if (Outer >= 0.0)
    {
        const vec3  Dir    = s_DirOuter[j].xyz;
        const vec3  Center = (vec3(C) + 0.5) * S;
        const vec3  V      = Center - P;
        const float d = length(V), rc = S * 0.8660254;
        if (d > rc && acos(clamp(dot(V, Dir) / d, -1.0, 1.0)) - asin(rc / d) > Outer) return -1.0;
    }
    return D2;
}

// light_cells::Hidden: a cell of level L >= 1 the window of level L - 1 covers - no lookup reaches it, it stays empty
bool Hidden(int L, ivec3 C)
{
    if (Push.m_bSkipHidden == 0u || L == 0) return false;
    const ivec3 O = Origin(L - 1);
    return all(greaterThanEqual(2 * C, O)) && all(lessThanEqual(2 * C + 1, O + kCells - 1));
}

// This thread's cell (level, world cell) and its slot in the cell list
void ThisCell(out int L, out ivec3 C, out uint Cell)
{
    L = int(gl_WorkGroupID.y);
    const uint  b = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    const ivec3 Block = ivec3(int(b & 3u), int((b >> 2) & 3u), int(b >> 4)) * 4;
    C    = Origin(L) + Block + ivec3(int(t & 3u), int((t >> 2) & 3u), int(t >> 4));
    Cell = uint(L * kLevelCells + Slot(C));
}

// Pass 0 and 2's light walk: in chunks of 64 lights, each light's cell range at this level (the CPU's ForEachCell bounds) into shared
// memory and the lights whose range meets the workgroup's block listed in light order; then each thread tests those for its cell.
// bFill: write the entries from First (else only count); with a head (SAMPLED) the importance into the entry's second uint (Split)
uint Walk(int L, ivec3 C, float S, bool bFill, uint First, uint Max)
{
    const ivec3 Lo0 = Origin(L), Hi0 = Lo0 + kCells - 1;
    const uint  b = gl_WorkGroupID.x;
    const ivec3 B0 = Lo0 + ivec3(int(b & 3u), int((b >> 2) & 3u), int(b >> 4)) * 4, B1 = B0 + 3;
    const uint  EntryBase = Push.m_CellBase + uint(2 * kLevels * kLevelCells);
    const bool  bHidden = Hidden(L, C);
    if (Hidden(L, B0) && Hidden(L, B1)) return 0u;                          // (the whole block: the workgroup skips the walk)
    uint n = 0u;
    for (uint c0 = 0u; c0 < Push.m_nLights; c0 += 64u)
    {
        const uint i = c0 + gl_LocalInvocationID.x;
        ivec3 Lo = ivec3(1), Hi = ivec3(0);
        if (i < Push.m_nLights)
        {
            const uint  A = LightAt(i);
            const vec3  P = LightPos(i);
            const float R = LightRadius(i);
            Lo = max(ivec3(floor((P - R) / S)), Lo0);
            Hi = min(ivec3(floor((P + R) / S)), Hi0);
            s_PosR[gl_LocalInvocationID.x]     = vec4(P, R);
            s_DirOuter[gl_LocalInvocationID.x] = vec4(uintBitsToFloat(Cells[A + 4u]), uintBitsToFloat(Cells[A + 5u]), uintBitsToFloat(Cells[A + 6u]), uintBitsToFloat(Cells[A + 7u]));
            s_Power[gl_LocalInvocationID.x]    = uintBitsToFloat(Cells[A + 8u]);
        }
        s_Lo[gl_LocalInvocationID.x] = Lo; s_Hi[gl_LocalInvocationID.x] = Hi;
        s_Hit[gl_LocalInvocationID.x] = (all(lessThanEqual(Lo, B1)) && all(greaterThanEqual(Hi, B0))) ? 1u : 0u;      // (meets the block)
        barrier();
        if (gl_LocalInvocationID.x == 0u)
        {
            uint m = 0u;
            for (uint j = 0u; j < 64u; ++j) if (s_Hit[j] != 0u) s_List[m++] = j;
            s_nList = m;
        }
        barrier();
        const uint m = bHidden ? 0u : s_nList;
        for (uint q = 0u; q < m; ++q)
        {
            const uint j = s_List[q];
            if (any(lessThan(C, s_Lo[j])) || any(greaterThan(C, s_Hi[j]))) continue;
            const float D2 = Reach(j, C, S);
            if (D2 < 0.0) continue;
            if (bFill && n < Max)
            {
                const uint i = c0 + j, e = EntryBase + 2u * (First + n);
                Cells[e] = i;
                if (Push.m_HeadCount == 0u) Cells[e + 1u] = 0u;
                else
                {
                    const float I = LightDistanceFalloff(sqrt(D2) / s_PosR[j].w, s_Power[j]);
                    Cells[e + 1u] = floatBitsToUint(I);
                    const uint T = min(Push.m_HeadCount, kMaxHead) + 1u;       // the head and the tail's largest
                    if (g_nTop < T || I > g_TopI[T - 1u])
                    {
                        uint p = min(g_nTop, T - 1u);
                        while (p > 0u && g_TopI[p - 1u] < I) { g_TopI[p] = g_TopI[p - 1u]; g_TopK[p] = g_TopK[p - 1u]; g_TopL[p] = g_TopL[p - 1u]; --p; }
                        g_TopI[p] = I; g_TopK[p] = n; g_TopL[p] = i;
                        g_nTop = min(g_nTop + 1u, T);
                    }
                }
            }
            ++n;
        }
        barrier();
    }
    return n;
}

// SplitCell: the head (the HeadCount most important; ties: the first in light order) back in light order, then the tail in light
// order with its cumulative probabilities. Walk kept the head and the tail's largest (g_Top), the importances in the entries' second
// uints. Two passes: the tail compacted (and its sum, from the end: the CPU sums from the start - the probabilities differ in the last
// bits), then the cumulative probabilities
void Split(uint First, uint n, uint H)
{
    const uint E = Push.m_CellBase + uint(2 * kLevels * kLevelCells) + 2u * First;
    precise const float Floor = max(g_TopI[H] * 0.01, 1e-12);       // (n > H: g_Top holds H + 1)
    // the head in light order (its entries' order in the list)
    for (uint a = 1u; a < H; ++a)
    {
        const uint K = g_TopK[a], Lt = g_TopL[a];
        uint p = a;
        while (p > 0u && g_TopK[p - 1u] > K) { g_TopK[p] = g_TopK[p - 1u]; g_TopL[p] = g_TopL[p - 1u]; --p; }
        g_TopK[p] = K; g_TopL[p] = Lt;
    }
    // the tail compacted after the head, from the end (an entry moves only to its own place or later: nothing unread is overwritten)
    precise float Sum = 0.0;
    int j = int(H) - 1;
    for (int k = int(n) - 1; k >= 0; --k)
    {
        if (j >= 0 && g_TopK[j] == uint(k)) { --j; continue; }
        const uint d = H + uint(k) - uint(j + 1), I = Cells[E + 2u * uint(k) + 1u];
        Cells[E + 2u * d] = Cells[E + 2u * uint(k)]; Cells[E + 2u * d + 1u] = I;
        Sum += max(uintBitsToFloat(I), Floor);
    }
    for (uint a = 0u; a < H; ++a) { Cells[E + 2u * a] = g_TopL[a]; Cells[E + 2u * a + 1u] = 0u; }
    precise float Cumulative = 0.0;
    for (uint k = H; k < n; ++k)
    {
        Cumulative += max(uintBitsToFloat(Cells[E + 2u * k + 1u]), Floor) / Sum;
        Cells[E + 2u * k + 1u] = floatBitsToUint(k + 1u == n ? 1.0 : Cumulative);
    }
}

void main()
{
    const uint Headers = Push.m_CellBase;
    if (Push.m_Pass == 1u)
    {
        // the exclusive prefix sum of the counts: 320 cells a thread, the 64 partial sums scanned in shared memory
        const uint t = gl_LocalInvocationID.x, nCells = uint(kLevels * kLevelCells), Per = (nCells + 63u) / 64u;
        uint Sum = 0u;
        for (uint c = t * Per; c < min(nCells, (t + 1u) * Per); ++c) Sum += Cells[Headers + 2u * c + 1u];
        s_Partial[t] = Sum;
        barrier();
        for (uint Off = 1u; Off < 64u; Off <<= 1)
        {
            const uint v = t >= Off ? s_Partial[t - Off] : 0u;
            barrier();
            s_Partial[t] += v;
            barrier();
        }
        uint First = s_Partial[t] - Sum;
        for (uint c = t * Per; c < min(nCells, (t + 1u) * Per); ++c) { const uint n = Cells[Headers + 2u * c + 1u]; Cells[Headers + 2u * c] = First; First += n; }
        return;
    }

    int L; ivec3 C; uint Cell;
    ThisCell(L, C, Cell);
    const float S = CellSize() * float(1 << L);
    if (Push.m_Pass == 0u) { Cells[Headers + 2u * Cell + 1u] = Walk(L, C, S, false, 0u, 0u); return; }

    // pass 2: the entries (at most what the CPU reserved: m_Bound), the split, the header
    const uint First = Cells[Headers + 2u * Cell];
    const uint Max   = First < Push.m_Bound ? Push.m_Bound - First : 0u;
    const uint n     = min(Walk(L, C, S, true, First, Max), Max);
    uint Head = 0u;
    if (Push.m_HeadCount > 0u && n > 0u)
    {
        const uint H = min(Push.m_HeadCount, kMaxHead);
        if (n <= H) { Head = n; for (uint k = 0u; k < n; ++k) Cells[Push.m_CellBase + uint(2 * kLevels * kLevelCells) + 2u * (First + k) + 1u] = 0u; }     // (all head: no probabilities)
        else { Split(First, n, H); Head = H; }
    }
    Cells[Headers + 2u * Cell + 1u] = n | (Head << 24);
}
