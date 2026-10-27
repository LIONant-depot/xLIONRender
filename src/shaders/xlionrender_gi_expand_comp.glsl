#version 450
// Part II, G4c (documentation/gi_dynamic_master_plan.md): the world's pick as ranges (gi::world::gpu_range) made into the batch's items, as
// the CPU's pick serves its surfels (gi::world::Update): a settled instance's surfel skipped by its tier (TracePeriod: buried, far), its rays
// by its uncertainty (RequestedRays), its offsets from the update's seed (world::PassOffsets). One invocation per surfel of the slice's
// ranges; the items appended (GICount[0])
layout(local_size_x = 64) in;

struct GIItem  { ivec4 m_Ids; vec4 m_Offsets; };        // instance, surfel, level, rays; off1, off2, shift (bits), its range (bits)
struct GIRange { ivec4 m_A; ivec4 m_B; };               // instance, level, first, count; flags, history, seed, start

layout(std430, set = 0, binding = 0) readonly  buffer data_b   { uint    GD[]; };
layout(std430, set = 0, binding = 1) readonly  buffer ranges_b { GIRange GIRanges[]; };
layout(std430, set = 0, binding = 2) writeonly buffer items_b  { GIItem  GIItems[]; };
layout(std430, set = 0, binding = 3)           buffer count_b  { uint    GICount[]; };

layout(push_constant) uniform push_t { int m_RangeFirst, m_nRanges, m_nThreads, m_StartBase; vec4 m_Eye; } Push;

const uint  gi_inst_v = 49u;
const int   gi_range_urgent_v = 1, gi_range_settled_v = 2;
const uint  gi_est_v  = 24u;

float GF(uint i) { return uintBitsToFloat(GD[i]); }
vec3  GV(uint i) { return vec3(uintBitsToFloat(GD[i]), uintBitsToFloat(GD[i + 1u]), uintBitsToFloat(GD[i + 2u])); }
uint  Hash32(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

void main()
{
    const int g = int(gl_GlobalInvocationID.x);
    if (g >= Push.m_nThreads) return;
    const int Pos = Push.m_StartBase + g;
    int lo = Push.m_RangeFirst, hi = Push.m_RangeFirst + Push.m_nRanges - 1;            // the range holding Pos
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (GIRanges[mid].m_B.w <= Pos) lo = mid; else hi = mid - 1; }
    const GIRange Rg = GIRanges[lo];
    const int Inst = Rg.m_A.x, Level = Rg.m_A.y, s = Rg.m_A.z + (Pos - Rg.m_B.w);
    if (Pos - Rg.m_B.w >= Rg.m_A.w) return;
    const uint R = GD[0] + uint(Inst) * gi_inst_v, u = uint(s);
    if (GD[R + 40u] == 0u || GD[R + 1u] != uint(Level)) return;          // gone, or another level since the pick
    const uint M = GD[R];
    if (u >= GD[M + 6u + 2u * uint(Level)]) return;
    // the tiers (TracePeriod): a settled instance's surfels, not a change's
    if ((Rg.m_B.x & gi_range_settled_v) != 0 && (Rg.m_B.x & gi_range_urgent_v) == 0)
    {
        int Period = 1;
        if (GF(GD[R + 3u] + u) < 0.01) Period = 8;                       // dead_retrace_v
        else if ((GD[11] & 2u) != 0u)
        {
            const uint Sb = GD[M + 5u + 2u * uint(Level)] + 7u * u;
            const vec3 C = GV(Sb);
            const vec3 W = vec3(dot(GV(R + 16u), C) + GF(R + 19u), dot(GV(R + 20u), C) + GF(R + 23u), dot(GV(R + 24u), C) + GF(R + 27u));
            const float d = length(W - Push.m_Eye.xyz) / (GF(Sb + 6u) * max(GF(R + 15u), 1e-9));
            Period = d < 12.0 ? 1 : d < 24.0 ? 2 : d < 48.0 ? 4 : 8;      // far_tier_radii_v
        }
        if (Period > 1 && ((Rg.m_B.y + s) % Period) != 0) return;
    }
    // the rays (RequestedRays)
    const int RPS = max(int(GD[10]), 1);
    int Rays = RPS;
    if ((GD[11] & 1u) != 0u)
    {
        const uint Eo = GD[R + 42u] + gi_est_v * u;
        if (GD[Eo + 11u] >= GD[9] && (GD[Eo + 14u] & 1u) == 0u)     // (settled, not watching: word 14's bit 0)
        {
            const vec3 Mn = GV(Eo), Vr = GV(Eo + 6u);
            const float Mean = 0.299 * Mn.x + 0.587 * Mn.y + 0.114 * Mn.z;
            const float Dev  = sqrt(max(0.299 * Vr.x + 0.587 * Vr.y + 0.114 * Vr.z, 0.0));
            const float uu   = Mean > 1e-4 ? Dev / Mean : 1.0;
            Rays = clamp(int(ceil(float(RPS) * uu * GF(12u))), max(1, RPS / 4), RPS);
        }
    }
    // the offsets (PassOffsets)
    const uint h1 = Hash32(uint(Rg.m_B.z) * 0x9E3779B9u ^ uint(Inst) * 0x85EBCA6Bu ^ u * 0xC2B2AE35u), h2 = Hash32(h1), h3 = Hash32(h2);
    GIItem It;
    It.m_Ids = ivec4(Inst, s, Level, Rays);
    It.m_Offsets = vec4(float(h1 >> 8) * (1.0 / 16777216.0), float(h2 >> 8) * (1.0 / 16777216.0), intBitsToFloat(int(h3 >> 26)), intBitsToFloat(lo));
    GIItems[atomicAdd(GICount[0], 1u)] = It;
}
