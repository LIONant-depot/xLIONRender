#version 450
// Part II, G4 (documentation/gi_dynamic_master_plan.md): the commit on the GPU. After a slice's pass (xlionrender_gi_surfel_pass_comp.glsl)
// each of its items' sums become the surfel's light, as gi::world does it on the CPU: FinishPass (the links' emitted light, E = pi mean(L),
// the validity from the back faces) and Update's commit (estimator::Add, the validity's blend, the basis fractions, the depth moments).
// The surfel's whole state is written back into the resident data (G4c: the GPU's only), what its pass found added to its range's stats
// (gi::world::gpu_range_stats: atomics)
layout(local_size_x = 64) in;

struct GIItem   { ivec4 m_Ids; vec4 m_Offsets; };
struct GIResult { float m_V[72]; };

layout(std430, set = 0, binding = 0)           buffer data_b    { uint     GD[]; };
layout(std430, set = 0, binding = 1) readonly  buffer items_b   { GIItem   GIItems[]; };
layout(std430, set = 0, binding = 2) readonly  buffer results_b { GIResult GIResults[]; };
layout(std430, set = 0, binding = 3)           buffer stats_b   { int      GIStats[]; };     // per range 8: rays, traced, unsupported, hits, at 3, at's flag
layout(std430, set = 0, binding = 4) readonly  buffer count_b   { uint     GICount[]; };

layout(push_constant) uniform push_t { int m_nItems, m_First, m_Count, m_Pad; } Push;

const uint  gi_inst_v  = 49u;        // gi::world::gpu_inst_v
const uint  gi_none_v  = 0xFFFFFFFFu;
const uint  gi_est_v   = 24u;        // gi::world::gpu_est_v (PackEst)
const float gi_pi_v    = 3.14159265;

float GF(uint i) { return uintBitsToFloat(GD[i]); }
vec3  GV(uint i) { return vec3(uintBitsToFloat(GD[i]), uintBitsToFloat(GD[i + 1u]), uintBitsToFloat(GD[i + 2u])); }
void  PF(uint i, float f) { GD[i] = floatBitsToUint(f); }
void  PV(uint i, vec3 v)  { GD[i] = floatBitsToUint(v.x); GD[i + 1u] = floatBitsToUint(v.y); GD[i + 2u] = floatBitsToUint(v.z); }
float Luma(vec3 c) { return 0.299 * c.x + 0.587 * c.y + 0.114 * c.z; }

void main()
{
    const int j = int(gl_GlobalInvocationID.x);
    if (j >= min(Push.m_Count, int(GICount[0]))) return;
    const int  k = Push.m_First + j;
    const int  Id = GIItems[k].m_Ids.x, Level = GIItems[k].m_Ids.z, R = GIItems[k].m_Ids.w;
    const uint u = uint(GIItems[k].m_Ids.y);
    const uint Rec = GD[0] + uint(Id) * gi_inst_v;
    if (GD[Rec + 40u] == 0u || GD[Rec + 1u] != uint(Level)) return;     // gone, or another level since the pick: the world drops it
    float V[72];
    for (int i = 0; i < 72; ++i) V[i] = GIResults[k].m_V[i];
    if ((GD[11] & 8u) != 0u)        // measurement only (setup::m_bAblateFreeze): the range's stats, the light left as it is
    {
        const int Rg0 = floatBitsToInt(GIItems[k].m_Offsets.w);
        atomicAdd(GIStats[8 * Rg0], R); atomicAdd(GIStats[8 * Rg0 + 1], 1);
        atomicAdd(GIStats[8 * Rg0 + 2], floatBitsToInt(V[62])); atomicAdd(GIStats[8 * Rg0 + 3], floatBitsToInt(V[63]));
        if ((GD[11] & 16u) != 0u)   // (setup::m_bAblateStats, a diagnostic run: the pass's work counters in the range's words 4..7)
        {
            atomicAdd(GIStats[8 * Rg0 + 4], floatBitsToInt(V[70])); atomicAdd(GIStats[8 * Rg0 + 5], floatBitsToInt(V[71]));
            atomicAdd(GIStats[8 * Rg0 + 6], floatBitsToInt(V[67])); atomicAdd(GIStats[8 * Rg0 + 7], floatBitsToInt(V[68]));
        }
        return;
    }

    // FinishPass: the emitters it sees (the links), E = pi mean(L), the validity from the back faces
    vec3 Sum = vec3(V[0], V[1], V[2]);
    vec3 BA[3] = vec3[3](vec3(V[3], V[4], V[5]), vec3(V[6], V[7], V[8]), vec3(V[9], V[10], V[11]));
    const uint Eb = GD[Rec + 44u] + 9u * u;
    if ((GD[11] & 64u) == 0u)       // (setup::m_bEmitterDirectExternal: lighting shades the emitters' direct light - not the links')
        for (int i = 0; i < 3; ++i) { const vec3 e = GV(Eb + 3u * uint(i)) * (float(R) / (3.0 * gi_pi_v)); Sum += e; BA[i] += e; }
    const vec3  Acc  = Sum * (gi_pi_v / float(R));
    const float Near = clamp(1.0 - (float(floatBitsToInt(V[60])) / float(R) - 0.2) / 0.3, 0.0, 1.0);
    const float All  = clamp(1.0 - (float(floatBitsToInt(V[61])) / float(R) - 0.8) / 0.15, 0.0, 1.0);
    const float AccValid = min(Near, All);

    // the estimator (estimator::Add, ShortBlend 0.08), its confidence before it
    const uint Eo = GD[Rec + 42u] + gi_est_v * u;
    vec3 Mean = GV(Eo), Short = GV(Eo + 3u), Var = GV(Eo + 6u);
    float Vbbr = GF(Eo + 9u), Inc = GF(Eo + 10u);
    uint  N = GD[Eo + 11u]; int Streak = int(GD[Eo + 12u]); uint nTaken = GD[Eo + 13u]; bool bWatch = (GD[Eo + 14u] & 1u) != 0u, bTook = (GD[Eo + 14u] & 2u) != 0u;
    vec3  Last = GV(Eo + 15u), Ref = GV(Eo + 18u);
    float PrevL = GF(Eo + 21u), DiffSq = GF(Eo + 22u); uint nDiff = GD[Eo + 23u];
    const float MinBlend = GF(8u);
    const uint  Warm     = GD[9];
    const float Conf = max(1.0 / float(N + 1u), MinBlend);
    vec3 y = Acc;
    if (MinBlend >= 1.0) { Mean = y; Short = y; Var = vec3(0); Vbbr = 1.0; Inc = 0.0; N = Warm; }          // Seed
    else
    {
        bool bDone = false;
        if (bWatch)
        {
            const float Ly = Luma(y), D = Ly - Luma(Ref);          // (estimator::Add: beyond half the light and 3 times the samples' noise)
            const float Noise = nDiff >= 3u ? sqrt(DiffSq / (2.0 * float(nDiff))) : 0.0;
            const int Sign = abs(D) > max(0.5 * max(Luma(Ref), 1e-3), 3.0 * Noise) ? (D > 0.0 ? 1 : -1) : 0;
            if (Sign == 0 && PrevL >= 0.0) { DiffSq += (Ly - PrevL) * (Ly - PrevL); ++nDiff; }
            PrevL = Ly;
            if (Sign != 0 && Sign == Streak && abs(Luma(y) - Luma(Last)) < 0.5 * abs(D) && (!bTook || N >= 4u)) { Mean = (Last + y) * 0.5; Short = Mean; Var = vec3(0); N = 2u; Streak = 0; Last = y; ++nTaken; bTook = true; Ref = Mean; bDone = true; }     // (estimator::Add: they agree; the next differs from the mean)
            else { Streak = Sign; Last = y; }
        }
        if (!bDone && N < Warm)
        {
            ++N;
            const float a = max(1.0 / float(N), MinBlend);
            Mean = Mean + (y - Mean) * a;
            Short = Mean; Var = vec3(0); Vbbr = 1.0; Inc = 0.0;
            if (bTook) Ref = Mean;
            if (N >= Warm) bWatch = false;
            bDone = true;
        }
        if (!bDone)
        {
            const vec3 Dev0 = sqrt(max(vec3(1e-5), Var));
            y = min(y, Short + Dev0 * 8.0 + vec3(0.1));
            const vec3 Delta = y - Short;
            Short = Short + (y - Short) * 0.08;
            const vec3 Delta2 = y - Short;
            Var = Var + (Delta * Delta2 - Var) * (0.08 * 0.5);
            const vec3 Dv = sqrt(max(vec3(1e-5), Var)), Diff = Mean - Short;
            const float Relative = Luma(abs(Diff) / Dv);
            Inc += (Relative - Inc) * 0.08;
            const float VbbrT = clamp(Luma(Short / Dv) * 0.5, 1.0 / 32.0, 1.0);
            const float x = clamp(Relative * max(0.02, Inc - 0.2), 0.0, 1.0);
            const float CatchUp = clamp(x * x * (3.0 - 2.0 * x), 1.0 / 256.0, 1.0) * Vbbr;
            Vbbr += (VbbrT - Vbbr) * 0.1;
            if (N < 256u) ++N;                                      // (gi::running_cap_v: the running mean under the blend)
            Mean = Mean + (y - Mean) * clamp(max(CatchUp, 1.0 / float(N)), 0.0, 1.0);
        }
    }
    PV(Eo, Mean); PV(Eo + 3u, Short); PV(Eo + 6u, Var); PF(Eo + 9u, Vbbr); PF(Eo + 10u, Inc);
    GD[Eo + 11u] = N; GD[Eo + 12u] = uint(Streak); GD[Eo + 13u] = nTaken; GD[Eo + 14u] = (bWatch ? 1u : 0u) | (bTook ? 2u : 0u); PV(Eo + 15u, Last); PV(Eo + 18u, Ref); PF(Eo + 21u, PrevL); PF(Eo + 22u, DiffSq); GD[Eo + 23u] = nDiff;

    // the light, the validity (on a transparent material: none), the fractions per basis direction, the moments
    PV(GD[Rec + 2u] + 3u * u, Mean);
    const uint Va = GD[Rec + 3u] + u;
    const float av = max(Conf, 0.05);
    float Valid = GF(Va) * (1.0 - av) + AccValid * av;
    const uint St = GD[Rec + 45u];
    if (Level == 0 && (GD[Rec + 47u] & 16u) != 0u && ((GD[St + u / 32u] >> (u % 32u)) & 1u) != 0u) Valid = 0.0;
    const vec3 Tot = BA[0] + BA[1] + BA[2];
    const float af = max(Conf, 0.1);
    const uint Fo = GD[Rec + 41u] + 9u * u;
    for (int i = 0; i < 3; ++i)
    {
        const vec3 F = vec3(Tot.x > 1e-9 ? BA[i].x / Tot.x : 1.0 / 3.0, Tot.y > 1e-9 ? BA[i].y / Tot.y : 1.0 / 3.0, Tot.z > 1e-9 ? BA[i].z / Tot.z : 1.0 / 3.0);
        const vec3 Fr = GV(Fo + 3u * uint(i));
        PV(Fo + 3u * uint(i), Fr + (F - Fr) * af);
    }
    const uint Mo = GD[Rec + 4u] + 32u * u, Mc = GD[Rec + 43u] + 16u * u;
    for (uint t = 0u; t < 16u; ++t)
    {
        const float c = V[12u + 3u * t + 2u];
        if (c == 0.0) continue;
        uint n = GD[Mc + t];
        if (n < 0xFFFFu) ++n;
        GD[Mc + t] = n;
        const float a = max(1.0 / float(n), 0.03);
        PF(Mo + 2u * t,      GF(Mo + 2u * t)      + (V[12u + 3u * t] / c - GF(Mo + 2u * t)) * a);
        PF(Mo + 2u * t + 1u, GF(Mo + 2u * t + 1u) + (V[12u + 3u * t + 1u] / c - GF(Mo + 2u * t + 1u)) * a);
    }
    if (Valid < 1e-6) Valid = 0.0;
    PF(Va, Valid);

    // its range's stats
    const int Rg = floatBitsToInt(GIItems[k].m_Offsets.w);
    atomicAdd(GIStats[8 * Rg], R); atomicAdd(GIStats[8 * Rg + 1], 1);
    const int nUns = floatBitsToInt(V[62]);
    atomicAdd(GIStats[8 * Rg + 2], nUns); atomicAdd(GIStats[8 * Rg + 3], floatBitsToInt(V[63]));
    if (nUns > 0 && atomicCompSwap(GIStats[8 * Rg + 7], 0, 1) == 0)
    {
        GIStats[8 * Rg + 4] = floatBitsToInt(V[64]); GIStats[8 * Rg + 5] = floatBitsToInt(V[65]); GIStats[8 * Rg + 6] = floatBitsToInt(V[66]);
    }
}
