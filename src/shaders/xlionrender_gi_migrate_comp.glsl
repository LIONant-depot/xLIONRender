#version 450
// Part II, G4c (documentation/gi_dynamic_master_plan.md): the light from the scene's last layout into its new one when it is packed again
// (gi::gpu_trace::Pack): the light lives only on the GPU. One workgroup a job. Kind 0: words copied (an instance's array at the same level).
// Kind 1: an instance at another level (gi::world::Update's level switch): its light and validity carried over (a child its parent's; a
// parent its children's mean), the rest new as world::ResetMoments makes it (the moments fully open, the fractions a third each, the
// estimators seeded with the light, the moment counts none)
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0) readonly buffer old_b  { uint GO[]; };
layout(std430, set = 0, binding = 1)          buffer new_b  { uint GD[]; };
layout(std430, set = 0, binding = 2) readonly buffer jobs_b { ivec4 GIJobs[]; };       // kind 0: old word, new word, count; kind 1: old record, new record

const uint gi_est_v = 24u;

vec3 OV(uint i) { return vec3(uintBitsToFloat(GO[i]), uintBitsToFloat(GO[i + 1u]), uintBitsToFloat(GO[i + 2u])); }
void PV(uint i, vec3 v) { GD[i] = floatBitsToUint(v.x); GD[i + 1u] = floatBitsToUint(v.y); GD[i + 2u] = floatBitsToUint(v.z); }

void main()
{
    const ivec4 J = GIJobs[gl_WorkGroupID.x];
    const uint  t = gl_LocalInvocationID.x;
    if (J.x == 0)
    {
        for (uint w = t; w < uint(J.w); w += 64u) GD[uint(J.z) + w] = GO[uint(J.y) + w];
        return;
    }
    const uint Ro = uint(J.y), Rn = uint(J.z);
    const uint M = GD[Rn], Lv = GD[Rn + 1u], n = GD[M + 6u + 2u * Lv];
    for (uint s = t; s < n; s += 64u)
    {
        vec3 E = vec3(0); float V = 0.0;
        if (Lv == 0u)       // a child: its parent's
        {
            const uint p = GD[GD[M + 9u] + s];
            E = OV(GO[Ro + 2u] + 3u * p); V = uintBitsToFloat(GO[GO[Ro + 3u] + p]);
        }
        else                // a parent: its children's mean
        {
            uint c = 0u;
            for (uint k = GD[GD[M + 24u] + s]; k < GD[GD[M + 24u] + s + 1u]; ++k)
            {
                const uint ch = GD[GD[M + 25u] + k];
                E += OV(GO[Ro + 2u] + 3u * ch); V += uintBitsToFloat(GO[GO[Ro + 3u] + ch]); ++c;
            }
            if (c > 0u) { E *= 1.0 / float(c); V /= float(c); }
        }
        PV(GD[Rn + 2u] + 3u * s, E);
        GD[GD[Rn + 3u] + s] = floatBitsToUint(V);
        const float Rr = 2.0 * uintBitsToFloat(GD[GD[M + 5u + 2u * Lv] + 7u * s + 6u]);
        for (uint q = 0u; q < 16u; ++q) { GD[GD[Rn + 4u] + 32u * s + 2u * q] = floatBitsToUint(Rr); GD[GD[Rn + 4u] + 32u * s + 2u * q + 1u] = floatBitsToUint(Rr * Rr); GD[GD[Rn + 43u] + 16u * s + q] = 0u; }
        for (uint q = 0u; q < 9u; ++q) GD[GD[Rn + 41u] + 9u * s + q] = floatBitsToUint(1.0 / 3.0);
        const uint Eo = GD[Rn + 42u] + gi_est_v * s;          // estimator::Seed of a new one
        PV(Eo, E); PV(Eo + 3u, E); PV(Eo + 6u, vec3(0)); GD[Eo + 9u] = floatBitsToUint(1.0); GD[Eo + 10u] = 0u;
        GD[Eo + 11u] = GD[9]; GD[Eo + 12u] = 0u; GD[Eo + 13u] = 0u; GD[Eo + 14u] = 0u; PV(Eo + 15u, vec3(0)); PV(Eo + 18u, vec3(0)); GD[Eo + 21u] = 0u; GD[Eo + 22u] = 0u; GD[Eo + 23u] = 0u;
    }
}
