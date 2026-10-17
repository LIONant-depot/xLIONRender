#version 450
// Part II, G4c (documentation/gi_dynamic_master_plan.md): an ambient cube's parents blended from the 6 it traces (gi::world::ExpandCube), on
// the GPU's light. One workgroup a cube instance (its entry: instance, set size, the set's surfels)
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0)          buffer data_b  { uint GD[]; };
layout(std430, set = 0, binding = 1) readonly buffer cubes_b { int  GICubes[]; };      // per cube 8: instance, count, up to 6 surfels

const uint gi_inst_v = 48u;

float GF(uint i) { return uintBitsToFloat(GD[i]); }
vec3  GV(uint i) { return vec3(uintBitsToFloat(GD[i]), uintBitsToFloat(GD[i + 1u]), uintBitsToFloat(GD[i + 2u])); }
void  PV(uint i, vec3 v) { GD[i] = floatBitsToUint(v.x); GD[i + 1u] = floatBitsToUint(v.y); GD[i + 2u] = floatBitsToUint(v.z); }

void main()
{
    const uint c = gl_WorkGroupID.x;
    const uint Inst = uint(GICubes[8u * c]), nSet = uint(GICubes[8u * c + 1u]);
    const uint R = GD[0] + Inst * gi_inst_v;
    if (GD[R + 40u] == 0u || GD[R + 1u] != 1u || nSet == 0u) return;
    const uint M = GD[R], n = GD[M + 8u], L = GD[M + 7u];         // level 1: the parents
    for (uint p = gl_LocalInvocationID.x; p < n; p += 64u)
    {
        bool bIn = false;
        for (uint k = 0u; k < nSet; ++k) bIn = bIn || uint(GICubes[8u * c + 2u + k]) == p;
        if (bIn) continue;
        const vec3 Np = GV(L + 7u * p + 3u);
        vec3 E = vec3(0), F0 = vec3(0), F1 = vec3(0), F2 = vec3(0); float W = 0.0;
        int Best = GICubes[8u * c + 2u]; float BestD = -2.0;
        for (uint k = 0u; k < nSet; ++k)
        {
            const uint q = uint(GICubes[8u * c + 2u + k]);
            const float d = dot(Np, GV(L + 7u * q + 3u));
            if (d > BestD) { BestD = d; Best = int(q); }
            const float cc = max(d, 0.0), w = cc * cc * (GF(GD[R + 3u] + q) + 1e-3);
            if (w <= 0.0) continue;
            E += GV(GD[R + 2u] + 3u * q) * w; W += w;
            F0 += GV(GD[R + 41u] + 9u * q) * w; F1 += GV(GD[R + 41u] + 9u * q + 3u) * w; F2 += GV(GD[R + 41u] + 9u * q + 6u) * w;
        }
        if (W <= 0.0)       // facing none of them: the nearest in direction
        {
            const uint b = uint(Best);
            PV(GD[R + 2u] + 3u * p, GV(GD[R + 2u] + 3u * b));
            for (uint i = 0u; i < 3u; ++i) PV(GD[R + 41u] + 9u * p + 3u * i, GV(GD[R + 41u] + 9u * b + 3u * i));
            continue;
        }
        const float r = 1.0 / W;
        PV(GD[R + 2u] + 3u * p, E * r);
        PV(GD[R + 41u] + 9u * p, F0 * r); PV(GD[R + 41u] + 9u * p + 3u, F1 * r); PV(GD[R + 41u] + 9u * p + 6u, F2 * r);
    }
}
