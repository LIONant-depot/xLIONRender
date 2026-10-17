#version 450
// Part II, G4c (documentation/gi_dynamic_master_plan.md): per instance the sums of its valid surfels' light (gi::world::UpdateMeans: the
// unsupported reads' value) and its bad surfels (world::BadSurfel: not finite or negative), for the world (SetGPUMeans). One workgroup an
// instance
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0) readonly  buffer data_b  { uint  GD[]; };
layout(std430, set = 0, binding = 1) writeonly buffer means_b { float GIMeans[]; };     // per instance: sum E V 3, sum V, bad

const uint gi_inst_v = 48u;

shared vec4  s_Sum[64];
shared float s_Bad[64];

void main()
{
    const uint i = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    const uint R = GD[0] + i * gi_inst_v;
    vec4 S = vec4(0); float Bad = 0.0;
    if (GD[R + 40u] != 0u)
    {
        const uint M = GD[R], n = GD[M + 6u + 2u * GD[R + 1u]];
        for (uint s = t; s < n; s += 64u)
        {
            const uint Ea = GD[R + 2u] + 3u * s;
            const vec3 E = vec3(uintBitsToFloat(GD[Ea]), uintBitsToFloat(GD[Ea + 1u]), uintBitsToFloat(GD[Ea + 2u]));
            const float V = uintBitsToFloat(GD[GD[R + 3u] + s]);
            S += vec4(E * V, V);
            const float a = E.x + E.y + E.z + V;
            Bad += (isnan(a) || isinf(a) || E.x < 0.0 || E.y < 0.0 || E.z < 0.0) ? 1.0 : 0.0;
        }
    }
    s_Sum[t] = S; s_Bad[t] = Bad;
    barrier();
    for (uint k = 32u; k > 0u; k >>= 1) { if (t < k) { s_Sum[t] += s_Sum[t + k]; s_Bad[t] += s_Bad[t + k]; } barrier(); }
    if (t == 0u) { GIMeans[5u * i] = s_Sum[0].x; GIMeans[5u * i + 1u] = s_Sum[0].y; GIMeans[5u * i + 2u] = s_Sum[0].z; GIMeans[5u * i + 3u] = s_Sum[0].w; GIMeans[5u * i + 4u] = s_Bad[0]; }
}
