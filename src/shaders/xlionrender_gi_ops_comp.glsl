#version 450
// Part II, G4 (documentation/gi_dynamic_master_plan.md): the world's changes to the estimators (gi::world::gpu_op: estimator::Lower at a
// dirty instance's pass and in ApplyAffect), on the GPU's copy of them, one surfel an invocation (gi::gpu_trace makes an instance's op
// one per surfel). Lower is the same in any order: the confidence goes down to the least of the ops' (atomicMin), the rest is the same
// value written by each
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0)          buffer data_b { uint  GD[]; };
layout(std430, set = 0, binding = 1) readonly buffer ops_b  { ivec4 GIOps[]; };     // instance, surfel, a (float bits), the moment counts again

layout(push_constant) uniform push_t { int m_nOps, m_Pad0, m_Pad1, m_Pad2; } Push;

const uint gi_inst_v = 47u;          // gi::world::gpu_inst_v
const uint gi_est_v  = 21u;

void main()
{
    const int i = int(gl_GlobalInvocationID.x);
    if (i >= Push.m_nOps) return;
    const ivec4 Op = GIOps[i];
    const uint Rec = GD[0] + uint(Op.x) * gi_inst_v, u = uint(Op.y);
    if (GD[Rec + 40u] == 0u) return;
    const float a = clamp(intBitsToFloat(Op.z), 0.0, 1.0);
    const uint  k = uint(max(1, int((1.0 - a) * float(GD[9]))));
    const uint Eo = GD[Rec + 42u] + gi_est_v * u;
    const uint Old = atomicMin(GD[Eo + 11u], k);
    if (Old > 0u) { GD[Eo + 14u] = 1u; GD[Eo + 12u] = 0u; GD[Eo + 18u] = GD[Eo]; GD[Eo + 19u] = GD[Eo + 1u]; GD[Eo + 20u] = GD[Eo + 2u]; }
    if (Op.w != 0) { const uint Mc = GD[Rec + 43u] + 16u * u; for (uint t = 0u; t < 16u; ++t) GD[Mc + t] = 0u; }
}
