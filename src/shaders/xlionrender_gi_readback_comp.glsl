#version 450
// Part II, G4c (documentation/gi_dynamic_master_plan.md): the GI's resident data copied into a host-visible buffer (gi::gpu_trace::ReadBack:
// the world's arrays from the GPU's light, for its tests, its dump, or the CPU taking over)
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0) readonly  buffer data_b { uint GD[]; };
layout(std430, set = 0, binding = 1) writeonly buffer out_b  { uint GOut[]; };

layout(push_constant) uniform push_t { int m_nWords, m_Pad0, m_Pad1, m_Pad2; } Push;

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    if (i < uint(Push.m_nWords)) GOut[i] = GD[i];
}
