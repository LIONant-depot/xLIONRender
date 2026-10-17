#version 450
// Part II, G3 (documentation/gi_dynamic_master_plan.md): the words of the GI's resident data that changed (gi::gpu_trace::Refresh), as
// (index, value) pairs, written into the device-local array. One region per run of a vkCmdCopyBuffer was the first way: the Debug
// build's validation compares every region with every other (thousands a frame: seconds)
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0)          buffer data_b  { uint  GD[]; };
layout(std430, set = 0, binding = 1) readonly buffer pairs_b { uvec2 GIPairs[]; };
layout(push_constant) uniform push_t { int m_nPairs, m_Pad0, m_Pad1, m_Pad2; } Push;

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    if (i < uint(Push.m_nPairs)) GD[GIPairs[i].x] = GIPairs[i].y;
}
