#version 450
#extension GL_EXT_control_flow_attributes : require
// M0's memory lab (render_leap_plan_2026-11-10.md section 2b, xlionrender_test_mem_lab.cpp): K dependent pointer chains an invocation
// over a permutation that is one cycle (Sattolo), each chain started at its own point of the cycle; m_Dense 1: the next index is the
// current + 1 (the same loads, no dependency on the data: the address pattern without the chase). The result consumed: Out[invocation]
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) readonly buffer next_b { uint Next[]; };
layout(std430, set = 0, binding = 1) writeonly buffer out_b { uint Out[]; };
layout(push_constant) uniform push_t { uint m_N, m_Steps, m_K, m_Dense; } Push;

void main()
{
    const uint g = gl_GlobalInvocationID.x, Total = gl_NumWorkGroups.x * 64u;
    uint Idx[8];
    [[unroll]] for (uint c = 0u; c < 8u; ++c) Idx[c] = uint(float(g * Push.m_K + c) / float(max(Total * Push.m_K, 1u)) * float(Push.m_N)) % Push.m_N;
    uint Acc = 0u;
    for (uint s = 0u; s < Push.m_Steps; ++s)
    {
        // (the K chains advanced together: their loads are independent of each other, each depends on its own last)
        [[unroll]] for (uint c = 0u; c < 8u; ++c)      // (unrolled: Idx[] stays in registers - a dynamic index put it in local memory and serialized the chains)
        {
            if (c >= Push.m_K) continue;
            const uint v = Next[Idx[c]];
            Idx[c] = Push.m_Dense != 0u ? (Idx[c] + 1u + (v & 0u)) % Push.m_N : v;
            Acc += v;
        }
    }
    Out[g] = Acc + Idx[0];
}
