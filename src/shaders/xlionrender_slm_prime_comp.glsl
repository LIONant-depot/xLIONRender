#version 450
// (C30 measurement, XLION_SLM_PRIME) a dispatch whose threads need ~6 KB of local memory each - more than any material shader's spill
// (~2.7 KB) - at the start of every frame: the driver's local memory is then sized for the largest need before any pass runs
layout(local_size_x = 64) in;
layout(push_constant) uniform push_t { uint m_N, m_Seed, m_Pad0, m_Pad1; } Push;
layout(std430, binding = 0) buffer out_t { uint m_V[]; } Out;
void main()
{
    uint A[1536];
    for (uint i = 0u; i < 1536u; ++i) A[i] = i * Push.m_Seed + gl_LocalInvocationID.x;
    uint S = 0u;
    for (uint i = 0u; i < Push.m_N; ++i) S += A[(i * 7u + gl_LocalInvocationID.x * 13u + Push.m_Seed) % 1536u];
    if (S == 0xFFFFFFFFu) Out.m_V[0] = S;      // (never: keeps the work)
}
