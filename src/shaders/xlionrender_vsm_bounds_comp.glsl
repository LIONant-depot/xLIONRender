#version 450
// The sun's pages' depth bounds (xLION_Docs shadows_gpu_plan.md S1b): a workgroup a page rendered this frame reduces its 128 x 128 depth
// texels to their least and greatest (16-bit, as stored) into its slot's entry - the sun mask's early-out reads them
// (xlionrender_sun_mask_frag.glsl VSMProvenLit). A slot keeps its entry while its page stays (a page renders again when it changes)
layout(local_size_x = 16, local_size_y = 16) in;
const int kPageTexels = 128;
layout(std430, set = 0, binding = 0) readonly buffer list_b   { ivec2 Pages[]; };       // the pages rendered: their slot's atlas origin
layout(std430, set = 0, binding = 1)          buffer bounds_b { uint  Bounds[]; };      // per slot: least depth (low 16 bits), greatest (high)
layout(set = 1, binding = 0) uniform sampler2D Atlas;
layout(push_constant) uniform push_t { uint m_Base, m_Count, m_Row, m_Pad; } Push;    // m_Row: slots per row of the pool

shared uint s_Min[256], s_Max[256];

void main()
{
    const ivec2 O = Pages[Push.m_Base + gl_WorkGroupID.x];
    const ivec2 T = O + ivec2(gl_LocalInvocationID.xy) * 8;
    uint m = 0xFFFFu, M = 0u;
    for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
    {
        const uint u = uint(round(texelFetch(Atlas, T + ivec2(x, y), 0).r * 65535.0));
        m = min(m, u); M = max(M, u);
    }
    const uint i = gl_LocalInvocationIndex;
    s_Min[i] = m; s_Max[i] = M;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u)
    {
        if (i < s) { s_Min[i] = min(s_Min[i], s_Min[i + s]); s_Max[i] = max(s_Max[i], s_Max[i + s]); }
        barrier();
    }
    if (i == 0u) Bounds[uint(O.y / kPageTexels) * Push.m_Row + uint(O.x / kPageTexels)] = s_Min[0] | (s_Max[0] << 16u);
}
