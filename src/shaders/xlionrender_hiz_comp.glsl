#version 450
// GPU cluster culling's Hi-Z pyramid (features/cull/xlionrender_gpu_cull.h): one level from the one below, a texel the farthest depth (depth:
// 0 near, 1 far) of the 2 x 2 under it - a last odd row or column has one. Level 0 is the scene's depth (window::CopyTextureToBuffer); a
// level's size is the one below's halved, rounded up, so a texel of level L covers the pixels [x 2^L, (x + 1) 2^L)
layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, set = 0, binding = 0) buffer hiz_b { float HiZ[]; };

layout(push_constant) uniform push_t { uint m_Src, m_SrcW, m_SrcH, m_Dst, m_DstW, m_DstH, m_Pad0, m_Pad1; } Push;

void main()
{
    const uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= Push.m_DstW || p.y >= Push.m_DstH) return;
    const uvec2 s0 = p * 2u, s1 = min(s0 + 1u, uvec2(Push.m_SrcW, Push.m_SrcH) - 1u);
    const float a = HiZ[Push.m_Src + s0.y * Push.m_SrcW + s0.x], b = HiZ[Push.m_Src + s0.y * Push.m_SrcW + s1.x];
    const float c = HiZ[Push.m_Src + s1.y * Push.m_SrcW + s0.x], d = HiZ[Push.m_Src + s1.y * Push.m_SrcW + s1.x];
    HiZ[Push.m_Dst + p.y * Push.m_DstW + p.x] = max(max(a, b), max(c, d));
}
