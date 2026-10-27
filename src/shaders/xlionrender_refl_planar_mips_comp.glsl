#version 450
// Reflections, rough planar receivers (documentation/reflections_strategy.md sec. 8): the planar view's pyramid - each level the 2 x 2
// of the one above, their colours weighed by their coverage (alpha: something drawn) so the empty texels do not darken it, the
// coverage averaged. Level 1 from the planar view's HDR, the next from the buffer (instance::renderPlanar; read by
// xlionrender_refl_trace_comp.glsl). A texel: two uints, the halves of rgb and the coverage
layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, set = 0, binding = 0) buffer mips_b { uvec2 Mips[]; };
layout(set = 1, binding = 0) uniform sampler2D Planar;

layout(push_constant) uniform push_t { int m_SrcW, m_SrcH, m_DstW, m_DstH; uint m_SrcBase, m_DstBase, m_bFromTexture, m_Pad; } Push;

vec4 Src(ivec2 q)
{
    q = min(q, ivec2(Push.m_SrcW - 1, Push.m_SrcH - 1));
    if (Push.m_bFromTexture != 0u) return texelFetch(Planar, q, 0);
    const uvec2 m = Mips[Push.m_SrcBase + uint(q.y * Push.m_SrcW + q.x)];
    return vec4(unpackHalf2x16(m.x), unpackHalf2x16(m.y));
}

void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Push.m_DstW || p.y >= Push.m_DstH) return;
    vec3 C = vec3(0.0); float A = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        const vec4 s = Src(2 * p + ivec2(k & 1, k >> 1));
        const float a = clamp(s.a, 0.0, 1.0);
        C += s.rgb * a; A += a;
    }
    const vec3 Col = A > 1e-4 ? C / A : vec3(0.0);
    Mips[Push.m_DstBase + uint(p.y * Push.m_DstW + p.x)] = uvec2(packHalf2x16(Col.rg), packHalf2x16(vec2(Col.b, A * 0.25)));
}
