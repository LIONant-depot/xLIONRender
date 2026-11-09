// A custom post node's frame (render graph stage I): included right after the node GLSL's #version 450 line, which then defines one function -
//      vec4 Node(ivec2 p, vec2 uv)     the output texel at pixel p (uv: its centre in 0..1)
// Src is the chain image (the node's input), ExposureTex the adapted exposure (1 x 1, .r); the result is the next chain image (alpha:
// coverage, keep Src's). The bindings and the push are post_comp's, so the renderer runs it as it runs the built-in post passes.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 1, binding = 0) uniform sampler2D Src;
layout(set = 1, binding = 1) uniform sampler2D ExposureTex;
layout(set = 2, binding = 0, rgba16f) uniform image2D Dst;
layout(push_constant) uniform push_t { int m_W, m_H, m_Mode; float m_A, m_B, m_C, m_D, m_E; } Push;     // (W, H: the size; the rest: the node's parameters, later)

vec4 Node(ivec2 p, vec2 uv);

void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Push.m_W || p.y >= Push.m_H) return;
    imageStore(Dst, p, Node(p, (vec2(p) + 0.5) / vec2(Push.m_W, Push.m_H)));
}
