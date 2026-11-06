#version 450
// XLION_NOISE_LAGS (measurement, Tomas 2026-11-06: noise by how far apart in time): the tone-mapped image's display luminance at every 4th pixel each
// way, four to a uint, into this frame's slot of a CPU-readable ring; the CPU compares frames at lags 1 to 60 at the real frame rate (the editor's
// screenshots spaced frames 123 ms apart)
layout(local_size_x = 64) in;
layout(set = 1, binding = 0) uniform sampler2D Image;
layout(std430, set = 0, binding = 0) buffer ring_b { uint Ring[]; };
layout(push_constant) uniform push_t { uint m_SubW, m_SubH, m_Base, m_Pad; } Push;

void main()
{
    const uint w = gl_GlobalInvocationID.x, nWords = (Push.m_SubW * Push.m_SubH + 3u) / 4u;
    if (w >= nWords) return;
    uint Packed = 0u;
    for (uint k = 0u; k < 4u; ++k)
    {
        const uint i = w * 4u + k;
        if (i >= Push.m_SubW * Push.m_SubH) break;
        const ivec2 p = ivec2(int(i % Push.m_SubW) * 4, int(i / Push.m_SubW) * 4);
        const vec3  c = texelFetch(Image, p, 0).rgb;
        const uint  L = uint(clamp(dot(c, vec3(0.2126, 0.7152, 0.0722)) * 255.0 + 0.5, 0.0, 255.0));
        Packed |= L << (8u * k);
    }
    Ring[Push.m_Base + w] = Packed;
}
