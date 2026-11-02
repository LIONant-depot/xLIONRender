#version 450
// The sun's contact shadow (XLION_CONTACT, Tomas's queue, 2026-11-02): what its shadow pages are too coarse for, a short march toward it through the G-buffer prepass's depth. One value a pixel of the view (Div: 1 or 2),
// read by SunShadowMasked (xlionrender_shadow_vsm.glsl, slot 12). The jitter by the pixel's place, not the frame: still without a TAA.
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, set = 0, binding = 0) readonly buffer table_b { uint GITable[]; };      // the gather's table: clip -> camera-centred world (0-15)
layout(set = 1, binding = 0) uniform sampler2D Depth;
layout(set = 2, binding = 0, r32f) writeonly uniform image2D Dst;
layout(push_constant) uniform push_t { mat4 m_W2C; vec4 m_ToSun; vec4 m_Params; uvec4 m_Size; } Push;   // ToSun: w the fill value, or < 0 a march; Params: length (world), steps, thickness (world), the distance it fades out by; Size: the view's, Div, the table's base

mat4 C2W(void)
{
    mat4 M;
    for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r) M[c][r] = uintBitsToFloat(GITable[Push.m_Size.w + uint(c * 4 + r)]);
    return M;
}
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 Size = ivec2(Push.m_Size.xy), S = Size / int(Push.m_Size.z);
    if (any(greaterThanEqual(p, S))) return;
    if (Push.m_ToSun.w >= 0.0) { imageStore(Dst, p, vec4(Push.m_ToSun.w)); return; }
    const ivec2 Q = min(p * int(Push.m_Size.z), Size - 1);
    const float z = texelFetch(Depth, Q, 0).r;
    float Lit = 1.0;
    if (z < 1.0)
    {
        const mat4 M = C2W();
        const vec4 Wh = M * vec4((vec2(Q) + 0.5) / vec2(Size) * 2.0 - 1.0, z, 1.0);
        const vec3 P = Wh.xyz / Wh.w;
        const float Dist = length(P);
        const float Fade = 1.0 - smoothstep(0.5 * Push.m_Params.w, Push.m_Params.w, Dist);
        if (Fade > 0.0)
        {
            const int   Steps = int(Push.m_Params.y);
            const float Jit = fract(52.9829189 * fract(dot(vec2(p), vec2(0.06711056, 0.00583715))));     // (interleaved gradient noise)
            const vec3  Start = P + Push.m_ToSun.xyz * (0.004 + 0.002 * Dist);                             // (off its own surface, more with the depth's error)
            for (int k = 0; k < Steps; ++k)
            {
                const float t = (float(k) + Jit) / float(Steps);
                const vec3  R = Start + Push.m_ToSun.xyz * (t * t * Push.m_Params.x);                     // (denser near the contact)
                const vec4  C = Push.m_W2C * vec4(R, 1.0);
                if (C.w <= 0.0) break;
                const vec3  n = C.xyz / C.w;
                const vec2  uv = n.xy * 0.5 + 0.5;
                if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) break;
                const float zs = texelFetch(Depth, ivec2(uv * vec2(Size)), 0).r;
                if (zs < n.z && zs < 1.0)
                {
                    // Behind what the depth shows, and not far behind: a thin caster's far side isn't known: assumed thickness
                    const vec4 Ws = M * vec4(n.xy, zs, 1.0);
                    if (length(R) - length(Ws.xyz / Ws.w) < Push.m_Params.z) { Lit = 1.0 - Fade; break; }
                }
            }
        }
    }
    imageStore(Dst, p, vec4(Lit));
}
