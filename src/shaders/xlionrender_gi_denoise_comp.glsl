#version 450
// The voxels' GI read cleaned on the screen (XLION_GI_DENOISE, xLION_Docs gi_resolve_plan_2026-11.md). A world cache's error is a voxel's: every pixel near a noisy corner shares it, so a small
// filter averages the same corners and changes nothing: the blobs are a cell across. So the strides follow the cell's size on the screen, to reach other corners. Same surface only (the plane, the normal).
// The light, not its AO: the AO is a contact's and stays sharp. Pixels in, pixels out; the header isn't touched.
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, set = 0, binding = 0) readonly buffer src_b   { uvec4 Src[]; };
layout(std430, set = 0, binding = 1)          buffer dst_b   { uvec4 Dst[]; };
layout(std430, set = 0, binding = 2) readonly buffer table_b { uint  GITable[]; };      // the gather's table: clip -> camera-centred world (0-15)
layout(std430, set = 0, binding = 3) readonly buffer ao_b    { vec2  AOData[]; };       // the half size AO (AO, distance)
layout(push_constant) uniform push_t { int m_W, m_H, m_TableBase, m_bAO; float m_CellPixels, m_MaxStride, m_Mult, m_Pad; } Push;   // m_CellPixels: a level 0 cell's pixels at distance 1
const uint screen_header_v = 8u;

vec3 ScreenNormal(in const uint w)
{
    const vec2 o = vec2(float((w >> 16) & 63u), float((w >> 22) & 63u)) / 63.0 * 2.0 - 1.0;
    vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Push.m_W || p.y >= Push.m_H) return;
    const uint  i = uint(p.y) * uint(Push.m_W) + uint(p.x);
    const uvec4 Q = Src[screen_header_v + i];
    Dst[screen_header_v + i] = Q;
    if ((Q.w & (1u << 30)) == 0u) return;
    uint t = uint(Push.m_TableBase);
    const mat4 C2W = mat4(uintBitsToFloat(uvec4(GITable[t + 0u], GITable[t + 1u], GITable[t + 2u], GITable[t + 3u])), uintBitsToFloat(uvec4(GITable[t + 4u], GITable[t + 5u], GITable[t + 6u], GITable[t + 7u])),
                          uintBitsToFloat(uvec4(GITable[t + 8u], GITable[t + 9u], GITable[t + 10u], GITable[t + 11u])), uintBitsToFloat(uvec4(GITable[t + 12u], GITable[t + 13u], GITable[t + 14u], GITable[t + 15u])));
    const vec2  Size = vec2(float(Push.m_W), float(Push.m_H));
    const vec2  Dy = unpackHalf2x16(Q.y);
    const float d = Dy.y;
    const vec3  n = ScreenNormal(Q.w);
    const vec4  Wp = C2W * vec4((vec2(p) + 0.5) / Size * 2.0 - 1.0, 0.5, 1.0);
    const vec3  Pp = normalize(Wp.xyz / Wp.w) * d;
    const float Ap = Push.m_bAO != 0 ? max(AOData[i].x, 0.05) : 1.0;
    vec3  Sum = vec3(unpackHalf2x16(Q.x), Dy.x) / Ap;
    float Wt = 1.0;
    const int s = max(1, int(min(Push.m_CellPixels / max(d, 0.1), Push.m_MaxStride) * Push.m_Mult + 0.5));
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x)
    {
        if (x == 0 && y == 0) continue;
        const ivec2 q = p + ivec2(x, y) * s;
        if (q.x < 0 || q.y < 0 || q.x >= Push.m_W || q.y >= Push.m_H) continue;
        const uint  j = uint(q.y) * uint(Push.m_W) + uint(q.x);
        const uvec4 R = Src[screen_header_v + j];
        if ((R.w & (1u << 30)) == 0u || ((R.w >> 28) & 3u) == 3u) continue;
        const float wn = pow(max(dot(ScreenNormal(R.w), n), 0.0), 32.0);
        if (wn < 0.01) continue;
        const vec2  Ry = unpackHalf2x16(R.y);
        const vec4  Wq = C2W * vec4((vec2(q) + 0.5) / Size * 2.0 - 1.0, 0.5, 1.0);
        const vec3  Pq = normalize(Wq.xyz / Wq.w) * Ry.y;
        const float wp = max(0.0, 1.0 - abs(dot(Pq - Pp, n)) / (0.01 + 0.005 * d));
        const float w = wn * wp;
        if (w <= 0.0) continue;
        const float Aq = Push.m_bAO != 0 ? max(AOData[j].x, 0.05) : 1.0;
        Sum += vec3(unpackHalf2x16(R.x), Ry.x) / Aq * w;
        Wt  += w;
    }
    const vec3 E = Sum / Wt * Ap;
    Dst[screen_header_v + i] = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, d)), Q.z, Q.w);
}
