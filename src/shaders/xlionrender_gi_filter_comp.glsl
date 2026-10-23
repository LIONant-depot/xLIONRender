#version 450
// Part II, G4 (documentation/gi_dynamic_master_plan.md): the shown light on the GPU, as gi::world::UpdateShown makes it on the CPU. One
// invocation per surfel of every instance (in the shown light's global order: the owners' table, and each instance's first in its record,
// word 46). Mode 0: pass 0, the surfel's light split by its basis fractions (E f 3). Mode 1: a filter pass, each surfel with its neighbors
// (the mesh's neighbor lists of the instance's level, weighted by their validity; none: copied). Mode 2: shown, saturated (SaturateOne),
// written into the renderer's ring for this frame (a uvec4 a surfel: the basis' 3 colors in RGB9E5, the validity's bits), with the depth moments (two halves a texel,
// rounded as the CPU's Half)
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0) readonly buffer data_b    { uint  GD[]; };
layout(std430, set = 0, binding = 1) readonly buffer owners_b  { uint  GIOwner[]; };
layout(std430, set = 0, binding = 2)          buffer scratch_b { float GIPass[]; };     // two arrays of 9 floats a surfel: [0, 9N), [9N, 18N)
layout(std430, set = 0, binding = 3) readonly buffer places_b  { int   GIPlace[]; };    // per instance its place in the ring region (-1: not drawn)
layout(std430, set = 0, binding = 4)          buffer light_b   { uvec4 GILight[]; };
layout(std430, set = 0, binding = 5)          buffer moment_b  { uint  GIMoment[]; };

layout(push_constant) uniform push_t { int m_Mode, m_N, m_Src, m_Base; float m_Saturation; int m_Pad0, m_Pad1, m_Pad2; } Push;

const uint gi_inst_v = 48u;          // gi::world::gpu_inst_v
const uint gi_none_v = 0xFFFFFFFFu;

// gi::PackRGB9E5, step for step (the same bits)
uint PackRGB9E5(vec3 c)
{
    c = vec3(c.x > 0.0 ? min(c.x, 65408.0) : 0.0, c.y > 0.0 ? min(c.y, 65408.0) : 0.0, c.z > 0.0 ? min(c.z, 65408.0) : 0.0);
    const float m = max(c.x, max(c.y, c.z));
    const int Floor = m > 0.0 ? int((floatBitsToUint(m) >> 23) & 0xFFu) - 127 : -16;
    int e = max(-16, Floor) + 16;
    float s = uintBitsToFloat(uint(127 - (e - 24)) << 23);
    if (floor(m * s + 0.5) >= 512.0) { ++e; s *= 0.5; }
    const uvec3 M = uvec3(floor(c * s + 0.5));
    return M.x | (M.y << 9) | (M.z << 18) | (uint(e) << 27);
}

float GF(uint i) { return uintBitsToFloat(GD[i]); }
vec3  GV(uint i) { return vec3(uintBitsToFloat(GD[i]), uintBitsToFloat(GD[i + 1u]), uintBitsToFloat(GD[i + 2u])); }

// details::Half: round half up, tiny to 0, overflow to inf
uint Half(float f)
{
    const uint x = floatBitsToUint(f), Sign = (x >> 16) & 0x8000u;
    const int  E = int((x >> 23) & 0xFFu) - 127 + 15;
    if (E <= 0)  return Sign;
    if (E >= 31) return Sign | 0x7C00u;
    const uint M = x & 0x7FFFFFu, H = Sign | (uint(E) << 10) | (M >> 13);
    return (H + ((M >> 12) & 1u)) & 0xFFFFu;
}

void main()
{
    // Mode 2: 16 invocations a surfel (t: its moment word), so that the stores into the host-visible rings are contiguous across invocations
    const uint t = Push.m_Mode == 2 ? gl_GlobalInvocationID.x % 16u : 0u;
    const uint g = Push.m_Mode == 2 ? gl_GlobalInvocationID.x / 16u : gl_GlobalInvocationID.x;
    if (g >= uint(Push.m_N)) return;
    const uint Inst = GIOwner[g], R = GD[0] + Inst * gi_inst_v;
    const uint s = g - GD[R + 46u], Level = GD[R + 1u], M = GD[R];
    const uint Src = uint(Push.m_Src) * 9u * uint(Push.m_N), Dst = (1u - uint(Push.m_Src)) * 9u * uint(Push.m_N);
    if (Push.m_Mode == 0)
    {
        const vec3 E = GV(GD[R + 2u] + 3u * s);
        for (uint i = 0u; i < 3u; ++i)
        {
            const vec3 f = GV(GD[R + 41u] + 9u * s + 3u * i), P = E * f * 3.0;
            GIPass[Dst + 9u * g + 3u * i] = P.x; GIPass[Dst + 9u * g + 3u * i + 1u] = P.y; GIPass[Dst + 9u * g + 3u * i + 2u] = P.z;
        }
        return;
    }
    if (Push.m_Mode == 1)
    {
        float a[9]; float W = 1.0;
        for (uint q = 0u; q < 9u; ++q) a[q] = GIPass[Src + 9u * g + q];
        const uint Nf = GD[M + 18u + 3u * Level];
        if (Nf != gi_none_v)
        {
            const uint Ni = GD[M + 19u + 3u * Level], Nw = GD[M + 20u + 3u * Level], First = GD[R + 46u];
            for (uint k = GD[Nf + s], k1 = GD[Nf + s + 1u]; k < k1; ++k)
            {
                const uint j = GD[Ni + k];
                const float w = GF(Nw + k) * GF(GD[R + 3u] + j);
                const uint gj = First + j;
                for (uint q = 0u; q < 9u; ++q) a[q] += GIPass[Src + 9u * gj + q] * w;
                W += w;
            }
        }
        const float r = 1.0 / W;
        for (uint q = 0u; q < 9u; ++q) GIPass[Dst + 9u * g + q] = a[q] * r;
        return;
    }
    // shown: saturated, into the ring (its region for this frame: m_Base surfels in), with the moments
    const int Place = GIPlace[Inst];
    if (Place < 0) return;
    const uint At = uint(Push.m_Base + Place + 1) + s;
    const uint Mo = GD[R + 4u] + 32u * s;
    GIMoment[16u * At + t] = Half(GF(Mo + 2u * t)) | (Half(GF(Mo + 2u * t + 1u)) << 16);
    if (t != 0u) return;
    const float Valid = GF(GD[R + 3u] + s);
    uint P[3];
    for (uint i = 0u; i < 3u; ++i)
    {
        vec3 C = vec3(GIPass[Src + 9u * g + 3u * i], GIPass[Src + 9u * g + 3u * i + 1u], GIPass[Src + 9u * g + 3u * i + 2u]);
        if (Push.m_Saturation != 1.0)
        {
            const float Y = 0.2126 * C.x + 0.7152 * C.y + 0.0722 * C.z;
            C = max(vec3(Y) + (C - vec3(Y)) * Push.m_Saturation, vec3(0.0));
        }
        P[i] = PackRGB9E5(C);
    }
    GILight[At] = uvec4(P[0], P[1], P[2], floatBitsToUint(Valid));
}
