#version 450
// Experiment X1 (xLION_Docs gi_next_level_analysis.md, XLION_GI_CLIP): the GI read through a camera-centred world clipmap. Levels of
// 64^3 cells, each cell's size twice the last's; a cell holds 6 faces (+x -x +y -y +z -z), a face a uvec4 of 24.8-style fixed point: the sum
// of weight x light (RGB) and of the weights. Mode 0: a range of surfels splat into the target, in their global order (the shown light,
// gpu_trace::Show) - an adder a cell of its disc, atomics; mode 1: a range of the target cleared. The surfels' own kernel (Wendland over 2
// radii); a surfel writes only the faces its normal looks toward; bSight: a ray per cell from the surfel keeps it from crossing a wall
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 5) readonly buffer owners_b { uint  GIOwner[]; };
layout(std430, set = 0, binding = 6) readonly buffer shown_b  { float GIShown[]; };
layout(std430, set = 0, binding = 7)          buffer clip_b   { uvec4 GIClip[]; };
layout(push_constant) uniform push_t { vec4 m_EyeCell; int m_First, m_Count, m_Mode, m_Sight, m_TlasRoot, m_PrimOffset, m_Pad0, m_Pad1; } Push;
#include "xlionrender_gi_pass_common.glsl"

const int   clip_dim_v = 64, clip_levels_v = 4;
const uint  gi_inst_clip_v = 50u;
const float clip_light_scale_v = 8192.0, clip_weight_scale_v = 8192.0;

float CF(uint i) { return uintBitsToFloat(GD[i]); }
vec3  CV(uint i) { return vec3(CF(i), CF(i + 1u), CF(i + 2u)); }

void main()
{
    if (Push.m_Mode == 1)
    {
        const uint i = uint(Push.m_First) + gl_GlobalInvocationID.x;
        if (int(gl_GlobalInvocationID.x) < Push.m_Count) GIClip[i] = uvec4(0u);
        return;
    }
    if (int(gl_GlobalInvocationID.x) >= Push.m_Count) return;
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset;
    const uint g    = uint(Push.m_First) + gl_GlobalInvocationID.x;
    const uint Inst = GIOwner[g], R = GD[0] + Inst * gi_inst_clip_v;
    const uint s    = g - GD[R + 46u], Level = GD[R + 1u], M = GD[R];
    const uint b    = GD[M + 5u + 2u * Level] + 7u * s;
    const mat3x4 L2W = mat3x4(vec4(CF(R + 16u), CF(R + 17u), CF(R + 18u), CF(R + 19u)), vec4(CF(R + 20u), CF(R + 21u), CF(R + 22u), CF(R + 23u)), vec4(CF(R + 24u), CF(R + 25u), CF(R + 26u), CF(R + 27u)));
    const vec3  Cm = CV(b), Nm = CV(b + 3u);
    const float Valid = CF(GD[R + 3u] + s);
    const vec3  Ctr = vec4(Cm, 1.0) * L2W;
    const vec3  Nw  = normalize(vec4(Nm, 0.0) * L2W);
    const float Scale = length(vec3(L2W[0].x, L2W[1].x, L2W[2].x));
    const float Reach = 2.0 * CF(b + 6u) * Scale;
    const vec3  T = normalize(cross(abs(Nm.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), Nm)), B = cross(Nm, T);
    const float a = 0.40824829, bb = 0.70710678, c = 0.57735027, d = 0.81649658;
    const vec3  D0 = T * -a + B * bb + Nm * c, D1 = T * -a - B * bb + Nm * c, D2 = T * d + Nm * c;
    const vec3  L0 = vec3(GIShown[9u * g], GIShown[9u * g + 1u], GIShown[9u * g + 2u]), L1 = vec3(GIShown[9u * g + 3u], GIShown[9u * g + 4u], GIShown[9u * g + 5u]), L2 = vec3(GIShown[9u * g + 6u], GIShown[9u * g + 7u], GIShown[9u * g + 8u]);
    // the faces it writes, and what each reads of its 3 lobes (GIBasisRead)
    uint  Face[3]; float FW[3]; vec3 FL[3];
    const mat3 Rot = transpose(mat3(L2W[0].xyz, L2W[1].xyz, L2W[2].xyz));      // (mesh -> world's rotation: its rows; a face in mesh space by its transpose)
    for (int k = 0; k < 3; ++k)
    {
        const bool  bNeg = Nw[k] < 0.0;
        Face[k] = uint(2 * k + (bNeg ? 1 : 0));
        FW[k]   = Nw[k] * Nw[k];
        vec3 dw = vec3(0.0); dw[k] = bNeg ? -1.0 : 1.0;
        const vec3 dm = normalize(dw * Rot);
        const vec3 w  = pow(max(vec3(dot(dm, D0), dot(dm, D1), dot(dm, D2)), vec3(0.0)), vec3(2.0));
        const float sw = w.x + w.y + w.z;
        const vec3 wn = sw > 1e-12 ? w / sw : vec3(1.0 / 3.0);
        FL[k] = L0 * wn.x + L1 * wn.y + L2 * wn.z;
    }
    // a level: its cells on the disc - over the 2 axes the normal looks least along, around the plane's height on the third
    int Major = abs(Nw.x) >= abs(Nw.y) && abs(Nw.x) >= abs(Nw.z) ? 0 : abs(Nw.y) >= abs(Nw.z) ? 1 : 2;
    const int U = (Major + 1) % 3, V = (Major + 2) % 3;
    for (int L = 0; L < clip_levels_v; ++L)
    {
        const float Cell = Push.m_EyeCell.w * float(1 << L);
        if (Cell * 8.0 < Reach && L + 1 < clip_levels_v) continue;       // (too many cells for what a surfel can tell: coarser levels hold it)
        const ivec3 Org = ivec3(floor(Push.m_EyeCell.xyz / Cell)) - ivec3(clip_dim_v / 2);
        const vec3  g0  = (Ctr - vec3(Reach)) / Cell - vec3(Org) - 0.5, g1 = (Ctr + vec3(Reach)) / Cell - vec3(Org) - 0.5;
        if (any(greaterThanEqual(g0, vec3(clip_dim_v))) || any(lessThan(g1, vec3(0.0)))) continue;
        const int u0 = max(int(ceil(g0[U])), 0), u1 = min(int(floor(g1[U])), clip_dim_v - 1);
        const int v0 = max(int(ceil(g0[V])), 0), v1 = min(int(floor(g1[V])), clip_dim_v - 1);
        for (int v = v0; v <= v1; ++v) for (int u = u0; u <= u1; ++u)
        {
            vec3 Pc; Pc[U] = (float(Org[U] + u) + 0.5) * Cell; Pc[V] = (float(Org[V] + v) + 0.5) * Cell; Pc[Major] = 0.0;
            // the plane's height at (u, v), 2 cells around it (a trilinear read sees the 2 layers each side of the surface)
            Pc[Major] = Ctr[Major] - (dot(Pc - Ctr, Nw) - (Pc[Major] - Ctr[Major]) * Nw[Major]) / Nw[Major];
            const int m0 = int(floor(Pc[Major] / Cell - float(Org[Major]) - 0.5));
            for (int m = max(m0 - 1, 0); m <= min(m0 + 2, clip_dim_v - 1); ++m)
            {
                ivec3 q; q[U] = u; q[V] = v; q[Major] = m;
                const vec3  Pq  = (vec3(Org + q) + 0.5) * Cell;
                const float f   = 1.0 - length(Pq - Ctr) / Reach;
                if (f <= 0.0) continue;
                const float w0  = f * f * f * f * (5.0 - 4.0 * f) * Valid;
                if (w0 < 1e-3) continue;
                if (Push.m_Sight != 0)
                {
                    const vec3  Ps = Pq - Nw * dot(Pq - Ctr, Nw);       // (the cell's point on the surfel's plane)
                    const vec3  Dr = Ps - Ctr;
                    const float Dl = length(Dr);
                    if (Dl > 1e-3 && Occluded(Ctr + Nw * 0.02, Dr * (1.0 / Dl), Dl)) continue;
                }
                const uint Base = (uint(L) * uint(clip_dim_v * clip_dim_v * clip_dim_v) + uint(q.x) + uint(q.y) * uint(clip_dim_v) + uint(q.z) * uint(clip_dim_v * clip_dim_v)) * 6u;
                for (int k = 0; k < 3; ++k)
                {
                    if (FW[k] < 0.02) continue;
                    const float w = w0 * FW[k];
                    const uvec3 Lq = uvec3(max(FL[k], vec3(0.0)) * w * clip_light_scale_v);
                    const uint  e  = Base + Face[k];
                    atomicAdd(GIClip[e].x, Lq.x); atomicAdd(GIClip[e].y, Lq.y); atomicAdd(GIClip[e].z, Lq.z); atomicAdd(GIClip[e].w, uint(w * clip_weight_scale_v));
                }
            }
        }
    }
}
