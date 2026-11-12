#version 450
// The GI bricks' publisher (XLION_GI_BRICKS, B0 - xlionrender_gi_brick.glsl has the layout): a thread a cell of a slab of one level's
// window (clip_dim_v cells a side around the anchor, planes SlabZ0 .. SlabZ0 + gl_NumWorkGroups.z * 4 - 1 along z), into its toroidal
// texel. Per face, the store's records at the cell (xlionrender_vox_levels.glsl): a face's lattice is vox_layers_v finer along its own
// axis, so the 4 layers whose read lands in this cell (4k - 2 .. 4k + 1) collapse into it, each its mean x its lean at the face's own
// normal, weighted by min(samples, 32) as the read weighs corners; the validity min(the weights, 32) / 32
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(push_constant) uniform brick_push { uint TableBase; int Level; int SlabZ0; uint Flags; } push;
layout(std430, set = 0, binding = 0) readonly buffer GIClipBuffer { uvec4 GIClip[]; };
layout(std430, set = 0, binding = 1) readonly buffer GIGatherTable { uint GITable[]; };
layout(set = 2, binding = 0, r11f_g11f_b10f) uniform writeonly image3D Face0;
layout(set = 2, binding = 1, r11f_g11f_b10f) uniform writeonly image3D Face1;
layout(set = 2, binding = 2, r11f_g11f_b10f) uniform writeonly image3D Face2;
layout(set = 2, binding = 3, r11f_g11f_b10f) uniform writeonly image3D Face3;
layout(set = 2, binding = 4, r11f_g11f_b10f) uniform writeonly image3D Face4;
layout(set = 2, binding = 5, r11f_g11f_b10f) uniform writeonly image3D Face5;
layout(set = 2, binding = 6, rgba8) uniform writeonly image3D Valid03;
layout(set = 2, binding = 7, rgba8) uniform writeonly image3D Valid45;

bool g_Lean = true;        // (XLION_GI_BRICKS flag 1: the lean moments off - store and bricks alike)
#define VOX_LEAN_V g_Lean
#include "xlionrender_vox_levels.glsl"

void Store(in const uint f, in const ivec3 t, in const vec3 v)
{
    const vec4 c = vec4(v, 0.0);
    switch (f)
    {
    case 0u: imageStore(Face0, t, c); break;
    case 1u: imageStore(Face1, t, c); break;
    case 2u: imageStore(Face2, t, c); break;
    case 3u: imageStore(Face3, t, c); break;
    case 4u: imageStore(Face4, t, c); break;
    default: imageStore(Face5, t, c); break;
    }
}

void main()
{
    g_Lean = (push.Flags & 1u) == 0u;
    const ivec3 w = ivec3(gl_GlobalInvocationID) + ivec3(0, 0, push.SlabZ0);
    if (any(greaterThanEqual(w, ivec3(clip_dim_v)))) return;
    const int   L    = push.Level;
    const float Cell = TF(21u) * float(1 << L);
    const ivec3 Org  = ivec3(floor(vec3(TF(22u), TF(23u), TF(24u)) / Cell)) - ivec3(clip_dim_v / 2);       // (ClipLevel's window)
    const ivec3 k    = Org + w;
    const ivec3 t    = k & ivec3(clip_dim_v - 1);       // (clip_dim_v a power of two: the toroidal texel, negative cells too)
    vec4 V03 = vec4(0.0), V45 = vec4(0.0);
    for (uint f = 0u; f < 6u; ++f)
    {
        const int a = int(f >> 1u);
        vec3 n = vec3(0.0); n[a] = (f & 1u) != 0u ? -1.0 : 1.0;
        vec3 Num = vec3(0.0); float Den = 0.0;
        for (int j = -2; j <= 1; ++j)
        {
            ivec3 q = k; q[a] = 4 * k[a] + j;
            const uint  h = VoxHash(L, q, f), e = VoxEntryOf(h);
            const uvec4 dU = GIClip[e + 1u];
            if (uintBitsToFloat(dU.x) != VoxTagOf(h)) continue;
            const vec4 v = uintBitsToFloat(GIClip[e]);
            if (v.w <= 0.0) continue;
            const float cw = min(v.w, 32.0);
            Num += v.rgb / v.w * VoxLeanScale(v, dU.w, dU, n, a) * cw; Den += cw;
        }
        const float V = round(min(Den, 32.0) / 32.0 * 255.0) / 255.0;      // (premultiplied by the validity as stored: RGBA8)
        Store(f, t, Den > 0.0 ? Num / Den * V : vec3(0.0));
        if (f < 4u) V03[f] = V; else V45[f - 4u] = V;
    }
    imageStore(Valid03, t, V03);
    imageStore(Valid45, t, V45);
}
