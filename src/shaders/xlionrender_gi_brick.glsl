// The GI bricks (XLION_GI_BRICKS, xLION_Docs gi_brick_textures_plan_2026-11-12.md v2): a store level's window (clip_dim_v cells a side around
// the anchor) published as 3D textures, read with the hardware's trilinear filter. One volume a face (+X -X +Y -Y +Z -Z), B10G11R11: the
// face's mean light premultiplied by its validity (0: no record); the validities in two RGBA8 volumes (faces 0-3, 4-5) - the filtered
// value over the filtered validity is the store's corner blend, each corner by min(samples, 32) (xlionrender_vox_levels.glsl ClipLevel).
// TOROIDAL: a cell's texel is its absolute index mod clip_dim_v, the sampler wraps (REPEAT) - world neighbours are texture neighbours, so
// the filter is right across the seam and the window needs no origin to be read: u = P / (Cell * clip_dim_v).
// The includer defines vec4 BrickTex(int L, uint i, vec3 u): level L's volume i (0-5 the faces, 6 the validities of faces 0-3, 7 of 4-5)
// sampled trilinear with a wrapping sampler at u, and includes xlionrender_vox_levels.glsl first
#ifndef XLIONRENDER_GI_BRICK_GLSL
#define XLIONRENDER_GI_BRICK_GLSL
const float brick_valid_min_v = 1e-3 / 32.0;       // (ClipLevel's Den > 1e-3 in samples: validity is samples / 32)
float g_BrickV = 0.0;       // (BrickRead: the faces' filtered validity, by their weights - min(samples, 32) / 32: the level's trust)
// The ambient cube at a world point for a normal: 3 filtered samples (a face per axis by the normal's sign, weights n^2), as ClipLevel:
// .w the weights of the faces that had light (0: none)
vec4 BrickRead(in const int L, in const vec3 Pw, in const vec3 n)
{
    const vec3 u = Pw / (TF(21u) * float(1 << L) * float(clip_dim_v));
    vec3 Sum = vec3(0.0); float SumW = 0.0, SumV = 0.0;
    for (int a = 0; a < 3; ++a)
    {
        const float wa = n[a] * n[a];
        if (wa < 0.02) continue;
        const uint  f = uint(2 * a + (n[a] < 0.0 ? 1 : 0));
        const float V = BrickTex(L, 6u + f / 4u, u)[f & 3u];
        if (V > brick_valid_min_v) { Sum += BrickTex(L, f, u).rgb / V * wa; SumW += wa; SumV += V * wa; }
    }
    g_BrickV = SumW > 0.0 ? SumV / SumW : 0.0;
    return SumW > 0.0 ? vec4(Sum / SumW, SumW) : vec4(0.0);
}
#endif
