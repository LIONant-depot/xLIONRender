#version 450
// A skinned mover into every mover page its box reaches in one instanced draw a cluster (plans/forward: drawSkinPages; Bistro Game 2026-11-06:
// 10 characters under a low sun were ~1000 mover pages a frame drawn one page and one cluster at a time on the CPU, 41000 draws, ~20 ms).
// gl_InstanceIndex (the draw's first instance + i) is the entry of the page list: its page; the world position as xlionrender_skin_geom_vert.glsl
// computes it, through the page's world -> clip, into the page's square of the whole shadow atlas (one viewport over all of it: the page's
// viewport transform folded into the position, as xlionrender_shadow_pairs_vert.glsl), clipped to the page by 4 clip distances
#include "xgeom_skin_mb_input_full.vert"

layout(set=2, binding = 0) uniform MeshUniforms
{
    mat4   w2C;            // (unused here: the page's)
    vec4   WorldOffset;    // xyz: the instance buffer's origin in that world
} mesh;

struct InstanceData
{
    vec4   Rows[3];
    vec4   Material[3];
    vec4   PrevRows[3];
};
layout(std430, set = 1, binding = 2) readonly buffer InstanceBuffer
{
    InstanceData Instances[];
} instances;

struct PageData
{
    mat4   W2C;            // world -> the page's clip space
    vec4   Atlas;          // x, y: the page's scale in the atlas's clip space; z, w: its offset
    vec4   Rect;           // (unused here)
};
layout(std430, set = 1, binding = 3) readonly buffer PageBuffer   { PageData Pages[]; };
layout(std430, set = 1, binding = 4) readonly buffer PageListBuffer { uint PageList[]; };

invariant gl_Position;
out float gl_ClipDistance[4];

void main()
{
    const mb_full_vertex VertData = getVertexData();
    const InstanceData   I        = instances.Instances[push.instanceIndex];
    const mat3x4         L2wRows  = mat3x4(I.Rows[0], I.Rows[1], I.Rows[2]);
    const vec4           World    = vec4(VertData.LocalPos * L2wRows + mesh.WorldOffset.xyz, 1.0);
    const PageData       P        = Pages[PageList[gl_InstanceIndex]];
    const vec4           C        = P.W2C * World;
    gl_ClipDistance[0] = C.w + C.x;
    gl_ClipDistance[1] = C.w - C.x;
    gl_ClipDistance[2] = C.w + C.y;
    gl_ClipDistance[3] = C.w - C.y;
    gl_Position = vec4(C.x * P.Atlas.x + C.w * P.Atlas.z, C.y * P.Atlas.y + C.w * P.Atlas.w, C.z, C.w);
}
