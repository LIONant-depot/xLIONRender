#version 450
// The sun's page/caster pairs (features/shadows/xlionrender_shadow_pairs.h): a rigid cluster into one of this frame's pages, its record
// by gl_InstanceIndex (xlionrender_shadow_pairs_comp.glsl). The world position as the one-page path computes it (xlionrender_static_geom_body.glsl),
// through the page's world -> clip, then into the page's square of the sun's atlas (one viewport over all of it: the page's viewport
// transform folded into the position; the sun is orthographic, w = 1), clipped to the page by 4 clip distances
#define XLION_SHADOW_RECORDS
#include "xgeom_static_mb_input_full.vert"

layout(set=2, binding = 0) uniform MeshUniforms
{
    mat4   w2C;            // (unused here: the page's)
    vec4   WorldOffset;    // xyz: the instance buffer's origin in that world
} mesh;

struct InstanceData
{
    vec4   Rows[3];
    vec4   Material[3];
};
layout(std430, set = 1, binding = 1) readonly buffer InstanceBuffer
{
    InstanceData Instances[];
} instances;

struct PageData
{
    mat4   W2C;            // world -> the page's clip space
    vec4   Atlas;          // x, y: the page's scale in the atlas's clip space; z, w: its offset
    vec4   Rect;           // (the pairs' test)
};
layout(std430, set = 1, binding = 3) readonly buffer PageBuffer
{
    PageData Pages[];
};

invariant gl_Position;
out float gl_ClipDistance[4];

void main()
{
    const mb_full_vertex VertData = getVertexData();
    const InstanceData   I        = instances.Instances[push.instanceIndex];
    const mat3x4         L2wRows  = mat3x4(I.Rows[0], I.Rows[1], I.Rows[2]);
    const vec4           World    = vec4(VertData.LocalPos * L2wRows + mesh.WorldOffset.xyz, 1.0);
    const PageData       P        = Pages[push.page];
    const vec4           C        = P.W2C * World;
    gl_ClipDistance[0] = C.w + C.x;
    gl_ClipDistance[1] = C.w - C.x;
    gl_ClipDistance[2] = C.w + C.y;
    gl_ClipDistance[3] = C.w - C.y;
    gl_Position = vec4(C.x * P.Atlas.x + C.w * P.Atlas.z, C.y * P.Atlas.y + C.w * P.Atlas.w, C.z, C.w);
}
