#version 450
// Forward plan, standard material for cluster geometry (xgeom_static): the constant metal/roughness parameters come from the
// draw's instance record (features/instances; written when they change), not from a per-draw uniform entry, so draws of the
// same mesh need no binding between them. Writes scene-linear HDR.
#include "mb_standard_pbr.frag"

struct InstanceData
{
    vec4   Rows[3];        // local -> the instance buffer's origin (the vertex shader's)
    vec4   Material[3];    // [0] base color rgb, metallic; [1] emissive rgb, roughness; [2] x AO, y F0
};
layout(std430, set = 1, binding = 1) readonly buffer InstanceBuffer
{
    InstanceData Instances[];
} instances;

layout(location = 8) flat in uint InInstance;       // the vertex shader's push.instanceIndex
layout(location = 0) out vec4 outColor;

void main()
{
    const InstanceData I = instances.Instances[InInstance];
    const vec3 C = PBRLighting
    ( vec3(0.0, 0.0, 1.0)           // tangent-space normal (no normal map)
    , I.Material[0].rgb
    , I.Material[2].x
    , I.Material[2].y
    , I.Material[1].w
    , I.Material[0].w
    , I.Material[1].rgb
    );
    outColor = vec4(C, 1.0);
}
