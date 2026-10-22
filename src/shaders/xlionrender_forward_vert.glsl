#version 450
// Forward plan vertex shader: xlionrender::vertex -> the mb_* VaryingFull interface (locations 0-5).
#include "mb_varying_definition_full.glsl"

layout(location = 0) in vec3 in_Pos;
layout(location = 1) in vec2 in_UV;
layout(location = 2) in vec4 in_Color;
layout(location = 3) in vec3 in_Normal;
layout(location = 4) in vec4 in_Tangent;    // xyz = tangent, w = binormal sign

layout(push_constant) uniform Push
{
    mat4 L2W;
    mat4 W2C;
} push;

layout(location = 0) out VaryingFull Out;
layout(location = 6) out vec2 OutUV1;           // no second UVs on this path: UV0
layout(location = 9) out vec4 OutGI;            // surfel GI (mb_standard_pbr.frag): none on this path

void main()
{
    const vec4 wPos    = push.L2W * vec4(in_Pos, 1.0);
    Out.wSpacePosition = wPos;
    gl_Position        = push.W2C * wPos;

    const mat3 RotMat  = mat3(push.L2W);    // ponytail: no inverse-transpose, fine for uniform scale
    Out.Normal         = normalize(RotMat * in_Normal);
    Out.Tangent.xyz    = normalize(RotMat * in_Tangent.xyz);
    Out.Tangent.w      = in_Tangent.w;
    Out.VertColor      = in_Color;
    OutGI              = vec4(0.0);
    Out.UV             = in_UV;
    OutUV1             = in_UV;

    // No shadows yet: z outside (-1,1) makes ShadowPCF return 1 without sampling
    Out.ShadowPosition = vec4(0.5, 0.5, 2.0, 1.0);
}
