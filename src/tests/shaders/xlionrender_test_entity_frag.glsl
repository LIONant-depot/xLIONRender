#version 450
// Test material with one entity-overridable parameter, declared as the material compiler declares it (xmaterial.plugin
// Compiler/xmaterial_compiler.cpp ParamTop/ReadExpr): HDR output = Tint, the block's first vec4 unless the draw's EntityParams
// entry (set 1 binding 10, indexed by the instance slot) has its Tint bit set.
#include "mb_standard_pbr.frag"

layout(std430, set = 1, binding = 9) readonly buffer MaterialParamsBuf { vec4 P[]; } g_Mat;
layout(push_constant) uniform XMatPush { layout(offset = 16) uint MaterialBase; } g_XMatPush;
struct XMatEntity { uvec4 H; vec4 Tint; vec4 P0; vec4 P1; };
layout(std430, set = 1, binding = 10) readonly buffer EntityParamsBuf { XMatEntity E[]; } g_Ent;
layout(location = 8) flat in uint InInstance;
#define XMAT_ENT(Bit, Field, Swz, Mat) ((g_Ent.E[InInstance].H.x & Bit) != 0u ? g_Ent.E[InInstance].Field.Swz : (Mat))

layout(location = 0) out vec4 outFragColor;

void main()
{
    outFragColor = vec4(XMAT_ENT(1u, Tint, xyz, g_Mat.P[g_XMatPush.MaterialBase + 0u].xyz), 1.0);
}
