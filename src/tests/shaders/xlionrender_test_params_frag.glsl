#version 450
// Test material reading its parameter block as the material compiler declares it (xmaterial.plugin Compiler/xmaterial_compiler.cpp
// ParamTop): HDR output = the block's second vec4, so the value proves binding 9 and the push constant's MaterialBase.
#include "mb_standard_pbr.frag"

layout(std430, set = 1, binding = 9) readonly buffer MaterialParamsBuf { vec4 P[]; } g_Mat;
layout(push_constant) uniform XMatPush { layout(offset = 16) uint MaterialBase; } g_XMatPush;

layout(location = 0) out vec4 outFragColor;

void main()
{
    outFragColor = vec4(g_Mat.P[g_XMatPush.MaterialBase + 1u].rgb, 1.0);
}
