#version 450
// Test material: HDR output = the second UVs the vertex shader handed over (the UV1 stream, or UV0 for a geometry
// without one) in RG, and UV0 in BA.
#include "mb_standard_pbr.frag"

layout(location = 0) out vec4 outFragColor;
layout(location = 6) in vec2 InUV1;

void main()
{
    outFragColor = vec4(InUV1, In.UV);
}
