#version 450
// Test material: HDR output = the vertex color the vertex shader handed over (the COLOR0 stream decoded from sRGB to linear,
// or the constant white of a geometry without one).
#include "mb_standard_pbr.frag"

layout(location = 0) out vec4 outFragColor;

void main()
{
    outFragColor = vec4(In.VertColor.rgb, 1.0);
}
