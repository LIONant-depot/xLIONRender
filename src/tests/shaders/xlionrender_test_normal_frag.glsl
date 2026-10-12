#version 450
// Test material: the world normal as a color, so a comparison checks positions and normals, not lighting.
#include "mb_varying_definition_full.glsl"

layout(location = 0) in VaryingFull In;
layout(location = 0) out vec4 outFragColor;

void main()
{
    outFragColor = vec4(normalize(In.Normal) * 0.5 + 0.5, 1.0);
}
