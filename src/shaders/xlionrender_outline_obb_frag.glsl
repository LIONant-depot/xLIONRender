#version 450
#extension GL_ARB_separate_shader_objects : enable

// Must match the OBB vertex stage's push-constant block (128 bytes std140) - this codebase
// requires every stage that touches the range to declare the identical struct.
layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 ViewportAndRadius;
    vec4 Color;
    vec4 Bounds;
    vec4 Axis;
} uniforms;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = uniforms.Color;
}