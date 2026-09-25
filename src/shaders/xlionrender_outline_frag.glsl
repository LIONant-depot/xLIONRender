#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 ViewportAndRadius;
    vec4 Color;
} uniforms;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = uniforms.Color;
}
