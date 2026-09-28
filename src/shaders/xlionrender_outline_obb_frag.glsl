#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 ViewportAndRadius; // xy viewport px; z radius px; w max fractional expansion
    vec4 Color;
    vec4 Bounds;            // xyz local box center; w cage growth in local units
    vec4 Axis;              // xyz local box half-extents; w near-parallel rejection angle (degrees)
} uniforms;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = uniforms.Color;
}
