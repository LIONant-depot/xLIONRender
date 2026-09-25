#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 Color;
} uniforms;

layout(location = 0) in vec3 inNormal;
layout(location = 0) out vec4 outColor;

// Deliberately simple: shades by the OBJECT-space normal, not a world-space light - a correct
// world-space light would need L2W in the push-constant block too, which (with L2C and Color already
// in it) would exceed Vulkan's guaranteed-minimum 128-byte push-constant budget. The apparent light
// direction rotates with each object as it tumbles - an accepted simplification, not a bug.
void main()
{
    vec3  N     = normalize(inNormal);
    float Shade = 0.4 + 0.6 * abs(N.y);
    outColor    = vec4(uniforms.Color.rgb * Shade, uniforms.Color.a);
}
