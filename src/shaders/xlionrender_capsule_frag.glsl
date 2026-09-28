#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 Color;
    vec4 RadiusHeightScale;
} uniforms;

layout(location = 0) in vec3 inNormal;
layout(location = 0) out vec4 outColor;

// Same shading as xlionrender_primitive_frag.glsl - object-space normal, no world-space light (see
// that shader's own comment on why).
void main()
{
    vec3  N     = normalize(inNormal);
    float Shade = 0.4 + 0.6 * abs(N.y);
    outColor    = vec4(uniforms.Color.rgb * Shade, uniforms.Color.a);
}
