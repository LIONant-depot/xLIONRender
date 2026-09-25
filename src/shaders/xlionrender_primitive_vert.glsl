#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

// Both stages declare the identical block - this codebase's own push-constant convention requires
// it (every stage that touches the range needs the same struct or nothing rasterizes).
layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 Color;
} uniforms;

layout(location = 0) out vec3 outNormal;

void main()
{
    gl_Position = uniforms.L2C * vec4(inPosition, 1.0);
    outNormal   = inNormal;
}
