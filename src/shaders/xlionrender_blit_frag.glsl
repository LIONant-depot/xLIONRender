#version 450
// A view's output (already display encoded) into the current pass as is - no color conversion. ImGui::Image is not used for
// views: the xGPU ImGui shader treats textures as linear and gamma encodes them again.
layout(binding = 0) uniform sampler2D SamplerOutput;

layout(location = 0) in  vec2 in_UV;
layout(location = 0) out vec4 outColor;

void main()
{
    outColor = texture(SamplerOutput, in_UV);
}
