#version 450
// Test material in the xmaterial layout (slot 0 shadow map, then textures; lighting UBO at set 2 binding 1):
// HDR output = texture * 3, so a value above 1.0 proves the texture binding and the scene-linear output.
#include "mb_standard_pbr.frag"

layout(binding = 1) uniform sampler2D SamplerAlbedo;

layout(location = 0) out vec4 outFragColor;

void main()
{
    outFragColor = vec4(texture(SamplerAlbedo, In.UV).rgb * 3.0, 1.0);
}
