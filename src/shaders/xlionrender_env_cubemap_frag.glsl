#version 450
// A cubemap texture (scene::sky::m_pCubemap) into the environment source's level 0: an octahedral map with folded borders. The
// pyramid and the prefilter then run as for the procedural sky.
#include "xlionrender_environment_atlas.glsl"

layout(binding = 0) uniform samplerCube SamplerCube;

layout(push_constant) uniform env_cubemap
{
    vec4 Params;        // x: the source's size
} P;

layout(location = 0) out vec4 outColor;

void main()
{
    const float Size  = P.Params.x;
    const float Inner = Size - 2.0 * kEnvBorder;
    const vec2  UV    = (gl_FragCoord.xy - kEnvBorder) / Inner * 2.0 - 1.0;
    // the source is 256^2: a 1K face would alias at level 0, a lower mip of the cube matches the octahedral texel instead
    const float Lod   = max(log2(float(textureSize(SamplerCube, 0).x) * 4.0 / Inner), 0.0);
    outColor = vec4(textureLod(SamplerCube, OctDecode(OctFold(UV)), Lod).rgb, 1.0);
}
