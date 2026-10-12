#version 450
// One level of the environment source's pyramid (features/environment) from the level above: each texel averages 4 bilinear
// samples of the parent around its direction (a 4x4 box over the parent's texels), folded borders like every octahedral map
// here. The prefilter reads the pyramid for filtered importance sampling (Krivanek & Colbert: a sample reads the level whose
// texels match its solid angle - few samples, no aliasing).
#include "xlionrender_environment_atlas.glsl"

layout(binding = 0) uniform sampler2D SamplerParent;

layout(push_constant) uniform downsample
{
    vec4 Params;        // x: this level's size, y: the parent's size
} P;

layout(location = 0) out vec4 outColor;

vec3 Parent(in const vec2 UV)
{
    const float Size  = P.Params.y;
    const float Inner = Size - 2.0 * kEnvBorder;
    return textureLod(SamplerParent, (kEnvBorder + (OctEncode(OctDecode(OctFold(UV))) * 0.5 + 0.5) * Inner) / Size, 0.0).rgb;
}

void main()
{
    const float Size  = P.Params.x;
    const float Inner = Size - 2.0 * kEnvBorder;
    const vec2  UV    = (gl_FragCoord.xy - kEnvBorder) / Inner * 2.0 - 1.0;
    const float q     = 0.5 / Inner;                    // a quarter of this level's texel, in [-1, 1] units
    outColor = vec4(0.25 * (Parent(UV + vec2(-q, -q)) + Parent(UV + vec2(q, -q)) + Parent(UV + vec2(-q, q)) + Parent(UV + vec2(q, q))), 1.0);
}
