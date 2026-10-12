#version 450
// The environment's source (features/environment): the procedural sky - ground / horizon / zenith gradient and the sun's disc - into
// an octahedral map with a kEnvBorder border (folded), drawn as a fullscreen triangle over the whole texture.
#include "xlionrender_environment_atlas.glsl"

layout(push_constant) uniform sky
{
    vec4 SunDir;        // xyz: toward the sun, w: cos of the disc's angular radius
    vec4 SunColor;      // rgb: color x intensity (radiance of the disc)
    vec4 Zenith;
    vec4 Horizon;
    vec4 Ground;        // w: the source's size
} P;

layout(location = 0) out vec4 outColor;

void main()
{
    const float Size  = P.Ground.w;
    const float Inner = Size - 2.0 * kEnvBorder;
    const vec2  UV    = (gl_FragCoord.xy - kEnvBorder) / Inner * 2.0 - 1.0;
    const vec3  D     = OctDecode(OctFold(UV));

    const float Up  = D.y;
    vec3 C = Up >= 0.0 ? mix(P.Horizon.rgb, P.Zenith.rgb, pow(Up, 0.5)) : mix(P.Horizon.rgb, P.Ground.rgb, pow(-Up, 0.35));
    if (dot(D, P.SunDir.xyz) >= P.SunDir.w) C += P.SunColor.rgb;
    outColor = vec4(C, 1.0);
}
