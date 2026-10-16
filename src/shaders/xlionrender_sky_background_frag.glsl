#version 450
// The sky as a background (scene::sky::m_bVisible): a cubemap seen from inside an infinitely far sphere. A fullscreen triangle at the far plane
// (xlionrender_clear_depth_vert.glsl), depth-tested against the scene so everything that was drawn stays in front; the color is the cubemap in the
// direction of the pixel, times the sky's intensity, and alpha 1 (coverage: the tone mapper shows the background color only where nothing was drawn).
layout(binding = 0) uniform samplerCube SamplerCube;

layout(push_constant) uniform sky_background
{
    mat4 InvW2C;        // clip -> the camera-centered world
    vec4 Params;        // x: the intensity, yz: the size of the view in pixels
} P;

layout(location = 0) out vec4 outColor;

void main()
{
    const vec2 NDC = gl_FragCoord.xy / P.Params.yz * 2.0 - 1.0;         // (the projection already flips Y for Vulkan: no second flip)
    const vec4 W   = P.InvW2C * vec4(NDC, 0.0, 1.0);
    const vec3 Dir = normalize(W.xyz / W.w);                            // a point of the pixel's ray on the near plane (the far one is at infinity: w = 0); the eye is the origin of the camera-centered world
    outColor = vec4(textureLod(SamplerCube, Dir, 0.0).rgb * P.Params.x, 1.0);
}
