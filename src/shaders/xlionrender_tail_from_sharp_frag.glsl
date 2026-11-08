#version 450
// The reflections' half-size thin G-buffer taken from the full-size one (the sharp receivers' G-buffer, drawn for the whole view): one texel
// of each 2 x 2 block, in the GI gather's checkerboard (the block's two middle texels alternating), its colour (normal, roughness, flags),
// motion and depth as they are. Drawing the scene a second time at half size cost 11,270 draws a frame on Bistro Game (2026-11-08).
layout(binding = 0) uniform sampler2D SharpColor;
layout(binding = 1) uniform sampler2D SharpMotion;
layout(binding = 2) uniform sampler2D SharpDepth;

layout(location = 0) in  vec2 in_UV;
layout(location = 0) out vec4 out_Color;
layout(location = 1) out vec4 out_Motion;

void main()
{
    const ivec2 p    = ivec2(gl_FragCoord.xy);
    const ivec2 Size = textureSize(SharpDepth, 0);
    const ivec2 Q    = min(p * 2 + ((((p.x ^ p.y) & 1) == 0) ? ivec2(0) : ivec2(1)), Size - 1);
    out_Color    = texelFetch(SharpColor,  Q, 0);
    out_Motion   = texelFetch(SharpMotion, Q, 0);
    gl_FragDepth = texelFetch(SharpDepth,  Q, 0).r;
}
