#version 450
// DebugView: one of the view's images (render_view::ForEachImage, by its render graph name) in place of the frame, nearest texel, scaled
// to the view. Value = texel * Scale + Bias, then by Mode: 0 colour (rgb, display encoded), 1 one channel as grey (Channel), 2 a normal
// (xyz * 0.5 + 0.5; two channels: blue 0.5), 3 depth (grey: white near, dark far - 1 - exp(-(1 - depth) * 1000 * Scale): the depth crowds near 1).
layout(binding = 0) uniform sampler2D Src;
layout(binding = 1) uniform sampler2D ExposureTex;      // (the frame's adapted exposure: Mode.w 2)

layout(push_constant) uniform PushConstants
{
    vec4  Remap;    // x Scale, y Bias, zw the output's size
    ivec4 Mode;     // x the mode, y the channel, z the mip, w 2: times the adapted exposure (an HDR image's auto colour: as the frame is exposed)
} P;

layout(location = 0) out vec4 outColor;

void main()
{
    const ivec2 Size = textureSize(Src, P.Mode.z);
    const ivec2 C    = clamp(ivec2(gl_FragCoord.xy / P.Remap.zw * vec2(Size)), ivec2(0), Size - 1);
    const float E    = P.Mode.w == 2 ? texelFetch(ExposureTex, ivec2(0), 0).r : 1.0;
    const vec4  V    = texelFetch(Src, C, P.Mode.z) * (P.Remap.x * E) + P.Remap.y;
    vec3 c;
    if(      P.Mode.x == 1 ) c = vec3(V[clamp(P.Mode.y, 0, 3)]);
    else if( P.Mode.x == 2 ) c = V.xyz * 0.5 + 0.5;
    else if( P.Mode.x == 3 ) c = vec3(1.0 - exp(-(1.0 - texelFetch(Src, C, P.Mode.z).x) * 1000.0 * P.Remap.x) + P.Remap.y);
    else                     c = V.rgb;
    c = clamp(c, 0.0, 1.0);
    if( P.Mode.x == 0 ) c = mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));    // (colour: m_Output is display encoded; data modes show their values as they are)
    outColor = vec4(c, 1.0);
}
