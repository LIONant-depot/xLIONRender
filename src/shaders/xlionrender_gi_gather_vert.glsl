#version 450
// The GI gather's full-screen triangle (xlionrender_gi_gather_common.glsl) at its GI buffer's depth: the classification wrote each pixel's
// as (index + 1) / 65536, so a pass tested EQUAL runs only on its own pixels - the others are rejected before their fragment runs
layout(location = 0) in vec2 in_Pos;
layout(push_constant) uniform gather_push { uint TableBase; uint Buffer; uint Div; uint Pad; } push;

void main()
{
    gl_Position = vec4(in_Pos, float(push.Buffer + 1u) / 65536.0, 1.0);
}
