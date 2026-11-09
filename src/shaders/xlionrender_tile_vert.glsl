#version 450
// The tile pass's full-screen triangle at its graph's depth (XLION_TILE_CLASSIFY, A5.2: xlionrender_tile_classify_frag.glsl wrote each
// pixel's graph as (graph + 1) / 65536; the draw tested EQUAL runs only on its own pixels). The push: the tile pass's (clip to world, then
// the graph in the low half of the word after it). Outputs as xlionrender_fullscreen_vert.glsl
layout(location = 0) in vec2 in_Pos;
layout(location = 0) out vec2 out_UV;
layout(push_constant) uniform tile_push { mat4 C2W; uint Graph; uint Pad0, Pad1, Pad2; } push;

void main()
{
    out_UV      = in_Pos * 0.5 + 0.5;
    gl_Position = vec4(in_Pos, float((push.Graph & 0xFFFFu) + 1u) / 65536.0, 1.0);
}
