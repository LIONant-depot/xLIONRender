#version 450
// Test material cut out everywhere (an alpha tested material whose texture's alpha is 0 all over): every fragment is discarded, the GI's
// material capture's too. What the capture reads there must be no albedo and no emission (xlionrender_test_gi_render.cpp).
#include "mb_standard_pbr.frag"

layout(location = 0) out vec4 outFragColor;

void main()
{
    discard;
}
