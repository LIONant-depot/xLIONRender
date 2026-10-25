// A texture-table material as the material compiler writes one (xmaterial.plugin Compiler/xmaterial_compiler.cpp TextureTop and
// TextureDecls): two textures, a one-vec4 parameter block, the texture ids after it (vec4 1: slot 0 in .xy, slot 1 in .zw). Included
// by xlionrender_test_table_frag.glsl (the table's set) and xlionrender_test_table_bindless_frag.glsl (XGPU_BINDLESS: the heaps by id).
// HDR output = (slot 0's red * P.x + slot 1's red * P.y) * 3
#include "mb_standard_pbr.frag"

layout(std430, set = 1, binding = 9) readonly buffer MaterialParamsBuf { vec4 P[]; } g_Mat;
layout(push_constant) uniform XMatPush { layout(offset = 16) uint MaterialBase; } g_XMatPush;
#define XMAT_BASE g_XMatPush.MaterialBase

#if defined(XGPU_BINDLESS)
    layout(set = 3, binding = 0) uniform texture2D xgpu_Textures2D[];
    layout(set = 3, binding = 2) uniform sampler   xgpu_Samplers[];
    #define XGPU_TEXTURE(Id) xgpu_Textures2D[Id]
    #define XGPU_SAMPLER(Id) xgpu_Samplers[Id]
    #define XMAT_TEXID(i)  floatBitsToUint(g_Mat.P[XMAT_BASE + 1u + (i) / 2u][((i) % 2u) * 2u])
    #define XMAT_SAMPID(i) floatBitsToUint(g_Mat.P[XMAT_BASE + 1u + (i) / 2u][((i) % 2u) * 2u + 1u])
    #define Slot0 sampler2D(XGPU_TEXTURE(XMAT_TEXID(0u)), XGPU_SAMPLER(XMAT_SAMPID(0u)))
    #define Slot1 sampler2D(XGPU_TEXTURE(XMAT_TEXID(1u)), XGPU_SAMPLER(XMAT_SAMPID(1u)))
#else
    layout(set = 3, binding = 0) uniform sampler2D Slot0;
    layout(set = 3, binding = 1) uniform sampler2D Slot1;
#endif

layout(location = 0) out vec4 outFragColor;

void main()
{
    const vec4 S = g_Mat.P[XMAT_BASE];
    outFragColor = vec4(vec3(texture(Slot0, In.UV).r * S.x + texture(Slot1, In.UV).r * S.y) * 3.0, 1.0);
}
