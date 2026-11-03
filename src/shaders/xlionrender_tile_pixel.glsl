// A material's tile variant (XLION_TILE_PIXEL, the material compiler; material_tiles_plan.md): its graph, as compiled, shaded full-screen from what the G-buffer prepass kept: no second raster, no overdraw.
// The pixel's material: the id's w; the block's vec4 before it says its graph (instance::SetMaterialTile): another graph's pixel waits for that graph's draw. A graph that reads what isn't kept (the second UV) has no variant: the compiler leaves it on the raster path
#include "xlionrender_tile_attr.glsl"
layout(binding = 1) uniform sampler2D TileDepth;
layout(binding = 2) uniform sampler2D TileNormal;
layout(binding = 3) uniform sampler2D TileId;
layout(binding = 4) uniform usampler2D TileAttr0;
layout(binding = 5) uniform usampler2D TileAttr1;
layout(binding = 6) uniform sampler2D TileColor;
vec2 g_TileDx, g_TileDy;
vec3 TileOct(in const vec2 o) { vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y)); if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0); return normalize(n); }
// A march toward a paged light through the prepass's depth: 4 steps, xlionrender_contact_comp.glsl's way
float TileContact(in const vec3 P, in const vec3 ToLight)
{
    const float Len = length(ToLight), Dist = length(P);
    const float Reach = min(UBOLight.ShadowLevels[0].w, Len * 0.9);
    if (Reach <= 0.0 || Dist > 15.0) return 1.0;
    const vec3  Dir = ToLight / Len;
    const vec3  Start = P + Dir * (0.004 + 0.002 * Dist);
    const vec2  Size = vec2(textureSize(TileDepth, 0));
    const float Jit = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    for (int k = 0; k < 4; ++k)
    {
        const float t = (float(k) + Jit) / 4.0;
        const vec4  C = UBOLight.TailW2C * vec4(Start + Dir * (t * t * Reach), 1.0);
        if (C.w <= 1e-4) return 1.0;
        const vec3  n = C.xyz / C.w;
        if (any(greaterThan(abs(n.xy), vec2(1.0)))) return 1.0;
        const float zs = texelFetch(TileDepth, ivec2((n.xy * 0.5 + 0.5) * Size), 0).r;
        if (zs >= n.z || zs >= 1.0) continue;
        const vec4  Ws = UBOLight.TailC2W * vec4(n.xy, zs, 1.0);
        if (C.w - 1.0 / Ws.w < 0.15) return smoothstep(7.5, 15.0, Dist);       // (blocked, fading out with the distance)
    }
    return 1.0;
}
void TilePixelSetup()
{
    const ivec2 Px = ivec2(gl_FragCoord.xy);
    const vec4 Id = texelFetch(TileId, Px, 0);
    if (Id.w < 0.5) discard;       // (nothing, masked, skinned: the raster path)
    const uint Base = uint(Id.w) - 1u;
    if (Base == 0u) discard;       // (no block: the raster path)
    if (floatBitsToUint(g_Mat.P[Base - 1u].x) != g_Tile.m_Graph) discard;
    g_TileBase = Base; InInstance = uint(Id.x) - 1u; g_TileFront = Id.y >= 0.0;
    // Its textures' ids, read once and checked: the heap's limits (xgpu_vulkan_texture_table.cpp); ponytail: without it the device was lost, the cause is not known
    for (uint i = 0u; i < XMAT_NTEX && i < 16u; ++i)
    {
        g_TileTex[i] = XMAT_TEXID_RAW(i); g_TileSamp[i] = XMAT_SAMPID_RAW(i);
        if (g_TileTex[i] == 0u || g_TileTex[i] >= 65536u || g_TileSamp[i] >= 1024u) discard;
    }
    const vec2 uv = (vec2(Px) + 0.5) / vec2(textureSize(TileId, 0));
    const vec4 Pc = g_Tile.m_C2W * vec4(uv * 2.0 - 1.0, texelFetch(TileDepth, Px, 0).r, 1.0);
    In.wSpacePosition = vec4(Pc.xyz / Pc.w, 1.0);
    g_TileDist = 1.0 / Pc.w;
    In.ShadowPosition = vec4(0.0);
    In.VertColor = texelFetch(TileColor, Px, 0);
    In.Normal = TileOct(texelFetch(TileNormal, Px, 0).xy);
    TileDecode(texelFetch(TileAttr0, Px, 0).xy, In.UV, g_TileDx, g_TileDy);
    const uint Tw = texelFetch(TileAttr1, Px, 0).x;
    In.Tangent = vec4(TileOct(unpackSnorm2x16(Tw)), (Tw & 1u) != 0u ? -1.0 : 1.0);
    InUV1 = In.UV;
    InGIRef = uvec2(0u, gi_screen_v);
}
#define texture(S, Uv) textureGrad(S, Uv, g_TileDx, g_TileDy)
