#version 450
// The material tile pass's prototype (XLION_TILE, material_tiles_plan.md): one graph (the imported opaque PBR, material compiler's log 8D45F86DC5F2C003), its main as compiled, shaded full-screen from
// what the G-buffer prepass kept: no second raster, no overdraw. ponytail: one graph by hand, all opaque texture-table materials assumed to be it, vertex color white; the compiler emits this for real if the measurement pays
#extension GL_ARB_separate_shader_objects  : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_nonuniform_qualifier : require
#define XGPU_BINDLESS
#define XLION_TILE_PIXEL
uint g_TileBase = 0u;
uint InInstance = 0u;
bool g_TileFront = true;
#define XLION_FRONT g_TileFront
#define XMAT_BASE g_TileBase
#define XMAT_PUSH_CLUSTER 0u
#define XMAT_PUSH_INSTANCE 0u
layout(push_constant) uniform TilePush { mat4 m_C2W; } g_Tile;
layout(std430, set = 1, binding = 9) readonly buffer MaterialParamsBuf { vec4 P[]; } g_Mat;
struct XMatEntity { uvec4 H; vec4 Tint; vec4 P0; vec4 P1; };
layout(std430, set = 1, binding = 10) readonly buffer EntityParamsBuf { XMatEntity E[]; } g_Ent;
#define XMAT_ENT(Bit, Field, Swz, Mat) ((g_Ent.E[InInstance].H.x & Bit) != 0u ? g_Ent.E[InInstance].Field.Swz : (Mat))
layout(set = 3, binding = 0) uniform texture2D xgpu_Textures2D[];
layout(set = 3, binding = 2) uniform sampler   xgpu_Samplers[];
#define XGPU_TEXTURE(Id) xgpu_Textures2D[nonuniformEXT(Id)]
#define XGPU_SAMPLER(Id) xgpu_Samplers[nonuniformEXT(Id)]
#define XMAT_TEXID(i)  floatBitsToUint(g_Mat.P[XMAT_BASE + 3u + (i) / 2u][((i) % 2u) * 2u])
#define XMAT_SAMPID(i) floatBitsToUint(g_Mat.P[XMAT_BASE + 3u + (i) / 2u][((i) % 2u) * 2u + 1u])
#define AlbedoSampler sampler2D(XGPU_TEXTURE(XMAT_TEXID(0u)), XGPU_SAMPLER(XMAT_SAMPID(0u)))
#define ORMESampler   sampler2D(XGPU_TEXTURE(XMAT_TEXID(1u)), XGPU_SAMPLER(XMAT_SAMPID(1u)))
#define NormalSampler sampler2D(XGPU_TEXTURE(XMAT_TEXID(2u)), XGPU_SAMPLER(XMAT_SAMPID(2u)))
struct { vec4 wSpacePosition; vec4 ShadowPosition; vec4 VertColor; vec4 Tangent; vec3 Normal; vec2 UV; } In;
vec2 InUV1 = vec2(0.0);
#define XMAT_VARYING_IN
#include "mb_standard_pbr.frag"
layout(location = 0) out vec4 outFragColor;

layout(binding = 1) uniform sampler2D TileDepth;
layout(binding = 2) uniform sampler2D TileNormal;
layout(binding = 3) uniform sampler2D TileId;
layout(binding = 4) uniform sampler2D TileAttr0;
layout(binding = 5) uniform sampler2D TileAttr1;
vec2 g_TileDx, g_TileDy;
vec3 TileOct(in const vec2 o) { vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y)); if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0); return normalize(n); }
#define texture(S, Uv) textureGrad(S, Uv, g_TileDx, g_TileDy)

void main()
{
    const ivec2 Px = ivec2(gl_FragCoord.xy);
    const vec4 Id = texelFetch(TileId, Px, 0);
    if (Id.w < 0.5) discard;       // (nothing, masked, skinned: the raster path)
    g_TileBase = uint(Id.w) - 1u; InInstance = uint(Id.x) - 1u; g_TileFront = Id.y >= 0.0;
    const vec4 A0 = texelFetch(TileAttr0, Px, 0), A1 = texelFetch(TileAttr1, Px, 0);
    const vec2 uv = (vec2(Px) + 0.5) / vec2(textureSize(TileId, 0));
    const vec4 Pc = g_Tile.m_C2W * vec4(uv * 2.0 - 1.0, texelFetch(TileDepth, Px, 0).r, 1.0);
    In.wSpacePosition = vec4(Pc.xyz / Pc.w, 1.0);
    In.ShadowPosition = vec4(0.0);
    In.VertColor = vec4(1.0);
    In.Normal = TileOct(texelFetch(TileNormal, Px, 0).xy);
    const bool bNeg = A1.w > 2.0;
    In.Tangent = vec4(TileOct(vec2(A1.z, bNeg ? A1.w - 4.0 : A1.w)), bNeg ? -1.0 : 1.0);
    In.UV = A0.xy; g_TileDx = A0.zw; g_TileDy = A1.xy;
    InGIRef = uvec2(0u, gi_screen_v);

const uint Tid0 = XMAT_TEXID(0u), Tid1 = XMAT_TEXID(1u), Tid2 = XMAT_TEXID(2u), Sid0 = XMAT_SAMPID(0u), Sid1 = XMAT_SAMPID(1u), Sid2 = XMAT_SAMPID(2u);
if (Tid0 - 1u >= 4000u || Tid1 - 1u >= 4000u || Tid2 - 1u >= 4000u || Sid0 >= 16u || Sid1 >= 16u || Sid2 >= 16u) { outFragColor = vec4(5.0, 0.0, 5.0, 1.0); return; }      // (ponytail: ids out of the range Bistro uses read as magenta; without this the prototype lost the device, cause not found: a material whose block isn t there yet?)
vec4 Albedo = texture(AlbedoSampler, vec2(In.UV.xy));
vec4 BaseColorFactor = vec4((g_Mat.P[XMAT_BASE + 0u].xyzw)[0],(g_Mat.P[XMAT_BASE + 0u].xyzw)[1],(g_Mat.P[XMAT_BASE + 0u].xyzw)[2],(g_Mat.P[XMAT_BASE + 0u].xyzw)[3]);
vec3 AlbedoTint = vec3(Albedo.xyz) * vec3(BaseColorFactor.xyz);
vec3 BaseColor = vec3(AlbedoTint.xyz) * vec3(In.VertColor.xyz);
vec4 ORME = texture(ORMESampler, vec2(In.UV.xy));
float RoughnessFactor = g_Mat.P[XMAT_BASE + 2u].w;
float Roughness = ORME.y * RoughnessFactor;
float MetallicFactor = g_Mat.P[XMAT_BASE + 2u].y;
float Metallic = ORME.z * MetallicFactor;
vec4 NormalTex = texture(NormalSampler, vec2(In.UV.xy));
float NormalStrength = g_Mat.P[XMAT_BASE + 2u].x;
vec3 Normal = vec3(vec4(NormalTex.xyzw).gr * 2.0 - 1.0, 0.0);
Normal.z = sqrt(max(1.0 - dot(Normal.xy, Normal.xy), 0.0));
Normal = normalize(vec3(Normal.xy * NormalStrength, Normal.z));
float One = 1.0;
float AOStrength = g_Mat.P[XMAT_BASE + 2u].z;
float AO = mix(One, ORME.x, AOStrength);
vec4 EmissiveColor = vec4((g_Mat.P[XMAT_BASE + 1u].xyzw)[0],(g_Mat.P[XMAT_BASE + 1u].xyzw)[1],(g_Mat.P[XMAT_BASE + 1u].xyzw)[2],(g_Mat.P[XMAT_BASE + 1u].xyzw)[3]);
vec3 EmissiveTint = vec3(BaseColor.xyz) * vec3(EmissiveColor.xyz);
vec3 Emissive = vec3(EmissiveTint.xyz) * vec3(ORME.www);
vec3 Out = PBRLighting(vec3(Normal.xyz), vec3(BaseColor.xyz), AO, 0.04, Roughness, Metallic, vec3(Emissive.xyz));
outFragColor = vec4(Out, 1.0);
}
