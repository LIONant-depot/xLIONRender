// The GI gather from the G-buffer prepass (gi_gbuffer_gather_plan.md, stage 2): shared by its classification pass and its passes, one a
// GI buffer (xlionrender_gi_classify_frag.glsl, xlionrender_gi_gather_frag.glsl; instance::renderScenePass). Each half-size pixel takes
// a full-size prepass sample of its block (a checkerboard): its depth, world normal (octahedral) and submesh instance (slot + 1, cluster).
// The table (this frame's region from push.TableBase, uints): 0-15 clip -> camera-centred world (columns), 16-18 the instance buffer's
// origin in that world, 19 the slots it lists; then per slot the first of its clusters' entries (0: none); an entry 4 uints - the GI's
// first surfel in GILight, its level (gi_none_v: none), its GI buffer's index (the pass that reads it), 0
#ifndef XLIONRENDER_GI_GATHER_COMMON_GLSL
#define XLIONRENDER_GI_GATHER_COMMON_GLSL

layout(binding = 1) uniform sampler2D GPreDepth;
layout(binding = 2) uniform sampler2D GPreNormal;
layout(binding = 3) uniform sampler2D GPreId;
layout(std430, set = 1, binding = 19) readonly buffer GIGatherTable { uint GITable[]; };

layout(push_constant) uniform gather_push
{
    uint TableBase;
    uint Buffer;        // this pass's GI buffer (its depth: (Buffer + 1) / 65536)
    uint Div;           // the read's resolution divisor
    uint Pad;
} push;

struct gather_sample
{
    bool  bHit;         // a GI surface is there
    ivec2 Q;            // the full-size texel
    vec3  P;            // camera-centred world position
    vec3  N;            // world normal (its visible side)
    uint  Slot, Cluster;
    uvec4 Entry;        // the table's
};

gather_sample GatherSample(in const ivec2 p)
{
    gather_sample S;
    S.bHit = false;
    // (a checkerboard of the block's two middle texels: both surfaces of an edge are sampled, and on average at the block's centre -
    // the upsample's half-size pixel centre; one texel for all left the edges unmatched: dashes of another surface's GI)
    const int D = int(push.Div);
    S.Q    = p * D + ivec2(((p.x ^ p.y) & 1) == 0 ? D / 2 - 1 : D / 2);
    const ivec2 Size = textureSize(GPreDepth, 0);
    S.Q    = min(S.Q, Size - 1);
    const vec2 Id = texelFetch(GPreId, S.Q, 0).xy;
    if (Id.x < 0.5) return S;
    S.Slot = uint(Id.x + 0.5) - 1u; S.Cluster = uint(Id.y + 0.5);
    const uint T = push.TableBase;
    if (S.Slot >= GITable[T + 19u]) return S;
    const uint First = GITable[T + 20u + S.Slot];
    if (First == 0u) return S;
    const uint e = T + First + 4u * S.Cluster;
    if (S.Cluster > 0xFFFFu || e + 3u >= uint(GITable.length())) return S;      // (an id no G-buffer writer gave: a material compiled before XLION_GPRE_OUT writes nothing defined there)
    S.Entry = uvec4(GITable[e], GITable[e + 1u], GITable[e + 2u], GITable[e + 3u]);
    if (S.Entry.y == gi_none_v) return S;
    const mat4  C2W = mat4(uintBitsToFloat(uvec4(GITable[T + 0u], GITable[T + 1u], GITable[T + 2u], GITable[T + 3u])),
                           uintBitsToFloat(uvec4(GITable[T + 4u], GITable[T + 5u], GITable[T + 6u], GITable[T + 7u])),
                           uintBitsToFloat(uvec4(GITable[T + 8u], GITable[T + 9u], GITable[T + 10u], GITable[T + 11u])),
                           uintBitsToFloat(uvec4(GITable[T + 12u], GITable[T + 13u], GITable[T + 14u], GITable[T + 15u])));
    const float z = texelFetch(GPreDepth, S.Q, 0).r;
    const vec4  W = C2W * vec4((vec2(S.Q) + 0.5) / vec2(Size) * 2.0 - 1.0, z, 1.0);
    S.P = W.xyz / W.w;
    const vec2 o = texelFetch(GPreNormal, S.Q, 0).xy;
    vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    S.N    = normalize(n);
    S.bHit = true;
    return S;
}

#endif
