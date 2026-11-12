#version 450
// The GI bricks' self-check (XLION_GI_BRICKS, B0: does the publication reproduce the store?): random surface points the gather saw
// (GIRefSample: world position, normal), each inside this level's window read twice - the store's read of the level (xlionrender_vox_levels.glsl
// ClipLevel, the gather's) and the bricks' 3 filtered samples (xlionrender_gi_brick.glsl BrickRead) - into a ring slot of counts:
// [0] points compared (the store had light), [1] of those the bricks had none, [2] sum of |bricks - store| / store (luminance) x 1000,
// [3] points off by more than 10 %, [4] the store's luminance x 100 summed, [5] the bricks' (the signed bias), [6] points outside the window
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x = 64) in;

layout(push_constant) uniform check_push { uint TableBase; int Level; uint nPixels; uint Seed; uint Slot; uint Flags; uint Pad1, Pad2; } push;
layout(std430, set = 0, binding = 0) readonly buffer GIClipBuffer { uvec4 GIClip[]; };
layout(std430, set = 0, binding = 1) readonly buffer GIGatherTable { uint GITable[]; };
layout(std430, set = 0, binding = 2) readonly buffer GIRefSampleBuffer { vec4 GIRefSample[]; };
layout(std430, set = 0, binding = 3) buffer StatsBuffer { uint Stats[]; };
layout(set = 1, binding = 0) uniform sampler3D GIBrick0;
layout(set = 1, binding = 1) uniform sampler3D GIBrick1;
layout(set = 1, binding = 2) uniform sampler3D GIBrick2;
layout(set = 1, binding = 3) uniform sampler3D GIBrick3;
layout(set = 1, binding = 4) uniform sampler3D GIBrick4;
layout(set = 1, binding = 5) uniform sampler3D GIBrick5;
layout(set = 1, binding = 6) uniform sampler3D GIBrick6;
layout(set = 1, binding = 7) uniform sampler3D GIBrick7;
vec4 BrickTex(in const int L, in const uint i, in const vec3 u)      // (one level's volumes bound a dispatch)
{
    switch (i)
    {
    case 0u: return textureLod(GIBrick0, u, 0.0); case 1u: return textureLod(GIBrick1, u, 0.0); case 2u: return textureLod(GIBrick2, u, 0.0);
    case 3u: return textureLod(GIBrick3, u, 0.0); case 4u: return textureLod(GIBrick4, u, 0.0); case 5u: return textureLod(GIBrick5, u, 0.0);
    case 6u: return textureLod(GIBrick6, u, 0.0); default: return textureLod(GIBrick7, u, 0.0);
    }
}

bool g_Lean = true;        // (XLION_GI_BRICKS flag 1: the lean moments off - store and bricks alike)
#define VOX_LEAN_V g_Lean
#include "xlionrender_vox_levels.glsl"
#include "xlionrender_gi_brick.glsl"

uint Mix(uint s) { s ^= s >> 16; s *= 0x7FEB352Du; s ^= s >> 15; s *= 0x846CA68Bu; s ^= s >> 16; return s; }
float Luma(in const vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main()
{
    uvec4 C = uvec4(0u); uvec3 D = uvec3(0u);       // (C: compared, none, error x 1000, over 10 %; D: store, bricks, outside)
    const uint  s  = Mix(gl_GlobalInvocationID.x * 0x9E3779B9u ^ push.Seed) % max(push.nPixels, 1u);
    const vec4  P  = GIRefSample[2u * s], N = GIRefSample[2u * s + 1u];
    const int   L  = push.Level;
    const float Cell = TF(21u) * float(1 << L);
    const ivec3 Org  = ivec3(floor(vec3(TF(22u), TF(23u), TF(24u)) / Cell)) - ivec3(clip_dim_v / 2);
    const vec3  g    = P.xyz / Cell - vec3(Org) - 0.5;
    if (P.w == 1.0 && dot(N.xyz, N.xyz) > 0.5)
    {
        if (any(lessThan(g, vec3(0.0))) || any(greaterThanEqual(g, vec3(clip_dim_v - 1)))) D.z = 1u;
        else
        {
            g_ClipFloat = true; g_Lean = (push.Flags & 1u) == 0u;
            const vec4 S = ClipLevel(L, P.xyz, N.xyz, vec3(TF(22u), TF(23u), TF(24u)), TF(21u));
            if (S.w > 0.0)
            {
                const vec4  B  = BrickRead(L, P.xyz, N.xyz);
                const float Ls = Luma(S.rgb), Lb = B.w > 0.0 ? Luma(B.rgb) : 0.0, E = abs(Lb - Ls) / max(Ls, 1e-4);
                C = uvec4(1u, B.w > 0.0 ? 0u : 1u, uint(min(E, 10.0) * 1000.0), E > 0.1 ? 1u : 0u);
                D.xy = uvec2(min(vec2(Ls, Lb), vec2(1000.0)) * 100.0);
            }
        }
    }
    C = subgroupAdd(C); D = subgroupAdd(D);
    if (subgroupElect())
    {
        const uint o = push.Slot * 8u;
        atomicAdd(Stats[o + 0u], C.x); atomicAdd(Stats[o + 1u], C.y); atomicAdd(Stats[o + 2u], C.z); atomicAdd(Stats[o + 3u], C.w);
        atomicAdd(Stats[o + 4u], D.x); atomicAdd(Stats[o + 5u], D.y); atomicAdd(Stats[o + 6u], D.z);
    }
}
