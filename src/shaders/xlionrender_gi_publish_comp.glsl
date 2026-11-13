#version 450
// The published light's resolve (XLION_VOX_PUBLISHED, gi_published_store_plan_2026-11-08.md): for each corner the gather asked for last
// frame (its request list, by parity), today's whole multi-level read (xlionrender_vox_levels.glsl VoxReadLevels) at the corner's centre,
// for its face's own normal, once - into the corner's published entry, which the gather's VoxReadPublished reads (8 corners a face, one
// level) instead of walking every level at every pixel. One thread a request; the list's count bounds them
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 64) in;

layout(push_constant) uniform publish_push { uint TableBase; uint Frame; uint Parity; uint Pad; } push;
layout(std430, set = 0, binding = 0) readonly buffer GIClipBuffer { uvec4 GIClip[]; };
layout(std430, set = 0, binding = 1) readonly buffer GIGatherTable { uint GITable[]; };
layout(std430, set = 0, binding = 2) buffer PubBuffer { uvec4 Pub[]; };
layout(std430, set = 0, binding = 2) buffer PubBufferU { uint PubU[]; };

#define VOX_PUBLISHED_BUFFER
#include "xlionrender_vox_levels.glsl"

void main()
{
    const uint k = gl_GlobalInvocationID.x;
    if (k >= min(PubU[push.Parity * 4u], pub_cap_v)) return;
    const uvec4 R = Pub[pub_list0_v + push.Parity * pub_cap_v + k];
    const ivec3 q = ivec3(R.xyz);
    const int   L = int(R.w >> 8u);
    const uint  Face = R.w & 255u;
    const int   a = int((Face & 63u) >> 1u);
    vec3 n = vec3(0.0); n[a] = (Face & 1u) != 0u ? -1.0 : 1.0;
    g_ClipFloat = true;
    g_VoxBucket = Face >= 64u ? 1u : 0u;
    const vec3 Pw = PubCornerWorld(q, L, a);
    const vec4 E  = VoxReadLevels(Pw - vec3(TF(25u), TF(26u), TF(27u)), n);
    const uvec2 h  = VoxHash(L, q, Face); const uint s = h.x % vox_slots_v;
    Pub[pub_entry0_v + s] = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, 0.0)), floatBitsToUint(VoxTagOf(h)), (push.Frame & 0x7FFFFFFFu) | (E.w > 0.0 ? 0x80000000u : 0u));
}
