#version 450
// Reflections: the hits' textures (documentation/reflections_strategy.md sec. 2: D(y) = L_e + albedo / pi (E_direct + E_indirect) + S,
// the albedo and the emission the material's own at the hit, not its per-triangle capture). A visibility buffer of the reflection rays'
// hits (xlionrender_refl_trace_comp.glsl's Hits: instance, triangle, barycentrics, E, S, the ray cone's width), shaded by each material's
// own GI-capture variant: the hits binned by their draw's material (instance::m_HitTables: per GI instance its material slots' bins), each
// hit a one-pixel quad of the shade target in its bin's range, its UV the barycentrics' and its UV radius the cone's (the texture's mip:
// the quad's derivatives) - xlionrender_gi_capture_vert.glsl reads them from Capture. Passes:
//   0 count:   per hit its bin's count
//   1 scan:    one workgroup - each bin's first, the indirect draws (albedo, emission), the counts kept for the CPU (it draws the bins
//              counted the frame its ring slot comes back: a bin hit for the first time shows its capture's albedo a frame)
//   2 scatter: per hit its place (its list entry's), its Capture entry
//   3 relight: per hit placed in a bin the CPU drew this frame: U = L_e + albedo E + S from the shade target, into the reflection records
layout(local_size_x = 256) in;

#include "xlionrender_gi_pass_common.glsl"           // (bindings 0-4: the GI's data - the meshes' UVs and triangles' slots)

struct draw_t { uint m_IndexCount, m_InstanceCount, m_FirstIndex; int m_VertexOffset; uint m_FirstInstance; };
layout(std430, set = 0, binding = 5)          buffer hits_b     { uvec4 Hits[]; };        // the trace's list: [0] the counts (by parity), then three a hit (the last: its key, its place)
layout(std430, set = 0, binding = 6) readonly buffer tables_b   { uint Tables[]; };       // per GI instance its slots' first (~0u: none), then the slots' bins (~0u: none); the drawn bins' bits after
layout(std430, set = 0, binding = 7)          buffer bins_b     { uint Bins[]; };         // per bin: count, first, fill (3 a bin)
layout(std430, set = 0, binding = 8)          buffer counts_b   { uint Counts[]; };       // per ring slot, per bin: this frame's count (the CPU reads it)
layout(std430, set = 0, binding = 9)          buffer draws_b    { draw_t Draws[]; };      // per pass (albedo, emission), per bin
layout(std430, set = 0, binding = 10)         buffer capture_b  { vec4 Capture[]; };      // per placed hit: { u, v, UV radius, - }, { normal (mesh), - }
layout(std430, set = 0, binding = 11)         buffer records_b  { uvec4 Records[]; };     // the view's records (the reflections' after the tail's)
layout(set = 1, binding = 0) uniform sampler2D Shade;         // the shade target: the albedo's rows, then the emission's

// m_Keys: the list's capacity; m_Parity: the frame's (its count); m_Half: W x H; m_FullBase: the full size's records' first (2 pixels a record);
// m_Bins: how many; m_Cap: the shade target's places a pass; m_Drawn: the drawn bins' bits' first word in Tables; m_Slot: the ring slot
// m_TableBase: this ring slot's Tables; m_TexMax: scene::m_ReflectionTextureMinReflectance (the shaded texels blend in above 3/4 of it)
layout(push_constant) uniform push_t { uint m_Pass, m_Keys, m_Half, m_FullBase, m_Bins, m_Cap, m_Drawn, m_Slot, m_Width, m_TableBase; float m_TexMax; uint m_Parity, m_Debug; } Push;     // (m_Debug 1: the shaded albedo itself, unlit - XLION_REFL_HITS_ALBEDO)
uint Count() { return min(Hits[0][Push.m_Parity], Push.m_Keys); }

uint BinOf(uvec4 A)
{
    if (A.x == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    const uint First = Tables[Push.m_TableBase + A.x];
    if (First == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    const uint M = MeshOf(InstRec(int(A.x)));
    if (GD[M + 16u] == 0xFFFFFFFFu) return 0xFFFFFFFFu;            // (no UVs)
    const uint Slot = GD[GD[M + 16u] + 2u * GD[M] + A.y];
    return Tables[Push.m_TableBase + First + Slot];
}

shared uint s_Sum[256];
// the count's and the scatter's per-workgroup bins (a hash of the bins its hits fall in): one global atomic a bin a workgroup - a
// mirror's hits are mostly one bin, and a million atomics on one counter cost tens of milliseconds
shared uint s_Key[256], s_Cnt[256], s_Base[256];
uint LocalSlot(uint b)
{
    uint h = (b * 2654435761u) >> 24u;
    for (uint t = 0u; t < 256u; ++t, h = (h + 1u) & 255u)
    {
        const uint Was = atomicCompSwap(s_Key[h], 0xFFFFFFFFu, b);
        if (Was == 0xFFFFFFFFu || Was == b) return h;
    }
    return 0xFFFFFFFFu;
}

void main()
{
    const uint k = gl_GlobalInvocationID.x;
    if (Push.m_Pass == 1u)
    {
        // each bin's first (an exclusive sum over the bins, 256 at a time) and its draws, clamped to the shade target's places
        uint Base = 0u;
        for (uint b0 = 0u; b0 < Push.m_Bins; b0 += 256u)
        {
            const uint b = b0 + gl_LocalInvocationIndex;
            const uint n = b < Push.m_Bins ? Bins[3u * b] : 0u;
            s_Sum[gl_LocalInvocationIndex] = n;
            barrier();
            for (uint o = 1u; o < 256u; o <<= 1u)
            {
                const uint v = gl_LocalInvocationIndex >= o ? s_Sum[gl_LocalInvocationIndex - o] : 0u;
                barrier();
                s_Sum[gl_LocalInvocationIndex] += v;
                barrier();
            }
            if (b < Push.m_Bins)
            {
                const uint First = Base + s_Sum[gl_LocalInvocationIndex] - n;
                const uint Kept  = First >= Push.m_Cap ? 0u : min(n, Push.m_Cap - First);
                Bins[3u * b + 1u] = First; Bins[3u * b + 2u] = 0u;
                Counts[Push.m_Slot * 4096u + b] = n;                      // (instance::hit_max_bins_v a ring slot)
                for (uint Pass = 0u; Pass < 2u; ++Pass)
                    Draws[Pass * Push.m_Bins + b] = draw_t(6u, Kept, 0u, 0, min(First, Push.m_Cap));      // (a quad, an instance a hit)
            }
            const uint Total = s_Sum[255];
            barrier();
            Base += Total;
        }
        return;
    }
    if (Push.m_Pass == 0u || Push.m_Pass == 2u)
    {
        const uvec4 A = k < Count() ? Hits[1u + 3u * k] : uvec4(0xFFFFFFFFu);
        const uint  b = BinOf(A);
        s_Key[gl_LocalInvocationIndex] = 0xFFFFFFFFu; s_Cnt[gl_LocalInvocationIndex] = 0u;
        barrier();
        const uint  h = b != 0xFFFFFFFFu ? LocalSlot(b) : 0xFFFFFFFFu;
        const uint  Mine = h != 0xFFFFFFFFu ? atomicAdd(s_Cnt[h], 1u) : 0u;
        barrier();
        const uint  i = gl_LocalInvocationIndex;
        if (s_Key[i] != 0xFFFFFFFFu) s_Base[i] = atomicAdd(Bins[3u * s_Key[i] + (Push.m_Pass == 0u ? 0u : 2u)], s_Cnt[i]);
        barrier();
        if (Push.m_Pass == 0u || k >= Count()) return;
        if (h == 0xFFFFFFFFu) return;
        const uint At = Bins[3u * b + 1u] + s_Base[h] + Mine;
        if (At >= Push.m_Cap || s_Base[h] + Mine >= Bins[3u * b]) return;
        Hits[3u + 3u * k].y = At;
        // the hit's UV, and its UV radius: the cone's width over the triangle's world-to-UV scale
        const uint  R = InstRec(int(A.x)), M = MeshOf(R);
        const uvec3 T = MTri(M, int(A.y));
        const uint  Uv = GD[M + 16u];
        const vec2  u0 = vec2(uintBitsToFloat(GD[Uv + 2u * T.x]), uintBitsToFloat(GD[Uv + 2u * T.x + 1u]));
        const vec2  u1 = vec2(uintBitsToFloat(GD[Uv + 2u * T.y]), uintBitsToFloat(GD[Uv + 2u * T.y + 1u]));
        const vec2  u2 = vec2(uintBitsToFloat(GD[Uv + 2u * T.z]), uintBitsToFloat(GD[Uv + 2u * T.z + 1u]));
        const float bu = uintBitsToFloat(A.z), bv = uintBitsToFloat(A.w);
        const vec2  UV = u0 * (1.0 - bu - bv) + u1 * bu + u2 * bv;
        const vec3  p0 = ToWorld(R, MPos(M, T.x)), p1 = ToWorld(R, MPos(M, T.y)), p2 = ToWorld(R, MPos(M, T.z));
        const float Aw = length(cross(p1 - p0, p2 - p0));
        const float Au = abs((u1.x - u0.x) * (u2.y - u0.y) - (u2.x - u0.x) * (u1.y - u0.y));
        const float Cone = unpackHalf2x16(Hits[2u + 3u * k].w).x;
        const float Ruv = 0.5 * Cone * sqrt(Au / max(Aw, 1e-12));
        const vec3  Nm = normalize(MNrm(M, T.x) * (1.0 - bu - bv) + MNrm(M, T.y) * bu + MNrm(M, T.z) * bv);
        Capture[2u * At]      = vec4(UV, Ruv, 0.0);
        Capture[2u * At + 1u] = vec4(Nm, 0.0);
        return;
    }
    // pass 3: relight the hits the CPU drew
    if (k == 0u) Hits[0][Push.m_Parity ^ 1u] = 0u;                     // (the next frame's count: nobody reads it this frame)
    if (k < Push.m_Bins) Bins[3u * k] = 0u;                            // (the bins' counts, for the next frame's: pass 2 was their last reader)
    if (k >= Count()) return;
    const uvec4 A = Hits[1u + 3u * k];
    const uint At = Hits[3u + 3u * k].y;
    if (At == 0xFFFFFFFFu) return;
    const uint b = BinOf(A);
    if ((Tables[Push.m_TableBase + Push.m_Drawn + b / 32u] & (1u << (b % 32u))) == 0u) return;

    const ivec2 t = ivec2(int(At % Push.m_Width), int(At / Push.m_Width));
    const vec3  Albedo   = texelFetch(Shade, t, 0).rgb;
    const vec3  Emission = texelFetch(Shade, t + ivec2(0, int(Push.m_Cap / Push.m_Width)), 0).rgb;
    const uvec4 B = Hits[2u + 3u * k];
    const uint  Key = Hits[3u + 3u * k].x;
    const vec3  E = vec3(unpackHalf2x16(B.x), unpackHalf2x16(B.y).x), S = vec3(unpackHalf2x16(B.y).y, unpackHalf2x16(B.z));
    const vec3  Ut = Push.m_Debug == 1u ? Albedo : Push.m_Debug == 2u ? vec3(1.0, 0.0, 1.0) : Emission + Albedo * E + S;
    const float Wt = smoothstep(0.75 * Push.m_TexMax, Push.m_TexMax, unpackHalf2x16(B.w).y);       // (the receiver's selection weight: no seam)
    if (Key < Push.m_Half)
    {
        const uint  i = Push.m_Half + Key;
        const uvec4 R = Records[i];
        const vec3  U = mix(vec3(unpackHalf2x16(R.x), unpackHalf2x16(R.y).x), Ut, Wt);
        Records[i] = uvec4(packHalf2x16(U.rg), packHalf2x16(vec2(U.b, unpackHalf2x16(R.y).y)), R.z, R.w);
    }
    else
    {
        // (the pair's x, y or z, w: its rgb and its b with the depth - its own two words only: the pair's other pixel is another thread's)
        const uint  f = Key - Push.m_Half, r = Push.m_FullBase + f / 2u;
        if ((f & 1u) == 0u) { const uvec2 R = Records[r].xy; const vec3 U = mix(vec3(unpackHalf2x16(R.x), unpackHalf2x16(R.y).x), Ut, Wt); Records[r].x = packHalf2x16(U.rg); Records[r].y = packHalf2x16(vec2(U.b, unpackHalf2x16(R.y).y)); }
        else                { const uvec2 R = Records[r].zw; const vec3 U = mix(vec3(unpackHalf2x16(R.x), unpackHalf2x16(R.y).x), Ut, Wt); Records[r].z = packHalf2x16(U.rg); Records[r].w = packHalf2x16(vec2(U.b, unpackHalf2x16(R.y).y)); }
    }
}
