// Surfel GI (features/gi/xlionrender_surfel_gi.h): the mesh's surfel data and this frame's surfel irradiance, read per vertex
// (xlionrender_static_geom_body.glsl) and per pixel (mb_standard_pbr.frag).
//  GIMesh: a header of 8 uints: [0] level 0 surfels' first uint, [1] level 1's, [2] the read grid's table, [3] its mask (0: no grid),
//          [4] its cell size (float bits), [5] its lists, [6] the table's longest probe; from 8: per vertex 2 uints (4 x 16-bit level 0 surfel indices, 0xFFFF none);
//          per surfel 8 floats: center xyz, radius, normal xyz, parent index (uint bits; level 0); the grid (surfel_mesh::BuildReadGrid):
//          per table entry 4 uints { key low, key high (0xFFFFFFFF: empty), first, count }, the lists of level 0 surfel indices
//  GILight: per surfel of the instance's active level a uvec4: its light around the 3 directions of the HL2 basis (GIBasisRead), each in
//          RGB9E5 (gi::PackRGB9E5), and the bits of its validity (0: buried)
//  GIMoment: per GILight surfel its depth moments (gi::MomentTexel): 16 hemi-octahedral texels of the mean and mean^2 distance its rays
//          travel (mesh units, capped at its reach)
#ifndef XLIONRENDER_SURFEL_GI_GLSL
#define XLIONRENDER_SURFEL_GI_GLSL

layout(std430, set = 1, binding = 5) readonly buffer GIMeshBuffer  { uint GIMesh[]; };
layout(std430, set = 1, binding = 6) readonly buffer GILightBuffer { uvec4 GILight[]; };
vec3 UnpackRGB9E5(in const uint p) { return vec3(p & 511u, (p >> 9) & 511u, (p >> 18) & 511u) * uintBitsToFloat(uint(127 + int(p >> 27) - 24) << 23); }     // (gi::UnpackRGB9E5)
layout(std430, set = 1, binding = 7) readonly buffer GIMomentBuffer { uint GIMoment[]; };    // per GILight surfel 16 texels: half mean, half mean^2

// The GI reference a vertex shader passes on (giLevel of the push constants): the level in the low 8 bits, gi_per_pixel_v when the
// pixels may read the grid, 0xFFFFFFFF: no GI
const uint gi_none_v      = 0xFFFFFFFFu;
const uint gi_probes_v    = 0xFFFFFFFEu;
const uint gi_screen_v    = 0xFFFFFFFDu;     // the GI read at half resolution (GIMesh is its buffer: SurfelGIScreen)     // no surfels of its own: the probe grid's light (GIMesh is the probe buffer: SurfelGIProbes)
const uint gi_per_pixel_v = 0x100u;
const uint gi_debug_shift_v = 9u;        // 2 bits (scene::gi_debug): 1 the surfels (each pixel the color of its dominant surfel), 2 the GI light only,
                                         // 3 the material capture (0x800: its emission pass); then 0x1000 the captured color of each pixel's triangle (what a
                                         // ray hitting it bounces). Kept apart: materials compiled before it still capture right
const uint gi_capture_emission_v = 0x800u;
const uint gi_capture_specular_v = 0x2000u;     // the capture's specular passes (reflections' secondary hits): F0's color; with 0x800 roughness, metallic
const uint gi_no_moments_v       = 0x4000u;     // the reads skip the depth-moment test (scene::gi_settings::m_bMoments off)
bool       g_GIMoments           = true;        // GIVisibility's moment test (the caller sets it from its GI reference)
// The most list entries the per-pixel read walks, the nearest first (surfel_mesh::SortReadLists): the far ones weigh little (a dense mesh
// lists 100+, the Golem 139 on average). Bits 16-23 of the GI reference override it (measurement: XLION_GI_PIXEL_CAP; 255: all)
const uint gi_pixel_cap_v = 32u;
uint       g_GIPixelCap   = gi_pixel_cap_v;
bool       g_GINoCorners  = false;      // measurement (XLION_GI_ABLATE=pixel_nocorners, GI reference bit 24): the per-pixel read without the corner weights
uint       g_GIAblateLookup = 0u;       // measurement (XLION_GI_ABLATE=pixel_lookuponly / pixel_nolookup, bits 25-26): 1 the cell lookup alone, 2 none
uint GIPixelCap(in const uint Ref) { const uint c = (Ref >> 16u) & 0xFFu; return c == 0u ? gi_pixel_cap_v : c == 255u ? 0xFFFFFFFFu : c; }
uint GIDebugCode(in const uint Ref) { return Ref == gi_none_v ? 0u : ((Ref >> gi_debug_shift_v) & 3u) | ((Ref >> 10u) & 0x4u); }    // scene::gi_debug

vec4 g_GIDebug = vec4(0.0);             // w > 0: the debug view's color replaces the lit one (GIDebugOr)
vec3 GIDebugOr(in const vec3 Lit) { return g_GIDebug.w > 0.0 ? g_GIDebug.rgb : Lit; }

vec3 GISurfelCenter(in const uint b) { return vec3(uintBitsToFloat(GIMesh[b]), uintBitsToFloat(GIMesh[b + 1u]), uintBitsToFloat(GIMesh[b + 2u])); }
vec3 GISurfelNormal(in const uint b) { return vec3(uintBitsToFloat(GIMesh[b + 4u]), uintBitsToFloat(GIMesh[b + 5u]), uintBitsToFloat(GIMesh[b + 6u])); }

// What a normal n reads of a surfel's light (its HL2 basis, gi::BasisDirs / BasisWeights): the 3 colors weighted by n's squared
// cosines to the surfel's basis directions (normalized). The surfel's own normal reads their mean; a normal-mapped one leans to its side
#ifndef XLION_GI_READ_KERNEL
#define XLION_GI_READ_KERNEL
// The read's kernel over a surfel's reach (2 radii), of f = 1 - d / 2r (gi_surfel_reconstruction_proposal.md v2.1, work order 3): Wendland's
// C2, (1 - q)^4 (4q + 1) = f^4 (5 - 4f) - flat at the center (f^2's slope there drew a bump per surfel where the light is noisy), 1 at
// the center, 0 with its slope at the reach (gi::ReadKernel's twin)
float GIReadKernel(float f) { return f * f * f * f * (5.0 - 4.0 * f); }
#endif
vec3 GIBasisRead(in const uint LightIndex, in const vec3 SurfelNormal, in const vec3 n)
{
    const vec3  T = normalize(cross(abs(SurfelNormal.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), SurfelNormal)), B = cross(SurfelNormal, T);
    const float a = 0.40824829, b = 0.70710678, c = 0.57735027, d = 0.81649658;
    const vec3  w = pow(max(vec3(dot(n, T * -a + B * b + SurfelNormal * c), dot(n, T * -a - B * b + SurfelNormal * c), dot(n, T * d + SurfelNormal * c)), vec3(0.0)), vec3(2.0));
    const float s = w.x + w.y + w.z;
    const vec3  wn = s > 1e-12 ? w / s : vec3(1.0 / 3.0);
    const uvec4 L = GILight[LightIndex];
    return UnpackRGB9E5(L.x) * wn.x + UnpackRGB9E5(L.y) * wn.y + UnpackRGB9E5(L.z) * wn.z;
}
float GIValidity(in const uint LightIndex) { return uintBitsToFloat(GILight[LightIndex].w); }

// The anti-leak terms of a read (Wicked Engine's surfel GI), times the falloff and normal weight:
//  - the plane distance: a point off the surfel's plane (across a corner, the far side of a seam) fades within 0.7 of its reach
//  - the depth moments: a point farther than the surfel's rays travel toward it is probably behind something (Chebyshev, cubed)
float GIVisibility(in const vec3 P, in const vec3 Center, in const vec3 Normal, in const float Radius, in const uint LightIndex)
{
    const vec3  D     = P - Center;
    const float Reach = 2.0 * Radius;
    float       w     = clamp(1.0 - abs(dot(D, Normal)) / (0.7 * Reach), 0.0, 1.0);
    if (w <= 0.0 || !g_GIMoments) return w;
    // The moments are looked up from the point lifted off the surface (DDGI's normal bias): a point on the surfel's own surface lies along
    // its tangent, where the texels mix in the rays that rise into nearby geometry (blinds, louvres) and read short - dark diamonds
    const vec3  Dq = D + Normal * (0.15 * Reach);
    const float d  = length(Dq);
    const vec3  T  = normalize(cross(abs(Normal.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), Normal)), B = cross(Normal, T);
    const vec3  Dd = Dq / d;
    const float x  = dot(Dd, T), y = dot(Dd, B), z = max(dot(Dd, Normal), 0.0);
    const float k  = 1.0 / max(abs(x) + abs(y) + z, 1e-6);
    // bilinear over the 4 nearest texels (texel centers at 0.5..3.5): the nearest one alone jumps at each texel's edge (wedges)
    const vec2  uv = clamp((vec2((x + y) * k, (x - y) * k) * 0.5 + 0.5) * 4.0 - 0.5, vec2(0.0), vec2(3.0));
    const ivec2 t0 = ivec2(floor(uv)), t1 = min(t0 + 1, ivec2(3));
    const vec2  fr = uv - vec2(t0);
    const uint  m0 = LightIndex * 16u;
    const vec2  M  = mix(mix(unpackHalf2x16(GIMoment[m0 + uint(t0.y * 4 + t0.x)]), unpackHalf2x16(GIMoment[m0 + uint(t0.y * 4 + t1.x)]), fr.x)
                       , mix(unpackHalf2x16(GIMoment[m0 + uint(t1.y * 4 + t0.x)]), unpackHalf2x16(GIMoment[m0 + uint(t1.y * 4 + t1.x)]), fr.x), fr.y);
    if (d > M.x)
    {
        const float Var = max(abs(M.x * M.x - M.y), 0.0025 * Reach * Reach);       // a floor (DDGI's): a small change of distance cannot flip the weight
        w *= pow(Var / (Var + (d - M.x) * (d - M.x) + 1e-12), 3.0);
    }
    return w;
}

// The read grid's list for the cell of the surface point P (mesh space, its normal N): false when P is off the surface's cells, or there
// is no grid. The cell is the one of P lifted 5% of a cell along N (surfel_mesh::ReadCellAt): a floor at a multiple of the cell size lies
// on a boundary, and the interpolated position falls a hair to either side of it
vec3 GIOctDecode(in const vec2 e)
{
    vec3 v = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) v.xy = (1.0 - abs(v.yx)) * vec2(e.x >= 0.0 ? 1.0 : -1.0, e.y >= 0.0 ? 1.0 : -1.0);
    return normalize(v);
}
// How far (sub-cells) a point is from a patch's sub-cells (4 x 4 across the cell in the patch's plane, the mask: surfel_mesh::MaskDistance):
// 0 inside one. The patch a point reads is the nearest (the strip by a wall's foot no sample fell in: its own side is still the nearer)
float GIMaskDistance(in const vec3 P, in const vec3 N, in const vec3 PatchN, in const float Cell, in const uint Mask)
{
    const vec3 Lp = (P + N * (0.05 * Cell)) / Cell;
    const vec3 A  = abs(PatchN);
    vec2 uv = A.x >= A.y && A.x >= A.z ? Lp.yz : A.y >= A.z ? Lp.xz : Lp.xy;
    uv -= floor(uv);
    if (((Mask >> (uint(clamp(uv.y * 4.0, 0.0, 3.0)) * 4u + uint(clamp(uv.x * 4.0, 0.0, 3.0)))) & 1u) != 0u) return 0.0;     // inside one (the usual case)
    float Best = 1e30;
    for (uint b = 0u; b < 16u; ++b)
    {
        if ((Mask & (1u << b)) == 0u) continue;
        const vec2 lo = vec2(float(b & 3u), float(b >> 2u)) * 0.25;
        const vec2 d  = max(max(lo - uv, uv - (lo + 0.25)), vec2(0.0));
        Best = min(Best, max(d.x, d.y) * 4.0);
    }
    return Best;
}
bool GIReadCellAt(in const ivec3 C, in const vec3 P, in const vec3 N, inout uint First, inout uint Count, inout vec3 PatchN, inout uint Patches, inout uint nPatches)
{
    const uint  Mask  = GIMesh[3];
    const float Cell  = uintBitsToFloat(GIMesh[4]);
    const uvec3 U     = uvec3(C) & 0x1FFFFFu;                       // surfel_grid::Key: 21 bits per axis
    const uint  KeyLo = U.z | (U.y << 21);
    const uint  KeyHi = (U.y >> 11) | (U.x << 10);
    uint h = ((uint(C.x) * 73856093u) ^ (uint(C.y) * 19349663u) ^ (uint(C.z) * 83492791u)) & Mask;
    for (uint n = 0u; n < GIMesh[6]; ++n, h = (h + 1u) & Mask)        // [6]: the table's longest probe
    {
        const uint e = GIMesh[2] + 4u * h;
        if (GIMesh[e + 1u] == 0xFFFFFFFFu) return false;
        if (GIMesh[e] != KeyLo || GIMesh[e + 1u] != KeyHi) continue;
        // the cell's patch that matches the point best (surfel_mesh::ReadCellList): the one whose sub-cells hold it first (a wall across
        // the cell splits its floor in two: the point's own side), then facing its way, then nearest its plane. Word 3: count | mask << 16
        Patches = GIMesh[5] + GIMesh[e + 2u]; nPatches = GIMesh[e + 3u];
        float Best = -1e30;
        for (uint p = 0u; p < GIMesh[e + 3u]; ++p)
        {
            const uint  a  = GIMesh[5] + GIMesh[e + 2u] + 4u * p;
            const vec3  Np = GIOctDecode(unpackSnorm2x16(GIMesh[a]));
            const float c  = dot(Np, N);
            if (c < 0.5) continue;
            const uint  w3 = GIMesh[a + 3u];
            const float Upper = c - abs(dot(P, Np) - uintBitsToFloat(GIMesh[a + 1u])) / Cell + 2.0;     // (its score inside its sub-cells)
            if (Upper <= Best) continue;                                                                // (it cannot win: no mask distance)
            const float Score = Upper - 2.0 * min(1.0, GIMaskDistance(P, N, Np, Cell, w3 >> 16u));
            if (Score > Best) { Best = Score; First = GIMesh[a + 2u]; Count = w3 & 0xFFFFu; PatchN = Np; }
        }
        return Best > -1e29;
    }
    return false;
}
bool GIReadCellLookup(in const vec3 P, in const vec3 N, out uint First, out uint Count, out vec3 PatchN, out uint Patches, out uint nPatches)
{
    First = 0u; Count = 0u; PatchN = N; Patches = 0u; nPatches = 0u;
    if (GIMesh[3] == 0u) return false;
    const float Cell = uintBitsToFloat(GIMesh[4]);
    const vec3  L    = (P + N * (0.05 * Cell)) / Cell;
    const ivec3 C    = ivec3(floor(L));
    if (GIReadCellAt(C, P, N, First, Count, PatchN, Patches, nPatches)) return true;
    // Not in the bake's cells but within a sample's spacing (an eighth of a cell: surfel_mesh::BuildReadGridAt) of their edge: the cells
    // across it - the bake's points and the pixel's land a hair apart there (a line along a rotated box's cell edges read nothing:
    // COVERAGE's magenta, the Cornell box's white dashes, Tomas 2026-10-30)
    const vec3  F = L - vec3(C);
    const ivec3 D = ivec3(F.x < 0.125 ? -1 : F.x > 0.875 ? 1 : 0, F.y < 0.125 ? -1 : F.y > 0.875 ? 1 : 0, F.z < 0.125 ? -1 : F.z > 0.875 ? 1 : 0);
    if (D == ivec3(0)) return false;
    for (int k = 1; k < 8; ++k)
    {
        const ivec3 O = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1) * D;
        if (O == ivec3(0) || ((k & 1) != 0 && D.x == 0) || ((k & 2) != 0 && D.y == 0) || ((k & 4) != 0 && D.z == 0)) continue;
        if (GIReadCellAt(C + O, P, N, First, Count, PatchN, Patches, nPatches)) return true;
    }
    return false;
}

// The last lookup (a pixel's reads ask for the same point: the per-pixel read, then its fallback): the patch choice scores every patch of
// the cell, the most of a dense mesh's read (the Golem: 30 ms of 34 at cap 32, maximized)
bool  g_GICellValid = false, g_GICellHit;
vec3  g_GICellP, g_GICellN, g_GICellPatchN;
uint  g_GICellFirst, g_GICellCount, g_GICellPatches, g_GICellnPatches;
bool GIReadCell(in const vec3 P, in const vec3 N, out uint First, out uint Count, out vec3 PatchN, out uint Patches, out uint nPatches)
{
    if (!(g_GICellValid && P == g_GICellP && N == g_GICellN))
    {
        g_GICellHit = GIReadCellLookup(P, N, g_GICellFirst, g_GICellCount, g_GICellPatchN, g_GICellPatches, g_GICellnPatches);
        g_GICellValid = true; g_GICellP = P; g_GICellN = N;
    }
    First = g_GICellFirst; Count = g_GICellCount; PatchN = g_GICellPatchN; Patches = g_GICellPatches; nPatches = g_GICellnPatches;
    return g_GICellHit;
}

// A part's list (surfel_mesh::PackReadGrid): its count word (header word 3, low 16 bits) is 0x8000 | its entries with corner weights | its
// all-seen ones << 7 (16 bits each, two to a word, after the others), else the count of entries as they are. At: the list's first word
uint GIListCount(in const uint Code) { return (Code & 0x8000u) != 0u ? (Code & 0x7Fu) + ((Code >> 7u) & 0x7Fu) : Code; }
uint GIListEntry(in const uint At, in const uint Code, in const uint i)
{
    const uint nF = (Code & 0x8000u) != 0u ? (Code & 0x7Fu) : 0xFFFFu;
    if (i < nF) return GIMesh[At + i];
    const uint h = i - nF, w = GIMesh[At + nF + (h >> 1u)];
    return ((h & 1u) == 0u ? (w & 0xFFFFu) : (w >> 16u)) | 0xFFFF0000u;
}

// How much the surface at P sees a listed surfel (surfel_mesh::ReadEntryWeight): the entry's 4 corner weights (4 bits each, the cell's
// corners in the patch's plane) blended by P's place in its cell
float GIReadWeight(in const uint e, in const vec3 P, in const vec3 N, in const vec3 PatchN)
{
    const float Cell = uintBitsToFloat(GIMesh[4]);
    const vec3  Lp   = (P + N * (0.05 * Cell)) / Cell;
    const vec3  A    = abs(PatchN);
    const vec2  uv   = A.x >= A.y && A.x >= A.z ? Lp.yz : A.y >= A.z ? Lp.xz : Lp.xy;
    const vec2  f    = uv - floor(uv);
    const vec4  w    = vec4(float((e >> 16u) & 15u), float((e >> 20u) & 15u), float((e >> 24u) & 15u), float((e >> 28u) & 15u));
    return mix(mix(w.x, w.y, f.x), mix(w.z, w.w, f.x), f.y) / 15.0;
}

// The debug view of the surfels: the level 0 surfel with the most weight at P (0xFFFFFFFF: none reaches it), and a color of its own
uint SurfelGIDominant(in const vec3 P, in const vec3 N)
{
    uint First, Count, Best = 0xFFFFFFFFu, Patches, nPatches;
    float BestW = 0.0;
    vec3 PatchN;
    GIReadCell(P, N, First, Count, PatchN, Patches, nPatches);
    for (uint i = 0u; i < GIListCount(Count); ++i)
    {
        const uint  e = GIListEntry(GIMesh[5] + First, Count, i), s = e & 0xFFFFu;       // the surfel, and how much the surface sees it (GIReadWeight)
        const uint  b = GIMesh[0] + 8u * s;
        const float c = dot(GISurfelNormal(b), N);
        const float f = max(0.0, 1.0 - length(P - GISurfelCenter(b)) / (2.0 * uintBitsToFloat(GIMesh[b + 3u])));
        const float w = c < 0.5 ? 0.0 : GIReadKernel(f) * c * GIReadWeight(e, P, N, PatchN);
        if (w > BestW) { BestW = w; Best = s; }
    }
    return Best;
}
// The debug view of the surface colors (scene::gi_debug::SURFACE_COLOR): the captured color of the pixel's triangle, its cluster's first
// triangle (the vertex shader passes the cluster in the GI reference's x) + gl_PrimitiveID; black: not captured yet
// The COVERAGE view's why for a pixel no read supports (debug only): magenta no cell or no part facing it, purple its part lists nothing,
// red its list has surfels but none weighs anything (out of reach, facing away, invalid, out of the corners' sight)
vec3 SurfelGIWhyUnsupported(in const vec3 P, in const vec3 N, in const uint LightOffset, in const uint Level)
{
    uint First, Count, Patches, nPatches;
    vec3 PatchN;
    if (!GIReadCell(P, N, First, Count, PatchN, Patches, nPatches)) return vec3(1.0, 0.0, 1.0);
    if (GIListCount(Count) == 0u) return vec3(0.45, 0.0, 0.75);
    return vec3(0.9, 0.05, 0.05);
}
bool SurfelGIIsPatch(in const uint s) { return s != 0xFFFFFFFFu && (GIMesh[GIMesh[0] + 8u * s + 7u] >> 31u) != 0u; }     // (PatchDefects: COVERAGE's yellow)
vec3 SurfelGITriangleColor(in const uint Cluster, in const uint Primitive)
{
    const uint t = GIMesh[7];
    if (t == 0u || Cluster >= GIMesh[t]) return vec3(0.0);
    return unpackUnorm4x8(GIMesh[t + 1u + GIMesh[t] + GIMesh[t + 1u + Cluster] + Primitive]).rgb;
}
vec3 SurfelDebugColor(in const uint s)
{
    uint h = s * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return vec3(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u)) / 255.0 * 0.85 + 0.15;
}

// sum(w E), sum(w) of the level 0 surfels that reach a point (mesh space position and normal): its cell of the read grid lists them.
// The weight of the vertex read (falloff over 2 radii squared, times the normal match, times the validity). 0: no grid, or off the
// surface's cells (the caller falls back to the vertex read)
// Reflections (gi_dynamic_master_plan.md, reflections layer 1): with g_GIReflDir set (mesh space, unit; 0: none) the read also weighs the
// same surfels' basis toward it - the light around the reflection direction, the rough reflection's (SurfelGIQuery: g_GIRefl)
vec3 g_GIReflDir = vec3(0.0);
vec4 g_GIRefl    = vec4(0.0);           // w > 0.5: g_GIReflDir's read (an irradiance, as g_GI's), for EnvLighting
vec3 g_GIReflSum = vec3(0.0), g_GIReflRelaxed = vec3(0.0);

// The probe grid's irradiance at a point (the eye's world: the scene pass's) and normal (xlionrender_gi_probes_comp.glsl's buffer, bound as GIMesh): its cell's 8
// probes, trilinear, each weighed by how much it faces the normal (a probe behind the surface lights it less), the ones inside a wall
// or of another cell (not yet updated after the grid moved) left out. w 0: none (the caller keeps its ambient)
vec4 SurfelGIProbes(in const vec3 P, in const vec3 N)
{
    if (GIMesh.length() < 16) return vec4(0.0);
    const ivec3 Org = ivec3(int(GIMesh[0]), int(GIMesh[1]), int(GIMesh[2])), Dim = ivec3(int(GIMesh[4]), int(GIMesh[5]), int(GIMesh[6]));
    const float Sp = uintBitsToFloat(GIMesh[3]);
    if (Sp <= 0.0 || Dim.x <= 0) return vec4(0.0);
    const vec3  Pw = P + vec3(uintBitsToFloat(GIMesh[8]), uintBitsToFloat(GIMesh[9]), uintBitsToFloat(GIMesh[10]));     // (P: the eye's world; the header: the eye)
    const vec3  F = Pw / Sp, B = floor(F), Fr = F - B;
    vec3 Sum = vec3(0.0); float W = 0.0;
    for (int k = 0; k < 8; ++k)
    {
        const ivec3 O = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1), C = ivec3(B) + O;
        if (any(lessThan(C, Org)) || any(greaterThanEqual(C, Org + Dim))) continue;
        const ivec3 Lc = C - Dim * ivec3(floor(vec3(C) / vec3(Dim)));      // (GLSL's % of a negative is undefined)
        const uint  b = 16u + 16u * uint(Lc.x + Dim.x * (Lc.y + Dim.y * Lc.z));
        if (b + 16u > uint(GIMesh.length())) continue;
        if ((GIMesh[b + 15u] & 3u) != 1u || ivec3(int(GIMesh[b + 12u]), int(GIMesh[b + 13u]), int(GIMesh[b + 14u])) != C) continue;
        const vec3  T = mix(vec3(1.0) - Fr, Fr, vec3(O));
        const vec3  Dp = vec3(C) * Sp - Pw;
        const float Face = dot(Dp, Dp) > 1e-8 ? 0.5 * (dot(normalize(Dp), N) + 1.0) : 1.0;
        const float w = T.x * T.y * T.z * (Face * Face + 0.05);
        // the irradiance of its L1 radiance toward N (the cosine lobe: pi, 2 pi / 3)
        const vec3 c0 = vec3(uintBitsToFloat(GIMesh[b]),      uintBitsToFloat(GIMesh[b + 1u]),  uintBitsToFloat(GIMesh[b + 2u]));
        const vec3 c1 = vec3(uintBitsToFloat(GIMesh[b + 3u]), uintBitsToFloat(GIMesh[b + 4u]),  uintBitsToFloat(GIMesh[b + 5u]));
        const vec3 c2 = vec3(uintBitsToFloat(GIMesh[b + 6u]), uintBitsToFloat(GIMesh[b + 7u]),  uintBitsToFloat(GIMesh[b + 8u]));
        const vec3 c3 = vec3(uintBitsToFloat(GIMesh[b + 9u]), uintBitsToFloat(GIMesh[b + 10u]), uintBitsToFloat(GIMesh[b + 11u]));
        const vec3 E = max(vec3(0.0), 3.14159265 * 0.282095 * c0 + 2.0943951 * 0.488603 * (c1 * N.x + c2 * N.y + c3 * N.z));
        Sum += E * w; W += w;
    }
    return W > 1e-4 ? vec4(Sum / W, 1.0) : vec4(0.0);
}

vec4 SurfelGIList(in const vec3 P, in const vec3 N, in const vec3 Nmap, in const uint LightOffset, in const uint Level, in const uint First, in const uint Count, in const vec3 PatchN, out vec4 Relaxed)
{
    vec4 Sum = vec4(0.0);
    Relaxed = vec4(0.0);                                            // the same without the moments' test (SurfelGIQuery's second tier)
    g_GIReflSum = vec3(0.0); g_GIReflRelaxed = vec3(0.0);
    const bool bRefl = dot(g_GIReflDir, g_GIReflDir) > 0.5;
    for (uint i = 0u; i < min(GIListCount(Count), g_GIPixelCap); ++i)            // (the lists are nearest first: surfel_mesh::SortReadLists)
    {
        const uint  e = GIListEntry(GIMesh[5] + First, Count, i), s = e & 0xFFFFu;       // the surfel, and how much the surface sees it (GIReadWeight)
        const uint  b = GIMesh[0] + 8u * s;
        const float c = dot(GISurfelNormal(b), N);
        if (c < 0.5) continue;
        const float f = max(0.0, 1.0 - length(P - GISurfelCenter(b)) / (2.0 * uintBitsToFloat(GIMesh[b + 3u])));
        if (f <= 0.0) continue;
        const float cn = smoothstep(0.5, 0.75, c);                  // fades in: a cut at 0.5 draws a line where smoothed normals sweep through it
        // Level 1 (far): the fine surfels still find what reaches the pixel, their parents give the light (the same coverage at both
        // levels: no pop at the switch)
        const uint  sl = Level == 1u ? (GIMesh[b + 7u] & 0xFFFFu) : s;      // (bit 31: a patch surfel)
        const uint  bl = Level == 1u ? GIMesh[1] + 8u * sl : b;
        const float w0 = GIReadKernel(f) * c * cn * (g_GINoCorners ? 1.0 : GIReadWeight(e, P, N, PatchN)) * GIValidity(LightOffset + sl);
        if (w0 <= 0.0) continue;
        const vec3  B = GIBasisRead(LightOffset + sl, GISurfelNormal(bl), Nmap);           // the normal-mapped normal reads the basis
        const vec3  Br = bRefl ? GIBasisRead(LightOffset + sl, GISurfelNormal(bl), g_GIReflDir) : vec3(0.0);
        Relaxed += vec4(B * w0, w0); g_GIReflRelaxed += Br * w0;
        const float w = w0 * GIVisibility(P, GISurfelCenter(bl), GISurfelNormal(bl), uintBitsToFloat(GIMesh[bl + 3u]), LightOffset + sl);
        if (w <= 0.0) continue;
        Sum += vec4(B * w, w); g_GIReflSum += Br * w;
    }
    return Sum;
}
// A GI read: the irradiance, how much stands behind it, and whether anything does. Unsupported is not dark: no surfel the point may read
// (off the surface's cells, its patch sees none of them, or all of them are out of sight or invalid) says nothing about its light, so its
// irradiance is not a black observation - the caller applies the policy (the materials: the vertex read where that one is supported, else
// no GI at all, the scene's ambient as without GI; the COVERAGE debug view shows which). The confidence is a convention, not an error bound:
// the read's total weight (falloff x normal x visibility x validity of the surfels it blends; about 0.25 and up is a full read)
struct GIQueryResult
{
    vec3  Irradiance;
    float Confidence;
    bool  bSupported;
};
// An instance's mean light (the slot before its first surfel, the renderer's upload)
vec3 SurfelGIMean(in const uint LightOffset) { return UnpackRGB9E5(GILight[LightOffset - 1u].x); }

// What a point no read supports shows (the last of the policy; gi::world::Unsupported the same): the mean light of the valid surfels its
// cell lists (every patch: by their reach and validity only - no sight, no facing), else the instance's mean. Not black, and local: the
// whole building's mean was a grey square in a dark room. Only the pixels that come to it pay for it
// An unsupported point's light (gi::world::Unsupported's twin), bounded: the patch of its cell that matches it (GIReadCell; never the whole
// cell, which mixes layers), its nearest gi_fallback_cap_v surfels (the lists are nearest first: surfel_mesh::SortReadLists), by reach
// and validity, the current light; else the instance's mean
const uint gi_fallback_cap_v = 32u;
vec3 SurfelGILocalMean(in const vec3 P, in const vec3 N, in const uint LightOffset, in const uint Level)
{
    uint First, Count, Patches, nPatches;
    vec3 PatchN;
    vec4 Sum = vec4(0.0);
    if (GIReadCell(P, N, First, Count, PatchN, Patches, nPatches))
        for (uint i = 0u; i < min(GIListCount(Count), gi_fallback_cap_v); ++i)
        {
            const uint  s  = GIListEntry(GIMesh[5] + First, Count, i) & 0xFFFFu;
            const uint  b  = GIMesh[0] + 8u * s;
            const float f  = max(0.0, 1.0 - length(P - GISurfelCenter(b)) / (2.0 * uintBitsToFloat(GIMesh[b + 3u])));
            const uint  sl = Level == 1u ? (GIMesh[b + 7u] & 0xFFFFu) : s;
            const float w  = f * GIValidity(LightOffset + sl);
            if (w <= 0.0) continue;
            const uvec4 L  = GILight[LightOffset + sl];
            Sum += vec4((UnpackRGB9E5(L.x) + UnpackRGB9E5(L.y) + UnpackRGB9E5(L.z)) * (w / 3.0), w);
        }
    return Sum.w > 1e-6 ? Sum.rgb / Sum.w : SurfelGIMean(LightOffset);
}

// The per-pixel read: the surfels of the cell's patch the point matches (GIReadCell), nothing else - no other patch's surfels, no surfels
// out of the patch's sight (those were the fail-open fallbacks: plausible light, from the wrong surface)
GIQueryResult SurfelGIQuery(in const vec3 P, in const vec3 N, in const vec3 Nmap, in const uint LightOffset, in const uint Level)
{
    uint First, Count, Patches, nPatches;
    vec3 PatchN;
    GIQueryResult R = GIQueryResult(vec3(0.0), 0.0, false);
    if (g_GIAblateLookup == 2u) { R.bSupported = true; R.Confidence = 1.0; return R; }     // (both: supported, black - the same path after)
    if (!GIReadCell(P, N, First, Count, PatchN, Patches, nPatches)) return R;
    if (g_GIAblateLookup == 1u) { R.bSupported = true; R.Confidence = 1.0; R.Irradiance = vec3(float(First & 1u) * 1e-30); return R; }     // (First used: the lookup kept)
    vec4 Relaxed;
    vec4 Sum = SurfelGIList(P, N, Nmap, LightOffset, Level, First, Count, PatchN, Relaxed);
    // The moments' test rejects every surfel (next to a wall's foot: their rays met the wall just past the point): its own surfels without
    // it - the corner weights (the bake's line of sight, the patch split at walls) still keep the other side out. Not the mean light: a
    // whole building's in a dark closet was a bright patch
    if (Sum.w <= 1e-6) { Sum = Relaxed; g_GIReflSum = g_GIReflRelaxed; }
    if (Sum.w <= 1e-6) return R;
    R.Irradiance = Sum.rgb / Sum.w; R.Confidence = Sum.w; R.bSupported = true;
    if (dot(g_GIReflDir, g_GIReflDir) > 0.5) g_GIRefl = vec4(g_GIReflSum / Sum.w, 1.0);
    return R;
}

#endif
