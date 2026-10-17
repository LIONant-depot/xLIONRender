#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require
// Part II, G2 (documentation/gi_dynamic_master_plan.md): a surfel pass on the GPU. Each work item's rays (gi::world::TraceChunk's) are
// traced by gi_lanes_v invocations, which write what TraceChunk accumulates before it finishes the pass (the light's sum, its share per basis
// direction, the depth moments' sums, the back-face and hit counts). Every function mirrors its CPU twin in xlionrender_surfel_gi.h; the
// data is gi::world::PackGPUSurfels's (its layout is written there) and PackGPU's
layout(local_size_x = 64) in;                                // (gi::world::gpu_group_v)
const int gi_lanes_v = 8;                                    // invocations per item (gi::world::gpu_lanes_v; the subgroup has 8 at least)

struct GINode     { vec4 m_B[6]; ivec4 m_Child; };
struct GITri4     { vec4 m_V0[3]; vec4 m_E1[3]; vec4 m_E2[3]; ivec4 m_Tri; };
struct GIInstance { vec4 m_W2L[3]; ivec4 m_Info; };
struct GIItem     { ivec4 m_Ids; vec4 m_Offsets; };          // instance, surfel, level, rays; off1, off2, shift (bits), -
struct GIResult   { float m_V[72]; };                         // sum 3, basis 9, moments 48, back, back all, unsupported, hits (int bits: gi::world::gpu_result), unsupported at 3, -

layout(std430, set = 0, binding = 0) readonly  buffer nodes_b     { GINode     GIBvhNodes[]; };
layout(std430, set = 0, binding = 1) readonly  buffer tri4_b      { GITri4     GIBvhTri4[]; };
layout(std430, set = 0, binding = 2) readonly  buffer instances_b { GIInstance GIBvhInstances[]; };
layout(std430, set = 0, binding = 3) readonly  buffer tlas_b      { int        GIBvhTlasLeaves[]; };
layout(std430, set = 0, binding = 4) readonly  buffer data_b      { uint       GD[]; };
layout(std430, set = 0, binding = 5) readonly  buffer items_b     { GIItem     GIItems[]; };
layout(std430, set = 0, binding = 6) writeonly buffer results_b   { GIResult   GIResults[]; };

layout(push_constant) uniform push_t { int m_nItems, m_TlasRoot, m_PrimOffset, m_First; } Push;     // (m_First: this dispatch's first item, a slice of the batch)

const uint  gi_none_v = 0xFFFFFFFFu;
const uint  gi_inst_v = 47u;                 // gi::world::gpu_inst_v
const float gi_pi_v   = 3.14159265;

float GF(uint i) { return uintBitsToFloat(GD[i]); }
vec3  GV(uint i) { return vec3(uintBitsToFloat(GD[i]), uintBitsToFloat(GD[i + 1u]), uintBitsToFloat(GD[i + 2u])); }

uint  InstRec(int i)  { return GD[0] + uint(i) * gi_inst_v; }
uint  MeshOf(uint R)  { return GD[R]; }

bool GISeeThroughBits(int Inst, int Tri)
{
    const uint R = InstRec(Inst), At = GD[R + 14u];
    return At != gi_none_v && ((GD[At + uint(Tri) / 32u] >> (uint(Tri) % 32u)) & 1u) != 0u;
}
#define GI_BVH_SEE_THROUGH
bool GISeeThrough(int Inst, int Tri) { return GISeeThroughBits(Inst, Tri); }
#include "xlionrender_gi_bvh.glsl"
int g_Shadows = 0, g_ReadEntries = 0, g_Probes = 0;      // (diagnostics, with g_GIBvhVisits: gi::world::gpu_result::m_Work)

// ---- an instance's placement
vec3 ToWorld(uint R, vec3 P)  { return vec3(dot(GV(R + 16u), P) + GF(R + 19u), dot(GV(R + 20u), P) + GF(R + 23u), dot(GV(R + 24u), P) + GF(R + 27u)); }
vec3 ToMesh(uint R, vec3 P)   { return vec3(dot(GV(R + 28u), P) + GF(R + 31u), dot(GV(R + 32u), P) + GF(R + 35u), dot(GV(R + 36u), P) + GF(R + 39u)); }
vec3 DirToWorld(uint R, vec3 D) { return vec3(dot(GV(R + 16u), D), dot(GV(R + 20u), D), dot(GV(R + 24u), D)); }
vec3 DirToMesh(uint R, vec3 D)  { return vec3(dot(GV(R + 28u), D), dot(GV(R + 32u), D), dot(GV(R + 36u), D)); }
vec3 NormalToWorld(uint R, vec3 N)      // world::NormalToWorld: the world-to-mesh transposed
{
    return normalize(vec3(GF(R + 28u) * N.x + GF(R + 32u) * N.y + GF(R + 36u) * N.z
                        , GF(R + 29u) * N.x + GF(R + 33u) * N.y + GF(R + 37u) * N.z
                        , GF(R + 30u) * N.x + GF(R + 34u) * N.y + GF(R + 38u) * N.z));
}

// ---- a mesh
vec3  MPos(uint M, uint v)    { return GV(GD[M + 1u] + 3u * v); }
vec3  MNrm(uint M, uint v)    { return GV(GD[M + 2u] + 3u * v); }
uvec3 MTri(uint M, int t)     { const uint b = GD[M + 3u] + 3u * uint(t); return uvec3(GD[b], GD[b + 1u], GD[b + 2u]); }
int   MVertSurfel(uint M, uint v, int k) { const uint w = GD[GD[M + 4u] + 2u * v + uint(k / 2)]; const uint s = (k % 2) == 0 ? (w & 0xFFFFu) : (w >> 16); return s == 0xFFFFu ? -1 : int(s); }
uint  MSurfel(uint M, uint Level, int s) { return GD[M + 5u + 2u * Level] + 7u * uint(s); }      // center 3, normal 3, radius
int   MParent(uint M, int s)  { return int(GD[GD[M + 9u] + uint(s)]); }

// ---- the solver's helpers
vec3 StratifiedCosineDir(vec3 N, int i, int R, float Off1, float Off2)
{
    const float u1 = fract((float(i) + 0.5) / float(R) + Off1), u2 = fract(float(i) * 0.61803398875 + Off2);
    const float r = sqrt(u1), phi = 2.0 * gi_pi_v * u2;
    const vec3 T = normalize(abs(N.x) > 0.5 ? cross(N, vec3(0, 1, 0)) : cross(N, vec3(1, 0, 0)));
    const vec3 B = cross(N, T);
    return normalize(T * (r * cos(phi)) + B * (r * sin(phi)) + N * sqrt(max(0.0, 1.0 - r * r)));
}
int MomentTexel(vec3 N, vec3 D)
{
    const vec3 T = normalize(cross(abs(N.x) < 0.9 ? vec3(1, 0, 0) : vec3(0, 1, 0), N)), B = cross(N, T);
    const float x = dot(D, T), y = dot(D, B), z = max(dot(D, N), 0.0);
    const float k = 1.0 / max(abs(x) + abs(y) + z, 1e-6);
    const float u = (x + y) * k, v = (x - y) * k;
    const int tu = clamp(int((u * 0.5 + 0.5) * 4.0), 0, 3), tv = clamp(int((v * 0.5 + 0.5) * 4.0), 0, 3);
    return tv * 4 + tu;
}
void BasisDirs(vec3 N, out vec3 D0, out vec3 D1, out vec3 D2)
{
    const vec3 T = normalize(cross(abs(N.x) < 0.9 ? vec3(1, 0, 0) : vec3(0, 1, 0), N)), B = cross(N, T);
    const float a = 0.40824829, b = 0.70710678, c = 0.57735027, d = 0.81649658;
    D0 = T * -a + B * b + N * c; D1 = T * -a + B * -b + N * c; D2 = T * d + N * c;
}
vec3 BasisWeights(vec3 D0, vec3 D1, vec3 D2, vec3 n)
{
    vec3 w = vec3(max(dot(D0, n), 0.0), max(dot(D1, n), 0.0), max(dot(D2, n), 0.0)); w *= w;
    const float Sum = w.x + w.y + w.z;
    return Sum > 1e-12 ? w / Sum : vec3(1.0 / 3.0);
}
vec3 SkyAt(vec3 D)
{
    const uint Lt = GD[2];
    if (GD[4] == 0u) return GV(Lt + 6u);
    const float x0 = D.x, y0 = D.z, z0 = -D.y;
    const float k = 1.0 / max(abs(x0) + abs(y0) + abs(z0), 1e-12);
    const float x = x0 * k, y = y0 * k, z = z0 * k;
    const float u = z >= 0.0 ? x : (1.0 - abs(y)) * (x >= 0.0 ? 1.0 : -1.0);
    const float v = z >= 0.0 ? y : (1.0 - abs(x)) * (y >= 0.0 ? 1.0 : -1.0);
    const float fx = clamp((u * 0.5 + 0.5) * 4.0 - 0.5, 0.0, 3.0), fy = clamp((v * 0.5 + 0.5) * 4.0 - 0.5, 0.0, 3.0);
    const int ix = min(int(fx), 2), iy = min(int(fy), 2);
    const float tx = fx - float(ix), ty = fy - float(iy);
    const uint O = Lt + 9u;
    const vec3 T00 = GV(O + 3u * uint(iy * 4 + ix)), T10 = GV(O + 3u * uint(iy * 4 + ix + 1)), T01 = GV(O + 3u * uint((iy + 1) * 4 + ix)), T11 = GV(O + 3u * uint((iy + 1) * 4 + ix + 1));
    return (T00 * (1.0 - tx) + T10 * tx) * (1.0 - ty) + (T01 * (1.0 - tx) + T11 * tx) * ty;
}
bool Occluded(vec3 O, vec3 D, float TMax)       // world::Occluded: the first hit ends it (see-through triangles block: as the CPU's shadow rays)
{
    g_GIBvhSeeThrough = false; g_GIBvhAnyHit = true; ++g_Shadows;
    const gi_hit H = GICast(O, D, TMax);
    g_GIBvhSeeThrough = true; g_GIBvhAnyHit = false;
    return H.m_Inst >= 0;
}
vec3 Direct(vec3 P, vec3 N)
{
    const uint Lt = GD[2];
    vec3 E = vec3(0);
    const vec3 O = P + N * 1e-3;
    const vec3 SunD = GV(Lt), SunC = GV(Lt + 3u);
    const float cs = -dot(N, SunD);
    if (cs > 0.0 && (SunC.x + SunC.y + SunC.z) > 0.0 && !Occluded(O, SunD * -1.0, 1e30)) E += SunC * cs;
    const uint nP = GD[3];
    for (uint i = 0u; i < nP; ++i)
    {
        const uint b = Lt + 57u + 7u * i;
        const vec3 D = GV(b) - P;
        const float d = length(D), s = d / GF(b + 6u);
        if (s >= 1.0 || d < 1e-6) continue;
        const float c = dot(N, D) / d;
        if (c <= 0.0) continue;
        if (Occluded(O, D * (1.0 / d), d - 2e-3)) continue;
        E += GV(b + 3u) * ((1.0 - s * s * s * s) / (1.0 + 2.0 * s * s) * c);
    }
    return E;
}

// ---- the reads at a hit (world::Indirect, ReadPoint, ReadVertex, Visibility, Unsupported, SurfaceAt)
float Visibility(uint R, uint M, uint Level, int s, vec3 P)
{
    const uint Sb = MSurfel(M, Level, s);
    const vec3 C = GV(Sb), N = GV(Sb + 3u);
    const vec3 Dv = P - C;
    const float Reach = 2.0 * GF(Sb + 6u);
    float w = clamp(1.0 - abs(dot(Dv, N)) / (0.7 * Reach), 0.0, 1.0);
    if (w <= 0.0) return w;
    const vec3 Q = Dv + N * (0.15 * Reach);
    const float d = length(Q);
    const vec3 U = Q * (1.0 / d);
    vec3 T = abs(N.x) < 0.9 ? vec3(0, -N.z, N.y) : vec3(N.z, 0, -N.x);
    T = T * (1.0 / length(T));
    const vec3 B = cross(N, T);
    const float x = dot(U, T), y = dot(U, B), z = max(dot(U, N), 0.0);
    const float k = 1.0 / max(abs(x) + abs(y) + z, 1e-6);
    const float u = clamp(((x + y) * k * 0.5 + 0.5) * 4.0 - 0.5, 0.0, 3.0);
    const float vv = clamp(((x - y) * k * 0.5 + 0.5) * 4.0 - 0.5, 0.0, 3.0);
    const int u0 = int(u), v0 = int(vv), u1 = min(u0 + 1, 3), v1 = min(v0 + 1, 3);
    const float fu = u - float(u0), fv = vv - float(v0);
    const uint Mo = GD[R + 4u] + 32u * uint(s);
    const float a00 = (1.0 - fu) * (1.0 - fv), a10 = fu * (1.0 - fv), a01 = (1.0 - fu) * fv, a11 = fu * fv;
    const uint t00 = 2u * uint(v0 * 4 + u0), t10 = 2u * uint(v0 * 4 + u1), t01 = 2u * uint(v1 * 4 + u0), t11 = 2u * uint(v1 * 4 + u1);
    const float Mean = GF(Mo + t00) * a00 + GF(Mo + t10) * a10 + GF(Mo + t01) * a01 + GF(Mo + t11) * a11;
    const float M2   = GF(Mo + t00 + 1u) * a00 + GF(Mo + t10 + 1u) * a10 + GF(Mo + t01 + 1u) * a01 + GF(Mo + t11 + 1u) * a11;
    const float Var  = max(abs(Mean * Mean - M2), 0.0025 * Reach * Reach);
    if (d > Mean) { const float r = Var / (Var + (d - Mean) * (d - Mean) + 1e-12); w *= r * r * r; }
    return w;
}
vec3 UnpackOct(uint W, out bool bOk)
{
    const float x0 = max(float(bitfieldExtract(int(W), 0, 16)) / 32767.0, -1.0), y0 = max(float(bitfieldExtract(int(W), 16, 16)) / 32767.0, -1.0);
    vec3 V = vec3(x0, y0, 1.0 - abs(x0) - abs(y0));
    if (V.z < 0.0) { V.x = (1.0 - abs(y0)) * (x0 >= 0.0 ? 1.0 : -1.0); V.y = (1.0 - abs(x0)) * (y0 >= 0.0 ? 1.0 : -1.0); }
    bOk = true;
    return V * (1.0 / sqrt(dot(V, V)));
}
float MaskDistance(uint Mask, vec2 uv)
{
    const int su = clamp(int(uv.x * 4.0), 0, 3), sv = clamp(int(uv.y * 4.0), 0, 3);
    if ((Mask & (1u << uint(sv * 4 + su))) != 0u) return 0.0;
    float Best = 1e30;
    for (int b = 0; b < 16; ++b)
    {
        if ((Mask & (1u << uint(b))) == 0u) continue;
        const float u0 = float(b % 4) / 4.0, v0 = float(b / 4) / 4.0, e = 0.25;
        const float du = max(max(u0 - uv.x, uv.x - (u0 + e)), 0.0), dv = max(max(v0 - uv.y, uv.y - (v0 + e)), 0.0);
        Best = min(Best, max(du, dv) * 4.0);
    }
    return Best;
}
ivec3 ReadAxes(vec3 N)
{
    const float x = abs(N.x), y = abs(N.y), z = abs(N.z);
    return x >= y && x >= z ? ivec3(1, 2, 0) : y >= z ? ivec3(0, 2, 1) : ivec3(0, 1, 2);
}
// The read grid's cell of P (lifted along N), its table entry (-1: none)
int ReadEntry(uint M, vec3 P, vec3 N)
{
    const uint Size = GD[M + 11u];
    if (Size == 0u) return -1;
    const float Cell = GF(M + 13u);
    const vec3 L = (P + N * (0.05 * Cell)) / Cell;
    const ivec3 C = ivec3(floor(L));
    const uint x21 = uint(C.x) & 0x1FFFFFu, y21 = uint(C.y) & 0x1FFFFFu, z21 = uint(C.z) & 0x1FFFFFu;
    const uint KLo = (y21 << 21) | z21, KHi = (x21 << 10) | (y21 >> 11);
    const uint Mask = Size - 1u;
    uint h = ((uint(C.x) * 73856093u) ^ (uint(C.y) * 19349663u) ^ (uint(C.z) * 83492791u)) & Mask;
    const uint Tb = GD[M + 10u];
    for (uint n = 0u; n <= Mask; ++n, h = (h + 1u) & Mask)
    {
        ++g_Probes;
        const uint E0 = GD[Tb + 4u * h], E1 = GD[Tb + 4u * h + 1u];
        if (E1 == 0xFFFFFFFFu) return -1;
        if (E0 == KLo && E1 == KHi) return int(h);
    }
    return -1;
}
// surfel_mesh::ReadCellList: the patch of P's cell that matches it best; its list (First, Count) and its normal. false: none
bool ReadCellList(uint M, vec3 P, vec3 N, out uint First, out uint Count, out vec3 PatchN)
{
    First = 0u; Count = 0u; PatchN = N;
    const int h = ReadEntry(M, P, N);
    if (h < 0) return false;
    const uint Tb = GD[M + 10u] + 4u * uint(h), Lb = GD[M + 12u];
    const float Cell = GF(M + 13u), iCell = 1.0 / Cell;
    const vec3 Lp = (P + N * (0.05 * Cell)) * iCell;
    const vec3 Fr = Lp - floor(Lp);
    float Best = -1e30; uint Bp = 0xFFFFFFFFu;
    for (uint p = 0u; p < GD[Tb + 3u]; ++p)
    {
        const uint Pa = Lb + GD[Tb + 2u] + 4u * p;
        bool bOk;
        const vec3 n = UnpackOct(GD[Pa], bOk);
        const float c = dot(n, N);
        if (c < 0.5) continue;
        const float ax = abs(n.x), ay = abs(n.y), az = abs(n.z);
        const int Au = ax >= ay && ax >= az ? 1 : 0, Av = ax >= ay && ax >= az ? 2 : ay >= az ? 2 : 1;
        const vec2 uv = vec2(Fr[Au], Fr[Av]);
        const uint Msk = GD[Pa + 3u] >> 16;
        const int su = min(int(uv.x * 4.0), 3), sv = min(int(uv.y * 4.0), 3);
        const float D = ((Msk >> uint(sv * 4 + su)) & 1u) != 0u ? 0.0 : MaskDistance(Msk, uv);
        const float Score = c - abs(dot(P, n) - uintBitsToFloat(GD[Pa + 1u])) * iCell + 2.0 - 2.0 * min(1.0, D);
        if (Score > Best) { Best = Score; Bp = p; }
    }
    if (Bp == 0xFFFFFFFFu) return false;
    const uint Pa = Lb + GD[Tb + 2u] + 4u * Bp;
    bool bOk;
    PatchN = UnpackOct(GD[Pa], bOk);
    First = GD[Pa + 2u]; Count = GD[Pa + 3u] & 0xFFFFu;
    return true;
}
// world::ReadPoint (P, N mesh space): E, false when unsupported
bool ReadPoint(int Id, vec3 P, vec3 N, out vec3 E)
{
    E = vec3(0);
    const uint R = InstRec(Id), M = MeshOf(R), Level = GD[R + 1u];
    uint First, Count; vec3 PatchN;
    if (!ReadCellList(M, P, N, First, Count, PatchN)) return false;
    const float Cell = GF(M + 13u);
    const vec3 Lp = (P + N * (0.05 * Cell)) / Cell;
    const ivec3 Ax = ReadAxes(PatchN);
    const float fu = Lp[Ax.x] - floor(Lp[Ax.x]), fv = Lp[Ax.y] - floor(Lp[Ax.y]);
    const float k0 = (1.0 - fu) * (1.0 - fv) / 15.0, k1 = fu * (1.0 - fv) / 15.0, k2 = (1.0 - fu) * fv / 15.0, k3 = fu * fv / 15.0;
    const uint nLight = GD[M + 6u + 2u * Level];
    const uint Lb = GD[M + 12u];
    vec3 S = vec3(0), RS = vec3(0); float W = 0.0, RW = 0.0;
    for (uint i = 0u; i < Count; ++i)
    {
        const uint e = GD[Lb + First + i];
        ++g_ReadEntries;
        const int s = int(e & 0xFFFFu);
        const uint Sb = MSurfel(M, 0u, s);
        const float c = dot(GV(Sb + 3u), N);
        if (c < 0.5) continue;
        const float f = 1.0 - length(P - GV(Sb)) / (2.0 * GF(Sb + 6u));
        if (f <= 0.0) continue;
        const int sl = Level == 1u ? MParent(M, s) : s;
        if (uint(sl) >= nLight) continue;
        const float t = min((c - 0.5) * 4.0, 1.0), cn = t * t * (3.0 - 2.0 * t);
        const float Corner = float((e >> 16) & 15u) * k0 + float((e >> 20) & 15u) * k1 + float((e >> 24) & 15u) * k2 + float((e >> 28) & 15u) * k3;
        const float w0 = f * f * c * cn * Corner * GF(GD[R + 3u] + uint(sl));
        if (w0 <= 0.0) continue;
        const vec3 El = GV(GD[R + 2u] + 3u * uint(sl));
        RS += El * w0; RW += w0;
        const float w = w0 * Visibility(R, M, Level, sl, P);
        if (w <= 0.0) continue;
        S += El * w; W += w;
    }
    if (W <= 1e-6) { S = RS; W = RW; }
    if (W <= 1e-6) return false;
    E = S * (1.0 / W);
    return true;
}
vec3 ReadVertex(uint R, uint M, uint Level, uint Vertex, out float W)
{
    const vec3 P = MPos(M, Vertex), N = MNrm(M, Vertex);
    vec3 Sum = vec3(0); W = 0.0;
    int Seen[4]; int nSeen = 0;
    for (int k = 0; k < 4; ++k)
    {
        int s = MVertSurfel(M, Vertex, k);
        if (s < 0) continue;
        if (Level == 1u)
        {
            s = MParent(M, s);
            bool bSeen = false;
            for (int j = 0; j < nSeen; ++j) bSeen = bSeen || Seen[j] == s;
            if (bSeen) continue;
            Seen[nSeen++] = s;
        }
        const uint Sb = MSurfel(M, Level, s);
        const float c = dot(GV(Sb + 3u), N);
        if (c < 0.5) continue;
        const float f = max(0.0, 1.0 - length(DirToWorld(R, P - GV(Sb))) / (2.0 * GF(Sb + 6u) * GF(R + 15u)));
        const float w = (f * f * c + 1e-6) * GF(GD[R + 3u] + uint(s));
        Sum += GV(GD[R + 2u] + 3u * uint(s)) * w;
        W += w;
    }
    return Sum;
}
bool Indirect(int Id, int Tri, float U, float V, out vec3 E)
{
    const uint R = InstRec(Id), M = MeshOf(R), Level = GD[R + 1u];
    const uvec3 T = MTri(M, Tri);
    const float B[3] = float[3](1.0 - U - V, U, V);
    if (GD[M + 17u] != 0u && GD[M + 11u] != 0u)
    {
        const vec3 P0 = MPos(M, T.x), P1 = MPos(M, T.y), P2 = MPos(M, T.z);
        return ReadPoint(Id, P0 * B[0] + P1 * B[1] + P2 * B[2], normalize(cross(P1 - P0, P2 - P0)), E);
    }
    vec3 Sum = vec3(0); float W = 0.0;
    for (int k = 0; k < 3; ++k)
    {
        float w;
        const vec3 Ev = ReadVertex(R, M, Level, T[k], w);
        Sum += Ev * B[k];
        W += w * B[k];
    }
    E = vec3(0);
    if (W <= 0.0) return false;
    E = Sum * (1.0 / W);
    return true;
}
vec3 Unsupported(int Id, int Tri, float U, float V)
{
    const uint R = InstRec(Id), M = MeshOf(R), Level = GD[R + 1u];
    const vec3 Mean = GV(R + 5u);
    if (GD[M + 11u] == 0u) return Mean;
    const uvec3 T = MTri(M, Tri);
    const float B[3] = float[3](1.0 - U - V, U, V);
    const vec3 P0 = MPos(M, T.x), P1 = MPos(M, T.y), P2 = MPos(M, T.z);
    const vec3 P = P0 * B[0] + P1 * B[1] + P2 * B[2], N = normalize(cross(P1 - P0, P2 - P0));
    const int h = ReadEntry(M, P, N);
    if (h < 0) return Mean;
    const uint Tb = GD[M + 10u] + 4u * uint(h), Lb = GD[M + 12u];
    const uint nLight = GD[M + 6u + 2u * Level];
    vec3 S = vec3(0); float W = 0.0;
    for (uint p = 0u; p < GD[Tb + 3u]; ++p)
    {
        const uint Pa = Lb + GD[Tb + 2u] + 4u * p;
        for (uint i = 0u; i < (GD[Pa + 3u] & 0xFFFFu); ++i)
        {
            const int s = int(GD[Lb + GD[Pa + 2u] + i] & 0xFFFFu);
            const uint Sb = MSurfel(M, 0u, s);
            const float f = max(0.0, 1.0 - length(P - GV(Sb)) / (2.0 * GF(Sb + 6u)));
            const int sl = Level == 1u ? MParent(M, s) : s;
            if (uint(sl) >= nLight) continue;
            const float w = f * GF(GD[R + 3u] + uint(sl));
            if (w <= 0.0) continue;
            S += GV(GD[R + 2u] + 3u * uint(sl)) * w; W += w;
        }
    }
    return W > 1e-6 ? S / W : Mean;
}
void SurfaceAt(int Id, int Tri, float U, float V, out vec3 Albedo, out vec3 Emission)
{
    const uint R = InstRec(Id), M = MeshOf(R);
    const bool bTri = GD[R + 11u] != gi_none_v;
    Albedo = bTri ? GV(GD[R + 11u] + 3u * uint(Tri)) : GV(R + 8u);
    Emission = vec3(0);
    const bool bSA = GD[R + 12u] != gi_none_v, bSE = GD[R + 13u] != gi_none_v;
    if (!bSA && !bSE) return;
    const uvec3 T = MTri(M, Tri);
    const float Bc[3] = float[3](1.0 - U - V, U, V);
    const vec3 P = MPos(M, T.x) * Bc[0] + MPos(M, T.y) * Bc[1] + MPos(M, T.z) * Bc[2];
    vec3 SA = vec3(0), SE = vec3(0); float W = 0.0;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 4; ++j)
        {
            const int s = MVertSurfel(M, T[k], j);
            if (s < 0) continue;
            const uint Sb = MSurfel(M, 0u, s);
            const float c = dot(GV(Sb + 3u), MNrm(M, T[k]));
            if (c < 0.5) continue;
            const float f = max(0.0, 1.0 - length(P - GV(Sb)) / (2.0 * GF(Sb + 6u)));
            const float w = (f * f * c + 1e-6) * Bc[k];
            if (bSA) SA += GV(GD[R + 12u] + 3u * uint(s)) * w;
            if (bSE) SE += GV(GD[R + 13u] + 3u * uint(s)) * w;
            W += w;
        }
    if (W <= 0.0) return;
    if (bSA && !bTri) Albedo = SA * (1.0 / W);
    Emission = SE * (GF(5u) / W);
}

// A work item's rays are spread over gi_lanes_v invocations (a cluster of its subgroup): lane l traces rays l, l + 8, ... and the
// cluster adds what they found. One invocation per item traced its rays one after the other: a small batch was bound by the rays'
// latency (Police: 426 items, 3.4k rays in 5 ms)
void main()
{
    const int  k     = Push.m_First + int(gl_GlobalInvocationID.x) / gi_lanes_v;
    const int  Lane  = int(gl_GlobalInvocationID.x) % gi_lanes_v;
    // (an invocation past the items still joins its cluster's sums, with nothing; so does an item whose instance is gone or changed level
    // since the batch was picked: the world drops it)
    const bool bItem = k < Push.m_nItems && GD[InstRec(GIItems[min(k, Push.m_nItems - 1)].m_Ids.x) + 40u] != 0u
                    && GD[InstRec(GIItems[min(k, Push.m_nItems - 1)].m_Ids.x) + 1u] == uint(GIItems[min(k, Push.m_nItems - 1)].m_Ids.z);
    g_GIBvhTlasRoot = Push.m_TlasRoot; g_GIBvhPrimOffset = Push.m_PrimOffset; g_GIBvhSeeThrough = true;
    vec3 Sum = vec3(0), BA0 = vec3(0), BA1 = vec3(0), BA2 = vec3(0), UnsAt = vec3(0);
    float MA[48]; for (int t = 0; t < 48; ++t) MA[t] = 0.0;
    int nBack = 0, nBackAll = 0, nUns = 0, nHits = 0, FirstUns = 0x7FFFFFFF;
    if (bItem)
    {
        const int Id = GIItems[k].m_Ids.x, s = GIItems[k].m_Ids.y, Rn = GIItems[k].m_Ids.w;
        const float Off1 = GIItems[k].m_Offsets.x, Off2 = GIItems[k].m_Offsets.y;
        const int Shift = floatBitsToInt(GIItems[k].m_Offsets.z);
        const uint R = InstRec(Id), M = MeshOf(R), Level = GD[R + 1u];
        const uint Sb = MSurfel(M, Level, s);
        const vec3 Sc = GV(Sb), Sn = GV(Sb + 3u);
        const float Reach = 2.0 * GF(Sb + 6u);
        const vec3 C = ToWorld(R, Sc), N = NormalToWorld(R, Sn);
        vec3 D0, D1, D2; BasisDirs(Sn, D0, D1, D2);
        const bool bSpread = GD[6] != 0u && Level == 0u && GD[M + 14u] != gi_none_v;
        const uint o0 = bSpread ? GD[GD[M + 14u] + uint(s)] : 0u, nO = bSpread ? GD[GD[M + 14u] + uint(s) + 1u] - o0 : 0u;
        const float MeanScale = GF(R + 15u);
        for (int r = Lane; r < Rn; r += gi_lanes_v)
        {
            vec3 Cr = C, Nr = N;
            if (nO > 1u) { const uint o = o0 + uint(r + Shift) % nO; Cr = ToWorld(R, GV(GD[M + 15u] + 3u * o)); Nr = NormalToWorld(R, GV(GD[M + 16u] + 3u * o)); }
            const vec3 D = StratifiedCosineDir(Nr, r, Rn, Off1, Off2);
            const gi_hit H = GICast(Cr + Nr * 1e-3, D, 1e30);
            vec3 Dm = DirToMesh(R, D);
            const float Lm = length(Dm);
            Dm = Dm * (1.0 / Lm);
            const vec3 Pe = ToMesh(R, Cr + D * (H.m_Inst < 0 ? Reach / Lm : H.m_T)) - Sc;
            const float Le = length(Pe);
            const float dd = min(Le, Reach);
            const int t = MomentTexel(Sn, Le > 1e-6 ? Pe * (1.0 / Le) : Dm);
            MA[3 * t] += dd; MA[3 * t + 1] += dd * dd; MA[3 * t + 2] += 1.0;
            const vec3 Bw = BasisWeights(D0, D1, D2, Dm);
            if (H.m_Inst < 0) { const vec3 Sky = SkyAt(D); Sum += Sky; BA0 += Sky * Bw.x; BA1 += Sky * Bw.y; BA2 += Sky * Bw.z; continue; }
            const uint HR = InstRec(H.m_Inst), HM = MeshOf(HR);
            const uvec3 HT = MTri(HM, H.m_Tri);
            const float B0 = 1.0 - H.m_U - H.m_V;
            const vec3 Nh = NormalToWorld(HR, MNrm(HM, HT.x) * B0 + MNrm(HM, HT.y) * H.m_U + MNrm(HM, HT.z) * H.m_V);
            if (dot(Nh, D) > 0.0) { nBack += H.m_T < Reach * MeanScale ? 1 : 0; ++nBackAll; continue; }
            vec3 Albedo, Emission;
            SurfaceAt(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Albedo, Emission);
            vec3 Q;
            const bool bQ = Indirect(H.m_Inst, H.m_Tri, H.m_U, H.m_V, Q);
            const vec3 Ph = ToWorld(HR, MPos(HM, HT.x) * B0 + MPos(HM, HT.y) * H.m_U + MPos(HM, HT.z) * H.m_V);
            if (!bQ) { if (nUns == 0) { UnsAt = Ph; FirstUns = r; } ++nUns; }
            ++nHits;
            const vec3 In = Direct(Ph, Nh) + (bQ ? Q : Unsupported(H.m_Inst, H.m_Tri, H.m_U, H.m_V));
            const vec3 Lr = Albedo * In * (1.0 / gi_pi_v) + (GD[7] != 0u ? Emission : vec3(0));
            Sum += Lr; BA0 += Lr * Bw.x; BA1 += Lr * Bw.y; BA2 += Lr * Bw.z;
        }
    }
    // the cluster's sums (the first unsupported ray's hit: the lane with the item's lowest such ray)
    Sum = subgroupClusteredAdd(Sum, gi_lanes_v); BA0 = subgroupClusteredAdd(BA0, gi_lanes_v); BA1 = subgroupClusteredAdd(BA1, gi_lanes_v); BA2 = subgroupClusteredAdd(BA2, gi_lanes_v);
    for (int t = 0; t < 48; ++t) MA[t] = subgroupClusteredAdd(MA[t], gi_lanes_v);
    nBack = subgroupClusteredAdd(nBack, gi_lanes_v); nBackAll = subgroupClusteredAdd(nBackAll, gi_lanes_v);
    nUns = subgroupClusteredAdd(nUns, gi_lanes_v); nHits = subgroupClusteredAdd(nHits, gi_lanes_v);
    const int First = subgroupClusteredMin(FirstUns, gi_lanes_v);
    UnsAt = subgroupClusteredAdd(FirstUns == First && First != 0x7FFFFFFF ? UnsAt : vec3(0), gi_lanes_v);
    const ivec4 Work = subgroupClusteredAdd(ivec4(g_GIBvhVisits, g_Shadows, g_ReadEntries, g_Probes), gi_lanes_v);
    if (!bItem || Lane != 0) return;
    GIResults[k].m_V[0] = Sum.x; GIResults[k].m_V[1] = Sum.y; GIResults[k].m_V[2] = Sum.z;
    GIResults[k].m_V[3] = BA0.x; GIResults[k].m_V[4] = BA0.y; GIResults[k].m_V[5] = BA0.z;
    GIResults[k].m_V[6] = BA1.x; GIResults[k].m_V[7] = BA1.y; GIResults[k].m_V[8] = BA1.z;
    GIResults[k].m_V[9] = BA2.x; GIResults[k].m_V[10] = BA2.y; GIResults[k].m_V[11] = BA2.z;
    for (int t = 0; t < 48; ++t) GIResults[k].m_V[12 + t] = MA[t];
    GIResults[k].m_V[60] = intBitsToFloat(nBack); GIResults[k].m_V[61] = intBitsToFloat(nBackAll); GIResults[k].m_V[62] = intBitsToFloat(nUns); GIResults[k].m_V[63] = intBitsToFloat(nHits);
    GIResults[k].m_V[64] = UnsAt.x; GIResults[k].m_V[65] = UnsAt.y; GIResults[k].m_V[66] = UnsAt.z; GIResults[k].m_V[67] = 0.0;
    for (int w = 0; w < 4; ++w) GIResults[k].m_V[68 + w] = intBitsToFloat(Work[w]);
}
