#version 450
// Point and spot light shadow pages: the request pass (features/shadows/xlionrender_shadow_local_pages.h; documentation/lighting_strategy.md
// P1d). Per pixel of the main view's depth and per shadowed light that reaches it: the face it is seen through, its footprint (the world
// distance to its neighbours, as the shading's derivatives) against the face's texel at that depth - the level whose texel fits it (as
// LocalShadow picks: xlionrender_local_lights.glsl) -, and a bit for every page its filter's taps fall in (the 3 x 3 taps and the
// normal offset). Then what the shading finds in this frame's table: counters for the report. The CPU reads both a ring late.
layout(local_size_x = 8, local_size_y = 8) in;

const int kLevels      = 5;         // local_pages::levels_v
const int kLevel0Pages = 16;        // local_pages::level0_pages_v
const int kPageTexels  = 128;
const int kFaceEntries = 341;       // local_pages::face_entries_v
const int kCounters    = 32;        // after the bits: 0 pixel-light pairs, 1 shaded by their level, 2 by a coarser, 3 by none, 5-9 per level
const int kMaxLights   = 32;

struct light_t
{
    vec4 PosRadius;                 // world position, radius
    vec4 DirTan;                    // spot: direction, tan of the half field of view; point: w 0
    vec4 Info;                      // x: the first face's table entry, y: near
};
struct frame_t
{
    mat4    C2W;                    // the main view's clip -> camera-centred world
    vec4    Eye;                    // the camera's world position
    vec4    P0;                     // x: lights, y: the resolution bias (levels), z: the table's first uint in Cells
    light_t Lights[kMaxLights];
};
layout(std430, set = 0, binding = 0) readonly buffer frame_b { frame_t F[]; };
layout(std430, set = 0, binding = 1) coherent buffer req_b  { uint Req[]; };
layout(std430, set = 0, binding = 2) readonly buffer cell_b { uint Cells[]; };
layout(set = 1, binding = 0) uniform sampler2D Depth;
layout(push_constant) uniform push_t { uint m_Frame, m_ReqBase, m_W, m_H, m_BitUints, m_Pad0, m_Pad1, m_Pad2; } Push;

shared uint s_Count[kCounters];

bool World(ivec2 p, out vec3 P)
{
    if (any(lessThan(p, ivec2(0))) || p.x >= int(Push.m_W) || p.y >= int(Push.m_H)) return false;
    const float d = texelFetch(Depth, p, 0).r;
    if (d >= 1.0) return false;
    const vec2 Ndc = (vec2(p) + 0.5) / vec2(Push.m_W, Push.m_H) * 2.0 - 1.0;
    const vec4 W = F[Push.m_Frame].C2W * vec4(Ndc, d, 1.0);
    P = W.xyz / W.w + F[Push.m_Frame].Eye.xyz;
    return true;
}

float AxisFootprint(ivec2 p, ivec2 Step, vec3 P)
{
    vec3 A, B;
    const bool bA = World(p + Step, A), bB = World(p - Step, B);
    return min(bA ? length(A - P) : 1e30, bB ? length(P - B) : 1e30);
}

// A face's basis (shadows::FaceBasis) and its axis for a point light's direction D (shadows::CubeFaceAxis order)
void FaceOf(vec3 D, out int Face, out vec3 Axis)
{
    const vec3 A = abs(D);
    if (A.x >= A.y && A.x >= A.z) { Face = D.x >= 0.0 ? 0 : 1; Axis = vec3(D.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0); }
    else if (A.y >= A.z)          { Face = D.y >= 0.0 ? 2 : 3; Axis = vec3(0.0, D.y >= 0.0 ? 1.0 : -1.0, 0.0); }
    else                          { Face = D.z >= 0.0 ? 4 : 5; Axis = vec3(0.0, 0.0, D.z >= 0.0 ? 1.0 : -1.0); }
}
void Basis(vec3 Fw, out vec3 R, out vec3 U)
{
    const vec3 Up = abs(Fw.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    R = normalize(cross(Up, Fw));
    U = cross(Fw, R);
}

uint LevelBase(int L) { uint b = 0u; for (int l = 0; l < L; ++l) b += uint((kLevel0Pages >> l) * (kLevel0Pages >> l)); return b; }

uint Entry(uint i) { return (Cells[uint(F[Push.m_Frame].P0.z) + (i >> 1)] >> ((i & 1u) * 16u)) & 0xFFFFu; }

void main()
{
    if (gl_LocalInvocationIndex < kCounters) s_Count[gl_LocalInvocationIndex] = 0u;
    barrier();

    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    vec3 P;
    if (World(p, P))
    {
        const float fx = AxisFootprint(p, ivec2(1, 0), P), fy = AxisFootprint(p, ivec2(0, 1), P);
        const float Foot = max(fx < 1e29 ? fx : 0.0, fy < 1e29 ? fy : 0.0);
        const int nLights = int(F[Push.m_Frame].P0.x);
        for (int li = 0; li < nLights; ++li)
        {
            const light_t Lt = F[Push.m_Frame].Lights[li];
            const vec3  D    = P - Lt.PosRadius.xyz;
            const float Dist = length(D);
            if (Dist >= Lt.PosRadius.w || Dist < 1e-6) continue;
            int  Face = 0;
            vec3 Axis;
            float TanHalf = 1.0;
            if (Lt.DirTan.w > 0.0) { Axis = Lt.DirTan.xyz; TanHalf = Lt.DirTan.w; }
            else FaceOf(D, Face, Axis);
            vec3 R, U;
            Basis(Axis, R, U);
            const float d = dot(D, Axis);
            if (d <= Lt.Info.y) continue;
            const vec2 UV = vec2(dot(D, R), dot(D, U)) / (d * TanHalf);
            if (any(greaterThan(abs(UV), vec2(1.0)))) continue;                         // outside the spot's frustum
            const float Texel0 = 2.0 * d * TanHalf / float(kLevel0Pages * kPageTexels);
            const int   Level  = clamp(int(floor(log2(max(Foot, 1e-9) / Texel0) + F[Push.m_Frame].P0.y)), 0, kLevels - 1);
            atomicAdd(s_Count[0], 1u);
            atomicAdd(s_Count[5 + Level], 1u);
            const uint FaceBase = uint(Lt.Info.x) + uint(Face * kFaceEntries);
            int Found = -1;
            for (int l = Level; l < kLevels; ++l)
            {
                const int   N  = (kLevel0Pages >> l) * kPageTexels;
                const ivec2 T  = ivec2(floor((UV * 0.5 + 0.5) * float(N)));
                const ivec2 Lo = clamp(T - 3, ivec2(0), ivec2(N - 1)) / kPageTexels, Hi = clamp(T + 3, ivec2(0), ivec2(N - 1)) / kPageTexels;
                const uint  B  = FaceBase + LevelBase(l);
                const int   W  = kLevel0Pages >> l;
                if (l == Level)
                    for (int y = Lo.y; y <= Hi.y; ++y) for (int x = Lo.x; x <= Hi.x; ++x)
                    {
                        const uint i = B + uint(y * W + x), w = Push.m_ReqBase + (i >> 5), b = 1u << (i & 31u);
                        if ((Req[w] & b) == 0u) atomicOr(Req[w], b);
                    }
                const ivec2 TLo = clamp(T - 1, ivec2(0), ivec2(N - 1)) / kPageTexels, THi = clamp(T + 1, ivec2(0), ivec2(N - 1)) / kPageTexels;
                bool bAll = true;
                for (int y = TLo.y; y <= THi.y; ++y) for (int x = TLo.x; x <= THi.x; ++x) bAll = bAll && Entry(B + uint(y * W + x)) != 0u;
                if (bAll) { Found = l; break; }
            }
            atomicAdd(s_Count[Found < 0 ? 3 : Found == Level ? 1 : 2], 1u);
        }
    }

    barrier();
    if (gl_LocalInvocationIndex < kCounters && s_Count[gl_LocalInvocationIndex] != 0u) atomicAdd(Req[Push.m_ReqBase + Push.m_BitUints + gl_LocalInvocationIndex], s_Count[gl_LocalInvocationIndex]);
}
