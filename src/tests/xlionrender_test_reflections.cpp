// Reflections, milestone A (documentation/reflections_strategy.md sec. 11, 12): the contracts' first measurements
//  - the metrics (xlionrender_test_metrics.h) checked on images whose answer is known
//  - the reflection debug view (scene::m_bDebugReflection): every pixel shows R^, its environment / probe / reflection specular
//  - the constant-environment invariant (sec. 11): under a uniform sky of radiance L, R^ = L E[F G2/G1] (the BRDF's directional
//    albedo, integrated here by VNDF sampling) on every receiver - a missing or doubled BRDF factor shows as a ratio
//  - the independent reference (sec. 11): a CPU path tracer over the GI's triangle BVH (gi::world::Cast / Occluded, the same
//    geometry the renderer draws) with the renderer's sky, sun, local lights' falloff and BRDF (GGX, correlated Smith, Schlick);
//    R^ at a receiver = VNDF samples of F G2/G1 times the radiance along each (emission, the hit's direct light with its highlights,
//    its indirect light; misses: the sky). Opaque rigid geometry only
//  - today's error table: the reflection lab (a chrome sphere, a rough gold one, a glossy floor, coloured walls, a glowing sign, the
//    sun and a lamp) - today's sky only and with one probe, against the reference, per receiver: RMAE (absolute floor, dark pixels
//    included) and dark-leak. Information, not a gate: the gates of B start from these numbers
#include "xlionrender_test_common.h"
#include "xlionrender_test_metrics.h"
#include "features/gi/xlionrender_surfel_gi.h"
#include "shaders/xlionrender_light_eval.glsl"
#include <random>
#include <chrono>

using namespace xlionrender::test;

namespace
{
    using v3 = xmath::fvec3;
    constexpr float pi_v = 3.14159265358979f, min_roughness_v = 0.045f;    // (mb_standard_pbr.frag MIN_ROUGHNESS)

    bool MakeMesh(xlionrender::instance& Renderer, const xprim_geom::mesh& Prim, xlionrender::mesh_id& Id) noexcept
    {
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        return Renderer.CreateMesh(Id, Verts, Prim.m_Indices) == nullptr;
    }

    xlionrender::gi::surfel_mesh RayMesh(const xprim_geom::mesh& Prim)
    {
        xlionrender::gi::surfel_mesh M;
        for (const auto& V : Prim.m_Vertices) { M.m_Pos.push_back({ V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }); M.m_Nrm.push_back({ V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }); M.m_UV.push_back({ V.m_Texcoord.m_X, V.m_Texcoord.m_Y }); }
        for (std::size_t i = 0; i + 2 < Prim.m_Indices.size(); i += 3) M.m_Tris.push_back({ int(Prim.m_Indices[i]), int(Prim.m_Indices[i + 1]), int(Prim.m_Indices[i + 2]) });
        M.Bake(0.08f, 0.25f);
        return M;
    }

    // The HDR target as RGB floats (row 0 = the top)
    bool ReadRGB(xgpu::device& Device, const xlionrender::render_view& View, rgb_image& Out)
    {
        std::vector<std::byte> Raw;
        int W, H;
        if (Device.ReadTexture(View.m_HDR, Raw, W, H)) return false;
        const auto* p = reinterpret_cast<const std::uint16_t*>(Raw.data());
        Out.resize(std::size_t(W) * H * 3);
        for (std::size_t i = 0; i < std::size_t(W) * H; ++i) for (int c = 0; c < 3; ++c) Out[i * 3 + c] = HalfToFloat(p[i * 4 + c]);
        return true;
    }

    v3   Mix(const v3& A, const v3& B, float t) noexcept { return A * (1 - t) + B * t; }
    v3   Mul(const v3& A, const v3& B) noexcept { return { A.m_X * B.m_X, A.m_Y * B.m_Y, A.m_Z * B.m_Z }; }
    float Lum(const v3& C) noexcept { return 0.2126f * C.m_X + 0.7152f * C.m_Y + 0.0722f * C.m_Z; }

    // The single-scattering GGX of mb_standard_pbr.frag: alpha = roughness^2, height-correlated Smith, Schlick
    struct ggx
    {
        float m_A;
        explicit ggx(float Roughness) noexcept : m_A(std::max(Roughness, min_roughness_v) * std::max(Roughness, min_roughness_v)) {}
        float Lambda(float c) const noexcept { c = std::max(c, 1e-6f); const float t2 = (1 - c * c) / (c * c); return 0.5f * (std::sqrt(1 + m_A * m_A * t2) - 1); }
        float D(float NdotH) const noexcept { const float a2 = m_A * m_A, d = NdotH * NdotH * (a2 - 1) + 1; return a2 / (pi_v * d * d); }
        float Vis(float NdotV, float NdotL) const noexcept       // G2 / (4 NdotV NdotL), as VisibilitySmithCorrelated
        {
            const float a2 = m_A * m_A;
            const float V = NdotL * std::sqrt(NdotV * NdotV * (1 - a2) + a2), L = NdotV * std::sqrt(NdotL * NdotL * (1 - a2) + a2);
            return 0.5f / std::max(V + L, 1e-6f);
        }
        // Heitz 2018: a visible normal for the view Ve (local frame, z the normal)
        v3 SampleVNDF(const v3& Ve, float U1, float U2) const noexcept
        {
            const v3 Vh = v3{ m_A * Ve.m_X, m_A * Ve.m_Y, Ve.m_Z }.NormalizeCopy();
            const float L2 = Vh.m_X * Vh.m_X + Vh.m_Y * Vh.m_Y;
            const v3 T1 = L2 > 0 ? v3{ -Vh.m_Y, Vh.m_X, 0 } * (1.0f / std::sqrt(L2)) : v3{ 1, 0, 0 };
            const v3 T2 = Vh.Cross(T1);
            const float r = std::sqrt(U1), phi = 2 * pi_v * U2;
            const float t1 = r * std::cos(phi), s = 0.5f * (1 + Vh.m_Z);
            const float t2 = (1 - s) * std::sqrt(std::max(0.0f, 1 - t1 * t1)) + s * r * std::sin(phi);
            const v3 Nh = T1 * t1 + T2 * t2 + Vh * std::sqrt(std::max(0.0f, 1 - t1 * t1 - t2 * t2));
            return v3{ m_A * Nh.m_X, m_A * Nh.m_Y, std::max(1e-6f, Nh.m_Z) }.NormalizeCopy();
        }
    };
    v3 Schlick(const v3& F0, float c) noexcept { const float f = std::pow(std::clamp(1 - c, 0.0f, 1.0f), 5.0f); return F0 + (v3{ 1, 1, 1 } - F0) * f; }

    struct frame
    {
        v3 m_T, m_B, m_N;
        explicit frame(const v3& N) noexcept : m_N(N)
        {
            const float s = N.m_Z >= 0 ? 1.0f : -1.0f, a = -1.0f / (s + N.m_Z), b = N.m_X * N.m_Y * a;      // Duff et al. 2017
            m_T = { 1 + s * N.m_X * N.m_X * a, s * b, -s * N.m_X };
            m_B = { b, s + N.m_Y * N.m_Y * a, -N.m_Y };
        }
        v3 ToLocal(const v3& V) const noexcept { return { V.Dot(m_T), V.Dot(m_B), V.Dot(m_N) }; }
        v3 ToWorld(const v3& V) const noexcept { return m_T * V.m_X + m_B * V.m_Y + m_N * V.m_Z; }
    };

    struct sample_rng
    {
        std::mt19937 m_G;
        std::uniform_real_distribution<float> m_U{ 0.0f, 1.0f };
        explicit sample_rng(std::uint32_t Seed) : m_G(Seed) {}
        float operator()() noexcept { return std::min(m_U(m_G), 0.99999994f); }
    };

    // One VNDF sample of a receiver's specular lobe: the direction and its weight F G2 / G1 (0: below the surface)
    struct spec_sample { v3 m_L; v3 m_W; };
    spec_sample SampleSpecular(const frame& Fr, const v3& Wo, const ggx& G, const v3& F0, sample_rng& Rng) noexcept
    {
        const v3 Ve = Fr.ToLocal(Wo);
        const v3 H  = G.SampleVNDF(Ve.m_Z > 1e-4f ? Ve : v3{ Ve.m_X, Ve.m_Y, 1e-4f }.NormalizeCopy(), Rng(), Rng());
        const v3 L  = H * (2 * Ve.Dot(H)) - Ve;
        if (L.m_Z <= 0) return { Fr.ToWorld(L), { 0, 0, 0 } };
        const float LV = G.Lambda(Ve.m_Z), LL = G.Lambda(L.m_Z);
        return { Fr.ToWorld(L), Schlick(F0, std::clamp(Ve.Dot(H), 0.0f, 1.0f)) * ((1 + LV) / (1 + LV + LL)) };
    }

    struct object { const xlionrender::gi::surfel_mesh* m_pRay; xlionrender::mesh_id m_Mesh; xmath::fmat4 m_L2W; xlionrender::material m_Mat; float (*m_pTexture)(float u, float v) = nullptr; };   // (m_pTexture: the albedo's factor at a UV)

    // The independent reference (see the top)
    struct reference
    {
        const xlionrender::gi::world&       m_World;
        const std::vector<object>&          m_Objects;
        const xlionrender::scene&           m_Scene;
        int                                 m_MaxDepth = 3;     // secondary hit 1, then two more bounces

        v3 Sky(const v3& D) const noexcept
        {
            const auto& S = m_Scene.m_Sky;
            const float Up = D.m_Y;
            const v3 C = Up >= 0 ? Mix(S.m_Horizon, S.m_Zenith, std::pow(Up, 0.5f)) : Mix(S.m_Horizon, S.m_Ground, std::pow(-Up, 0.35f));
            return C * S.m_Intensity;
        }

        v3 Trace(const v3& O, const v3& D, int Depth, sample_rng& Rng) const noexcept
        {
            const auto H = m_World.Cast(O, D);
            if (H.m_Inst < 0) return Sky(D);
            v3 N = m_World.HitNormal(H).NormalizeCopy();
            if (N.Dot(D) > 0) N = N * -1.0f;
            const auto& Ob = m_Objects[std::size_t(H.m_Inst)];
            if (Ob.m_pTexture == nullptr || Ob.m_pRay->m_UV.size() != Ob.m_pRay->m_Pos.size()) return Shade(m_World.HitPosition(H), N, D * -1.0f, Ob.m_Mat, Depth, Rng);
            const auto& T = Ob.m_pRay->m_Tris[std::size_t(H.m_Tri)];
            const auto& A = Ob.m_pRay->m_UV[std::size_t(T[0])]; const auto& B = Ob.m_pRay->m_UV[std::size_t(T[1])]; const auto& C = Ob.m_pRay->m_UV[std::size_t(T[2])];
            const float b0 = 1 - H.m_U - H.m_V;
            auto M = Ob.m_Mat;
            M.m_BaseColor = M.m_BaseColor * Ob.m_pTexture(A[0] * b0 + B[0] * H.m_U + C[0] * H.m_V, A[1] * b0 + B[1] * H.m_U + C[1] * H.m_V);
            return Shade(m_World.HitPosition(H), N, D * -1.0f, M, Depth, Rng);
        }

        v3 Shade(const v3& P, const v3& N, const v3& Wo, const xlionrender::material& M, int Depth, sample_rng& Rng) const noexcept
        {
            const v3 Kd = M.m_BaseColor * (1 - M.m_Metallic);
            const v3 F0 = Mix(v3{ M.m_F0, M.m_F0, M.m_F0 }, M.m_BaseColor, M.m_Metallic);
            const ggx G(M.m_Roughness);
            const float NdotV = std::max(N.Dot(Wo), 1e-4f);
            const v3 O = P + N * 1e-3f;
            v3 L = M.m_Emissive;
            const auto Direct = [&](const v3& Ld, const v3& Radiance)
            {
                const float NdotL = N.Dot(Ld);
                if (NdotL <= 0) return;
                const v3 Hh = (Ld + Wo).NormalizeCopy();
                const v3 Spec = Schlick(F0, std::clamp(Ld.Dot(Hh), 0.0f, 1.0f)) * (G.D(std::max(N.Dot(Hh), 0.0f)) * G.Vis(NdotV, NdotL));
                L = L + Mul(Kd * (1 / pi_v) + Spec, Radiance) * NdotL;
            };
            const auto& Sun = m_Scene.m_Sun;
            if (Sun.m_Intensity > 0)
            {
                const v3 Ld = (Sun.m_Direction * -1.0f).NormalizeCopy();
                if (N.Dot(Ld) > 0 && !m_World.Occluded(O, Ld)) Direct(Ld, Sun.m_Color * Sun.m_Intensity);
            }
            for (const auto& Lt : m_Scene.m_LocalLights)
            {
                const v3 Dv = Lt.m_Position - P;
                const float d = Dv.Length();
                if (d >= Lt.m_Radius || d < 1e-5f) continue;
                const v3 Ld = Dv * (1 / d);
                float Att = xlionrender::light_eval::LightDistanceFalloff(d / Lt.m_Radius, Lt.m_Intensity);
                if (Lt.m_Type == xlionrender::local_light::type::SPOT)
                    Att *= xlionrender::light_eval::LightSpotCone((Ld * -1.0f).Dot(Lt.m_Direction.NormalizeCopy()), std::cos(Lt.m_OuterAngle), std::cos(Lt.m_InnerAngle));
                if (Att <= 0 || N.Dot(Ld) <= 0 || (Lt.m_bCastShadows && m_World.Occluded(O, Ld, d - 2e-3f))) continue;
                Direct(Ld, Lt.m_Color * Att);
            }
            if (Depth >= m_MaxDepth) return L;
            // One more bounce: the specular lobe with probability pS (VNDF), else the diffuse one (cosine)
            const float pS = Lum(Kd) <= 0 ? 1.0f : std::clamp(Lum(Schlick(F0, NdotV)) / (Lum(Schlick(F0, NdotV)) + Lum(Kd)), 0.1f, 0.9f);
            const frame Fr(N);
            if (Rng() < pS)
            {
                const auto S = SampleSpecular(Fr, Wo, G, F0, Rng);
                if (Lum(S.m_W) > 0) L = L + Mul(S.m_W, Trace(O, S.m_L, Depth + 1, Rng)) * (1 / pS);
            }
            else
            {
                const float r = std::sqrt(Rng()), phi = 2 * pi_v * Rng();
                const v3 Ld = Fr.ToWorld({ r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0f, 1 - r * r)) });
                L = L + Mul(Kd, Trace(O, Ld, Depth + 1, Rng)) * (1 / (1 - pS));
            }
            return L;
        }

        // R^ at a receiver: the mean of F G2/G1 times the radiance along its VNDF samples
        v3 Reflected(const v3& P, const v3& N, const v3& Wo, const xlionrender::material& M, int Samples, sample_rng& Rng) const noexcept
        {
            const v3 F0 = Mix(v3{ M.m_F0, M.m_F0, M.m_F0 }, M.m_BaseColor, M.m_Metallic);
            const ggx G(M.m_Roughness);
            const frame Fr(N);
            v3 Sum{ 0, 0, 0 };
            for (int s = 0; s < Samples; ++s)
            {
                const auto S = SampleSpecular(Fr, Wo, G, F0, Rng);
                if (Lum(S.m_W) > 0) Sum = Sum + Mul(S.m_W, Trace(P + N * 1e-3f, S.m_L, 1, Rng));
            }
            return Sum * (1.0f / float(Samples));
        }

        // The lobe's directional albedo E[F G2/G1] (the constant-environment answer per unit radiance)
        static v3 DirectionalAlbedo(const v3& N, const v3& Wo, const xlionrender::material& M, int Samples, sample_rng& Rng) noexcept
        {
            const v3 F0 = Mix(v3{ M.m_F0, M.m_F0, M.m_F0 }, M.m_BaseColor, M.m_Metallic);
            const ggx G(M.m_Roughness);
            const frame Fr(N);
            v3 Sum{ 0, 0, 0 };
            for (int s = 0; s < Samples; ++s) Sum = Sum + SampleSpecular(Fr, Wo, G, F0, Rng).m_W;
            return Sum * (1.0f / float(Samples));
        }
    };
}

int main()
{
    {
        const auto Why = SelfTestMetrics();
        CHECK(Why.empty(), "metrics: %s", Why.c_str());
        std::printf("metrics: the self-test passes\n");
    }

    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    const auto SpherePrim = xprim_geom::uvsphere::Generate(64, 64, 1, 0.5f);
    for (const auto& V : SpherePrim.m_Vertices)     // (its normals were (x R w S, y / S, z R w S): right only for a radius of 1, up to 37 degrees off here)
        CHECK(std::abs(v3{ V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }.NormalizeCopy().Dot({ V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }) - 1) < 1e-4f, "uvsphere: a normal is not the sphere's");
    const auto BoxPrim    = xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 });
    xlionrender::mesh_id Sphere, Box;
    CHECK(MakeMesh(Renderer, SpherePrim, Sphere) && MakeMesh(Renderer, BoxPrim, Box), "meshes");
    const auto RaySphere = RayMesh(SpherePrim), RayBox = RayMesh(BoxPrim);

    constexpr int size_v = 128;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 0.8f });
    Camera.setViewport({ 0, 0, size_v, size_v });

    const auto Run = [&](const std::vector<object>& Objects, xlionrender::scene& Scene, rgb_image& Out, int Frames) -> bool
    {
        Scene.m_Draws.clear();
        for (const auto& O : Objects) Scene.m_Draws.push_back({ .m_Mesh = O.m_Mesh, .m_L2W = O.m_L2W, .m_Material = O.m_Mat });
        Scene.m_bDebugReflection = true;
        for (int f = 0; f < Frames; ++f) if (!App.RenderFrame(Scene, View, Camera)) return false;
        return ReadRGB(Device, View, Out);
    };
    const auto MakeWorld = [&](const std::vector<object>& Objects, xlionrender::gi::world& World)
    {
        for (const auto& O : Objects) World.Add(*O.m_pRay, O.m_L2W, O.m_Mat.m_BaseColor);
    };
    // A pixel's camera ray (the axes' signs are found once below, as in xlionrender_test_light_reference.cpp)
    int SX = 0, SY = 0;
    const auto Ray = [&](int x, int y, int sx, int sy)
    {
        const xmath::fmat4 C2W = Camera.getC2W();
        const float nx = float(sx) * (2.0f * (float(x) + 0.5f) / size_v - 1.0f), ny = float(sy) * (1.0f - 2.0f * (float(y) + 0.5f) / size_v);
        const auto At = [&](float z) { const xmath::fvec4 P = C2W * xmath::fvec4{ nx, ny, z, 1.0f }; return v3{ P.m_X, P.m_Y, P.m_Z } * (1.0f / P.m_W); };
        auto D = (At(0.6f) - At(0.2f)).NormalizeCopy();
        return D.Dot(Camera.getForward()) < 0 ? D * -1.0f : D;
    };

    //--------------------------------------------------------------------------------------------------------------------------------
    // The constant-environment invariant: four spheres under a uniform sky (no sun, no lights): the renderer's R^ against L E[F G2/G1]
    {
        Camera.LookAt(v3{ 0.5f, 1.3f, -5.0f }, v3{ 0.3f, 0.3f, 0 });     // (off-centre: a mirrored mapping must not fit)
        const xlionrender::material Mats[] =
        { { .m_BaseColor = { 0.95f, 0.93f, 0.88f }, .m_Metallic = 1, .m_Roughness = 0.05f }       // chrome
        , { .m_BaseColor = { 1.00f, 0.78f, 0.34f }, .m_Metallic = 1, .m_Roughness = 0.40f }       // rough gold
        , { .m_BaseColor = { 0.50f, 0.50f, 0.50f }, .m_Metallic = 0, .m_Roughness = 0.20f }       // glossy dielectric
        , { .m_BaseColor = { 0.20f, 0.40f, 0.80f }, .m_Metallic = 0, .m_Roughness = 0.70f } };    // rough dielectric
        std::vector<object> Objects;
        for (int i = 0; i < 4; ++i) Objects.push_back({ &RaySphere, Sphere, xmath::fmat4::fromTranslation({ -1.8f + 1.2f * float(i), 0.6f, 0 }), Mats[i] });
        xlionrender::scene Scene;
        Scene.m_GI.m_bAO = false;      // (images compared between paths: the screen's ambient occlusion is gi_render's)
        Scene.m_Sun.m_Intensity = 0;
        Scene.m_Sky = { .m_Zenith = { 0.6f, 0.6f, 0.6f }, .m_Horizon = { 0.6f, 0.6f, 0.6f }, .m_Ground = { 0.6f, 0.6f, 0.6f }, .m_Intensity = 1 };
        rgb_image Gpu;
        CHECK(Run(Objects, Scene, Gpu, 4), "render the uniform sky");
        xlionrender::gi::world World;
        MakeWorld(Objects, World);

        // The pixel to ray mapping: the signs whose rays hit exactly the pixels the GPU shaded (everything else is the black background)
        double Best = -1, Second = -1;
        for (int sx : { 1, -1 }) for (int sy : { 1, -1 })
        {
            int nAgree = 0;
            for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
                nAgree += (World.Cast(Camera.getPosition(), Ray(x, y, sx, sy)).m_Inst >= 0) == (Lum(&Gpu[(std::size_t(y) * size_v + x) * 3]) > 1e-4f);
            const double A = double(nAgree) / (size_v * size_v);
            if (A > Best) { Second = Best; Best = A; SX = sx; SY = sy; } else Second = std::max(Second, A);
        }
        std::printf("pixel to ray mapping: NDC signs %d %d, agree %.2f%% (the next best %.2f%%)\n", SX, SY, 100 * Best, 100 * Second);
        CHECK(Best > 0.99 && Second < Best - 0.02, "no unambiguous pixel to ray mapping (best %.3f, next %.3f)", Best, Second);

        rgb_image Ref(Gpu.size(), 0.0f);
        std::vector<mask> Masks(4, mask(size_v * size_v, 0));
        sample_rng Rng(7);
        for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
        {
            const std::size_t i = std::size_t(y) * size_v + x;
            const v3 D = Ray(x, y, SX, SY);
            const auto H = World.Cast(Camera.getPosition(), D);
            if (H.m_Inst < 0) continue;
            const v3 N = World.HitNormal(H).NormalizeCopy();
            if (N.Dot(D) > -0.2f) continue;         // the silhouette: the renderer's and the rays' normals differ there (excluded from both)
            Masks[std::size_t(H.m_Inst)][i] = 1;
            const v3 E = Mul(reference::DirectionalAlbedo(N, D * -1.0f, Objects[std::size_t(H.m_Inst)].m_Mat, 2048, Rng), v3{ 0.6f, 0.6f, 0.6f });
            Ref[i * 3] = E.m_X; Ref[i * 3 + 1] = E.m_Y; Ref[i * 3 + 2] = E.m_Z;
        }
        const char* Names[] = { "chrome", "rough gold", "glossy dielectric", "rough dielectric" };
        double Worst = 0;
        for (int k = 0; k < 4; ++k)
        {
            double Ratio = 0, n = 0;
            for (std::size_t i = 0; i < Masks[k].size(); ++i) if (Masks[k][i]) { Ratio += Lum(&Gpu[i * 3]) / std::max(Lum(&Ref[i * 3]), 1e-6f); ++n; }
            const double E = RMAE(Gpu, Ref, Masks[k], 1e-3);
            std::printf("constant environment, %-17s: RMAE %.4f, mean renderer / reference %.4f (%d pixels)\n", Names[k], E, Ratio / std::max(n, 1.0), int(n));
            Worst = std::max(Worst, E);
        }
        DumpRGB("refl_const_gpu", Gpu, size_v, size_v);
        DumpRGB("refl_const_ref", Ref, size_v, size_v);
        CHECK(Worst < 0.05, "the constant-environment invariant fails: RMAE %.4f (a missing or doubled BRDF factor, or the DFG table)", Worst);
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    // The reflection lab and today's error table
    {
        Camera.LookAt(v3{ 0.6f, 1.6f, -4.2f }, v3{ 0.2f, 0.5f, 1.0f });
        enum : int { floor_v, back_v, left_v, sign_v, chrome_v, gold_v };
        std::vector<object> Objects =
        { { &RayBox,    Box,    xmath::fmat4::fromTranslation({ 0, -0.5f, 0 }) * xmath::fmat4::fromScale({ 10, 1, 10 }), { .m_BaseColor = { 0.45f, 0.45f, 0.45f }, .m_Roughness = 0.10f } }
        , { &RayBox,    Box,    xmath::fmat4::fromTranslation({ 0, 1.5f, 3.5f }) * xmath::fmat4::fromScale({ 8, 3, 0.4f }), { .m_BaseColor = { 0.75f, 0.12f, 0.10f }, .m_Roughness = 0.8f } }
        , { &RayBox,    Box,    xmath::fmat4::fromTranslation({ -2.8f, 1.5f, 0 }) * xmath::fmat4::fromScale({ 0.4f, 3, 7 }), { .m_BaseColor = { 0.10f, 0.60f, 0.15f }, .m_Roughness = 0.8f } }
        , { &RayBox,    Box,    xmath::fmat4::fromTranslation({ 0.9f, 1.9f, 3.25f }) * xmath::fmat4::fromScale({ 1.4f, 0.5f, 0.1f }), { .m_BaseColor = { 0.05f, 0.05f, 0.05f }, .m_Roughness = 0.6f, .m_Emissive = { 6, 4, 1.5f } } }
        , { &RaySphere, Sphere, xmath::fmat4::fromTranslation({ -0.4f, 0.6f, 0.4f }) * xmath::fmat4::fromScale({ 1.2f, 1.2f, 1.2f }), { .m_BaseColor = { 0.95f, 0.93f, 0.88f }, .m_Metallic = 1, .m_Roughness = 0.05f } }
        , { &RaySphere, Sphere, xmath::fmat4::fromTranslation({ 1.3f, 0.5f, 0.9f }), { .m_BaseColor = { 1.00f, 0.78f, 0.34f }, .m_Metallic = 1, .m_Roughness = 0.35f } } };
        xlionrender::scene Scene;
        Scene.m_GI.m_bAO = false;      // (images compared between paths: the screen's ambient occlusion is gi_render's)
        Scene.m_Sun   = { .m_Direction = { 0.35f, -1, 0.45f }, .m_Intensity = 2.5f };
        Scene.m_Sky.m_Intensity = 1;
        Scene.m_LocalLights = { { .m_Position = { -1.2f, 1.4f, -0.8f }, .m_Radius = 4, .m_Color = { 1, 0.9f, 0.75f }, .m_Intensity = 3 } };

        xlionrender::gi::world World;
        MakeWorld(Objects, World);
        const reference Reference{ World, Objects, Scene };

        // Today, the sky only (first: the render sets the camera's projection the rays below are made from)
        rgb_image SkyOnly, WithProbe;
        CHECK(Run(Objects, Scene, SkyOnly, 4), "render today, the sky only");

        // The reference (and the receivers' masks: the floor, the two spheres; the silhouettes left out)
        constexpr int samples_v = 64;
        rgb_image Ref(std::size_t(size_v) * size_v * 3, 0.0f);
        mask MFloor(size_v * size_v, 0), MChrome = MFloor, MGold = MFloor, MAll = MFloor;
        const auto T0 = std::chrono::steady_clock::now();
        std::uint64_t nRays = 0;
        for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
        {
            const std::size_t i = std::size_t(y) * size_v + x;
            const v3 D = Ray(x, y, SX, SY);
            const auto H = World.Cast(Camera.getPosition(), D);
            if (H.m_Inst < 0) continue;
            const v3 N = World.HitNormal(H).NormalizeCopy();
            if (N.Dot(D) > -0.2f) continue;
            const int Id = H.m_Inst;
            if (Id != floor_v && Id != chrome_v && Id != gold_v) continue;
            (Id == floor_v ? MFloor : Id == chrome_v ? MChrome : MGold)[i] = 1;
            MAll[i] = 1;
            sample_rng Rng(std::uint32_t(i * 7919 + 1));
            const v3 R = Reference.Reflected(World.HitPosition(H), N, D * -1.0f, Objects[std::size_t(Id)].m_Mat, samples_v, Rng);
            Ref[i * 3] = R.m_X; Ref[i * 3 + 1] = R.m_Y; Ref[i * 3 + 2] = R.m_Z;
            nRays += samples_v;
        }
        const double RefMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
        double MeanRef = 0, n = 0;
        for (std::size_t i = 0; i < MAll.size(); ++i) if (MAll[i]) { MeanRef += Lum(&Ref[i * 3]); ++n; }
        MeanRef /= std::max(n, 1.0);
        const double Floor = 1e-3 * MeanRef;
        std::printf("reference: %d samples per pixel, %.0f ms on one thread (%llu receiver samples); the floor %.2e (1e-3 of the receivers' mean %.4f)\n"
            , samples_v, RefMs, static_cast<unsigned long long>(nRays), Floor, MeanRef);
        CHECK(MeanRef > 0.01, "the reference's receivers are dark (%f): the lab is wrong", MeanRef);

        // Today with one probe over the lab
        Scene.m_Probes = { { .m_Position = { 0.2f, 1.2f, 0.4f }, .m_BoxMin = { -2.6f, -0.6f, -5 }, .m_BoxMax = { 4.9f, 3, 3.3f }, .m_Blend = 0.3f } };
        Renderer.m_ProbeFacesPerFrame = 6;
        CHECK(Run(Objects, Scene, WithProbe, 4), "render today, with a probe");
        CHECK(Renderer.getProbeCaptures() >= 1, "the probe was not captured");

        std::printf("today's error table (R^ against the reference; RMAE with the absolute floor, dark pixels included | dark-leak):\n");
        std::printf("  %-10s %-24s %-24s\n", "receiver", "sky only", "sky + one probe");
        const std::pair<const char*, const mask*> Rows[] = { { "floor", &MFloor }, { "chrome", &MChrome }, { "rough gold", &MGold }, { "all", &MAll } };
        for (const auto& [pName, pM] : Rows)
        {
            int nDark = 0, nIn = 0;
            for (std::size_t i = 0; i < pM->size(); ++i) if ((*pM)[i]) { ++nIn; nDark += Lum(&Ref[i * 3]) < Floor; }
            std::printf("  %-10s RMAE %6.3f | leak %8.3f   RMAE %6.3f | leak %8.3f   (%d pixels, %d dark)\n", pName
                , RMAE(SkyOnly, Ref, *pM, Floor), DarkLeak(SkyOnly, Ref, *pM, Floor), RMAE(WithProbe, Ref, *pM, Floor), DarkLeak(WithProbe, Ref, *pM, Floor), nIn, nDark);
        }

        // The reference's own noise (bias / noise over 4 independent runs of the chrome sphere, 16 samples each, against the 64-sample one)
        std::vector<rgb_image> Runs(4, rgb_image(Ref.size(), 0.0f));
        for (int r = 0; r < 4; ++r)
            for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
            {
                const std::size_t i = std::size_t(y) * size_v + x;
                if (!MChrome[i]) continue;
                const v3 D = Ray(x, y, SX, SY);
                const auto H = World.Cast(Camera.getPosition(), D);
                sample_rng Rng(std::uint32_t(i * 104729 + 31 * r + 17));
                const v3 R = Reference.Reflected(World.HitPosition(H), World.HitNormal(H).NormalizeCopy(), D * -1.0f, Objects[chrome_v].m_Mat, 16, Rng);
                Runs[r][i * 3] = R.m_X; Runs[r][i * 3 + 1] = R.m_Y; Runs[r][i * 3 + 2] = R.m_Z;
            }
        const auto BN = BiasNoise(Runs, Ref, MChrome, Floor);
        std::printf("reference self-consistency, chrome: 4 runs of 16 samples against 64: mean %.4f, per-run noise %.4f\n", BN.m_Bias, BN.m_Noise);
        CHECK(BN.m_Bias < 0.05, "the reference does not agree with itself (%f)", BN.m_Bias);

        DumpRGB("refl_lab_ref", Ref, size_v, size_v);
        DumpRGB("refl_lab_sky", SkyOnly, size_v, size_v);
        DumpRGB("refl_lab_probe", WithProbe, size_v, size_v);
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    // Milestone B: the lab again, its meshes with GI (rigid cluster meshes with surfels: the GI's BVH and data), the traced reflections on
    // (scene::m_ReflectionMaxRoughness). R^ against the same independent reference, per receiver - today's numbers and the traced ones
    {
        Camera.LookAt(v3{ 0.6f, 1.6f, -4.2f }, v3{ 0.2f, 0.5f, 1.0f });
        struct gi_object { xprim_geom::mesh m_Prim; v3 m_At; float m_Scale; xlionrender::material m_Mat; };
        auto Sized = [](float X, float Y, float Z, int n) { return xprim_geom::cube::Generate(n, n, n, n, xprim_geom::float3{ X, Y, Z }); };
        enum : int { floor_v, back_v, left_v, sign_v, chrome_v, gold_v };
        std::vector<gi_object> Lab =
        { { Sized(10, 1, 10, 20),  { 0, -0.5f, 0 },      1,    { .m_BaseColor = { 0.45f, 0.45f, 0.45f }, .m_Roughness = 0.10f } }
        , { Sized(8, 3, 0.4f, 16), { 0, 1.5f, 3.5f },    1,    { .m_BaseColor = { 0.75f, 0.12f, 0.10f }, .m_Roughness = 0.8f } }
        , { Sized(0.4f, 3, 7, 16), { -2.8f, 1.5f, 0 },   1,    { .m_BaseColor = { 0.10f, 0.60f, 0.15f }, .m_Roughness = 0.8f } }
        , { Sized(1.4f, 0.5f, 0.1f, 4), { 0.9f, 1.9f, 3.25f }, 1, { .m_BaseColor = { 0.05f, 0.05f, 0.05f }, .m_Roughness = 0.6f, .m_Emissive = { 6, 4, 1.5f } } }
        , { SpherePrim,            { -0.4f, 0.6f, 0.4f }, 1.2f, { .m_BaseColor = { 0.95f, 0.93f, 0.88f }, .m_Metallic = 1, .m_Roughness = 0.05f } }
        , { SpherePrim,            { 1.3f, 0.5f, 0.9f },  1,    { .m_BaseColor = { 1.00f, 0.78f, 0.34f }, .m_Metallic = 1, .m_Roughness = 0.35f } } };
        std::vector<cluster_buffers> Buffers(Lab.size());
        std::vector<xlionrender::gi::surfel_mesh> Rays(Lab.size());
        std::vector<object> Objects;
        xlionrender::scene Scene;
        Scene.m_GI.m_bAO = false;      // (images compared between paths: the screen's ambient occlusion is gi_render's)
        for (std::size_t k = 0; k < Lab.size(); ++k)
        {
            xlionrender::mesh_id Id;
            auto Surfels = MakeSurfels(Lab[k].m_Prim, 0.15f, 0.45f);
            for (const auto& V : Lab[k].m_Prim.m_Vertices) Surfels.m_UV.push_back({ V.m_Texcoord.m_X, V.m_Texcoord.m_Y });     // (after the bake: the hits' UVs, no surfel capture)
            CHECK(MakeStaticBuffers(Device, Lab[k].m_Prim, 5.0f, Buffers[k]) && !Renderer.CreateClusterGeom(Id, Buffers[k].Geom()) && !Renderer.SetMeshGI(Id, std::move(Surfels)), "the lab's GI mesh %zu", k);
            Rays[k] = RayMesh(Lab[k].m_Prim);
            const auto L2W = xmath::fmat4::fromTranslation({ Lab[k].m_At.m_X, Lab[k].m_At.m_Y, Lab[k].m_At.m_Z }) * xmath::fmat4::fromScale({ Lab[k].m_Scale, Lab[k].m_Scale, Lab[k].m_Scale });
            Objects.push_back({ &Rays[k], Id, L2W, Lab[k].m_Mat });
            Scene.m_Draws.push_back({ .m_Mesh = Id, .m_L2W = L2W, .m_Material = Lab[k].m_Mat, .m_Instance = 100 + k });
        }
        Scene.m_Sun   = { .m_Direction = { 0.35f, -1, 0.45f }, .m_Intensity = 2.5f };
        Scene.m_Sky.m_Intensity = 1;
        Scene.m_LocalLights = { { .m_Position = { -1.2f, 1.4f, -0.8f }, .m_Radius = 4, .m_Color = { 1, 0.9f, 0.75f }, .m_Intensity = 3 } };
        Scene.m_GI = { .m_bEnable = true, .m_EmissiveScale = 1 };       // (the reference's emission lights as it shows: the GI's default lights it 100 times brighter)
        Scene.m_GI.m_bAO = false;      // (against the path traced reference: it has no screen-space occlusion)
        Scene.m_bDebugReflection = true;
        Scene.m_Probes.clear();

        xlionrender::gi::world World;
        MakeWorld(Objects, World);
        const reference Reference{ World, Objects, Scene };
        rgb_image Today, Traced;
        App.m_bQuietTimes = true;
        for (int f = 0; f < 240; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "settle the GI");
        CHECK(ReadRGB(Device, View, Today), "read today (GI on, no traced reflections)");
        Scene.m_ReflectionMaxRoughness = 0.3f;
        Scene.m_bReflectionJitter = true;                   // (the 16 frames' mean: a new ray a texel each)
        Scene.m_bReflectionReconstruct = false;             // (B measures the estimator itself: its one ray a texel, no reconstruction)
        // (the traced reflections are one ray a texel, a new one each frame: the mean of 16 frames is the estimator's expectation - its
        // bias, what B gates; the noise is the reconstruction's, milestone D)
        const auto Mean = [&](int Frames, rgb_image& Out) -> bool
        {
            for (int f = 0; f < 4; ++f) if (!App.RenderFrame(Scene, View, Camera)) return false;
            rgb_image One;
            Out.clear();
            for (int f = 0; f < Frames; ++f)
            {
                if (!App.RenderFrame(Scene, View, Camera) || !ReadRGB(Device, View, One)) return false;
                if (Out.empty()) Out.assign(One.size(), 0.0f);
                for (std::size_t k = 0; k < One.size(); ++k) Out[k] += One[k] / float(Frames);
            }
            return true;
        };
        CHECK(Mean(16, Traced), "render the traced reflections");
        rgb_image Planar;
        Scene.m_ReflectionPlane = { 0, 1, 0, 0 };          // (the floor's top: y = 0)
        CHECK(Mean(16, Planar), "render the planar view");
        Scene.m_ReflectionPlane = { 0, 0, 0, 0 };

        rgb_image Ref(Today.size(), 0.0f);
        mask MFloor(size_v * size_v, 0), MChrome = MFloor, MGold = MFloor;
        for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
        {
            const std::size_t i = std::size_t(y) * size_v + x;
            const v3 D = Ray(x, y, SX, SY);
            const auto H = World.Cast(Camera.getPosition(), D);
            if (H.m_Inst < 0) continue;
            const v3 N = World.HitNormal(H).NormalizeCopy();
            if (N.Dot(D) > -0.2f) continue;
            const int Id = H.m_Inst;
            if (Id != floor_v && Id != chrome_v && Id != gold_v) continue;
            (Id == floor_v ? MFloor : Id == chrome_v ? MChrome : MGold)[i] = 1;
            sample_rng Rng(std::uint32_t(i * 7919 + 1));
            const v3 R = Reference.Reflected(World.HitPosition(H), N, D * -1.0f, Objects[std::size_t(Id)].m_Mat, 64, Rng);
            Ref[i * 3] = R.m_X; Ref[i * 3 + 1] = R.m_Y; Ref[i * 3 + 2] = R.m_Z;
        }
        double MeanRef = 0, n = 0;
        for (std::size_t i = 0; i < MFloor.size(); ++i) if (MFloor[i] || MChrome[i] || MGold[i]) { MeanRef += Lum(&Ref[i * 3]); ++n; }
        const double Floor = 1e-3 * MeanRef / std::max(n, 1.0);
        std::printf("milestone B, the lab with GI (R^ against the reference; RMAE | dark-leak):\n  %-10s %-24s %-24s %-24s\n", "receiver", "today (probes / sky)", "traced (sharp: r < 0.3)", "+ the floor's planar view");
        const std::pair<const char*, const mask*> Rows[] = { { "floor", &MFloor }, { "chrome", &MChrome }, { "rough gold", &MGold } };
        double ChromeTraced = 0, ChromeToday = 0, FloorTraced = 0, FloorToday = 0, FloorPlanar = 0;
        for (const auto& [pName, pM] : Rows)
        {
            const double Et = RMAE(Today, Ref, *pM, Floor), Er = RMAE(Traced, Ref, *pM, Floor), Ep = RMAE(Planar, Ref, *pM, Floor);
            std::printf("  %-10s RMAE %6.3f | leak %6.3f   RMAE %6.3f | leak %6.3f   RMAE %6.3f | leak %6.3f\n", pName, Et, DarkLeak(Today, Ref, *pM, Floor), Er, DarkLeak(Traced, Ref, *pM, Floor), Ep, DarkLeak(Planar, Ref, *pM, Floor));
            if (pM == &MChrome) { ChromeTraced = Er; ChromeToday = Et; }
            if (pM == &MFloor) { FloorTraced = Er; FloorToday = Et; FloorPlanar = Ep; }
        }
        DumpRGB("refl_b_ref", Ref, size_v, size_v);
        DumpRGB("refl_b_today", Today, size_v, size_v);
        DumpRGB("refl_b_traced", Traced, size_v, size_v);
        DumpRGB("refl_b_planar", Planar, size_v, size_v);
        CHECK(FloorPlanar < FloorTraced, "the planar view does not improve the floor over the traced rays (%.3f against %.3f)", FloorPlanar, FloorTraced);
        CHECK(FloorTraced < 0.7 * FloorToday, "the traced reflections do not improve the floor (%.3f against today's %.3f)", FloorTraced, FloorToday);
        CHECK(ChromeTraced < 0.5 * ChromeToday, "the traced reflections do not improve the chrome sphere (%.3f against today's %.3f)", ChromeTraced, ChromeToday);
        // A rough floor (roughness 0.3): the planar view filtered by the lobe's footprint (its pyramid), against the reference, today's
        // probes and one traced ray a texel (no reconstruction yet: noisy)
        {
            Objects[floor_v].m_Mat.m_Roughness = 0.3f;
            Scene.m_Draws[floor_v].m_Material.m_Roughness = 0.3f;
            rgb_image RToday, RTraced, RPlanar, RRef(Today.size(), 0.0f);
            const auto Shot = [&](float MaxRough, bool bPlane, rgb_image& Out) -> bool
            {
                Scene.m_ReflectionMaxRoughness = MaxRough;
                Scene.m_ReflectionPlane = bPlane ? xmath::fvec4{ 0, 1, 0, 0 } : xmath::fvec4{ 0, 0, 0, 0 };
                return Mean(16, Out);
            };
            CHECK(Shot(0, false, RToday) && Shot(0.5f, false, RTraced) && Shot(0.5f, true, RPlanar), "render the rough floor");
            Scene.m_ReflectionPlane = { 0, 0, 0, 0 };
            for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
            {
                const std::size_t i = std::size_t(y) * size_v + x;
                if (!MFloor[i]) continue;
                const v3 D = Ray(x, y, SX, SY);
                const auto H = World.Cast(Camera.getPosition(), D);
                sample_rng Rng(std::uint32_t(i * 7919 + 3));
                const v3 R = Reference.Reflected(World.HitPosition(H), World.HitNormal(H).NormalizeCopy(), D * -1.0f, Objects[floor_v].m_Mat, 256, Rng);
                RRef[i * 3] = R.m_X; RRef[i * 3 + 1] = R.m_Y; RRef[i * 3 + 2] = R.m_Z;
            }
            const double Et = RMAE(RToday, RRef, MFloor, Floor), Er = RMAE(RTraced, RRef, MFloor, Floor), Ep = RMAE(RPlanar, RRef, MFloor, Floor);
            std::printf("a rough floor (0.3), RMAE: today %.3f, one traced ray a texel %.3f, the filtered planar view %.3f\n", Et, Er, Ep);
            DumpRGB("refl_b_rough_ref", RRef, size_v, size_v);
            DumpRGB("refl_b_rough_planar", RPlanar, size_v, size_v);
            DumpRGB("refl_b_rough_traced", RTraced, size_v, size_v);
            CHECK(Ep < Et && Ep < Er, "the filtered planar view is not the best on the rough floor (%.3f; today %.3f, traced %.3f)", Ep, Et, Er);

            // Without the jitter (scene::m_bReflectionJitter, the default until D accumulates) each texel keeps its ray: the rough floor
            // stays still from frame to frame (only the GI's own updates move it); with it, it changes every frame
            const auto Change = [&](bool bJitter) -> double
            {
                Scene.m_bReflectionJitter = bJitter;
                Scene.m_ReflectionMaxRoughness = 0.5f;
                rgb_image A, B;
                for (int f = 0; f < 4; ++f) if (!App.RenderFrame(Scene, View, Camera)) return -1;
                if (!ReadRGB(Device, View, A) || !App.RenderFrame(Scene, View, Camera) || !ReadRGB(Device, View, B)) return -1;
                double d = 0, n = 0;
                for (std::size_t i = 0; i < MFloor.size(); ++i) if (MFloor[i]) { d += std::abs(Lum(&A[i * 3]) - Lum(&B[i * 3])); ++n; }
                return d / std::max(n, 1.0);
            };
            const double Still = Change(false), Moving = Change(true);
            std::printf("the rough floor's change a frame: %.5f without the jitter, %.5f with it\n", Still, Moving);
            CHECK(Still >= 0 && Moving > 0 && Still * 10 < Moving, "the reflections still move without the jitter (%.5f against %.5f)", Still, Moving);

            // Milestone D, the reconstruction (scene::m_bReflectionReconstruct), on the rough floor traced (no planar view), against one
            // ray a texel: (1) flicker - the floor's mean change a frame, still camera; (2) one frame against the reference, beside the
            // estimator's 16-frame mean (what a perfect reconstruction of its noise would reach); (3) lag - the lamp off, the frames until
            // the floor follows the estimator's mean of the new light; (4) ghosting - the chrome sphere moved away, the frames until its
            // reflection is gone from the floor
            {
                Scene.m_ReflectionMaxRoughness = 0.5f;
                Scene.m_bReflectionJitter = true;
                const auto FloorMean = [&](const rgb_image& I) { double s = 0, n = 0; for (std::size_t i = 0; i < MFloor.size(); ++i) if (MFloor[i]) { s += Lum(&I[i * 3]); ++n; } return s / std::max(n, 1.0); };
                const auto Diff = [&](const rgb_image& A, const rgb_image& B) { double s = 0, n = 0; for (std::size_t i = 0; i < MFloor.size(); ++i) if (MFloor[i]) { s += std::abs(Lum(&A[i * 3]) - Lum(&B[i * 3])); ++n; } return s / std::max(n, 1.0); };
                const auto Flicker = [&](bool bRec, rgb_image& Last) -> double
                {
                    Scene.m_bReflectionReconstruct = bRec;
                    rgb_image A, B;
                    for (int f = 0; f < 30; ++f) if (!App.RenderFrame(Scene, View, Camera)) return -1;
                    if (!ReadRGB(Device, View, A)) return -1;
                    double d = 0;
                    for (int f = 0; f < 8; ++f) { if (!App.RenderFrame(Scene, View, Camera) || !ReadRGB(Device, View, B)) return -1; d += Diff(A, B); A = B; }
                    Last = A;
                    return d / 8;
                };
                rgb_image RawOne, RecOne, RawMean;
                const double FRaw = Flicker(false, RawOne), FRec = Flicker(true, RecOne);
                Scene.m_bReflectionReconstruct = false;
                CHECK(Mean(16, RawMean), "the estimator's mean");
                const double ERaw = RMAE(RawOne, RRef, MFloor, Floor), ERec = RMAE(RecOne, RRef, MFloor, Floor), EMean = RMAE(RawMean, RRef, MFloor, Floor);
                std::printf("milestone D, the rough floor (0.3): change a frame %.5f -> %.5f reconstructed; RMAE of one frame %.3f -> %.3f (the estimator's 16-frame mean %.3f)\n", FRaw, FRec, ERaw, ERec, EMean);
                DumpRGB("refl_d_rough_raw", RawOne, size_v, size_v);
                DumpRGB("refl_d_rough_rec", RecOne, size_v, size_v);
                CHECK(FRec >= 0 && FRec < 0.25 * FRaw, "the reconstruction does not calm the floor (%.5f against %.5f a frame)", FRec, FRaw);
                CHECK(ERec < ERaw && ERec < 1.15 * EMean, "a reconstructed frame is not near the estimator's mean (%.3f; one ray %.3f, the mean %.3f)", ERec, ERaw, EMean);

                // lag: the lamp off - the frames until the floor's mean is within 3% of where it settles, reconstructed and as traced (the
                // GI's own response is in both: the reconstruction's lag is the difference)
                const auto Lag = [&](bool bRec, std::vector<double>& Track) -> int
                {
                    Scene.m_bReflectionReconstruct = bRec;
                    const auto Lamp = Scene.m_LocalLights;
                    rgb_image I;
                    Scene.m_LocalLights.clear();
                    for (int f = 0; f < 90; ++f) if (!App.RenderFrame(Scene, View, Camera)) return -1;
                    if (!ReadRGB(Device, View, I)) return -1;
                    const double Off = FloorMean(I);
                    Scene.m_LocalLights = Lamp;
                    for (int f = 0; f < 90; ++f) if (!App.RenderFrame(Scene, View, Camera)) return -1;
                    Scene.m_LocalLights.clear();
                    int n = -1;
                    for (int f = 1; f <= 16 && n < 0; ++f)
                    {
                        if (!App.RenderFrame(Scene, View, Camera) || !ReadRGB(Device, View, I)) return -1;
                        const double e = std::abs(FloorMean(I) - Off) / std::max(Off, 1e-6);
                        Track.push_back(e);
                        if (e < 0.03) n = f;
                    }
                    Scene.m_LocalLights = Lamp;
                    return n;
                };
                std::vector<double> TRaw, TRec;
                const int LagRaw = Lag(false, TRaw), LagRec = Lag(true, TRec);
                std::printf("milestone D, the lamp off: the floor within 3%% of where it settles after %d frames as traced, %d reconstructed (errors:", LagRaw, LagRec);
                for (double e : TRec) std::printf(" %.3f", e);
                std::printf(")\n");
                CHECK(LagRaw > 0 && LagRec > 0 && LagRec <= LagRaw + 3, "the reconstruction lags the lamp (%d frames, %d as traced)", LagRec, LagRaw);

                // ghosting: the chrome sphere moved 1.9 m aside - its reflection on the floor must leave with it. Where the floor showed it
                // (the floor's texels whose settled images with and without the sphere differ by more than a fifth of the floor's mean),
                // the part of that difference still there each frame after the move (signed toward the sphere's image), reconstructed and
                // as traced: the GI's own response to the move (the sphere's bounce and occlusion fade over many frames) is in both, so the
                // reconstruction's ghost is what it keeps beyond the traced estimator's - the frames until it is within 0.05 of it
                const auto Ghost = [&](bool bRec, std::vector<double>& Track) -> int
                {
                    Scene.m_bReflectionReconstruct = bRec;
                    const auto Was = Scene.m_Draws[chrome_v].m_L2W, Away = xmath::fmat4::fromTranslation({ -1.9f, 0.6f, 0.4f }) * xmath::fmat4::fromScale({ 1.2f, 1.2f, 1.2f });
                    rgb_image Gone, With, Now;
                    Scene.m_Draws[chrome_v].m_L2W = Away;
                    for (int f = 0; f < 120; ++f) if (!App.RenderFrame(Scene, View, Camera)) return -1;
                    if (!ReadRGB(Device, View, Gone)) return -1;
                    Scene.m_Draws[chrome_v].m_L2W = Was;
                    for (int f = 0; f < 120; ++f) if (!App.RenderFrame(Scene, View, Camera)) return -1;
                    if (!ReadRGB(Device, View, With)) return -1;
                    const double Cut = 0.2 * FloorMean(Gone);
                    double Was0 = 0, nM = 0;
                    for (std::size_t i = 0; i < MFloor.size(); ++i) if (MFloor[i] && std::abs(Lum(&With[i * 3]) - Lum(&Gone[i * 3])) > Cut) { Was0 += std::abs(Lum(&With[i * 3]) - Lum(&Gone[i * 3])); ++nM; }
                    Scene.m_Draws[chrome_v].m_L2W = Away;
                    int n = -1;
                    for (int f = 1; f <= 8 && nM > 0; ++f)
                    {
                        if (!App.RenderFrame(Scene, View, Camera) || !ReadRGB(Device, View, Now)) return -1;
                        double Left = 0;
                        for (std::size_t i = 0; i < MFloor.size(); ++i) if (MFloor[i] && std::abs(Lum(&With[i * 3]) - Lum(&Gone[i * 3])) > Cut) Left += (Lum(&Now[i * 3]) - Lum(&Gone[i * 3])) * (Lum(&With[i * 3]) > Lum(&Gone[i * 3]) ? 1.0 : -1.0);     // (signed: the noise cancels)
                        const double e = Left / std::max(Was0, 1e-9);
                        Track.push_back(e);
                        n = f;
                    }
                    Scene.m_Draws[chrome_v].m_L2W = Was;
                    return n;
                };
                std::vector<double> GRaw, GRec;
                const int GhostRaw = Ghost(false, GRaw), GhostRec = Ghost(true, GRec);
                int Kept = -1;
                for (std::size_t f = 0; f < GRec.size() && f < GRaw.size() && Kept < 0; ++f) if (GRec[f] <= GRaw[f] + 0.05) Kept = int(f) + 1;
                std::printf("milestone D, the chrome sphere moved away: its reflection on the floor within 0.05 of the traced estimator's after %d frames (left:", Kept);
                for (double e : GRec) std::printf(" %.2f", e);
                std::printf(" | traced:");
                for (double e : GRaw) std::printf(" %.2f", e);
                std::printf(")\n");
                CHECK(GhostRaw == 8 && GhostRec == 8 && Kept > 0 && Kept <= 2, "the reconstruction ghosts the moved sphere (%d frames)", Kept);
                Scene.m_bReflectionReconstruct = false;
                Scene.m_bReflectionJitter = false;
            }
        }

        // Textures at the hit (the hits' shading: xlionrender_refl_hits_comp.glsl): the back wall's albedo a checker of 32 x 32 squares
        // (0.25 m: finer than its triangles, so the per-triangle capture can only average them) through a real material; the chrome
        // sphere and the floor traced (roughness 0.05, 0.10: both sharp, at full size), one ray a texel, 16 frames' mean, against the
        // reference given the same texture
        {
            static constexpr std::uint32_t s_Frag[] = {
                #include "xlionrender_test_albedo_frag.h"
            };
            xgpu::shader Frag;
            CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "albedo shader");
            static constexpr int tex_v = 256, squares_v = 32;
            const auto Checker = [](float u, float v) -> float { const int a = int(std::floor(u * squares_v)), b = int(std::floor(v * squares_v)); return ((a + b) & 1) ? 0.9f : 0.15f; };
            std::vector<std::uint32_t> Texels;
            std::vector<xgpu::texture::setup::mip_size> Mips;
            {
                std::vector<float> L(std::size_t(tex_v) * tex_v);
                for (int y = 0; y < tex_v; ++y) for (int x = 0; x < tex_v; ++x) L[std::size_t(y) * tex_v + x] = Checker((float(x) + 0.5f) / tex_v, (float(y) + 0.5f) / tex_v);
                for (int n = tex_v; n >= 1; n /= 2)
                {
                    const std::size_t At = Texels.size();
                    for (float l : L) { const auto c = std::uint32_t(std::lround(std::clamp(l, 0.0f, 1.0f) * 255)); Texels.push_back(c | (c << 8) | (c << 16) | 0xff000000u); }
                    Mips.push_back({ std::uint32_t((Texels.size() - At) * 4) });
                    if (n == 1) break;
                    std::vector<float> H(std::size_t(n / 2) * (n / 2));
                    for (int y = 0; y < n / 2; ++y) for (int x = 0; x < n / 2; ++x)
                        H[std::size_t(y) * (n / 2) + x] = 0.25f * (L[std::size_t(2 * y) * n + 2 * x] + L[std::size_t(2 * y) * n + 2 * x + 1] + L[std::size_t(2 * y + 1) * n + 2 * x] + L[std::size_t(2 * y + 1) * n + 2 * x + 1]);
                    L = std::move(H);
                }
            }
            xgpu::texture Tex;
            CHECK(!Device.Create(Tex, { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = tex_v, .m_Height = tex_v, .m_MipSizes = Mips
                                      , .m_AllFacesData = { reinterpret_cast<const std::byte*>(Texels.data()), Texels.size() * 4 }, .m_isGamma = false }), "checker texture");
            xlionrender::material_id Mat;
            const auto Slots = std::array<xgpu::texture*, 2>{ nullptr, &Tex };
            CHECK(!Renderer.CreateMaterial(Mat, Frag, Slots), "checker material");
            static float (*s_Checker)(float, float) = +[](float u, float v) -> float { const int a = int(std::floor(u * squares_v)), b = int(std::floor(v * squares_v)); return ((a + b) & 1) ? 0.9f : 0.15f; };
            Objects[back_v].m_pTexture = s_Checker;
            Objects[floor_v].m_Mat.m_Roughness = 0.05f;
            Scene.m_Draws[floor_v].m_Material.m_Roughness = 0.05f;
            Scene.m_Draws[back_v].m_Materials = { Mat };
            Scene.m_ReflectionMaxRoughness = 0.3f;
            Scene.m_ReflectionPlane = { 0, 0, 0, 0 };
            Scene.m_bReflectionJitter = true;
            Scene.m_bReflectionReconstruct = false;
            for (int f = 0; f < 4; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "the textured wall");
            for (int f = 0; f < 300 && Renderer.isGICapturing(); ++f) CHECK(App.RenderFrame(Scene, View, Camera), "capturing the textured wall");
            for (int f = 0; f < 120; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "settle the textured wall");
            rgb_image TRef(Today.size(), 0.0f);
            for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
            {
                const std::size_t i = std::size_t(y) * size_v + x;
                if (!MChrome[i] && !MFloor[i]) continue;
                const v3 D = Ray(x, y, SX, SY);
                const auto H = World.Cast(Camera.getPosition(), D);
                sample_rng Rng(std::uint32_t(i * 7919 + 5));
                const v3 R = Reference.Reflected(World.HitPosition(H), World.HitNormal(H).NormalizeCopy(), D * -1.0f, Objects[std::size_t(H.m_Inst)].m_Mat, 64, Rng);
                TRef[i * 3] = R.m_X; TRef[i * 3 + 1] = R.m_Y; TRef[i * 3 + 2] = R.m_Z;
            }
            rgb_image Captured, Textured;
            Scene.m_ReflectionTextureMaxRoughness = 0;
            CHECK(Mean(16, Captured), "the hits as captured");
            Scene.m_ReflectionTextureMaxRoughness = 0.3f;
            CHECK(Mean(16, Textured), "the hits shaded");
            const double CC = RMAE(Captured, TRef, MChrome, Floor), CT = RMAE(Textured, TRef, MChrome, Floor), FC = RMAE(Captured, TRef, MFloor, Floor), FT = RMAE(Textured, TRef, MFloor, Floor);
            std::printf("textures at the hit (a 32 x 32 checker on the back wall): chrome RMAE %.3f as captured -> %.3f shaded, floor %.3f -> %.3f\n", CC, CT, FC, FT);
            DumpRGB("refl_tex_ref", TRef, size_v, size_v);
            DumpRGB("refl_tex_captured", Captured, size_v, size_v);
            DumpRGB("refl_tex_shaded", Textured, size_v, size_v);
            CHECK(FT < 0.9 * FC && CT <= CC * 1.02, "the hits' textures do not bring the floor closer to the reference (%.3f against %.3f; chrome %.3f against %.3f)", FT, FC, CT, CC);     // (the chrome sees the wall little: unchanged)

            // The cutoff's sweep (scene::m_ReflectionTextureMaxRoughness): the floor at roughness 0.1 .. 0.5, its reflection as captured and
            // shaded, against the reference - how much the hits' textures still bring as the lobe widens (a table; no gate)
            std::printf("textures at the hit, the floor's roughness against what the shading brings (RMAE as captured -> shaded):");
            for (const float Rho : { 0.10f, 0.20f, 0.30f, 0.40f, 0.50f })
            {
                Objects[floor_v].m_Mat.m_Roughness = Rho;
                Scene.m_Draws[floor_v].m_Material.m_Roughness = Rho;
                Scene.m_ReflectionMaxRoughness = 0.6f;
                rgb_image SRef(Today.size(), 0.0f), SC, ST;
                for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
                {
                    const std::size_t i = std::size_t(y) * size_v + x;
                    if (!MFloor[i]) continue;
                    const v3 D = Ray(x, y, SX, SY);
                    const auto H = World.Cast(Camera.getPosition(), D);
                    sample_rng Rng(std::uint32_t(i * 7919 + 11));
                    const v3 R = Reference.Reflected(World.HitPosition(H), World.HitNormal(H).NormalizeCopy(), D * -1.0f, Objects[floor_v].m_Mat, 64, Rng);
                    SRef[i * 3] = R.m_X; SRef[i * 3 + 1] = R.m_Y; SRef[i * 3 + 2] = R.m_Z;
                }
                Scene.m_ReflectionTextureMaxRoughness = 0;
                CHECK(Mean(16, SC), "the sweep, captured");
                Scene.m_ReflectionTextureMaxRoughness = 1.0f;
                CHECK(Mean(16, ST), "the sweep, shaded");
                std::printf("  %.1f: %.3f -> %.3f", Rho, RMAE(SC, SRef, MFloor, Floor), RMAE(ST, SRef, MFloor, Floor));
            }
            std::printf("\n");

            // The screen trace (scene::m_bReflectionScreen: SSR over the probes, the BVH trace off) on the near-mirror floor: where the
            // floor's mirror ray meets a sphere (on screen) its reflection changes - the sphere where it is, not the probes' view -, where
            // it meets nothing (the sky: off screen) the probes stay; its coverage reconstructed (no hit / miss speckle: the floor's change
            // a frame stays small). (Against the path-traced reference the floor's error is the lighting's - the lamp against the GI)
            {
                Objects[floor_v].m_Mat.m_Roughness = 0.05f;
                Scene.m_Draws[floor_v].m_Material.m_Roughness = 0.05f;
                Scene.m_ReflectionMaxRoughness = 0;
                Scene.m_bReflectionJitter = true;
                Scene.m_bReflectionReconstruct = true;
                std::vector<std::uint8_t> MSphere(MFloor.size(), 0), MSky(MFloor.size(), 0);
                for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
                {
                    const std::size_t i = std::size_t(y) * size_v + x;
                    if (!MFloor[i]) continue;
                    const v3 D = Ray(x, y, SX, SY);
                    const auto H = World.Cast(Camera.getPosition(), D);
                    const v3 N = World.HitNormal(H).NormalizeCopy();
                    const v3 R = D - N * (2.0f * D.Dot(N));
                    const auto M = World.Cast(World.HitPosition(H) + N * 1e-3f, R);
                    if (M.m_Inst == chrome_v || M.m_Inst == gold_v) MSphere[i] = 1;
                    else if (M.m_Inst < 0) MSky[i] = 1;
                }
                const auto Settled = [&](bool bScreen, rgb_image& I) -> double
                {
                    Scene.m_bReflectionScreen = bScreen;
                    rgb_image A, B;
                    for (int f = 0; f < 30; ++f) if (!App.RenderFrame(Scene, View, Camera)) return -1;
                    if (!ReadRGB(Device, View, A)) return -1;
                    double d = 0, n = 0;
                    for (int f = 0; f < 4; ++f)
                    {
                        if (!App.RenderFrame(Scene, View, Camera) || !ReadRGB(Device, View, B)) return -1;
                        for (std::size_t i = 0; i < MFloor.size(); ++i) if (MFloor[i]) { d += std::abs(Lum(&A[i * 3]) - Lum(&B[i * 3])); ++n; }
                        A = B;
                    }
                    I = A;
                    return d / std::max(n, 1.0);
                };
                rgb_image SP, SS;
                const double FP = Settled(false, SP), FS = Settled(true, SS);
                const auto Change = [&](const std::vector<std::uint8_t>& Mk) { double s = 0, n = 0; for (std::size_t i = 0; i < Mk.size(); ++i) if (Mk[i]) { s += std::abs(Lum(&SS[i * 3]) - Lum(&SP[i * 3])); ++n; } return std::pair{ s / std::max(n, 1.0), n }; };
                const auto [CSphere, nSphere] = Change(MSphere);
                const auto [CSky, nSky] = Change(MSky);
                std::printf("the screen trace (SSR), the floor at roughness 0.05: its change where the mirror ray meets a sphere %.5f (%.0f px), meets nothing %.5f (%.0f px); change a frame %.5f -> %.5f\n", CSphere, nSphere, CSky, nSky, FP, FS);
                DumpRGB("refl_ssr_off", SP, size_v, size_v);
                DumpRGB("refl_ssr_on", SS, size_v, size_v);
                CHECK(FP >= 0 && FS >= 0 && nSphere > 50 && nSky > 50, "the screen trace's floor masks (%.0f, %.0f px)", nSphere, nSky);
                CHECK(CSphere > 4.0 * CSky, "the screen trace does not show the spheres in the floor (change %.5f there, %.5f where it meets nothing)", CSphere, CSky);
                CHECK(FS < 0.02, "the screen trace's floor flickers (%.5f a frame)", FS);
                Scene.m_bReflectionScreen = false;
                Scene.m_bReflectionReconstruct = false;
            }
            Objects[floor_v].m_Mat.m_Roughness = 0.10f;
            Scene.m_Draws[floor_v].m_Material.m_Roughness = 0.10f;
            Scene.m_ReflectionTextureMaxRoughness = 0.3f;
            Scene.m_Draws[back_v].m_Materials = {};
            Objects[back_v].m_pTexture = nullptr;
            Scene.m_bReflectionJitter = false;
            for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "the wall again");
            Renderer.DestroyMaterial(Mat);
            Device.Destroy(std::move(Tex));
        }
        Scene.m_ReflectionMaxRoughness = 0;
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
