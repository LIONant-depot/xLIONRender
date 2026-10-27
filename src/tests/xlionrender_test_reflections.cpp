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
        for (const auto& V : Prim.m_Vertices) { M.m_Pos.push_back({ V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }); M.m_Nrm.push_back({ V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }); }
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

    struct object { const xlionrender::gi::surfel_mesh* m_pRay; xlionrender::mesh_id m_Mesh; xmath::fmat4 m_L2W; xlionrender::material m_Mat; };

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
            return Shade(m_World.HitPosition(H), N, D * -1.0f, m_Objects[std::size_t(H.m_Inst)].m_Mat, Depth, Rng);
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

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
