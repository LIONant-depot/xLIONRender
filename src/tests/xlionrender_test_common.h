#ifndef XLIONRENDER_TEST_COMMON_H
#define XLIONRENDER_TEST_COMMON_H
#pragma once

// Shared by the xLIONRender tests: a window + renderer, validation-message counting, HDR readback, a test sphere.

#include "xlionrender.h"
#include "xlionrender_test_no_dialogs.h"
#include "dependencies/xprim_geom/source/xprim_geom.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <algorithm>
#include <array>

#define CHECK(COND, ...) do { if (!(COND)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); return 1; } } while(0)

namespace xlionrender::test
{
    inline int g_nErrors = 0;
    inline void LogError  (std::string_view Msg) { ++g_nErrors; std::printf("[error] %.*s\n", int(Msg.size()), Msg.data()); }
    inline void LogWarning(std::string_view Msg) { std::printf("[warn] %.*s\n", int(Msg.size()), Msg.data()); }

    inline float HalfToFloat(std::uint16_t H) noexcept
    {
        const int   Exp = (H >> 10) & 0x1f, Mant = H & 0x3ff;
        const float S   = (H >> 15) ? -1.0f : 1.0f;
        if (Exp == 0)  return S * std::ldexp(float(Mant), -24);
        if (Exp == 31) return Mant ? NAN : S * INFINITY;
        return S * std::ldexp(float(Mant | 0x400), Exp - 25);
    }

    // Red channel of the HDR target as floats (row 0 = top of the image)
    inline bool ReadHDR(xgpu::device& Device, const render_view& View, std::vector<float>& Red)
    {
        std::vector<std::byte> Raw;
        int W, H;
        if (Device.ReadTexture(View.m_HDR, Raw, W, H)) return false;
        const auto* p = reinterpret_cast<const std::uint16_t*>(Raw.data());
        Red.resize(std::size_t(W) * H);
        for (std::size_t i = 0; i < Red.size(); ++i) Red[i] = HalfToFloat(p[i * 4]);
        return true;
    }

    // To look at a ReadHDR image: with XLIONRENDER_TEST_DUMP=<folder>, writes <folder>/<Name>.pgm (red, x / (1 + x) to 8 bits)
    inline void Dump(const char* pName, const std::vector<float>& Red) noexcept
    {
        const char* pDir = std::getenv("XLIONRENDER_TEST_DUMP");
        if (pDir == nullptr) return;
        const int Side = int(std::sqrt(double(Red.size())));
        if (FILE* f = std::fopen((std::string(pDir) + "/" + pName + ".pgm").c_str(), "wb"))
        {
            std::fprintf(f, "P5 %d %d 255\n", Side, Side);
            for (float V : Red) std::fputc(int(255.0f * std::max(V, 0.0f) / (1.0f + std::max(V, 0.0f))), f);
            std::fclose(f);
        }
    }

    inline xprim_geom::mesh MakeSphere() noexcept { return xprim_geom::uvsphere::Generate(32, 32, 1, 0.5f); }

    // A mesh's surfels for GI (gi::surfel_mesh::Bake) in its vertex order - what the xgeom compiler's GI options will bake
    inline gi::surfel_mesh MakeSurfels(const xprim_geom::mesh& Prim, float Radius, float ParentRadius) noexcept
    {
        gi::surfel_mesh S;
        for (const auto& V : Prim.m_Vertices)
        {
            S.m_Pos.push_back({ V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z });
            S.m_Nrm.push_back({ V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z });
        }
        for (std::size_t i = 0; i + 2 < Prim.m_Indices.size(); i += 3) S.m_Tris.push_back({ int(Prim.m_Indices[i]), int(Prim.m_Indices[i + 1]), int(Prim.m_Indices[i + 2]) });
        S.Bake(Radius, ParentRadius);
        return S;
    }

    // Octahedral encoding into [0,1]^2, the inverse of oct_decode in the xgeom_* vertex shaders
    inline void OctEncode(float X, float Y, float Z, float& U, float& V) noexcept
    {
        const float L = std::abs(X) + std::abs(Y) + std::abs(Z);
        X /= L; Y /= L;
        if (Z < 0)
        {
            const float OX = X;
            X = (1 - std::abs(Y))  * (OX >= 0 ? 1.f : -1.f);
            Y = (1 - std::abs(OX)) * (Y  >= 0 ? 1.f : -1.f);
        }
        U = X * 0.5f + 0.5f;
        V = Y * 0.5f + 0.5f;
    }

    inline std::uint32_t Quantize(float V01, std::uint32_t Max) noexcept { return static_cast<std::uint32_t>(std::lround(std::clamp(V01, 0.f, 1.f) * Max)); }


    // A mesh in the GPU format of the xgeom_* loaders: index, vertex, extras (contiguous, as setStreamingBuffers takes them) + the
    // cluster SSBO. One cluster; positions must lie in [-Scale, Scale].
    struct cluster_buffers
    {
        std::array<xgpu::buffer, 3>     m_Streams;
        xgpu::buffer                    m_ClusterData;
        xgpu::buffer                    m_Colors;                // the optional COLOR0 stream (SetColor); empty: the renderer binds white
        xgpu::buffer                    m_UV1;                   // the optional UV1 stream (SetUV1); empty: the shaders read UV0
        std::uint32_t                   m_nIndices = 0;
        float                           m_Scale    = 1;         // positions lie in [-m_Scale, m_Scale]

        cluster_geom Geom(geom_kind Kind = geom_kind::RIGID, xgpu::buffer* pBones = nullptr, std::uint32_t MaxInfluences = 1) noexcept
        {
            cluster_geom G{ .m_Kind = Kind, .m_pStreams = m_Streams.data(), .m_pColors = m_Colors.m_Private ? &m_Colors : nullptr, .m_pUV1 = m_UV1.m_Private ? &m_UV1 : nullptr, .m_pClusterData = &m_ClusterData, .m_pBones = pBones, .m_MaxInfluences = MaxInfluences };
            G.m_BBox.m_Min = xmath::fvec3(-m_Scale);
            G.m_BBox.m_Max = xmath::fvec3( m_Scale);
            G.m_Clusters  = { { 0, m_nIndices, 0, G.m_BBox } };       // (its box: the GPU culls by it)
            G.m_Submeshes = { { 0, 1, 0 } };
            return G;
        }

        // Every vertex the same RGBA8 (sRGB) color, nVertices of them
        bool SetColor(xgpu::device& Device, std::size_t nVertices, std::uint32_t RGBA) noexcept
        {
            std::vector<std::uint32_t> Colors(nVertices, RGBA);
            return !Device.Create(m_Colors, { .m_Type = xgpu::buffer::type::VERTEX, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = int(Colors.size()), .m_pData = Colors.data() });
        }

        // Every vertex the same UV1, nVertices of them
        bool SetUV1(xgpu::device& Device, std::size_t nVertices, float U, float V) noexcept
        {
            std::vector<std::array<float, 2>> UVs(nVertices, { U, V });
            return !Device.Create(m_UV1, { .m_Type = xgpu::buffer::type::VERTEX, .m_EntryByteSize = sizeof(UVs[0]), .m_EntryCount = int(UVs.size()), .m_pData = UVs.data() });
        }
    };

    namespace details
    {
        struct cluster_data { float m_PosScale[4]; float m_PosTranslation[4]; float m_UVScaleTranslation[4]; };     // posScale.w: base bone (skin)

        inline std::int16_t QuantizePos(float P, float Scale) noexcept { return static_cast<std::int16_t>(std::clamp(std::lround((P / Scale + 1.0f) * 32767.5f - 32768.0f), -32768l, 32767l)); }

        template<typename T_VERT, typename T_EXTRA>
        bool Upload(xgpu::device& Device, const xprim_geom::mesh& Prim, const std::vector<T_VERT>& Pos, const std::vector<T_EXTRA>& Extras, float Scale, cluster_buffers& Out) noexcept
        {
            std::vector<std::uint16_t> Idx16(Prim.m_Indices.begin(), Prim.m_Indices.end());
            const cluster_data Cluster{ { Scale, Scale, Scale, 0 }, { 0, 0, 0, 0 }, { 1, 1, 0, 0 } };
            Out.m_nIndices = static_cast<std::uint32_t>(Idx16.size());
            Out.m_Scale    = Scale;
            return !Device.Create(Out.m_Streams[0],  { .m_Type = xgpu::buffer::type::INDEX,   .m_EntryByteSize = sizeof(std::uint16_t), .m_EntryCount = int(Idx16.size()),  .m_pData = Idx16.data() })
                && !Device.Create(Out.m_Streams[1],  { .m_Type = xgpu::buffer::type::VERTEX,  .m_EntryByteSize = sizeof(T_VERT),        .m_EntryCount = int(Pos.size()),    .m_pData = const_cast<T_VERT*>(Pos.data()) })
                && !Device.Create(Out.m_Streams[2],  { .m_Type = xgpu::buffer::type::VERTEX,  .m_EntryByteSize = sizeof(T_EXTRA),       .m_EntryCount = int(Extras.size()), .m_pData = const_cast<T_EXTRA*>(Extras.data()) })
                && !Device.Create(Out.m_ClusterData, { .m_Type = xgpu::buffer::type::STORAGE, .m_EntryByteSize = sizeof(cluster_data),  .m_EntryCount = 1, .m_pData = const_cast<cluster_data*>(&Cluster) });
        }
    }

    // xgeom_static::geom layout (xgeom_static.plugin/source/xgeom_static.h): int16 position + 16 bits of low normal/tangent bits and the
    // binormal sign; extras: uint16 UV, 8-bit high halves of the 12-bit oct normal and 12/11-bit oct tangent
    inline bool MakeStaticBuffers(xgpu::device& Device, const xprim_geom::mesh& Prim, float Scale, cluster_buffers& Out) noexcept
    {
        struct vert  { std::int16_t m_X, m_Y, m_Z, m_Extra; };
        struct extra { std::uint16_t m_UV[2]; std::uint8_t m_OctNormal[2]; std::uint8_t m_OctTangent[2]; };
        std::vector<vert>  Pos(Prim.m_Vertices.size());
        std::vector<extra> Extras(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Pos.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            float NU, NV, TU, TV;
            OctEncode(V.m_Normal.m_X,  V.m_Normal.m_Y,  V.m_Normal.m_Z,  NU, NV);
            OctEncode(V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, TU, TV);
            const std::uint32_t Nx = Quantize(NU, 4095), Ny = Quantize(NV, 4095), Tx = Quantize(TU, 4095), Ty = Quantize(TV, 2047);
            const std::uint32_t Extra = (Nx & 15) | ((Ny & 15) << 4) | ((Tx & 15) << 8) | ((Ty & 7) << 12) | ((V.m_Tangent.m_D < 0 ? 1u : 0u) << 15);
            Pos[i]    = { details::QuantizePos(V.m_Position.m_X, Scale), details::QuantizePos(V.m_Position.m_Y, Scale), details::QuantizePos(V.m_Position.m_Z, Scale), static_cast<std::int16_t>(static_cast<std::uint16_t>(Extra)) };
            Extras[i] = { { static_cast<std::uint16_t>(Quantize(V.m_Texcoord.m_X, 65535)), static_cast<std::uint16_t>(Quantize(V.m_Texcoord.m_Y, 65535)) }
                        , { static_cast<std::uint8_t>(Nx >> 4), static_cast<std::uint8_t>(Ny >> 4) }
                        , { static_cast<std::uint8_t>(Tx >> 4), static_cast<std::uint8_t>(Ty >> 3) } };
        }
        return details::Upload(Device, Prim, Pos, Extras, Scale, Out);
    }

    // xgeom_skin::geom layout (xgeom_skin.plugin/source/xgeom_skin.h): int16 position + 6 packed bytes (bone offsets and weights; all zero:
    // bone 0 of the cluster, which with one influence weighs 1); extras: uint16 UV, 16-bit oct normal, 16-bit oct tangent X, 15-bit Y + sign
    inline bool MakeSkinBuffers(xgpu::device& Device, const xprim_geom::mesh& Prim, float Scale, cluster_buffers& Out) noexcept
    {
        struct vert  { std::int16_t m_X, m_Y, m_Z; std::uint8_t m_Packed[6]; };
        struct extra { std::uint16_t m_UV[2]; std::uint16_t m_OctNormal[2]; std::uint16_t m_OctTangentX; std::uint16_t m_OctTangentY_Sign; };
        static_assert(sizeof(vert) == 12 && sizeof(extra) == 12);
        std::vector<vert>  Pos(Prim.m_Vertices.size());
        std::vector<extra> Extras(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Pos.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            float NU, NV, TU, TV;
            OctEncode(V.m_Normal.m_X,  V.m_Normal.m_Y,  V.m_Normal.m_Z,  NU, NV);
            OctEncode(V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, TU, TV);
            Pos[i]    = { details::QuantizePos(V.m_Position.m_X, Scale), details::QuantizePos(V.m_Position.m_Y, Scale), details::QuantizePos(V.m_Position.m_Z, Scale), {} };
            Extras[i] = { { static_cast<std::uint16_t>(Quantize(V.m_Texcoord.m_X, 65535)), static_cast<std::uint16_t>(Quantize(V.m_Texcoord.m_Y, 65535)) }
                        , { static_cast<std::uint16_t>(Quantize(NU, 65535)), static_cast<std::uint16_t>(Quantize(NV, 65535)) }
                        , static_cast<std::uint16_t>(Quantize(TU, 65535))
                        , static_cast<std::uint16_t>(Quantize(TV, 32767) | (V.m_Tangent.m_D < 0 ? 0x8000u : 0u)) };
        }
        return details::Upload(Device, Prim, Pos, Extras, Scale, Out);
    }

    // A material made of one fragment shader (SPIR-V words) with no texture of its own: slot 0, the shadow map, is left white
    template<std::size_t N>
    bool MakeTestMaterial(xgpu::device& Device, instance& Renderer, const std::uint32_t(&Words)[N], xgpu::shader& Shader, material_id& Id) noexcept
    {
        if (Device.Create(Shader, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(Words), N } } })) return false;
        static xgpu::texture* const s_NoShadow[] = { nullptr };
        return Renderer.CreateMaterial(Id, Shader, s_NoShadow) == nullptr;
    }

    // Window, device and renderer, plus one frame of rendering
    struct app
    {
        xgpu::instance          m_Instance;
        xgpu::device            m_Device;
        xgpu::window            m_Window;
        xlionrender::instance   m_Renderer;
        bool                    m_bGotTimes = false;

        bool Init()
        {
            if (xgpu::CreateInstance(m_Instance, { .m_bDebugMode = true, .m_bEnableRenderDoc = false, .m_pLogErrorFunc = LogError, .m_pLogWarning = LogWarning })) return false;
            if (m_Instance.Create(m_Device))                                    return false;
            if (m_Device.Create(m_Window, { .m_Width = 256, .m_Height = 256 })) return false;
            return m_Renderer.Init(m_Device) == nullptr;
        }

        instance::gpu_times m_LastTimes;
        bool                m_bQuietTimes = false;

        bool RenderFrame(const scene& Scene, render_view& View, xgpu::tools::view& Camera, const instance::overlay_fn& Overlay = {})
        {
            if (!m_Instance.ProcessInputEvents()) return false;
            while (m_Window.BeginRendering()) if (!m_Instance.ProcessInputEvents()) return false;
            m_Renderer.Render(m_Window, Scene, View, Camera, Overlay);
            { auto Cmd = m_Window.getCmdBuffer(); }
            m_Window.PageFlip();
            instance::gpu_times T;
            if (m_Renderer.getGPUTimes(T)) { m_bGotTimes = true; m_LastTimes = T; if (!m_bQuietTimes) std::printf("GPU: scene %.3f ms, tone map %.3f ms\n", T.m_SceneMs, T.m_ToneMapMs); }
            return true;
        }
    };
}

#endif
