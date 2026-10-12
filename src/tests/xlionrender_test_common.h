#ifndef XLIONRENDER_TEST_COMMON_H
#define XLIONRENDER_TEST_COMMON_H
#pragma once

// Shared by the xLIONRender tests: a window + renderer, validation-message counting, HDR readback, a test sphere.

#include "xlionrender.h"
#include "dependencies/xprim_geom/source/xprim_geom.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

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

    inline xprim_geom::mesh MakeSphere() noexcept { return xprim_geom::uvsphere::Generate(32, 32, 1, 0.5f); }

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

        bool RenderFrame(const scene& Scene, render_view& View, xgpu::tools::view& Camera, const instance::overlay_fn& Overlay = {})
        {
            if (!m_Instance.ProcessInputEvents()) return false;
            while (m_Window.BeginRendering()) if (!m_Instance.ProcessInputEvents()) return false;
            m_Renderer.Render(m_Window, Scene, View, Camera, Overlay);
            { auto Cmd = m_Window.getCmdBuffer(); }
            m_Window.PageFlip();
            instance::gpu_times T;
            if (m_Renderer.getGPUTimes(T)) { m_bGotTimes = true; std::printf("GPU: scene %.3f ms, tone map %.3f ms\n", T.m_SceneMs, T.m_ToneMapMs); }
            return true;
        }
    };
}

#endif
