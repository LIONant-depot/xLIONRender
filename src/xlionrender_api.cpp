#include "xlionrender_api.h"
#include "xlionrender_renderer.h"
#include "xlionrender_internal.h"
#include "xlionrender_system.h"
#include <algorithm>
#include <cstring>
#include <format>
#include <string>
#include <unordered_map>

namespace xlionrender
{
    inline renderer g_Renderer;
    inline xgpu::device* g_pDevice = nullptr;           // the host's device (Init), for the resource system of the core

    // One render system per live world (every open Level has its own xecs::game_mgr::instance); the host names the
    // world it is drawing/picking for by that instance's address.
    inline std::unordered_map<const void*, system*> g_Systems;

    void SetActiveSystemInternal(const void* pWorld, system* pSystem) noexcept
    {
        if (pSystem) g_Systems[pWorld] = pSystem;
        else         g_Systems.erase(pWorld);
    }

    static system* FindSystem(const void* pWorld) noexcept
    {
        auto It = g_Systems.find(pWorld);
        return It == g_Systems.end() ? nullptr : It->second;
    }

    bool Init(xgpu::device& Device) noexcept
    {
        g_pDevice = &Device;
        return g_Renderer.Init(Device) && g_TextRenderer.Init(Device);
    }

    xgpu::device* GetDeviceInternal() noexcept { return g_pDevice; }

    void Draw(const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept
    {
        if (auto* pSystem = FindSystem(pWorld)) pSystem->Collect();
        g_Renderer.Draw(CmdBuffer, W2C, ViewportW, ViewportH);
        g_TextRenderer.Draw(CmdBuffer, W2C, ViewportW, ViewportH);          // after the solids: the text is blended over them
    }

    std::uint64_t Pick(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT) noexcept
    {
        auto* pSystem = FindSystem(pWorld);
        return pSystem ? pSystem->Pick(Origin, Dir, MaxT) : xecs::component::entity::invalid_entity_v;
    }

    void SetSelectedEntity(std::uint64_t EntityValue) noexcept
    {
        g_Renderer.SetSelected(EntityValue);
    }

    namespace
    {
        struct editor_impl final : xRenderEditor
        {
            std::uint32_t Version() const noexcept override { return kVersion; }
            void          Release() noexcept override { delete this; }
            bool          Init(xgpu::device& Device) noexcept override { return xlionrender::Init(Device); }
            void          Draw(const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float W, float H) noexcept override { xlionrender::Draw(pWorld, CmdBuffer, W2C, W, H); }
            std::uint64_t Pick(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT) noexcept override { return xlionrender::Pick(pWorld, Origin, Dir, MaxT); }
            void          SetSelectedEntity(std::uint64_t EntityValue) noexcept override { xlionrender::SetSelectedEntity(EntityValue); }
            int           DescribeText(const void* pWorld, std::uint64_t EntityValue, char* pOut, int Capacity) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                if (!pSystem || !pOut || Capacity <= 0) return -1;
                std::string Text;
                if (!pSystem->DescribeText(EntityValue, Text)) return -1;
                const int Length = static_cast<int>(std::min<std::size_t>(Text.size(), static_cast<std::size_t>(Capacity - 1)));
                std::memcpy(pOut, Text.data(), static_cast<std::size_t>(Length));
                pOut[Length] = 0;
                return Length;
            }
            int           DescribeTextDraw(char* pOut, int Capacity) noexcept override
            {
                if (!pOut || Capacity <= 0) return -1;
                const auto& S = g_TextRenderer.getLastStats();
                const std::string Text = std::format("DescribeTextDraw: ok\nLabels={}\nGlyphs={}\nDraws={}\nDropped={}",S.m_Labels, S.m_Glyphs, S.m_Draws, S.m_Dropped);
                const int Length = static_cast<int>(std::min<std::size_t>(Text.size(), static_cast<std::size_t>(Capacity - 1)));
                std::memcpy(pOut, Text.data(), static_cast<std::size_t>(Length));
                pOut[Length] = 0;
                return Length;
            }
        };
    }

    // Not exported (no header declares it outside this DLL) - only system::Collect, compiled into
    // this same DLL (xlionrender_system.h), ever calls this. "No one submits shapes except internally
    // to this DLL" - this is the one function that makes that literally true.
    void SubmitInternal(shape Shape, const xmath::fmat4& L2W, const xmath::fvec3& Scale, const xmath::fvec3& Color, std::uint64_t EntityValue) noexcept
    {
        g_Renderer.Submit(Shape, L2W, Scale, Color, EntityValue);
    }
}

extern "C" __declspec(dllexport) xlionrender::xRenderEditor* XLionRender_CreateEditor() noexcept
{
    return new xlionrender::editor_impl;
}
