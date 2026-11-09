#include "xlionrender_api.h"
#include "xlionrender_renderer.h"
#include "xlionrender_internal.h"
#include "xlionrender_system.h"
#include "xprofile.h"
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
        XPROFILE_MEM_OWNER("LIONRender: the editor's renderer");
        return g_Renderer.Init(Device) && g_TextRenderer.Init(Device);
    }

    xgpu::device* GetDeviceInternal() noexcept { return g_pDevice; }
    std::uint64_t GetSelectedInternal() noexcept { return g_Renderer.getSelected(); }

    void Draw(const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH, view View, const roles* pRoles) noexcept
    {
        XPROFILE_SCOPE("LIONRender Draw");
        XPROFILE_MEM_OWNER("LIONRender: the editor's renderer");
        if (auto* pSystem = FindSystem(pWorld)) pSystem->Collect(View, pRoles);
        g_Renderer.Draw(CmdBuffer, W2C, ViewportW, ViewportH, pRoles);
        g_TextRenderer.Draw(CmdBuffer, W2C, ViewportW, ViewportH);          // after the solids: the text is blended over them
    }

    std::uint64_t Pick(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT, view View, const roles* pRoles) noexcept
    {
        auto* pSystem = FindSystem(pWorld);
        return pSystem ? pSystem->Pick(Origin, Dir, MaxT, View, pRoles) : xecs::component::entity::invalid_entity_v;
    }

    bool EntityBox(const void* pWorld, std::uint64_t EntityValue, xmath::fvec3& Center, xmath::fquat& Rotation, xmath::fvec3& HalfExtents) noexcept
    {
        auto* pSystem = FindSystem(pWorld);
        return pSystem && pSystem->EntityBox(EntityValue, Center, Rotation, HalfExtents);
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
            void          Draw(const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float W, float H, view View) noexcept override { xlionrender::Draw(pWorld, CmdBuffer, W2C, W, H, View); }
            std::uint64_t Pick(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT, view View) noexcept override { return xlionrender::Pick(pWorld, Origin, Dir, MaxT, View); }
            void          SetSelectedEntity(std::uint64_t EntityValue) noexcept override { xlionrender::SetSelectedEntity(EntityValue); }
            bool          EntityBox(const void* pWorld, std::uint64_t EntityValue, xmath::fvec3& Center, xmath::fquat& Rotation, xmath::fvec3& HalfExtents) noexcept override { return xlionrender::EntityBox(pWorld, EntityValue, Center, Rotation, HalfExtents); }
            void          DrawRoles(const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float W, float H, view View, const roles& Roles) noexcept override { xlionrender::Draw(pWorld, CmdBuffer, W2C, W, H, View, &Roles); }
            std::uint64_t PickRoles(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT, view View, const roles& Roles) noexcept override { return xlionrender::Pick(pWorld, Origin, Dir, MaxT, View, &Roles); }
            int           DescribeRoleDraw(char* pOut, int Capacity) noexcept override
            {
                if (!pOut || Capacity <= 0) return -1;
                const auto& S = g_Renderer.getLastRoleStats();
                const std::string Text = std::format("DescribeRoleDraw: ok\nDrawnContext={}\nDrawnDocument={}\nFaded={}", S.m_Context, S.m_Document, S.m_bFaded ? 1 : 0);
                const int Length = static_cast<int>(std::min<std::size_t>(Text.size(), static_cast<std::size_t>(Capacity - 1)));
                std::memcpy(pOut, Text.data(), static_cast<std::size_t>(Length));
                pOut[Length] = 0;
                return Length;
            }
            void          RenderScene(const void* pWorld, xgpu::window& Window, xgpu::tools::view& Camera, int W, int H, view View, overlay_fn pOverlay, void* pUser, const roles* pRoles) noexcept override
            {
                if (auto* pSystem = FindSystem(pWorld)) pSystem->RenderScene(Window, Camera, W, H, View, pOverlay, pUser, pRoles);
            }
            void          BlitScene(const void* pWorld, xgpu::cmd_buffer& CmdBuffer) noexcept override
            {
                if (auto* pSystem = FindSystem(pWorld)) pSystem->BlitScene(CmdBuffer);
            }
            int           DescribeScene(const void* pWorld, char* pOut, int Capacity) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                if (!pSystem || !pOut || Capacity <= 0) return -1;
                const std::string Text = pSystem->DescribeScene();
                const int Length = static_cast<int>(std::min<std::size_t>(Text.size(), static_cast<std::size_t>(Capacity - 1)));
                std::memcpy(pOut, Text.data(), static_cast<std::size_t>(Length));
                pOut[Length] = 0;
                return Length;
            }
            void          SetViewOptions(const void* pWorld, const view_options& Options) noexcept override
            {
                if (auto* pSystem = FindSystem(pWorld)) pSystem->m_ViewOptions = Options;
            }
            float         GITraceMs(const void* pWorld) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                return pSystem ? pSystem->GITraceMs() : 0.0f;
            }
            int           NotDrawnBoxes(const void* pWorld, wire_box* pOut, int MaxBoxes) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                if (!pSystem || !pSystem->m_pScene) return 0;
                const auto Boxes = pSystem->m_pScene->m_Renderer.getNotDrawn();
                if (pOut && MaxBoxes > 0) std::copy_n(Boxes.begin(), std::min<std::size_t>(Boxes.size(), static_cast<std::size_t>(MaxBoxes)), pOut);
                return static_cast<int>(Boxes.size());
            }
            int           OldMaterials(const void* pWorld, std::uint64_t* pOut, int MaxEntities) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                if (!pSystem) return 0;
                const auto& V = pSystem->m_OldMaterials;
                if (pOut && MaxEntities > 0) std::copy_n(V.begin(), std::min<std::size_t>(V.size(), static_cast<std::size_t>(MaxEntities)), pOut);
                return static_cast<int>(V.size());
            }
            std::uint64_t PickNoMeshes(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT, view View) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                return pSystem ? pSystem->Pick(Origin, Dir, MaxT, View, nullptr, false) : ~0ull;
            }
            int           DescribeGraph(const void* pWorld, char* pOut, int Capacity) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                if (!pSystem) return -1;
                const std::string Text = pSystem->m_pScene ? pSystem->m_pScene->m_View.m_Graph.Text() : std::string();
                if (pOut && Capacity > 0)
                {
                    const std::size_t n = std::min<std::size_t>(Text.size(), static_cast<std::size_t>(Capacity - 1));
                    std::memcpy(pOut, Text.data(), n);
                    pOut[n] = 0;
                }
                return static_cast<int>(Text.size());
            }
            int           DescribePost(const void* pWorld, char* pOut, int Capacity) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                if (!pSystem) return -1;
                std::string Text;
                if (pSystem->m_pScene) { const auto& V = pSystem->m_pScene->m_View; Text = post::Describe(V.m_PostDesc, V.m_PostPlan, V.m_PostError, V.m_PostSnapshot, V.m_PostNoted, V.m_PostRan); }
                if (pOut && Capacity > 0)
                {
                    const std::size_t n = std::min<std::size_t>(Text.size(), static_cast<std::size_t>(Capacity - 1));
                    std::memcpy(pOut, Text.data(), n);
                    pOut[n] = 0;
                }
                return static_cast<int>(Text.size());
            }
            bool          EditPost(const void* pWorld, const char* pEdit, char* pOut, int Capacity) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                std::string Reply; bool bOk = false;
                if (!pSystem || !pSystem->m_pScene) Reply = "no such world";
                else if (std::string_view E(pEdit ? pEdit : ""); E.starts_with("insertable "))
                {
                    for (const auto T : post::Insertable(pSystem->m_pScene->m_View.m_PostDesc, std::atoi(pEdit + 11))) Reply += std::string(post::node_names_v[std::size_t(T)]) + " ";
                    bOk = true;
                }
                else if (E == "source") { Reply = std::format("{:016X}", pSystem->m_pScene->m_RenderGraph); bOk = true; }      // (the RenderGraph the chain comes from; 0: the built-in one)
                else if (E == "describe") { const auto& V = pSystem->m_pScene->m_View; Reply = post::Describe(V.m_PostDesc, V.m_PostPlan, V.m_PostError, V.m_PostSnapshot, V.m_PostNoted, V.m_PostRan); bOk = true; }      // (the Post chain view's text, for the console)
                else if (E == "stages")         // (the frame stages: the ones bypassed, then every scope the last frame ran - a line each)
                {
                    const auto& V = pSystem->m_pScene->m_View;
                    Reply = "off:"; for (const auto& S : V.m_StagesOff) Reply += " " + S;
                    const auto T = V.m_Graph.Text();
                    for (std::size_t b = 0; b < T.size(); )
                    {
                        std::size_t e = T.find('\n', b); if (e == std::string::npos) e = T.size();
                        if (T.compare(b, 2, "S\t") == 0) { const auto f1 = T.find('\t', b + 2), f2 = T.find('\t', f1 + 1); if (f1 < e) Reply += "\n" + T.substr(f1 + 1, std::min(f2, e) - f1 - 1); }
                        b = e + 1;
                    }
                    bOk = true;
                }
                else if (E.starts_with("stage "))      // "stage <name> 0|1" on the view (a Level with a RenderGraph: the session edits its descriptor instead)
                {
                    char Name[64] = {}; int On = 1;
                    auto& V = pSystem->m_pScene->m_View;
                    if (std::sscanf(pEdit + 6, "%63s %d", Name, &On) != 2) Reply = "stage <name> 0|1";
                    else if (!stage::Bypassable(Name)) Reply = std::format("the stage '{}' cannot be bypassed", Name);
                    else { std::erase(V.m_StagesOff, std::string(Name)); if (!On) V.m_StagesOff.push_back(Name); Reply = "ok"; bOk = true; }
                }
                else if (E.starts_with("param "))      // "param <name> [value]" on the view (a Level with a RenderGraph: the session edits its descriptor instead)
                {
                    char Name[64] = {}; float Value = 0;
                    auto& Ov = pSystem->m_pScene->m_View.StageParams();
                    const int n = std::sscanf(pEdit + 6, "%63s %f", Name, &Value);
                    const auto* pP = n >= 1 ? stage::Parameter(Name) : nullptr;
                    if (!pP) Reply = "param <name> [value]: ao.radius, ao.thickness, refl.ssr_steps, refl.ssr_max_dist, refl.ssr_thickness";
                    else { std::erase_if(Ov, [&](const auto& P) { return P.first == Name; }); if (n == 2) Ov.emplace_back(Name, stage::param{ Name, std::string(pP->m_Type), Value }.Word()); Reply = "ok"; bOk = true; }
                }
                else bOk = post::Edit(pSystem->m_pScene->m_View.m_PostDesc, E, Reply);
                if (pOut && Capacity > 0) { const std::size_t n = std::min<std::size_t>(Reply.size(), static_cast<std::size_t>(Capacity - 1)); std::memcpy(pOut, Reply.data(), n); pOut[n] = 0; }
                return bOk;
            }
            std::uint64_t PickPixel(const void* pWorld, int X, int Y, float* pDistance) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                if (pDistance) *pDistance = 0.0f;
                return pSystem ? pSystem->PickPixel(X, Y, pDistance) : ~0ull;
            }
            bool          UpgradeOldMaterials(const void* pWorld, std::uint64_t EntityValue) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                return pSystem && pSystem->UpgradeOldMaterialsOf(EntityValue);
            }
            int           DumpGI(const void* pWorld, const char* pPath) noexcept override
            {
                auto* pSystem = FindSystem(pWorld);
                return pSystem && pPath ? pSystem->DumpGI(pPath) : -1;
            }
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
                const std::string Text = std::format("DescribeTextDraw: ok\nLabels={}\nGlyphs={}\nDraws={}\nDropped={}\nCalls={}\nSeen={}\nWithLayout={}\nReady={}", S.m_Labels, S.m_Glyphs, S.m_Draws, S.m_Dropped, S.m_Calls, S.m_Seen, S.m_WithLayout, S.m_bReady ? 1 : 0);
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

// Before a copy of this DLL is unloaded (a Level closing - xlevel_engine_copies.h): its own scheduler's workers stop. Each worker thread
// holds a reference to the module its code is in (the C runtime's thread start), so while they run FreeLibrary never unloads the copy:
// it stayed loaded, its file locked (the close waited a second on it), 2026-11-01
extern "C" __declspec(dllexport) void XLionRender_BeforeUnload() noexcept
{
    if (auto& S = xscheduler::g_System; S.getWorkerCount()) S.Shutdown();
}
