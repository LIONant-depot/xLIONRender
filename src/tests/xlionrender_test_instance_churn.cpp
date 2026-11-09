// A renderer made, used and destroyed again and again on one device (the editor's Play/Stop: each Play renders with a renderer of its own)
// gives its memory back: after two warm-up rounds, the process's private memory grows less than 8 MB a round on average.
//  - each round: a renderer, a sphere and a slab, a sun with shadows, a view, 20 frames, then everything destroyed
//  - XLION_CHURN_ROUNDS (default 8) rounds; the private memory printed after each
#include "xlionrender_test_common.h"
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>
#include <memory>

using namespace xlionrender::test;
namespace xgpu { void PrintVulkanAllocStats( const char* pLabel ) noexcept; }      // (xgpu_vulkan_instance.cpp: XGPU_ALLOC_STATS)

namespace
{
    double PrivateMB() noexcept
    {
        PROCESS_MEMORY_COUNTERS_EX C{};
        K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&C), sizeof(C));
        return double(C.PrivateUsage) / (1024.0 * 1024.0);
    }

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

    constexpr std::uint32_t s_AOComp[] = {
        #include "xlionrender_ao_comp.h"
    };
    constexpr std::uint32_t s_DepthFrag[] = {
        #include "xlionrender_depth_frag.h"
    };
    constexpr std::uint32_t s_ClearVert[] = {
        #include "xlionrender_clear_depth_vert.h"
    };
    template<std::size_t N> xgpu::shader::setup::raw_data Raw(const std::uint32_t(&W)[N]) noexcept { return { { reinterpret_cast<const std::int32_t*>(W), N } }; }

    // A graphics pipeline and its instance (as the renderer's shadow clear), made and destroyed 50 times. bDestroy: through Device.Destroy
    // (the renderer's way); else the handles only dropped
    bool GraphicsChurn(app& App, bool bDestroy) noexcept
    {
        for (int k = 0; k < 50; ++k)
        {
            xgpu::vertex_descriptor VD;
            auto A2 = std::array{ xgpu::vertex_descriptor::attribute{ .m_Offset = 0, .m_Format = xgpu::vertex_descriptor::format::FLOAT_2D } };
            if (App.m_Device.Create(VD, { .m_VertexSize = 8, .m_Attributes = A2 })) return false;
            xgpu::shader Frag, Vert;
            if (App.m_Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = Raw(s_DepthFrag) })) return false;
            if (App.m_Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX, .m_Sharer = Raw(s_ClearVert) })) return false;
            auto Shaders = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            xgpu::pipeline::setup Setup{ .m_VertexDescriptor = VD, .m_Shaders = Shaders };
            Setup.m_Primitive.m_Cull = xgpu::pipeline::primitive::cull::NONE;
            Setup.m_DepthStencil.m_DepthCompare = xgpu::pipeline::depth_stencil::depth_compare::ALWAYS;
            xgpu::pipeline Pipe; xgpu::pipeline_instance Inst;
            if (App.m_Device.Create(Pipe, Setup)) return false;
            if (App.m_Device.Create(Inst, { .m_PipeLine = Pipe })) return false;
            if (bDestroy) { App.m_Device.Destroy(std::move(Inst)); App.m_Device.Destroy(std::move(Pipe)); }
        }
        return true;
    }

    // One Play: a renderer of its own on the app's device, a few frames, all destroyed
    bool Round(app& App) noexcept
    {
        static const int s_Stage = [] { const char* p = xlionrender::Env("XLION_CHURN_STAGE"); return p ? std::atoi(p) : 9; }();      // (bisection: -1 one compute pipeline 50 times, 0 Init only, 1 + a view, 2 + meshes, 9 all)
        if (s_Stage == -2 || s_Stage == -3) return GraphicsChurn(App, s_Stage == -2);      // (-2: Device.Destroy, -3: the handles dropped)
        if (s_Stage == -4 || s_Stage == -5 || s_Stage == -6)      // (-4: a render target, -5: a texture with data, -6: a render target and its pass)
        {
            for (int k = 0; k < 50; ++k)
            {
                xgpu::texture Tex;
                const std::uint32_t White = 0xffffffffu;
                const auto Mips = std::array{ xgpu::texture::setup::mip_size{ sizeof(White) } };
                if (s_Stage == -5) { if (App.m_Device.Create(Tex, { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips, .m_AllFacesData = { reinterpret_cast<const std::byte*>(&White), sizeof(White) }, .m_isGamma = false })) return false; }
                else if (App.m_Device.Create(Tex, { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = 256, .m_Height = 256, .m_isGamma = false })) return false;
                if (s_Stage == -6)
                {
                    xgpu::renderpass Pass;
                    auto A = std::array{ xgpu::renderpass::attachment{ Tex } };
                    if (App.m_Device.Create(Pass, { .m_Attachments = A })) return false;
                }
            }
            return true;
        }
        if (s_Stage < 0)
        {
            for (int k = 0; k < 50; ++k)
            {
                xgpu::shader Shader; xgpu::compute Compute;
                if (App.m_Device.Create(Shader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_AOComp), std::size(s_AOComp) } } })) return false;
                if (App.m_Device.Create(Compute, { .m_pShader = &Shader, .m_nBuffers = 5, .m_nTextures = 4, .m_PushConstantsSize = 64, .m_pName = "churn" })) return false;
            }
            return true;
        }
        auto pRenderer = std::make_unique<xlionrender::instance>();
        if (pRenderer->Init(App.m_Device, xlionrender::Env("XLION_CHURN_PREVIEW") ? xlionrender::instance::setup::Preview() : xlionrender::instance::setup{})) return false;
        if (const char* p = xlionrender::Env("XLION_CHURN_SLEEP_MS")) Sleep(DWORD(std::atoi(p)));     // (diagnosis: the driver's background work done before the destroy)
        if (s_Stage == 0) { pRenderer.reset(); return true; }
        xlionrender::mesh_id Sphere, Box;
        if (!MakeMesh(*pRenderer, MakeSphere(), Sphere) || !MakeMesh(*pRenderer, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), Box)) return false;
        xlionrender::render_view View;
        if (pRenderer->CreateView(View, { .m_Width = 256, .m_Height = 256 })) return false;
        xgpu::tools::view Camera;
        if (s_Stage <= 2) { pRenderer->DestroyView(View); pRenderer->DestroyMesh(Sphere); pRenderer->DestroyMesh(Box); pRenderer.reset(); return true; }
        Camera.setFov(xmath::radian{ 1.0f });
        Camera.LookAt(xmath::fvec3{ 0, 2.5f, -3.0f }, xmath::fvec3{ 0, 0, 0 });
        xlionrender::scene Scene;
        Scene.m_Draws   = { { .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.55f, 0 }) * xmath::fmat4::fromScale({ 10.0f, 0.1f, 10.0f }) }
                          , { .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ 0, 0.3f, 0 }) } };
        Scene.m_Sun     = { .m_Direction = { 0.3f, -1.0f, 0.2f }, .m_Intensity = 3, .m_ShadowTileSize = 0.5f };
        Scene.m_Ambient = { 0.1f, 0.1f, 0.1f };
        for (int f = 0; f < 20; ++f)
        {
            if (!App.m_Instance.ProcessInputEvents()) return false;
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) return false;
            pRenderer->Render(App.m_Window, Scene, View, Camera, {});
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
        }
        pRenderer->DestroyView(View);
        pRenderer->DestroyMesh(Sphere);
        pRenderer->DestroyMesh(Box);
        pRenderer.reset();
        return true;
    }
}

int main()
{
    app App;
    App.m_bQuietTimes = true;
    CHECK(App.Init(), "The app (its own renderer stays idle)");
    const char* pN = xlionrender::Env("XLION_CHURN_ROUNDS");
    const int nRounds = pN ? std::max(4, std::atoi(pN)) : 8;
    double After2 = 0, Last = 0;
    for (int r = 0; r < nRounds; ++r)
    {
        CHECK(Round(App), "Round %d", r);
        for (int f = 0; f < 6; ++f)      // (frames with no renderer: the device's deferred destroys run - the editor keeps drawing after Stop)
        {
            if (!App.m_Instance.ProcessInputEvents()) break;
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
        }
        Last = PrivateMB();
        if (r == 1) After2 = Last;
        std::printf("round %d: private memory %.1f MB\n", r, Last);
        if (std::getenv("XGPU_ALLOC_STATS")) { char L[32]; std::snprintf(L, sizeof(L), "round %d", r); xgpu::PrintVulkanAllocStats(L); }
    }
    const double PerRound = (Last - After2) / double(nRounds - 2);
    std::printf("after the warm-up: %+.2f MB a round\n", PerRound);
    const char* pMax = xlionrender::Env("XLION_CHURN_MAX_MB");     // (diagnosis: a high one lets the run reach the device's destruction - the validation layer's leak list)
    CHECK(PerRound < (pMax ? std::atof(pMax) : 8.0), "A renderer made and destroyed keeps %.1f MB a round", PerRound);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
