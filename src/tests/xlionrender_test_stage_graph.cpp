// Frame stages as data (render graph R1, xlionrender_stage_graph.h). The lowering refuses what does not match (a name not registered, the
// bindings or the push not the kernel's). Then the equivalence: the light tiles hand-written (view A) against their description (view B),
// a scene with local lights, the tiles' lists read back each frame - equal bit for bit (the lit image is not: two views of the prepass
// path differ by their frame-indexed noise). First A against A (both
// hand-written): if two views differ the harness is not trusted. The Frame graph's "light tiles" scope from the stage is the same.
#include "xlionrender_test_common.h"
#include "plans/forward/xlionrender_plan_forward.h"
#include <cstring>

using namespace xlionrender::test;
namespace stage = xlionrender::stage;

static bool Read( xgpu::device& Device, const xlionrender::render_view& V, std::vector<std::byte> (&S)[2] )
{
    int W, H;
    const xgpu::texture* T[] = { &V.m_HDR, &V.m_Output };
    for( int i = 0; i < 2; ++i ) if( Device.ReadTexture(*T[i], S[i], W, H) ) return false;
    return true;
}

// the scope's line and its access lines from the graph's text
static std::string Scope( const std::string& Text, const char* pName )
{
    const auto b = Text.find(std::string("\t") + pName + "\t"); if( b == std::string::npos ) return {};
    const auto s = Text.rfind('\n', b) + 1;
    auto e = Text.find("\nS\t", b);
    return Text.substr(s, e == std::string::npos ? std::string::npos : e - s);
}

int main( int argc, char** argv )
{
    const int nFrames = argc > 1 ? std::atoi(argv[1]) : 4;

    // the lowering's refusals
    {
        stage::registry R;
        xgpu::compute Fake;
        R.m_Kernels.Add("light_tiles", { &Fake, 2, 1, 0, 80, "instance: m_LightTilesCompute" });
        for( const char* p : { "Lights: light ring", "Lights: the screen tiles' lists" } ) R.m_Buffers.Add(p);
        R.m_Textures.Add("Render target: depth");
        for( const char* p : { "view.width", "view.height", "lights.local_base", "lights.local_count" } ) { const int i = R.m_Frame.Add(p); R.SetU32(i, 1); }
        const int C2W = R.m_Frame.Add("frame.c2w"); { float M[16]{}; R.SetFloats(C2W, M, 16); }
        const auto& D = xlionrender::details::LightTilesStage();
        CHECK(stage::Lower(D, R).m_Error.empty(), "the light tiles lower: %s", stage::Lower(D, R).m_Error.c_str());
        auto Bad = D; Bad.m_Passes[0].m_Kernel = "nope";
        CHECK(stage::Lower(Bad, R).m_Error.find("no kernel 'nope'") != std::string::npos, "an unknown kernel is refused");
        Bad = D; Bad.m_Passes[0].m_Buffers[0].m_Name = "nope";
        CHECK(stage::Lower(Bad, R).m_Error.find("no buffer 'nope'") != std::string::npos, "an unknown buffer is refused");
        Bad = D; Bad.m_Passes[0].m_Textures.clear();
        CHECK(stage::Lower(Bad, R).m_Error.find("bindings do not match") != std::string::npos, "a missing binding is refused");
        Bad = D; Bad.m_Passes[0].m_Push.push_back({ "", 7u });
        CHECK(stage::Lower(Bad, R).m_Error.find("pushes 84 bytes") != std::string::npos, "a push of the wrong size is refused: %s", stage::Lower(Bad, R).m_Error.c_str());
        Bad = D; Bad.m_Passes[0].m_Push[1].m_Frame = "nope";
        CHECK(stage::Lower(Bad, R).m_Error.find("no frame value 'nope'") != std::string::npos, "an unknown frame value is refused");
        std::printf("lowering: refusals as expected\n");
    }

    // the tile pass (and its light tiles) is on with XLION_TILE (read at the first frame)
#ifdef _WIN32
    _putenv_s("XLION_TILE", "1");
#else
    setenv("XLION_TILE", "1", 1);
#endif
    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    // cluster geometry: the G-buffer prepass lays its depth before the light tiles (plain meshes are not in it)
    cluster_buffers SphereB, BoxB;
    xlionrender::mesh_id Sphere, Box;
    CHECK(MakeStaticBuffers(App.m_Device, MakeSphere(), 0.5f, SphereB) && !Renderer.CreateClusterGeom(Sphere, SphereB.Geom()), "the sphere");
    CHECK(MakeStaticBuffers(App.m_Device, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), 0.5f, BoxB) && !Renderer.CreateClusterGeom(Box, BoxB.Geom()), "the box");
    xlionrender::scene Scene;
    Scene.m_Draws = { { .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.55f, 0 }) * xmath::fmat4::fromScale({ 10.0f, 0.1f, 10.0f }) }
                    , { .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ -0.4f, 0.3f, 0 }) }
                    , { .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0.7f, -0.1f, 0.5f }) * xmath::fmat4::fromScale({ 0.4f, 0.8f, 0.4f }) } };
    Scene.m_Sun.m_Intensity = 0.5f;
    Scene.m_LocalLights.push_back({ .m_Type = xlionrender::local_light::type::POINT, .m_Position = { -0.6f, 0.5f, -0.6f }, .m_Radius = 1.2f, .m_Color = { 1, 0.6f, 0.3f }, .m_Intensity = 6.0f, .m_bCastShadows = false });
    Scene.m_LocalLights.push_back({ .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { 1.2f, 0.9f, -0.5f }, .m_Direction = { 0, -1, 0.5f }, .m_Radius = 1.6f, .m_Color = { 0.3f, 0.6f, 1 }, .m_Intensity = 8.0f, .m_bCastShadows = false });
    Scene.m_Post.m_bTAA = false; Scene.m_Post.m_Bloom = 0.0f; Scene.m_Post.m_bAutoExposure = false;

    const int W = 805, H = 377;
    // a frame, then the tiles' lists copied into a buffer the CPU reads (xGPU reads back textures only)
    xgpu::buffer Copy;
    const auto Frame = [&]( const xlionrender::scene& S, xlionrender::render_view& V, xgpu::tools::view& Cam, std::vector<std::byte> (&Img)[2], std::vector<std::uint32_t>& Tiles ) -> bool
    {
        auto& Win = App.m_Window;
        if( !App.m_Instance.ProcessInputEvents() ) return false;
        while( Win.BeginRendering() ) if( !App.m_Instance.ProcessInputEvents() ) return false;
        Renderer.Render(Win, S, V, Cam);
        const int n = V.m_LightTiles.m_Private ? int(V.m_LightTiles.getEntryCount()) : 0;
        if( n && Copy.m_Private && int(Copy.getEntryCount()) != n ) App.m_Device.Destroy(std::move(Copy));
        if( n && !Copy.m_Private && App.m_Device.Create(Copy, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = n, .m_pName = "test: light tiles read back" }) ) return false;
        if( n ) { const xgpu::buffer::copy C{ 0, 0, n }; Win.CopyBuffer(V.m_LightTiles, Copy, std::span{ &C, 1 }); }
        { auto Cmd = Win.getCmdBuffer(); }
        Win.PageFlip();
        if( !Read(App.m_Device, V, Img) ) return false;
        Tiles.assign(std::size_t(n), 0u);
        if( n ) (void)Copy.MemoryMap(0, n, [&]( void* p ) { std::memcpy(Tiles.data(), p, std::size_t(n) * 4); });
        return true;
    };
    for( int Control = 1; Control >= 0; --Control )         // 1: hand-written against itself (the harness's own check), 0: against the stage
    {
        xlionrender::render_view A, B;
        CHECK(!Renderer.CreateView(A, { .m_Width = W, .m_Height = H }) && !Renderer.CreateView(B, { .m_Width = W, .m_Height = H }), "CreateView");
        A.m_bStageLightTiles = false;
        B.m_bStageLightTiles = !Control;
        std::string Why;
        for( int f = 0; f < nFrames && Why.empty(); ++f )
        {
            xgpu::tools::view Camera;
            Camera.setFov(xmath::radian{ 1.0f });
            Camera.setViewport({ 0, 0, W, H });
            Camera.LookAt(xmath::fvec3{ 0.05f * float(f), 0.3f, -2.5f }, xmath::fvec3{ 0, 0, 0 });
            std::vector<std::byte> SA[2], SB[2];
            std::vector<std::uint32_t> LA, LB;
            CHECK(Frame(Scene, A, Camera, SA, LA), "Render A");
            CHECK(Frame(Scene, B, Camera, SB, LB), "Render B");
            const auto TA = Scope(A.m_Graph.Text(), "light tiles"), TB = Scope(B.m_Graph.Text(), "light tiles");
            CHECK(!TA.empty() && !LA.empty(), "the light tiles ran (the scene has local lights; XLION_TILE=1)");
            if( TA != TB ) Why = std::format("frame {}: the Frame graph's scope differs:\n{}\n--\n{}", f, TA, TB);
            if( Why.empty() && LA != LB ) { std::size_t k = 0; while( k < std::min(LA.size(), LB.size()) && LA[k] == LB[k] ) ++k; Why = std::format("frame {}: the tiles' lists differ ({} vs {} words), the first at word {}", f, LA.size(), LB.size(), k); }
            if( f == nFrames - 1 && Control )
            {
                int nTiles = 0, nLit = 0;
                for( std::size_t t = 16; t + 16 <= LA.size(); t += 16 ) { ++nTiles; nLit += LA[t] != 0; }
                std::printf("tiles %d, with lights %d (header %u %u)\n", nTiles, nLit, LA[0], LA[1]);
                CHECK(nLit > 0 && nLit < nTiles, "some tiles have lights, some not (a test that can tell)");
            }
        }
        CHECK(B.m_LightTilesPlan.m_Error.empty(), "the stage lowered: %s", B.m_LightTilesPlan.m_Error.c_str());
        if( !Control ) CHECK(!B.m_LightTilesPlan.m_Passes.empty(), "view B ran the stage");
        const char* pWhat = Control ? "hand-written against itself" : "hand-written against the stage";
        Renderer.DestroyView(A); Renderer.DestroyView(B);
        if( !Why.empty() ) { std::printf("DIFFERENT: %s - %s\n", pWhat, Why.c_str()); std::printf(Control ? "FAIL: the harness is not trusted\n" : "FAIL\n"); return 1; }
        std::printf("same: %s, %d frames\n", pWhat, nFrames);
    }
    if( Copy.m_Private ) App.m_Device.Destroy(std::move(Copy));
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
