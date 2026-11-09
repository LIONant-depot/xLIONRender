// Custom post nodes (render graph stage I1): a node with its own compute shader (xlionrender_post_node.glsl's frame + one function),
// compiled here with glslc the way the RenderGraph compiler will, given to the view as SPIR-V.
//  - an identity node after TAA: the output equals the chain without it, bit for bit (two views, the harness's way: the same lit image)
//  - a node that changes the image (it halves it): the output differs - the node really runs
//  - a shader edit (identity -> half -> identity) replaces the pipeline; the output follows
//  - a custom node with no SPIR-V is skipped: the chain without it
//  - a description with a custom node in a bad place, or a bad shader index, is refused by name (CPU)
#include "xlionrender_test_common.h"
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace xlionrender::test;
namespace post = xlionrender::post;

#define MUST(COND, ...) do { if (!(COND)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); std::exit(1); } } while(0)

static std::vector<std::int32_t> Compile( const char* pName, const char* pBody )
{
    const auto Dir = std::filesystem::temp_directory_path() / "xlionrender_test_post_custom";
    std::filesystem::create_directories(Dir);
    const auto Src = Dir / (std::string(pName) + ".comp"), Spv = Dir / (std::string(pName) + ".spv");
    { std::ofstream F(Src); F << "#version 450\n#extension GL_GOOGLE_include_directive : require\n#include \"xlionrender_post_node.glsl\"\n" << pBody << "\n"; }
    const char* pSdk = std::getenv("VULKAN_SDK");
    MUST(pSdk, "VULKAN_SDK is not set (glslc)");
    const std::string Inc = (std::filesystem::path(__FILE__).parent_path().parent_path() / "shaders").string();
    const std::string Cmd = std::format("\"\"{}\\Bin\\glslc.exe\" -fshader-stage=compute --target-env=vulkan1.2 -I \"{}\" \"{}\" -o \"{}\"\"", pSdk, Inc, Src.string(), Spv.string());
    MUST(std::system(Cmd.c_str()) == 0, "glslc failed on %s", pName);
    std::ifstream F(Spv, std::ios::binary);
    std::vector<char> B((std::istreambuf_iterator<char>(F)), {});
    std::vector<std::int32_t> W(B.size() / 4);
    std::memcpy(W.data(), B.data(), W.size() * 4);
    return W;
}

static std::vector<std::byte> Output( xgpu::device& Device, const xlionrender::render_view& V )
{
    std::vector<std::byte> R; int W, H;
    MUST(!Device.ReadTexture(V.m_Output, R, W, H), "read the output");
    return R;
}

int main()
{
    // CPU: where a custom node may go
    {
        auto D = post::desc::Default(); D.m_Nodes.insert(D.m_Nodes.begin() + 1, { post::node_type::CUSTOM, true, 0 });
        const auto P = post::Lower(D, 640, 360);
        MUST(P.m_Error.empty() && P.m_Passes[1].m_Kernel == post::kernel::CUSTOM && P.m_Passes[2].m_Bind[0].m_Image == post::image::CHAIN, "a custom node after TAA lowers (the bloom then reads the chain - its output): %s", P.m_Error.c_str());
        auto Bad = post::desc::Default(); Bad.m_Nodes.insert(Bad.m_Nodes.begin(), { post::node_type::CUSTOM, true, 0 });
        MUST(post::Lower(Bad, 64, 64).m_Error.find("TAA must come first") != std::string::npos, "a custom node before TAA is refused");
        auto Idx = post::desc::Default(); Idx.m_Nodes.insert(Idx.m_Nodes.begin() + 1, { post::node_type::CUSTOM, true, 9 });
        MUST(post::Lower(Idx, 64, 64).m_Error.find("shader index 9") != std::string::npos, "a bad shader index is refused");
        auto Two = D; Two.m_Nodes.insert(Two.m_Nodes.begin() + 2, { post::node_type::CUSTOM, true, 0 });
        MUST(post::Lower(Two, 64, 64).m_Error.find("shader 0 is used by node 1") != std::string::npos, "two nodes one image refused");
    }

    const auto Identity = Compile("identity", "vec4 Node(ivec2 p, vec2 uv) { return texelFetch(Src, p, 0); }");
    const auto Half     = Compile("half",     "vec4 Node(ivec2 p, vec2 uv) { vec4 c = texelFetch(Src, p, 0); return vec4(c.rgb * 0.5, c.a); }");

    app App;
    MUST(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    xlionrender::mesh_id Sphere;
    {
        const auto Prim = MakeSphere();
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        MUST(!Renderer.CreateMesh(Sphere, Verts, Prim.m_Indices), "CreateMesh");
    }
    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_Material = { .m_BaseColor = { 0.8f, 0.8f, 0.8f }, .m_Roughness = 0.3f } });
    Scene.m_Sun.m_Direction = { 0.3f, -1, 0.5f }; Scene.m_Sun.m_Intensity = 30; Scene.m_Ambient = { 0.05f, 0.05f, 0.08f };
    Scene.m_Post.m_bTAA = true; Scene.m_Post.m_Bloom = 0.4f; Scene.m_Post.m_bAutoExposure = true; Scene.m_Post.m_BloomThreshold = 0.6f;

    xlionrender::render_view A, B;
    MUST(!Renderer.CreateView(A, { .m_Width = 400, .m_Height = 240 }) && !Renderer.CreateView(B, { .m_Width = 400, .m_Height = 240 }), "CreateView");
    A.m_PostFixedDt = B.m_PostFixedDt = 1.0f / 30.0f;
    B.m_pPostInputs = &A;
    B.m_PostDesc.m_Nodes.insert(B.m_PostDesc.m_Nodes.begin() + 1, { post::node_type::CUSTOM, true, 0 });      // TAA -> custom -> bloom ...
    const auto Frames = [&](int n)
    {
        for (int f = 0; f < n; ++f)
        {
            xgpu::tools::view C; C.setFov(xmath::radian{ 1.0f }); C.setViewport({ 0, 0, 400, 240 });
            C.LookAt(xmath::fvec3{ 0.05f * float(f), 0.1f, -2.0f }, xmath::fvec3{ 0, 0, 0 });
            MUST(App.RenderFrame(Scene, A, C) && App.RenderFrame(Scene, B, C), "Render");
        }
        return std::pair{ Output(App.m_Device, A), Output(App.m_Device, B) };
    };
    // no SPIR-V: the node is skipped, the chain as without it
    { const auto [a, b] = Frames(4); MUST(a == b, "a custom node with no shader is skipped (the output as without it)"); std::printf("ok: no shader - skipped\n"); }
    // identity: the same output
    B.m_PostCustomSpirv[0] = Identity;
    { const auto [a, b] = Frames(4); MUST(a == b, "an identity node: the same output"); MUST(B.m_PostCustomMade[0] != 0, "its pipeline made"); std::printf("ok: identity - identical\n"); }
    // half: different (it runs), then back to identity (the pipeline replaced twice)
    B.m_PostCustomSpirv[0] = Half;
    { const auto [a, b] = Frames(4); MUST(a != b, "a node that halves the image changes the output"); std::printf("ok: half - differs\n"); }
    // a bad shader (not SPIR-V): the last good (half) stays - the output still differs, the pipeline is the half's
    {
        const auto HalfMade = B.m_PostCustomMade[0];
        B.m_PostCustomSpirv[0] = std::vector<std::int32_t>(64, 0x12345678);
        const auto [a, b] = Frames(3);
        MUST(a != b && B.m_PostCustomMade[0] == HalfMade, "a bad shader keeps the last good one (the half)");
        std::printf("ok: a bad shader - the last good kept\n");
    }
    B.m_PostCustomSpirv[0] = Identity;
    { const auto [a, b] = Frames(4); MUST(a == b, "the shader back to identity: the same output again"); std::printf("ok: edit back - identical\n"); }
    // two views with different shaders in one frame: a third view (the same input) with the half, B with the identity
    {
        xlionrender::render_view C3;
        MUST(!Renderer.CreateView(C3, { .m_Width = 400, .m_Height = 240 }), "CreateView third");
        C3.m_PostFixedDt = A.m_PostFixedDt; C3.m_pPostInputs = &A; C3.m_PostDesc = B.m_PostDesc; C3.m_PostCustomSpirv[0] = Half;
        for (int f = 0; f < 3; ++f)
        {
            xgpu::tools::view C; C.setFov(xmath::radian{ 1.0f }); C.setViewport({ 0, 0, 400, 240 }); C.LookAt(xmath::fvec3{ 0, 0.1f, -2.0f }, xmath::fvec3{ 0, 0, 0 });
            MUST(App.RenderFrame(Scene, A, C) && App.RenderFrame(Scene, B, C) && App.RenderFrame(Scene, C3, C), "Render three views");
        }
        const auto a = Output(App.m_Device, A), b = Output(App.m_Device, B), c = Output(App.m_Device, C3);
        MUST(a == b && a != c, "two views, two shaders: the identity view equals the plain one, the half view differs");
        // closing a view with its custom pipeline right after it rendered (frames in flight)
        xgpu::tools::view C; C.setFov(xmath::radian{ 1.0f }); C.setViewport({ 0, 0, 400, 240 }); C.LookAt(xmath::fvec3{ 0, 0.1f, -2.0f }, xmath::fvec3{ 0, 0, 0 });
        MUST(App.RenderFrame(Scene, C3, C), "Render before closing");
        Renderer.DestroyView(C3);
        std::printf("ok: two views, two shaders; closed in flight\n");
    }
    // resize: the node's image made again at the new size
    MUST(!Renderer.CreateView(A, { .m_Width = 320, .m_Height = 200 }) && !Renderer.CreateView(B, { .m_Width = 320, .m_Height = 200 }), "resize");
    {
        for (int f = 0; f < 3; ++f)
        {
            xgpu::tools::view C; C.setFov(xmath::radian{ 1.0f }); C.setViewport({ 0, 0, 320, 200 }); C.LookAt(xmath::fvec3{ 0, 0.1f, -2.0f }, xmath::fvec3{ 0, 0, 0 });
            MUST(App.RenderFrame(Scene, A, C) && App.RenderFrame(Scene, B, C), "Render after resize");
        }
        MUST(Output(App.m_Device, A) == Output(App.m_Device, B), "after a resize: still identical");
        std::printf("ok: resize\n");
    }
    Renderer.DestroyView(A); Renderer.DestroyView(B);
    MUST(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
