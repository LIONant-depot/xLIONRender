#include "xlionrender_instance.h"
#include "plans/forward/xlionrender_plan_forward.h"
#include <cstring>

namespace xlionrender
{
    namespace
    {
        constexpr std::uint32_t s_ForwardVert[] = {
            #include "xlionrender_forward_vert.h"
        };
        constexpr std::uint32_t s_ForwardStandardFrag[] = {
            #include "xlionrender_forward_standard_frag.h"
        };
        constexpr std::uint32_t s_StaticGeomVert[] = {
            #include "xlionrender_static_geom_vert.h"
        };
        constexpr std::uint32_t s_GICaptureVert[] = {
            #include "xlionrender_gi_capture_vert.h"
        };
        constexpr std::uint32_t s_StaticGeomGIVert[] = {
            #include "xlionrender_static_geom_gi_vert.h"
        };
        constexpr std::uint32_t s_SkinGeomVert[] = {
            #include "xlionrender_skin_geom_vert.h"
        };
        constexpr std::uint32_t s_DefaultMaterialFrag[] = {
            #include "xlionrender_default_material_frag.h"
        };
        constexpr std::uint32_t s_DepthFrag[] = {
            #include "xlionrender_depth_frag.h"
        };
        constexpr std::uint32_t s_EnvSkyFrag[] = {
            #include "xlionrender_env_sky_frag.h"
        };
        constexpr std::uint32_t s_EnvPrefilterFrag[] = {
            #include "xlionrender_env_prefilter_frag.h"
        };
        constexpr std::uint32_t s_EnvDownsampleFrag[] = {
            #include "xlionrender_env_downsample_frag.h"
        };
        constexpr std::uint32_t s_OutlineMaskFrag[] = {
            #include "xlionrender_outline_mask_frag.h"
        };
        constexpr std::uint32_t s_OutlineCompositeFrag[] = {
            #include "xlionrender_outline_composite_frag.h"
        };
#if 0       // The first attempt (the primitives' OBB outline shader over the scene): not used
        constexpr std::uint32_t s_OutlineObbVert[] = {
            #include "xlionrender_outline_obb_vert.h"
        };
        constexpr std::uint32_t s_OutlineObbFrag[] = {
            #include "xlionrender_outline_obb_frag.h"
        };
#endif
        constexpr std::uint32_t s_SkyBackgroundFrag[] = {
            #include "xlionrender_sky_background_frag.h"
        };
        constexpr std::uint32_t s_EnvCubemapFrag[] = {
            #include "xlionrender_env_cubemap_frag.h"
        };
        constexpr std::uint32_t s_EnvCubeToOctFrag[] = {
            #include "xlionrender_env_cube_to_oct_frag.h"
        };
        constexpr std::uint32_t s_EnvDFGFrag[] = {
            #include "xlionrender_env_dfg_frag.h"
        };
        constexpr std::uint32_t s_GeomStandardFrag[] = {
            #include "xlionrender_geom_standard_frag.h"
        };
        constexpr std::uint32_t s_ShadowRemapFrag[] = {
            #include "xlionrender_shadow_remap_frag.h"
        };
        constexpr std::uint32_t s_ClearDepthVert[] = {
            #include "xlionrender_clear_depth_vert.h"
        };
        constexpr std::uint32_t s_FullscreenVert[] = {
            #include "xlionrender_fullscreen_vert.h"
        };
        constexpr std::uint32_t s_ToneMapFrag[] = {
            #include "xlionrender_tonemap_frag.h"
        };
        constexpr std::uint32_t s_BlitFrag[] = {
            #include "xlionrender_blit_frag.h"
        };

        template<std::size_t N>
        xgpu::shader::setup::raw_data SPIRV(const std::uint32_t(&Words)[N]) noexcept
        {
            return { { reinterpret_cast<const std::int32_t*>(Words), N } };
        }

        constexpr int max_draws_per_frame_v         = 16384;    // ponytail: one material UBO entry per draw, ring of this size (the stress city needs ~7k); the constant material belongs in the persistent instance record
        constexpr int max_shadow_draws_per_frame_v  = 8192;     // ponytail: one UBO entry per (tile, caster) the shadow pass draws, ring of this size

        struct fullscreen_vert { float m_X, m_Y; };

        // xgpu::buffer has no move, so Device.Destroy(std::move(B)) queues a copy and B keeps a reference: take the handle out
        // instead, its destructor queues it in the device's deferred destroy
        void ReleaseBuffer( xgpu::buffer& Buffer ) noexcept
        {
            xgpu::buffer Released;
            Released.m_Private.swap(Buffer.m_Private);
        }

        // Puts Slot in the first dead entry (its generation moves on, so old ids stay invalid) or at the end
        template<typename T_SLOT>
        std::pair<std::uint32_t, std::uint32_t> AddSlot( std::vector<T_SLOT>& Slots, T_SLOT&& Slot ) noexcept
        {
            Slot.m_bAlive = true;
            for( std::uint32_t i = 0; i < Slots.size(); ++i )
            {
                if( Slots[i].m_bAlive ) continue;
                Slot.m_Generation = Slots[i].m_Generation + 1;
                Slots[i]          = std::move(Slot);
                return { i, Slots[i].m_Generation };
            }
            Slots.push_back(std::move(Slot));
            return { static_cast<std::uint32_t>(Slots.size() - 1), 0 };
        }

        template<typename T_SLOT, typename T_ID>
        T_SLOT* FindSlot( std::vector<T_SLOT>& Slots, T_ID Id ) noexcept
        {
            if( Id.m_Index >= Slots.size() ) return nullptr;
            auto& S = Slots[Id.m_Index];
            return (S.m_bAlive && S.m_Generation == Id.m_Generation) ? &S : nullptr;
        }
    }

    //------------------------------------------------------------------------------------------------

    instance::~instance( void ) noexcept
    {
        if( m_bReady == false ) return;
        for( auto& V : m_CaptureViews ) DestroyView(V);
        m_Device.Destroy(std::move(m_CubeToOctInstance));
        m_Device.Destroy(std::move(m_CubeToOctPipeline));
        m_Device.Destroy(std::move(m_ProbeAtlasClearPass));
        m_Device.Destroy(std::move(m_ProbeAtlasPass));
        m_Device.Destroy(std::move(m_ProbeAtlas));
        for( auto& M : m_Meshes ) if( M.m_bAlive && !M.m_bCluster ) { ReleaseBuffer(M.m_VertexBuffer); ReleaseBuffer(M.m_IndexBuffer); }
        for( auto& M : m_Meshes ) if( M.m_bAlive && M.m_pGI ) ReleaseBuffer(M.m_GIBuffer);
        ReleaseBuffer(m_GILightRing);
        ReleaseBuffer(m_GIMomentRing);
        ReleaseBuffer(m_GIDummy);
        for( auto& M : m_Materials ) if( M.m_bAlive ) for( std::size_t k = 0; k < M.m_Pipeline.size(); ++k )
        {
            if( M.m_Instance[k].m_Private ) m_Device.Destroy(std::move(M.m_Instance[k]));
            if( M.m_Pipeline[k].m_Private ) m_Device.Destroy(std::move(M.m_Pipeline[k]));
        }
        m_Device.Destroy(std::move(m_ForwardInstance));
        m_Device.Destroy(std::move(m_StaticGeomInstance));
        m_Device.Destroy(std::move(m_StaticGeomPipeline));
        ReleaseBuffer(m_StaticGeomUBO);
        ReleaseBuffer(m_InstanceBuffer);
        ReleaseBuffer(m_InstanceRing);
        ReleaseBuffer(m_ShadowUBO);
        for( std::size_t k = 0; k < m_ShadowPipeline.size(); ++k ) { m_Device.Destroy(std::move(m_ShadowInstance[k])); m_Device.Destroy(std::move(m_ShadowPipeline[k])); }
        m_Device.Destroy(std::move(m_ShadowClearInstance));
        m_Device.Destroy(std::move(m_ShadowClearPipeline));
        m_Device.Destroy(std::move(m_ShadowClearPass));
        m_Device.Destroy(std::move(m_ShadowPass));
        m_Device.Destroy(std::move(m_RemapInstance));
        m_Device.Destroy(std::move(m_RemapPipeline));
        m_Device.Destroy(std::move(m_ScratchPass));
        m_Device.Destroy(std::move(m_ShadowScratch));
        m_Device.Destroy(std::move(m_ShadowAtlas));
        m_Device.Destroy(std::move(m_ForwardPipeline));
        m_Device.Destroy(std::move(m_ToneMapPipeline));
        m_Device.Destroy(std::move(m_BlitPipeline));
        m_Device.Destroy(std::move(m_WhiteTexture));
        ReleaseBuffer(m_LightUBO);
        ReleaseBuffer(m_LightRing);
        for( auto* p : { &m_EnvSkyInstance, &m_EnvPrefilterInstance, &m_EnvDFGInstance, &m_EnvCubemapInstance, &m_SkyBackgroundInstance } ) m_Device.Destroy(std::move(*p));
        m_Device.Destroy(std::move(m_SkyBackgroundPipeline));
        m_Device.Destroy(std::move(m_OutlineInstance));
        m_Device.Destroy(std::move(m_OutlinePipeline));
        for( auto& P : m_MaskPipeline ) m_Device.Destroy(std::move(P));
        m_Device.Destroy(std::move(m_OutlineCompositePipeline));
        for( auto& I : m_EnvDownsampleInstance ) m_Device.Destroy(std::move(I));
        for( auto* p : { &m_EnvSkyPipeline, &m_EnvPrefilterPipeline, &m_EnvDFGPipeline, &m_EnvDownsamplePipeline, &m_EnvCubemapPipeline } ) m_Device.Destroy(std::move(*p));
        for( auto* p : { &m_EnvAtlasPass, &m_EnvDFGPass } )                                 m_Device.Destroy(std::move(*p));
        for( auto& P : m_EnvSourcePass ) m_Device.Destroy(std::move(P));
        for( auto* p : { &m_EnvAtlas, &m_EnvDFG } )                                         m_Device.Destroy(std::move(*p));
        m_GICaptures.clear();
        m_Device.Destroy(std::move(m_GICapturePass));
        m_Device.Destroy(std::move(m_GICaptureTex));
        ReleaseBuffer(m_GICaptureVB);
        ReleaseBuffer(m_GICaptureIB);
        m_Device.Destroy(std::move(m_GIConstCaptureInstance));
        m_Device.Destroy(std::move(m_GIConstCapturePipeline));
        for( auto& T : m_EnvSource ) m_Device.Destroy(std::move(T));
        ReleaseBuffer(m_CellRing);
        ReleaseBuffer(m_MaterialUBO);
        ReleaseBuffer(m_FullscreenVB);
        ReleaseBuffer(m_FullscreenIB);
    }

    //------------------------------------------------------------------------------------------------

    // Every lit pipeline's samplers: 0 the shadow atlas (clamp), 1-7 the material's textures (white where it has none), 8 the
    // environment atlas, 9 the DFG table (both clamp, linear)
    std::vector<xgpu::pipeline::sampler> instance::LitSamplers( void ) const noexcept
    {
        constexpr auto Clamp = std::array{ xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP };
        std::vector<xgpu::pipeline::sampler> Samplers(lit_samplers_v);
        Samplers[0] = { .m_AddressMode = Clamp };
        Samplers[8] = { .m_bDisableAnisotropy = true, .m_AddressMode = Clamp };
        Samplers[9]  = { .m_bDisableAnisotropy = true, .m_AddressMode = Clamp };
        Samplers[10] = { .m_bDisableAnisotropy = true, .m_AddressMode = Clamp };
        return Samplers;
    }

    std::vector<xgpu::pipeline_instance::sampler_binding> instance::LitBindings( std::span<xgpu::texture* const> Textures ) noexcept
    {
        std::vector<xgpu::pipeline_instance::sampler_binding> Bindings;
        Bindings.reserve(lit_samplers_v);
        Bindings.emplace_back(m_ShadowAtlas);
        for( std::size_t i = 1; i < 8; ++i ) Bindings.emplace_back( i < Textures.size() && Textures[i] ? *Textures[i] : m_WhiteTexture );
        Bindings.emplace_back(m_EnvAtlas);
        Bindings.emplace_back(m_EnvDFG);
        Bindings.emplace_back(m_ProbeAtlas);
        return Bindings;
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::Init( xgpu::device& Device, const shadows::setup& ShadowSetup ) noexcept
    {
        m_Device      = Device;
        m_ShadowSetup = ShadowSetup;
        m_LocalShadows.Init(ShadowSetup);

        // Vertex layouts
        {
            auto Attributes = std::array
            { xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(vertex, m_Pos),     .m_Format = xgpu::vertex_descriptor::format::FLOAT_3D }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(vertex, m_UV),      .m_Format = xgpu::vertex_descriptor::format::FLOAT_2D }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(vertex, m_Color),   .m_Format = xgpu::vertex_descriptor::format::UINT8_4D_NORMALIZED }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(vertex, m_Normal),  .m_Format = xgpu::vertex_descriptor::format::FLOAT_3D }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(vertex, m_Tangent), .m_Format = xgpu::vertex_descriptor::format::FLOAT_4D }
            };
            if( auto Err = Device.Create(m_MeshVD, { .m_VertexSize = sizeof(vertex), .m_Attributes = Attributes }); Err ) return Err;

            auto Attributes2D = std::array{ xgpu::vertex_descriptor::attribute{ .m_Offset = 0, .m_Format = xgpu::vertex_descriptor::format::FLOAT_2D } };
            if( auto Err = Device.Create(m_FullscreenVD, { .m_VertexSize = sizeof(fullscreen_vert), .m_Attributes = Attributes2D }); Err ) return Err;
        }

        // Uniforms (dynamic UBO rings) - layouts in shaders/xlionrender_forward_standard_frag.glsl
        if( auto Err = Device.Create(m_LightUBO,    { .m_Type = xgpu::buffer::type::UNIFORM, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::ubo_lighting), .m_EntryCount = 16 }); Err ) return Err;
        if( auto Err = Device.Create(m_LightRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::local_light_data), .m_EntryCount = ring_frames_v * light_region_v }); Err ) return Err;
        if( auto Err = Device.Create(m_GILightRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = ring_frames_v * gi_light_region_v * 3 }); Err ) return Err;     // 3 vec4 a surfel: the HL2 basis
        if( auto Err = Device.Create(m_GIMomentRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = ring_frames_v * gi_light_region_v * gi::moment_texels_v }); Err ) return Err;
        {
            const std::uint32_t Zeros[8] = {};     // a GIMesh header: no surfels, no read grid
            if( auto Err = Device.Create(m_GIDummy, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(Zeros), .m_EntryCount = 1, .m_pData = const_cast<std::uint32_t*>(Zeros) }); Err ) return Err;
        }
        if( auto Err = Device.Create(m_CellRing,  { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = ring_frames_v * max_cell_uints_v }); Err ) return Err;
        if( auto Err = Device.Create(m_MaterialUBO, { .m_Type = xgpu::buffer::type::UNIFORM, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::ubo_material), .m_EntryCount = max_draws_per_frame_v }); Err ) return Err;

        // White 1x1 texture bound where the shadow map will go
        {
            const std::uint32_t White = 0xffffffffu;
            const auto          Mips  = std::array{ xgpu::texture::setup::mip_size{ sizeof(White) } };
            xgpu::texture::setup Setup{ .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                      , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&White), sizeof(White) }, .m_isGamma = false };
            if( auto Err = Device.Create(m_WhiteTexture, Setup); Err ) return Err;
        }

        // The shadow atlas: every lit pipeline samples it in slot 0
        {
            const int Clipmap = shadows::atlas_tiles_v * ShadowSetup.m_ClipmapTileSize, Local = ShadowSetup.m_LocalAtlasSize;
            if( auto Err = Device.Create(m_ShadowAtlas,   { .m_Format = xgpu::texture::format::DEPTH_U16, .m_Width = Clipmap + Local, .m_Height = std::max(Clipmap, Local), .m_isGamma = false }); Err ) return Err;
            if( auto Err = Device.Create(m_ShadowScratch, { .m_Format = xgpu::texture::format::DEPTH_F32, .m_Width = ShadowSetup.m_ScratchSize, .m_Height = ShadowSetup.m_ScratchSize, .m_isGamma = false }); Err ) return Err;
        }

        // The environment probe's textures (filled by updateEnvironment)
        for( int k = 0; k < env_source_levels_v; ++k )
            if( auto Err = Device.Create(m_EnvSource[k], { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = env_source_size_v >> k, .m_Height = env_source_size_v >> k, .m_isGamma = false }); Err ) return Err;
        if( auto Err = Device.Create(m_EnvAtlas,  { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = 384, .m_Height = 256, .m_isGamma = false }); Err ) return Err;
        if( auto Err = Device.Create(m_EnvDFG,    { .m_Format = xgpu::texture::format::R16G16_SFLOAT, .m_Width = env_dfg_size_v, .m_Height = env_dfg_size_v, .m_isGamma = false }); Err ) return Err;

        // The GI's material capture: its target, pass, the quads (4 corners a surfel, twice: albedo and emission) and their indices
        if( auto Err = Device.Create(m_GICaptureTex, { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = gi_capture_w_v, .m_Height = gi_capture_h_v, .m_isGamma = false }); Err ) return Err;
        {
            // Cleared to 0: a texel a cut out (alpha tested) material discards reads back as no albedo and no emission - the pass's default
            // clear (0.45 gray) made every cut out surfel emit 0.45, a light at EmissiveScale
            auto A = std::array{ xgpu::renderpass::attachment{ m_GICaptureTex } };
            if( auto Err = Device.Create(m_GICapturePass, { .m_Attachments = A, .m_ClearColorR = 0, .m_ClearColorG = 0, .m_ClearColorB = 0, .m_ClearColorA = 0 }); Err ) return Err;
        }
        if( auto Err = Device.Create(m_GICaptureVB, { .m_Type = xgpu::buffer::type::VERTEX, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(float) * 2, .m_EntryCount = 8 * gi_capture_max_v }); Err ) return Err;
        if( auto Err = Device.Create(m_GICaptureIB, { .m_Type = xgpu::buffer::type::INDEX, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = 6 * gi_capture_max_v }); Err ) return Err;
        (void)m_GICaptureIB.MemoryMap(0, 6 * gi_capture_max_v, [](void* p)
        {
            auto* I = static_cast<std::uint32_t*>(p);
            for( std::uint32_t q = 0; q < std::uint32_t(gi_capture_max_v); ++q ) { const std::uint32_t v = 4 * q; const std::uint32_t Q[6] = { v, v + 1, v + 2, v, v + 2, v + 3 }; std::memcpy(I + 6 * q, Q, sizeof(Q)); }
        });
        if( auto Err = Device.Create(m_ProbeAtlas,{ .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = probe_slots_x_v * 384, .m_Height = probe_slots_v / probe_slots_x_v * 256, .m_isGamma = false }); Err ) return Err;

        // Forward standard-material pipeline
        {
            xgpu::shader Vert, Frag;
            if( auto Err = Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_ForwardVert) }); Err ) return Err;
            if( auto Err = Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_ForwardStandardFrag) }); Err ) return Err;

            auto Shaders      = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            const auto Samplers = LitSamplers();
            auto UniformBinds = std::array
            { xgpu::pipeline::uniform_binds{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 2, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 3, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // lights
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 4, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // light cells
            };
            if( auto Err = Device.Create(m_ForwardPipeline, { .m_VertexDescriptor = m_MeshVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_forward)
                                                            , .m_UniformBinds = UniformBinds, .m_Samplers = Samplers }); Err ) return Err;

            auto Bindings = LitBindings({});
            if( auto Err = Device.Create(m_ForwardInstance, { .m_PipeLine = m_ForwardPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;
        }

        // xgeom_static pipeline: the static geom editor's vertex decode + the same standard material fragment
        {
            auto Attributes = std::array
            { xgpu::vertex_descriptor::attribute{ .m_Offset = 0, .m_Format = xgpu::vertex_descriptor::format::SINT16_4D, .m_iStream = 0 }   // xgeom_static::geom::vertex
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 0, .m_Format = xgpu::vertex_descriptor::format::UINT16_2D, .m_iStream = 1 }   // vertex_extras::m_UV
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 4, .m_Format = xgpu::vertex_descriptor::format::UINT8_4D,  .m_iStream = 1 }   // vertex_extras::m_OctNormal + m_OctTangent
            };
            if( auto Err = Device.Create(m_StaticGeomVD, { .m_bUseStreaming = true, .m_Topology = xgpu::vertex_descriptor::topology::TRIANGLE_LIST, .m_VertexSize = 0, .m_Attributes = Attributes }); Err ) return Err;

            // xgeom_skin::geom::vertex (int16 x3 + 6 packed bytes) and vertex_extras (UV, oct normal, oct tangent X, tangent Y + sign); the same
            // formats the skin geom editor uses (R16G16B16A16_SINT / R16_UINT / R16G16_UINT: required vertex formats on every driver)
            auto SkinAttributes = std::array
            { xgpu::vertex_descriptor::attribute{ .m_Offset = 0,  .m_Format = xgpu::vertex_descriptor::format::SINT16_4D, .m_iStream = 0 }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 6,  .m_Format = xgpu::vertex_descriptor::format::UINT16_1D, .m_iStream = 0 }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 8,  .m_Format = xgpu::vertex_descriptor::format::UINT16_2D, .m_iStream = 0 }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 0,  .m_Format = xgpu::vertex_descriptor::format::UINT16_2D, .m_iStream = 1 }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 4,  .m_Format = xgpu::vertex_descriptor::format::UINT16_2D, .m_iStream = 1 }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 8,  .m_Format = xgpu::vertex_descriptor::format::UINT16_1D, .m_iStream = 1 }
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 10, .m_Format = xgpu::vertex_descriptor::format::UINT16_1D, .m_iStream = 1 }
            };
            if( auto Err = Device.Create(m_SkinGeomVD, { .m_bUseStreaming = true, .m_Topology = xgpu::vertex_descriptor::topology::TRIANGLE_LIST, .m_VertexSize = 0, .m_Attributes = SkinAttributes }); Err ) return Err;
            if( auto Err = Device.Create(m_StaticGeomUBO, { .m_Type = xgpu::buffer::type::UNIFORM, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::ubo_geom_view), .m_EntryCount = 64 }); Err ) return Err;
            {
                // Device-local (GPU_READ: created from zeros, then only written by GPU copies); the ring: host-visible, a region per frame
                std::vector<details::instance_data> Zeros(max_instances_v + max_transient_v, details::instance_data{});
                void* pZeros = Zeros.data();
                if( auto Err = Device.Create(m_InstanceBuffer, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(details::instance_data), .m_EntryCount = int(Zeros.size()), .m_pData = pZeros }); Err ) return Err;
                if( auto Err = Device.Create(m_InstanceRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::instance_data), .m_EntryCount = ring_frames_v * (max_instances_v + max_transient_v) }); Err ) return Err;
            }
            m_InstanceSlots.Init(max_instances_v);

            xgpu::shader Vert, Frag;
            // the GI variant: the constant material draws read the surfel GI (bindings 5, 6)
            if( auto Err = Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_StaticGeomGIVert) }); Err ) return Err;
            // the constant material comes from the instance record (xlionrender_geom_standard_frag.glsl): no per-draw material entry
            if( auto Err = Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_GeomStandardFrag) }); Err ) return Err;

            auto Shaders      = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            const auto Samplers = LitSamplers();
            auto UniformBinds = std::array
            { xgpu::pipeline::uniform_binds{ .m_BindIndex = 0, .m_Usage = { .m_bVertex = true },   .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 0, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 1, .m_Usage = { .m_bVertex = true, .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // instances (transform; constant material)
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 3, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // lights
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 4, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // light cells
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 5, .m_Usage = { .m_bVertex = true, .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // GI: the mesh's surfels (+ the per-pixel read)
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 6, .m_Usage = { .m_bVertex = true, .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // GI: the surfels' light
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 7, .m_Usage = { .m_bVertex = true, .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // GI: their depth moments
            };
            if( auto Err = Device.Create(m_StaticGeomPipeline, { .m_VertexDescriptor = m_StaticGeomVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                                               , .m_UniformBinds = UniformBinds, .m_Samplers = Samplers }); Err ) return Err;

            auto Bindings = LitBindings({});
            if( auto Err = Device.Create(m_StaticGeomInstance, { .m_PipeLine = m_StaticGeomPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;

            // and its GI capture (the constant material's albedo and emission, from the instance record)
            xgpu::shader CaptureVert;
            if( auto Err = Device.Create(CaptureVert, { .m_Type = xgpu::shader::type::bit::VERTEX, .m_Sharer = SPIRV(s_GICaptureVert) }); Err ) return Err;
            auto CaptureShaders = std::array<const xgpu::shader*, 2>{ &Frag, &CaptureVert };
            xgpu::pipeline::setup CaptureSetup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = CaptureShaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                              , .m_UniformBinds = UniformBinds, .m_Samplers = Samplers };
            CaptureSetup.m_Primitive.m_Cull = xgpu::pipeline::primitive::cull::NONE;
            if( auto Err = Device.Create(m_GIConstCapturePipeline, CaptureSetup); Err ) return Err;
            if( auto Err = Device.Create(m_GIConstCaptureInstance, { .m_PipeLine = m_GIConstCapturePipeline, .m_SamplersBindings = Bindings }); Err ) return Err;
        }

        // Tone map pipeline (fullscreen triangle sampling the view's HDR target)
        {
            xgpu::shader Vert, Frag;
            if( auto Err = Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_FullscreenVert) }); Err ) return Err;
            if( auto Err = Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_ToneMapFrag) }); Err ) return Err;

            auto Shaders  = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            auto Samplers = std::array{ xgpu::pipeline::sampler{} };
            xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_tonemap), .m_Samplers = Samplers };
            Setup.m_Primitive.m_Cull                 = xgpu::pipeline::primitive::cull::NONE;
            Setup.m_DepthStencil.m_bDepthTestEnable  = false;
            Setup.m_DepthStencil.m_bDepthWriteEnable = false;
            if( auto Err = Device.Create(m_ToneMapPipeline, Setup); Err ) return Err;

            // Blit: the same fullscreen triangle, alpha blended into whatever pass is current
            xgpu::shader Blit;
            if( auto Err = Device.Create(Blit, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_BlitFrag) }); Err ) return Err;
            auto BlitShaders = std::array<const xgpu::shader*, 2>{ &Blit, &Vert };
            xgpu::pipeline::setup BlitSetup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = BlitShaders, .m_Samplers = Samplers, .m_Blend = xgpu::pipeline::blend::getAlphaOriginal() };
            BlitSetup.m_Primitive.m_Cull                 = xgpu::pipeline::primitive::cull::NONE;
            BlitSetup.m_DepthStencil.m_bDepthTestEnable  = false;
            BlitSetup.m_DepthStencil.m_bDepthWriteEnable = false;
            if( auto Err = Device.Create(m_BlitPipeline, BlitSetup); Err ) return Err;

            if( auto Err = Device.Create(m_FullscreenVB, { .m_Type = xgpu::buffer::type::VERTEX, .m_EntryByteSize = sizeof(fullscreen_vert), .m_EntryCount = 3 }); Err ) return Err;
            (void)m_FullscreenVB.MemoryMap(0, 3, [](void* p) { const fullscreen_vert V[] = { {-1,-1}, {3,-1}, {-1,3} }; std::memcpy(p, V, sizeof(V)); });
            if( auto Err = Device.Create(m_FullscreenIB, { .m_Type = xgpu::buffer::type::INDEX, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = 3 }); Err ) return Err;
            (void)m_FullscreenIB.MemoryMap(0, 3, [](void* p) { const std::uint32_t I[] = { 0, 1, 2 }; std::memcpy(p, I, sizeof(I)); });
        }

        if( auto Err = Device.Create(m_DefaultMaterialFrag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_DefaultMaterialFrag) }); Err ) return Err;

        // Shadows: the atlas's passes (a clear for its first use - a never rendered attachment cannot be loaded -, then load), the
        // tile clear (depth 1, ALWAYS) and the casters' depth pipelines: depth bias against acne, depth clamp so casters nearer to the
        // light than a level's range still cast ("pancaking")
        {
            auto Attachments = std::array{ xgpu::renderpass::attachment{ m_ShadowAtlas } };
            if( auto Err = Device.Create(m_ShadowClearPass, { .m_Attachments = Attachments }); Err ) return Err;
            if( auto Err = Device.Create(m_ShadowPass,      { .m_Attachments = Attachments, .m_bClearOnRender = false }); Err ) return Err;
            auto Scratch = std::array{ xgpu::renderpass::attachment{ m_ShadowScratch } };
            if( auto Err = Device.Create(m_ScratchPass,     { .m_Attachments = Scratch }); Err ) return Err;

            xgpu::shader Depth, ClearVert, ForwardVert, RigidVert, SkinVert;
            if( auto Err = Device.Create(Depth,       { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_DepthFrag) }); Err ) return Err;
            if( auto Err = Device.Create(ClearVert,   { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_ClearDepthVert) }); Err ) return Err;
            if( auto Err = Device.Create(ForwardVert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_ForwardVert) }); Err ) return Err;
            if( auto Err = Device.Create(RigidVert,   { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_StaticGeomVert) }); Err ) return Err;
            if( auto Err = Device.Create(SkinVert,    { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_SkinGeomVert) }); Err ) return Err;

            {
                auto Shaders = std::array<const xgpu::shader*, 2>{ &Depth, &ClearVert };
                xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders };
                Setup.m_Primitive.m_Cull                = xgpu::pipeline::primitive::cull::NONE;
                Setup.m_DepthStencil.m_DepthCompare     = xgpu::pipeline::depth_stencil::depth_compare::ALWAYS;
                if( auto Err = Device.Create(m_ShadowClearPipeline, Setup); Err ) return Err;
                if( auto Err = Device.Create(m_ShadowClearInstance, { .m_PipeLine = m_ShadowClearPipeline }); Err ) return Err;
            }

            // The local maps' remap: scratch -> atlas, depth written by the shader (ALWAYS)
            {
                xgpu::shader Remap;
                if( auto Err = Device.Create(Remap, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_ShadowRemapFrag) }); Err ) return Err;
                auto Shaders  = std::array<const xgpu::shader*, 2>{ &Remap, &ClearVert };
                auto Samplers = std::array{ xgpu::pipeline::sampler{} };
                xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_shadow_remap), .m_Samplers = Samplers };
                Setup.m_Primitive.m_Cull            = xgpu::pipeline::primitive::cull::NONE;
                Setup.m_DepthStencil.m_DepthCompare = xgpu::pipeline::depth_stencil::depth_compare::ALWAYS;
                if( auto Err = Device.Create(m_RemapPipeline, Setup); Err ) return Err;
                auto Bindings = std::array{ xgpu::pipeline_instance::sampler_binding{ m_ShadowScratch } };
                if( auto Err = Device.Create(m_RemapInstance, { .m_PipeLine = m_RemapPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;
            }

            if( auto Err = Device.Create(m_ShadowUBO, { .m_Type = xgpu::buffer::type::UNIFORM, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::ubo_geom_view), .m_EntryCount = max_shadow_draws_per_frame_v }); Err ) return Err;

            auto Binds = std::array
            { xgpu::pipeline::uniform_binds{ .m_BindIndex = 0, .m_Usage = { .m_bVertex = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 0, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 1, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 2, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }
            };
            const auto MakeCaster = [&](int k, xgpu::vertex_descriptor& VD, const xgpu::shader& Vert, std::size_t PushSize, std::size_t nBinds) -> xgpu::device::error*
            {
                auto Shaders = std::array<const xgpu::shader*, 2>{ &Depth, &Vert };
                xgpu::pipeline::setup Setup{ .m_VertexDescriptor = VD, .m_Shaders = Shaders, .m_PushConstantsSize = PushSize, .m_UniformBinds = std::span{ Binds.data(), nBinds } };
                Setup.m_Primitive.m_Cull                        = xgpu::pipeline::primitive::cull::NONE;
                Setup.m_DepthStencil.m_DepthBiasConstantFactor  = 2.0f;
                Setup.m_DepthStencil.m_DepthBiasSlopeFactor     = 4.0f;
                Setup.m_DepthStencil.m_bDepthBiasEnable         = true;
                Setup.m_DepthStencil.m_bDepthClampEnable        = true;
                if( auto Err = Device.Create(m_ShadowPipeline[k], Setup); Err ) return Err;
                return Device.Create(m_ShadowInstance[k], { .m_PipeLine = m_ShadowPipeline[k] });
            };
            if( auto Err = MakeCaster(0, m_MeshVD,       ForwardVert, sizeof(details::push_forward),     0); Err ) return Err;     // CreateMesh meshes
            if( auto Err = MakeCaster(1, m_StaticGeomVD, RigidVert,   sizeof(details::push_static_geom), 3); Err ) return Err;     // RIGID
            if( auto Err = MakeCaster(2, m_SkinGeomVD,   SkinVert,    sizeof(details::push_skin_geom),   4); Err ) return Err;     // SKIN

            // The selection's mask: the same vertex paths, the mask's fragment (it samples the scene's depth), no bias, the depth test and write of its own
            {
                xgpu::shader MaskFrag;
                if( auto Err = Device.Create(MaskFrag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_OutlineMaskFrag) }); Err ) return Err;
                const auto MakeMask = [&](int k, xgpu::vertex_descriptor& VD, const xgpu::shader& Vert, std::size_t PushSize, std::size_t nBinds) -> xgpu::device::error*
                {
                    auto Shaders  = std::array<const xgpu::shader*, 2>{ &MaskFrag, &Vert };
                    auto Samplers = std::array{ xgpu::pipeline::sampler{ .m_bDisableAnisotropy = true } };
                    xgpu::pipeline::setup Setup{ .m_VertexDescriptor = VD, .m_Shaders = Shaders, .m_PushConstantsSize = PushSize, .m_UniformBinds = std::span{ Binds.data(), nBinds }, .m_Samplers = Samplers };
                    Setup.m_Primitive.m_Cull = xgpu::pipeline::primitive::cull::NONE;
                    return Device.Create(m_MaskPipeline[k], Setup);
                };
                if( auto Err = MakeMask(0, m_MeshVD,       ForwardVert, sizeof(details::push_forward),     0); Err ) return Err;
                if( auto Err = MakeMask(1, m_StaticGeomVD, RigidVert,   sizeof(details::push_static_geom), 3); Err ) return Err;
                if( auto Err = MakeMask(2, m_SkinGeomVD,   SkinVert,    sizeof(details::push_skin_geom),   4); Err ) return Err;
                m_bOutlineReady = true;
            }
        }

        // The environment probe's passes and pipelines (fullscreen triangles; the sky and the prefilter rebuild only on a sky change)
        {
            xgpu::shader ClearVert2, Sky, Prefilter, DFG, Downsample, Cubemap;
            if( auto Err = Device.Create(Cubemap,    { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_EnvCubemapFrag) }); Err ) return Err;
            if( auto Err = Device.Create(ClearVert2, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_ClearDepthVert) }); Err ) return Err;
            if( auto Err = Device.Create(Sky,        { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_EnvSkyFrag) }); Err ) return Err;
            if( auto Err = Device.Create(Prefilter,  { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_EnvPrefilterFrag) }); Err ) return Err;
            if( auto Err = Device.Create(DFG,        { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_EnvDFGFrag) }); Err ) return Err;
            if( auto Err = Device.Create(Downsample, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_EnvDownsampleFrag) }); Err ) return Err;

            for( int k = 0; k < env_source_levels_v; ++k )
            {
                auto A = std::array{ xgpu::renderpass::attachment{ m_EnvSource[k] } };
                if( auto Err = Device.Create(m_EnvSourcePass[k], { .m_Attachments = A }); Err ) return Err;
            }
            auto A1 = std::array{ xgpu::renderpass::attachment{ m_EnvAtlas } };
            auto A2 = std::array{ xgpu::renderpass::attachment{ m_EnvDFG } };
            if( auto Err = Device.Create(m_EnvAtlasPass,  { .m_Attachments = A1 }); Err ) return Err;
            if( auto Err = Device.Create(m_EnvDFGPass,    { .m_Attachments = A2 }); Err ) return Err;

            const auto Clamp = xgpu::pipeline::sampler{ .m_bDisableAnisotropy = true, .m_AddressMode = std::array{ xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP } };
            const auto MakePipe = [&](xgpu::pipeline& Pipe, const xgpu::shader& Frag, std::size_t PushSize, int nSamplers) -> xgpu::device::error*
            {
                auto Shaders  = std::array<const xgpu::shader*, 2>{ &Frag, &ClearVert2 };
                std::vector<xgpu::pipeline::sampler> Samplers(nSamplers, Clamp);
                xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders, .m_PushConstantsSize = PushSize, .m_Samplers = Samplers };
                Setup.m_Primitive.m_Cull            = xgpu::pipeline::primitive::cull::NONE;
                Setup.m_DepthStencil.m_DepthCompare = xgpu::pipeline::depth_stencil::depth_compare::ALWAYS;
                return Device.Create(Pipe, Setup);
            };
            if( auto Err = MakePipe(m_EnvSkyPipeline,        Sky,        sizeof(details::push_env_sky),        0); Err ) return Err;
            if( auto Err = MakePipe(m_EnvPrefilterPipeline,  Prefilter,  sizeof(details::push_env_prefilter),  env_source_levels_v); Err ) return Err;
            if( auto Err = MakePipe(m_EnvDFGPipeline,        DFG,        sizeof(details::push_env_dfg),        0); Err ) return Err;
            if( auto Err = MakePipe(m_EnvDownsamplePipeline, Downsample, sizeof(details::push_env_downsample), 1); Err ) return Err;
            if( auto Err = MakePipe(m_EnvCubemapPipeline,    Cubemap,    sizeof(details::push_env_cube_to_oct), 1); Err ) return Err;
            // The sky as a background: depth-tested against the scene at the far plane, nothing written to the depth
            {
                xgpu::shader SkyBackground;
                if( auto Err = Device.Create(SkyBackground, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_SkyBackgroundFrag) }); Err ) return Err;
                auto Shaders  = std::array<const xgpu::shader*, 2>{ &SkyBackground, &ClearVert2 };
                auto Samplers = std::array{ Clamp };
                xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_sky_background), .m_Samplers = Samplers };
                Setup.m_Primitive.m_Cull                 = xgpu::pipeline::primitive::cull::NONE;
                Setup.m_DepthStencil.m_DepthCompare      = xgpu::pipeline::depth_stencil::depth_compare::LESS_OR_EQUAL;
                Setup.m_DepthStencil.m_bDepthWriteEnable = false;
                if( auto Err = Device.Create(m_SkyBackgroundPipeline, Setup); Err ) return Err;
            }
#if 0       // The first attempt: the primitives' OBB outline shader over the scene (it needs a mesh that is box-like): replaced by the mask and the composite below
            // The selection's outline: the editor's outline shader (the primitives' selection outline), here over the scene: cull none, the depth bias
            // away from the eye (the object's own pixels, at the same depth, fail the test: only the ring around its silhouette shows), nothing written
            {
                xgpu::shader Vert, Frag;
                if( auto Err = Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (std::int32_t*)s_OutlineObbVert, std::size(s_OutlineObbVert) } } }); Err ) return Err;
                if( auto Err = Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (std::int32_t*)s_OutlineObbFrag, std::size(s_OutlineObbFrag) } } }); Err ) return Err;
                auto Attributes = std::array
                { xgpu::vertex_descriptor::attribute{ .m_Offset = 0,                  .m_Format = xgpu::vertex_descriptor::format::FLOAT_3D }
                , xgpu::vertex_descriptor::attribute{ .m_Offset = 3 * sizeof(float),  .m_Format = xgpu::vertex_descriptor::format::FLOAT_3D }
                };
                if( auto Err = Device.Create(m_OutlineVD, xgpu::vertex_descriptor::setup{ .m_Topology = xgpu::vertex_descriptor::topology::TRIANGLE_LIST, .m_VertexSize = 6 * sizeof(float), .m_Attributes = Attributes }); Err ) return Err;
                auto Shaders = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
                if( auto Err = Device.Create(m_OutlinePipeline, xgpu::pipeline::setup{ .m_VertexDescriptor = m_OutlineVD, .m_Shaders = Shaders
                    , .m_PushConstantsSize = sizeof(details::push_outline_obb)
                    , .m_Primitive    = { .m_Cull = xgpu::pipeline::primitive::cull::NONE }
                    , .m_DepthStencil = { .m_DepthBiasConstantFactor = 1.25f, .m_DepthBiasSlopeFactor = 1.75f, .m_bDepthWriteEnable = false, .m_bDepthBiasEnable = true } }); Err ) return Err;
                if( auto Err = Device.Create(m_OutlineInstance, { .m_PipeLine = m_OutlinePipeline }); Err ) return Err;
            }
#endif
            // The selection's outline composite: over the tone-mapped image, alpha blended (the alpha channel as coverage: a over what is there), no depth
            {
                xgpu::shader Frag;
                if( auto Err = Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_OutlineCompositeFrag) }); Err ) return Err;
                auto Shaders  = std::array<const xgpu::shader*, 2>{ &Frag, &ClearVert2 };
                auto Samplers = std::array{ xgpu::pipeline::sampler{ .m_bDisableAnisotropy = true } };
                xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_outline_composite), .m_Samplers = Samplers };
                Setup.m_Primitive.m_Cull                 = xgpu::pipeline::primitive::cull::NONE;
                Setup.m_DepthStencil.m_bDepthTestEnable  = false;
                Setup.m_DepthStencil.m_bDepthWriteEnable = false;
                Setup.m_Blend.m_bEnable = true;
                Setup.m_Blend.m_ColorSrcFactor = xgpu::pipeline::blend::factor::SRC_ALPHA; Setup.m_Blend.m_ColorDstFactor = xgpu::pipeline::blend::factor::ONE_MINUS_SRC_ALPHA;
                Setup.m_Blend.m_AlphaSrcFactor = xgpu::pipeline::blend::factor::ONE;       Setup.m_Blend.m_AlphaDstFactor = xgpu::pipeline::blend::factor::ONE_MINUS_SRC_ALPHA;
                if( auto Err = Device.Create(m_OutlineCompositePipeline, Setup); Err ) return Err;
            }
            if( auto Err = Device.Create(m_EnvSkyInstance, { .m_PipeLine = m_EnvSkyPipeline }); Err ) return Err;
            if( auto Err = Device.Create(m_EnvDFGInstance, { .m_PipeLine = m_EnvDFGPipeline }); Err ) return Err;
            {
                std::vector<xgpu::pipeline_instance::sampler_binding> Levels;
                for( auto& T : m_EnvSource ) Levels.emplace_back(T);
                if( auto Err = Device.Create(m_EnvPrefilterInstance, { .m_PipeLine = m_EnvPrefilterPipeline, .m_SamplersBindings = Levels }); Err ) return Err;
            }
            for( int k = 1; k < env_source_levels_v; ++k )
            {
                auto Parent = std::array{ xgpu::pipeline_instance::sampler_binding{ m_EnvSource[k - 1] } };
                if( auto Err = Device.Create(m_EnvDownsampleInstance[k], { .m_PipeLine = m_EnvDownsamplePipeline, .m_SamplersBindings = Parent }); Err ) return Err;
            }
        }

        // GPU timings: 0 = scene start, 1 = scene end, 2 = tone map end
        if( auto Err = Device.Create(m_Timestamps, { .m_Count = 4 }); Err ) return Err;     // 0 shadows start, 1 scene start, 2 scene end, 3 tone map end

        // Reflection probes: the atlas' passes (a clear for its first use, then load), the capture views, the cube -> octahedral pass
        {
            auto A = std::array{ xgpu::renderpass::attachment{ m_ProbeAtlas } };
            if( auto Err = Device.Create(m_ProbeAtlasClearPass, { .m_Attachments = A }); Err ) return Err;
            if( auto Err = Device.Create(m_ProbeAtlasPass,      { .m_Attachments = A, .m_bClearOnRender = false }); Err ) return Err;
            m_bReady = true;                        // CreateView needs the tone map pipeline: everything it uses exists by now
            for( auto& V : m_CaptureViews ) if( auto Err = CreateView(V, { .m_Width = probe_face_size_v, .m_Height = probe_face_size_v }); Err ) return Err;

            xgpu::shader ClearVert3, CubeToOct;
            if( auto Err = Device.Create(ClearVert3, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_ClearDepthVert) }); Err ) return Err;
            if( auto Err = Device.Create(CubeToOct,  { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_EnvCubeToOctFrag) }); Err ) return Err;
            auto Shaders = std::array<const xgpu::shader*, 2>{ &CubeToOct, &ClearVert3 };
            std::vector<xgpu::pipeline::sampler> Samplers(6, xgpu::pipeline::sampler{ .m_bDisableAnisotropy = true, .m_AddressMode = std::array{ xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP } });
            xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_env_cube_to_oct), .m_Samplers = Samplers };
            Setup.m_Primitive.m_Cull            = xgpu::pipeline::primitive::cull::NONE;
            Setup.m_DepthStencil.m_DepthCompare = xgpu::pipeline::depth_stencil::depth_compare::ALWAYS;
            if( auto Err = Device.Create(m_CubeToOctPipeline, Setup); Err ) return Err;
            std::vector<xgpu::pipeline_instance::sampler_binding> Faces;
            for( auto& V : m_CaptureViews ) Faces.emplace_back(V.m_HDR);
            if( auto Err = Device.Create(m_CubeToOctInstance, { .m_PipeLine = m_CubeToOctPipeline, .m_SamplersBindings = Faces }); Err ) return Err;
        }

        m_bReady = true;
        return nullptr;
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::CreateMesh( mesh_id& Id, std::span<const vertex> Vertices, std::span<const std::uint32_t> Indices ) noexcept
    {
        assert(m_bReady && !Vertices.empty() && !Indices.empty());

        mesh_slot M;
        if( auto Err = m_Device.Create(M.m_VertexBuffer, { .m_Type = xgpu::buffer::type::VERTEX, .m_EntryByteSize = sizeof(vertex),        .m_EntryCount = static_cast<int>(Vertices.size()) }); Err ) return Err;
        if( auto Err = m_Device.Create(M.m_IndexBuffer,  { .m_Type = xgpu::buffer::type::INDEX,  .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = static_cast<int>(Indices.size())  }); Err ) return Err;
        (void)M.m_VertexBuffer.MemoryMap(0, static_cast<int>(Vertices.size()), [&](void* p) { std::memcpy(p, Vertices.data(), Vertices.size_bytes()); });
        (void)M.m_IndexBuffer .MemoryMap(0, static_cast<int>(Indices.size()),  [&](void* p) { std::memcpy(p, Indices.data(),  Indices.size_bytes());  });
        M.m_nIndices = static_cast<int>(Indices.size());
        M.m_BBox.m_Min = M.m_BBox.m_Max = xmath::fvec3(Vertices[0].m_Pos[0], Vertices[0].m_Pos[1], Vertices[0].m_Pos[2]);
        for( auto& V : Vertices ) { const xmath::fvec3 P(V.m_Pos[0], V.m_Pos[1], V.m_Pos[2]); M.m_BBox.m_Min = xmath::fvec3::Min(M.m_BBox.m_Min, P); M.m_BBox.m_Max = xmath::fvec3::Max(M.m_BBox.m_Max, P); }
        const auto [Index, Gen] = AddSlot(m_Meshes, std::move(M));
        Id = { Index, Gen };
        return nullptr;
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::CreateClusterGeom( mesh_id& Id, cluster_geom Geom ) noexcept
    {
        assert(m_bReady && Geom.m_pStreams && Geom.m_pClusterData && (Geom.m_Kind != geom_kind::SKIN || Geom.m_pBones));
        mesh_slot M;
        M.m_Geom     = std::move(Geom);
        M.m_bCluster = true;
        const auto [Index, Gen] = AddSlot(m_Meshes, std::move(M));
        Id = { Index, Gen };
        return nullptr;
    }

    //------------------------------------------------------------------------------------------------

    //------------------------------------------------------------------------------------------------

    void instance::DestroyMesh( mesh_id Id ) noexcept
    {
        if( findMesh(Id) == nullptr ) return;
        auto& M = m_Meshes[Id.m_Index];
        if( M.m_bCluster == false )
        {
            ReleaseBuffer(M.m_VertexBuffer);
            ReleaseBuffer(M.m_IndexBuffer);
        }
        if( M.m_pGI ) { ReleaseBuffer(M.m_GIBuffer); M.m_pGI.reset(); }
        if( M.m_nOutlineIndices > 0 ) { ReleaseBuffer(M.m_OutlineVB); ReleaseBuffer(M.m_OutlineIB); M.m_nOutlineIndices = 0; }
        M.m_Geom = {};
        M.m_bAlive     = false;
    }

    //------------------------------------------------------------------------------------------------

    // The vertex shader's GIMesh (xlionrender_static_geom_body.glsl): [0] level 0's first uint, [1] level 1's; per vertex 2 uints
    // (4 x 16-bit surfel indices, 0xFFFF none); per surfel 8: center xyz, radius, normal xyz, parent (uint bits)
    xgpu::device::error* instance::SetOutlineProxy( mesh_id Id, std::span<const float> PosNrm, std::span<const std::uint32_t> Indices, const xmath::fvec3& Center, const xmath::fvec3& Half ) noexcept
    {
        if( findMesh(Id) == nullptr || PosNrm.empty() || Indices.empty() ) return nullptr;
        auto& M = m_Meshes[Id.m_Index];
        if( M.m_nOutlineIndices > 0 ) return nullptr;
        if( auto Err = m_Device.Create(M.m_OutlineVB, { .m_Type = xgpu::buffer::type::VERTEX, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 6 * sizeof(float), .m_EntryCount = int(PosNrm.size() / 6) }); Err ) return Err;
        (void)M.m_OutlineVB.MemoryMap(0, int(PosNrm.size() / 6), [&](void* p) { std::memcpy(p, PosNrm.data(), PosNrm.size() * sizeof(float)); });
        if( auto Err = m_Device.Create(M.m_OutlineIB, { .m_Type = xgpu::buffer::type::INDEX, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = int(Indices.size()) }); Err ) return Err;
        (void)M.m_OutlineIB.MemoryMap(0, int(Indices.size()), [&](void* p) { std::memcpy(p, Indices.data(), Indices.size() * sizeof(std::uint32_t)); });
        M.m_OutlineCenter = Center; M.m_OutlineHalf = Half;
        M.m_nOutlineIndices = int(Indices.size());
        return nullptr;
    }

    xgpu::device::error* instance::SetMeshGI( mesh_id Id, gi::surfel_mesh&& Surfels, std::vector<std::uint32_t>&& Packed ) noexcept
    {
        if( findMesh(Id) == nullptr ) return nullptr;
        auto& M = m_Meshes[Id.m_Index];
        assert(M.m_bCluster && M.m_Geom.m_Kind == geom_kind::RIGID && Surfels.m_Levels[0].size() < 0xFFFF);
        if( Surfels.m_ReadTable.empty() ) Surfels.BuildReadGrid();
        M.m_pGI = std::make_unique<gi::surfel_mesh>(std::move(Surfels));
        static std::uint64_t s_GIStamp = 0;
        M.m_GIStamp = ++s_GIStamp;
        if( Packed.empty() ) return buildMeshGI(M, {});
        return createMeshGI(M, Packed);                                       // (Phase 7: packed on a worker: PackMeshGI)
    }

    // The layout of shaders/xlionrender_surfel_gi.glsl: a header of 8 uints ([0] level 0 surfels, [1] level 1 surfels, [2] the read grid's
    // table, [3] its mask (0: none), [4] its cell (float bits), [5] its lists, [6] its longest probe), per vertex 2 uints, the surfels, the
    // grid. [7]: the triangles' colors (0: none), the SURFACE_COLOR view's: the clusters' count, each cluster's first triangle, a color
    // per triangle (unorm 4x8) - a pixel's triangle is its cluster's first + gl_PrimitiveID. Pure: a worker can pack it (no colors)
    std::vector<std::uint32_t> PackMeshGI( const gi::surfel_mesh& Surfels, std::span<const std::uint32_t> ClusterFirstTri, std::span<const xmath::fvec3> TriColors ) noexcept
    {
        const std::uint32_t nVerts = std::uint32_t(Surfels.m_Pos.size());
        std::vector<std::uint32_t> Data(8 + 2 * nVerts);
        Data[0] = std::uint32_t(Data.size());
        for( std::uint32_t v = 0; v < nVerts; ++v )
        {
            const auto& I = Surfels.m_VertSurfels[v];
            const auto  U = [&](int k) { return I[k] < 0 ? 0xFFFFu : std::uint32_t(I[k]); };
            Data[8 + 2 * v] = U(0) | (U(1) << 16);
            Data[9 + 2 * v] = U(2) | (U(3) << 16);
        }
        for( int L = 0; L < 2; ++L )
        {
            if( L == 1 ) Data[1] = std::uint32_t(Data.size());
            const auto& Level = Surfels.m_Levels[L];
            for( std::size_t s = 0; s < Level.size(); ++s )
            {
                const auto& S = Level[s];
                const float F[7] = { S.m_Center.m_X, S.m_Center.m_Y, S.m_Center.m_Z, S.m_Radius, S.m_Normal.m_X, S.m_Normal.m_Y, S.m_Normal.m_Z };
                for( float f : F ) Data.push_back(std::bit_cast<std::uint32_t>(f));
                Data.push_back(L == 0 ? std::uint32_t(Surfels.m_ParentOf[s]) | (s >= Surfels.m_FirstPatch ? 0x80000000u : 0u) : 0u);      // bit 31: a patch (the COVERAGE view's yellow)
            }
        }
        if( !Surfels.m_ReadTable.empty() )
        {
            Data[2] = std::uint32_t(Data.size());
            Data[3] = std::uint32_t(Surfels.m_ReadTable.size() - 1);
            Data[4] = std::bit_cast<std::uint32_t>(Surfels.m_ReadCell);
            Data[6] = Surfels.m_ReadMaxProbe;
            for( const auto& E : Surfels.m_ReadTable ) Data.insert(Data.end(), E.begin(), E.end());
            Data[5] = std::uint32_t(Data.size());
            Data.insert(Data.end(), Surfels.m_ReadList.begin(), Surfels.m_ReadList.end());
        }
        if( !TriColors.empty() )
        {
            Data[7] = std::uint32_t(Data.size());
            Data.push_back(std::uint32_t(ClusterFirstTri.size()));
            Data.insert(Data.end(), ClusterFirstTri.begin(), ClusterFirstTri.end());
            const auto U8 = [](float f) { return std::uint32_t(std::clamp(f, 0.0f, 1.0f) * 255.0f + 0.5f); };
            for( const auto& c : TriColors ) Data.push_back(U8(c.m_X) | (U8(c.m_Y) << 8) | (U8(c.m_Z) << 16) | (255u << 24));
        }
        return Data;
    }

    xgpu::device::error* instance::buildMeshGI( mesh_slot& M, std::span<const xmath::fvec3> TriColors ) noexcept
    {
        std::vector<std::uint32_t> First;
        if( !TriColors.empty() ) for( const auto& C : M.m_Geom.m_Clusters ) First.push_back(C.m_iIndex / 3);
        return createMeshGI(M, PackMeshGI(*M.m_pGI, First, TriColors));
    }

    xgpu::device::error* instance::createMeshGI( mesh_slot& M, std::span<const std::uint32_t> Data ) noexcept
    {
        if( M.m_GIBuffer.m_Private ) ReleaseBuffer(M.m_GIBuffer);
        return m_Device.Create(M.m_GIBuffer, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = int(Data.size()), .m_pData = const_cast<std::uint32_t*>(Data.data()) });
    }

    //------------------------------------------------------------------------------------------------

    const instance::mesh_slot* instance::findMesh( mesh_id Id ) const noexcept
    {
        return FindSlot(const_cast<std::vector<mesh_slot>&>(m_Meshes), Id);
    }

    //------------------------------------------------------------------------------------------------

    instance::material_slot* instance::findMaterial( material_id Id ) noexcept
    {
        return FindSlot(m_Materials, Id);
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::CreateMaterial( material_id& Id, const xgpu::shader& FragmentHDR, std::span<xgpu::texture* const> Textures, bool bAlpha ) noexcept
    {
        assert(m_bReady && !Textures.empty());

        material_slot M;
        M.m_bAlpha = bAlpha;
        M.m_Fragment = FragmentHDR;
        M.m_Textures.clear();
        for( auto* p : Textures ) M.m_Textures.push_back(p ? *p : xgpu::texture{});

        // The rigid pipeline now, so a material that cannot build reports it here (the skin one is built on its first skinned draw)
        if( getMaterialInstance(M, geom_kind::RIGID) == nullptr ) return VGPU_ERROR(xgpu::device::error::FAILURE, "The material's pipeline could not be created");

        const auto [Index, Gen] = AddSlot(m_Materials, std::move(M));
        Id = { Index, Gen };
        return nullptr;
    }

    //------------------------------------------------------------------------------------------------

    xgpu::pipeline_instance* instance::getMaterialInstance( material_slot& M, geom_kind Kind ) noexcept
    {
        const auto k = static_cast<std::size_t>(Kind);
        if( M.m_Instance[k].m_Private ) return &M.m_Instance[k];

        const bool bSkin = Kind == geom_kind::SKIN;
        xgpu::shader Vert;
        if( m_Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX, .m_Sharer = bSkin ? SPIRV(s_SkinGeomVert) : SPIRV(s_StaticGeomGIVert) }) ) return nullptr;     // RIGID: reads the surfel GI

        // Slot 0 (shadow map) clamps, the material textures use the default sampler - as the geom editors
        assert(M.m_Textures.size() <= 8);       // slots 8, 9: the environment
        const auto Samplers = LitSamplers();

        auto Shaders      = std::array<const xgpu::shader*, 2>{ &M.m_Fragment, &Vert };
        using ub = xgpu::pipeline::uniform_binds;
        const auto VertexSSBO   = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = ub::type::SSBO_STATIC }; };
        const auto FragmentSSBO = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = { .m_bFragment = true }, .m_Type = ub::type::SSBO_STATIC }; };
        const auto BothSSBO     = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = { .m_bVertex = true, .m_bFragment = true }, .m_Type = ub::type::SSBO_STATIC }; };
        std::vector<ub> UniformBinds
        { ub{ .m_BindIndex = 0, .m_Usage = { .m_bVertex = true },   .m_Type = ub::type::UBO_DYNAMIC }
        , ub{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = ub::type::UBO_DYNAMIC }
        , VertexSSBO(0)                                         // clusters
        , VertexSSBO(1)                                         // RIGID: instances, SKIN: bones
        };
        if( bSkin ) UniformBinds.push_back(VertexSSBO(2));     // SKIN: instances
        UniformBinds.push_back(FragmentSSBO(3));                // lights
        UniformBinds.push_back(FragmentSSBO(4));                // light cells
        UniformBinds.push_back(BothSSBO(5));                    // the GI: the mesh's surfels (RIGID: the vertex read; both: the per-pixel read
        UniformBinds.push_back(BothSSBO(6));                    // of mb_standard_pbr.frag, off on the skin path), their light
        UniformBinds.push_back(BothSSBO(7));                    //   and their depth moments
        const auto Binds = std::span{ UniformBinds };
        if( m_Device.Create(M.m_Pipeline[k], { .m_VertexDescriptor = bSkin ? m_SkinGeomVD : m_StaticGeomVD, .m_Shaders = Shaders
                                             , .m_PushConstantsSize = bSkin ? sizeof(details::push_skin_geom) : sizeof(details::push_static_geom)
                                             , .m_UniformBinds = Binds, .m_Samplers = Samplers }) ) return nullptr;

        std::vector<xgpu::pipeline_instance::sampler_binding> Bindings;
        Bindings = LitBindings(M.TexturePointers());
        if( m_Device.Create(M.m_Instance[k], { .m_PipeLine = M.m_Pipeline[k], .m_SamplersBindings = Bindings }) )
        {
            m_Device.Destroy(std::move(M.m_Pipeline[k]));
            return nullptr;
        }
        return &M.m_Instance[k];
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::CreateDefaultMaterial( material_id& Id, std::span<xgpu::texture* const> Textures ) noexcept
    {
        return CreateMaterial(Id, m_DefaultMaterialFrag, Textures);
    }

    //------------------------------------------------------------------------------------------------

    // A material's GI capture pipeline: its fragment shader in capture mode behind the capture vertex shader (the quads of
    // shaders/xlionrender_gi_capture_vert.glsl), the rigid path's bindings (binding 0: the capture's surfel data)
    xgpu::pipeline_instance* instance::getCaptureInstance( material_slot& M ) noexcept
    {
        constexpr auto k = capture_path_v;
        if( M.m_Instance[k].m_Private ) return &M.m_Instance[k];
        xgpu::shader Vert;
        if( m_Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX, .m_Sharer = SPIRV(s_GICaptureVert) }) ) return nullptr;
        const auto Samplers = LitSamplers();
        auto Shaders = std::array<const xgpu::shader*, 2>{ &M.m_Fragment, &Vert };
        using ub = xgpu::pipeline::uniform_binds;
        const auto VertexSSBO   = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = ub::type::SSBO_STATIC }; };
        const auto FragmentSSBO = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = { .m_bFragment = true }, .m_Type = ub::type::SSBO_STATIC }; };
        const auto BothSSBO     = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = { .m_bVertex = true, .m_bFragment = true }, .m_Type = ub::type::SSBO_STATIC }; };
        auto Binds = std::array
        { ub{ .m_BindIndex = 0, .m_Usage = { .m_bVertex = true },   .m_Type = ub::type::UBO_DYNAMIC }
        , ub{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = ub::type::UBO_DYNAMIC }
        , VertexSSBO(0), VertexSSBO(1), FragmentSSBO(3), FragmentSSBO(4), BothSSBO(5), BothSSBO(6), BothSSBO(7)
        };
        xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                   , .m_UniformBinds = std::span{ Binds }, .m_Samplers = Samplers };
        Setup.m_Primitive.m_Cull = xgpu::pipeline::primitive::cull::NONE;
        if( m_Device.Create(M.m_Pipeline[k], Setup) ) return nullptr;
        auto Bindings = LitBindings(M.TexturePointers());
        if( m_Device.Create(M.m_Instance[k], { .m_PipeLine = M.m_Pipeline[k], .m_SamplersBindings = Bindings }) )
        {
            m_Device.Destroy(std::move(M.m_Pipeline[k]));
            return nullptr;
        }
        return &M.m_Instance[k];
    }

    //------------------------------------------------------------------------------------------------

    void instance::DestroyMaterial( material_id Id ) noexcept
    {
        auto* pM = findMaterial(Id);
        if( pM == nullptr ) return;
        for( std::size_t k = 0; k < pM->m_Pipeline.size(); ++k )
        {
            if( pM->m_Instance[k].m_Private ) m_Device.Destroy(std::move(pM->m_Instance[k]));
            if( pM->m_Pipeline[k].m_Private ) m_Device.Destroy(std::move(pM->m_Pipeline[k]));
        }
        pM->m_Textures.clear();
        pM->m_Fragment = {};
        pM->m_bAlive = false;
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::CreateView( render_view& View, const render_view::setup& Setup ) noexcept
    {
        assert(m_bReady && Setup.m_Width > 0 && Setup.m_Height > 0);
        DestroyView(View);

        View.m_Width  = Setup.m_Width;
        View.m_Height = Setup.m_Height;
        if( auto Err = m_Device.Create(View.m_HDR,    { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = Setup.m_Width, .m_Height = Setup.m_Height, .m_isGamma = false }); Err ) return Err;
        if( auto Err = m_Device.Create(View.m_Depth,  { .m_Format = xgpu::texture::format::DEPTH_F32,           .m_Width = Setup.m_Width, .m_Height = Setup.m_Height, .m_isGamma = false }); Err ) return Err;
        if( auto Err = m_Device.Create(View.m_Output, { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM,      .m_Width = Setup.m_Width, .m_Height = Setup.m_Height, .m_isGamma = false }); Err ) return Err;

        {
            auto Attachments = std::array{ xgpu::renderpass::attachment{ View.m_HDR }, xgpu::renderpass::attachment{ View.m_Depth } };
            if( auto Err = m_Device.Create(View.m_ScenePass, { .m_Attachments = Attachments, .m_ClearColorR = 0, .m_ClearColorG = 0, .m_ClearColorB = 0, .m_ClearColorA = 0 }); Err ) return Err;
        }
        {
            auto Attachments = std::array{ xgpu::renderpass::attachment{ View.m_Output } };
            if( auto Err = m_Device.Create(View.m_OutputPass, { .m_Attachments = Attachments }); Err ) return Err;
        }
        {
            auto Attachments = std::array{ xgpu::renderpass::attachment{ View.m_Output }, xgpu::renderpass::attachment{ View.m_Depth } };
            if( auto Err = m_Device.Create(View.m_OverlayPass, { .m_Attachments = Attachments, .m_bClearOnRender = false }); Err ) return Err;
        }

        auto Bindings = std::array{ xgpu::pipeline_instance::sampler_binding{ View.m_HDR } };
        if( auto Err = m_Device.Create(View.m_ToneMapInstance, { .m_PipeLine = m_ToneMapPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;

        auto BlitBindings = std::array{ xgpu::pipeline_instance::sampler_binding{ View.m_Output } };
        return m_Device.Create(View.m_BlitInstance, { .m_PipeLine = m_BlitPipeline, .m_SamplersBindings = BlitBindings });
    }

    //------------------------------------------------------------------------------------------------

    int instance::DumpGI( const char* pPath ) const noexcept
    {
        if( !m_pGIWorld ) return -1;
        const int n = m_pGIWorld->Dump(pPath);
        if( n >= 0 ) (void)DumpGIFrames((std::string(pPath) + ".frames.csv").c_str());
        return n;
    }

    int instance::DumpGIFrames( const char* pPath ) const noexcept
    {
        std::FILE* f = nullptr;
    #ifdef _WIN32
        if( fopen_s(&f, pPath, "w") != 0 ) f = nullptr;
    #else
        f = std::fopen(pPath, "w");
    #endif
        if( f == nullptr ) return -1;
        std::fprintf(f, "# GI frame log (documentation/gi_dynamic_master_plan.md, Phase 0): times in ms (main thread), LongestSurfelUs in us; the trace's worker time is TraceMs (the main thread waits for it); Hits evaluate the direct light (shadow rays)\n");
        std::fprintf(f, "frame,UpdateGIMs,TotalMs,TlasLightsMs,LevelsMs,PickMs,LinksMs,EmitMs,MeanMs,TraceMs,CommitMs,CubeMs,StatsMs,FilterMs,RingCopyMs,MomentsMs,CaptureMs,LoadMs,"
                       "Updated,Surfels,Rays,Hits,LongestSurfelUs,LinkRebuilds,Links,Emitters,Moves,MovesMs,MarkedDirty,HistoryResets,Dirty,UploadBytes,Dropped,LinkBytes,LightBytes,FilterBytes,Unsupported,Bad,Cubes,CubeParents,Pending,PendingRays,ChunkRays,RayUsEst,Rounds,CastMs,DirectMs,ReadMs,FairSurfels,LinkPending,LinkSurfels,UrgentSurfels,PreMs,OverAllowMs,TraceCpuMs,LinkCpuMs,LinkLongestUs,TokenUnits\n");
        for( const auto& F : m_GIFrames )
        {
            const auto& S = F.m_Stats;
            std::fprintf(f, "%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%lld,%.1f,%d,%d,%d,%d,%.3f,%d,%d,%d,%zu,%d,%zu,%zu,%zu,%d,%d,%d,%d,%d,%d,%d,%.1f,%d,%.3f,%.3f,%.3f,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%d\n"
                , (unsigned long long)F.m_Frame, S.m_UpdateGIMs, S.m_TotalMs, S.m_TlasLightsMs, S.m_LevelsMs, S.m_PickMs, S.m_LinksMs, S.m_EmitMs, S.m_MeanMs, S.m_TraceMs, S.m_CommitMs, S.m_CubeMs, S.m_StatsMs
                , S.m_FilterMs, S.m_RingCopyMs, S.m_MomentsMs, S.m_CaptureMs, S.m_LoadMs
                , S.m_Updated, S.m_Surfels, S.m_Rays, S.m_Hits, S.m_LongestSurfelUs, S.m_nLinkRebuilds, S.m_nLinks, S.m_nEmitters, S.m_nMoves, S.m_MovesMs, S.m_nMarkedDirty, S.m_nHistoryResets, S.m_Dirty
                , S.m_UploadBytes, S.m_nDropped, S.m_LinkBytes, S.m_LightBytes, S.m_FilterBytes, S.m_nUnsupported, S.m_nBad, S.m_nCubes, S.m_CubeParents, S.m_Pending, S.m_PendingRays, S.m_ChunkRays, double(S.m_RayUsEst), S.m_Rounds, S.m_CastMs, S.m_DirectMs, S.m_ReadMs, S.m_FairSurfels, S.m_LinkPending, S.m_LinkSurfels, S.m_UrgentSurfels, S.m_PreMs, S.m_OverAllowMs, S.m_TraceCpuMs, S.m_LinkCpuMs, S.m_LinkLongestUs, S.m_TokenUnits);
        }
        std::fclose(f);
        return int(m_GIFrames.size());
    }

    xgpu::device::error* instance::createOutlineTargets( render_view& View ) noexcept
    {
        if( View.m_Mask.m_Private && View.m_MaskWidth == View.m_Width && View.m_MaskHeight == View.m_Height ) return nullptr;
        destroyOutlineTargets(View);
        if( auto Err = m_Device.Create(View.m_Mask,      { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = View.m_Width, .m_Height = View.m_Height, .m_isGamma = false }); Err ) return Err;
        if( auto Err = m_Device.Create(View.m_MaskDepth, { .m_Format = xgpu::texture::format::DEPTH_F32,     .m_Width = View.m_Width, .m_Height = View.m_Height, .m_isGamma = false }); Err ) return Err;
        {
            auto Attachments = std::array{ xgpu::renderpass::attachment{ View.m_Mask }, xgpu::renderpass::attachment{ View.m_MaskDepth } };
            if( auto Err = m_Device.Create(View.m_MaskPass, { .m_Attachments = Attachments, .m_ClearColorR = 0, .m_ClearColorG = 0, .m_ClearColorB = 0, .m_ClearColorA = 0 }); Err ) return Err;
        }
        {
            auto Attachments = std::array{ xgpu::renderpass::attachment{ View.m_Output } };
            if( auto Err = m_Device.Create(View.m_OutlinePass, { .m_Attachments = Attachments, .m_bClearOnRender = false }); Err ) return Err;
        }
        auto DepthBinding = std::array{ xgpu::pipeline_instance::sampler_binding{ View.m_Depth } };
        for( int k = 0; k < 3; ++k )
            if( auto Err = m_Device.Create(View.m_MaskInstance[k], { .m_PipeLine = m_MaskPipeline[k], .m_SamplersBindings = DepthBinding }); Err ) return Err;
        auto MaskBinding = std::array{ xgpu::pipeline_instance::sampler_binding{ View.m_Mask } };
        if( auto Err = m_Device.Create(View.m_OutlineInstance, { .m_PipeLine = m_OutlineCompositePipeline, .m_SamplersBindings = MaskBinding }); Err ) return Err;
        View.m_MaskWidth = View.m_Width; View.m_MaskHeight = View.m_Height;
        return nullptr;
    }

    void instance::destroyOutlineTargets( render_view& View ) noexcept
    {
        for( auto& I : View.m_MaskInstance ) if( I.m_Private ) m_Device.Destroy(std::move(I));
        if( View.m_OutlineInstance.m_Private ) m_Device.Destroy(std::move(View.m_OutlineInstance));
        if( View.m_MaskPass.m_Private )        m_Device.Destroy(std::move(View.m_MaskPass));
        if( View.m_OutlinePass.m_Private )     m_Device.Destroy(std::move(View.m_OutlinePass));
        if( View.m_Mask.m_Private )            m_Device.Destroy(std::move(View.m_Mask));
        if( View.m_MaskDepth.m_Private )       m_Device.Destroy(std::move(View.m_MaskDepth));
        View.m_MaskWidth = View.m_MaskHeight = 0;
    }

    void instance::DestroyView( render_view& View ) noexcept
    {
        destroyOutlineTargets(View);
        if( View.m_ToneMapInstance.m_Private ) m_Device.Destroy(std::move(View.m_ToneMapInstance));
        if( View.m_BlitInstance.m_Private )    m_Device.Destroy(std::move(View.m_BlitInstance));
        if( View.m_ScenePass.m_Private )       m_Device.Destroy(std::move(View.m_ScenePass));
        if( View.m_OutputPass.m_Private )      m_Device.Destroy(std::move(View.m_OutputPass));
        if( View.m_OverlayPass.m_Private )     m_Device.Destroy(std::move(View.m_OverlayPass));
        if( View.m_HDR.m_Private )             m_Device.Destroy(std::move(View.m_HDR));
        if( View.m_Depth.m_Private )           m_Device.Destroy(std::move(View.m_Depth));
        if( View.m_Output.m_Private )          m_Device.Destroy(std::move(View.m_Output));
        View.m_Width = View.m_Height = 0;
    }

    //------------------------------------------------------------------------------------------------

    void instance::Blit( xgpu::cmd_buffer& CmdBuffer, render_view& View ) noexcept
    {
        if( View.m_BlitInstance.m_Private == nullptr ) return;
        CmdBuffer.setPipelineInstance(View.m_BlitInstance);
        CmdBuffer.setBuffer(m_FullscreenVB);
        CmdBuffer.setBuffer(m_FullscreenIB);
        CmdBuffer.Draw(3);
    }

    //------------------------------------------------------------------------------------------------

    bool instance::getGPUTimes( gpu_times& Times ) noexcept
    {
        if( m_Timestamps.getResults(m_TimestampNs) == false ) return false;
        const auto& T = m_TimestampNs;
        Times.m_ShadowMs  = (T[1] - T[0]) * 1e-6;
        Times.m_SceneMs   = (T[2] - T[1]) * 1e-6;
        Times.m_ToneMapMs = (T[3] - T[2]) * 1e-6;
        return true;
    }
}
