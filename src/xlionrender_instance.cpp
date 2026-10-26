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
        constexpr std::uint32_t s_StaticGeomGIRecordsVert[] = {
            #include "xlionrender_static_geom_gi_records_vert.h"
        };
        constexpr std::uint32_t s_CullComp[] = {
            #include "xlionrender_cull_comp.h"
        };
        constexpr std::uint32_t s_HiZComp[] = {
            #include "xlionrender_hiz_comp.h"
        };
        constexpr std::uint32_t s_VSMRequestComp[] = {
            #include "xlionrender_vsm_request_comp.h"
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
        constexpr std::uint32_t s_GIScreenFrag[] = {
            #include "xlionrender_gi_screen_frag.h"
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
        for( auto& M : m_Meshes ) if( M.m_bAlive ) for( auto& S : M.m_GISections ) ReleaseBuffer(S.m_Buffer);
        ReleaseBuffer(m_GILightRing);
        ReleaseBuffer(m_GIMomentRing);
        if( m_GILightOther.m_Private ) ReleaseBuffer(m_GILightOther);
        if( m_GIMomentOther.m_Private ) ReleaseBuffer(m_GIMomentOther);
        ReleaseBuffer(m_GIDummy);
        ReleaseBuffer(m_WhiteColorStream);
        ReleaseBuffer(m_NoUV1Stream);
        for( auto& M : m_Materials ) if( M.m_bAlive ) releaseInstances(M);
        assert(m_SharedPipelines.empty());
        m_Device.Destroy(std::move(m_ForwardInstance));
        m_Device.Destroy(std::move(m_StaticGeomInstance));
        m_Device.Destroy(std::move(m_StaticGeomPipeline));
        m_Device.Destroy(std::move(m_DepthPrepassInstance));
        m_Device.Destroy(std::move(m_DepthPrepassPipeline));
        m_Device.Destroy(std::move(m_GIScreenInstance));
        m_Device.Destroy(std::move(m_GIScreenPipeline));
        m_Device.Destroy(std::move(m_GIScreenDepthInstance));
        m_Device.Destroy(std::move(m_GIScreenDepthPipeline));
        ReleaseBuffer(m_StaticGeomUBO);
        ReleaseBuffer(m_InstanceBuffer);
        ReleaseBuffer(m_InstanceRing);
        ReleaseBuffer(m_MaterialParams);
        ReleaseBuffer(m_MaterialParamsRing);
        ReleaseBuffer(m_EntityParams);
        ReleaseBuffer(m_EntityParamsRing);
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

    std::vector<xgpu::pipeline_instance::sampler_binding> instance::LitBindings( std::span<xgpu::texture* const> Textures, bool bShadowAtlas ) noexcept
    {
        std::vector<xgpu::pipeline_instance::sampler_binding> Bindings;
        Bindings.reserve(lit_samplers_v);
        Bindings.emplace_back(bShadowAtlas ? m_ShadowAtlas : m_WhiteTexture);
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
        if( auto Err = Device.Create(m_GILightRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = ring_frames_v * gi_light_region_v }); Err ) return Err;     // a uvec4 a surfel: the HL2 basis' 3 colors in RGB9E5, the validity
        if( auto Err = Device.Create(m_GIMomentRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = ring_frames_v * gi_light_region_v * gi::moment_texels_v }); Err ) return Err;
        {
            const std::uint32_t Zeros[8] = {};     // a GIMesh header: no surfels, no read grid
            if( auto Err = Device.Create(m_GIDummy, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(Zeros), .m_EntryCount = 1, .m_pData = const_cast<std::uint32_t*>(Zeros) }); Err ) return Err;
        }
        {
            std::uint32_t White = 0xFFFFFFFFu;
            if( auto Err = Device.Create(m_WhiteColorStream, { .m_Type = xgpu::buffer::type::VERTEX, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(White), .m_EntryCount = 1, .m_pData = &White }); Err ) return Err;
            float NoUV1[2] = { -3.0e38f, -3.0e38f };   // mb_input_definition_full.vert mb_UV1: below -1e38 reads UV0
            if( auto Err = Device.Create(m_NoUV1Stream, { .m_Type = xgpu::buffer::type::VERTEX, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(NoUV1), .m_EntryCount = 1, .m_pData = NoUV1 }); Err ) return Err;
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
            if( auto Err = Device.Create(m_ShadowAtlas,   { .m_Format = xgpu::texture::format::DEPTH_U16, .m_Width = 2 * Clipmap + Local, .m_Height = std::max(Clipmap, Local), .m_isGamma = false }); Err ) return Err;
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
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 0, .m_Format = xgpu::vertex_descriptor::format::UINT8_4D_NORMALIZED, .m_iStream = 2 }   // optional: COLOR0
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 0, .m_Format = xgpu::vertex_descriptor::format::FLOAT_2D, .m_iStream = 3 }              // optional: UV1
            };
            if( auto Err = Device.Create(m_StaticGeomVD, { .m_bUseStreaming = true, .m_Topology = xgpu::vertex_descriptor::topology::TRIANGLE_LIST, .m_VertexSize = 0, .m_Attributes = Attributes, .m_bDynamicStride = true }); Err ) return Err;

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
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 0,  .m_Format = xgpu::vertex_descriptor::format::UINT8_4D_NORMALIZED, .m_iStream = 2 }  // optional: COLOR0
            , xgpu::vertex_descriptor::attribute{ .m_Offset = 0,  .m_Format = xgpu::vertex_descriptor::format::FLOAT_2D, .m_iStream = 3 }             // optional: UV1
            };
            if( auto Err = Device.Create(m_SkinGeomVD, { .m_bUseStreaming = true, .m_Topology = xgpu::vertex_descriptor::topology::TRIANGLE_LIST, .m_VertexSize = 0, .m_Attributes = SkinAttributes, .m_bDynamicStride = true }); Err ) return Err;
            if( auto Err = Device.Create(m_StaticGeomUBO, { .m_Type = xgpu::buffer::type::UNIFORM, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::ubo_geom_view), .m_EntryCount = 64 }); Err ) return Err;
            {
                // Device-local (GPU_READ: created from zeros, then only written by GPU copies); the ring: host-visible, a region per frame
                std::vector<details::instance_data> Zeros(max_instances_v + max_transient_v, details::instance_data{});
                void* pZeros = Zeros.data();
                if( auto Err = Device.Create(m_InstanceBuffer, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(details::instance_data), .m_EntryCount = int(Zeros.size()), .m_pData = pZeros }); Err ) return Err;
                if( auto Err = Device.Create(m_InstanceRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::instance_data), .m_EntryCount = ring_frames_v * (max_instances_v + max_transient_v) }); Err ) return Err;
            }
            m_InstanceSlots.Init(max_instances_v);
            {
                std::vector<xmath::fvec4> Zeros(max_material_params_v, xmath::fvec4(0, 0, 0, 0));
                void* pZeros = Zeros.data();
                if( auto Err = Device.Create(m_MaterialParams, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(xmath::fvec4), .m_EntryCount = max_material_params_v, .m_pData = pZeros }); Err ) return Err;
                if( auto Err = Device.Create(m_MaterialParamsRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(xmath::fvec4), .m_EntryCount = ring_frames_v * param_ring_v }); Err ) return Err;
                std::vector<entity_params> EntityZeros(max_instances_v + max_transient_v);     // a 64 B entry per slot (every slot a shader can index), like the instance buffer
                void* pEntityZeros = EntityZeros.data();
                if( auto Err = Device.Create(m_EntityParams, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(entity_params), .m_EntryCount = int(EntityZeros.size()), .m_pData = pEntityZeros }); Err ) return Err;
                // ponytail: sized like the instance ring (3 x 5 MiB); shrink to the pending slots only if the memory matters
                if( auto Err = Device.Create(m_EntityParamsRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(entity_params), .m_EntryCount = ring_frames_v * int(EntityZeros.size()) }); Err ) return Err;
            }

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
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 9, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // MaterialParams (not read: the materials' layout)
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 10, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }  // EntityParams (not read)
            };
            if( auto Err = Device.Create(m_StaticGeomPipeline, { .m_VertexDescriptor = m_StaticGeomVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                                               , .m_UniformBinds = UniformBinds, .m_Samplers = Samplers }); Err ) return Err;

            auto Bindings = LitBindings({});
            if( auto Err = Device.Create(m_StaticGeomInstance, { .m_PipeLine = m_StaticGeomPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;

            // the depth prepass: the same vertex shader and layout, depth only (no color written)
            {
                xgpu::shader DepthOnly;
                if( auto Err = Device.Create(DepthOnly, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_DepthFrag) }); Err ) return Err;
                auto PrepassShaders = std::array<const xgpu::shader*, 2>{ &DepthOnly, &Vert };
                xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_StaticGeomVD, .m_Shaders = PrepassShaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                           , .m_UniformBinds = UniformBinds, .m_Samplers = Samplers };
                Setup.m_Blend.m_ColorWriteMask = 0;
                if( auto Err = Device.Create(m_DepthPrepassPipeline, Setup); Err ) return Err;
                if( auto Err = Device.Create(m_DepthPrepassInstance, { .m_PipeLine = m_DepthPrepassPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;
            }

            // the GI read at half resolution: the same layout + the buffer it writes; at the depth laid first, no color. Both of its phases
            // read each draw's push constants from the view's records (binding 12): a run of clusters is one indirect draw
            {
                xgpu::shader RecordsVert, DepthOnly;
                if( auto Err = Device.Create(RecordsVert, { .m_Type = xgpu::shader::type::bit::VERTEX, .m_Sharer = SPIRV(s_StaticGeomGIRecordsVert) }); Err ) return Err;
                if( auto Err = Device.Create(DepthOnly, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_DepthFrag) }); Err ) return Err;
                const xgpu::pipeline::uniform_binds RecordsBind{ .m_BindIndex = 12, .m_Usage = { .m_bVertex = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC };
                {
                    auto DepthShaders = std::array<const xgpu::shader*, 2>{ &DepthOnly, &RecordsVert };
                    std::vector<xgpu::pipeline::uniform_binds> DepthBinds(UniformBinds.begin(), UniformBinds.end());
                    DepthBinds.push_back(RecordsBind);
                    xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_StaticGeomVD, .m_Shaders = DepthShaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                               , .m_UniformBinds = DepthBinds, .m_Samplers = Samplers };
                    Setup.m_Blend.m_ColorWriteMask = 0;
                    if( auto Err = Device.Create(m_GIScreenDepthPipeline, Setup); Err ) return Err;
                    if( auto Err = Device.Create(m_GIScreenDepthInstance, { .m_PipeLine = m_GIScreenDepthPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;
                }
                xgpu::shader ScreenFrag;
                if( auto Err = Device.Create(ScreenFrag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_GIScreenFrag) }); Err ) return Err;
                auto ScreenShaders = std::array<const xgpu::shader*, 2>{ &ScreenFrag, &RecordsVert };
                std::vector<xgpu::pipeline::uniform_binds> ScreenBinds(UniformBinds.begin(), UniformBinds.end());
                ScreenBinds.push_back({ .m_BindIndex = 11, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC });
                ScreenBinds.push_back(RecordsBind);
                ScreenBinds.push_back({ .m_BindIndex = 13, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC });     // last frame's (temporal)
                xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_StaticGeomVD, .m_Shaders = ScreenShaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                           , .m_UniformBinds = ScreenBinds, .m_Samplers = Samplers };
                Setup.m_Blend.m_ColorWriteMask = 0;
                Setup.m_DepthStencil.m_bDepthWriteEnable = false;
                Setup.m_DepthStencil.m_DepthCompare      = xgpu::pipeline::depth_stencil::depth_compare::LESS_OR_EQUAL;
                if( auto Err = Device.Create(m_GIScreenPipeline, Setup); Err ) return Err;
                if( auto Err = Device.Create(m_GIScreenInstance, { .m_PipeLine = m_GIScreenPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;
            }

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
        if( auto Err = Device.Create(m_StaticGeomGIVert,    { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_StaticGeomGIVert) }); Err ) return Err;     // RIGID: reads the surfel GI
        if( auto Err = Device.Create(m_SkinGeomVert,        { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_SkinGeomVert) }); Err ) return Err;
        if( auto Err = Device.Create(m_GICaptureVert,       { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_GICaptureVert) }); Err ) return Err;
        if( auto Err = Device.Create(m_StaticGeomGIRecordsVert, { .m_Type = xgpu::shader::type::bit::VERTEX, .m_Sharer = SPIRV(s_StaticGeomGIRecordsVert) }); Err ) return Err;     // RIGID, GPU culled
        if( m_pCull = std::make_unique<gpu_cull>(); !m_pCull->Init(Device, s_CullComp, s_HiZComp) ) m_pCull.reset();      // (no drawIndirectCount: the CPU path)

        // The sun's virtual shadow pages (features/shadows/xlionrender_shadow_vsm.h): the pool is the clipmap's part of the atlas; the request
        // pass, its frame data (a ring slot each), its bits and counters (zeroed by a copy, copied back to a ring slot)
        m_VSM.Init({ .m_PoolTexels = sunAtlasWidth(), .m_PoolHeight = shadows::atlas_tiles_v * ShadowSetup.m_ClipmapTileSize });     // (2048 slots)
        if( const char* p = std::getenv("XLION_VSM_DEBUG") ) m_VSMDebugView = std::atoi(p);
        {
            using usage = xgpu::buffer::setup::usage;
            const int nMax = shadows::vsm::max_levels_v * m_VSM.getSetup().m_Window * m_VSM.getSetup().m_Window;
            const int nReq = (nMax + 31) / 32 + 32 + 2 * nMax;      // bits, counters, depth ranges - any level count renderShadows may grow to
            std::vector<std::uint32_t> Zeros(std::size_t(nReq), 0u);
            if( auto Err = Device.Create(m_VSMRequestShader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = SPIRV(s_VSMRequestComp) }); Err ) return Err;
            if( auto Err = Device.Create(m_VSMRequest, { .m_pShader = &m_VSMRequestShader, .m_nBuffers = 3, .m_nTextures = 1, .m_PushConstantsSize = 32 }); Err ) return Err;
            if( auto Err = Device.Create(m_VSMFrames,  { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::vsm_frame), .m_EntryCount = ring_frames_v }); Err ) return Err;
            if( auto Err = Device.Create(m_VSMReq,     { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = nReq, .m_pData = Zeros.data() }); Err ) return Err;
            if( auto Err = Device.Create(m_VSMZeros,   { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = nReq, .m_pData = Zeros.data() }); Err ) return Err;
            if( auto Err = Device.Create(m_VSMReqHost, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = ring_frames_v * nReq }); Err ) return Err;
            if( auto Err = Device.Create(m_VSMTimes,   { .m_Count = 2 }); Err ) return Err;
        }

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
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 8, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }     // SKIN: the BoneMap
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
            if( auto Err = MakeCaster(2, m_SkinGeomVD,   SkinVert,    sizeof(details::push_skin_geom),   5); Err ) return Err;     // SKIN

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
                if( auto Err = MakeMask(2, m_SkinGeomVD,   SkinVert,    sizeof(details::push_skin_geom),   5); Err ) return Err;
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
        for( auto& S : M.m_GISections ) ReleaseBuffer(S.m_Buffer);
        M.m_GISections.clear(); M.m_GIClusterSection.clear(); M.m_GIClusterVertexDelta.clear();
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
            for( const auto& E : Surfels.m_PackedReadTable ) Data.insert(Data.end(), E.begin(), E.end());       // the packed lists (surfel_mesh::PackReadGrid)
            Data[5] = std::uint32_t(Data.size());
            Data.insert(Data.end(), Surfels.m_PackedReadList.begin(), Surfels.m_PackedReadList.end());
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

    xgpu::device::error* instance::SetMeshGISections( mesh_id Id, std::vector<gi_section_in>&& Sections, std::vector<std::uint16_t>&& ClusterSection, std::vector<int>&& ClusterVertexDelta ) noexcept
    {
        if( findMesh(Id) == nullptr ) return nullptr;
        auto& M = m_Meshes[Id.m_Index];
        assert(M.m_bCluster && M.m_Geom.m_Kind == geom_kind::RIGID && ClusterSection.size() == M.m_Geom.m_Clusters.size() && ClusterVertexDelta.size() == ClusterSection.size());
        for( auto& S : M.m_GISections ) ReleaseBuffer(S.m_Buffer);
        M.m_GISections.clear();
        static std::uint64_t s_GIStamp = 1ull << 63;            // (apart from SetMeshGI's)
        M.m_GIStamp = ++s_GIStamp;
        for( auto& In : Sections )
        {
            assert(In.m_Surfels.m_Levels[0].size() < 0xFFFF);
            auto& S = M.m_GISections.emplace_back();
            if( In.m_Surfels.m_ReadTable.empty() ) In.m_Surfels.BuildReadGrid();
            S.m_pGI    = std::make_unique<gi::surfel_mesh>(std::move(In.m_Surfels));
            S.m_nOwned = In.m_nOwned;
            S.m_Ghosts = std::move(In.m_Ghosts);
            const auto Packed = In.m_Packed.empty() ? PackMeshGI(*S.m_pGI, {}, {}) : std::move(In.m_Packed);
            if( auto Err = m_Device.Create(S.m_Buffer, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = int(Packed.size()), .m_pData = const_cast<std::uint32_t*>(Packed.data()) }); Err ) return Err;
        }
        M.m_GIClusterSection     = std::move(ClusterSection);
        M.m_GIClusterVertexDelta = std::move(ClusterVertexDelta);
        return nullptr;
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

    xgpu::device::error* instance::CreateMaterial( material_id& Id, const xgpu::shader& FragmentHDR, std::span<xgpu::texture* const> Textures, const material_state& State, std::span<const float> Params, bool bTextureTable ) noexcept
    {
        assert(m_bReady && !Textures.empty() && Params.size() % 4 == 0);

        material_slot M;
        M.m_State = State;
        M.m_Fragment = FragmentHDR;
        M.m_Textures.clear();
        for( auto* p : Textures ) M.m_Textures.push_back(p ? *p : xgpu::texture{});

        // A texture-table material: its textures (all but slot 0, the shadow atlas) into its table, their ids after its parameter block
        std::vector<float> Block(Params.begin(), Params.end());
        if( bTextureTable )
        {
            std::vector<xgpu::texture_table::slot> Slots;
            for( std::size_t i = 1; i < M.m_Textures.size(); ++i )
            {
                const auto& T = M.m_Textures[i];
                Slots.push_back(T.m_Private ? xgpu::texture_table::slot{ T.getID(), T.getSampler() } : xgpu::texture_table::slot{});
            }
            if( auto Err = m_Device.Create(M.m_Table, { .m_Slots = Slots }); Err ) return Err;
            M.m_bTable = true;
            M.m_IDsAt  = std::uint32_t(Block.size() / 4);
            Block.resize(Block.size() + 4 * ((Slots.size() + 1) / 2), 0.0f);
            for( std::size_t i = 0; i < Slots.size(); ++i )
            {
                Block[4 * M.m_IDsAt + 2 * i + 0] = std::bit_cast<float>(Slots[i].m_Texture.m_Value);
                Block[4 * M.m_IDsAt + 2 * i + 1] = std::bit_cast<float>(Slots[i].m_Sampler.m_Value);
            }
        }

        // The rigid pipeline now, so a material that cannot build reports it here (the skin one is built on its first skinned draw)
        if( getMaterialInstance(M, geom_kind::RIGID) == nullptr ) return VGPU_ERROR(xgpu::device::error::FAILURE, "The material's pipeline could not be created");

        // Its parameter block: a range of MaterialParams (first fit, else the top), copied over by the next updateInstances
        Params = Block;
        if( const int n = int(Params.size() / 4); n > 0 )
        {
            M.m_ParamRange = m_ParamAlloc.allocate(std::uint32_t(n));
            const int At = M.m_ParamRange.offset == OffsetAllocator::Allocation::NO_SPACE ? -1 : int(M.m_ParamRange.offset);
            if( At < 0 )
            {
                releaseInstances(M);
                return VGPU_ERROR(xgpu::device::error::FAILURE, "No room left in MaterialParams for the material's parameters");
            }
            M.m_ParamBase  = std::uint32_t(At);
            M.m_ParamCount = std::uint32_t(n);
            m_ParamPending.emplace_back(At, std::vector<float>(Params.begin(), Params.end()));
        }

        const auto [Index, Gen] = AddSlot(m_Materials, std::move(M));
        Id = { Index, Gen };
        return nullptr;
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::SetMaterialParams( material_id Id, std::span<const float> Params, std::uint32_t FirstVec4 ) noexcept
    {
        auto* pM = findMaterial(Id);
        const std::uint32_t nBlock = pM == nullptr ? 0 : pM->m_bTable ? pM->m_IDsAt : pM->m_ParamCount;     // (not a table's ids)
        if( pM == nullptr || Params.size() % 4 || FirstVec4 + Params.size() / 4 > nBlock ) return VGPU_ERROR(xgpu::device::error::FAILURE, "Not a range of the material's parameter block");
        if( !Params.empty() ) m_ParamPending.emplace_back(int(pM->m_ParamBase + FirstVec4), std::vector<float>(Params.begin(), Params.end()));
        return nullptr;
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::SetMaterialTexture( material_id Id, int iSlot, xgpu::texture& Texture ) noexcept
    {
        auto* pM = findMaterial(Id);
        if( pM == nullptr || !pM->m_bTable || iSlot < 0 || iSlot >= pM->m_Table.getSlotCount() ) return VGPU_ERROR(xgpu::device::error::FAILURE, "Not a texture slot of a texture-table material");
        const xgpu::texture_table::slot Slot{ Texture.getID(), Texture.getSampler() };
        if( pM->m_Table.Update(iSlot, Slot) ) return VGPU_ERROR(xgpu::device::error::FAILURE, "The material's texture table could not be updated");
        m_Device.Destroy(std::move(pM->m_Textures[iSlot + 1]));      // (a frame in flight may still read it)
        pM->m_Textures[iSlot + 1] = Texture;

        // The ids in the block: one vec4 through the ring (written whole: its other slot as it is)
        const std::uint32_t At = pM->m_IDsAt + std::uint32_t(iSlot / 2);
        std::vector<float> V(4, 0.0f);
        for( int j = 0; j < 2; ++j )
            if( const int s = (iSlot & ~1) + j; s < pM->m_Table.getSlotCount() )
            {
                const auto S = pM->m_Table.getSlot(s);
                V[2 * j + 0] = std::bit_cast<float>(S.m_Texture.m_Value);
                V[2 * j + 1] = std::bit_cast<float>(S.m_Sampler.m_Value);
            }
        m_ParamPending.emplace_back(int(pM->m_ParamBase + At), std::move(V));
        return nullptr;
    }

    //------------------------------------------------------------------------------------------------

    // The index buffer, the core streams, then the optional ones: a stream the geometry lacks reads a constant default (stride 0)
    void instance::BindClusterStreams( xgpu::cmd_buffer& Cmd, const cluster_geom& G ) noexcept
    {
        const auto Streams = std::array{ &G.m_pStreams[0], &G.m_pStreams[1], &G.m_pStreams[2], G.m_pColors ? G.m_pColors : &m_WhiteColorStream, G.m_pUV1 ? G.m_pUV1 : &m_NoUV1Stream };
        Cmd.setStreamingBuffers(std::span<xgpu::buffer* const>{ Streams }, (G.m_pColors ? 0u : 1u << 2) | (G.m_pUV1 ? 0u : 1u << 3));    // stream 2: COLOR0, 3: UV1
    }

    //------------------------------------------------------------------------------------------------

    // A material's settings in its pipeline (material_state): culling, raster, blend, depth
    void instance::ApplyState( xgpu::pipeline::setup& Setup, const material_state& S ) noexcept
    {
        using blend = xgpu::pipeline::blend;
        if( S.m_bDoubleSided ) Setup.m_Primitive.m_Cull   = xgpu::pipeline::primitive::cull::NONE;
        if( S.m_bWireframe )   Setup.m_Primitive.m_Raster = xgpu::pipeline::primitive::raster::WIRELINE;
        Setup.m_DepthStencil.m_bDepthTestEnable  = S.m_bDepthTest;
        Setup.m_DepthStencil.m_bDepthWriteEnable = S.m_bDepthWrite;
        // The color blends; the alpha (what was drawn: the tone map's coverage) only grows: "over" for translucent, kept for additive
        if( S.m_Blend == material_state::blend::ALPHA ) { Setup.m_Blend = blend::getAlphaOriginal(); Setup.m_Blend.m_AlphaDstFactor = blend::factor::ONE_MINUS_SRC_ALPHA; }
        if( S.m_Blend == material_state::blend::ADD )   { Setup.m_Blend = blend::getAdd(); Setup.m_Blend.m_AlphaSrcFactor = blend::factor::ZERO; Setup.m_Blend.m_AlphaDstFactor = blend::factor::ONE; }
    }

    //------------------------------------------------------------------------------------------------

    xgpu::pipeline_instance* instance::getMaterialInstance( material_slot& M, geom_kind Kind, bool bShadow ) noexcept
    {
        const auto k = (bShadow ? shadow_path_v : 0) + static_cast<std::size_t>(Kind);
        if( M.m_Instance[k].m_Private ) return &M.m_Instance[k];
        // a shadow caster renders into the atlas
        return makeInstance(M, k, !bShadow, [&](xgpu::pipeline& Pipeline) { return buildMaterialPipeline(Pipeline, M, Kind, bShadow); });
    }

    xgpu::pipeline_instance* instance::getRecordsInstance( material_slot& M ) noexcept
    {
        if( M.m_Instance[records_path_v].m_Private ) return &M.m_Instance[records_path_v];
        return makeInstance(M, records_path_v, true, [&](xgpu::pipeline& Pipeline) { return buildMaterialPipeline(Pipeline, M, geom_kind::RIGID, false, true); });
    }

    xgpu::device::error* instance::buildMaterialPipeline( xgpu::pipeline& Pipeline, const material_slot& M, geom_kind Kind, bool bShadow, bool bRecords ) noexcept
    {
        const bool bSkin = Kind == geom_kind::SKIN;

        // Slot 0 (shadow map) clamps, the material textures use the default sampler - as the geom editors (a texture-table material: no
        // material texture in set 0)
        assert(M.m_bTable || M.m_Textures.size() <= 8);       // slots 8, 9: the environment
        const auto Samplers = LitSamplers();

        auto Shaders      = std::array<const xgpu::shader*, 2>{ &M.m_Fragment, bSkin ? &m_SkinGeomVert : bRecords ? &m_StaticGeomGIRecordsVert : &m_StaticGeomGIVert };
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
        if( bSkin ) UniformBinds.push_back(VertexSSBO(8));     // SKIN: the BoneMap (xgeom_skin palettes)
        UniformBinds.push_back(FragmentSSBO(9));                // MaterialParams (every material declares it, read or not)
        UniformBinds.push_back(FragmentSSBO(10));               // EntityParams
        if( bRecords ) UniformBinds.push_back(VertexSSBO(12)); // the draws' records (GPU culling)
        const auto Binds = std::span{ UniformBinds };
        xgpu::pipeline::setup Setup{ .m_VertexDescriptor = bSkin ? m_SkinGeomVD : m_StaticGeomVD, .m_Shaders = Shaders
                                   , .m_PushConstantsSize = bSkin ? sizeof(details::push_skin_geom) : sizeof(details::push_static_geom)
                                   , .m_UniformBinds = Binds, .m_Samplers = Samplers, .m_nTextureSlots = TableSlots(M) };
        if( bShadow )       // the casters' state (instance::Init MakeCaster); the fragment cuts out and stops (ubo_lighting m_LightParams.w)
        {
            Setup.m_Primitive.m_Cull                        = xgpu::pipeline::primitive::cull::NONE;
            Setup.m_DepthStencil.m_DepthBiasConstantFactor  = 2.0f;
            Setup.m_DepthStencil.m_DepthBiasSlopeFactor     = 4.0f;
            Setup.m_DepthStencil.m_bDepthBiasEnable         = true;
            Setup.m_DepthStencil.m_bDepthClampEnable        = true;
        }
        else ApplyState(Setup, M.m_State);
        return m_Device.Create(Pipeline, Setup);
    }

    //------------------------------------------------------------------------------------------------

    // What a path's pipeline depends on: the fragment shader, the state the vertex paths apply (ApplyState; the GI capture and the shadow
    // casters use none), the path. The fragment's address is safe: every material of an entry holds a copy of the shader
    instance::pipeline_key instance::PipelineKey( const material_slot& M, std::size_t k ) noexcept
    {
        std::uint32_t State = 0;
        if( k < capture_path_v || k == records_path_v )
        {
            const auto& S = M.m_State;
            State = (S.isBlended() ? 1u + std::uint32_t(S.m_Blend) : 0u) | (S.m_bDoubleSided ? 1u << 4 : 0u) | (S.m_bWireframe ? 1u << 5 : 0u)
                  | (S.m_bDepthTest ? 1u << 6 : 0u) | (S.m_bDepthWrite ? 1u << 7 : 0u);
        }
        return { M.m_Fragment.m_Private.get(), State, k };
    }

    xgpu::pipeline_instance* instance::makeInstance( material_slot& M, std::size_t k, bool bShadowAtlas, const std::function<xgpu::device::error*(xgpu::pipeline&)>& Build ) noexcept
    {
        const auto Key = PipelineKey(M, k);
        auto& P = m_SharedPipelines[Key];
        if( P.m_Pipeline.m_Private == nullptr && Build(P.m_Pipeline) ) { m_SharedPipelines.erase(Key); return nullptr; }

        // A texture-table material's set 0 holds the renderer's textures only: the same for every material of the pipeline (its copy)
        if( M.m_bTable && P.m_Globals.m_Private ) M.m_Instance[k] = P.m_Globals;
        else
        {
            auto Bindings = LitBindings(M.m_bTable ? std::vector<xgpu::texture*>{} : M.TexturePointers(), bShadowAtlas);
            if( m_Device.Create(M.m_Instance[k], { .m_PipeLine = P.m_Pipeline, .m_SamplersBindings = Bindings }) )
            {
                if( P.m_nUsers == 0 ) { m_Device.Destroy(std::move(P.m_Pipeline)); m_SharedPipelines.erase(Key); }
                return nullptr;
            }
            if( M.m_bTable ) P.m_Globals = M.m_Instance[k];
        }
        ++P.m_nUsers;
        return &M.m_Instance[k];
    }

    // Set 3's size: a texture-table material's slots (bindless: the heaps, any count), 0 otherwise. The pipeline key needs no part of it: the
    // fragment shader decides it
    std::uint8_t instance::TableSlots( const material_slot& M ) const noexcept
    {
        if( !M.m_bTable ) return 0;
        return std::uint8_t(std::max(1, M.m_Table.getSlotCount()));
    }

    // Its instances, and the pipelines no other material uses
    void instance::releaseInstances( material_slot& M ) noexcept
    {
        for( std::size_t k = 0; k < M.m_Instance.size(); ++k )
        {
            if( M.m_Instance[k].m_Private == nullptr ) continue;
            m_Device.Destroy(std::move(M.m_Instance[k]));
            M.m_Instance[k].m_Private.reset();      // (no move: Destroy queued a copy)
            const auto It = m_SharedPipelines.find(PipelineKey(M, k));
            assert(It != m_SharedPipelines.end());
            if( --It->second.m_nUsers == 0 )
            {
                if( It->second.m_Globals.m_Private ) m_Device.Destroy(std::move(It->second.m_Globals));
                m_Device.Destroy(std::move(It->second.m_Pipeline));
                m_SharedPipelines.erase(It);
            }
        }
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
        return makeInstance(M, k, true, [&](xgpu::pipeline& Pipeline) -> xgpu::device::error*
        {
            const auto Samplers = LitSamplers();
            auto Shaders = std::array<const xgpu::shader*, 2>{ &M.m_Fragment, &m_GICaptureVert };
            using ub = xgpu::pipeline::uniform_binds;
            const auto VertexSSBO   = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = ub::type::SSBO_STATIC }; };
            const auto FragmentSSBO = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = { .m_bFragment = true }, .m_Type = ub::type::SSBO_STATIC }; };
            const auto BothSSBO     = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = { .m_bVertex = true, .m_bFragment = true }, .m_Type = ub::type::SSBO_STATIC }; };
            auto Binds = std::array
            { ub{ .m_BindIndex = 0, .m_Usage = { .m_bVertex = true },   .m_Type = ub::type::UBO_DYNAMIC }
            , ub{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = ub::type::UBO_DYNAMIC }
            , VertexSSBO(0), VertexSSBO(1), FragmentSSBO(3), FragmentSSBO(4), BothSSBO(5), BothSSBO(6), BothSSBO(7), FragmentSSBO(9), FragmentSSBO(10)
            };
            xgpu::pipeline::setup Setup{ .m_VertexDescriptor = m_FullscreenVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                       , .m_UniformBinds = std::span{ Binds }, .m_Samplers = Samplers, .m_nTextureSlots = TableSlots(M) };
            Setup.m_Primitive.m_Cull = xgpu::pipeline::primitive::cull::NONE;
            return m_Device.Create(Pipeline, Setup);
        });
    }

    //------------------------------------------------------------------------------------------------

    void instance::DestroyMaterial( material_id Id ) noexcept
    {
        auto* pM = findMaterial(Id);
        if( pM == nullptr ) return;
        releaseInstances(*pM);
        if( pM->m_ParamCount )
        {
            std::erase_if(m_ParamPending, [&](const auto& P) { return P.first >= int(pM->m_ParamBase) && P.first < int(pM->m_ParamBase + pM->m_ParamCount); });     // created (or a texture set) and destroyed before a render
            m_ParamAlloc.free(pM->m_ParamRange);
            pM->m_ParamRange = {};
            pM->m_ParamCount = 0;
        }
        pM->m_Textures.clear();
        pM->m_Fragment = {};
        if( pM->m_bTable ) { m_Device.Destroy(std::move(pM->m_Table)); pM->m_Table.m_Private.reset(); pM->m_bTable = false; }
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
        if( m_pGIWorld->m_Setup.m_bGPUOwned )          // the light is the GPU's: the file is written when it is read back (a frame or two later)
        {
            m_GIDumpPath = pPath; m_bGIDumpAsked = false;
            int n = 0;
            for( int i = 0; i < m_pGIWorld->InstanceCount(); ++i ) n += int(m_pGIWorld->InstanceLight(i).size());
            return n;
        }
        const int n = m_pGIWorld->Dump(pPath);
        if( n >= 0 ) (void)DumpGIFrames((std::string(pPath) + ".frames.csv").c_str());
        return n;
    }

    std::string instance::DescribeGIRead( const xmath::fvec3& O, const xmath::fvec3& D ) const noexcept
    {
        if( !m_pGIWorld ) return "GIReadAt: no GI";
        const auto& W = *m_pGIWorld;
        const auto  H = W.Cast(gi::v3{ O.m_X, O.m_Y, O.m_Z }, gi::v3{ D.m_X, D.m_Y, D.m_Z });
        if( H.m_Inst < 0 || std::size_t(H.m_Inst) >= m_GITrack.size() || !m_GITrack[std::size_t(H.m_Inst)].m_pMesh ) return "GIReadAt: no hit";
        const auto& M = *m_GITrack[std::size_t(H.m_Inst)].m_pMesh;
        const auto& T = M.m_Tris[std::size_t(H.m_Tri)];
        const float B[3] = { 1 - H.m_U - H.m_V, H.m_U, H.m_V };
        const gi::v3 P = M.m_Pos[T[0]] * B[0] + M.m_Pos[T[1]] * B[1] + M.m_Pos[T[2]] * B[2];
        gi::v3 N = (M.m_Pos[T[1]] - M.m_Pos[T[0]]).Cross(M.m_Pos[T[2]] - M.m_Pos[T[0]]).NormalizeCopy();
        if( N.Dot(gi::v3{ D.m_X, D.m_Y, D.m_Z }) > 0 ) N = N * -1.0f;
        const auto& E  = W.InstanceLight(H.m_Inst);
        const auto& Va = W.InstanceValid(H.m_Inst);
        const auto  Lum = [&](std::size_t s) { return s < E.size() ? 0.2126f * E[s].m_X + 0.7152f * E[s].m_Y + 0.0722f * E[s].m_Z : -1.0f; };
        std::string S = std::format("GIReadAt: instance {} (surfels {}, owned {}), triangle {} (material slot {}), at {:.3f},{:.3f},{:.3f} facing {:.2f},{:.2f},{:.2f}, {:.2f} away\n",
                                    H.m_Inst, E.size(), W.OwnedSurfels(H.m_Inst), H.m_Tri, std::size_t(H.m_Tri) < M.m_TriMaterial.size() ? int(M.m_TriMaterial[std::size_t(H.m_Tri)]) : -1, P.m_X, P.m_Y, P.m_Z, N.m_X, N.m_Y, N.m_Z, H.m_T);
        gi::v3 PatchN = N;
        const auto List = M.ReadCellList(P, N, &PatchN);
        S += std::format("cell list: {} entries (patch facing {:.2f},{:.2f},{:.2f}); per entry: surfel, owned, its distance, its plane offset, its facing, corners (0-15), its light, validity\n", List.size(), PatchN.m_X, PatchN.m_Y, PatchN.m_Z);
        for( const auto e : List )
        {
            const std::size_t s = e & 0xFFFFu;
            const auto& Sf = M.m_Levels[0][s];
            S += std::format("  {:5} {} {:6.3f} {:+6.3f} {:5.2f}  {:2} {:2} {:2} {:2}  {:8.4f} {:4.2f}\n", s, s < W.OwnedSurfels(H.m_Inst) ? "o" : "g", (Sf.m_Center - P).Length(), (Sf.m_Center - P).Dot(N), Sf.m_Normal.Dot(N),
                             (e >> 16) & 15u, (e >> 20) & 15u, (e >> 24) & 15u, (e >> 28) & 15u, Lum(s), s < Va.size() ? Va[s] : -1.0f);
        }
        S += "vertex surfels:";
        for( int k = 0; k < 3; ++k ) for( const int s : M.m_VertSurfels[std::size_t(T[std::size_t(k)])] )
            if( s >= 0 ) S += std::format(" {}({:.4f})", s, Lum(std::size_t(s)));
        return S;
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
                       "Updated,Surfels,Rays,Hits,LongestSurfelUs,LinkRebuilds,Links,Emitters,Moves,MovesMs,MarkedDirty,HistoryResets,Dirty,UploadBytes,Dropped,LinkBytes,LightBytes,FilterBytes,Unsupported,Bad,Cubes,CubeParents,Pending,PendingRays,ChunkRays,RayUsEst,Rounds,CastMs,DirectMs,ReadMs,FairSurfels,LinkPending,LinkSurfels,UrgentSurfels,PreMs,OverAllowMs,TraceCpuMs,LinkCpuMs,LinkLongestUs,TokenUnits,GPUTraceMs,GPUPacks,GPUCpuMs,GPUSliceRays,TimeMs,MeanLum,SceneGpuMs\n");
        for( const auto& F : m_GIFrames )
        {
            const auto& S = F.m_Stats;
            std::fprintf(f, "%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%lld,%.1f,%d,%d,%d,%d,%.3f,%d,%d,%d,%zu,%d,%zu,%zu,%zu,%d,%d,%d,%d,%d,%d,%d,%.1f,%d,%.3f,%.3f,%.3f,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%d,%.3f,%d,%.3f,%d,%.1f,%.6g,%.3f\n"
                , (unsigned long long)F.m_Frame, S.m_UpdateGIMs, S.m_TotalMs, S.m_TlasLightsMs, S.m_LevelsMs, S.m_PickMs, S.m_LinksMs, S.m_EmitMs, S.m_MeanMs, S.m_TraceMs, S.m_CommitMs, S.m_CubeMs, S.m_StatsMs
                , S.m_FilterMs, S.m_RingCopyMs, S.m_MomentsMs, S.m_CaptureMs, S.m_LoadMs
                , S.m_Updated, S.m_Surfels, S.m_Rays, S.m_Hits, S.m_LongestSurfelUs, S.m_nLinkRebuilds, S.m_nLinks, S.m_nEmitters, S.m_nMoves, S.m_MovesMs, S.m_nMarkedDirty, S.m_nHistoryResets, S.m_Dirty
                , S.m_UploadBytes, S.m_nDropped, S.m_LinkBytes, S.m_LightBytes, S.m_FilterBytes, S.m_nUnsupported, S.m_nBad, S.m_nCubes, S.m_CubeParents, S.m_Pending, S.m_PendingRays, S.m_ChunkRays, double(S.m_RayUsEst), S.m_Rounds, S.m_CastMs, S.m_DirectMs, S.m_ReadMs, S.m_FairSurfels, S.m_LinkPending, S.m_LinkSurfels, S.m_UrgentSurfels, S.m_PreMs, S.m_OverAllowMs, S.m_TraceCpuMs, S.m_LinkCpuMs, S.m_LinkLongestUs, S.m_TokenUnits, S.m_GPUTraceMs, S.m_GPUPacks, S.m_GPUCpuMs, S.m_GPUSliceRays, S.m_TimeMs, double(S.m_MeanLum), F.m_SceneGpuMs);
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

    xgpu::device::error* instance::createCullTargets( render_view& View ) noexcept
    {
        if( View.m_HiZ.m_Private ) return nullptr;
        auto Attachments = std::array{ xgpu::renderpass::attachment{ View.m_HDR }, xgpu::renderpass::attachment{ View.m_Depth } };
        if( auto Err = m_Device.Create(View.m_ScenePassLoad, { .m_Attachments = Attachments, .m_bClearOnRender = false }); Err ) return Err;
        const std::vector<float> Zeros(std::size_t(gpu_cull::HiZFloats(View.m_Width, View.m_Height)), 0.0f);     // (a device-local buffer is made from its first data)
        return m_Device.Create(View.m_HiZ, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(float)
                                           , .m_EntryCount = int(Zeros.size()), .m_pData = const_cast<float*>(Zeros.data()) });
    }

    xgpu::device::error* instance::createGIScreenTargets( render_view& View ) noexcept
    {
        const int W = (View.m_Width + 1) / 2, H = (View.m_Height + 1) / 2;
        if( View.m_GIScreenPass.m_Private && View.m_GIScreenWidth == W && View.m_GIScreenHeight == H ) return nullptr;
        destroyGIScreenTargets(View);
        if( auto Err = m_Device.Create(View.m_GIScreenColor, { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = W, .m_Height = H, .m_isGamma = false }); Err ) return Err;
        if( auto Err = m_Device.Create(View.m_GIScreenDepth, { .m_Format = xgpu::texture::format::DEPTH_F32, .m_Width = W, .m_Height = H, .m_isGamma = false }); Err ) return Err;
        {
            auto Attachments = std::array{ xgpu::renderpass::attachment{ View.m_GIScreenColor }, xgpu::renderpass::attachment{ View.m_GIScreenDepth } };
            if( auto Err = m_Device.Create(View.m_GIScreenPass, { .m_Attachments = Attachments, .m_ClearColorR = 0, .m_ClearColorG = 0, .m_ClearColorB = 0, .m_ClearColorA = 0 }); Err ) return Err;
        }
        std::vector<std::array<std::uint32_t, 4>> Data(screen_header_v + std::size_t(W) * std::size_t(H), std::array<std::uint32_t, 4>{});
        if( auto Err = m_Device.Create(View.m_GIScreenZeros, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = int(Data.size()), .m_pData = Data.data() }); Err ) return Err;
        Data[0] = { std::uint32_t(W), std::uint32_t(H), 0u, 0u };
        if( auto Err = m_Device.Create(View.m_GIScreenBuffer, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = int(Data.size()), .m_pData = Data.data() }); Err ) return Err;
        if( auto Err = m_Device.Create(View.m_GIScreenPrev, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = int(Data.size()), .m_pData = Data.data() }); Err ) return Err;
        if( auto Err = m_Device.Create(View.m_GIScreenHeader, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = ring_frames_v * screen_header_v }); Err ) return Err;
        if( auto Err = m_Device.Create(View.m_GIScreenCounters, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = ring_frames_v }); Err ) return Err;
        View.m_bGIScreenPrev = false;
        if( auto Err = m_Device.Create(View.m_GIScreenRecords, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::push_static_geom), .m_EntryCount = ring_frames_v * screen_draws_v }); Err ) return Err;
        if( auto Err = m_Device.Create(View.m_GIScreenCommands, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::draw_indexed_indirect), .m_EntryCount = ring_frames_v * screen_draws_v }); Err ) return Err;
        View.m_GIScreenWidth = W; View.m_GIScreenHeight = H;
        return nullptr;
    }

    void instance::destroyGIScreenTargets( render_view& View ) noexcept
    {
        if( View.m_GIScreenPass.m_Private )   m_Device.Destroy(std::move(View.m_GIScreenPass));
        if( View.m_GIScreenColor.m_Private )  m_Device.Destroy(std::move(View.m_GIScreenColor));
        if( View.m_GIScreenDepth.m_Private )  m_Device.Destroy(std::move(View.m_GIScreenDepth));
        if( View.m_GIScreenBuffer.m_Private ) m_Device.Destroy(std::move(View.m_GIScreenBuffer));
        if( View.m_GIScreenZeros.m_Private )  m_Device.Destroy(std::move(View.m_GIScreenZeros));
        if( View.m_GIScreenPrev.m_Private )   m_Device.Destroy(std::move(View.m_GIScreenPrev));
        if( View.m_GIScreenHeader.m_Private ) m_Device.Destroy(std::move(View.m_GIScreenHeader));
        if( View.m_GIScreenCounters.m_Private ) m_Device.Destroy(std::move(View.m_GIScreenCounters));
        View.m_bGIScreenPrev = false;
        if( View.m_GIScreenRecords.m_Private )  m_Device.Destroy(std::move(View.m_GIScreenRecords));
        if( View.m_GIScreenCommands.m_Private ) m_Device.Destroy(std::move(View.m_GIScreenCommands));
        View.m_GIScreenWidth = View.m_GIScreenHeight = 0;
    }

    void instance::DestroyView( render_view& View ) noexcept
    {
        destroyOutlineTargets(View);
        destroyGIScreenTargets(View);
        if( View.m_ToneMapInstance.m_Private ) m_Device.Destroy(std::move(View.m_ToneMapInstance));
        if( View.m_BlitInstance.m_Private )    m_Device.Destroy(std::move(View.m_BlitInstance));
        if( View.m_ScenePass.m_Private )       m_Device.Destroy(std::move(View.m_ScenePass));
        if( View.m_ScenePassLoad.m_Private )   m_Device.Destroy(std::move(View.m_ScenePassLoad));
        if( View.m_HiZ.m_Private )             ReleaseBuffer(View.m_HiZ);
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
        ++m_GPUTimesSerial;
        const auto& T = m_TimestampNs;
        Times.m_ShadowMs  = (T[1] - T[0]) * 1e-6;
        Times.m_SceneMs   = (T[2] - T[1]) * 1e-6;
        Times.m_ToneMapMs = (T[3] - T[2]) * 1e-6;
        return true;
    }
}
