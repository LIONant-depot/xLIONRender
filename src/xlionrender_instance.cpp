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
        for( auto* p : { &m_EnvSkyInstance, &m_EnvPrefilterInstance, &m_EnvDFGInstance } ) m_Device.Destroy(std::move(*p));
        for( auto& I : m_EnvDownsampleInstance ) m_Device.Destroy(std::move(I));
        for( auto* p : { &m_EnvSkyPipeline, &m_EnvPrefilterPipeline, &m_EnvDFGPipeline, &m_EnvDownsamplePipeline } ) m_Device.Destroy(std::move(*p));
        for( auto* p : { &m_EnvAtlasPass, &m_EnvDFGPass } )                                 m_Device.Destroy(std::move(*p));
        for( auto& P : m_EnvSourcePass ) m_Device.Destroy(std::move(P));
        for( auto* p : { &m_EnvAtlas, &m_EnvDFG } )                                         m_Device.Destroy(std::move(*p));
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
        if( auto Err = Device.Create(m_ProbeAtlas,{ .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = 1536, .m_Height = 1024, .m_isGamma = false }); Err ) return Err;

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
            if( auto Err = Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_StaticGeomVert) }); Err ) return Err;
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
            };
            if( auto Err = Device.Create(m_StaticGeomPipeline, { .m_VertexDescriptor = m_StaticGeomVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                                               , .m_UniformBinds = UniformBinds, .m_Samplers = Samplers }); Err ) return Err;

            auto Bindings = LitBindings({});
            if( auto Err = Device.Create(m_StaticGeomInstance, { .m_PipeLine = m_StaticGeomPipeline, .m_SamplersBindings = Bindings }); Err ) return Err;
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
        }

        // The environment probe's passes and pipelines (fullscreen triangles; the sky and the prefilter rebuild only on a sky change)
        {
            xgpu::shader ClearVert2, Sky, Prefilter, DFG, Downsample;
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
        M.m_Geom = {};
        M.m_bAlive     = false;
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

    xgpu::device::error* instance::CreateMaterial( material_id& Id, const xgpu::shader& FragmentHDR, std::span<xgpu::texture* const> Textures ) noexcept
    {
        assert(m_bReady && !Textures.empty());

        material_slot M;
        M.m_pFragment = &FragmentHDR;
        M.m_Textures.assign(Textures.begin(), Textures.end());

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
        if( m_Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX, .m_Sharer = bSkin ? SPIRV(s_SkinGeomVert) : SPIRV(s_StaticGeomVert) }) ) return nullptr;

        // Slot 0 (shadow map) clamps, the material textures use the default sampler - as the geom editors
        assert(M.m_Textures.size() <= 8);       // slots 8, 9: the environment
        const auto Samplers = LitSamplers();

        auto Shaders      = std::array<const xgpu::shader*, 2>{ M.m_pFragment, &Vert };
        using ub = xgpu::pipeline::uniform_binds;
        const auto VertexSSBO   = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = ub::type::SSBO_STATIC }; };
        const auto FragmentSSBO = [](int i) { return ub{ .m_BindIndex = i, .m_Usage = { .m_bFragment = true }, .m_Type = ub::type::SSBO_STATIC }; };
        std::vector<ub> UniformBinds
        { ub{ .m_BindIndex = 0, .m_Usage = { .m_bVertex = true },   .m_Type = ub::type::UBO_DYNAMIC }
        , ub{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = ub::type::UBO_DYNAMIC }
        , VertexSSBO(0)                                         // clusters
        , VertexSSBO(1)                                         // RIGID: instances, SKIN: bones
        };
        if( bSkin ) UniformBinds.push_back(VertexSSBO(2));     // SKIN: instances
        UniformBinds.push_back(FragmentSSBO(3));                // lights
        UniformBinds.push_back(FragmentSSBO(4));                // light cells
        const auto Binds = std::span{ UniformBinds };
        if( m_Device.Create(M.m_Pipeline[k], { .m_VertexDescriptor = bSkin ? m_SkinGeomVD : m_StaticGeomVD, .m_Shaders = Shaders
                                             , .m_PushConstantsSize = bSkin ? sizeof(details::push_skin_geom) : sizeof(details::push_static_geom)
                                             , .m_UniformBinds = Binds, .m_Samplers = Samplers }) ) return nullptr;

        std::vector<xgpu::pipeline_instance::sampler_binding> Bindings;
        Bindings = LitBindings(M.m_Textures);
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

    void instance::DestroyView( render_view& View ) noexcept
    {
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
