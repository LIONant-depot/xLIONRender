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

        constexpr int max_draws_per_frame_v = 1024;     // ponytail: one material UBO entry per draw, ring of this size; grow with the scene graph work

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
        m_Device.Destroy(std::move(m_ForwardPipeline));
        m_Device.Destroy(std::move(m_ToneMapPipeline));
        m_Device.Destroy(std::move(m_BlitPipeline));
        m_Device.Destroy(std::move(m_WhiteTexture));
        ReleaseBuffer(m_LightUBO);
        ReleaseBuffer(m_MaterialUBO);
        ReleaseBuffer(m_FullscreenVB);
        ReleaseBuffer(m_FullscreenIB);
    }

    //------------------------------------------------------------------------------------------------

    xgpu::device::error* instance::Init( xgpu::device& Device ) noexcept
    {
        m_Device = Device;

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
        if( auto Err = Device.Create(m_MaterialUBO, { .m_Type = xgpu::buffer::type::UNIFORM, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::ubo_material), .m_EntryCount = max_draws_per_frame_v }); Err ) return Err;

        // White 1x1 texture bound where the shadow map will go
        {
            const std::uint32_t White = 0xffffffffu;
            const auto          Mips  = std::array{ xgpu::texture::setup::mip_size{ sizeof(White) } };
            xgpu::texture::setup Setup{ .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                      , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&White), sizeof(White) }, .m_isGamma = false };
            if( auto Err = Device.Create(m_WhiteTexture, Setup); Err ) return Err;
        }

        // Forward standard-material pipeline
        {
            xgpu::shader Vert, Frag;
            if( auto Err = Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_ForwardVert) }); Err ) return Err;
            if( auto Err = Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_ForwardStandardFrag) }); Err ) return Err;

            auto Shaders      = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            auto Samplers     = std::array{ xgpu::pipeline::sampler{} };
            auto UniformBinds = std::array
            { xgpu::pipeline::uniform_binds{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 2, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            };
            if( auto Err = Device.Create(m_ForwardPipeline, { .m_VertexDescriptor = m_MeshVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_forward)
                                                            , .m_UniformBinds = UniformBinds, .m_Samplers = Samplers }); Err ) return Err;

            auto Bindings = std::array{ xgpu::pipeline_instance::sampler_binding{ m_WhiteTexture } };
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
            if( auto Err = Device.Create(m_StaticGeomUBO, { .m_Type = xgpu::buffer::type::UNIFORM, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(details::ubo_static_geom_mesh), .m_EntryCount = max_draws_per_frame_v }); Err ) return Err;

            xgpu::shader Vert, Frag;
            if( auto Err = Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = SPIRV(s_StaticGeomVert) }); Err ) return Err;
            if( auto Err = Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = SPIRV(s_ForwardStandardFrag) }); Err ) return Err;

            auto Shaders      = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            auto Samplers     = std::array{ xgpu::pipeline::sampler{} };
            auto UniformBinds = std::array
            { xgpu::pipeline::uniform_binds{ .m_BindIndex = 0, .m_Usage = { .m_bVertex = true },   .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 2, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
            , xgpu::pipeline::uniform_binds{ .m_BindIndex = 0, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }
            };
            if( auto Err = Device.Create(m_StaticGeomPipeline, { .m_VertexDescriptor = m_StaticGeomVD, .m_Shaders = Shaders, .m_PushConstantsSize = sizeof(details::push_static_geom)
                                                               , .m_UniformBinds = UniformBinds, .m_Samplers = Samplers }); Err ) return Err;

            auto Bindings = std::array{ xgpu::pipeline_instance::sampler_binding{ m_WhiteTexture } };
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

        // GPU timings: 0 = scene start, 1 = scene end, 2 = tone map end
        if( auto Err = Device.Create(m_Timestamps, { .m_Count = 3 }); Err ) return Err;

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
        std::vector<xgpu::pipeline::sampler> Samplers(M.m_Textures.size());
        Samplers[0] = xgpu::pipeline::sampler{ .m_AddressMode = std::array{ xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP } };

        auto Shaders      = std::array<const xgpu::shader*, 2>{ M.m_pFragment, &Vert };
        auto UniformBinds = std::array
        { xgpu::pipeline::uniform_binds{ .m_BindIndex = 0, .m_Usage = { .m_bVertex = true },   .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
        , xgpu::pipeline::uniform_binds{ .m_BindIndex = 1, .m_Usage = { .m_bFragment = true }, .m_Type = xgpu::pipeline::uniform_binds::type::UBO_DYNAMIC }
        , xgpu::pipeline::uniform_binds{ .m_BindIndex = 0, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }
        , xgpu::pipeline::uniform_binds{ .m_BindIndex = 1, .m_Usage = xgpu::shader::type{ xgpu::shader::type::bit::VERTEX }, .m_Type = xgpu::pipeline::uniform_binds::type::SSBO_STATIC }   // SKIN: bones
        };
        const auto Binds = std::span{ UniformBinds.data(), bSkin ? std::size_t{4} : std::size_t{3} };
        if( m_Device.Create(M.m_Pipeline[k], { .m_VertexDescriptor = bSkin ? m_SkinGeomVD : m_StaticGeomVD, .m_Shaders = Shaders
                                             , .m_PushConstantsSize = bSkin ? sizeof(details::push_skin_geom) : sizeof(details::push_static_geom)
                                             , .m_UniformBinds = Binds, .m_Samplers = Samplers }) ) return nullptr;

        std::vector<xgpu::pipeline_instance::sampler_binding> Bindings;
        Bindings.reserve(M.m_Textures.size());
        for( auto* pTexture : M.m_Textures ) Bindings.emplace_back( pTexture ? *pTexture : m_WhiteTexture );
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
        Times.m_SceneMs   = (T[1] - T[0]) * 1e-6;
        Times.m_ToneMapMs = (T[2] - T[1]) * 1e-6;
        return true;
    }
}
