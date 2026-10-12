// Plan A - forward. instance::Render lives here so each plan is one file to read.
#include "xlionrender_instance.h"
#include "plans/forward/xlionrender_plan_forward.h"
#include <chrono>

namespace xlionrender
{
    namespace
    {
        // FNV-1a over a caster's identity and state: a different value is a different (or moved) caster
        std::uint64_t Hash(std::uint64_t H, const void* pData, std::size_t Size) noexcept
        {
            for( auto* p = static_cast<const std::uint8_t*>(pData); Size--; ++p ) H = (H ^ *p) * 1099511628211ull;
            return H;
        }
    }

    // Every draw's slot in the instance buffer this frame: a draw with an instance id keeps its persistent slot (written only
    // when its transform changes or the origin moves), one without takes a transient slot written now. What is written goes
    // into this frame's region of the CPU ring, and the GPU copies it into the device-local instance buffer before the passes
    // that read it (features/instances/xlionrender_instance_slots.h).
    void instance::updateInstances( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye ) noexcept
    {
        const std::uint64_t Frame = m_Device.getFrameIndex();
        if( Frame != m_InstanceFrame )
        {
            m_InstanceFrame  = Frame;
            m_TransientCount = 0;
            m_RingUsed       = 0;
        }

        // The origin (camera-relative "little world" for the stored rows): re-snapped when the eye moves a grid cell away
        const auto Far = [&](int i) { return std::abs(Eye[i] - m_Origin[i]) > origin_grid_v; };
        if( !m_bOriginSet || Far(0) || Far(1) || Far(2) )
        {
            for( int i = 0; i < 3; ++i ) m_Origin[i] = std::round(Eye[i] / origin_grid_v) * origin_grid_v;
            m_bOriginSet = true;
            m_InstanceSlots.MarkAllPending();
        }

        // This frame's ring region: written in order, copied by contiguous runs
        constexpr int    region_v = max_instances_v + max_transient_v;
        const int        Base     = int(Frame % ring_frames_v) * region_v;
        std::vector<xgpu::buffer::copy> Regions;
        (void)m_InstanceRing.MemoryMap(Base, region_v, [&](void* pMemory)
        {
            auto* pRing = static_cast<details::instance_data*>(pMemory);
            const auto Put = [&](std::uint32_t Slot, const xmath::fmat4& L2W, const material& Mat)
            {
                pRing[m_RingUsed].set(L2W, m_Origin, Mat);
                const int Src = Base + m_RingUsed++;
                if( !Regions.empty() && Regions.back().m_SrcEntry + Regions.back().m_Count == Src && Regions.back().m_DstEntry + Regions.back().m_Count == int(Slot) ) ++Regions.back().m_Count;
                else Regions.push_back({ Src, int(Slot), 1 });
            };

            m_DrawSlot.resize(Scene.m_Draws.size());
            for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
            {
                const auto&         D  = Scene.m_Draws[i];
                const std::uint32_t ID = static_cast<std::uint32_t>(D.m_Instance);
                std::uint32_t Slot = ID ? m_InstanceSlots.Acquire(ID, D.m_L2W, Frame, D.m_Material) : instance_slots::none_v;
                if( Slot == instance_slots::none_v )
                {
                    if( m_TransientCount >= max_transient_v || m_RingUsed >= region_v ) { m_DrawSlot[i] = instance_slots::none_v; continue; }     // ponytail: dropped
                    Slot = std::uint32_t(max_instances_v + m_TransientCount++);
                    Put(Slot, D.m_L2W, D.m_Material);
                }
                m_DrawSlot[i] = Slot;
            }
            m_InstanceSlots.Flush(region_v - m_RingUsed, Put);     // what does not fit this frame stays pending (several renders per frame)
        });
        Window.CopyBuffer(m_InstanceRing, m_InstanceBuffer, Regions);
    }

    //------------------------------------------------------------------------------------------------

    // The local lights for this render: into this frame's region of the light buffer, and in modes CELLS / SAMPLED their world
    // cells into the cell buffer (features/lights/xlionrender_light_cells.h). Fills the lighting block's local part.
    void instance::writeLocalLights( const scene& Scene, const xmath::fvec3& Eye, details::ubo_lighting& Light ) noexcept
    {
        const std::uint64_t Frame = m_Device.getFrameIndex();
        if( Frame != m_LightFrame ) { m_LightFrame = Frame; m_LightRingUsed = 0; m_CellRingUsed = 0; }

        const int nLights = std::min({ int(Scene.m_LocalLights.size()), max_lights_v, max_lights_v - m_LightRingUsed });
        const int LightBase = int(Frame % ring_frames_v) * max_lights_v + m_LightRingUsed;
        if( nLights > 0 )
            (void)m_LightRing.MemoryMap(LightBase, nLights, [&](void* pMemory)
            {
                auto* pOut = static_cast<details::local_light_data*>(pMemory);
                for( int i = 0; i < nLights; ++i )
                {
                    const auto& LL    = Scene.m_LocalLights[i];
                    const bool  bSpot = LL.m_Type == local_light::type::SPOT;
                    const auto  Map   = i < int(m_LocalMaps.size()) ? m_LocalMaps[i] : shadows::local_map{};
                    auto&       Out   = pOut[i];
                    Out.m_PosRadius   = xmath::fvec4(LL.m_Position - Eye, LL.m_Radius);
                    Out.m_Color       = xmath::fvec4(LL.m_Color * LL.m_Intensity, std::cos(LL.m_InnerAngle));
                    Out.m_DirCosOuter = bSpot ? xmath::fvec4(LL.m_Direction.NormalizeCopy(), std::cos(LL.m_OuterAngle)) : xmath::fvec4(0, 0, 0, -2);
                    Out.m_Shadow      = { float(shadows::atlas_tiles_v * m_ShadowSetup.m_ClipmapTileSize + Map.m_X), float(Map.m_Y), float(Map.m_Size), bSpot ? shadows::SpotTanHalf(LL) : 1.0f };
                }
            });
        m_LightRingUsed += std::max(nLights, 0);

        auto Mode = m_LightMode;
        int  CellBase = 0;
        m_LightBuildMs = 0;
        if( Mode != light_mode::EXACT && nLights > 0 )
        {
            const auto T0 = std::chrono::steady_clock::now();
            m_CellLights.resize(nLights);
            for( int i = 0; i < nLights; ++i )
            {
                const auto& LL = Scene.m_LocalLights[i];
                const float Lum = 0.2126f * LL.m_Color.m_X + 0.7152f * LL.m_Color.m_Y + 0.0722f * LL.m_Color.m_Z;
                const bool  bSpot = LL.m_Type == local_light::type::SPOT;
                m_CellLights[i] = { LL.m_Position, LL.m_Radius, std::max(Lum * LL.m_Intensity, 1e-6f), LL.m_Direction.NormalizeCopy(), bSpot ? LL.m_OuterAngle : -1.0f };
            }
            m_LightCells.Build(m_CellLights, Eye, m_LightCellSize, Mode == light_mode::SAMPLED ? m_LightHead : 0);
            m_LightBuildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();

            const int nUints = int(m_LightCells.m_Data.size());
            if( nUints > max_cell_uints_v - m_CellRingUsed ) Mode = light_mode::EXACT;      // ponytail: too many entries for the buffer
            else
            {
                CellBase = int(Frame % ring_frames_v) * max_cell_uints_v + m_CellRingUsed;
                (void)m_CellRing.MemoryMap(CellBase, nUints, [&](void* pMemory) { std::memcpy(pMemory, m_LightCells.m_Data.data(), nUints * sizeof(std::uint32_t)); });
                m_CellRingUsed += nUints;
            }
        }

        m_LightModeUsed  = Mode;
        m_LightCellUints = Mode == light_mode::EXACT ? 0 : int(m_LightCells.m_Data.size());
        Light.m_LocalCount = { float(nLights), float(int(Mode)), float(m_LightSeed++ & 0xffffff), float(m_LightHead) };
        Light.m_LocalBase  = { float(LightBase), float(CellBase), m_LightCellSize, float(m_LightTailSamples) };
        for( int L = 0; L < light_cells::levels_v; ++L )
            Light.m_CellOrigin[L] = { float(m_LightCells.m_Origin[L][0]), float(m_LightCells.m_Origin[L][1]), float(m_LightCells.m_Origin[L][2]), 0 };
    }

    //------------------------------------------------------------------------------------------------

    // Shadows: which casters changed (world bounds), what must be rendered - the sun's clipmap tiles and the local lights' maps -, then:
    //  - the local maps' casters into the scratch (in batches that fit it), each batch remapped into the atlas (linear distance,
    //    octahedral for point lights)
    //  - the clipmap tiles straight into the atlas (each cleared and drawn with the casters over it)
    // Before the scene pass, which samples the atlas.
    void instance::renderShadows( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye, float ProjScale ) noexcept
    {
        if( !m_bShadowAtlasReady )
        {
            auto Cmd = Window.StartRenderPass(m_ShadowClearPass);      // the atlas's first use: cleared to far, so it can be loaded and sampled
            m_bShadowAtlasReady = true;
        }

        m_nShadowTilesRendered = 0;

        // The casters and what changed since the last frame: new and gone casters, and skinned ones (old and new bounds)
        struct draw_caster { const draw* m_pDraw; const mesh_slot* m_pMesh; xmath::fbbox m_World; std::uint32_t m_Slot; };
        std::vector<draw_caster>                         Draws;
        std::vector<xmath::fbbox>                        Changed;
        std::unordered_map<std::uint64_t, xmath::fbbox>  Now;
        Draws.reserve(Scene.m_Draws.size());
        Now.reserve(Scene.m_Draws.size());
        m_DrawWorld.assign(Scene.m_Draws.size(), xmath::fbbox{});
        for( std::size_t iDraw = 0; iDraw < Scene.m_Draws.size(); ++iDraw )
        {
            const auto& D     = Scene.m_Draws[iDraw];
            const auto* pMesh = findMesh(D.m_Mesh);
            if( pMesh == nullptr ) continue;
            if( pMesh->m_bCluster && m_DrawSlot[iDraw] == instance_slots::none_v ) continue;

            const bool bSkin = pMesh->m_bCluster && pMesh->m_Geom.m_Kind == geom_kind::SKIN;
            std::uint64_t Key = 1469598103934665603ull;
            Key = Hash(Key, &D.m_Mesh, sizeof(D.m_Mesh));
            Key = Hash(Key, &D.m_L2W, sizeof(D.m_L2W));
            Key = Hash(Key, D.m_Materials.data(), D.m_Materials.size() * sizeof(material_id));     // alpha test, displacement
            const xmath::fbbox World = (pMesh->m_bCluster ? pMesh->m_Geom.m_BBox : pMesh->m_BBox).TransformCopy(D.m_L2W);
            m_DrawWorld[iDraw] = World;
            Draws.push_back({ &D, pMesh, World, m_DrawSlot[iDraw] });
            Now.emplace(Key, World);

            const auto It = m_PrevCasters.find(Key);
            if( It == m_PrevCasters.end() || bSkin ) Changed.push_back(World);
            if( It != m_PrevCasters.end() && bSkin ) Changed.push_back(It->second);
        }
        for( const auto& [Key, World] : m_PrevCasters ) if( !Now.contains(Key) ) Changed.push_back(World);
        m_PrevCasters = std::move(Now);

        const auto& Sun = Scene.m_Sun;
        if( Sun.m_bCastShadows ) m_Clipmap.Update(Sun.m_Direction, Sun.m_ShadowTileSize, Eye, Changed, m_bShadowCache, m_ShadowTiles);
        else                     m_ShadowTiles.clear();
        m_LocalShadows.Update(Scene.m_LocalLights, Changed, m_bShadowCache, Eye, ProjScale, m_ShadowUpdatesPerFrame, m_LocalMaps, m_LocalToRender);

        m_nShadowTilesRendered = static_cast<int>(m_ShadowTiles.size() + m_LocalToRender.size());
        if( m_nShadowTilesRendered == 0 ) return;

        // Draws the casters that pass Keep with W2C into the current viewport
        const auto DrawCasters = [&]( xgpu::cmd_buffer& Cmd, const xmath::fmat4& W2C, const auto& Keep )
        {
            auto& ViewBlock = m_ShadowUBO.allocEntry<details::ubo_geom_view>();     // absolute world: the instances' origin as is
            ViewBlock.m_w2C         = W2C;
            ViewBlock.m_WorldOffset = xmath::fvec4(m_Origin, 0);
            const cluster_geom* pBoundGeom = nullptr;       // binds only when the mesh changes

            for( std::size_t i = 0; i < Draws.size(); ++i )
            {
                if( !Keep(i) ) continue;
                const auto& D = *Draws[i].m_pDraw;
                const auto& M = *Draws[i].m_pMesh;

                if( M.m_bCluster == false )
                {
                    pBoundGeom = nullptr;
                    Cmd.setPipelineInstance(m_ShadowInstance[0]);
                    Cmd.setPushConstants(details::push_forward{ D.m_L2W, W2C });
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(M.m_VertexBuffer));
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(M.m_IndexBuffer));
                    Cmd.Draw(M.m_nIndices);
                    continue;
                }

                const auto& G     = M.m_Geom;
                const bool  bSkin = G.m_Kind == geom_kind::SKIN;
                if( &G != pBoundGeom )
                {
                    std::array<xgpu::buffer*, 3> StaticBuffers{ G.m_pClusterData, bSkin ? G.m_pBones : &m_InstanceBuffer, &m_InstanceBuffer };
                    Cmd.setPipelineInstance(m_ShadowInstance[bSkin ? 2 : 1], std::span{ StaticBuffers.data(), bSkin ? std::size_t{3} : std::size_t{2} });
                    Cmd.setDynamicUBO(m_ShadowUBO, 0);
                    Cmd.setStreamingBuffers({ G.m_pStreams, 3 });
                    pBoundGeom = &G;
                }

                for( const auto& Sub : G.m_Submeshes )
                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
                        const auto& C = G.m_Clusters[c];
                        if( bSkin ) Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Draws[i].m_Slot });
                        else        Cmd.setPushConstants(details::push_static_geom{ c, Draws[i].m_Slot });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                    }
            }
        };

        const auto Viewport = [&]( xgpu::cmd_buffer& Cmd, int X, int Y, int S )
        {
            Cmd.setViewport(float(X), float(Y), float(S), float(S));
            Cmd.setScissor(X, Y, S, S);
        };

        // Local maps: batches whose faces fit in the scratch (a point light: 6 faces of half its map size in two blocks of its map
        // size, a spot: one face of its map size)
        struct face_batch { int m_iRender; std::array<shadows::buddy::pos, 2> m_Blocks; };
        std::vector<face_batch> Batch;
        const int LocalX = shadows::atlas_tiles_v * m_ShadowSetup.m_ClipmapTileSize;     // the local part's origin in the atlas
        bool      bClipmapDone = false;

        const auto FacePos = [&]( const face_batch& B, int Face, int Size, bool bPoint ) -> shadows::buddy::pos
        {
            if( !bPoint ) return B.m_Blocks[0];
            const auto& Block = B.m_Blocks[Face / 4];
            const int   Half  = Size / 2, Sub = Face % 4;
            return { Block.m_X + (Sub % 2) * Half, Block.m_Y + (Sub / 2) * Half };
        };

        const auto Flush = [&]()
        {
            if( !Batch.empty() )
            {
                auto Cmd = Window.StartRenderPass(m_ScratchPass);
                for( const auto& B : Batch )
                {
                    const auto& R      = m_LocalToRender[B.m_iRender];
                    const auto& L      = Scene.m_LocalLights[R.m_Light];
                    const bool  bPoint = L.m_Type == local_light::type::POINT;
                    const float Near   = shadows::LocalNear(L.m_Radius);
                    const auto  Keep   = [&](std::size_t i) { return shadows::Overlaps(Draws[i].m_World, L.m_Position, L.m_Radius); };   // ponytail: the sphere, not the face's frustum
                    for( int f = 0, nFaces = bPoint ? 6 : 1; f < nFaces; ++f )
                    {
                        const auto P = FacePos(B, f, R.m_Map.m_Size, bPoint);
                        Viewport(Cmd, P.m_X, P.m_Y, bPoint ? R.m_Map.m_Size / 2 : R.m_Map.m_Size);
                        DrawCasters(Cmd, bPoint ? shadows::FaceW2C(L.m_Position, shadows::CubeFaceAxis(f), 1.0f, Near, L.m_Radius)
                                                : shadows::FaceW2C(L.m_Position, L.m_Direction.NormalizeCopy(), shadows::SpotTanHalf(L), Near, L.m_Radius), Keep);
                    }
                }
            }

            auto Cmd = Window.StartRenderPass(m_ShadowPass);

            if( !bClipmapDone )
            {
                bClipmapDone = true;
                std::vector<shadows::lbox> LightBounds;         // the clipmap culls in light space
                LightBounds.reserve(Draws.size());
                for( const auto& C : Draws ) LightBounds.push_back(m_Clipmap.Bounds(C.m_World));
                const int S = m_ShadowSetup.m_ClipmapTileSize;
                for( const auto& T : m_ShadowTiles )
                {
                    Viewport(Cmd, T.m_AtlasX * S, T.m_AtlasY * S, S);
                    Cmd.setPipelineInstance(m_ShadowClearInstance);
                    Cmd.setBuffer(m_FullscreenVB);
                    Cmd.setBuffer(m_FullscreenIB);
                    Cmd.Draw(3);
                    DrawCasters(Cmd, T.m_W2C, [&](std::size_t i) { return shadows::clipmap::Overlaps(LightBounds[i], T.m_MinX, T.m_MinY, T.m_Size); });
                }
            }

            for( const auto& B : Batch )
            {
                const auto& R      = m_LocalToRender[B.m_iRender];
                const auto& L      = Scene.m_LocalLights[R.m_Light];
                const bool  bPoint = L.m_Type == local_light::type::POINT;
                details::push_shadow_remap Push{};
                Push.m_Tile   = { LocalX + R.m_Map.m_X, R.m_Map.m_Y, R.m_Map.m_Size, bPoint ? R.m_Map.m_Size / 2 : R.m_Map.m_Size };
                Push.m_Params = { shadows::LocalNear(L.m_Radius), L.m_Radius, bPoint ? 1.0f : shadows::SpotTanHalf(L), bPoint ? 1.0f : 0.0f };
                for( int f = 0; f < (bPoint ? 6 : 1); ++f )
                {
                    const auto P = FacePos(B, f, R.m_Map.m_Size, bPoint);
                    Push.m_Faces[f * 2] = P.m_X;
                    Push.m_Faces[f * 2 + 1] = P.m_Y;
                }
                Viewport(Cmd, Push.m_Tile[0], Push.m_Tile[1], R.m_Map.m_Size);
                Cmd.setPipelineInstance(m_RemapInstance);
                Cmd.setPushConstants(Push);
                Cmd.setBuffer(m_FullscreenVB);
                Cmd.setBuffer(m_FullscreenIB);
                Cmd.Draw(3);
            }
            Batch.clear();
            m_ScratchAlloc.Reset(m_ShadowSetup.m_ScratchSize, m_ShadowSetup.m_MinLocalSize);
        };

        m_ScratchAlloc.Reset(m_ShadowSetup.m_ScratchSize, m_ShadowSetup.m_MinLocalSize);
        for( int r = 0; r < int(m_LocalToRender.size()); ++r )
        {
            const auto& R      = m_LocalToRender[r];
            const bool  bPoint = Scene.m_LocalLights[R.m_Light].m_Type == local_light::type::POINT;
            const auto  TryAlloc = [&](face_batch& B)
            {
                if( !m_ScratchAlloc.Alloc(R.m_Map.m_Size, B.m_Blocks[0]) ) return false;
                return !bPoint || m_ScratchAlloc.Alloc(R.m_Map.m_Size, B.m_Blocks[1]);
            };
            face_batch B{ r, {} };
            if( !TryAlloc(B) )
            {
                Flush();
                if( !TryAlloc(B) ) continue;        // larger than the scratch: setup error (m_ScratchSize >= 2 x m_MaxLocalSize)
            }
            Batch.push_back(B);
        }
        if( !Batch.empty() || !bClipmapDone ) Flush();
    }

    //------------------------------------------------------------------------------------------------

    void instance::Render( xgpu::window& Window, const scene& Scene, render_view& View, xgpu::tools::view& Camera, const overlay_fn& Overlay ) noexcept
    {
        assert(m_bReady && View.m_Width > 0);

        Camera.setViewport({ 0, 0, View.m_Width, View.m_Height });

        // Camera-centered world (same as the editors): precision stays near the eye in large worlds
        const xmath::fvec3 Eye     = Camera.getPosition();
        const xmath::fmat4 W2C     = Camera.getW2C() * xmath::fmat4::fromTranslation(Eye);
        const xmath::fmat4 ToEye   = xmath::fmat4::fromTranslation(-Eye);

        const float ProjScale = float(View.m_Height) / (2.0f * xmath::Tan(xmath::radian{ Camera.getFov().m_Value * 0.5f }));    // pixels per unit at distance 1
        using clock = std::chrono::steady_clock;
        const auto Ms = [](clock::time_point A, clock::time_point B) { return std::chrono::duration<double, std::milli>(B - A).count(); };
        const auto T0 = clock::now();
        updateInstances(Window, Scene, Eye);
        const auto T1 = clock::now();
        Window.WriteTimestamp(m_Timestamps, 0);                 // GPU: the shadow passes start here (after the instance copy)
        renderShadows(Window, Scene, Eye, ProjScale);
        const auto T2 = clock::now();

        // Scene -> HDR
        {
            auto Cmd = Window.StartRenderPass(View.m_ScenePass);
            Cmd.WriteTimestamp(m_Timestamps, 1);

            {
                const auto& L     = Scene.m_Sun;
                auto&       Light = m_LightUBO.allocEntry<details::ubo_lighting>();
                Light.m_LightColor        = xmath::fvec4(L.m_Color * L.m_Intensity, 1);
                Light.m_AmbientLightColor = xmath::fvec4(Scene.m_Ambient, 0);
                Light.m_wSpaceLightPos    = xmath::fvec4(-L.m_Direction.NormalizeCopy(), 0);
                Light.m_wSpaceEyePos      = xmath::fvec4(0, 0, 0, 1);       // camera-centered
                Light.m_LightParams       = { 0, L.m_Temperature, 1, 0 };

                Light.m_W2Light = xmath::fmat4::fromIdentity();
                for( int c = 0; c < 3; ++c )
                {
                    const float Axis[3][3] = { { m_Clipmap.m_X.m_X, m_Clipmap.m_X.m_Y, m_Clipmap.m_X.m_Z }, { m_Clipmap.m_Y.m_X, m_Clipmap.m_Y.m_Y, m_Clipmap.m_Y.m_Z }, { m_Clipmap.m_Z.m_X, m_Clipmap.m_Z.m_Y, m_Clipmap.m_Z.m_Z } };
                    for( int r = 0; r < 3; ++r ) Light.m_W2Light(r, c) = Axis[r][c];
                }
                Light.m_CameraPos = xmath::fvec4(Eye, 1);
                for( int i = 0; i < shadows::clipmap_levels_v; ++i )
                {
                    const auto& Lv = m_Clipmap.m_Levels[i];
                    Light.m_ShadowLevels[i] = { float(Lv.m_OriginX), float(Lv.m_OriginY), Lv.m_TileSize, Lv.m_DepthCenter };
                }
                Light.m_ShadowParams = { L.m_bCastShadows ? 1.0f : 0.0f, m_Clipmap.DepthHalfRange(0), float(m_ShadowSetup.m_ClipmapTileSize), 0 };

                writeLocalLights(Scene, Eye, Light);
            }

            auto& ViewBlock = m_StaticGeomUBO.allocEntry<details::ubo_geom_view>();      // camera-centered
            ViewBlock.m_w2C         = W2C;
            ViewBlock.m_WorldOffset = xmath::fvec4(m_Origin - Eye, 0);

            // Frustum culling in the camera-centered world (the bounds move by -Eye)
            const frustum Frustum = frustum::FromW2C(W2C);

            // The visible draws, sorted by mesh then first material: runs of the same state need no binding between draws
                        m_Visible.clear();
            for( std::size_t iDraw = 0; iDraw < Scene.m_Draws.size(); ++iDraw )
            {
                const auto& D     = Scene.m_Draws[iDraw];
                const auto* pMesh = findMesh(D.m_Mesh);
                if( pMesh == nullptr ) continue;
                if( m_bFrustumCull )
                {
                    xmath::fbbox Box = m_DrawWorld[iDraw];
                    Box.m_Min = Box.m_Min - Eye;
                    Box.m_Max = Box.m_Max - Eye;
                    if( !Frustum.Overlaps(Box) ) continue;
                }
                m_Visible.push_back({ pMesh, D.m_Materials.empty() ? ~0u : D.m_Materials[0].m_Index, std::uint32_t(iDraw) });
            }
            std::sort(m_Visible.begin(), m_Visible.end(), [](const visible_draw& a, const visible_draw& b)
            {
                return a.m_pMesh != b.m_pMesh ? a.m_pMesh < b.m_pMesh : a.m_Material != b.m_Material ? a.m_Material < b.m_Material : a.m_iDraw < b.m_iDraw;
            });
            m_nDrawsRendered = int(m_Visible.size());

            const xgpu::pipeline_instance* pBound     = nullptr;      // the state the last cluster draw left bound
            const cluster_geom*            pBoundGeom = nullptr;
            for( const auto& Vis : m_Visible )
            {
                const std::size_t iDraw = Vis.m_iDraw;
                const auto& D     = Scene.m_Draws[iDraw];
                const auto* pMesh = Vis.m_pMesh;

                const xmath::fmat4 L2w  = ToEye * D.m_L2W;
                const std::uint32_t Slot = m_DrawSlot[iDraw];

                // The standard (constant) material: pipeline first, the dynamic UBOs bind to the active pipeline's layout
                const auto BindStandard = [&](xgpu::pipeline_instance& Instance, std::span<xgpu::buffer*> StaticBuffers)
                {
                    Cmd.setPipelineInstance(Instance, StaticBuffers);
                    Cmd.setDynamicUBO(m_LightUBO, 1);
                    auto& Mat = m_MaterialUBO.allocEntry<details::ubo_material>();
                    Mat.m_BaseColor = xmath::fvec4(D.m_Material.m_BaseColor, 1);
                    Mat.m_Params    = { D.m_Material.m_Metallic, D.m_Material.m_Roughness, D.m_Material.m_AO, D.m_Material.m_F0 };
                    Mat.m_Emissive  = xmath::fvec4(D.m_Material.m_Emissive, 0);
                    Cmd.setDynamicUBO(m_MaterialUBO, 2);
                };

                if( pMesh->m_bCluster == false )
                {
                    pBound = nullptr;
                    std::array<xgpu::buffer*, 2> LightBuffers{ &m_LightRing, &m_CellRing };
                    BindStandard(m_ForwardInstance, LightBuffers);
                    Cmd.setPushConstants(details::push_forward{ L2w, W2C });
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_VertexBuffer));
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_IndexBuffer));
                    Cmd.Draw(pMesh->m_nIndices);
                    continue;
                }

                const auto& G     = pMesh->m_Geom;
                const bool  bSkin = G.m_Kind == geom_kind::SKIN;
                if( Slot == instance_slots::none_v ) continue;
                std::array<xgpu::buffer*, 5> StaticBuffers = bSkin ? std::array<xgpu::buffer*, 5>{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_LightRing, &m_CellRing }
                                                                   : std::array<xgpu::buffer*, 5>{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, nullptr };
                const auto  Static = std::span{ StaticBuffers.data(), bSkin ? std::size_t{5} : std::size_t{4} };

                for( const auto& Sub : G.m_Submeshes )
                {
                    auto* pMat      = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                    auto* pInstance = pMat ? getMaterialInstance(*pMat, G.m_Kind) : nullptr;
                    if( pInstance == nullptr )
                    {
                        if( bSkin ) continue;       // ponytail: no constant-material fallback on the skin path; the editors give every slot a material
                        pInstance = &m_StaticGeomInstance;      // the constant material, read from the instance record
                    }
                    if( pInstance != pBound || &G != pBoundGeom )
                    {
                        Cmd.setPipelineInstance(*pInstance, Static);
                        Cmd.setDynamicUBO(m_LightUBO, 1);
                        Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                        Cmd.setStreamingBuffers({ G.m_pStreams, 3 });
                        pBound     = pInstance;
                        pBoundGeom = &G;
                    }

                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
                        const auto& C = G.m_Clusters[c];
                        if( bSkin ) Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Slot });
                        else        Cmd.setPushConstants(details::push_static_geom{ c, Slot });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                    }
                }
            }

            Cmd.WriteTimestamp(m_Timestamps, 2);
        }

        // HDR -> exposure -> LionMapper -> sRGB output (alpha = coverage)
        {
            auto Cmd = Window.StartRenderPass(View.m_OutputPass);
            Cmd.setPipelineInstance(View.m_ToneMapInstance);
            Cmd.setPushConstants(details::push_tonemap{ { View.m_Exposure, 0, 0, 0 }, View.m_Background });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
            Cmd.WriteTimestamp(m_Timestamps, 3);
        }

        // Editor overlays over the tone-mapped image, depth-tested against the scene
        if( Overlay )
        {
            auto Cmd = Window.StartRenderPass(View.m_OverlayPass);
            Overlay(Cmd);
        }

        const auto T3 = clock::now();
        m_CPUTimes = { Ms(T0, T1), Ms(T1, T2), m_LightBuildMs, Ms(T0, T3) };
    }
}
