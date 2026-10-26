// Plan A - forward. instance::Render lives here so each plan is one file to read.
#include "xlionrender_instance.h"
#include "plans/forward/xlionrender_plan_forward.h"
#include <chrono>
#include <optional>

namespace xlionrender
{
    namespace
    {
        constexpr std::uint32_t s_GISurfelPassComp[] = {
            #include "xlionrender_gi_surfel_pass_comp.h"
        };
        constexpr std::uint32_t s_GIScatterComp[] = {
            #include "xlionrender_gi_scatter_comp.h"
        };
        constexpr std::uint32_t s_GICommitComp[] = {
            #include "xlionrender_gi_commit_comp.h"
        };
        constexpr std::uint32_t s_GIOpsComp[] = {
            #include "xlionrender_gi_ops_comp.h"
        };
        constexpr std::uint32_t s_GIFilterComp[] = {
            #include "xlionrender_gi_filter_comp.h"
        };
        constexpr std::uint32_t s_GIExpandComp[] = {
            #include "xlionrender_gi_expand_comp.h"
        };
        constexpr std::uint32_t s_GIMeansComp[] = {
            #include "xlionrender_gi_means_comp.h"
        };
        constexpr std::uint32_t s_GICubeComp[] = {
            #include "xlionrender_gi_cube_comp.h"
        };
        constexpr std::uint32_t s_GIMigrateComp[] = {
            #include "xlionrender_gi_migrate_comp.h"
        };
        constexpr std::uint32_t s_GIReadBackComp[] = {
            #include "xlionrender_gi_readback_comp.h"
        };
        constexpr std::uint32_t s_GILinksComp[] = {        // experiment D: the links and their solve
            #include "xlionrender_gi_links_comp.h"
        };
        constexpr std::uint32_t s_GISolveComp[] = {
            #include "xlionrender_gi_solve_comp.h"
        };
        constexpr std::uint32_t s_GIGhostsComp[] = {        // GI sections: the ghosts' copy
            #include "xlionrender_gi_ghosts_comp.h"
        };
        constexpr std::uint32_t s_GIProbesComp[] = {        // the probe grid (what has no surfels: skinned meshes)
            #include "xlionrender_gi_probes_comp.h"
        };

        // FNV-1a over a caster's identity and state: a different value is a different (or moved) caster
        std::uint64_t Hash(std::uint64_t H, const void* pData, std::size_t Size) noexcept
        {
            for( auto* p = static_cast<const std::uint8_t*>(pData); Size--; ++p ) H = (H ^ *p) * 1099511628211ull;
            return H;
        }

        // A planned cluster draw's identity (instance::m_ClusterPlanKey): its mesh and entity (never 0: not planned)
        std::uint64_t ClusterPlanKey(const draw& D) noexcept
        {
            return Hash(Hash(1469598103934665603ull, &D.m_Mesh, sizeof(D.m_Mesh)), &D.m_UserID, sizeof(D.m_UserID)) | 1;
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
            m_ParamRingUsed  = 0;
            m_nSlotsWritten  = 0;
        }

        // The origin (camera-relative "little world" for the stored rows): re-snapped when the eye moves a grid cell away
        const auto Far = [&](int i) { return std::abs(Eye[i] - m_Origin[i]) > origin_grid_v; };
        if( !m_bOriginSet || Far(0) || Far(1) || Far(2) )
        {
            for( int i = 0; i < 3; ++i ) m_Origin[i] = std::round(Eye[i] / origin_grid_v) * origin_grid_v;
            m_bOriginSet = true;
            m_InstanceSlots.MarkAllPending();
        }

        // This frame's ring region: written in order, copied by contiguous runs. The entity parameters: the same entries of their own ring
        constexpr int    region_v = max_instances_v + max_transient_v;
        const int        Base     = int(Frame % ring_frames_v) * region_v;
        std::vector<xgpu::buffer::copy> Regions;
        (void)m_EntityParamsRing.MemoryMapFrameRegion(Base, region_v, [&](void* pEntityMemory) {        // (this frame's slot: no wait for the frame in flight)
        (void)m_InstanceRing.MemoryMapFrameRegion(Base, region_v, [&](void* pMemory)
        {
            auto* pRing    = static_cast<details::instance_data*>(pMemory);
            auto* pEntRing = static_cast<entity_params*>(pEntityMemory);
            const auto Put = [&](std::uint32_t Slot, const xmath::fmat4& L2W, const material& Mat, const entity_params& Ent)
            {
                pRing[m_RingUsed].set(L2W, m_Origin, Mat);
                pEntRing[m_RingUsed] = Ent;
                const int Src = Base + m_RingUsed++;
                if( !Regions.empty() && Regions.back().m_SrcEntry + Regions.back().m_Count == Src && Regions.back().m_DstEntry + Regions.back().m_Count == int(Slot) ) ++Regions.back().m_Count;
                else Regions.push_back({ Src, int(Slot), 1 });
            };

            m_DrawSlot.resize(Scene.m_Draws.size());
            m_DrawChanged.assign(Scene.m_Draws.size(), 0);
            for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
            {
                const auto&         D  = Scene.m_Draws[i];
                const std::uint32_t ID = static_cast<std::uint32_t>(D.m_Instance);
                bool bChanged = true;
                std::uint32_t Slot = ID ? m_InstanceSlots.Acquire(ID, D.m_L2W, Frame, D.m_Material, &bChanged, D.m_EntityParams) : instance_slots::none_v;
                m_DrawChanged[i] = bChanged;
                if( Slot == instance_slots::none_v )
                {
                    if( m_TransientCount >= max_transient_v || m_RingUsed >= region_v ) { m_DrawSlot[i] = instance_slots::none_v; continue; }     // ponytail: dropped
                    Slot = std::uint32_t(max_instances_v + m_TransientCount++);
                    Put(Slot, D.m_L2W, D.m_Material, D.m_EntityParams);
                }
                m_DrawSlot[i] = Slot;
            }
            m_nSlotsWritten += m_InstanceSlots.Flush(region_v - m_RingUsed, Put);     // what does not fit this frame stays pending (several renders per frame)
        }); });
        Window.CopyBuffer(m_InstanceRing, m_InstanceBuffer, Regions);
        Window.CopyBuffer(m_EntityParamsRing, m_EntityParams, Regions);

        // New and changed parameter blocks: packed into this frame's ring region (in order: a later write of the same vec4s wins), then
        // copied into MaterialParams at their own offsets; what does not fit waits for the next frame, still in order
        if( m_ParamPending.empty() ) return;
        const int ParamBase = int(Frame % ring_frames_v) * param_ring_v;
        std::vector<xgpu::buffer::copy> Blocks;
        std::size_t nDone = 0;
        (void)m_MaterialParamsRing.MemoryMap(ParamBase, param_ring_v, [&](void* pMemory)
        {
            for( ; nDone < m_ParamPending.size(); ++nDone )         // (after what this frame's earlier renders wrote: their copies have not run yet)
            {
                const auto& [At, Floats] = m_ParamPending[nDone];
                const int n = int(Floats.size() / 4);
                assert(n <= param_ring_v);
                if( m_ParamRingUsed + n > param_ring_v ) break;
                std::memcpy(static_cast<float*>(pMemory) + 4 * std::size_t(m_ParamRingUsed), Floats.data(), Floats.size() * sizeof(float));
                Blocks.push_back({ ParamBase + m_ParamRingUsed, At, n });
                m_ParamRingUsed += n;
            }
        });
        m_ParamPending.erase(m_ParamPending.begin(), m_ParamPending.begin() + nDone);
        Window.CopyBuffer(m_MaterialParamsRing, m_MaterialParams, Blocks);
    }

    //------------------------------------------------------------------------------------------------

    static int g_nSceneBinds = 0, g_nSceneDraws = 0;
    static double g_BindMs = 0, g_DrawMs = 0;     // measurement only (XLION_FRAME_TIMES): the main scene pass's binds and cluster draws
    // Measurement only (XLION_GPU_TIMES): GPU timestamps of the GI update's parts (slots: before Submit, after it, before Show, after it)
    namespace { struct gi_times { xgpu::timestamps m_T; bool m_bOk = false; std::vector<std::uint64_t> m_Ns; double m_Sum[3] = {}; int m_n = 0; }; }
    static gi_times* GITimes(const void* pKey, xgpu::device& Device) noexcept
    {
        static const bool s_bOn = std::getenv("XLION_GPU_TIMES") != nullptr;
        if( !s_bOn ) return nullptr;
        static std::unordered_map<const void*, gi_times> s_Map;      // ponytail: per renderer, never erased (a measurement hook)
        auto [It, bNew] = s_Map.try_emplace(pKey);
        if( bNew ) It->second.m_bOk = Device.Create(It->second.m_T, { .m_Count = 4 }) == nullptr;
        if( !It->second.m_bOk ) return nullptr;
        auto& G = It->second;
        if( G.m_T.getResults(G.m_Ns) && G.m_Ns.size() >= 4 && G.m_Ns[3] > G.m_Ns[0] )
        {
            for( int k = 0; k < 3; ++k ) G.m_Sum[k] += (G.m_Ns[k + 1] - G.m_Ns[k]) * 1e-6;
            if( ++G.m_n == 120 ) { std::printf("GIGPU submit %.2f, between %.2f, show %.2f ms\n", G.m_Sum[0] / G.m_n, G.m_Sum[1] / G.m_n, G.m_Sum[2] / G.m_n); std::fflush(stdout); G.m_n = 0; for( auto& v : G.m_Sum ) v = 0; }
        }
        return &G;
    }


    // Surfel GI (features/gi/xlionrender_surfel_gi.h): the draws of meshes with surfels and a persistent id become world instances
    // (added, moved, removed when no longer drawn), the scene's lights its lights, one Update on the ray budget; then every drawn
    // instance's active-level light into this frame's region of the GI light ring (m_DrawGI: where, which level)
    void instance::updateGI( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye, float ProjScale ) noexcept
    {
        m_DrawGI.assign(Scene.m_Draws.size(), { 0u, 0xFFFFFFFFu });
        m_DrawGISec.assign(Scene.m_Draws.size(), {});
        if( !Scene.m_GI.m_bEnable ) return;
        const auto TGI = std::chrono::steady_clock::now();
        const auto MsSince = [](std::chrono::steady_clock::time_point T) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T).count(); };
        if( !m_pGIWorld ) m_pGIWorld = std::make_unique<gi::world>();
        auto& W = *m_pGIWorld;
        ++m_GIRender;
        if( Scene.m_GI.m_bGPUTrace && !m_pGITrace && !m_bGITraceFailed )
        {
            m_pGITrace = std::make_unique<gi::gpu_trace>();
            if( !m_pGITrace->Init(m_Device, { s_GISurfelPassComp, s_GIScatterComp, s_GICommitComp, s_GIOpsComp, s_GIFilterComp, s_GIExpandComp, s_GIMeansComp, s_GICubeComp, s_GIMigrateComp, s_GIReadBackComp, s_GILinksComp, s_GISolveComp, s_GIGhostsComp, s_GIProbesComp }) )
                { m_pGITrace.reset(); m_bGITraceFailed = true; }
        }
        // The light lives on the GPU (m_bGPUOwned). Turned off, the GPU's light comes back into the world first (a read back: one frame
        // more on the GPU), then the CPU takes over
        if( m_pGITrace && !Scene.m_GI.m_bGPUTrace && !m_bGIToCPU && W.m_Setup.m_bGPUOwned ) m_bGIToCPU = true;
        bool                bGPU    = m_pGITrace && (Scene.m_GI.m_bGPUTrace || m_bGIToCPU);
        const std::uint64_t GIFrame = m_Device.getFrameIndex();
        W.m_Setup.m_bGPUTrace = bGPU;
        W.m_Setup.m_bGPUShown = bGPU;                               // (G4: the shown light too: Show below, straight into the ring)
        W.m_Setup.m_bGPUOwned = bGPU;                               // (G4c: the light lives there: the pick hands out ranges)
        W.m_Setup.m_bEmitterSampling = Scene.m_GI.m_bEmitterSampling;      // (G5: no links)
        const auto TCollect = std::chrono::steady_clock::now();
        if( m_pGITrace ) m_pGITrace->Collect(W, GIFrame);         // (the GPU's last slice; a read back asked for: into the world's arrays)
        const double CollectMs = MsSince(TCollect);
        if( m_bGIToCPU && m_bGIToCPUAsked && m_pGITrace && !m_pGITrace->ReadBackPending() )        // the world has the GPU's light: the CPU's now
        {
            m_pGITrace.reset(); m_bGIToCPU = m_bGIToCPUAsked = false; bGPU = false;
            W.m_Setup.m_bGPUTrace = W.m_Setup.m_bGPUShown = W.m_Setup.m_bGPUOwned = false;
            W.DirtyAll();                                           // (its passes in flight are dropped: traced again)
        }
        static std::unordered_map<const void*, bool> s_TransportAsked;     // measurement (XLION_GI_TRANSPORT): compiled at the dump, below (ponytail: never erased)
        if( !m_GIDumpPath.empty() && m_bGIDumpAsked && (!m_pGITrace || !m_pGITrace->ReadBackPending()) )        // DumpGI: the GPU's light came back
        {
            if( std::getenv("XLION_GI_TRANSPORT") ) s_TransportAsked[this] = true;
            (void)W.Dump(m_GIDumpPath.c_str()); (void)DumpGIFrames((m_GIDumpPath + ".frames.csv").c_str());
            m_GIDumpPath.clear(); m_bGIDumpAsked = false;
        }

        std::vector<int> DrawId(Scene.m_Draws.size(), -1);
        std::vector<std::vector<int>> DrawSecIds(Scene.m_Draws.size());     // GI sections: per draw its sections' world instances
        for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
        {
            const auto&         D  = Scene.m_Draws[i];
            const std::uint32_t ID = std::uint32_t(D.m_Instance);
            const auto*         pM = findMesh(D.m_Mesh);
            if( ID == 0 || pM == nullptr || (!pM->m_pGI && pM->m_GISections.empty()) ) continue;
            // One world instance for the geometry's surfels (Key: the draw's), or one a section (gi_chunks_plan.md: the object is still one -
            // moved, recolored, removed as a whole; its sections' ghosts copy their owners). true: it was added
            const auto Register = [&](int& Id, const gi::surfel_mesh& Mesh, bool bCapture, int Section)
            {
            // New surfels (a recompile): by the stamp, not the address - the new ones can be allocated where the old ones were
            if( Id >= 0 && m_GITrack[Id].m_GIStamp != pM->m_GIStamp ) { W.Remove(Id); Id = -1; }
            // What the GI takes from the draw's materials: which slots it sees through (transparent ones), and its colors (captured)
            const auto TakeMaterials = [&](int Id)
            {
                std::vector<bool> Alpha(D.m_Materials.size(), false);
                for( std::size_t k = 0; k < D.m_Materials.size(); ++k ) if( const auto* pMat = findMaterial(D.m_Materials[k]); pMat && MaterialVisibilityOf(pMat->m_State).m_Opacity == material_visibility::opacity::BLENDED ) Alpha[k] = true;
                W.SetSeeThrough(Id, Alpha);
                const_cast<mesh_slot*>(pM)->m_bTriColors = false;              // the SURFACE_COLOR view's table: made again from the new capture
                // a capture still waiting for this instance is of materials it no longer has (one in flight does not publish: processGICapture)
                std::erase_if(m_GICaptures, [&](const gi_capture_job& J) { return J.m_GIId == Id && !J.m_bRecorded; });
                if( bCapture && !Mesh.m_SurfelMaterial.empty() )      // its surfels' and its triangles' colors: captured from its materials (processGICapture; a section: its own)
                {
                    m_GICaptures.push_back({ .m_GIId = Id, .m_Mesh = D.m_Mesh, .m_Materials = D.m_Materials, .m_Slot = i < m_DrawSlot.size() ? m_DrawSlot[i] : 0u, .m_Section = Section });
                    for( std::size_t t = 0; t < Mesh.m_Tris.size(); t += gi_capture_max_v )
                        m_GICaptures.push_back({ .m_GIId = Id, .m_Mesh = D.m_Mesh, .m_Materials = D.m_Materials, .m_Slot = i < m_DrawSlot.size() ? m_DrawSlot[i] : 0u
                                               , .m_bTriangles = true, .m_First = std::uint32_t(t), .m_Section = Section });
                }
            };
            const auto Role = gi::role((D.m_Instance >> draw_gi_role_shift_v) & 3u);
            const bool bAdded = Id < 0;
            if( Id < 0 )
            {
                const auto& Mat = D.m_Material;
                Id = W.Add(Mesh, D.m_L2W, Mat.m_BaseColor * (1 - Mat.m_Metallic));
                W.SetRole(Id, Role);
                m_GITrack.resize(std::max<std::size_t>(m_GITrack.size(), Id + 1));
                m_GITrack[Id] = { D.m_L2W, &Mesh, pM->m_GIStamp, 0, D.m_Materials };
                TakeMaterials(Id);
                if( bCapture && !Mesh.m_SurfelMaterial.empty() ) W.SetProvisional(Id, true);     // (gi_dynamic_master_plan.md 7a: until its captures are in)
            }
            else
            {
                if( m_GITrack[Id].m_Materials != D.m_Materials )      // its materials were made again (a material compiled again) or changed
                {
                    m_GITrack[Id].m_Materials = D.m_Materials;
                    TakeMaterials(Id);
                }
                W.SetRole(Id, Role);                                   // (nothing when it is the same)
                if( std::memcmp(&m_GITrack[Id].m_L2W, &D.m_L2W, sizeof(xmath::fmat4)) != 0 )
                {
                    W.SetTransform(Id, D.m_L2W);
                    m_GITrack[Id].m_L2W = D.m_L2W;
                }
            }
            m_GITrack[Id].m_Render = m_GIRender;
            return bAdded;
            };
            if( pM->m_pGI )
            {
                auto [It, bNew] = m_GIIds.try_emplace(ID, -1);
                (void)Register(It->second, *pM->m_pGI, true, -1);
                DrawId[i] = It->second;
                continue;
            }
            // its sections: one world instance each, then their ghosts named by the owners' instances (equal levels: the plan's gate)
            auto& Ids = DrawSecIds[i];
            bool bAny = false;
            for( std::size_t k = 0; k < pM->m_GISections.size(); ++k )
            {
                auto [It, bNew] = m_GISecIds.try_emplace((std::uint64_t(ID) << 16) | std::uint64_t(k), -1);
                bAny |= Register(It->second, *pM->m_GISections[k].m_pGI, true, int(k));
                Ids.push_back(It->second);
            }
            if( bAny )
                for( std::size_t k = 0; k < pM->m_GISections.size(); ++k )
                {
                    const auto& S = pM->m_GISections[k];
                    std::vector<std::array<int, 2>> Owners;
                    Owners.reserve(S.m_Ghosts.size());
                    for( const auto& G : S.m_Ghosts ) Owners.push_back({ G.m_Chunk < Ids.size() ? Ids[G.m_Chunk] : -1, int(G.m_Local) });
                    W.SetGhosts(Ids[k], S.m_nOwned, std::move(Owners));
                }
        }
        const auto Retire = [&](auto& Ids)
        {
            for( auto It = Ids.begin(); It != Ids.end(); )
            {
                if( m_GITrack[It->second].m_Render != m_GIRender )
                {
                    W.Remove(It->second);
                    auto& T = m_GITrack[It->second];
                    if( T.m_Offset >= 0 ) m_GIFree.push_back({ T.m_Offset, T.m_Size + 1 });     // its place in the rings
                    T.m_Offset = -1;
                    It = Ids.erase(It);
                }
                else ++It;
            }
        };
        Retire(m_GIIds);
        Retire(m_GISecIds);

        // The lights: the sun, the sky (or the flat ambient as a sky), the local lights as points (ponytail: spots as points)
        gi::lights L;
        L.m_SunDirection = Scene.m_Sun.m_Direction.NormalizeCopy();
        L.m_SunColor     = Scene.m_Sun.m_bGI ? Scene.m_Sun.m_Color * Scene.m_Sun.m_Intensity : xmath::fvec3{ 0, 0, 0 };     // (GI off: direct light only)
        L.m_Sky          = Scene.m_Sky.m_Intensity <= 0 ? Scene.m_Ambient * (1.0f / gi::pi_v)
                         : Scene.m_Sky.m_pCubemap      ? m_EnvMean * Scene.m_Sky.m_Intensity                                  // (updateEnvironment)
                         :                               (Scene.m_Sky.m_Zenith + Scene.m_Sky.m_Horizon) * (0.5f * Scene.m_Sky.m_Intensity);
        if( Scene.m_Sky.m_Intensity > 0 && Scene.m_Sky.m_pCubemap && m_bEnvOct )
        {
            for( std::size_t k = 0; k < L.m_SkyOct.size(); ++k ) L.m_SkyOct[k] = m_EnvOct[k] * Scene.m_Sky.m_Intensity;
            L.m_bSkyOct = true;
        }
        L.m_Points.reserve(Scene.m_LocalLights.size());
        for( const auto& LL : Scene.m_LocalLights ) if( LL.m_bGI )
        {
            gi::lights::point P{ LL.m_Position, LL.m_Color * LL.m_Intensity, LL.m_Radius };
            if( LL.m_Type == local_light::type::SPOT )      // its cone, as the renderer's (local_light_data: cos outer, cos inner)
            {
                const auto Dn = LL.m_Direction.NormalizeCopy();
                P.m_Dir = { Dn.m_X, Dn.m_Y, Dn.m_Z }; P.m_CosOuter = std::cos(std::max(LL.m_OuterAngle, LL.m_InnerAngle)); P.m_CosInner = std::cos(LL.m_InnerAngle);
            }
            L.m_Points.push_back(P);
        }
        if( auto It = s_TransportAsked.find(this); It != s_TransportAsked.end() && It->second )      // measurement: experiment D at a real size
        {
            It->second = false;
            const int Rays = std::max(16, std::atoi(std::getenv("XLION_GI_TRANSPORT")));
            const auto T = W.CompileTransport(L, Rays);
            std::vector<gi::v3> Ref; std::vector<char> Near;
            for( int i = 0; i < W.InstanceCount(); ++i )
            {
                const auto& E = W.InstanceLight(i);
                const std::size_t nRows = T.m_InstFirst[std::size_t(i) + 1] - T.m_InstFirst[std::size_t(i)];
                for( std::size_t r = 0; r < nRows; ++r ) { Ref.push_back(r < E.size() ? E[r] : gi::v3{ 0, 0, 0 }); Near.push_back(W.Level(i) == 0 && E.size() == nRows); }
            }
            const auto Lum = [](const gi::v3& e) { return 0.2126 * e.m_X + 0.7152 * e.m_Y + 0.0722 * e.m_Z; };
            std::size_t MaxLinks = 0; for( std::size_t r = 0; r + 1 < T.m_First.size(); ++r ) MaxLinks = std::max<std::size_t>(MaxLinks, T.m_First[r + 1] - T.m_First[r]);
            std::printf("GITRANSPORT %zu surfels, %d rays each: compiled in %.0f ms; %zu links (%.1f a surfel, at most %zu; %.1f MB at 16 B); rays: %.1f%% sky, %.1f%% unsupported (%.2f%% dropped), %.1f%% far, %.1f%% back\n",
                        T.m_D.size(), Rays, T.m_CompileMs, T.m_Col.size(), double(T.m_Col.size()) / double(std::max<std::size_t>(T.m_D.size(), 1)), MaxLinks, T.m_Col.size() * 16.0 / 1048576.0,
                        100.0 * T.m_nSky / double(T.m_nRays), 100.0 * T.m_nUnsupported / double(T.m_nRays), 100.0 * T.m_nDropped / double(T.m_nRays), 100.0 * T.m_nFar / double(T.m_nRays), 100.0 * T.m_nBack / double(T.m_nRays));
            for( const int KI : { 0, 16, 32, 116, 132 } )        // (1xx: the tail to whole instances)
            {
            const int K = KI % 100; const bool bInst = KI >= 100;
            const auto Tk = K > 0 ? W.TruncateTransport(T, K, bInst) : gi::world::transport{};
            const auto& TT = K > 0 ? Tk : T;
            if( K > 0 ) std::printf("GITRANSPORT top %d + the tail through the %s: %zu links + %zu aggregate (%.1f MB at 16 B)\n", K, bInst ? "instances" : "parents", TT.m_Col.size(), TT.m_AGroup.size(), (TT.m_Col.size() + TT.m_AGroup.size()) * 16.0 / 1048576.0);
            std::vector<gi::v3> E;
            double SolveMs = 0;
            for( int it = 1; it <= 20; ++it )
            {
                const auto Ts = std::chrono::steady_clock::now();
                gi::world::SolveTransport(TT, E, 1);
                SolveMs += MsSince(Ts);
                if( it != 5 && it != 20 ) continue;
                double a = 0, b = 0, d2 = 0, Top = 1e-9; std::size_t nN = 0;
                for( std::size_t r = 0; r < E.size(); ++r ) if( Near[r] ) { a += Lum(E[r]); b += Lum(Ref[r]); d2 += (Lum(E[r]) - Lum(Ref[r])) * (Lum(E[r]) - Lum(Ref[r])); Top = std::max(Top, Lum(Ref[r])); ++nN; }
                std::printf("GITRANSPORT iteration %d: the near surfels' total light %.2f%% off the GPU's converged light, %.2f%% RMS (of the brightest); %.2f ms an iteration on the CPU\n",
                            it, 100 * std::abs(a - b) / std::max(b, 1e-9), 100 * std::sqrt(d2 / double(std::max<std::size_t>(nN, 1))) / Top, SolveMs / it);
            }
            }
            std::fflush(stdout);
        }
        W.m_Setup.m_FilterPasses  = std::clamp(Scene.m_GI.m_FilterPasses, 0, 16);   // what is shown only: no re-solve
        W.m_Setup.m_RaysPerSurfel = std::clamp(Scene.m_GI.m_RaysPerSurfel, 1, 256);
        W.m_Setup.m_Saturation    = std::clamp(Scene.m_GI.m_Saturation, 0.0f, 4.0f);     // what is shown only: no re-solve
        W.m_Setup.m_ProjScale     = ProjScale;
        W.m_Setup.m_TraceAllowanceMs = std::max(Scene.m_GI.m_TraceAllowanceMs, 0.0f);
        W.m_Setup.m_OverheadMs       = float(m_GIOverheadMs);                      // Phase 6b: this renderer's GI work outside Update (its last frames), from the allowance
        W.m_Setup.m_PixelsPerSurfel = std::max(Scene.m_GI.m_PixelsPerSurfel, 0.0f);
        if( W.m_Setup.m_EmissiveScale != Scene.m_GI.m_EmissiveScale )          // a new scale: every instance re-solves
        {
            W.m_Setup.m_EmissiveScale = Scene.m_GI.m_EmissiveScale;
            W.DirtyAll();
        }
        // G3, G4 (gi_dynamic_master_plan.md): the rays on the GPU, over frames. This frame's slice dispatched after the update (it commits
        // what came in, picks when the batch is all in); the last frame's slice into the world before it (Collect, at the top: before
        // anything here changes the world, whose copy of the GPU's state must be current first)
        // Diagnostic (gi_dynamic_master_plan.md 7a, smoke/gi_startup.py --captures-first): with XLION_GI_HOLD_CAPTURES set, no solve until
        // the material captures have been done for 10 frames (the control: the solve never sees an uncaptured material)
        static const bool s_bHold = std::getenv("XLION_GI_HOLD_CAPTURES") != nullptr;
        {   // (gi_dynamic_master_plan.md 7a) an instance is provisional until no capture of it is queued: all of them in (or dropped)
            std::vector<char> Queued(std::size_t(W.InstanceCount()), 0);
            for( const auto& J : m_GICaptures ) if( std::size_t(J.m_GIId) < Queued.size() ) Queued[std::size_t(J.m_GIId)] = 1;
            const auto Release = [&](const auto& Ids) { for( const auto& [Key, Id] : Ids ) if( std::size_t(Id) < Queued.size() && !Queued[std::size_t(Id)] && W.IsProvisional(Id) ) W.SetProvisional(Id, false); };
            Release(m_GIIds);
            Release(m_GISecIds);        // (the sections' too: they were never released - their hits brought nothing, no colour bounce)
        }
        m_GICapturesQuiet = m_GICaptures.empty() ? m_GICapturesQuiet + 1 : 0;
        if( s_bHold && m_GICapturesQuiet < 10 ) return;
        // Measurement only (GI review, experiment A: the ablations). XLION_GI_ABLATE holds any of: "freeze" (after XLION_GI_FREEZE_S seconds,
        // default 15, of this renderer's GI the slices are traced, their stats kept, the light not committed: the cache stays as it settled), "hits" (frozen, and the
        // pass's hits read a constant instead of the cache), "pixel_vertex" (the pixels read their vertices' surfels only: no per-pixel walk),
        // "pixel_nomoments" (the per-pixel read without its moment test), "pixel_none" (no GI read in shading at all; the solver still runs)
        static const std::string s_Ablate = [] { const char* p = std::getenv("XLION_GI_ABLATE"); return std::string(p ? p : ""); }();
        static const double      s_FreezeS = [] { const char* p = std::getenv("XLION_GI_FREEZE_S"); return p ? std::atof(p) : 15.0; }();
        if( !s_Ablate.empty() )
        {
            static std::unordered_map<const void*, std::chrono::steady_clock::time_point> s_Start;     // ponytail: per renderer, never erased (a measurement hook)
            const auto It = s_Start.try_emplace(this, std::chrono::steady_clock::now()).first;
            const bool bHits   = s_Ablate.find("hits") != std::string::npos;
            const bool bFrozen = (bHits || s_Ablate.find("freeze") != std::string::npos) && MsSince(It->second) > s_FreezeS * 1000.0;
            W.m_Setup.m_bAblateHitRead = bFrozen && bHits;
            W.m_Setup.m_bAblateFreeze  = bFrozen;
            W.m_Setup.m_bAblateStats   = bFrozen && s_Ablate.find("stats") != std::string::npos;     // ("freeze,stats": the diagnostic run, never timed)
            W.m_Setup.m_bAblateFallback = bFrozen && s_Ablate.find("fallback") != std::string::npos; // ("freeze,fallback": only the unsupported hits' walk stubbed)
        }
        m_GIStats = W.Update(L, Eye, bGPU && Scene.m_GI.m_GPUBudgetMs > 0 ? m_pGITrace->BatchRays() : Scene.m_GI.m_RayBudget, std::uint32_t(m_GIRender));     // (no GPU budget: the scene's rays, a slice each)
        if( bGPU )
        {
            const auto TSubmit = std::chrono::steady_clock::now();
            if( auto* pT = GITimes(this, m_Device) ) Window.WriteTimestamp(pT->m_T, 0);
            m_pGITrace->Submit(Window, W, L, GIFrame, Scene.m_GI.m_GPUBudgetMs, Eye);
            // The link solve, on by default (Tomas 2026-10-24: it is what settles the light): the links built on the GPU once the rays settled (30
            // frames), then solved every frame and the rays stop (moving objects still get them). XLION_GI_LINKS=<rays>[,settle frames[,surfels a
            // frame]] changes it; XLION_GI_LINKS=0 turns it off
            static const std::array<int, 3> s_Links = [] { std::array<int, 3> A{ 64, 30, 2048 }; if (const char* p = std::getenv("XLION_GI_LINKS")) (void)std::sscanf(p, "%d,%d,%d", &A[0], &A[1], &A[2]); return A; }();
            if( s_Links[0] > 0 ) m_pGITrace->PumpLinks(Window, W, L, GIFrame, s_Links[0], s_Links[1], s_Links[2]);
            // The probe grid (skinned meshes' GI): 32 x 16 x 32 probes a meter apart around the eye, 1024 updated a frame
            static const bool s_bNoProbes = std::getenv("XLION_GI_NO_PROBES") != nullptr;     // (measurement: A/B)
            if( Scene.m_GI.m_bProbes && !s_bNoProbes ) (void)m_pGITrace->UpdateProbes(Window, Eye, Scene.m_GI.m_ProbeSpacing, { 32, 16, 32 }, 1024, GIFrame);
            if( auto* pT = GITimes(this, m_Device) ) Window.WriteTimestamp(pT->m_T, 1);
            if( (m_bGIToCPU && !m_bGIToCPUAsked) || (!m_GIDumpPath.empty() && !m_bGIDumpAsked) )     // the GPU's light back into the world (the next Collect)
                if( m_pGITrace->ReadBack(Window, GIFrame) ) { if( m_bGIToCPU ) m_bGIToCPUAsked = true; if( !m_GIDumpPath.empty() ) m_bGIDumpAsked = true; }
            m_GIStats.m_GPUTraceMs = m_pGITrace->LastMs(); m_GIStats.m_GPUPacks = m_pGITrace->Packs(); m_GIStats.m_GPUCpuMs = CollectMs + MsSince(TSubmit);
            m_GIStats.m_GPUSliceRays = m_pGITrace->LastRays();
        }
        const auto TUpload = std::chrono::steady_clock::now();

        // Phase 3 (documentation/gi_dynamic_master_plan.md): every frame slot of the rings keeps every instance at the same place (its mean,
        // then its surfels); a slot is written only what changed since it was last written (each instance keeps, per slot, the surfels
        // it has not had, or "all of it"). Each frame's slot is then one complete version of the light, never mixed with another frame's:
        // the GPU reads slot f while the CPU brings slot f + 1 up to date
        const auto PackMoments = [&](std::uint32_t* pOut, const gi::moments& M)
        {
            for( int t = 0; t < gi::moment_texels_v; ++t ) pOut[t] = details::Half(M[2 * t]) | (std::uint32_t(details::Half(M[2 * t + 1])) << 16);
        };
        const auto TMoments = std::chrono::steady_clock::now();
        if( !bGPU ) for( const auto& [i, s] : W.LastUpdated() )     // (on the GPU: Show writes them)
        {
            if( std::size_t(i) >= m_GITrack.size() ) continue;
            auto& T = m_GITrack[i];
            const auto& M = W.InstanceMoments(i);
            if( T.m_MomentsReset == W.MomentsReset(i) && T.m_Moments.size() == M.size() * gi::moment_texels_v )
            {
                PackMoments(T.m_Moments.data() + std::size_t(s) * gi::moment_texels_v, M[s]);
                for( int f = 0; f < ring_frames_v; ++f ) if( !T.m_bMomFull[f] ) T.m_MomPending[f].push_back(s);
            }
        }
        m_GIStats.m_MomentsMs += MsSince(TMoments);

        // The places: a new instance, or one with a new size (a level switch), gets one, and every slot all of it. When the region is too
        // fragmented, every place is given again (all of it to every slot). ponytail: first fit, free blocks not merged (the compaction does)
        const auto Alloc = [&](int Size) -> int
        {
            for( auto It = m_GIFree.begin(); It != m_GIFree.end(); ++It )
                if( It->second >= Size ) { const int At = It->first; It->first += Size; It->second -= Size; if( It->second == 0 ) m_GIFree.erase(It); return At; }
            if( m_GIAllocTop + Size > m_GIRegion ) return -1;
            const int At = m_GIAllocTop; m_GIAllocTop += Size; return At;
        };
        const auto Place = [&](gi_track& T, int n) -> bool
        {
            if( T.m_Offset >= 0 ) m_GIFree.push_back({ T.m_Offset, T.m_Size + 1 });
            T.m_Size = n; T.m_Offset = Alloc(n + 1);
            T.m_bFull.fill(true); T.m_bMomFull.fill(true);
            for( auto& P : T.m_Pending ) P.clear();
            for( auto& P : T.m_MomPending ) P.clear();
            return T.m_Offset >= 0;
        };
        // A device-local (GPU_READ) buffer is made from its data: zeros of its size (the host-visible one: none)
        std::vector<std::uint32_t> ZeroWords;
        const auto Zeros = [&](xgpu::buffer::setup::usage U, std::size_t nWords) -> std::optional<void*> { if( U != xgpu::buffer::setup::usage::GPU_READ ) return std::nullopt; ZeroWords.assign(nWords, 0u); return static_cast<void*>(ZeroWords.data()); };
        const auto ForEachGIId = [&](auto&& Fn) { for( const auto& [K, Id] : m_GIIds ) Fn(Id); for( const auto& [K, Id] : m_GISecIds ) Fn(Id); };
        bool bCompact = false;
        ForEachGIId([&](int Id)
        {
            auto& T = m_GITrack[Id];
            const int n = int(W.InstanceLight(Id).size());
            if( (T.m_Offset < 0 || T.m_Size != n) && !Place(T, n) ) bCompact = true;
        });
        if( bCompact )
        {
            // The region grown to what is drawn (a power of 2, 2^22 at most: 64 MB of light, 256 MB of moments host-visible) - its rings made
            // again, every place given again (ponytail: never shrunk)
            std::size_t Need = 0;
            ForEachGIId([&](int Id) { Need += W.InstanceLight(Id).size() + 1; });
            if( Need > std::size_t(m_GIRegion) && m_GIRegion < (1 << 22) )
            {
                int Region = m_GIRegion;
                while( std::size_t(Region) < Need && Region < (1 << 22) ) Region <<= 1;
                xgpu::buffer Light, Moments;
                const auto Usage = m_bGIRingsLocal ? xgpu::buffer::setup::usage::GPU_READ : xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ;
                if( !m_Device.Create(Light, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = Usage, .m_EntryByteSize = 16, .m_EntryCount = ring_frames_v * Region, .m_pData = Zeros(Usage, std::size_t(ring_frames_v) * Region * 4) })
                 && !m_Device.Create(Moments, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = Usage, .m_EntryByteSize = 4, .m_EntryCount = ring_frames_v * Region * gi::moment_texels_v, .m_pData = Zeros(Usage, std::size_t(ring_frames_v) * Region * gi::moment_texels_v) }) )
                {
                    // (xgpu::buffer has no move: the handles swapped; the old ones' destructors queue them in the device's deferred destroy)
                    m_GILightRing.m_Private.swap(Light.m_Private); m_GIMomentRing.m_Private.swap(Moments.m_Private);
                    m_GIRegion = Region;
                    xgpu::buffer L2, M2;                                    // the other kind: made again at its next switch, at this size
                    m_GILightOther.m_Private.swap(L2.m_Private); m_GIMomentOther.m_Private.swap(M2.m_Private);
                    m_GIOtherRegion = 0;
                }
            }
            m_GIFree.clear(); m_GIAllocTop = 0;
            for( auto& T : m_GITrack ) T.m_Offset = -1;
            ForEachGIId([&](int Id) { (void)Place(m_GITrack[Id], int(W.InstanceLight(Id).size())); });
        }
        ForEachGIId([&](int Id)
        {
            auto& T = m_GITrack[Id];
            if( bGPU ) { T.m_bFull.fill(true); T.m_bMomFull.fill(true); T.m_MomentsReset = ~0u; return; }     // (the GPU writes them; back on the CPU: all of them again)
            const auto& M = W.InstanceMoments(Id);
            if( T.m_MomentsReset != W.MomentsReset(Id) || T.m_Moments.size() != M.size() * gi::moment_texels_v )     // all of them again
            {
                T.m_Moments.resize(M.size() * gi::moment_texels_v);
                for( std::size_t k = 0; k < M.size(); ++k ) PackMoments(T.m_Moments.data() + k * gi::moment_texels_v, M[k]);
                T.m_MomentsReset = W.MomentsReset(Id);
                T.m_bMomFull.fill(true);
            }
            if( W.ShownFull(Id) ) T.m_bFull.fill(true);
            else for( int f = 0; f < ring_frames_v; ++f ) if( !T.m_bFull[f] ) T.m_Pending[f].insert(T.m_Pending[f].end(), W.ShownChanged(Id).begin(), W.ShownChanged(Id).end());
        });

        // The rings of the kind this frame writes: device-local while the GPU traces (Show writes them), host-visible for the CPU world
        if( bGPU != m_bGIRingsLocal )
        {
            if( m_GIOtherRegion != m_GIRegion )
            {
                const auto Usage = bGPU ? xgpu::buffer::setup::usage::GPU_READ : xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ;
                xgpu::buffer Light, Moments;
                if( !m_Device.Create(Light, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = Usage, .m_EntryByteSize = 16, .m_EntryCount = ring_frames_v * m_GIRegion, .m_pData = Zeros(Usage, std::size_t(ring_frames_v) * m_GIRegion * 4) })
                 && !m_Device.Create(Moments, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = Usage, .m_EntryByteSize = 4, .m_EntryCount = ring_frames_v * m_GIRegion * gi::moment_texels_v, .m_pData = Zeros(Usage, std::size_t(ring_frames_v) * m_GIRegion * gi::moment_texels_v) }) )
                {
                    m_GILightOther.m_Private.swap(Light.m_Private); m_GIMomentOther.m_Private.swap(Moments.m_Private);
                    m_GIOtherRegion = m_GIRegion;
                }
            }
            if( m_GIOtherRegion == m_GIRegion )
            {
                m_GILightRing.m_Private.swap(m_GILightOther.m_Private); m_GIMomentRing.m_Private.swap(m_GIMomentOther.m_Private);
                m_bGIRingsLocal = bGPU;
                ForEachGIId([&](int Id) { m_GITrack[Id].m_bFull.fill(true); m_GITrack[Id].m_bMomFull.fill(true); });     // (the CPU's: all of it again)
            }
        }
        const bool bShowOnGPU = bGPU && m_bGIRingsLocal;

        // This frame's slot: what each drawn instance has not had
        const std::uint64_t Frame = m_Device.getFrameIndex();
        const int Slot = int(Frame % ring_frames_v);
        const int Base = Slot * m_GIRegion;
        // scene::gi_debug in the shaders' bits (xlionrender_surfel_gi.glsl GIDebugCode): 1..3 at bit 9, 4 and 8 at bits 12 and 13
        const std::uint32_t GIDebugBits = ((std::uint32_t(Scene.m_GI.m_Debug) & 3u) << 9) | ((std::uint32_t(Scene.m_GI.m_Debug) & 0xCu) << 10)
                                        | (Scene.m_GI.m_bMoments ? 0u : 0x4000u);                   // gi_no_moments_v
        const auto TRing = std::chrono::steady_clock::now();
        std::size_t Entries = 0, MomentUints = 0;
        const auto RingWrite = [&](void* pMemory)     // a uvec4 a surfel: 3 colors in RGB9E5, the validity's bits (null: the rings are local, Show writes them)
        {
            auto* pOut = static_cast<std::array<std::uint32_t, 4>*>(pMemory);
            const auto WriteOne = [&](int Id, std::pair<std::uint32_t, std::uint32_t>& DrawGI)
            {
                auto& T = m_GITrack[Id];
                if( T.m_Offset < 0 ) { ++m_GIStats.m_nDropped; return; }  // ponytail: no GI for what does not fit
                const auto& E = W.BasisLight(Id);              // shown: the HL2 basis, spatially filtered (the solver keeps its own)
                const auto& V = W.InstanceValid(Id);
                const auto Write = [&](std::size_t k)
                {
                    pOut[std::size_t(T.m_Offset) + 1 + k] = { gi::PackRGB9E5(E[k][0]), gi::PackRGB9E5(E[k][1]), gi::PackRGB9E5(E[k][2]), std::bit_cast<std::uint32_t>(V[k]) };
                };
                // The slot before its surfels: its mean light, what a pixel no read supports shows (SurfelGIMean: the solver's policy too,
                // gi::world::Unsupported). Then its surfels
                const auto& Mean = W.MeanLight(Id);
                if( pOut ) { const std::uint32_t m = gi::PackRGB9E5(Mean); pOut[std::size_t(T.m_Offset)] = { m, m, m, std::bit_cast<std::uint32_t>(1.0f) }; }
                ++Entries;
                if( bGPU ) {}                                       // (its surfels: Show)
                else if( T.m_bFull[Slot] ) { for( std::size_t k = 0; k < E.size(); ++k ) Write(k); Entries += E.size(); T.m_bFull[Slot] = false; }
                else { for( const int k : T.m_Pending[Slot] ) if( std::size_t(k) < E.size() ) Write(std::size_t(k)); Entries += T.m_Pending[Slot].size(); }
                T.m_Pending[Slot].clear();
                DrawGI = { std::uint32_t(Base + T.m_Offset + 1), std::uint32_t(W.Level(Id)) | (Scene.m_GI.m_bPerPixel ? 0x100u : 0u) | GIDebugBits };     // xlionrender_surfel_gi.glsl gi_per_pixel_v, gi_debug_shift_v
                {   // measurement only (XLION_GI_ABLATE, updateGI above): the pixel read's variants
                    static const std::string s_Px = [] { const char* p = std::getenv("XLION_GI_ABLATE"); return std::string(p ? p : ""); }();
                    if( s_Px.find("pixel_vertex") != std::string::npos )    DrawGI.second &= ~0x100u;
                    if( s_Px.find("pixel_nomoments") != std::string::npos ) DrawGI.second |= 0x4000u;
                    if( s_Px.find("pixel_nolocal") != std::string::npos )   DrawGI.second |= 0x8000u;     // (mb_standard_pbr.frag: no SurfelGILocalMean)
                    if( s_Px.find("pixel_nocorners") != std::string::npos ) DrawGI.second |= 0x01000000u; // (the per-pixel read without its corner weights)
                    if( s_Px.find("pixel_none") != std::string::npos )      DrawGI = { 0u, 0xFFFFFFFFu };
                    static const std::uint32_t s_Cap = [] { const char* p = std::getenv("XLION_GI_PIXEL_CAP"); return p ? std::uint32_t(std::clamp(std::atoi(p), 0, 255)) : 0u; }();
                    if( s_Cap && DrawGI.second != 0xFFFFFFFFu ) DrawGI.second |= s_Cap << 16;     // (xlionrender_surfel_gi.glsl GIPixelCap)
                }
            };
            for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
            {
                if( DrawId[i] >= 0 ) WriteOne(DrawId[i], m_DrawGI[i]);
                if( DrawSecIds[i].empty() ) continue;
                m_DrawGISec[i].assign(DrawSecIds[i].size(), { 0u, 0xFFFFFFFFu });
                for( std::size_t k = 0; k < DrawSecIds[i].size(); ++k ) WriteOne(DrawSecIds[i][k], m_DrawGISec[i][k]);
            }
        };
        if( bShowOnGPU ) RingWrite(nullptr);
        else (void)m_GILightRing.MemoryMapFrameRegion(std::size_t(Base), std::size_t(m_GIRegion), RingWrite);
        m_GIStats.m_RingCopyMs = MsSince(TRing);
        const auto TMomentsCopy = std::chrono::steady_clock::now();

        // The surfels' depth moments at the same index (gi::moment_texels_v per surfel, each a mean and a mean^2 in two 16-bit floats)
        if( !bGPU ) (void)m_GIMomentRing.MemoryMapFrameRegion(std::size_t(Base) * gi::moment_texels_v, std::size_t(m_GIRegion) * gi::moment_texels_v, [&](void* pMemory)
        {
            auto* pOut = static_cast<std::uint32_t*>(pMemory);
            ForEachGIId([&](int Id)
            {
                auto& T = m_GITrack[Id];
                if( T.m_Offset < 0 ) return;
                std::uint32_t* pAt = pOut + (std::size_t(T.m_Offset) + 1) * gi::moment_texels_v;
                if( T.m_bMomFull[Slot] ) { std::memcpy(pAt, T.m_Moments.data(), T.m_Moments.size() * sizeof(std::uint32_t)); MomentUints += T.m_Moments.size(); T.m_bMomFull[Slot] = false; }
                else for( const int k : T.m_MomPending[Slot] )
                {
                    if( std::size_t(k + 1) * gi::moment_texels_v > T.m_Moments.size() ) continue;
                    std::memcpy(pAt + std::size_t(k) * gi::moment_texels_v, T.m_Moments.data() + std::size_t(k) * gi::moment_texels_v, gi::moment_texels_v * sizeof(std::uint32_t));
                    MomentUints += gi::moment_texels_v;
                }
                T.m_MomPending[Slot].clear();
            });
        });
        // G4: on the GPU the shown light of every drawn instance into this frame's region (its places: the means written above)
        if( bGPU )
        {
            // per instance { its place (-1: not drawn), its mean light in RGB9E5 (written before its surfels when the rings are local) }
            std::vector<int> Places(2 * std::size_t(W.InstanceCount()), -1);
            const auto PlaceOf = [&](int Id) { if( Id >= 0 && m_GITrack[Id].m_Offset >= 0 ) { Places[2 * std::size_t(Id)] = m_GITrack[Id].m_Offset; Places[2 * std::size_t(Id) + 1] = bShowOnGPU ? int(gi::PackRGB9E5(W.MeanLight(Id))) : -1; } };
            for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
            {
                PlaceOf(DrawId[i]);
                for( const int Id : DrawSecIds[i] ) PlaceOf(Id);
            }
            if( auto* pT = GITimes(this, m_Device) ) Window.WriteTimestamp(pT->m_T, 2);
            (void)m_pGITrace->Show(Window, Places, m_GILightRing, m_GIMomentRing, Base, W.m_Setup.m_FilterPasses, W.m_Setup.m_Saturation);
            if( auto* pT = GITimes(this, m_Device) ) Window.WriteTimestamp(pT->m_T, 3);
        }
        m_GIStats.m_UploadBytes += Entries * 3 * sizeof(xmath::fvec4) + MomentUints * sizeof(std::uint32_t);
        m_GIStats.m_MomentsMs += MsSince(TMomentsCopy);
        m_GIStats.m_UploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - TUpload).count();
        m_GIStats.m_UpdateGIMs = MsSince(TGI);
        if( m_GIStart.time_since_epoch().count() == 0 ) m_GIStart = TGI;
        m_GIStats.m_TimeMs = std::chrono::duration<double, std::milli>(TGI - m_GIStart).count();
        m_GIOverheadMs += (std::max(m_GIStats.m_UpdateGIMs - m_GIStats.m_TotalMs, 0.0) - m_GIOverheadMs) * 0.2;      // (the outside part, slowly)
        m_GIStats.m_LoadMs = m_GILoadMsAcc; m_GILoadMsAcc = 0;
        {   gpu_times T; const bool bT = getGPUTimes(T);
            m_GIFrames.push_back({ m_Device.getFrameIndex(), m_GIStats, bT ? T.m_SceneMs : -1.0 });     // (the scene pass' GPU time: the per-pixel GI read is in it)
            static const bool s_bTimes = std::getenv("XLION_GPU_TIMES") != nullptr;     // measurement: the GPU passes, every 120 frames
            if( s_bTimes && bT && m_Device.getFrameIndex() % 120 == 0 ) std::printf("GPUTIMES shadows %.2f scene %.2f tonemap %.2f GItrace %.2f ms; CPU Render %.2f ms (instances %.2f, shadows %.2f)\n", T.m_ShadowMs, T.m_SceneMs, T.m_ToneMapMs, m_GIStats.m_GPUTraceMs, m_CPUTimes.m_RenderMs, m_CPUTimes.m_InstancesMs, m_CPUTimes.m_ShadowsMs), std::fflush(stdout);
        }
        if( m_GIFrames.size() > gi_frame_log_v ) m_GIFrames.erase(m_GIFrames.begin(), m_GIFrames.begin() + std::ptrdiff_t(m_GIFrames.size() - gi_frame_log_v));
    }

    //------------------------------------------------------------------------------------------------

    // The sun's virtual pages: this frame's page table in the cell ring, written once a frame - first in the frame's region (the same
    // reset as writeLocalLights', which then appends after it). Returns its first uint
    int instance::vsmTableBase( void ) noexcept
    {
        const std::uint64_t Frame = m_Device.getFrameIndex();
        if( Frame != m_LightFrame ) { m_LightFrame = Frame; m_LightRingUsed = 0; m_CellRingUsed = 0; }
        if( m_VSMTableFrame == Frame ) return m_VSMTableBase;
        m_VSM.BuildTable(m_VSMTable);
        const int n = int(m_VSMTable.size());
        m_VSMTableBase = int(Frame % ring_frames_v) * max_cell_uints_v + m_CellRingUsed;
        (void)m_CellRing.MemoryMapFrameRegion(m_VSMTableBase, n, [&](void* p) { std::memcpy(p, m_VSMTable.data(), std::size_t(n) * sizeof(std::uint32_t)); });   // (this frame's slot: no wait for the frame in flight)
        m_CellRingUsed += n;
        m_VSMTableFrame = Frame;
        return m_VSMTableBase;
    }

    // The sun's virtual pages: the request pass over the main view's depth (shaders/xlionrender_vsm_request_comp.glsl), after the scene
    // pass. Its bits and counters go to this frame's ring slot, read when the slot comes round (renderShadows), with these windows
    void instance::vsmRequestPass( xgpu::window& Window, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye ) noexcept
    {
        const int Slot = int(m_Device.getFrameIndex() % ring_frames_v), nBits = vsmBitUints(), nReq = nBits + 32;
        const auto& VS = m_VSM.getSetup();
        details::vsm_frame F;
        F.m_C2W = W2C.Inverse();
        F.m_Eye = xmath::fvec4(Eye, 1);
        F.m_LX  = xmath::fvec4(m_VSM.m_X, 0); F.m_LY = xmath::fvec4(m_VSM.m_Y, 0); F.m_LZ = xmath::fvec4(m_VSM.m_Z, 0);
        F.m_P0  = { m_VSM.Level0Texel(), float(VS.m_Levels), float(VS.m_Window), m_VSM.ResolutionBias() };
        for( int i = 0; i < shadows::vsm::max_levels_v; ++i )
        {
            const auto& Lv = m_VSM.m_Levels[i];
            F.m_Levels[i] = { float(Lv.m_OriginX), float(Lv.m_OriginY), Lv.m_PageSize, Lv.m_DepthCenter };
        }
        (void)m_VSMFrames.MemoryMapFrameRegion(Slot, 1, [&](void* p) { std::memcpy(p, &F, sizeof(F)); });                                           // (this frame's slot: no wait)
        const xgpu::buffer::copy Z{ 0, 0, nReq };
        Window.CopyBuffer(m_VSMZeros, m_VSMReq, std::span{ &Z, 1 });
        Window.WriteTimestamp(m_VSMTimes, 0);
        const std::uint32_t Push[8] = { std::uint32_t(Slot), 0u, std::uint32_t(vsmTableBase()), std::uint32_t(View.m_Width), std::uint32_t(View.m_Height), std::uint32_t(nBits), 0u, 0u };
        auto Buffers  = std::array<xgpu::buffer*, 3>{ &m_VSMFrames, &m_VSMReq, &m_CellRing };
        auto Textures = std::array<const xgpu::texture*, 1>{ &View.m_Depth };
        Window.Dispatch(m_VSMRequest, Buffers, Textures, Push, sizeof(Push), (View.m_Width + 7) / 8, (View.m_Height + 7) / 8);
        Window.WriteTimestamp(m_VSMTimes, 1);
        const xgpu::buffer::copy C{ 0, Slot * nReq, nReq };
        Window.CopyBuffer(m_VSMReq, m_VSMReqHost, std::span{ &C, 1 });
        m_VSMSnap[Slot] = m_VSM.Snapshot();
    }

    // The local lights and the captured reflection probes for this render: into this frame's region of the light buffer, and in
    // modes CELLS / SAMPLED their world cells into the cell buffer (features/lights/xlionrender_light_cells.h). Fills the lighting
    // block's local and probe parts.
    void instance::writeLocalLights( const scene& Scene, const xmath::fvec3& Eye, details::ubo_lighting& Light ) noexcept
    {
        const std::uint64_t Frame = m_Device.getFrameIndex();
        if( Frame != m_LightFrame ) { m_LightFrame = Frame; m_LightRingUsed = 0; m_CellRingUsed = 0; }

        const int nLights = std::min({ int(Scene.m_LocalLights.size()), max_lights_v, light_region_v - m_LightRingUsed });
        const int LightBase = int(Frame % ring_frames_v) * light_region_v + m_LightRingUsed;
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
                    Out.m_Shadow      = { float(sunAtlasWidth() + Map.m_X), float(Map.m_Y), float(Map.m_Size), bSpot ? shadows::SpotTanHalf(LL) : 1.0f };
                }
            });
        m_LightRingUsed += std::max(nLights, 0);

        // The captured probes, innermost (smallest box) first: the shader lets the first ones that cover a point win
        m_ProbeOrder.clear();
        for( int i = 0; i < int(m_ProbeStates.size()) && i < int(Scene.m_Probes.size()); ++i ) if( m_ProbeStates[i].m_bValid ) m_ProbeOrder.push_back(i);
        const auto Volume = [&](int i) { const auto D = Scene.m_Probes[i].m_BoxMax - Scene.m_Probes[i].m_BoxMin; return D.m_X * D.m_Y * D.m_Z; };
        std::sort(m_ProbeOrder.begin(), m_ProbeOrder.end(), [&](int a, int b) { return Volume(a) < Volume(b); });
        const int nProbes   = std::min(int(m_ProbeOrder.size()), light_region_v - m_LightRingUsed);
        const int ProbeBase = int(Frame % ring_frames_v) * light_region_v + m_LightRingUsed;
        if( nProbes > 0 )
            (void)m_LightRing.MemoryMap(ProbeBase, nProbes, [&](void* pMemory)
            {
                auto* pOut = static_cast<details::probe_data*>(pMemory);
                for( int k = 0; k < nProbes; ++k )
                {
                    const auto& Pr   = Scene.m_Probes[m_ProbeOrder[k]];
                    const int   Slot = m_ProbeStates[m_ProbeOrder[k]].m_Slot;
                    pOut[k] = { xmath::fvec4(Pr.m_Position - Eye, Pr.m_Blend), xmath::fvec4(Pr.m_BoxMin - Eye, 0), xmath::fvec4(Pr.m_BoxMax - Eye, 0)
                              , { float(Slot % probe_slots_x_v * 384), float(Slot / probe_slots_x_v * 256), 0, 0 } };
                }
            });
        m_LightRingUsed += std::max(nProbes, 0);

        auto Mode = m_LightMode;
        int  CellBase = 0;
        m_LightBuildMs = 0;
        int  ProbeCells = -1;
        if( Mode != light_mode::EXACT && (nLights > 0 || nProbes > 0) )
        {
            const auto T0 = std::chrono::steady_clock::now();
            m_CellLights.resize(std::max(nLights, 0));
            for( int i = 0; i < nLights; ++i )
            {
                const auto& LL = Scene.m_LocalLights[i];
                const float Lum = 0.2126f * LL.m_Color.m_X + 0.7152f * LL.m_Color.m_Y + 0.0722f * LL.m_Color.m_Z;
                const bool  bSpot = LL.m_Type == local_light::type::SPOT;
                m_CellLights[i] = { LL.m_Position, LL.m_Radius, std::max(Lum * LL.m_Intensity, 1e-6f), LL.m_Direction.NormalizeCopy(), bSpot ? LL.m_OuterAngle : -1.0f };
            }
            m_LightCells.Build(m_CellLights, Eye, m_LightCellSize, Mode == light_mode::SAMPLED ? m_LightHead : 0);
            if( nProbes > 0 )
            {
                m_ProbeBoxes.resize(nProbes);
                for( int k = 0; k < nProbes; ++k ) m_ProbeBoxes[k] = { Scene.m_Probes[m_ProbeOrder[k]].m_BoxMin, Scene.m_Probes[m_ProbeOrder[k]].m_BoxMax };
                ProbeCells = int(m_LightCells.AddBoxes(m_ProbeBoxes));
            }
            m_LightBuildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();

            const int nUints = int(m_LightCells.m_Data.size());
            if( nUints > max_cell_uints_v - m_CellRingUsed ) { Mode = light_mode::EXACT; ProbeCells = -1; }    // ponytail: too many entries for the buffer
            else
            {
                CellBase = int(Frame % ring_frames_v) * max_cell_uints_v + m_CellRingUsed;
                (void)m_CellRing.MemoryMap(CellBase, nUints, [&](void* pMemory) { std::memcpy(pMemory, m_LightCells.m_Data.data(), nUints * sizeof(std::uint32_t)); });
                m_CellRingUsed += nUints;
            }
        }

        m_LightModeUsed  = Mode;
        m_LightCellUints = Mode == light_mode::EXACT ? 0 : int(m_LightCells.m_Data.size());
        m_LightStats.m_nLights        = nLights;
        m_LightStats.m_nDropped       = int(Scene.m_LocalLights.size()) - nLights;
        m_LightStats.m_nOverflowCells = 0;
        if( Mode != light_mode::EXACT ) for( const auto n : m_LightCells.m_Count ) m_LightStats.m_nOverflowCells += n >= light_cells::overflow_v;
        Light.m_LocalCount = { float(nLights), float(int(Mode)), float(m_LightSeed++ & 0xffffff), float(m_LightHead) };
        Light.m_LocalBase  = { float(LightBase), float(CellBase), m_LightCellSize, float(m_LightTailSamples) };
        Light.m_ProbeCount = { float(std::max(nProbes, 0)), float(ProbeBase), float(ProbeCells), 0 };
        for( int L = 0; L < light_cells::levels_v; ++L )
            Light.m_CellOrigin[L] = { float(m_LightCells.m_Origin[L][0]), float(m_LightCells.m_Origin[L][1]), float(m_LightCells.m_Origin[L][2]), 0 };
    }

    //------------------------------------------------------------------------------------------------

    // The environment probe (scene::m_Sky): the DFG table once, then the sky into the octahedral source and its prefiltered atlas
    // whenever the sky changes. Off (m_Intensity 0): the flat ambient lights.
    void instance::updateEnvironment( xgpu::window& Window, const scene& Scene ) noexcept
    {
        const auto& Sky = Scene.m_Sky;
        m_bEnvOn        = Sky.m_Intensity > 0;
        m_EnvIntensity  = Sky.m_Intensity;

        // The cubemap's mean color (the GI's sky): its roughest prefiltered level, read back once the GPU is done with it
        if( m_EnvMeanFrame && m_Device.getFrameIndex() >= m_EnvMeanFrame + ring_frames_v + 1 )
        {
            m_EnvMeanFrame = 0;
            std::vector<std::byte> Raw;
            int W = 0, H = 0;
            if( m_Device.ReadTexture(m_EnvAtlas, Raw, W, H) == nullptr && W >= 376 && H >= 136 )
            {
                const auto* P = reinterpret_cast<const std::uint16_t*>(Raw.data());
                xmath::fvec3 S{ 0, 0, 0 };
                for( int y = 128; y < 136; ++y ) for( int x = 368; x < 376; ++x )      // level 5: (368, 128) 8x8 (prefilterInto)
                {
                    const auto* t = P + 4 * (std::size_t(y) * W + x);
                    S = S + xmath::fvec3(details::HalfToFloat(t[0]), details::HalfToFloat(t[1]), details::HalfToFloat(t[2]));
                }
                m_EnvMean = S * (1.0f / 64.0f);
                for( int j = 0; j < 4; ++j ) for( int i = 0; i < 4; ++i )              // the level's inner texels (kEnvBorder 2 of its 8)
                {
                    const auto* t = P + 4 * (std::size_t(130 + j) * W + 370 + i);
                    m_EnvOct[std::size_t(j * 4 + i)] = xmath::fvec3(details::HalfToFloat(t[0]), details::HalfToFloat(t[1]), details::HalfToFloat(t[2]));
                }
                m_bEnvOct = true;
            }
        }
        if( !m_bEnvOn && m_bEnvDFGReady ) return;        // (the first render builds everything once: bound textures must have been rendered)

        if( !m_bEnvDFGReady )
        {
            auto Cmd = Window.StartRenderPass(m_EnvDFGPass);
            Cmd.setPipelineInstance(m_EnvDFGInstance);
            Cmd.setPushConstants(details::push_env_dfg{ { float(env_dfg_size_v), 512, 0, 0 } });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
            m_bEnvDFGReady = true;
        }

        const float Key[] = { Sky.m_Zenith.m_X, Sky.m_Zenith.m_Y, Sky.m_Zenith.m_Z, Sky.m_Horizon.m_X, Sky.m_Horizon.m_Y, Sky.m_Horizon.m_Z, Sky.m_Ground.m_X, Sky.m_Ground.m_Y, Sky.m_Ground.m_Z };
        const std::uint64_t K = (Sky.m_pCubemap ? Hash(1469598103934665603ull, &Sky.m_pCubemap, sizeof(Sky.m_pCubemap)) : Hash(1469598103934665603ull, Key, sizeof(Key))) | 1;
        if( K == m_EnvKey ) return;
        m_EnvKey = K;
        ++m_nEnvBuilds;

        if( Sky.m_pCubemap )
        {
            if( m_pEnvCubemap != Sky.m_pCubemap )       // its instance binds the texture
            {
                m_Device.Destroy(std::move(m_EnvCubemapInstance));
                auto Bind = std::array{ xgpu::pipeline_instance::sampler_binding{ *const_cast<xgpu::texture*>(Sky.m_pCubemap) } };
                m_pEnvCubemap = m_Device.Create(m_EnvCubemapInstance, { .m_PipeLine = m_EnvCubemapPipeline, .m_SamplersBindings = Bind }) == nullptr ? Sky.m_pCubemap : nullptr;
            }
            if( m_pEnvCubemap )
            {
                {
                    auto Cmd = Window.StartRenderPass(m_EnvSourcePass[0]);        // (its pass ends with the scope, before the pyramid's)
                    Cmd.setPipelineInstance(m_EnvCubemapInstance);
                    Cmd.setPushConstants(details::push_env_cube_to_oct{ { float(env_source_size_v), 0, 0, 0 } });
                    Cmd.setBuffer(m_FullscreenVB);
                    Cmd.setBuffer(m_FullscreenIB);
                    Cmd.Draw(3);
                }
                prefilterInto(Window, m_EnvAtlasPass, 0, 0);
                m_EnvMeanFrame = std::max<std::uint64_t>(m_Device.getFrameIndex(), 1);
                return;
            }
        }
        {
            auto Cmd = Window.StartRenderPass(m_EnvSourcePass[0]);
            Cmd.setPipelineInstance(m_EnvSkyInstance);
            Cmd.setPushConstants(details::push_env_sky{ { 0, 1, 0, 2 }, { 0, 0, 0, 0 }                     // no sun disc: the sun is the direct light
                                                      , xmath::fvec4(Sky.m_Zenith, 0), xmath::fvec4(Sky.m_Horizon, 0), xmath::fvec4(Sky.m_Ground, float(env_source_size_v)) });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
        }
        prefilterInto(Window, m_EnvAtlasPass, 0, 0);
    }

    // The environment source's pyramid (level 0 written by the caller) and its prefilter into Pass' atlas at (OriginX, OriginY):
    // the 6 roughness levels of the sky layout (xlionrender_environment_atlas.glsl)
    void instance::prefilterInto( xgpu::window& Window, xgpu::renderpass& Pass, float OriginX, float OriginY ) noexcept
    {
        for( int k = 1; k < env_source_levels_v; ++k )
        {
            auto Cmd = Window.StartRenderPass(m_EnvSourcePass[k]);
            Cmd.setPipelineInstance(m_EnvDownsampleInstance[k]);
            Cmd.setPushConstants(details::push_env_downsample{ { float(env_source_size_v >> k), float(env_source_size_v >> (k - 1)), 0, 0 } });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
        }
        auto Cmd = Window.StartRenderPass(Pass);
        constexpr float Rects[6][3] = { { 0, 0, 256 }, { 256, 0, 128 }, { 256, 128, 64 }, { 320, 128, 32 }, { 352, 128, 16 }, { 368, 128, 8 } };     // xlionrender_environment_atlas.glsl
        for( int L = 0; L < 6; ++L )
        {
            const float X = OriginX + Rects[L][0], Y = OriginY + Rects[L][1], S = Rects[L][2];
            Cmd.setViewport(X, Y, S, S);
            Cmd.setScissor(int(X), int(Y), int(S), int(S));
            Cmd.setPipelineInstance(m_EnvPrefilterInstance);
            Cmd.setPushConstants(details::push_env_prefilter{ { X, Y, S, 0 }, { L / 5.0f, float(env_source_size_v), L == 0 ? 1.0f : 64.0f, 0 } });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
        }
    }

    //------------------------------------------------------------------------------------------------

    // Reflection probes: the nearest probe_slots_v (eye to box) are resident, each in a slot of the atlas. A changed probe (new,
    // moved, newly resident, a changed caster in its box, a changed light reaching its box, or a sun / sky / ambient change) is
    // re-captured, m_ProbeFacesPerFrame faces per frame (never captured first, then the nearest): each face a scene pass from the
    // probe's center; with its 6 faces done, the faces -> the octahedral source -> the pyramid and the prefilter into its slot.
    // A change during a capture lets it finish and captures again (a continuous change never starves it), except a moved probe.
    void instance::updateProbes( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye ) noexcept
    {
        if( !m_bProbeAtlasReady ) { auto Cmd = Window.StartRenderPass(m_ProbeAtlasClearPass); m_bProbeAtlasReady = true; }   // bound by every lit pipeline
        const int nProbes = int(Scene.m_Probes.size());
        m_ProbeStates.resize(nProbes);
        if( m_ProbeCapturing >= nProbes ) m_ProbeCapturing = -1;
        if( nProbes == 0 ) return;

        const auto Touches = [](const xmath::fbbox& A, const xmath::fbbox& B)
        {
            return A.m_Max.m_X >= B.m_Min.m_X && A.m_Min.m_X <= B.m_Max.m_X && A.m_Max.m_Y >= B.m_Min.m_Y && A.m_Min.m_Y <= B.m_Max.m_Y
                && A.m_Max.m_Z >= B.m_Min.m_Z && A.m_Min.m_Z <= B.m_Max.m_Z;
        };
        const auto BoxDistance = [&](const reflection_probe& Pr)
        {
            float D2 = 0;
            for( int a = 0; a < 3; ++a ) { const float d = std::max({ Pr.m_BoxMin[a] - Eye[a], 0.0f, Eye[a] - Pr.m_BoxMax[a] }); D2 += d * d; }
            return D2;
        };

        // Residency: the nearest probe_slots_v keep or get a slot, the others lose theirs
        // ponytail: no hysteresis - a probe at the boundary can swap in and out as the camera moves
        std::vector<bool> Wanted(nProbes, true);
        if( nProbes > probe_slots_v )
        {
            std::vector<std::pair<float, int>> Rank(nProbes);
            for( int i = 0; i < nProbes; ++i ) Rank[i] = { BoxDistance(Scene.m_Probes[i]), i };
            std::nth_element(Rank.begin(), Rank.begin() + probe_slots_v, Rank.end());
            for( int k = probe_slots_v; k < nProbes; ++k )
            {
                const int i  = Rank[k].second;
                auto&     St = m_ProbeStates[i];
                Wanted[i] = false;
                if( St.m_Slot >= 0 ) { St.m_Slot = -1; St.m_bValid = false; if( m_ProbeCapturing == i ) m_ProbeCapturing = -1; }
            }
        }
        std::array<bool, probe_slots_v> Used{};
        for( const auto& St : m_ProbeStates ) if( St.m_Slot >= 0 ) Used[St.m_Slot] = true;
        for( int i = 0, Free = 0; i < nProbes; ++i )
        {
            auto& St = m_ProbeStates[i];
            if( St.m_Slot >= 0 || !Wanted[i] ) continue;
            while( Free < probe_slots_v && Used[Free] ) ++Free;
            if( Free == probe_slots_v ) break;
            St.m_Slot = Free; Used[Free] = true; St.m_bValid = false; St.m_bDirty = true;
        }

        // The lights every capture sees (sun, sky, ambient)
        float G[] = { Scene.m_Sun.m_Direction.m_X, Scene.m_Sun.m_Direction.m_Y, Scene.m_Sun.m_Direction.m_Z, Scene.m_Sun.m_Color.m_X, Scene.m_Sun.m_Color.m_Y, Scene.m_Sun.m_Color.m_Z
                    , Scene.m_Sun.m_Intensity, Scene.m_Sun.m_Temperature, Scene.m_Sun.m_bCastShadows ? 1.0f : 0.0f, Scene.m_Ambient.m_X, Scene.m_Ambient.m_Y, Scene.m_Ambient.m_Z
                    , Scene.m_Sky.m_Zenith.m_X, Scene.m_Sky.m_Zenith.m_Y, Scene.m_Sky.m_Zenith.m_Z, Scene.m_Sky.m_Horizon.m_X, Scene.m_Sky.m_Horizon.m_Y, Scene.m_Sky.m_Horizon.m_Z
                    , Scene.m_Sky.m_Ground.m_X, Scene.m_Sky.m_Ground.m_Y, Scene.m_Sky.m_Ground.m_Z, Scene.m_Sky.m_Intensity };
        const std::uint64_t GlobalKey = Hash(1469598103934665603ull, G, sizeof(G));

        for( int i = 0; i < nProbes; ++i )
        {
            const auto& Pr = Scene.m_Probes[i];
            auto&       St = m_ProbeStates[i];
            if( St.m_Slot < 0 ) continue;
            const float K[] = { Pr.m_Position.m_X, Pr.m_Position.m_Y, Pr.m_Position.m_Z, Pr.m_BoxMin.m_X, Pr.m_BoxMin.m_Y, Pr.m_BoxMin.m_Z, Pr.m_BoxMax.m_X, Pr.m_BoxMax.m_Y, Pr.m_BoxMax.m_Z };
            const std::uint64_t Key = Hash(1469598103934665603ull, K, sizeof(K)) | 1;
            if( Key != St.m_Key ) { St.m_Key = Key; St.m_bValid = false; St.m_bDirty = true; if( m_ProbeCapturing == i ) m_ProbeNextFace = 0; }

            const xmath::fbbox Box{ Pr.m_BoxMin, Pr.m_BoxMax };
            for( const auto& C : m_FrameChanged ) if( Touches(C, Box) ) { St.m_bDirty = true; break; }

            // ponytail: every resident probe against every light (32 x lights sphere-box tests); the world cells' lists when it shows
            std::uint64_t LightKey = GlobalKey;
            for( const auto& LL : Scene.m_LocalLights )
            {
                const xmath::fbbox Sphere{ LL.m_Position - xmath::fvec3(LL.m_Radius), LL.m_Position + xmath::fvec3(LL.m_Radius) };
                if( !Touches(Sphere, Box) ) continue;
                const float L[] = { float(LL.m_Type), LL.m_Position.m_X, LL.m_Position.m_Y, LL.m_Position.m_Z, LL.m_Direction.m_X, LL.m_Direction.m_Y, LL.m_Direction.m_Z, LL.m_Radius
                                  , LL.m_InnerAngle, LL.m_OuterAngle, LL.m_Color.m_X, LL.m_Color.m_Y, LL.m_Color.m_Z, LL.m_Intensity, LL.m_bCastShadows ? 1.0f : 0.0f };
                LightKey = Hash(LightKey, L, sizeof(L));
            }
            if( LightKey != St.m_LightKey ) { if( St.m_LightKey != 0 ) St.m_bDirty = true; St.m_LightKey = LightKey; }
        }

        for( int Budget = std::max(m_ProbeFacesPerFrame, 1); Budget > 0; )
        {
            if( m_ProbeCapturing < 0 )
            {
                float Best = 1e30f;
                for( int i = 0; i < nProbes; ++i )
                {
                    if( !m_ProbeStates[i].m_bDirty || m_ProbeStates[i].m_Slot < 0 ) continue;
                    const float d = (Scene.m_Probes[i].m_Position - Eye).Length() - (m_ProbeStates[i].m_bValid ? 0.0f : 1e20f);
                    if( d < Best ) { Best = d; m_ProbeCapturing = i; }
                }
                if( m_ProbeCapturing < 0 ) return;
                m_ProbeNextFace = 0;
                m_ProbeStates[m_ProbeCapturing].m_bDirty = false;    // a change from now on marks it again
            }

            const auto& Pr = Scene.m_Probes[m_ProbeCapturing];
            for( ; Budget > 0 && m_ProbeNextFace < 6; --Budget, ++m_ProbeNextFace )
                renderScenePass(Window, Scene, m_CaptureViews[m_ProbeNextFace]
                               , shadows::FaceW2C(xmath::fvec3{ 0, 0, 0 }, shadows::CubeFaceAxis(m_ProbeNextFace), 1.0f, 0.05f, 2000.0f), Pr.m_Position, false);
            if( m_ProbeNextFace < 6 ) return;

            {
                auto Cmd = Window.StartRenderPass(m_EnvSourcePass[0]);
                Cmd.setPipelineInstance(m_CubeToOctInstance);
                Cmd.setPushConstants(details::push_env_cube_to_oct{ { float(env_source_size_v), 0, 0, 0 } });
                Cmd.setBuffer(m_FullscreenVB);
                Cmd.setBuffer(m_FullscreenIB);
                Cmd.Draw(3);
            }
            auto& St = m_ProbeStates[m_ProbeCapturing];
            prefilterInto(Window, m_ProbeAtlasPass, float(St.m_Slot % probe_slots_x_v * 384), float(St.m_Slot / probe_slots_x_v * 256));
            St.m_bValid = true;
            ++m_nProbeCaptures;
            m_ProbeCapturing = -1;
        }
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

        // The casters and what changed since the last render: new and gone casters, moved or re-materialed ones, and skinned
        // ones (old and new bounds). Draws with an instance id compare against their slot's last state (no hashing, bounds kept
        // while unchanged); draws without one go through a hash of their state.
        struct draw_caster { const draw* m_pDraw; const mesh_slot* m_pMesh; xmath::fbbox m_World; std::uint32_t m_Slot; std::uint32_t m_BoxedBase; };     // m_BoxedBase: in m_ClusterSkip (~0: none)
        std::vector<draw_caster>                         Draws;
        std::vector<xmath::fbbox>                        Changed;
        std::unordered_map<std::uint64_t, xmath::fbbox>  Now;
        Draws.reserve(Scene.m_Draws.size());
        m_DrawWorld.assign(Scene.m_Draws.size(), xmath::fbbox{});
        if( m_SlotCasters.empty() ) m_SlotCasters.resize(max_instances_v);
        const std::uint64_t Pass = ++m_ShadowPassCount;
        m_CasterSlots.clear();
        const auto MaterialsHash = [](const draw& D) { return Hash(1469598103934665603ull, D.m_Materials.data(), D.m_Materials.size() * sizeof(material_id)); };     // alpha test, displacement
        const auto WorldOf       = [](const mesh_slot& M, const draw& D) { return (M.m_bCluster ? M.m_Geom.m_BBox : M.m_BBox).TransformCopy(D.m_L2W); };
        for( std::size_t iDraw = 0; iDraw < Scene.m_Draws.size(); ++iDraw )
        {
            const auto& D     = Scene.m_Draws[iDraw];
            const auto* pMesh = findMesh(D.m_Mesh);
            if( pMesh == nullptr ) continue;
            const std::uint32_t Slot = m_DrawSlot[iDraw];
            if( pMesh->m_bCluster && Slot == instance_slots::none_v ) continue;

            // The editor's view of huge meshes. Boxes only: a cluster mesh casts no shadow (gone from the casters, its tiles re-render, and again when it is back).
            // A budget: the clusters the last scene pass drew as their boxes cast none either (the same draw: m_ClusterPlanKey); BoxedHash is part of the caster's
            // state, so the tiles under it re-render when they change, and when the budget is turned on or off
            if( pMesh->m_bCluster && m_bBoxesOnly ) { m_DrawWorld[iDraw] = WorldOf(*pMesh, D); continue; }
            std::uint32_t BoxedBase = ~0u;
            std::uint64_t BoxedHash = 0;
            if( pMesh->m_bCluster && m_FrameBudgetMs > 0 && iDraw < m_ClusterBase.size() && m_ClusterBase[iDraw] != ~0u && m_ClusterPlanKey[iDraw] == ClusterPlanKey(D)
             && std::size_t(m_ClusterBase[iDraw]) + pMesh->m_Geom.m_Clusters.size() <= m_ClusterSkip.size() )
            {
                const std::uint32_t Base = m_ClusterBase[iDraw];
                std::uint64_t H = 1469598103934665603ull;
                for( std::uint32_t c = 0, n = std::uint32_t(pMesh->m_Geom.m_Clusters.size()); c < n; ++c )
                    if( m_ClusterSkip[Base + c] == cluster_boxed_v ) { H = Hash(H, &c, sizeof(c)); BoxedBase = Base; }
                if( BoxedBase != ~0u ) BoxedHash = H;
            }

            const bool   bSkin = pMesh->m_bCluster && pMesh->m_Geom.m_Kind == geom_kind::SKIN;
            xmath::fbbox World;
            if( Slot < std::uint32_t(max_instances_v) )
            {
                auto&               C     = m_SlotCasters[Slot];
                const std::uint64_t MHash = MaterialsHash(D);
                const bool bChanged = m_DrawChanged[iDraw] || !C.m_bValid || bSkin || C.m_Mesh.m_Index != D.m_Mesh.m_Index
                                   || C.m_Mesh.m_Generation != D.m_Mesh.m_Generation || C.m_MaterialsHash != MHash || C.m_BoxedHash != BoxedHash;
                if( bChanged )
                {
                    if( C.m_bValid ) Changed.push_back(C.m_World);
                    C = { WorldOf(*pMesh, D), D.m_Mesh, MHash, BoxedHash, Pass, true };
                    Changed.push_back(C.m_World);
                }
                C.m_Seen = Pass;
                World    = C.m_World;
                m_CasterSlots.push_back(Slot);
            }
            else
            {
                std::uint64_t Key = 1469598103934665603ull;
                Key = Hash(Key, &D.m_Mesh, sizeof(D.m_Mesh));
                Key = Hash(Key, &D.m_L2W, sizeof(D.m_L2W));
                Key = Hash(Key, D.m_Materials.data(), D.m_Materials.size() * sizeof(material_id));
                Key = Hash(Key, &BoxedHash, sizeof(BoxedHash));
                World = WorldOf(*pMesh, D);
                Now.emplace(Key, World);
                const auto It = m_PrevCasters.find(Key);
                if( It == m_PrevCasters.end() || bSkin ) Changed.push_back(World);
                if( It != m_PrevCasters.end() && bSkin ) Changed.push_back(It->second);
            }
            m_DrawWorld[iDraw] = World;
            Draws.push_back({ &D, pMesh, World, Slot, BoxedBase });
        }
        for( const auto& [Key, World] : m_PrevCasters ) if( !Now.contains(Key) ) Changed.push_back(World);
        m_PrevCasters = std::move(Now);
        m_FrameChanged = Changed;                           // the probes re-capture where casters changed
        for( auto Slot : m_CasterSlotsPrev )                                // persistent casters no longer drawn
        {
            auto& C = m_SlotCasters[Slot];
            if( C.m_bValid && C.m_Seen != Pass ) { Changed.push_back(C.m_World); C.m_bValid = false; }
        }
        std::swap(m_CasterSlots, m_CasterSlotsPrev);

        const auto& Sun = Scene.m_Sun;
        // The sun: its virtual pages (m_bVirtualShadows), or the clipmap. A switch starts both over
        if( m_bVirtualShadows != m_bVSMLast ) { m_Clipmap.Invalidate(); m_VSM.Invalidate(); m_bVSMLast = m_bVirtualShadows; }
        m_VSM.m_ToRender.clear();
        if( Sun.m_bCastShadows && m_bVirtualShadows )
        {
            // The requests of this ring slot's last request pass (a ring ago: done) and its counters, with the windows that made them
            const int Slot = int(m_Device.getFrameIndex() % ring_frames_v), nBits = vsmBitUints(), nReq = nBits + 32;
            m_VSMRequests.clear();
            if( m_VSMSnap[Slot].m_bValid )
            {
                // (a ring ago: done - no wait for the frame in flight, which cost 2.3 ms of CPU; one copy out, then decoded: the mapped
                // memory can be the GPU's, where every read crosses the bus - a word at a time it cost 1 ms)
                std::vector<std::uint32_t> Got(static_cast<std::size_t>(nReq));
                (void)m_VSMReqHost.MemoryMapFrameRegion(Slot * nReq, nReq, [&](void* p) { std::memcpy(Got.data(), p, Got.size() * sizeof(std::uint32_t)); });
                m_VSM.DecodeRequests({ Got.data(), std::size_t(nBits) }, m_VSMSnap[Slot], m_VSMRequests);
                std::memcpy(m_VSMReport.m_Counters.data(), Got.data() + nBits, m_VSMReport.m_Counters.size() * sizeof(std::uint32_t));
                m_VSMReport.m_bCounters = true;
                m_VSMSnap[Slot].m_bValid = false;
            }
            if( std::vector<std::uint64_t> Ns; m_VSMTimes.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull && Ns[1] >= Ns[0] )
                m_VSMReport.m_RequestGpuMs = double(Ns[1] - Ns[0]) * 1e-6;
            // Level 0's texel: the clipmap's level 0 texel / 4, at most 2 mm (a scene's tile can be tens of metres: the static geom
            // preview's is the model's radius / 2); the screen picks the level, so the 2 mm only show where a pixel asks for them. The
            // levels: at least the setup's, more (up to max_levels_v) when the coarsest would not cover what the clipmap covers
            const float Level0Texel = std::min(Sun.m_ShadowTileSize / float(4 * m_ShadowSetup.m_ClipmapTileSize), shadows::vsm::max_level0_texel_v);
            {
                const auto& VS    = m_VSM.getSetup();
                const float Cover = Sun.m_ShadowTileSize * float(shadows::clipmap_tiles_v << (shadows::clipmap_levels_v - 1));
                const int   Need  = int(std::ceil(std::log2(std::max(Cover / (Level0Texel * float(VS.m_Window * shadows::vsm::page_texels_v)), 1.0f)))) + 1;
                if( Need > VS.m_Levels && VS.m_Levels < shadows::vsm::max_levels_v ) { auto S = VS; S.m_Levels = std::min(Need, shadows::vsm::max_levels_v); m_VSM.Init(S); }
            }
            m_VSM.Update(Sun.m_Direction, Level0Texel, Eye, Changed, m_bShadowCache, m_VSMRequests, m_Device.getFrameIndex());
            m_VSMReport.m_Pages = m_VSM.m_Stats;
            m_ShadowTiles.clear();
        }
        else if( Sun.m_bCastShadows ) m_Clipmap.Update(Sun.m_Direction, Sun.m_ShadowTileSize, Eye, Changed, m_bShadowCache, m_ShadowTiles);
        else                   { m_ShadowTiles.clear(); m_Clipmap.Invalidate(); m_VSM.Invalidate(); }      // the casters changing meanwhile reach no tile: all of them when it casts again
        m_LocalShadows.Update(Scene.m_LocalLights, Changed, m_bShadowCache, Eye, ProjScale, m_ShadowUpdatesPerFrame, m_LocalMaps, m_LocalToRender);
        m_LightStats.m_nShadowed = m_LightStats.m_nShadowFallback = 0;
        for( std::size_t i = 0; i < Scene.m_LocalLights.size(); ++i ) if( Scene.m_LocalLights[i].m_bCastShadows )
        {
            ++m_LightStats.m_nShadowed;
            m_LightStats.m_nShadowFallback += m_LocalMaps[i].m_Size == 0;
        }
        m_LightStats.m_nShadowStale = m_LocalShadows.StaleCount();

        m_nShadowTilesRendered = static_cast<int>(m_ShadowTiles.size() + m_VSM.m_ToRender.size() + m_LocalToRender.size());
        if( m_nShadowTilesRendered == 0 ) return;

        // Draws the casters that pass Keep with W2C into the current viewport
        // Draws the casters of Indices that pass Keep with W2C into the current viewport
        const auto DrawCasters = [&]( xgpu::cmd_buffer& Cmd, const xmath::fmat4& W2C, std::span<const std::uint32_t> Indices, const auto& Keep )
        {
            auto& ViewBlock = m_ShadowUBO.allocEntry<details::ubo_geom_view>();     // absolute world: the instances' origin as is
            ViewBlock.m_w2C         = W2C;
            ViewBlock.m_WorldOffset = xmath::fvec4(m_Origin, 0);
            const cluster_geom*            pBoundGeom = nullptr;       // binds only when the mesh or the caster changes
            const void*                    pBound     = nullptr;       // (the caster's shared state: texture-table materials of one pipeline share it)
            bool                           bCutLight  = false;

            for( const std::uint32_t i : Indices )
            {
                if( !Keep(i) ) continue;
                const auto& D = *Draws[i].m_pDraw;
                const auto& M = *Draws[i].m_pMesh;

                if( M.m_bCluster == false )
                {
                    if( !D.m_Material.m_bCastShadows ) continue;        // (the standard material's choice: the editors' ground grid does not cast)
                    pBoundGeom = nullptr;
                    pBound     = nullptr;
                    Cmd.setPipelineInstance(m_ShadowInstance[0]);
                    Cmd.setPushConstants(details::push_forward{ D.m_L2W, W2C });
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(M.m_VertexBuffer));
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(M.m_IndexBuffer));
                    Cmd.Draw(M.m_nIndices);
                    continue;
                }

                const auto& G     = M.m_Geom;
                const bool  bSkin = G.m_Kind == geom_kind::SKIN;
                for( const auto& Sub : G.m_Submeshes )
                {
                    // The material's settings: no shadow, or a masked one's cutout (its own caster: the material's fragment, which stops after the cut)
                    auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                    const auto Vis = pMat ? MaterialVisibilityOf(pMat->m_State) : material_visibility{};
                    if( !Vis.m_bCastsShadow ) continue;
                    auto* pCut = Vis.m_Opacity == material_visibility::opacity::MASKED ? getMaterialInstance(*pMat, G.m_Kind, true) : nullptr;
                    auto* pInstance = pCut ? pCut : &m_ShadowInstance[bSkin ? 2 : 1];
                    if( pInstance->m_Private.get() != pBound || &G != pBoundGeom )
                    {
                        if( pCut )
                        {
                            if( !bCutLight )        // the fragment's "shadow pass" flag (ubo_lighting m_LightParams.w), once per call
                            {
                                m_LightUBO.allocEntry<details::ubo_lighting>().m_LightParams = xmath::fvec4(0, 0, 1, 1);
                                bCutLight = true;
                            }
                            std::array<xgpu::buffer*, 11> Buffers = bSkin ? std::array<xgpu::buffer*, 11>{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy, &m_MaterialParams, &m_EntityParams }
                                                                          : std::array<xgpu::buffer*, 11>{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, nullptr, nullptr };
                            Cmd.setPipelineInstance(*pCut, std::span{ Buffers.data(), bSkin ? std::size_t{11} : std::size_t{9} });
                            if( pMat->m_bTable ) Cmd.setTextureTable(pMat->m_Table);
                            Cmd.setDynamicUBO(m_LightUBO, 1);
                        }
                        else
                        {
                            std::array<xgpu::buffer*, 4> StaticBuffers{ G.m_pClusterData, bSkin ? G.m_pBones : &m_InstanceBuffer, &m_InstanceBuffer, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy };
                            Cmd.setPipelineInstance(*pInstance, std::span{ StaticBuffers.data(), bSkin ? std::size_t{4} : std::size_t{2} });
                        }
                        Cmd.setDynamicUBO(m_ShadowUBO, 0);
                        BindClusterStreams(Cmd, G);
                        pBound     = pInstance->m_Private.get();
                        pBoundGeom = &G;
                    }
                    else if( pCut && pMat->m_bTable ) Cmd.setTextureTable(pMat->m_Table);     // (the same state, another material: its table - bindless: the same set, nothing)

                    const std::uint32_t MaterialBase = pCut ? pMat->m_ParamBase : 0u;
                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
                        if( Draws[i].m_BoxedBase != ~0u && m_ClusterSkip[Draws[i].m_BoxedBase + c] == cluster_boxed_v ) continue;     // drawn as its box: no shadow
                        const auto& C = G.m_Clusters[c];
                        if( bSkin ) Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Draws[i].m_Slot, 0u, MaterialBase });
                        else        Cmd.setPushConstants(details::push_static_geom{ c, Draws[i].m_Slot, 0u, 0xFFFFFFFFu, MaterialBase });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                    }
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
        std::vector<std::uint32_t> LightCasters;        // a local light's casters (reused per light)
        const int LocalX = sunAtlasWidth();                                                // the local part's origin in the atlas
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
                    // The light's casters once (its sphere), then per face the ones in that face's frustum
                    LightCasters.clear();
                    for( std::uint32_t i = 0; i < Draws.size(); ++i ) if( shadows::Overlaps(Draws[i].m_World, L.m_Position, L.m_Radius) ) LightCasters.push_back(i);
                    for( int f = 0, nFaces = bPoint ? 6 : 1; f < nFaces; ++f )
                    {
                        const auto P = FacePos(B, f, R.m_Map.m_Size, bPoint);
                        Viewport(Cmd, P.m_X, P.m_Y, bPoint ? R.m_Map.m_Size / 2 : R.m_Map.m_Size);
                        const xmath::fmat4 FaceW2C = bPoint ? shadows::FaceW2C(L.m_Position, shadows::CubeFaceAxis(f), 1.0f, Near, L.m_Radius)
                                                            : shadows::FaceW2C(L.m_Position, L.m_Direction.NormalizeCopy(), shadows::SpotTanHalf(L), Near, L.m_Radius);
                        const frustum Face = frustum::FromW2C(FaceW2C);
                        DrawCasters(Cmd, FaceW2C, LightCasters, [&](std::uint32_t i) { return !m_bShadowFaceCull || Face.Overlaps(Draws[i].m_World, false); });
                    }
                }
            }

            auto Cmd = Window.StartRenderPass(m_ShadowPass);

            if( !bClipmapDone )
            {
                bClipmapDone = true;
                // The casters culled in light space (only when a tile or page renders): their light-space boxes, kept from frame to frame -
                // a draw's again when its world box changes, all when the light's basis does (an epoch, the switch)
                auto&                       LightBounds = m_CasterLight;
                std::vector<std::uint32_t>  AllCasters;
                if( !m_ShadowTiles.empty() || !m_VSM.m_ToRender.empty() )
                {
                    const std::array<xmath::fvec3, 3> Basis = m_bVirtualShadows ? std::array{ m_VSM.m_X, m_VSM.m_Y, m_VSM.m_Z } : std::array{ m_Clipmap.m_X, m_Clipmap.m_Y, m_Clipmap.m_Z };
                    if( std::memcmp(&Basis, &m_CasterLightBasis, sizeof(Basis)) != 0 ) { m_CasterLightWorld.clear(); m_CasterLightBasis = Basis; }
                    m_CasterLight.resize(Draws.size());
                    m_CasterLightWorld.resize(Draws.size(), xmath::fbbox{ xmath::fvec3(1, 1, 1), xmath::fvec3(-1, -1, -1) });
                    AllCasters.reserve(Draws.size());
                    for( std::uint32_t i = 0; i < Draws.size(); ++i )
                    {
                        if( std::memcmp(&m_CasterLightWorld[i], &Draws[i].m_World, sizeof(xmath::fbbox)) != 0 )
                        {
                            m_CasterLight[i]      = m_bVirtualShadows ? m_VSM.Bounds(Draws[i].m_World) : m_Clipmap.Bounds(Draws[i].m_World);
                            m_CasterLightWorld[i] = Draws[i].m_World;
                        }
                        AllCasters.push_back(i);
                    }
                }
                // The sun's virtual pages: each in its slot of the atlas, cleared, its casters (light-space rect test)
                for( const auto& P : m_VSM.m_ToRender )
                {
                    Viewport(Cmd, P.m_AtlasX, P.m_AtlasY, shadows::vsm::page_texels_v);
                    Cmd.setPipelineInstance(m_ShadowClearInstance);
                    Cmd.setBuffer(m_FullscreenVB);
                    Cmd.setBuffer(m_FullscreenIB);
                    Cmd.Draw(3);
                    DrawCasters(Cmd, P.m_W2C, AllCasters, [&](std::uint32_t i) { return shadows::clipmap::Overlaps(LightBounds[i], P.m_MinX, P.m_MinY, P.m_Size); });
                }
                const int S = m_ShadowSetup.m_ClipmapTileSize;
                for( const auto& T : m_ShadowTiles )
                {
                    Viewport(Cmd, T.m_AtlasX * S, T.m_AtlasY * S, S);
                    Cmd.setPipelineInstance(m_ShadowClearInstance);
                    Cmd.setBuffer(m_FullscreenVB);
                    Cmd.setBuffer(m_FullscreenIB);
                    Cmd.Draw(3);
                    DrawCasters(Cmd, T.m_W2C, AllCasters, [&](std::uint32_t i) { return shadows::clipmap::Overlaps(LightBounds[i], T.m_MinX, T.m_MinY, T.m_Size); });
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

    // The scene into View's HDR target (and depth), lit, from a camera-centered W2C at Eye: the main view, and the reflection
    // probes' capture faces (bMain false: no timestamps). Shadows, instances and cells of this frame are already up to date.
    void instance::renderScenePass( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye, bool bMain ) noexcept
    {
        const xmath::fmat4 ToEye = xmath::fmat4::fromTranslation(-Eye);
        // Scene -> HDR
        {
            {
                const auto& L     = Scene.m_Sun;
                auto&       Light = m_LightUBO.allocEntry<details::ubo_lighting>();
                Light.m_LightColor        = xmath::fvec4(L.m_Color * L.m_Intensity, 1);
                Light.m_AmbientLightColor = xmath::fvec4(Scene.m_Ambient, 0);
                Light.m_wSpaceLightPos    = xmath::fvec4(-L.m_Direction.NormalizeCopy(), 0);
                Light.m_wSpaceEyePos      = xmath::fvec4(0, 0, 0, 1);       // camera-centered
                Light.m_LightParams       = { 0, L.m_Temperature, 1, 0 };

                // The sun's shadow: the virtual pages (its basis, levels and this frame's table) or the clipmap. With the pages on, the
                // clipmap's flag is off: materials compiled before the pages read only it, and sample no sun shadow rather than pages as tiles
                const bool bVSM = m_bVirtualShadows && L.m_bCastShadows;
                const xmath::fvec3 BX = bVSM ? m_VSM.m_X : m_Clipmap.m_X, BY = bVSM ? m_VSM.m_Y : m_Clipmap.m_Y, BZ = bVSM ? m_VSM.m_Z : m_Clipmap.m_Z;
                Light.m_W2Light = xmath::fmat4::fromIdentity();
                for( int c = 0; c < 3; ++c )
                {
                    const float Axis[3][3] = { { BX.m_X, BX.m_Y, BX.m_Z }, { BY.m_X, BY.m_Y, BY.m_Z }, { BZ.m_X, BZ.m_Y, BZ.m_Z } };
                    for( int r = 0; r < 3; ++r ) Light.m_W2Light(r, c) = Axis[r][c];
                }
                Light.m_CameraPos = xmath::fvec4(Eye, 1);
                for( int i = 0; i < shadows::clipmap_levels_v; ++i )
                {
                    const auto& Lv = m_Clipmap.m_Levels[i];
                    Light.m_ShadowLevels[i] = { float(Lv.m_OriginX), float(Lv.m_OriginY), Lv.m_TileSize, Lv.m_DepthCenter };
                }
                Light.m_ShadowParams = { L.m_bCastShadows && !bVSM ? 1.0f : 0.0f, m_Clipmap.DepthHalfRange(0), float(m_ShadowSetup.m_ClipmapTileSize), 0 };
                const auto& VS = m_VSM.getSetup();
                Light.m_VSMParams  = bVSM ? xmath::fvec4{ 1.0f, m_VSM.Level0Texel(), float(VS.m_Levels), float(vsmTableBase()) } : xmath::fvec4{ 0, 0, 0, 0 };
                Light.m_VSMParams2 = { float(VS.m_Window), float(shadows::vsm::page_texels_v), float(VS.SlotsPerRow()), float(m_VSMDebugView > 0 ? m_VSMDebugView : m_bVSMTapTranslation ? 1 : 0) };
                for( int i = 0; i < shadows::vsm::max_levels_v; ++i )
                {
                    const auto& Lv = m_VSM.m_Levels[i];
                    Light.m_VSMLevels[i] = { float(Lv.m_OriginX), float(Lv.m_OriginY), Lv.m_PageSize, Lv.m_DepthCenter };
                }
                Light.m_VSMParams3 = { m_VSM.ResolutionBias(), 0, 0, 0 };

                writeLocalLights(Scene, Eye, Light);
                Light.m_EnvParams = { m_bEnvOn ? 1.0f : 0.0f, m_EnvIntensity, 0, 0 };

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

            // The editor's view of huge meshes (m_bBoxesOnly, m_FrameBudgetMs): the clusters left out, and their boxes
            const bool bPlan = bMain && (m_bBoxesOnly || m_FrameBudgetMs > 0);
            if( bMain )
            {
                m_NotDrawn.clear();
                m_ClusterSkip.clear();
                m_ClusterBase.assign(Scene.m_Draws.size(), ~0u);
                m_ClusterPlanKey.assign(Scene.m_Draws.size(), 0);
                m_BudgetStats.m_nClusters = m_BudgetStats.m_nDrawn = 0;
                m_BudgetStats.m_Allowance = std::uint64_t(m_IndexAllowance);
                if( bPlan ) planClusters(Scene, Eye, Frustum);
            }

            // A submesh the half-resolution GI pass reads (and the depth prepass lays): opaque, single sided, depth written and tested, filled
            const auto Opaque = [&]( const material_slot* pMat )
            {
                if( pMat == nullptr ) return true;
                const auto& St = pMat->m_State;
                return !(St.m_Blend != material_state::blend::OFF || St.m_bDoubleSided || !St.m_bDepthWrite || !St.m_bDepthTest || St.m_bWireframe);
            };
            // The GI read at half resolution (scene::gi_settings::m_bHalfRes; XLION_GI_HALFRES=0/1 overrides): the main view, no GI debug view. The rigid
            // opaque clusters' depth into the view's half-size target, then the GI draws' own read at it (the material's: GIPixelGI, with
            // the surface's normal) into the view's buffer; the scene pass's GI draws below upsample it (SurfelGIScreen) instead of reading
            // the surfels each pixel - a quarter of the reads
            static const int s_HalfResEnv = [] { const char* p = std::getenv("XLION_GI_HALFRES"); return p ? std::atoi(p) : -1; }();     // (measurement: 0 off, 1 on)
            const bool bScreen = bMain && (s_HalfResEnv >= 0 ? s_HalfResEnv != 0 : Scene.m_GI.m_bHalfRes) && Scene.m_GI.m_Debug == gi_debug::NONE
                              && !(bPlan && m_bBoxesOnly) && createGIScreenTargets(View) == nullptr;
            if( bMain ) m_GIStats.m_nScreenDraws = m_GIStats.m_nScreenCalls = 0;
            if( bMain && !bScreen ) View.m_bGIScreenPrev = false;      // (a frame without it: nothing to reproject from)
            if( bScreen )
            {
                // Temporal (XLION_GI_TEMPORAL=0: off): last frame's buffer becomes the one reprojected from; this frame's is cleared (a pixel no
                // GI surface covers keeps nothing of an older frame) and gets the header: the size, the frame's 2x2 pattern, whether last
                // frame's is there, its world to clip and the eye's move (xlionrender_gi_screen_frag.glsl)
                static const int s_TemporalEnv = [] { const char* p = std::getenv("XLION_GI_TEMPORAL"); return p ? std::atoi(p) : -1; }();     // (0 off, 1/2/4 the rate)
                const bool bTemporal = s_TemporalEnv >= 0 ? s_TemporalEnv != 0 : Scene.m_GI.m_bTemporal;
                const int  Rate      = s_TemporalEnv > 0 ? s_TemporalEnv : Scene.m_GI.m_TemporalRate;
                const std::uint32_t RateLog2 = Rate >= 4 ? 2u : Rate >= 2 ? 1u : 0u;
                const int  CSlot = int(m_Device.getFrameIndex() % ring_frames_v);
                (void)View.m_GIScreenCounters.MemoryMapFrameRegion(CSlot, 1, [&](void* p)         // (this slot's frame is done: its counters)
                {
                    const auto* pC = static_cast<const std::uint32_t*>(p);
                    m_GIStats.m_nScreenFresh = int(pC[0]); m_GIStats.m_nScreenReprojected = int(pC[1]); m_GIStats.m_nScreenRejected = int(pC[2]); m_GIStats.m_nScreenAged = int(pC[3]);
                });
                std::swap(View.m_GIScreenBuffer, View.m_GIScreenPrev);
                if( View.m_bGIScreenPerPixel != Scene.m_GI.m_bPerPixel ) View.m_bGIScreenPrev = false;
                View.m_bGIScreenPerPixel = Scene.m_GI.m_bPerPixel;
                {   // A camera cut (a jump past 2 m, or the view turned past 45 degrees): no history - a surface seen from both sides of a
                    // cut would reproject, its reflection read toward the old eye. The forward: clip w's row of world to clip (column major)
                    const auto Fwd = []( const xmath::fmat4& M ) { const float* p = reinterpret_cast<const float*>(&M); const xmath::fvec3 F{ p[3], p[7], p[11] }; return F / std::max(F.Length(), 1e-12f); };
                    const xmath::fvec3 Jump = Eye - View.m_GIScreenPrevEye;
                    if( View.m_bGIScreenPrev && (Jump.Dot(Jump) > 4.0f || Fwd(W2C).Dot(Fwd(View.m_GIScreenPrevW2C)) < 0.7071f) ) View.m_bGIScreenPrev = false;
                }
                {
                    const xgpu::buffer::copy C{ screen_header_v, screen_header_v, View.m_GIScreenWidth * View.m_GIScreenHeight };
                    Window.CopyBuffer(View.m_GIScreenZeros, View.m_GIScreenBuffer, std::span{ &C, 1 });
                }
                {
                    const int HBase = int(m_Device.getFrameIndex() % ring_frames_v) * screen_header_v;
                    const xmath::fvec3 Move = Eye - View.m_GIScreenPrevEye;
                    (void)View.m_GIScreenHeader.MemoryMapFrameRegion(HBase, screen_header_v, [&](void* p)
                    {
                        auto* pH = static_cast<std::uint32_t*>(p);
                        std::memset(pH, 0, screen_header_v * 16);
                        pH[0] = std::uint32_t(View.m_GIScreenWidth); pH[1] = std::uint32_t(View.m_GIScreenHeight);
                        pH[2] = View.m_GIScreenFrame; pH[3] = (bTemporal && View.m_bGIScreenPrev ? 1u : 0u) | (RateLog2 << 1);
                        std::memcpy(pH + 4, &View.m_GIScreenPrevW2C, sizeof(float) * 16);        // (entries 1-4: its columns, as the UBO's mat4)
                        const float M[3] = { Move[0], Move[1], Move[2] };
                        std::memcpy(pH + 20, M, sizeof(M));
                    });
                    const xgpu::buffer::copy C{ HBase, 0, screen_header_v };
                    Window.CopyBuffer(View.m_GIScreenHeader, View.m_GIScreenBuffer, std::span{ &C, 1 });
                }
                View.m_GIScreenPrevW2C = W2C; View.m_GIScreenPrevEye = Eye; View.m_bGIScreenPrev = true; ++View.m_GIScreenFrame;
                // Its draws: a record (the cluster's push constants) and a command a cluster, in runs of one bind state - phase 0 a run a
                // geometry, phase 1 a run a geometry and GI buffer - each run one indirect draw (this frame's region of the view's buffers)
                m_ScreenRecords.clear(); m_ScreenCommands.clear(); m_ScreenRuns.clear();
                const int Base = int(m_Device.getFrameIndex() % ring_frames_v) * screen_draws_v;
                for( int Phase = 0; Phase < 2; ++Phase )        // 0: the depth, 1: the GI read at it
                {
                    for( const auto& Vis : m_Visible )
                    {
                        const std::size_t iDraw = Vis.m_iDraw;
                        const auto&       D     = Scene.m_Draws[iDraw];
                        const auto*       pMesh = Vis.m_pMesh;
                        const std::uint32_t Slot = m_DrawSlot[iDraw];
                        if( !pMesh->m_bCluster || Slot == instance_slots::none_v ) continue;
                        const auto& G = pMesh->m_Geom;
                        if( G.m_Kind == geom_kind::SKIN ) continue;
                        const bool bSections = !pMesh->m_GISections.empty() && pMesh->m_GIClusterSection.size() == G.m_Clusters.size();
                        const auto GI = iDraw < m_DrawGI.size() ? m_DrawGI[iDraw] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };
                        if( Phase == 1 && !(bSections ? iDraw < m_DrawGISec.size() && !m_DrawGISec[iDraw].empty() : pMesh->m_pGI && GI.second != 0xFFFFFFFFu) ) continue;
                        const std::uint32_t SkipBase = bPlan ? m_ClusterBase[iDraw] : ~0u;
                        for( const auto& Sub : G.m_Submeshes )
                        {
                            if( !Opaque(Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr) ) continue;
                            for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                            {
                                if( SkipBase != ~0u && m_ClusterSkip[SkipBase + c] ) continue;
                                if( m_ScreenRecords.size() == std::size_t(screen_draws_v) ) break;
                                const xgpu::buffer*        pBuf = nullptr;
                                details::push_static_geom  Rec{ c, Slot, 0u, 0xFFFFFFFFu, 0u };
                                if( Phase == 1 )
                                {
                                    const std::uint16_t k = bSections ? pMesh->m_GIClusterSection[c] : 0;
                                    pBuf = bSections ? &pMesh->m_GISections[k].m_Buffer : &pMesh->m_GIBuffer;
                                    if( bSections )
                                    {
                                        const auto SGI = k < m_DrawGISec[iDraw].size() ? m_DrawGISec[iDraw][k] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };
                                        if( SGI.second == 0xFFFFFFFFu ) continue;
                                        Rec = { c, Slot, SGI.first, SGI.second, 0u, pMesh->m_GIClusterVertexDelta[c] };
                                    }
                                    else Rec = { c, Slot, GI.first, GI.second, 0u };
                                }
                                const auto& C = G.m_Clusters[c];
                                if( m_ScreenRuns.empty() || m_ScreenRuns.back().m_Phase != Phase || m_ScreenRuns.back().m_pGeom != &G || m_ScreenRuns.back().m_pGI != pBuf )
                                    m_ScreenRuns.push_back({ Phase, &G, pBuf, int(m_ScreenRecords.size()), 0 });
                                ++m_ScreenRuns.back().m_Count;
                                m_ScreenCommands.push_back({ C.m_nIndices, 1u, C.m_iIndex, std::int32_t(C.m_iVertex), std::uint32_t(Base + int(m_ScreenRecords.size())) });
                                m_ScreenRecords.push_back(Rec);
                                m_GIStats.m_nScreenDraws += Phase;
                            }
                        }
                    }
                }
                const int nScreen = int(m_ScreenRecords.size());
                if( nScreen )
                {
                    (void)View.m_GIScreenRecords.MemoryMapFrameRegion(Base, nScreen, [&](void* p) { std::memcpy(p, m_ScreenRecords.data(), m_ScreenRecords.size() * sizeof(m_ScreenRecords[0])); });
                    (void)View.m_GIScreenCommands.MemoryMapFrameRegion(Base, nScreen, [&](void* p) { std::memcpy(p, m_ScreenCommands.data(), m_ScreenCommands.size() * sizeof(m_ScreenCommands[0])); });
                }
                m_GIStats.m_nScreenCalls = int(m_ScreenRuns.size());

                auto SCmd = Window.StartRenderPass(View.m_GIScreenPass);
                for( const auto& Run : m_ScreenRuns )
                {
                    const auto& G = *Run.m_pGeom;
                    if( Run.m_Phase == 0 )
                    {
                        std::array<xgpu::buffer*, 10> Buffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, &View.m_GIScreenRecords };
                        SCmd.setPipelineInstance(m_GIScreenDepthInstance, std::span{ Buffers });
                    }
                    else
                    {
                        std::array<xgpu::buffer*, 12> Buffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, const_cast<xgpu::buffer*>(Run.m_pGI), &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, &View.m_GIScreenBuffer, &View.m_GIScreenRecords, &View.m_GIScreenPrev };
                        SCmd.setPipelineInstance(m_GIScreenInstance, std::span{ Buffers });
                    }
                    SCmd.setDynamicUBO(m_LightUBO, 1);
                    SCmd.setDynamicUBO(m_StaticGeomUBO, 0);
                    BindClusterStreams(SCmd, G);
                    SCmd.DrawIndexedIndirect(View.m_GIScreenCommands, Base + Run.m_First, Run.m_Count);
                }
                SCmd.Release();         // (the pass ends: a copy goes outside it)
                {   // (its counters into this frame's slot: read a ring later, above)
                    const xgpu::buffer::copy C{ 6, CSlot, 1 };
                    Window.CopyBuffer(View.m_GIScreenBuffer, View.m_GIScreenCounters, std::span{ &C, 1 });
                }
            }

            // a skinned mesh's GI: the probe grid (no surfels of its own; the dummy: none). ponytail: the main view's only (its header carries
            // the main eye: the reflection views' positions are their own eye's)
            xgpu::buffer* const Probes = bMain && Scene.m_GI.m_bEnable && Scene.m_GI.m_bProbes && m_pGITrace && m_pGITrace->ProbeBuffer() ? m_pGITrace->ProbeBuffer() : nullptr;

            // A rigid cluster draw's GI (the scene pass's draws, CPU or GPU culled): per draw and material, a cluster's GI buffer (binding 5)
            // and its push constants (or record)
            const auto RigidGI = [&]( std::size_t iDraw, const material_slot* pMat )
            {
                const auto* pMesh  = findMesh(Scene.m_Draws[iDraw].m_Mesh);
                const bool  bRigid = pMesh->m_Geom.m_Kind != geom_kind::SKIN;
                const auto  GI     = iDraw < m_DrawGI.size() ? m_DrawGI[iDraw] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };
                const bool  bSections0 = bRigid && !pMesh->m_GISections.empty() && pMesh->m_GIClusterSection.size() == pMesh->m_Geom.m_Clusters.size();
                // read at half resolution (the pass above): the buffer as its GI buffer, gi_screen_v as its level - one buffer for every section
                const bool  bScreenSub = bScreen && bRigid && Opaque(pMat) && (bSections0 ? iDraw < m_DrawGISec.size() && !m_DrawGISec[iDraw].empty() : pMesh->m_pGI && GI.second != 0xFFFFFFFFu);
                const bool  bSections = bSections0 && !bScreenSub;
                // a rigid mesh without surfels of its own reads the probe grid as well (its GI buffer: the grid; its level: gi_probes_v)
                const bool  bRigidProbes = bRigid && Probes && !pMesh->m_pGI;
                return [=, this, &View]( std::uint32_t c, std::uint32_t Slot, std::uint32_t MaterialBase ) -> std::pair<xgpu::buffer*, details::push_static_geom>
                {
                    if( bScreenSub ) return { &View.m_GIScreenBuffer, { c, Slot, 0u, 0xFFFFFFFDu, MaterialBase } };      // (xlionrender_surfel_gi.glsl gi_screen_v)
                    if( bSections )         // GI sections: the cluster's section - its buffer, its place in the light ring, its vertices
                    {
                        const std::uint16_t k = pMesh->m_GIClusterSection[c];
                        const auto SGI = k < m_DrawGISec[iDraw].size() ? m_DrawGISec[iDraw][k] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };
                        return { const_cast<xgpu::buffer*>(&pMesh->m_GISections[k].m_Buffer), { c, Slot, SGI.first, SGI.second, MaterialBase, pMesh->m_GIClusterVertexDelta[c] } };
                    }
                    xgpu::buffer* const pBuf = pMesh->m_pGI ? const_cast<xgpu::buffer*>(&pMesh->m_GIBuffer) : bRigidProbes ? Probes : const_cast<xgpu::buffer*>(&m_GIDummy);
                    return { pBuf, { c, Slot, GI.first, bRigidProbes ? 0xFFFFFFFEu : GI.second, MaterialBase } };     // (0xFFFFFFFE: xlionrender_surfel_gi.glsl gi_probes_v)
                };
            };

            // GPU culling (features/cull, documentation/gpu_cluster_culling.md): the main view's opaque rigid clusters with a material, a
            // candidate each, in runs (a bind state each; a prepass run a geometry), culled here - outside the pass - into their runs' commands
            static const bool s_bNoPrepass = std::getenv("XLION_NO_PREPASS") != nullptr;
            static const bool s_bNoGPUCull      = std::getenv("XLION_NO_GPU_CULL") != nullptr;
            static const bool s_bNoGPUOcclusion = std::getenv("XLION_NO_GPU_OCCLUSION") != nullptr;
            m_CullCandidates.clear(); m_CullMatrices.clear(); m_CullRuns.clear(); m_CullRunOf.clear(); m_CullSub.clear();
            m_CullSubBase.assign(Scene.m_Draws.size(), ~0u);
            bool          bOcclude   = false;       // the two passes: pass 1 (seen last frame), the Hi-Z, pass 2 (the rest)
            bool          bPass2     = false;       // pass 2's lists drawn (occlusion, or a frozen cull that had them)
            std::uint32_t CullSecond = 0;           // pass 2's commands: after pass 1's
            if( bMain && m_pCull && m_bGPUCull && !s_bNoGPUCull && !(bPlan && m_bBoxesOnly) )      // (a frame budget: the clusters it keeps)
            {
                const auto RunOf = [&]( xgpu::pipeline_instance* pInstance, material_slot* pMat, const cluster_geom* pGeom, xgpu::buffer* pGI )
                {
                    const auto [It, bNew] = m_CullRunOf.try_emplace({ pInstance, pMat, pGeom, pGI }, std::uint32_t(m_CullRuns.size()));
                    if( bNew ) m_CullRuns.push_back({ pInstance, pMat, pGeom, pGI });
                    return It->second;
                };
                for( const auto& Vis : m_Visible )
                {
                    const std::size_t   iDraw = Vis.m_iDraw;
                    const auto&         D     = Scene.m_Draws[iDraw];
                    const auto*         pMesh = Vis.m_pMesh;
                    const std::uint32_t Slot  = m_DrawSlot[iDraw];
                    if( !pMesh->m_bCluster || Slot == instance_slots::none_v || pMesh->m_Geom.m_Kind == geom_kind::SKIN || m_CullMatrices.size() == std::size_t(gpu_cull::max_draws_v) ) continue;
                    const auto&         G     = pMesh->m_Geom;
                    const auto          Draw  = std::uint32_t(m_CullMatrices.size());
                    const auto          Base  = std::uint32_t(m_CullSub.size());
                    const auto          VisBase = m_pCull->VisKey(Slot, std::uint32_t(G.m_Clusters.size()));
                    const std::uint32_t SkipBase = bPlan ? m_ClusterBase[iDraw] : ~0u;
                    m_CullSub.resize(Base + G.m_Submeshes.size(), 0);
                    bool bAny = false;
                    for( std::size_t s = 0; s < G.m_Submeshes.size(); ++s )
                    {
                        const auto& Sub  = G.m_Submeshes[s];
                        auto*       pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                        // (past the caps: the CPU path; a submesh adds at most a run a cluster and its prepass run)
                        if( pMat == nullptr || !Opaque(pMat) || m_CullCandidates.size() + Sub.m_nClusters > std::size_t(gpu_cull::max_candidates_v)
                            || m_CullRuns.size() + Sub.m_nClusters + 1 > std::size_t(gpu_cull::max_runs_v) ) continue;
                        auto* pInstance = getRecordsInstance(*pMat);
                        if( pInstance == nullptr ) continue;
                        const auto GIOf = RigidGI(iDraw, pMat);
                        // (a run looked up again only when the GI buffer changes: GI sections; the prepass run: the geometry's)
                        const std::uint32_t Prepass = s_bNoPrepass ? ~0u : RunOf(nullptr, nullptr, &G, nullptr);
                        std::uint32_t       Run     = ~0u;
                        const xgpu::buffer* pRunBuf = nullptr;
                        for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                        {
                            if( SkipBase != ~0u && m_ClusterSkip[SkipBase + c] ) continue;      // its box instead, or outside the view (planClusters)
                            const auto [pBuf, Rec] = GIOf(c, Slot, pMat->m_ParamBase);
                            if( Run == ~0u || pBuf != pRunBuf ) { Run = RunOf(pInstance, pMat, &G, pBuf); pRunBuf = pBuf; }
                            ++m_CullRuns[Run].m_Max;
                            if( Prepass != ~0u ) ++m_CullRuns[Prepass].m_Max;
                            const auto& C    = G.m_Clusters[c];
                            const auto  Ctr  = C.m_BBox.getCenter();
                            const auto  Half = (C.m_BBox.m_Max - C.m_BBox.m_Min) * 0.5f;
                            m_CullCandidates.push_back({ .m_Run = Run, .m_RunFirst = 0
                                                       , .m_PrepassRun = Prepass, .m_PrepassFirst = 0
                                                       , .m_Draw = Draw, .m_nIndices = std::uint32_t(C.m_nIndices), .m_iFirstIndex = std::uint32_t(C.m_iIndex), .m_VertexOffset = std::int32_t(C.m_iVertex)
                                                       , .m_Record = Rec, .m_VisKey = VisBase + c, .m_Center = { Ctr.m_X, Ctr.m_Y, Ctr.m_Z, 0 }, .m_Half = { Half.m_X, Half.m_Y, Half.m_Z, 0 } });
                        }
                        m_CullSub[Base + s] = 1;
                        bAny = true;
                    }
                    if( bAny ) { m_CullSubBase[iDraw] = Base; m_CullMatrices.push_back(W2C * ToEye * D.m_L2W); }
                    else m_CullSub.resize(Base);
                }
                std::uint32_t First = 0;        // a run's commands: its candidates' room
                for( auto& R : m_CullRuns ) { R.m_First = First; First += R.m_Max; }
                CullSecond = First;
                for( auto& K : m_CullCandidates )
                {
                    K.m_RunFirst = m_CullRuns[K.m_Run].m_First;
                    if( K.m_PrepassRun != ~0u ) K.m_PrepassFirst = m_CullRuns[K.m_PrepassRun].m_First;
                }
                m_CullPrepass.resize(m_CullRuns.size());
                for( std::size_t r = 0; r < m_CullRuns.size(); ++r ) m_CullPrepass[r] = m_CullRuns[r].m_pInstance == nullptr;
                // Frozen (m_bFreezeCull): the last cull's lists as they are - no cull, no Hi-Z - while the runs are the same
                if( m_bFreezeCull && m_CullFrozen.m_bValid && m_CullFrozen.m_nRuns == m_CullRuns.size() && m_CullFrozen.m_Second == CullSecond ) bPass2 = m_CullFrozen.m_bPass2;
                else
                {
                    // A camera cut (a jump past 2 m, or the view turned past 45 degrees) tries the occlusion again at once (adaptive: gpu_cull::WantOcclusion)
                    const auto Fwd = []( const xmath::fmat4& M ) { const float* p = reinterpret_cast<const float*>(&M); const xmath::fvec3 F{ p[3], p[7], p[11] }; return F / std::max(F.Length(), 1e-12f); };
                    const xmath::fvec3 Jump = Eye - m_CullPrevEye;
                    const bool bCut = Jump.Dot(Jump) > 4.0f || Fwd(W2C).Dot(m_CullPrevFwd) < 0.7071f;
                    m_CullPrevEye = Eye; m_CullPrevFwd = Fwd(W2C);
                    m_pCull->Begin(m_Device.getFrameIndex());
                    bOcclude = bPass2 = m_bGPUOcclusion && !s_bNoGPUOcclusion && m_pCull->WantOcclusion(bCut, m_GPUOcclusionMinTriangles) && createCullTargets(View) == nullptr;
                    m_pCull->Cull(Window, m_CullCandidates, m_CullMatrices, m_CullPrepass, CullSecond, bOcclude);
                    static const bool s_bCullTimes = std::getenv("XLION_GPU_TIMES") != nullptr;
                    if( s_bCullTimes && m_Device.getFrameIndex() % 120 == 0 )
                        std::printf("CULLGPU pass 1 cull %.3f, pyramid + pass 2 cull %.3f ms; occlusion %s (it hid %.0f%% of the triangles it tested); %d clusters, %lld triangles shaded\n"
                                   , m_pCull->m_CullGpuMs, m_pCull->m_OccludeGpuMs, m_pCull->isOcclusionSkipped() ? "skipped" : "on", 100.0 * m_pCull->m_OccludedShare, m_pCull->m_Drawn, (long long)m_pCull->m_Triangles), std::fflush(stdout);
                    m_CullFrozen = { true, m_CullRuns.size(), CullSecond, bOcclude };
                }
            }
            const auto CulledOnGPU = [&]( std::size_t iDraw, const cluster_geom& G, const cluster_geom::submesh& Sub )
            {
                return m_CullSubBase[iDraw] != ~0u && m_CullSub[m_CullSubBase[iDraw] + std::size_t(&Sub - G.m_Submeshes.data())] != 0;
            };

            std::optional<xgpu::cmd_buffer> CmdPass;     // (GPU culling's occlusion ends the pass for its Hi-Z and loads it again: the same Cmd)
            CmdPass.emplace(Window.StartRenderPass(View.m_ScenePass));
            auto& Cmd = *CmdPass;
            if( bMain ) Cmd.WriteTimestamp(m_Timestamps, 1);
            // Measurement only (XLION_GPU_TIMES): the main view's scene pass split - its start, the depth prepass's end, its end (SCENESPLIT)
            static const bool s_bSplit = std::getenv("XLION_GPU_TIMES") != nullptr;
            static std::unordered_map<const void*, xgpu::timestamps> s_Split;      // ponytail: per renderer, never erased (a measurement hook)
            xgpu::timestamps* pSplit = nullptr;
            if( s_bSplit && bMain )
            {
                auto [It, bNew] = s_Split.try_emplace(this);
                if( bNew && m_Device.Create(It->second, { .m_Count = 3 }) ) s_Split.erase(It);
                else pSplit = &It->second;
            }
            if( pSplit )
            {
                static std::vector<std::uint64_t> s_Ns;
                static double s_Pre = 0, s_Shade = 0; static int s_n = 0;
                if( pSplit->getResults(s_Ns) && s_Ns.size() >= 3 && s_Ns[2] > s_Ns[1] && s_Ns[1] > s_Ns[0] )
                {
                    s_Pre += double(s_Ns[1] - s_Ns[0]) * 1e-6; s_Shade += double(s_Ns[2] - s_Ns[1]) * 1e-6;
                    if( ++s_n == 120 ) { std::printf("SCENESPLIT prepass %.3f, the rest (shading, sky, blended) %.3f ms\n", s_Pre / s_n, s_Shade / s_n); std::fflush(stdout); s_n = 0; s_Pre = s_Shade = 0; }
                }
                Cmd.WriteTimestamp(*pSplit, 0);
            }

            // A GPU culled run's draws in a pass (0: pass 1, 1: pass 2)
            const auto DrawCulled = [&]( std::size_t r, int Pass )
            {
                const auto& R = m_CullRuns[r];
                Cmd.DrawIndexedIndirectCount(m_pCull->m_Commands, int(R.m_First + (Pass ? CullSecond : 0u)), m_pCull->m_Counts, int(r) + (Pass ? gpu_cull::max_runs_v : 0), int(R.m_Max));
            };
            // The GPU culled prepass runs (a run a geometry), a pass's
            const auto PrepassCulled = [&]( int Pass )
            {
                for( std::size_t r = 0; r < m_CullRuns.size(); ++r )
                {
                    const auto& R = m_CullRuns[r];
                    if( R.m_pInstance ) continue;
                    std::array<xgpu::buffer*, 10> Buffers{ R.m_pGeom->m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, &m_pCull->m_Records };
                    Cmd.setPipelineInstance(m_GIScreenDepthInstance, std::span{ Buffers });     // (the depth prepass's pipeline reading the records)
                    Cmd.setDynamicUBO(m_LightUBO, 1);
                    Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                    BindClusterStreams(Cmd, *R.m_pGeom);
                    DrawCulled(r, Pass);
                }
            };

            const void*                    pBound     = nullptr;      // the state the last cluster draw left bound (texture-table materials of one pipeline share it)
            const cluster_geom*            pBoundGeom = nullptr;
            const xgpu::buffer*            pBoundGIBuf = nullptr;     // (GI sections: its section's buffer)

            // A cluster mesh's submesh with its material (null: the constant one)
            const auto DrawSubmesh = [&]( std::size_t iDraw, const cluster_geom::submesh& Sub, material_slot* pMat )
            {
                const auto& D     = Scene.m_Draws[iDraw];
                const auto& G     = findMesh(D.m_Mesh)->m_Geom;
                const bool  bSkin = G.m_Kind == geom_kind::SKIN;
                const std::uint32_t Slot     = m_DrawSlot[iDraw];
                const std::uint32_t SkipBase = bPlan ? m_ClusterBase[iDraw] : ~0u;
                auto* pInstance = pMat ? getMaterialInstance(*pMat, G.m_Kind) : nullptr;
                if( pInstance == nullptr )
                {
                    if( bSkin ) return;         // ponytail: no constant-material fallback on the skin path; the editors give every slot a material
                    pInstance = &m_StaticGeomInstance;      // the constant material, read from the instance record
                }
                const std::uint32_t MaterialBase = pMat ? pMat->m_ParamBase : 0u;
                xgpu::buffer* const SkinGI = bSkin && Probes ? Probes : &m_GIDummy;
                const auto GIOf = RigidGI(iDraw, pMat);         // (rigid: a cluster's GI buffer and push constants)
                const auto Bind = [&](xgpu::buffer* pGIBuffer)
                {
                    const auto TB = std::chrono::steady_clock::now();
                    struct bind_lap { std::chrono::steady_clock::time_point t; ~bind_lap() { g_BindMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); } } BindLap{ TB };
                    if( !bSkin )
                    {
                        std::array<xgpu::buffer*, 9> GIBuffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing
                                                              , pGIBuffer, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams };
                        Cmd.setPipelineInstance(*pInstance, std::span{ GIBuffers });     // every rigid pipeline reads the GI
                    }
                    else
                    {
                        std::array<xgpu::buffer*, 11> Static{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_LightRing, &m_CellRing, SkinGI, &m_GILightRing, &m_GIMomentRing, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy, &m_MaterialParams, &m_EntityParams };
                        Cmd.setPipelineInstance(*pInstance, std::span{ Static });      // SKIN: the GI bindings of the material fragment (no surfels: the probe grid), the BoneMap, the parameters
                    }
                    if( pMat && pMat->m_bTable && pInstance != &m_StaticGeomInstance ) Cmd.setTextureTable(pMat->m_Table);
                    Cmd.setDynamicUBO(m_LightUBO, 1);
                    Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                    BindClusterStreams(Cmd, G);
                    pBound       = pInstance->m_Private.get();
                    pBoundGeom   = &G;
                    pBoundGIBuf  = pGIBuffer;
                    ++g_nSceneBinds;
                    ++m_nSceneBinds;
                };
                xgpu::buffer* const pGI0 = bSkin ? SkinGI : GIOf(Sub.m_iCluster, Slot, MaterialBase).first;
                if( pInstance->m_Private.get() != pBound || &G != pBoundGeom || (!bSkin && pBoundGIBuf != pGI0) ) Bind(pGI0);
                else if( pMat && pMat->m_bTable && pInstance != &m_StaticGeomInstance ) Cmd.setTextureTable(pMat->m_Table);     // (the same state, another material: its table - bindless: the same set, nothing)

                for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                {
                    if( SkipBase != ~0u && m_ClusterSkip[SkipBase + c] ) continue;      // its box instead, or outside the view (planClusters)
                    const auto& C = G.m_Clusters[c];
                    if( bSkin ) { Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Slot, SkinGI != &m_GIDummy ? 1u : 0u, MaterialBase }); }
                    else
                    {
                        const auto [pBuf, Push] = GIOf(c, Slot, MaterialBase);
                        if( pBuf != pBoundGIBuf ) Bind(pBuf);       // (GI sections: the cluster's section's buffer)
                        Cmd.setPushConstants(Push);
                    }
                    { const auto TD = std::chrono::steady_clock::now(); Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex)); g_DrawMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - TD).count(); }
                    ++g_nSceneDraws;
                }
            };

            // The depth prepass: the opaque rigid clusters' depth first (no GI read, no material: the same vertex shader, no color), so their
            // shading below runs once a pixel - at the depth written here (LESS_OR_EQUAL), every pixel behind it rejected before its fragment
            // runs. Left out (they shade as before, tested against it): blended, masked (their discard), double sided (the prepass culls),
            // no depth write or test, wireframe, skin. XLION_NO_PREPASS=1: none (measurement)
            if( !s_bNoPrepass )
            {
                PrepassCulled(0);
                if( bPass2 && !bOcclude ) PrepassCulled(1);     // (frozen: both lists now)
                pBound = nullptr;
                const cluster_geom* pPrepassGeom = nullptr;
                for( const auto& Vis : m_Visible )
                {
                    const std::size_t iDraw = Vis.m_iDraw;
                    const auto&       D     = Scene.m_Draws[iDraw];
                    const auto*       pMesh = Vis.m_pMesh;
                    const std::uint32_t Slot = m_DrawSlot[iDraw];
                    if( !pMesh->m_bCluster || Slot == instance_slots::none_v || (bPlan && m_bBoxesOnly) ) continue;
                    const auto& G = pMesh->m_Geom;
                    if( G.m_Kind == geom_kind::SKIN ) continue;
                    const std::uint32_t SkipBase = bPlan ? m_ClusterBase[iDraw] : ~0u;
                    for( const auto& Sub : G.m_Submeshes )
                    {
                        if( CulledOnGPU(iDraw, G, Sub) ) continue;
                        const auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                        if( pMat )
                        {
                            const auto& St = pMat->m_State;
                            if( St.m_Blend != material_state::blend::OFF || St.m_bDoubleSided || !St.m_bDepthWrite || !St.m_bDepthTest || St.m_bWireframe ) continue;
                        }
                        if( &G != pPrepassGeom )
                        {
                            std::array<xgpu::buffer*, 9> Buffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams };
                            Cmd.setPipelineInstance(m_DepthPrepassInstance, std::span{ Buffers });
                            Cmd.setDynamicUBO(m_LightUBO, 1);
                            Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                            BindClusterStreams(Cmd, G);
                            pPrepassGeom = &G;
                            pBound = nullptr;           // (the shading's binds start again)
                        }
                        for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                        {
                            if( SkipBase != ~0u && m_ClusterSkip[SkipBase + c] ) continue;
                            const auto& C = G.m_Clusters[c];
                            Cmd.setPushConstants(details::push_static_geom{ c, Slot, 0u, 0xFFFFFFFFu, 0u });       // (no GI: the vertex shader's read skipped)
                            Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                        }
                    }
                }
            }

            if( pSplit ) Cmd.WriteTimestamp(*pSplit, 1);

            // GPU culling's second pass: the depth laid so far (the clusters seen last frame, the CPU path's) into the Hi-Z pyramid, every other
            // cluster in the view tested against it; the visible ones' depth too, in the pass loaded again - then the shading of both passes
            if( bOcclude )
            {
                CmdPass.reset();        // (the pass ends: the copy and the cull run outside it)
                m_pCull->Occlude(Window, View.m_Depth, View.m_HiZ, View.m_Width, View.m_Height);
                CmdPass.emplace(Window.StartRenderPass(View.m_ScenePassLoad));
                if( !s_bNoPrepass ) PrepassCulled(1);
                pBound = nullptr;
            }

            // Translucent and additive submeshes: after the opaque ones and the sky, by priority, then back to front
            struct blended_draw { std::int8_t m_Priority; float m_Distance; std::uint32_t m_iDraw; const cluster_geom::submesh* m_pSub; material_slot* m_pMat; };
            std::vector<blended_draw> Blended;

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
                    Mat.m_Emissive  = xmath::fvec4(D.m_Material.m_Emissive, D.m_Material.m_bEditorGrid ? 1.0f : 0.0f);
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

                if( Slot == instance_slots::none_v ) continue;
                if( bPlan && m_bBoxesOnly ) continue;                    // its box instead (planClusters)
                for( const auto& Sub : pMesh->m_Geom.m_Submeshes )
                {
                    auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                    if( pMat && pMat->m_State.isBlended() )
                    {
                        const auto& B = m_DrawWorld[iDraw];
                        Blended.push_back({ pMat->m_State.m_SortPriority, ((B.m_Min + B.m_Max) * 0.5f - Eye).LengthSq(), std::uint32_t(iDraw), &Sub, pMat });
                        continue;
                    }
                    if( !CulledOnGPU(iDraw, pMesh->m_Geom, Sub) ) DrawSubmesh(iDraw, Sub, pMat);
                }
            }

            // The GPU culled runs, by bind state: a state is bound once (texture-table materials of one pipeline share it: a run of another
            // material sets its table - bindless: nothing - and its material base; the rest: the records the cull wrote)
            {
                std::vector<std::uint32_t> Order;
                for( std::uint32_t r = 0; r < std::uint32_t(m_CullRuns.size()); ++r ) if( m_CullRuns[r].m_pInstance ) Order.push_back(r);
                const auto Key = [&]( std::uint32_t r ) { const auto& R = m_CullRuns[r]; return std::tuple{ R.m_pInstance->m_Private.get(), R.m_pGeom, R.m_pGI }; };
                std::stable_sort(Order.begin(), Order.end(), [&]( std::uint32_t a, std::uint32_t b ) { return Key(a) < Key(b); });
                const void* pRunBound = nullptr; const cluster_geom* pRunGeom = nullptr; const xgpu::buffer* pRunGI = nullptr;
                for( const auto r : Order )
                {
                    const auto& R = m_CullRuns[r];
                    const auto& G = *R.m_pGeom;
                    if( R.m_pInstance->m_Private.get() != pRunBound || &G != pRunGeom || R.m_pGI != pRunGI )
                    {
                        std::array<xgpu::buffer*, 10> Buffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, R.m_pGI, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, &m_pCull->m_Records };
                        Cmd.setPipelineInstance(*R.m_pInstance, std::span{ Buffers });
                        if( R.m_pMat->m_bTable ) Cmd.setTextureTable(R.m_pMat->m_Table);
                        Cmd.setDynamicUBO(m_LightUBO, 1);
                        Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                        BindClusterStreams(Cmd, G);
                        pRunBound = R.m_pInstance->m_Private.get(); pRunGeom = &G; pRunGI = R.m_pGI;
                        ++g_nSceneBinds;
                        ++m_nSceneBinds;
                    }
                    else if( R.m_pMat->m_bTable ) Cmd.setTextureTable(R.m_pMat->m_Table);
                    Cmd.setPushConstants(details::push_static_geom{ 0u, 0u, 0u, 0xFFFFFFFFu, R.m_pMat->m_ParamBase });     // (the fragment's material base)
                    DrawCulled(r, 0);
                    if( bPass2 ) DrawCulled(r, 1);
                }
            }
            pBound = nullptr;

            // The sky as the background: the cubemap seen from inside, behind everything that was drawn
            if( bMain && Scene.m_Sky.m_bVisible && Scene.m_Sky.m_pCubemap && Scene.m_Sky.m_Intensity > 0 )
            {
                if( m_pSkyBackground != Scene.m_Sky.m_pCubemap )       // its instance binds the texture
                {
                    m_Device.Destroy(std::move(m_SkyBackgroundInstance));
                    auto Bind = std::array{ xgpu::pipeline_instance::sampler_binding{ *const_cast<xgpu::texture*>(Scene.m_Sky.m_pCubemap) } };
                    m_pSkyBackground = m_Device.Create(m_SkyBackgroundInstance, { .m_PipeLine = m_SkyBackgroundPipeline, .m_SamplersBindings = Bind }) == nullptr ? Scene.m_Sky.m_pCubemap : nullptr;
                }
                if( m_pSkyBackground )
                {
                    Cmd.setPipelineInstance(m_SkyBackgroundInstance);
                    Cmd.setPushConstants(details::push_sky_background{ W2C.Inverse(), xmath::fvec4(Scene.m_Sky.m_Intensity, float(View.m_Width), float(View.m_Height), 0) });
                    Cmd.setBuffer(m_FullscreenVB);
                    Cmd.setBuffer(m_FullscreenIB);
                    Cmd.Draw(3);
                }
            }

            // The blended ones, over everything (ponytail: sorted per submesh by its draw's box center; per-pixel order would need OIT)
            std::sort(Blended.begin(), Blended.end(), [](const blended_draw& a, const blended_draw& b) { return a.m_Priority != b.m_Priority ? a.m_Priority < b.m_Priority : a.m_Distance > b.m_Distance; });
            pBound = nullptr;
            for( const auto& B : Blended ) DrawSubmesh(B.m_iDraw, *B.m_pSub, B.m_pMat);

#if 0       // The first attempt: the primitives' OBB outline shader over the scene (it needs a mesh that is box-like): replaced by renderOutline
            // The selection's outline, with the editor's outline shader: the selected draw's mesh expanded in screen space, over what was drawn
            // (the depth bias puts it behind the object's own pixels: only the ring around the silhouette shows, and what is in front of it hides it)
            if( bMain && Scene.m_SelectedUser != ~0ull )
                for( const auto& D : Scene.m_Draws )
                {
                    if( D.m_UserID != Scene.m_SelectedUser ) continue;
                    const auto* pMesh = findMesh(D.m_Mesh);
                    if( pMesh && pMesh->m_nOutlineIndices > 0 )
                    {
                        constexpr float kTwoPi = 6.28318530717958647692f;
                        static const auto s_Start = std::chrono::steady_clock::now();
                        const float Pulse = 0.975f + 0.125f * std::sin(std::chrono::duration<float>(std::chrono::steady_clock::now() - s_Start).count() * kTwoPi);     // the primitives' outline's: 1 Hz
                        const float Growth = 0.3f * std::max({ pMesh->m_OutlineHalf.m_X, pMesh->m_OutlineHalf.m_Y, pMesh->m_OutlineHalf.m_Z });                   // the primitives': 0.15 on a half of 0.5
                        Cmd.setPipelineInstance(m_OutlineInstance);
                        Cmd.setPushConstants(details::push_outline_obb
                            { W2C * (ToEye * D.m_L2W)
                            , xmath::fvec4(float(View.m_Width), float(View.m_Height), 6.0f, 0.08f)
                            , xmath::fvec4(std::min(1.0f, Pulse), std::min(1.0f, Pulse * 0.65f), std::min(1.0f, Pulse * 0.15f), 1.0f)
                            , xmath::fvec4(pMesh->m_OutlineCenter.m_X, pMesh->m_OutlineCenter.m_Y, pMesh->m_OutlineCenter.m_Z, Growth)
                            , xmath::fvec4(pMesh->m_OutlineHalf.m_X, pMesh->m_OutlineHalf.m_Y, pMesh->m_OutlineHalf.m_Z, 32.0f) });
                        Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_OutlineVB));
                        Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_OutlineIB));
                        Cmd.Draw(pMesh->m_nOutlineIndices);
                    }
                    break;
                }
#endif

            if( bMain ) Cmd.WriteTimestamp(m_Timestamps, 2);
            if( pSplit ) Cmd.WriteTimestamp(*pSplit, 2);
        }
    }

    //------------------------------------------------------------------------------------------------

    // The editor's view of huge meshes (m_bBoxesOnly, m_FrameBudgetMs): per visible cluster draw, which of its clusters the scene pass leaves out (m_ClusterSkip) and the boxes
    // the editor draws instead (m_NotDrawn). Boxes only: every cluster mesh is its box. A budget: the clusters in the view, the biggest on the screen first (the size of their
    // box over its distance to the eye), while they fit the allowance of indices (updateBudget); the others are their boxes. Clusters outside the view: left out, no box
    void instance::planClusters( const scene& Scene, const xmath::fvec3& Eye, const frustum& Frustum ) noexcept
    {
        constexpr std::size_t kMaxBoxes = 8192;                 // the editor draws them as lines: the most important ones (a budget's are in order of size on the screen)
        const auto AddBox = [&]( const xmath::fmat4& L2W, const xmath::fbbox& B, std::uint64_t Entity )
        {
            if( m_NotDrawn.size() >= kMaxBoxes ) return;
            const xmath::fvec3 C = B.getCenter(), E = B.m_Max - B.m_Min;
            wire_box Box;
            for( int r = 0; r < 3; ++r )
            {
                for( int k = 0; k < 3; ++k ) Box.m_Rows[r][k] = L2W(r, k) * E[k];
                Box.m_Rows[r][3] = L2W(r, 0) * C.m_X + L2W(r, 1) * C.m_Y + L2W(r, 2) * C.m_Z + L2W(r, 3);
            }
            Box.m_Entity = Entity;
            m_NotDrawn.push_back(Box);
        };

        m_ClusterPicks.clear();
        std::uint64_t Visible = 0;
        const xmath::fmat4 ToEye = xmath::fmat4::fromTranslation(-Eye);
        for( const auto& Vis : m_Visible )
        {
            if( Vis.m_pMesh->m_bCluster == false ) continue;
            const auto& D = Scene.m_Draws[Vis.m_iDraw];
            const auto& G = Vis.m_pMesh->m_Geom;
            const std::uint32_t Base = std::uint32_t(m_ClusterSkip.size());
            m_ClusterBase[Vis.m_iDraw] = Base;
            m_ClusterPlanKey[Vis.m_iDraw] = ClusterPlanKey(D);
            m_ClusterSkip.resize(m_ClusterSkip.size() + G.m_Clusters.size(), m_bBoxesOnly ? cluster_boxed_v : std::uint8_t(0));
            if( m_bBoxesOnly ) { AddBox(D.m_L2W, G.m_BBox, D.m_UserID); continue; }

            const xmath::fmat4 L2w = ToEye * D.m_L2W;           // camera-centered, as the frustum
            for( const auto& Sub : G.m_Submeshes )
                for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                {
                    const auto& B = G.m_Clusters[c].m_BBox;
                    xmath::fbbox W;
                    for( int i = 0; i < 8; ++i )
                    {
                        const xmath::fvec3 P = L2w * xmath::fvec3( i & 1 ? B.m_Max.m_X : B.m_Min.m_X, i & 2 ? B.m_Max.m_Y : B.m_Min.m_Y, i & 4 ? B.m_Max.m_Z : B.m_Min.m_Z );
                        if( i == 0 ) W.m_Min = W.m_Max = P;
                        else { W.m_Min = xmath::fvec3::Min(W.m_Min, P); W.m_Max = xmath::fvec3::Max(W.m_Max, P); }
                    }
                    if( !Frustum.Overlaps(W) ) { m_ClusterSkip[Base + c] = cluster_culled_v; continue; }
                    const xmath::fvec3 Center = (W.m_Min + W.m_Max) * 0.5f, Half = (W.m_Max - W.m_Min) * 0.5f;
                    const float R2 = Half.m_X * Half.m_X + Half.m_Y * Half.m_Y + Half.m_Z * Half.m_Z;
                    const float D2 = std::max(Center.m_X * Center.m_X + Center.m_Y * Center.m_Y + Center.m_Z * Center.m_Z, 1.0e-6f);
                    m_ClusterPicks.push_back({ R2 / D2, Base + c, G.m_Clusters[c].m_nIndices, Vis.m_iDraw, c, &G });
                    Visible += G.m_Clusters[c].m_nIndices;
                }
        }
        m_IndicesVisible          = Visible;
        m_BudgetStats.m_nClusters = int(m_ClusterPicks.size());
        m_BudgetStats.m_nDrawn    = int(m_ClusterPicks.size());
        if( m_bBoxesOnly ) { m_BudgetStats.m_nDrawn = 0; return; }
        if( m_IndexAllowance <= 0 || double(Visible) <= m_IndexAllowance ) return;

        // Over the allowance: the biggest on the screen first, until the next one does not fit (the first one always does)
        std::sort(m_ClusterPicks.begin(), m_ClusterPicks.end(), []( const cluster_pick& a, const cluster_pick& b ) { return a.m_Priority > b.m_Priority; });
        std::uint64_t Drawn  = 0;
        int           nDrawn = 0;
        bool          bFull  = false;
        for( const auto& P : m_ClusterPicks )
        {
            if( !bFull && (nDrawn == 0 || double(Drawn + P.m_nIndices) <= m_IndexAllowance) ) { Drawn += P.m_nIndices; ++nDrawn; continue; }
            bFull = true;                                       // the first that does not fit: it and every one after it are their boxes
            m_ClusterSkip[P.m_iSkip] = cluster_boxed_v;
            const auto& D = Scene.m_Draws[P.m_iDraw];
            AddBox(D.m_L2W, P.m_pGeom->m_Clusters[P.m_iCluster].m_BBox, D.m_UserID);
        }
        m_BudgetStats.m_nDrawn = nDrawn;
    }

    // The frame budget's allowance (planClusters), decided over a few frames at a time (what the last change did is in them): while the frames are slower than the budget it
    // shrinks with them (to a floor: something is always drawn, whatever else makes the frame slow), while they are well under it grows back, to no limit once all fits
    void instance::updateBudget( void ) noexcept
    {
        const auto   Now = std::chrono::steady_clock::now();
        const double Ms  = m_LastRender == std::chrono::steady_clock::time_point{} ? 0.0 : std::chrono::duration<double, std::milli>(Now - m_LastRender).count();
        m_LastRender = Now;
        if( m_FrameBudgetMs <= 0 || m_bBoxesOnly ) { m_IndexAllowance = 0; m_BudgetWindowMs = 0; m_BudgetWindowFrames = 0; return; }
        if( Ms <= 0 || Ms > 1000 ) return;                      // the first frame, or a stall (a load, a breakpoint): not the geometry's doing

        m_BudgetWindowMs += Ms;
        if( ++m_BudgetWindowFrames < 4 ) return;
        const double FrameMs = m_BudgetWindowMs / m_BudgetWindowFrames;
        m_BudgetWindowMs     = 0;
        m_BudgetWindowFrames = 0;
        m_BudgetStats.m_FrameMs = float(FrameMs);

        constexpr double kMinIndices = 3.0 * 250000;           // 250K triangles
        const double     Budget      = double(m_FrameBudgetMs);
        const double     Visible     = double(m_IndicesVisible);
        if( FrameMs > Budget * 1.05 )
        {
            const double From = m_IndexAllowance > 0 ? std::min(m_IndexAllowance, Visible) : Visible;
            m_IndexAllowance  = std::max(kMinIndices, From * std::clamp(Budget / FrameMs, 0.5, 0.9));
        }
        else if( m_IndexAllowance > 0 && FrameMs < Budget * 0.8 )
        {
            m_IndexAllowance *= 1.2;
            if( m_IndexAllowance >= Visible ) m_IndexAllowance = 0;
        }
    }

    //------------------------------------------------------------------------------------------------

    // The GI's material capture, one job at a time: recorded (the mesh's surfels sorted by material slot, each a 2 x 2 quad through its
    // slot's material in capture mode: the albedo in the top half, the emission in the bottom), then a few frames later (the GPU done
    // with it) read back: each surfel's color the mean of its block, into the solver
    void instance::processGICapture( xgpu::window& Window ) noexcept
    {
        if( m_GICaptures.empty() || !m_pGIWorld ) return;
        auto&       J     = m_GICaptures.front();
        const auto  Frame = m_Device.getFrameIndex();
        const auto* pM    = findMesh(J.m_Mesh);
        const gi::surfel_mesh* pG = pM == nullptr ? nullptr : J.m_Section < 0 ? pM->m_pGI.get()
                                  : std::size_t(J.m_Section) < pM->m_GISections.size() ? pM->m_GISections[std::size_t(J.m_Section)].m_pGI.get() : nullptr;
        if( pG == nullptr ) { m_GICaptures.pop_front(); return; }     // (its buffer goes with it)
        const auto& G     = *pG;
        std::vector<gi::surfel_mesh::surfel_material> TriMats;
        std::vector<gi::v3>               TriN;
        if( J.m_bTriangles )
        {
            const std::size_t End = std::min<std::size_t>(G.m_Tris.size(), J.m_First + std::size_t(gi_capture_max_v));
            for( std::size_t t = J.m_First; t < End; ++t )
            {
                const auto& T = G.m_Tris[t];
                gi::surfel_mesh::surfel_material S;
                S.m_iMaterial = t < G.m_TriMaterial.size() ? G.m_TriMaterial[t] : 0u;
                if( G.m_UV.size() == G.m_Pos.size() )
                {
                    const auto& A = G.m_UV[T[0]]; const auto& B = G.m_UV[T[1]]; const auto& C = G.m_UV[T[2]];
                    S.m_U        = (A[0] + B[0] + C[0]) / 3;
                    S.m_V        = (A[1] + B[1] + C[1]) / 3;
                    S.m_UVRadius = 0.5f * std::sqrt(0.5f * std::abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])));     // a square of its UV area
                }
                TriMats.push_back(S);
                const gi::v3 F = (G.m_Pos[T[1]] - G.m_Pos[T[0]]).Cross(G.m_Pos[T[2]] - G.m_Pos[T[0]]);
                TriN.push_back(F.Length() > 1e-12f ? F.NormalizeCopy() : G.m_Nrm[T[0]]);
            }
        }
        const auto& Mats  = J.m_bTriangles ? TriMats : G.m_SurfelMaterial;
        const int   n     = int(Mats.size());

        if( !J.m_bRecorded )
        {
            if( n == 0 || n > gi_capture_max_v ) { m_GICaptures.pop_front(); return; }
            J.m_Order.resize(n);
            for( int i = 0; i < n; ++i ) J.m_Order[i] = i;
            std::stable_sort(J.m_Order.begin(), J.m_Order.end(), [&](int a, int b) { return Mats[a].m_iMaterial < Mats[b].m_iMaterial; });

            std::vector<xmath::fvec4> Data(2 * std::size_t(n));
            const auto& L0 = G.m_Levels[0];
            for( int i = 0; i < n; ++i )
            {
                const auto& S = Mats[J.m_Order[i]];
                Data[2 * i]     = xmath::fvec4(S.m_U, S.m_V, S.m_UVRadius, 0);
                Data[2 * i + 1] = xmath::fvec4(J.m_bTriangles ? TriN[J.m_Order[i]] : L0[J.m_Order[i]].m_Normal, 0);
            }
            if( m_Device.Create(J.m_Data, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(xmath::fvec4), .m_EntryCount = 2 * n, .m_pData = Data.data() }) )
            { m_GICaptures.pop_front(); return; }

            (void)m_GICaptureVB.MemoryMap(0, 8 * std::size_t(n), [&](void* p)
            {
                auto* V = static_cast<float*>(p);
                for( int Pass = 0; Pass < 2; ++Pass )
                    for( int i = 0; i < n; ++i )
                    {
                        const float x0 = float(2 * (i % (gi_capture_w_v / 2))), y0 = float(2 * (i / (gi_capture_w_v / 2)) + Pass * gi_capture_h_v / 2);
                        const float X[4] = { x0, x0 + 2, x0 + 2, x0 }, Y[4] = { y0, y0, y0 + 2, y0 + 2 };
                        for( int c = 0; c < 4; ++c )
                        {
                            float* q = V + 2 * (std::size_t(Pass) * 4 * n + 4 * std::size_t(i) + c);
                            q[0] = X[c] / gi_capture_w_v * 2 - 1;
                            q[1] = Y[c] / gi_capture_h_v * 2 - 1;
                        }
                    }
            });

            auto  Cmd = Window.StartRenderPass(m_GICapturePass);
            auto& ViewBlock = m_StaticGeomUBO.allocEntry<details::ubo_geom_view>();
            std::memset(&ViewBlock, 0, sizeof(ViewBlock));
            auto& Light = m_LightUBO.allocEntry<details::ubo_lighting>();
            std::memset(&Light, 0, sizeof(Light));
            // GIMesh: the empty one. The capture's GI reference is not gi_none (its debug bits say "capture"), so the material shader still runs
            // its GI fallback (SurfelGILocalMean at the capture's position 0) before the capture output replaces it: with the mesh's read grid
            // that walked every list of the cell at the mesh's origin for every capture pixel (the Golem: about 4 s of GPU a batch, 31 batches)
            std::array<xgpu::buffer*, 9> Buffers{ &J.m_Data, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams };
            for( int a = 0; a < n; )
            {
                const std::uint32_t Slot = Mats[J.m_Order[a]].m_iMaterial;
                int b = a;
                while( b < n && Mats[J.m_Order[b]].m_iMaterial == Slot ) ++b;
                auto* pMat  = Slot < J.m_Materials.size() ? findMaterial(J.m_Materials[Slot]) : nullptr;
                auto* pInst = pMat ? getCaptureInstance(*pMat) : &m_GIConstCaptureInstance;
                if( pInst == nullptr ) pInst = &m_GIConstCaptureInstance;
                Cmd.setPipelineInstance(*pInst, std::span{ Buffers });
                if( pMat && pMat->m_bTable && pInst != &m_GIConstCaptureInstance ) Cmd.setTextureTable(pMat->m_Table);
                Cmd.setDynamicUBO(m_LightUBO, 1);
                Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                Cmd.setBuffer(m_GICaptureVB);
                Cmd.setBuffer(m_GICaptureIB);
                for( int Pass = 0; Pass < 2; ++Pass )
                {
                    Cmd.setPushConstants(details::push_static_geom{ Pass ? std::uint32_t(-n) : 0u, J.m_Slot, 0u, (3u << 9) | (Pass ? 0x800u : 0u), pInst == &m_GIConstCaptureInstance ? 0u : pMat->m_ParamBase });
                    Cmd.Draw(6 * (b - a), 6 * a, Pass * 4 * n);
                }
                a = b;
            }
            J.m_bRecorded = true;
            J.m_Frame     = Frame;
            return;
        }

        if( Frame < J.m_Frame + ring_frames_v + 1 ) return;         // the GPU is done with it after the frames in flight
        // measurement only (GI review): the read back's steps, printed when the whole takes more than half a second
        const auto TCap = std::chrono::steady_clock::now();
        double CapMs[6] = {};
        auto CapLap = [&, T = TCap](int k) mutable { const auto N = std::chrono::steady_clock::now(); CapMs[k] += std::chrono::duration<double, std::milli>(N - T).count(); T = N; };
        struct cap_report { double* p; std::chrono::steady_clock::time_point t0; ~cap_report() { const double All = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if( All > 500 ) std::printf("GICAPTURE %.0f ms: read back %.0f, unpack %.0f, surfel/triangle albedo %.0f, emission %.0f, all albedo %.0f, buildMeshGI %.0f\n", All, p[0], p[1], p[2], p[3], p[4], p[5]), std::fflush(stdout); } } CapReport{ CapMs, TCap };
        std::vector<std::byte> Raw;
        int W = 0, H = 0;
        const bool bRead = m_Device.ReadTexture(m_GICaptureTex, Raw, W, H) == nullptr;
        CapLap(0);
        if( bRead && W == gi_capture_w_v && H == gi_capture_h_v )
        {
            const auto Half = details::HalfToFloat;
            const auto* P = reinterpret_cast<const std::uint16_t*>(Raw.data());
            const auto Texel = [&](int x, int y) { const auto* t = P + 4 * (std::size_t(y) * W + x); return xmath::fvec3(Half(t[0]), Half(t[1]), Half(t[2])); };
            std::vector<xmath::fvec3> Albedo(n), Emission(n);
            for( int i = 0; i < n; ++i )
            {
                const int x0 = 2 * (i % (W / 2)), y0 = 2 * (i / (W / 2));
                const auto Block = [&](int y) { return (Texel(x0, y) + Texel(x0 + 1, y) + Texel(x0, y + 1) + Texel(x0 + 1, y + 1)) * 0.25f; };
                Albedo[J.m_Order[i]]   = Block(y0);
                Emission[J.m_Order[i]] = Block(y0 + H / 2);                           // as captured (the world scales it: scene::gi_settings::m_EmissiveScale)
            }
            CapLap(1);
            // Captured from materials the instance no longer has (they were made again while this was in flight): not published - its new capture is queued
            const bool bCurrent = std::size_t(J.m_GIId) < m_GITrack.size() && m_GITrack[J.m_GIId].m_Materials == J.m_Materials;
            if( bCurrent && !J.m_bTriangles ) { m_pGIWorld->SetSurfelMaterial(J.m_GIId, std::move(Albedo), std::move(Emission)); CapLap(2); }
            else if( bCurrent )
            {
                m_pGIWorld->SetTriangleAlbedo(J.m_GIId, J.m_First, Albedo);
                CapLap(2);
                m_pGIWorld->SetTriangleEmission(J.m_GIId, J.m_First, Emission);
                CapLap(3);
                if( J.m_Section < 0 && J.m_First + std::size_t(n) >= G.m_Tris.size() )      // its last batch: the SURFACE_COLOR view's table (the mesh's first instance)
                {
                    const auto& All = m_pGIWorld->TriangleAlbedo(J.m_GIId);
                    CapLap(4);
                    if( auto* pSlot = const_cast<mesh_slot*>(pM); pSlot->m_GIBuffer.m_Private && pSlot->m_bTriColors == false )
                    {
                        std::vector<xmath::fvec3> C(All.begin(), All.end());
                        (void)buildMeshGI(*pSlot, C);
                        CapLap(5);
                        pSlot->m_bTriColors = true;
                    }
                }
                else if( J.m_Section >= 0 && J.m_First + std::size_t(n) >= G.m_Tris.size() )   // a section's last batch: its buffer again, with its triangles' table
                {
                    // A section's triangles are its clusters' (SectionGeometry: in the whole's order), so a cluster's first is the count of the
                    // section's triangles before it; the clusters of other sections name nothing (their draws bind their own section's buffer)
                    auto* pSlot = const_cast<mesh_slot*>(pM);
                    const auto& Cl = pSlot->m_Geom.m_Clusters;
                    std::vector<std::uint32_t> Order(Cl.size()), First(Cl.size(), 0);
                    for( std::uint32_t c = 0; c < Order.size(); ++c ) Order[c] = c;
                    std::sort(Order.begin(), Order.end(), [&](std::uint32_t x, std::uint32_t y) { return Cl[x].m_iIndex < Cl[y].m_iIndex; });
                    std::uint32_t Off = 0;
                    for( const auto c : Order )
                        if( c < pSlot->m_GIClusterSection.size() && pSlot->m_GIClusterSection[c] == std::uint16_t(J.m_Section) ) { First[c] = Off; Off += Cl[c].m_nIndices / 3; }
                    const auto& All = m_pGIWorld->TriangleAlbedo(J.m_GIId);
                    if( Off == All.size() )
                    {
                        auto& S = pSlot->m_GISections[std::size_t(J.m_Section)];
                        const std::vector<xmath::fvec3> Colors(All.begin(), All.end());
                        const auto Packed = PackMeshGI(*S.m_pGI, First, Colors);
                        xgpu::buffer B;         // (swapped in: the old one's destructor queues it in the device's deferred destroy)
                        if( !m_Device.Create(B, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = int(Packed.size()), .m_pData = const_cast<std::uint32_t*>(Packed.data()) }) )
                            S.m_Buffer.m_Private.swap(B.m_Private);
                        CapLap(5);
                    }
                }
            }
        }
        m_GICaptures.pop_front();                                       // its buffer goes with it (xgpu's deferred destroy)
    }

    //------------------------------------------------------------------------------------------------

    // The selection's outline (Scene.m_SelectedUser's draw), made of two passes. The mask: the object's own triangles through the casters' vertex
    // paths (rigid, skin, plain meshes) into a target of the view's size with a depth buffer of its own (the nearest fragment stays); the fragment
    // writes R 1 (covered) and G (that fragment is not behind what the scene drew: the scene's depth sampled). The composite: a fullscreen triangle
    // over the tone-mapped image, scissored to the object's projected bounds (and the outline's reach): the edge from the fraction of a ring of
    // taps that is selection (xlionrender_outline_composite_frag.glsl). Nothing is selected: nothing runs.
    void instance::renderOutline( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye ) noexcept
    {
        if( Scene.m_SelectedUser == ~0ull || !m_bOutlineReady ) return;
        std::size_t iDraw = Scene.m_Draws.size();
        for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i ) if( Scene.m_Draws[i].m_UserID == Scene.m_SelectedUser ) { iDraw = i; break; }
        if( iDraw >= Scene.m_Draws.size() || iDraw >= m_DrawSlot.size() || iDraw >= m_DrawWorld.size() ) return;
        const auto& D     = Scene.m_Draws[iDraw];
        const auto* pMesh = findMesh(D.m_Mesh);
        if( pMesh == nullptr || (pMesh->m_bCluster && m_DrawSlot[iDraw] == instance_slots::none_v) ) return;
        if( pMesh->m_bCluster && m_bBoxesOnly ) return;                 // drawn as its box (the editor draws it in the selection's color)
        if( createOutlineTargets(View) ) return;

        // The object's bounds on the screen (the world box's 8 corners): where the composite has anything to do. A corner behind the eye: the whole view
        constexpr float kRadius = 3.0f;
        const xmath::fbbox& B = m_DrawWorld[iDraw];
        float MinX = 1e30f, MinY = 1e30f, MaxX = -1e30f, MaxY = -1e30f;
        bool  bWhole = false;
        for( int i = 0; i < 8; ++i )
        {
            const xmath::fvec3 P{ (i & 1 ? B.m_Max.m_X : B.m_Min.m_X) - Eye.m_X, (i & 2 ? B.m_Max.m_Y : B.m_Min.m_Y) - Eye.m_Y, (i & 4 ? B.m_Max.m_Z : B.m_Min.m_Z) - Eye.m_Z };
            const xmath::fvec4 C = W2C * xmath::fvec4(P, 1.0f);
            if( C.m_W <= 1.0e-4f ) { bWhole = true; break; }
            const float X = (C.m_X / C.m_W * 0.5f + 0.5f) * float(View.m_Width), Y = (C.m_Y / C.m_W * 0.5f + 0.5f) * float(View.m_Height);
            MinX = std::min(MinX, X); MaxX = std::max(MaxX, X); MinY = std::min(MinY, Y); MaxY = std::max(MaxY, Y);
        }
        int X0 = 0, Y0 = 0, X1 = View.m_Width, Y1 = View.m_Height;
        if( !bWhole )
        {
            const float Pad = kRadius + 3.0f;
            X0 = std::clamp(int(std::floor(MinX - Pad)), 0, View.m_Width);  X1 = std::clamp(int(std::ceil(MaxX + Pad)), 0, View.m_Width);
            Y0 = std::clamp(int(std::floor(MinY - Pad)), 0, View.m_Height); Y1 = std::clamp(int(std::ceil(MaxY + Pad)), 0, View.m_Height);
            if( X1 <= X0 || Y1 <= Y0 ) return;       // off the screen
        }

        {
            auto Cmd = Window.StartRenderPass(View.m_MaskPass);
            const xmath::fmat4 ToEye = xmath::fmat4::fromTranslation(-Eye);
            auto& ViewBlock = m_ShadowUBO.allocEntry<details::ubo_geom_view>();       // camera-centered, as the scene pass's
            ViewBlock.m_w2C         = W2C;
            ViewBlock.m_WorldOffset = xmath::fvec4(m_Origin - Eye, 0);
            if( pMesh->m_bCluster == false )
            {
                Cmd.setPipelineInstance(View.m_MaskInstance[0]);
                Cmd.setPushConstants(details::push_forward{ ToEye * D.m_L2W, W2C });
                Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_VertexBuffer));
                Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_IndexBuffer));
                Cmd.Draw(pMesh->m_nIndices);
            }
            else
            {
                const auto& G     = pMesh->m_Geom;
                const bool  bSkin = G.m_Kind == geom_kind::SKIN;
                std::array<xgpu::buffer*, 4> StaticBuffers{ G.m_pClusterData, bSkin ? G.m_pBones : &m_InstanceBuffer, &m_InstanceBuffer, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy };
                Cmd.setPipelineInstance(View.m_MaskInstance[bSkin ? 2 : 1], std::span{ StaticBuffers.data(), bSkin ? std::size_t{4} : std::size_t{2} });
                Cmd.setDynamicUBO(m_ShadowUBO, 0);
                BindClusterStreams(Cmd, G);
                const std::uint32_t Slot = m_DrawSlot[iDraw];
                for( const auto& Sub : G.m_Submeshes )
                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
                        if( isClusterSkipped(iDraw, c) ) continue;              // the scene pass left it out (its box, or outside the view)
                        const auto& C = G.m_Clusters[c];
                        if( bSkin ) Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Slot });
                        else        Cmd.setPushConstants(details::push_static_geom{ c, Slot });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                    }
            }
        }
        {
            auto Cmd = Window.StartRenderPass(View.m_OutlinePass);
            constexpr float kTwoPi = 6.28318530717958647692f;
            static const auto s_Start = std::chrono::steady_clock::now();
            const float Pulse = 0.975f + 0.125f * std::sin(std::chrono::duration<float>(std::chrono::steady_clock::now() - s_Start).count() * kTwoPi);       // the primitives' outline's: 1 Hz
            Cmd.setScissor(X0, Y0, X1 - X0, Y1 - Y0);
            Cmd.setPipelineInstance(View.m_OutlineInstance);
            Cmd.setPushConstants(details::push_outline_composite{ xmath::fvec4(std::min(1.0f, Pulse), std::min(1.0f, Pulse * 0.65f), std::min(1.0f, Pulse * 0.15f), 1.0f), xmath::fvec4(kRadius, 0.35f, 0.12f, 0.0f) });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
        }
    }

    //------------------------------------------------------------------------------------------------

    void instance::Render( xgpu::window& Window, const scene& Scene, render_view& View, xgpu::tools::view& CameraIn, const overlay_fn& Overlay ) noexcept
    {
        assert(m_bReady && View.m_Width > 0);
        static const bool s_bNoBudget = std::getenv("XLION_NO_FRAME_BUDGET") != nullptr;    // measurement only (GI review): every cluster drawn, whatever the host's budget
        if( s_bNoBudget ) m_FrameBudgetMs = 0;

        // The render target's viewport is for this render: on a copy. The host's own view keeps the viewport it has (the editor's panel, in screen
        // coordinates): its click-pick of the same frame builds its ray from it (it used to get the target's: every click missed)
        xgpu::tools::view Camera = CameraIn;
        Camera.setViewport({ 0, 0, View.m_Width, View.m_Height });

        // Camera-centered world (same as the editors): precision stays near the eye in large worlds
        const xmath::fvec3 Eye     = Camera.getPosition();
        const xmath::fmat4 W2C     = Camera.getW2C() * xmath::fmat4::fromTranslation(Eye);

        const float ProjScale = float(View.m_Height) / (2.0f * xmath::Tan(xmath::radian{ Camera.getFov().m_Value * 0.5f }));    // pixels per unit at distance 1
        using clock = std::chrono::steady_clock;
        const auto Ms = [](clock::time_point A, clock::time_point B) { return std::chrono::duration<double, std::milli>(B - A).count(); };
        const auto T0 = clock::now();
        m_nSceneBinds = 0;
        // Measurement only (XLION_GPU_TIMES): the whole Render on the GPU (its start, after the GI and the captures, after the tone map, its end)
        static const bool s_bTimes = std::getenv("XLION_GPU_TIMES") != nullptr;
        static std::unordered_map<const void*, xgpu::timestamps> s_TS;      // ponytail: per renderer, never erased (a measurement hook)
        xgpu::timestamps* pTS = nullptr;
        if( s_bTimes )
        {
            auto [It, bNew] = s_TS.try_emplace(this);
            if( bNew && m_Device.Create(It->second, { .m_Count = 7 }) ) s_TS.erase(It);
            else pTS = &It->second;
        }
        if( pTS )
        {
            static std::vector<std::uint64_t> s_Ns;
            static double s_Sum[6] = {}; static int s_n = 0;
            if( pTS->getResults(s_Ns) && s_Ns.size() >= 7 && s_Ns[6] > s_Ns[0] )
            {
                for( int k = 0; k < 6; ++k ) s_Sum[k] += (s_Ns[k + 1] - s_Ns[k]) * 1e-6;
                if( ++s_n == 120 ) { std::printf("RENDERGPU environment %.2f, instances %.2f, GI update %.2f, captures %.2f, shadows+scene+tonemap %.2f, overlays+outline %.2f ms\n", s_Sum[0] / s_n, s_Sum[1] / s_n, s_Sum[2] / s_n, s_Sum[3] / s_n, s_Sum[4] / s_n, s_Sum[5] / s_n); std::fflush(stdout); s_n = 0; for( auto& v : s_Sum ) v = 0; }
            }
            Window.WriteTimestamp(*pTS, 0);
        }
        // The probe atlas and the shadow atlas are bound by every lit pipeline, the GI capture's below included: cleared before anything samples them
        // (the first frame's capture used to read them undefined: a Vulkan validation error)
        if( !m_bProbeAtlasReady ) { auto Cmd = Window.StartRenderPass(m_ProbeAtlasClearPass); m_bProbeAtlasReady = true; }
        if( !m_bShadowAtlasReady ) { auto Cmd = Window.StartRenderPass(m_ShadowClearPass); m_bShadowAtlasReady = true; }
        updateEnvironment(Window, Scene);
        if( pTS ) Window.WriteTimestamp(*pTS, 1);
        updateInstances(Window, Scene, Eye);
        if( pTS ) Window.WriteTimestamp(*pTS, 2);
        updateGI(Window, Scene, Eye, ProjScale);
        if( pTS ) Window.WriteTimestamp(*pTS, 3);                                   // CPU (getGIStats); the probes' captures read it too
        {
            const auto TCapture = std::chrono::steady_clock::now();
            processGICapture(Window);                           // the surfels' material colors (one mesh at a time)
            if( !m_GIFrames.empty() ) m_GIFrames.back().m_Stats.m_CaptureMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - TCapture).count();
        }
        const auto T1 = clock::now();
        if( pTS ) Window.WriteTimestamp(*pTS, 4);
        Window.WriteTimestamp(m_Timestamps, 0);                 // GPU: the shadow passes start here (after the instance copy)
        renderShadows(Window, Scene, Eye, ProjScale);
        updateProbes(Window, Scene, Eye);
        const auto T2 = clock::now();

        updateBudget();
        const auto T2b = clock::now();
        renderScenePass(Window, Scene, View, W2C, Eye, true);
        if( m_bVirtualShadows && Scene.m_Sun.m_bCastShadows ) vsmRequestPass(Window, View, W2C, Eye);      // (the main view's depth: the pages it asks for)
        const auto T2c = clock::now();

        // HDR -> exposure -> LionMapper -> sRGB output (alpha = coverage)
        {
            auto Cmd = Window.StartRenderPass(View.m_OutputPass);
            Cmd.setPipelineInstance(View.m_ToneMapInstance);
            Cmd.setPushConstants(details::push_tonemap{ { View.m_Exposure, 0, 0, 0 }, View.m_Background });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
            Cmd.WriteTimestamp(m_Timestamps, 3);
            if( pTS ) Cmd.WriteTimestamp(*pTS, 5);
        }

        // Editor overlays over the tone-mapped image, depth-tested against the scene
        if( Overlay )
        {
            auto Cmd = Window.StartRenderPass(View.m_OverlayPass);
            Overlay(Cmd);
        }

        // The selection's outline last, over the overlays too: the host's grid is opaque floor and painted over the outline of an object standing on it
        renderOutline(Window, Scene, View, W2C, Eye);
        if( pTS ) Window.WriteTimestamp(*pTS, 6);

        const auto T3 = clock::now();
        m_CPUTimes = { Ms(T0, T1), Ms(T1, T2), m_LightBuildMs, Ms(T0, T3) };
        // Measurement only (XLION_LEDGER): the lighting ledger (documentation/lighting_strategy.md sec. 10), median, p95 and p99 of 120 frames
        // per row, per renderer (its view's size names it). Rows that do not exist yet (reconstruction, rays, cache maintenance) print 0;
        // direct shading is fused in the scene pass, so the scene pass is the row (paired variants attribute it). The GPU times are the
        // previous frames': each timing set once, whoever read it (getGPUTimes: the GI's log or the host may have)
        static const bool s_bLedger = std::getenv("XLION_LEDGER") != nullptr;
        if( s_bLedger )
        {
            struct ledger { std::array<std::vector<double>, 4> m_Rows; std::uint64_t m_Serial = 0; };
            static std::unordered_map<const void*, ledger> s_Ledger;    // ponytail: per renderer, never erased (a measurement hook)
            auto& L      = s_Ledger[this];
            auto& s_Rows = L.m_Rows;                                    // CPU records + grid, CPU shadow preparation (+ probes), GPU shadows, GPU scene pass
            s_Rows[0].push_back(m_LightBuildMs); s_Rows[1].push_back(Ms(T1, T2));
            gpu_times G;
            (void)getGPUTimes(G);
            if( m_GPUTimesSerial != L.m_Serial && m_TimestampNs.size() >= 3 )
            {
                L.m_Serial = m_GPUTimesSerial;
                s_Rows[2].push_back((m_TimestampNs[1] - m_TimestampNs[0]) * 1e-6); s_Rows[3].push_back((m_TimestampNs[2] - m_TimestampNs[1]) * 1e-6);
            }
            if( s_Rows[0].size() == 120 )
            {
                double Med[4] = {}, P95[4] = {}, P99[4] = {};
                for( int r = 0; r < 4; ++r ) if( auto& V = s_Rows[r]; !V.empty() )
                {
                    std::sort(V.begin(), V.end());
                    Med[r] = V[V.size() / 2]; P95[r] = V[std::min(V.size() - 1, V.size() * 95 / 100)]; P99[r] = V[std::min(V.size() - 1, V.size() * 99 / 100)];
                    V.clear();
                }
                std::printf("LEDGER view %dx%d records+grid cpu %.3f/%.3f/%.3f, shadow prep gpu %.3f/%.3f/%.3f cpu %.3f/%.3f/%.3f, direct shading (scene pass) gpu %.3f/%.3f/%.3f, reconstruction 0, rays 0, cache 0 ms (median/p95/p99 of 120); lights %d (dropped %d), shadowed %d (fallback %d, stale %d), overflow cells %d; sun pages %s: request pass %.3f ms, rendered %d + net %d, rejected %d, resident %d + net %d of %d slots, load %.2f, dynamic bias %.2f, pixels by their level %.1f%% / coarser %.1f%% / none %.1f%%\n"
                           , View.m_Width, View.m_Height, Med[0], P95[0], P99[0], Med[2], P95[2], P99[2], Med[1], P95[1], P99[1], Med[3], P95[3], P99[3], m_LightStats.m_nLights, m_LightStats.m_nDropped, m_LightStats.m_nShadowed
                           , m_LightStats.m_nShadowFallback, m_LightStats.m_nShadowStale, m_LightStats.m_nOverflowCells
                           , m_bVirtualShadows ? "on" : "off", m_VSMReport.m_RequestGpuMs, m_VSMReport.m_Pages.m_Rendered, m_VSMReport.m_Pages.m_RenderedPinned, m_VSMReport.m_Pages.m_Rejected, m_VSMReport.m_Pages.m_ResidentFine, m_VSMReport.m_Pages.m_ResidentPinned, m_VSM.getSetup().Slots(), m_VSMReport.m_Pages.m_PoolLoad, m_VSMReport.m_Pages.m_DynamicBias
                           , m_VSMReport.m_Counters[0] ? 100.0 * m_VSMReport.m_Counters[1] / m_VSMReport.m_Counters[0] : 0.0, m_VSMReport.m_Counters[0] ? 100.0 * m_VSMReport.m_Counters[2] / m_VSMReport.m_Counters[0] : 0.0
                           , m_VSMReport.m_Counters[0] ? 100.0 * m_VSMReport.m_Counters[3] / m_VSMReport.m_Counters[0] : 0.0);
                std::fflush(stdout);
            }
        }
        // Measurement only (XLION_FRAME_TIMES): Render's CPU parts, the mean of 120 frames
        static const bool s_bFrameTimes = std::getenv("XLION_FRAME_TIMES") != nullptr;
        if( s_bFrameTimes )
        {
            static double Sum[5] = {}, nB = 0, nD = 0; static int n = 0;
            nB += g_nSceneBinds; nD += g_nSceneDraws; g_nSceneBinds = g_nSceneDraws = 0;
            static double BMs = 0, DMs = 0; BMs += g_BindMs; DMs += g_DrawMs; g_BindMs = g_DrawMs = 0;
            const double V[5] = { Ms(T0, T1), Ms(T1, T2), Ms(T2, T2b), Ms(T2b, T2c), Ms(T2c, T3) };
            for( int k = 0; k < 5; ++k ) Sum[k] += V[k];
            if( ++n == 120 ) { std::printf("RENDERCPU up to the shadows %.2f, shadows+probes %.2f, budget %.2f, scene pass %.2f, tone map+overlays+outline %.2f ms; scene binds %.0f (%.2f ms), cluster draws %.0f (%.2f ms) a frame\n", Sum[0] / n, Sum[1] / n, Sum[2] / n, Sum[3] / n, Sum[4] / n, nB / n, BMs / n, nD / n, DMs / n); std::fflush(stdout); n = 0; nB = nD = 0; BMs = DMs = 0; for( auto& v : Sum ) v = 0; }
        }
    }
}
