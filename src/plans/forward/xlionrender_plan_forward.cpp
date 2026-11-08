// Plan A - forward. instance::Render lives here so each plan is one file to read.
#include "../../xlionrender_env.h"
#include "xlionrender_instance.h"
#include "plans/forward/xlionrender_plan_forward.h"
// The profiler (xprofile / Tracy): the frame's passes as CPU and GPU zones - in LIONRender.dll; the tests are built without it
#if __has_include("xprofile.h")
    #include "xprofile.h"
#else
    #define XPROFILE_SCOPE(NAME)
    #define XPROFILE_GPU_SCOPE(TARGET, NAME)
#endif
#include <map>
#include <chrono>
#include <optional>

namespace xlionrender
{
    namespace
    {
        constexpr std::uint32_t s_GISurfelPassComp[] = {
            #include "xlionrender_gi_surfel_pass_comp.h"
        };
        constexpr std::uint32_t s_GISurfelPassCompNoClustered[] = {
            #include "xlionrender_gi_surfel_pass_comp_noclustered.h"
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
        constexpr std::uint32_t s_GIRefComp[] = {           // the GI's ground truth (XLION_GI_REF)
            #include "xlionrender_gi_ref_comp.h"
        };
        constexpr std::uint32_t s_GIVoxComp[] = {           // the voxel rays
            #include "xlionrender_gi_vox_comp.h"
        };
        constexpr std::uint32_t s_GIClipComp[] = {          // experiment X1: the clipmap (XLION_GI_CLIP)
            #include "xlionrender_gi_clip_comp.h"
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
            const auto Put = [&](std::uint32_t Slot, const xmath::fmat4& L2W, const xmath::fmat4& Prev, const material& Mat, const entity_params& Ent)
            {
                pRing[m_RingUsed].set(L2W, Prev, m_Origin, Mat);
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
                    Put(Slot, D.m_L2W, D.m_L2W, D.m_Material, D.m_EntityParams);       // (a transient slot: no motion of its own)
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
    namespace { struct gi_times { xgpu::timestamps m_T; bool m_bOk = false; std::vector<std::uint64_t> m_Ns; double m_Sum[6] = {}; int m_n = 0; }; }
    static gi_times* GITimes(const void* pKey, xgpu::device& Device) noexcept
    {
        static const bool s_bOn = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
        if( !s_bOn ) return nullptr;
        static std::unordered_map<const void*, gi_times> s_Map;      // ponytail: per renderer, never erased (a measurement hook)
        auto [It, bNew] = s_Map.try_emplace(pKey);
        if( bNew ) It->second.m_bOk = Device.Create(It->second.m_T, { .m_Count = 6 }) == nullptr;
        if( !It->second.m_bOk ) return nullptr;
        auto& G = It->second;
        // (slots: 0 before Submit, 4 after it, 5 after the links' pump, 1 after the probes; 2 before Show, 3 after it)
        if( G.m_T.getResults(G.m_Ns) && G.m_Ns.size() >= 6 && G.m_Ns[3] > G.m_Ns[0] && G.m_Ns[1] >= G.m_Ns[5] && G.m_Ns[5] >= G.m_Ns[4] && G.m_Ns[4] >= G.m_Ns[0] )
        {
            for( int k = 0; k < 3; ++k ) G.m_Sum[k] += (G.m_Ns[k + 1] - G.m_Ns[k]) * 1e-6;
            G.m_Sum[3] += (G.m_Ns[4] - G.m_Ns[0]) * 1e-6; G.m_Sum[4] += (G.m_Ns[5] - G.m_Ns[4]) * 1e-6; G.m_Sum[5] += (G.m_Ns[1] - G.m_Ns[5]) * 1e-6;
            if( ++G.m_n == 120 ) { std::printf("GIGPU submit %.2f (the rays' slice %.2f, the links: relight + solve + ghosts %.2f, the probes %.2f), between %.2f, show %.2f ms\n", G.m_Sum[0] / G.m_n, G.m_Sum[3] / G.m_n, G.m_Sum[4] / G.m_n, G.m_Sum[5] / G.m_n, G.m_Sum[1] / G.m_n, G.m_Sum[2] / G.m_n); std::fflush(stdout); G.m_n = 0; for( auto& v : G.m_Sum ) v = 0; }
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
            // GroupNonUniformClustered only when the device reports VK_SUBGROUP_FEATURE_CLUSTERED_BIT (every platform)
            const std::span<const std::uint32_t> GISurfelPass = m_Device.hasSubgroupClustered()
                ? std::span<const std::uint32_t>{ s_GISurfelPassComp }
                : std::span<const std::uint32_t>{ s_GISurfelPassCompNoClustered };
            if( !m_pGITrace->Init(m_Device, { GISurfelPass, s_GIScatterComp, s_GICommitComp, s_GIOpsComp, s_GIFilterComp, s_GIExpandComp, s_GIMeansComp, s_GICubeComp, s_GIMigrateComp, s_GIReadBackComp, s_GILinksComp, s_GISolveComp, s_GIGhostsComp, s_GIProbesComp, s_GIClipComp, s_GIRefComp, s_GIVoxComp }) )
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
        if( m_pGITrace ) { XPROFILE_SCOPE("GI: collect"); m_pGITrace->Collect(W, GIFrame); }         // (the GPU's last slice; a read back asked for: into the world's arrays)
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
            if( xlionrender::Env("XLION_GI_TRANSPORT") ) s_TransportAsked[this] = true;
            (void)W.Dump(m_GIDumpPath.c_str()); (void)DumpGIFrames((m_GIDumpPath + ".frames.csv").c_str());
            m_GIDumpPath.clear(); m_bGIDumpAsked = false;
        }

        std::vector<int> DrawId(Scene.m_Draws.size(), -1);
        std::vector<std::vector<int>> DrawSecIds(Scene.m_Draws.size());     // GI sections: per draw its sections' world instances
        for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
        {
            XPROFILE_SCOPE("GI: register");
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
                // the masked slots' opacity masks (alpha-tested rays): from the cache, or a capture queued once a material (after the erase above: it removed them)
                std::erase_if(m_GIMaskWaiting, [&](const gi_mask_wait& M) { return M.m_GIId == Id; });
                static const bool s_bNoGIMasks = xlionrender::Env("XLION_GI_NO_MASKS") != nullptr;     // (measurement: no mask captures)
                for( std::size_t k = 0; k < (s_bNoGIMasks ? 0 : D.m_Materials.size()); ++k )
                {
                    const auto* pMat = findMaterial(D.m_Materials[k]);
                    if( pMat == nullptr || MaterialVisibilityOf(pMat->m_State).m_Opacity != material_visibility::opacity::MASKED || Mesh.m_UV.size() != Mesh.m_Pos.size() ) continue;
                    const std::uint64_t Key = (std::uint64_t(D.m_Materials[k].m_Index) << 32) | D.m_Materials[k].m_Generation;
                    if( const auto It = m_GIMaskCache.find(Key); It != m_GIMaskCache.end() ) { W.SetSlotMask(Id, std::uint32_t(k), It->second); continue; }
                    const bool bQueued = std::ranges::any_of(m_GIMaskWaiting, [&](const gi_mask_wait& M) { return M.m_Key == Key; });
                    m_GIMaskWaiting.push_back({ Id, std::uint32_t(k), Key });
                    if( !bQueued ) m_GICaptures.push_back({ .m_GIId = Id, .m_Mesh = D.m_Mesh, .m_Materials = D.m_Materials, .m_Slot = i < m_DrawSlot.size() ? m_DrawSlot[i] : 0u, .m_Section = Section, .m_MaskSlot = int(k) });
                }
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
            m_GITrack[Id].m_Render   = m_GIRender;
            m_GITrack[Id].m_DrawSlot = i < m_DrawSlot.size() ? m_DrawSlot[i] : 0u;
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
        XPROFILE_SCOPE("GI: retire and lights");
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
            const int Rays = std::max(16, std::atoi(xlionrender::Env("XLION_GI_TRANSPORT")));
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
        // lighting shades the emitters' direct light (its emitter lights: one switch, never both) - the GI keeps their bounces; switched:
        // every instance re-solves (what it shows changes)
        if( const bool bExternal = m_bEmitterLights || Scene.m_GI.m_bEmitterLights; W.m_Setup.m_bEmitterDirectExternal != bExternal )
        {
            W.m_Setup.m_bEmitterDirectExternal = bExternal;
            W.LowerAll();
        }
        // G3, G4 (gi_dynamic_master_plan.md): the rays on the GPU, over frames. This frame's slice dispatched after the update (it commits
        // what came in, picks when the batch is all in); the last frame's slice into the world before it (Collect, at the top: before
        // anything here changes the world, whose copy of the GPU's state must be current first)
        // Diagnostic (gi_dynamic_master_plan.md 7a, smoke/gi_startup.py --captures-first): with XLION_GI_HOLD_CAPTURES set, no solve until
        // the material captures have been done for 10 frames (the control: the solve never sees an uncaptured material)
        static const bool s_bHold = xlionrender::Env("XLION_GI_HOLD_CAPTURES") != nullptr;
        {   // (gi_dynamic_master_plan.md 7a) an instance is provisional until no capture of it is queued: all of them in (or dropped)
            XPROFILE_SCOPE("GI: provisional");
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
        static const std::string s_Ablate = [] { const char* p = xlionrender::Env("XLION_GI_ABLATE"); return std::string(p ? p : ""); }();
        static const double      s_FreezeS = [] { const char* p = xlionrender::Env("XLION_GI_FREEZE_S"); return p ? std::atof(p) : 15.0; }();
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
        { XPROFILE_SCOPE("GI: world update"); m_GIStats = W.Update(L, Eye, bGPU && Scene.m_GI.m_GPUBudgetMs > 0 ? m_pGITrace->BatchRays() : Scene.m_GI.m_RayBudget, std::uint32_t(m_GIRender)); }     // (no GPU budget: the scene's rays, a slice each)
        if( bGPU )
        {
            const auto TSubmit = std::chrono::steady_clock::now();
            // The relight's budget: the GI's slice less its steady cost (the update and Show, measured while no relight was owed: a few
            // frames old, smoothed) - a light change's catch-up never takes the GI past its slice (XLION_GI_RELIGHT_MS still overrides)
            m_GIQuietFrames = m_pGITrace->isRelighting() ? 0 : m_GIQuietFrames + 1;
            if( !m_bGISliceTSTried ) { m_bGISliceTSTried = true; m_bGISliceTS = m_Device.Create(m_GISliceTS, { .m_Count = 5 }) == nullptr; }
            if( m_bGISliceTS )
            {
                XPROFILE_SCOPE("GI: slice times");
                if( m_GISliceTS.getResults(m_GISliceNs) && m_GISliceNs.size() >= 5 && m_GISliceNs[1] >= m_GISliceNs[4] && m_GISliceNs[4] > m_GISliceNs[0] && m_GISliceNs[3] > m_GISliceNs[2] && m_GIQuietFrames > ring_frames_v + 1 )
                {
                    const double Show = double(m_GISliceNs[3] - m_GISliceNs[2]) * 1e-6;
                    const double Ms = double(m_GISliceNs[1] - m_GISliceNs[0]) * 1e-6 + Show, NoProbes = double(m_GISliceNs[4] - m_GISliceNs[0]) * 1e-6 + Show;
                    m_GISteadyMs = m_GISteadyMs < 0 ? Ms : 0.9 * m_GISteadyMs + 0.1 * Ms;
                    m_GISteadyNoProbesMs = m_GISteadyNoProbesMs < 0 ? NoProbes : 0.9 * m_GISteadyNoProbesMs + 0.1 * NoProbes;
                }
                // (a relight round skips the probe grid's update - the skinned meshes' GI waits a few frames: its share is the relight's)
                if( m_GISteadyMs >= 0 ) m_pGITrace->SetRelightBudgetMs(std::max(float(m_GISliceMs - (m_pGITrace->isRelighting() ? m_GISteadyNoProbesMs : m_GISteadyMs)), 0.1f));
                Window.WriteTimestamp(m_GISliceTS, 0);
            }
            // the relight's sun visibility: the GI's sun shadow map (renderShadows draws it later this frame: a sun that turned waits for it)
            {
                const xmath::fvec3 Z = Scene.m_Sun.m_Direction.NormalizeCopy();
                const bool bSame = m_GISunKey[0] == Z.m_X && m_GISunKey[1] == Z.m_Y && m_GISunKey[2] == Z.m_Z;
                m_pGITrace->SetSunMap(m_bGISunMapValid && bSame ? &m_GISunMap : nullptr, m_GISunW2C, m_GISunTexel, m_GISunDepthPerMeter, m_bGISunMapMade && !bSame && !m_bGISunRays);
            }
            if( auto* pT = GITimes(this, m_Device) ) Window.WriteTimestamp(pT->m_T, 0);
            { XPROFILE_SCOPE("GI: submit"); m_pGITrace->Submit(Window, W, L, GIFrame, Scene.m_GI.m_GPUBudgetMs, Eye); }
            if( auto* pT = GITimes(this, m_Device) ) Window.WriteTimestamp(pT->m_T, 4);
            // The link solve, on by default (Tomas 2026-10-24: it is what settles the light): the links built on the GPU once the rays settled (30
            // frames), then solved every frame and the rays stop (moving objects still get them). XLION_GI_LINKS=<rays>[,settle frames[,surfels a
            // frame]] changes it; XLION_GI_LINKS=0 turns it off
            static const std::array<int, 3> s_Links = [] { std::array<int, 3> A{ 64, 30, 2048 }; if (const char* p = xlionrender::Env("XLION_GI_LINKS")) (void)xlionrender::Scan(p, "%d,%d,%d", &A[0], &A[1], &A[2]); return A; }();
            static const bool s_bNoSurfels = xlionrender::Env("XLION_GI_NO_SURFELS") != nullptr;     // disables surfel GI; voxel GI unaffected
            if( s_Links[0] > 0 && !s_bNoSurfels ) { XPROFILE_SCOPE("GI: links"); m_pGITrace->PumpLinks(Window, W, L, GIFrame, s_Links[0], s_Links[1], s_Links[2]); }
            if( auto* pT = GITimes(this, m_Device) )
            {
                Window.WriteTimestamp(pT->m_T, 5);
                if( GIFrame % 600 == 0 ) { const auto C = W.FilterCounts(); std::printf("GIFILTER %zu surfels, %zu neighbor links (%.1f a surfel): a pass reads ~%.0f MB\n", C[0], C[1], double(C[1]) / double(std::max<std::size_t>(C[0], 1)), double(C[1]) * (36 + 12) / 1048576.0 + double(C[0]) * 72 / 1048576.0); std::fflush(stdout); }
            }
            // The probe grid (skinned meshes' GI): 32 x 16 x 32 probes a meter apart around the eye, 1024 updated a frame
            static const bool s_bNoProbes = [] { const char* p = xlionrender::Env("XLION_GI_NO_PROBES"); return p != nullptr && p[0] != '0'; }();     // (the editor's default 1 with the voxel GI; 0: the grid on - the skinned meshes' GI)
            if( m_bGISliceTS ) Window.WriteTimestamp(m_GISliceTS, 4);
            static int s_ProbeRuns = 0, s_ProbeSkips = 0;      // (XLION_GPU_TIMES, measurement: whether the grid updates - it waits while the GI relights)
            if( xlionrender::Env("XLION_GPU_TIMES") && GIFrame % 600 == 0 ) std::printf("GIPROBES %p the last 600 frames: updated %d, skipped while relighting %d (on %d, GI %d, probes setting %d, no-probes switch %d)\n", (const void*)this, s_ProbeRuns, s_ProbeSkips, int(Scene.m_GI.m_bProbes && !s_bNoProbes), int(Scene.m_GI.m_bEnable), int(Scene.m_GI.m_bProbes), int(s_bNoProbes)), std::fflush(stdout), s_ProbeRuns = s_ProbeSkips = 0;
            if( Scene.m_GI.m_bProbes && !s_bNoProbes && !s_bNoSurfels && m_pGITrace->isRelighting() ) ++s_ProbeSkips;
            if( Scene.m_GI.m_bProbes && !s_bNoProbes && (s_bNoSurfels || !m_pGITrace->isRelighting()) )
            {
                ++s_ProbeRuns;
                XPROFILE_SCOPE("GI: probes");
                // the grid's readers: the drawn meshes without surfels of their own (skinned, moving) - their probes every frame, the rest
                // 64 a frame (a full pass ~4 s)
                std::vector<gi::v3> Focus;
                for( const auto& D : Scene.m_Draws )
                {
                    if( D.m_Instance != 0 && hasGI(D.m_Instance) ) continue;
                    const auto* pM = findMesh(D.m_Mesh);
                    if( pM == nullptr ) continue;
                    const auto& B = pM->m_bCluster ? pM->m_Geom.m_BBox : pM->m_BBox;
                    xmath::fvec3 Lo{ 1e30f, 1e30f, 1e30f }, Hi{ -1e30f, -1e30f, -1e30f };
                    for( int c = 0; c < 8; ++c )
                    {
                        const xmath::fvec3 P = D.m_L2W * xmath::fvec3{ (c & 1) ? B.m_Max.m_X : B.m_Min.m_X, (c & 2) ? B.m_Max.m_Y : B.m_Min.m_Y, (c & 4) ? B.m_Max.m_Z : B.m_Min.m_Z };
                        Lo = xmath::fvec3::Min(Lo, P); Hi = xmath::fvec3::Max(Hi, P);
                    }
                    Focus.push_back(Lo); Focus.push_back(Hi);
                }
                static const std::array<float, 4> s_ProbeRead = [] { std::array<float, 4> A{ 0.0f, 64.0f, 1.0f, 0.0f }; if( const char* p = xlionrender::Env("XLION_GI_READ_PROBES") ) (void)xlionrender::Scan(p, "%f %f %f %f", &A[0], &A[1], &A[2], &A[3]); return A; }();
                (void)m_pGITrace->UpdateProbes(Window, Eye, s_ProbeRead[0] > 0.0f ? s_ProbeRead[0] : Scene.m_GI.m_ProbeSpacing, { 32, 16, 32 }, Focus, 192, int(s_ProbeRead[1]), GIFrame, int(s_ProbeRead[2]), s_ProbeRead[3] != 0.0f);
            }
            if( auto* pT = GITimes(this, m_Device) ) Window.WriteTimestamp(pT->m_T, 1);
            if( m_bGISliceTS ) Window.WriteTimestamp(m_GISliceTS, 1);
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
        XPROFILE_SCOPE("GI: places and rings");
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
                    static const std::string s_Px = [] { const char* p = xlionrender::Env("XLION_GI_ABLATE"); return std::string(p ? p : ""); }();
                    if( s_Px.find("pixel_vertex") != std::string::npos )    DrawGI.second &= ~0x100u;
                    if( s_Px.find("pixel_nomoments") != std::string::npos ) DrawGI.second |= 0x4000u;
                    if( s_Px.find("pixel_nolocal") != std::string::npos )   DrawGI.second |= 0x8000u;     // (mb_standard_pbr.frag: no SurfelGILocalMean)
                    if( s_Px.find("pixel_nocorners") != std::string::npos ) DrawGI.second |= 0x01000000u; // (the per-pixel read without its corner weights)
                    if( s_Px.find("pixel_lookuponly") != std::string::npos ) DrawGI.second |= 0x02000000u; // (the per-pixel read's cell lookup alone: then unsupported)
                    if( s_Px.find("pixel_nolookup") != std::string::npos )  DrawGI.second |= 0x04000000u; // (the per-pixel read unsupported at once)
                    if( s_Px.find("pixel_none") != std::string::npos )      DrawGI = { 0u, 0xFFFFFFFFu };
                    static const std::uint32_t s_Cap = [] { const char* p = xlionrender::Env("XLION_GI_PIXEL_CAP"); return p ? std::uint32_t(std::clamp(std::atoi(p), 0, 255)) : 0u; }();
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
        { XPROFILE_SCOPE("GI: ring"); if( bShowOnGPU ) RingWrite(nullptr);
        else (void)m_GILightRing.MemoryMapFrameRegion(std::size_t(Base), std::size_t(m_GIRegion), RingWrite); }
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
            if( m_bGISliceTS ) Window.WriteTimestamp(m_GISliceTS, 2);
            static const bool s_bNoSurfelsShow = xlionrender::Env("XLION_GI_NO_SURFELS") != nullptr;     // disables surfel GI; voxel GI unaffected
            XPROFILE_SCOPE("GI: show"); if( !s_bNoSurfelsShow ) (void)m_pGITrace->Show(Window, Places, m_GILightRing, m_GIMomentRing, Base, W.m_Setup.m_FilterPasses, W.m_Setup.m_Saturation, Scene.m_GI.m_bShowBlend);
            // Experiment X1 (xLION_Docs gi_next_level_analysis.md): XLION_GI_CLIP=<1 splat, 2 with a sight ray per cell>[,<level 0's cell>[,<surfels a frame>]]
            static const std::array<float, 3> s_Clip = [] { std::array<float, 3> A{ 0.0f, 0.5f, 4096.0f }; if( const char* p = xlionrender::Env("XLION_GI_CLIP") ) (void)xlionrender::Scan(p, "%f,%f,%f", &A[0], &A[1], &A[2]); return A; }();
            if( s_Clip[0] > 0.0f ) (void)m_pGITrace->ClipStep(Window, Eye, s_Clip[1], int(s_Clip[2]), s_Clip[0] > 1.5f);
            if( m_bGISliceTS ) Window.WriteTimestamp(m_GISliceTS, 3);
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
            static const bool s_bTimes = xlionrender::Env("XLION_GPU_TIMES") != nullptr;     // measurement: the GPU passes, every 120 frames
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
        beginCellFrame(Frame);
        if( m_VSMTableFrame == Frame ) return m_VSMTableBase;
        m_VSM.BuildTable(m_VSMTable);
        {   // the previous epoch's table right after it (the same size): the shading reads it where this epoch has no page yet; then the movers'
            std::vector<std::uint32_t> More;
            m_VSM.BuildPrevTable(More);
            m_VSMTable.insert(m_VSMTable.end(), More.begin(), More.end());
            m_VSM.BuildMoverTable(More);
            m_VSMTable.insert(m_VSMTable.end(), More.begin(), More.end());
            // the mover maps (XLION_SH_MOVER_MAPS): their count, then 8 floats each (xlionrender_shadow_vsm.glsl VSMMoverMaps)
            m_VSMTable.push_back(std::uint32_t(m_MoverMaps.size()));
            for( const auto& M : m_MoverMaps ) { const float* f = &M.m_X0; for( int k = 0; k < 8; ++k ) m_VSMTable.push_back(std::bit_cast<std::uint32_t>(f[k])); }
        }
        const int n = int(m_VSMTable.size());
        m_VSMTableBase = int(Frame % ring_frames_v) * m_CellStride + m_CellRingUsed;
        (void)m_CellRing.MemoryMapFrameRegion(m_VSMTableBase, n, [&](void* p) { std::memcpy(p, m_VSMTable.data(), std::size_t(n) * sizeof(std::uint32_t)); });   // (this frame's slot: no wait for the frame in flight)
        m_CellRingUsed += n;
        m_VSMTableFrame = Frame;
        return m_VSMTableBase;
    }

    // The local lights' pages: this frame's table in the cell ring, written once a frame (appended in the frame's region). Returns its first uint
    int instance::localTableBase( void ) noexcept
    {
        const std::uint64_t Frame = m_Device.getFrameIndex();
        beginCellFrame(Frame);
        if( m_LocalTableFrame == Frame ) return m_LocalTableBase;
        m_LocalPages.BuildTables(m_LocalTable);       // ([mover offset][static][mover]: xlionrender_local_lights.glsl LocalTexel)
        {   // the lamps' maps of the skinned movers (LocalLampMaps): a light slot's first map and count, then the maps (sorted by slot)
            std::stable_sort(m_LampMaps.begin(), m_LampMaps.end(), [](const lamp_map& a, const lamp_map& b) { return a.m_Slot < b.m_Slot; });
            const std::size_t Hd = m_LocalTable.size();
            m_LocalTable.push_back(std::uint32_t(max_local_lights_v));
            m_LocalTable.resize(Hd + 1 + 2 * max_local_lights_v, 0u);
            for( std::size_t k = 0; k < m_LampMaps.size(); ++k )
            {
                const auto s = m_LampMaps[k].m_Slot;
                if( s >= std::uint32_t(max_local_lights_v) ) continue;
                if( m_LocalTable[Hd + 2 + 2 * s] == 0 ) m_LocalTable[Hd + 1 + 2 * s] = std::uint32_t(k);
                ++m_LocalTable[Hd + 2 + 2 * s];
            }
            for( const auto& M : m_LampMaps ) { const float* f = M.m_Axis; for( int k = 0; k < 15; ++k ) m_LocalTable.push_back(std::bit_cast<std::uint32_t>(f[k])); m_LocalTable.push_back(M.m_Slot); }
        }
        const int n = std::max(int(m_LocalTable.size()), 1);
        m_LocalTable.resize(std::size_t(n), 0u);
        m_LocalTableBase = int(Frame % ring_frames_v) * m_CellStride + m_CellRingUsed;
        (void)m_CellRing.MemoryMapFrameRegion(m_LocalTableBase, n, [&](void* p) { std::memcpy(p, m_LocalTable.data(), std::size_t(n) * sizeof(std::uint32_t)); });
        m_CellRingUsed += n;
        m_LocalTableFrame = Frame;
        return m_LocalTableBase;
    }

    // The local lights' pages: the request pass over the main view's depth (shaders/xlionrender_local_request_comp.glsl), after the scene
    // pass, for the shadowed lights with a light slot. Its bits and counters go to this frame's ring slot, read when the slot comes round
    void instance::localRequestPass( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye ) noexcept
    {
        const int Slot = int(m_Device.getFrameIndex() % ring_frames_v), nBits = localBitUints(), nReq = nBits + 32;
        details::local_request_frame F{};
        F.m_C2W = W2C.Inverse();
        F.m_Eye = xmath::fvec4(Eye, 1);
        int n = 0;
        for( int i = 0; i < int(Scene.m_LocalLights.size()) && n < max_local_lights_v; ++i )
        {
            const int LS = m_LocalPages.SlotOfLight(i);
            if( LS < 0 ) continue;
            const auto& L = Scene.m_LocalLights[std::size_t(i)];
            const bool  bSpot = L.m_Type == local_light::type::SPOT;
            F.m_Lights[n].m_PosRadius = xmath::fvec4(L.m_Position, L.m_Radius);
            F.m_Lights[n].m_DirTan    = bSpot ? xmath::fvec4(L.m_Direction.NormalizeCopy(), shadows::SpotTanHalf(L)) : xmath::fvec4(0, 0, 0, 0);
            F.m_Lights[n].m_Info      = { float(LS * shadows::local_pages::faces_per_light_v * shadows::local_pages::face_entries_v), shadows::LocalNear(L.m_Radius), m_LocalPages.MovingBias(LS)
                                        , L.m_SourceRadius > 0.0f ? 2.0f : 0.0f };     // w: the levels coarser its contact-hardening reads (kLocalSoftLevels)
            ++n;
        }
        for( int ls = 0; ls < max_local_lights_v; ++ls ) m_LocalReqGen[Slot][std::size_t(ls)] = m_LocalPages.Generation(ls);
        if( n == 0 ) return;
        F.m_P0 = { float(n), m_LocalPages.ResolutionBias(), float(localTableBase()), 0 };
        (void)m_LocalFrames.MemoryMapFrameRegion(Slot, 1, [&](void* p) { std::memcpy(p, &F, sizeof(F)); });
        const xgpu::buffer::copy Z{ 0, 0, nReq };
        Window.CopyBuffer(m_VSMZeros, m_LocalReq, std::span{ &Z, 1 });
        const std::uint32_t Push[8] = { std::uint32_t(Slot), 0u, std::uint32_t(View.m_Width), std::uint32_t(View.m_Height), std::uint32_t(nBits), 0u, 0u, 0u };
        auto Buffers  = std::array<xgpu::buffer*, 3>{ &m_LocalFrames, &m_LocalReq, &m_CellRing };
        auto Textures = std::array<const xgpu::texture*, 1>{ &View.m_Depth };
        Window.Dispatch(m_LocalRequest, Buffers, Textures, Push, sizeof(Push), (View.m_Width + 7) / 8, (View.m_Height + 7) / 8);
        const xgpu::buffer::copy C{ 0, Slot * nReq, nReq };
        Window.CopyBuffer(m_LocalReq, m_LocalReqHost, std::span{ &C, 1 });
        m_LocalReqValid[Slot] = true;
    }

    // The sun's virtual pages: the request pass over the main view's depth (shaders/xlionrender_vsm_request_comp.glsl), after the scene
    // pass. Its bits and counters go to this frame's ring slot, read when the slot comes round (renderShadows), with these windows
    void instance::vsmRequestPass( xgpu::window& Window, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye ) noexcept
    {
        const int Slot = int(m_Device.getFrameIndex() % ring_frames_v), nBits = vsmBitUints(), nReq = vsmReqUints();
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
        // (each pixel's own level, and the contact-hardening's coarser levels shared by each 2 x 2 quad's lanes)
        const std::uint32_t Push[8] = { std::uint32_t(Slot), 0u, std::uint32_t(vsmTableBase()), std::uint32_t(View.m_Width), std::uint32_t(View.m_Height), std::uint32_t(nBits), std::uint32_t(m_VSMSoftLevels), std::bit_cast<std::uint32_t>(m_VSMLevelBand) };     // (m_Pad1: the levels' blend)
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
    // A frame's first write into the light and cell rings: their use starts again; the cell ring first grows to the largest need seen
    // (a frame whose cells did not fit rendered EXACT and recorded it: m_CellNeed) - xGPU keeps the old buffer while frames use it
    void instance::beginCellFrame( std::uint64_t Frame ) noexcept
    {
        if( Frame == m_LightFrame ) return;
        m_LightFrame = Frame; m_LightRingUsed = 0; m_CellRingUsed = 0;
        m_bLightCellsTimed = false;
        if( std::vector<std::uint64_t> Ns; m_LightCellsTimes.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull && Ns[1] >= Ns[0] )
            m_LightBuildGpuMs = double(Ns[1] - Ns[0]) * 1e-6;
        if( m_CellNeed <= m_CellStride ) return;
        const int Stride = ((m_CellNeed + m_CellNeed / 4) + 65535) / 65536 * 65536;
        m_Device.Destroy(std::move(m_CellRing));
        if( m_Device.Create(m_CellRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = ring_frames_v * Stride, .m_bWriteOnly = true }) == nullptr )
            m_CellStride = Stride;
        else (void)m_Device.Create(m_CellRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = ring_frames_v * m_CellStride, .m_bWriteOnly = true });
        m_CellNeed = 0;
    }

    bool instance::readLightCells( std::vector<std::uint32_t>& Out ) noexcept
    {
        Out.resize(std::size_t(m_LightCellUints));
        if( m_LightCellUints == 0 ) return false;
        return m_CellRing.MemoryMap(m_LightCellBase, m_LightCellUints, [&](void* p) { std::memcpy(Out.data(), p, Out.size() * sizeof(std::uint32_t)); }) == nullptr;
    }

    // The GI's emitters as lights (m_bEmitterLights; lighting_gi_interface_notice.md "emissive surfaces as lights"): each emitting surfel a
    // one-sided disc of its area - the GI's own form factor, A cos / (d^2 + A) (shaders/xlionrender_local_lights.glsl LocalEmitterAtt) -, its
    // radiance the GI's (its emissive scale in), no shadow pages; the strongest m_EmitterLightsMax by power, the rest left out. Made again
    // only when the view's generation changes (an emitter added, moved, its emission or the scale changed)
    void instance::refreshEmitterLights( bool bOn ) noexcept
    {
        const auto* pW = m_pGIWorld.get();
        if( !bOn || pW == nullptr ) { m_EmitterLights.clear(); m_EmitterLightsGen = 0; m_nEmittersLeftOut = m_nEmittersMerged = 0; return; }
        const std::uint64_t Gen = pW->EmittersGen();
        if( Gen == m_EmitterLightsGen && m_EmitterLightsMaxUsed == m_EmitterLightsMax ) return;     // (the same emitters, the same limit)
        m_EmitterLightsGen = Gen; m_EmitterLightsMaxUsed = m_EmitterLightsMax;
        m_EmitterLights.clear();
        const auto Lum = []( const xmath::fvec3& c ) { return 0.2126f * c.m_X + 0.7152f * c.m_Y + 0.0722f * c.m_Z; };
        pW->ForEachEmitter([&]( const auto& E )
        {
            const float L = Lum(E.m_Radiance);
            if( L <= 0 || E.m_Area <= 0 ) return;
            const float R = std::clamp(std::sqrt(E.m_Area * L / m_EmitterCutoff), std::sqrt(E.m_Area), 50.0f);     // (A / (d^2 + A) x L = cutoff)
            m_EmitterLights.push_back({ E.m_Pos, E.m_Normal, E.m_Radiance, E.m_Area, R });
        });
        // Over m_EmitterLightsMax: the strongest kept, each of the others folded into the nearest kept one facing its way - its flux
        // (radiance x area) added to that one's radiance: the light kept, moved to its neighbour. None facing its way: left out (counted) -
        // folded into one facing elsewhere it would light what it never reached (a panel's back into its front: +17% under it)
        // ponytail: O(folded x kept) on a generation change - a grid of the kept ones when a level's emitters show it
        const std::size_t Max = std::size_t(std::max(m_EmitterLightsMax, 1));
        m_nEmittersLeftOut = 0;
        m_nEmittersMerged  = int(m_EmitterLights.size() > Max ? m_EmitterLights.size() - Max : 0);
        if( m_EmitterLights.size() > Max )
        {
            std::nth_element(m_EmitterLights.begin(), m_EmitterLights.begin() + std::ptrdiff_t(Max), m_EmitterLights.end()
                            , [&]( const emitter_light& a, const emitter_light& b ) { return Lum(a.m_Radiance) * a.m_Area > Lum(b.m_Radiance) * b.m_Area; });
            // (each group a larger disc: its area the sum, at the centre of its flux, the flux kept - radiance = flux / area)
            std::vector<xmath::fvec3> Flux(Max, xmath::fvec3{ 0, 0, 0 }), At(Max, xmath::fvec3{ 0, 0, 0 });
            std::vector<float>        Area(Max, 0.0f), W(Max, 0.0f);
            for( std::size_t d = Max; d < m_EmitterLights.size(); ++d )
            {
                const auto& E = m_EmitterLights[d];
                std::size_t Best = Max;
                float       BestD = std::numeric_limits<float>::max();
                for( std::size_t k = 0; k < Max; ++k )
                {
                    const auto&  K  = m_EmitterLights[k];
                    const xmath::fvec3 D = K.m_Pos - E.m_Pos;
                    const float  d2 = D.Dot(D);
                    if( d2 < BestD && K.m_Normal.Dot(E.m_Normal) > 0.5f ) { BestD = d2; Best = k; }
                }
                if( Best == Max ) { ++m_nEmittersLeftOut; --m_nEmittersMerged; continue; }
                const float w = Lum(E.m_Radiance) * E.m_Area;
                Flux[Best] += E.m_Radiance * E.m_Area; Area[Best] += E.m_Area; At[Best] += E.m_Pos * w; W[Best] += w;
            }
            m_EmitterLights.resize(Max);
            for( std::size_t k = 0; k < Max; ++k )
            {
                auto& K = m_EmitterLights[k];
                if( Area[k] <= 0 ) continue;
                const float w = Lum(K.m_Radiance) * K.m_Area;
                K.m_Pos      = (K.m_Pos * w + At[k]) / std::max(w + W[k], 1e-12f);
                K.m_Radiance = (K.m_Radiance * K.m_Area + Flux[k]) / (K.m_Area + Area[k]);
                K.m_Area    += Area[k];
                K.m_Radius   = std::clamp(std::sqrt(K.m_Area * Lum(K.m_Radiance) / m_EmitterCutoff), std::sqrt(K.m_Area), 50.0f);     // (its reach grows with it)
            }
        }
    }

    void instance::writeLocalLights( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye, details::ubo_lighting& Light ) noexcept
    {
        const std::uint64_t Frame = m_Device.getFrameIndex();
        beginCellFrame(Frame);
        refreshEmitterLights(m_bEmitterLights || Scene.m_GI.m_bEmitterLights);

        // the scene's lights, then the emitters' (after them: no shadow pages - the local pages index the scene's)
        const int nScene  = std::min({ int(Scene.m_LocalLights.size()), max_lights_v, light_region_v - m_LightRingUsed });
        const int nEm     = std::max(0, std::min({ int(m_EmitterLights.size()), max_lights_v - nScene, light_region_v - m_LightRingUsed - nScene }));
        const int nLights = nScene + nEm;
        const int LightBase = int(Frame % ring_frames_v) * light_region_v + m_LightRingUsed;
        if( nLights > 0 )
            (void)m_LightRing.MemoryMap(LightBase, nLights, [&](void* pMemory)
            {
                auto* pOut = static_cast<details::local_light_data*>(pMemory);
                for( int k = 0; k < nEm; ++k )      // (an emitter: DirCosOuter w -3, its normal; Shadow w its area, no pages)
                {
                    const auto& E = m_EmitterLights[std::size_t(k)];
                    auto& Out = pOut[nScene + k];
                    Out.m_PosRadius   = xmath::fvec4(E.m_Pos - Eye, E.m_Radius);
                    Out.m_Color       = xmath::fvec4(E.m_Radiance, 1.0f);
                    Out.m_DirCosOuter = xmath::fvec4(E.m_Normal, -3.0f);
                    Out.m_Shadow      = { 0.0f, 0.0f, 0.0f, E.m_Area };
                }
                for( int i = 0; i < nScene; ++i )
                {
                    const auto& LL    = Scene.m_LocalLights[i];
                    const bool  bSpot = LL.m_Type == local_light::type::SPOT;
                    const int   LSlot = m_LocalPages.SlotOfLight(i);
                    auto&       Out   = pOut[i];
                    Out.m_PosRadius   = xmath::fvec4(LL.m_Position - Eye, LL.m_Radius);
                    Out.m_Color       = xmath::fvec4(LL.m_Color * LL.m_Intensity, std::cos(LL.m_InnerAngle));
                    Out.m_DirCosOuter = bSpot ? xmath::fvec4(LL.m_Direction.NormalizeCopy(), std::cos(LL.m_OuterAngle)) : xmath::fvec4(0, 0, 0, -2);
                    Out.m_Shadow      = { float(std::max(LSlot, 0) * shadows::local_pages::faces_per_light_v * shadows::local_pages::face_entries_v), std::max(LL.m_SourceRadius, 0.0f), LSlot >= 0 ? 1.0f : 0.0f, bSpot ? shadows::SpotTanHalf(LL) : 1.0f };     // (y: its source radius - contact-hardening)
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
        static const int s_ForcedMode = []                  // measurement only (XLION_LIGHT_MODE): the editor's mode forced
        {
            const char* p = xlionrender::Env("XLION_LIGHT_MODE");
            if( p == nullptr ) return -1;
            const std::string_view M(p);
            return M == "exact" ? int(light_mode::EXACT) : M == "cells" ? int(light_mode::CELLS) : M == "sampled" ? int(light_mode::SAMPLED) : M == "screen" ? int(light_mode::SCREEN) : M == "auto" ? int(light_mode::AUTO) : -1;
        }();
        if( s_ForcedMode >= 0 ) Mode = light_mode(s_ForcedMode);
        if( Mode == light_mode::AUTO ) Mode = nLights >= m_LightScreenThreshold ? light_mode::SCREEN : light_mode::CELLS;
        int  CellBase = 0;
        m_LightBuildMs = 0;
        int  ProbeCells = -1;
        if( Mode != light_mode::EXACT && (nLights > 0 || nProbes > 0) )
        {
            const auto T0 = std::chrono::steady_clock::now();
            m_CellLights.resize(std::max(nLights, 0));
            for( int k = 0; k < nEm; ++k )
            {
                const auto& E = m_EmitterLights[std::size_t(k)];
                m_CellLights[std::size_t(nScene + k)] = { E.m_Pos, E.m_Radius, std::max((0.2126f * E.m_Radiance.m_X + 0.7152f * E.m_Radiance.m_Y + 0.0722f * E.m_Radiance.m_Z) * E.m_Area, 1e-6f), E.m_Normal, -1.0f };
            }
            for( int i = 0; i < nScene; ++i )
            {
                const auto& LL = Scene.m_LocalLights[i];
                const float Lum = 0.2126f * LL.m_Color.m_X + 0.7152f * LL.m_Color.m_Y + 0.0722f * LL.m_Color.m_Z;
                const bool  bSpot = LL.m_Type == local_light::type::SPOT;
                m_CellLights[i] = { LL.m_Position, LL.m_Radius, std::max(Lum * LL.m_Intensity, 1e-6f), LL.m_Direction.NormalizeCopy(), bSpot ? LL.m_OuterAngle : -1.0f };
            }
            // the GPU build: the windows and a bound of the entries here, the lists by shaders/xlionrender_light_cells_comp.glsl (below)
            // (it runs in a device-local scratch - the cell ring is host memory, slow for the build's reads - copied into the ring after)
            // (the same lights and windows as the scratch's last build - a still camera over static lights -: its lists again, no build)
            bool                bGPU  = m_bLightCellsGPU && nLights > 0;
            const bool          bHead = Mode == light_mode::SAMPLED || Mode == light_mode::SCREEN;      // (the head/tail split)
            const std::uint64_t Key   = bGPU ? light_cells::Key(m_CellLights, Eye, m_LightCellSize, bHead ? m_LightHead : 0) : 0u;
            const bool          bSame = bGPU && Key == m_CellScratchKey && m_CellScratchUints > 0;
            if( bSame ) m_LightCells.Reuse(Eye, m_LightCellSize, m_CellScratchBound);
            const std::uint32_t Bound = bSame ? m_CellScratchBound : bGPU ? m_LightCells.Reserve(m_CellLights, Eye, m_LightCellSize) : 0u;
            if( bGPU && !bSame ) m_LightCells.GPUInput(m_CellLights, m_CellInput);
            if( bSame ) m_CellInput.clear();
            if( bGPU && int(m_LightCells.m_Data.size() + m_CellInput.size()) > m_CellScratchUints )
            {
                const int Need = int(m_LightCells.m_Data.size() + m_CellInput.size()), Size = ((Need + Need / 4) + 65535) / 65536 * 65536;
                const std::vector<std::uint32_t> Zeros(std::size_t(Size), 0u);
                m_Device.Destroy(std::move(m_CellScratch));
                m_CellScratchUints = m_Device.Create(m_CellScratch, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = Size, .m_pData = const_cast<std::uint32_t*>(Zeros.data()) }) == nullptr ? Size : 0;
                bGPU = m_CellScratchUints > 0;
                m_CellScratchKey = 0;
            }
            if( !bGPU ) m_LightCells.Build(m_CellLights, Eye, m_LightCellSize, bHead ? m_LightHead : 0);
            const int nLists = int(m_LightCells.m_Data.size());
            if( nProbes > 0 )
            {
                m_ProbeBoxes.resize(nProbes);
                for( int k = 0; k < nProbes; ++k ) m_ProbeBoxes[k] = { Scene.m_Probes[m_ProbeOrder[k]].m_BoxMin, Scene.m_Probes[m_ProbeOrder[k]].m_BoxMax };
                ProbeCells = int(m_LightCells.AddBoxes(m_ProbeBoxes));
            }
            m_LightBuildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();

            const int nUints = int(m_LightCells.m_Data.size()) + (bGPU ? int(m_CellInput.size()) : 0);
            if( nUints > m_CellStride - m_CellRingUsed )        // (the ring grows for the next frame: this one, EXACT)
            {
                m_CellNeed = std::max(m_CellNeed, m_CellRingUsed + nUints);
                Mode = light_mode::EXACT; ProbeCells = -1;
            }
            else
            {
                CellBase = int(Frame % ring_frames_v) * m_CellStride + m_CellRingUsed;
                const int From = bGPU ? nLists : 0, nCPU = int(m_LightCells.m_Data.size()) - From;     // (the GPU writes the lists: only the probes' part)
                if( nCPU > 0 ) (void)m_CellRing.MemoryMapFrameRegion(CellBase + From, nCPU, [&](void* pMemory) { std::memcpy(pMemory, m_LightCells.m_Data.data() + From, std::size_t(nCPU) * sizeof(std::uint32_t)); });
                if( bGPU )
                {
                    // the input into the ring, then the scratch (the lists at 0, the input after them); count, scan, fill (+ the head/tail
                    // split) - a workgroup a 4^3 block of a level's window -, the lists back into the ring
                    const int InputBase = CellBase + int(m_LightCells.m_Data.size()), nInput = int(m_CellInput.size());
                    if( !m_bLightCellsTimed ) Window.WriteTimestamp(m_LightCellsTimes, 0);
                    const xgpu::buffer::copy In{ InputBase, nLists, nInput }, Out{ 0, CellBase, nLists };
                    if( !bSame )
                    {
                        (void)m_CellRing.MemoryMapFrameRegion(InputBase, nInput, [&](void* pMemory) { std::memcpy(pMemory, m_CellInput.data(), std::size_t(nInput) * sizeof(std::uint32_t)); });
                        Window.CopyBuffer(m_CellRing, m_CellScratch, std::span{ &In, 1 });
                        const std::uint32_t Head = bHead ? std::uint32_t(std::max(m_LightHead, 0)) : 0u;
                        auto Buffers = std::array<xgpu::buffer*, 1>{ &m_CellScratch };
                        constexpr int Blocks = (light_cells::cells_v / 4) * (light_cells::cells_v / 4) * (light_cells::cells_v / 4);
                        for( std::uint32_t Pass = 0; Pass < 3; ++Pass )
                        {
                            const std::uint32_t Push[8] = { Pass, std::uint32_t(nLists), std::uint32_t(nLights), 0u, Head, Bound, m_LightCells.m_bSkipHidden ? 1u : 0u, 0u };
                            Window.Dispatch(m_LightCellsCompute, Buffers, Push, sizeof(Push), Pass == 1 ? 1 : Blocks, Pass == 1 ? 1 : light_cells::levels_v);
                        }
                        m_CellScratchKey = Key; m_CellScratchBound = Bound; ++m_nCellBuilds;
                    }
                    Window.CopyBuffer(m_CellScratch, m_CellRing, std::span{ &Out, 1 });
                    if( !m_bLightCellsTimed ) { Window.WriteTimestamp(m_LightCellsTimes, 1); m_bLightCellsTimed = true; }      // (the frame's first build: the view's)
                }
                m_CellRingUsed += nUints;
            }
        }

        if( nLights <= 0 && nProbes <= 0 ) Mode = light_mode::EXACT;    // (no cells written: the shader must not read headers at CellBase)
        m_LightModeUsed  = Mode;
        m_LightCellBase  = CellBase;
        m_LightCellUints = Mode == light_mode::EXACT ? 0 : int(m_LightCells.m_Data.size());
        m_LightStats.m_nLights        = nLights;
        m_LightStats.m_nDropped       = int(Scene.m_LocalLights.size()) - nScene;
        m_LightStats.m_nEmitters      = nEm;
        m_LightStats.m_nEmittersLeftOut = m_nEmittersLeftOut + int(m_EmitterLights.size()) - nEm;
        m_LightStats.m_nEmittersMerged  = m_nEmittersMerged;
        m_LightStats.m_nOverflowCells = 0;           // (a cell's list never overflows: 2-uint headers)
        m_LightStats.m_CellRingMB     = double(ring_frames_v) * double(m_CellStride) * 4.0 / (1024.0 * 1024.0);
        Light.m_LocalCount = { float(nLights), float(int(Mode)), float(m_LightSeed++ & 0xffffff), float(m_bLightTailVisibility ? -m_LightTailCandidates : m_LightTailCandidates) };
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
    // The reflections' screen trace (scene::m_bReflectionScreen; XLION_REFL_SCREEN=1 turns it on where the scene has no setting for it - a Level, 2026-11-07)
    static bool ReflectionScreen( const scene& Scene ) noexcept { static const bool s_bOn = xlionrender::Env("XLION_REFL_SCREEN") != nullptr; return Scene.m_bReflectionScreen || s_bOn; }

    void instance::updateProbes( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye ) noexcept
    {
        if( !m_bProbeAtlasReady ) { auto Cmd = Window.StartRenderPass(m_ProbeAtlasClearPass); m_bProbeAtlasReady = true; }   // bound by every lit pipeline
        const int nProbes = int(Scene.m_Probes.size());
        m_ProbeStates.resize(nProbes);
        if( m_ProbeCapturing >= nProbes ) m_ProbeCapturing = -1;
        if( static const bool s_bTrace = xlionrender::Env("XLION_PROBE_TRACE") != nullptr; s_bTrace )     // measurement: the probes' state, every 120 frames
        {
            static int s_n = 0;
            if( ++s_n % 120 == 0 )
            {
                int nValid = 0, nDirty = 0, nSlot = 0;
                for( const auto& St : m_ProbeStates ) { nValid += St.m_bValid; nDirty += St.m_bDirty; nSlot += St.m_Slot >= 0; }
                std::printf("PROBES %d in the scene, %d with a slot, %d valid, %d dirty; capturing %d (face %d)\n", nProbes, nSlot, nValid, nDirty, m_ProbeCapturing, m_ProbeNextFace); std::fflush(stdout);
            }
        }
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
        if( nProbes > m_ProbeSlots )
        {
            std::vector<std::pair<float, int>> Rank(nProbes);
            for( int i = 0; i < nProbes; ++i ) Rank[i] = { BoxDistance(Scene.m_Probes[i]), i };
            std::nth_element(Rank.begin(), Rank.begin() + m_ProbeSlots, Rank.end());
            for( int k = m_ProbeSlots; k < nProbes; ++k )
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
            while( Free < m_ProbeSlots && Used[Free] ) ++Free;
            if( Free == m_ProbeSlots ) break;
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
            if( Key != St.m_Key ) { St.m_Key = Key; St.m_bValid = false; St.m_bDirty = true; if( m_ProbeCapturing == i ) m_ProbeNextFace = 0; if( xlionrender::Env("XLION_PROBE_TRACE") ) std::printf("PROBEMOVE %d at frame %llu\n", i, (unsigned long long)m_Device.getFrameIndex()); }

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
            // measurement (XLION_PROBE_TRACE): a capture frame's GPU cost - its faces, its prefilter - printed past 10 ms
            static const bool s_bCost = xlionrender::Env("XLION_PROBE_TRACE") != nullptr;
            static xgpu::timestamps s_TS; static bool s_bTS = false; static std::vector<std::uint64_t> s_Ns;
            if( s_bCost && !s_bTS ) s_bTS = m_Device.Create(s_TS, { .m_Count = 3 }) == nullptr;
            if( s_bCost && s_bTS )
            {
                if( s_TS.getResults(s_Ns) && s_Ns.size() >= 3 && s_Ns[1] >= s_Ns[0] )
                {
                    const double Faces = (s_Ns[1] - s_Ns[0]) * 1e-6, Pre = s_Ns[2] >= s_Ns[1] ? (s_Ns[2] - s_Ns[1]) * 1e-6 : 0.0;
                    if( Faces + Pre > 10 ) { std::printf("PROBECOST faces %.2f, prefilter %.2f ms (frame %llu)\n", Faces, Pre, (unsigned long long)m_Device.getFrameIndex()); std::fflush(stdout); }
                }
                Window.WriteTimestamp(s_TS, 0);
            }
            for( ; Budget > 0 && m_ProbeNextFace < 6; --Budget, ++m_ProbeNextFace )
                renderScenePass(Window, Scene, m_CaptureViews[m_ProbeNextFace]
                               , shadows::FaceW2C(xmath::fvec3{ 0, 0, 0 }, shadows::CubeFaceAxis(m_ProbeNextFace), 1.0f, 0.05f, 2000.0f), Pr.m_Position, false);
            if( s_bCost && s_bTS ) Window.WriteTimestamp(s_TS, 1);
            if( m_ProbeNextFace < 6 ) { if( s_bCost && s_bTS ) Window.WriteTimestamp(s_TS, 2); return; }

            {
                auto Cmd = Window.StartRenderPass(m_EnvSourcePass[0]);
                Cmd.setPipelineInstance(m_CubeToOctInstance);
                Cmd.setPushConstants(details::push_env_cube_to_oct{ { float(env_source_size_v), 0.05f, 2000.0f, 0 } });     // (the faces' near, far: FaceW2C's above)
                Cmd.setBuffer(m_FullscreenVB);
                Cmd.setBuffer(m_FullscreenIB);
                Cmd.Draw(3);
            }
            auto& St = m_ProbeStates[m_ProbeCapturing];
            prefilterInto(Window, m_ProbeAtlasPass, float(St.m_Slot % probe_slots_x_v * 384), float(St.m_Slot / probe_slots_x_v * 256));
            if( s_bCost && s_bTS ) Window.WriteTimestamp(s_TS, 2);
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
        m_MoverMaps.clear(); m_LampMaps.clear();      // (this frame's, if any: none read last frame's)
        // Measurement only (XLION_GPU_TIMES): its CPU time by part - the pages' choice, the casters (light bounds, pairs), the recording -
        // SHADOWCPU, the largest of 120 frames (the sun turning added ~3.8 ms of GPU idle before the scene pass, 2026-10-30)
        static const bool s_bShCpu = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
        static double s_ShMax[5] = {}; static int s_ShFrames = 0, s_ShBusy = 0;
        struct sh_cpu
        {
            using clk = std::chrono::steady_clock;
            bool m_bOn; clk::time_point m_T[6]; int m_n = 0;
            void Lap( void ) noexcept { if( m_bOn && m_n < 5 ) m_T[++m_n] = clk::now(); }
            ~sh_cpu()
            {
                if( !m_bOn ) return;
                while( m_n < 5 ) m_T[++m_n] = clk::now();
                double d[5], t = 0; for( int k = 0; k < 5; ++k ) t += d[k] = std::chrono::duration<double, std::milli>(m_T[k + 1] - m_T[k]).count();
                if( t > 0.5 ) { for( int k = 0; k < 5; ++k ) s_ShMax[k] += d[k]; ++s_ShBusy; }       // (the frames with page work: their mean)
                if( ++s_ShFrames == 120 )
                {
                    if( s_ShBusy ) std::printf("SHADOWCPU mean of %d busy frames: changes + decode %.3f, vsm::Update %.3f, after it (local pages, table) %.3f, casters %.3f, recording %.3f ms\n", s_ShBusy, s_ShMax[0] / s_ShBusy, s_ShMax[1] / s_ShBusy, s_ShMax[2] / s_ShBusy, s_ShMax[3] / s_ShBusy, s_ShMax[4] / s_ShBusy), std::fflush(stdout);
                    s_ShFrames = s_ShBusy = 0; for( auto& v : s_ShMax ) v = 0;
                }
            }
        } ShCpu{ s_bShCpu, { std::chrono::steady_clock::now() } };
        if( !m_bShadowAtlasReady )
        {
            auto Cmd = Window.StartRenderPass(m_ShadowClearPass);      // the atlas's first use: cleared to far, so it can be loaded and sampled
            m_bShadowAtlasReady = true;
        }

        m_nShadowTilesRendered = 0;
        m_ShadowDrawStats      = {};

        // The casters and what changed since the last render: new and gone casters, moved or re-materialed ones, and skinned
        // ones (old and new bounds). Draws with an instance id compare against their slot's last state (no hashing, bounds kept
        // while unchanged); draws without one go through a hash of their state.
        struct draw_caster { const draw* m_pDraw; const mesh_slot* m_pMesh; xmath::fbbox m_World; std::uint32_t m_Slot; std::uint32_t m_BoxedBase; bool m_bMover; bool m_bGPU; bool m_bCut = false; };     // m_BoxedBase: in m_ClusterSkip (~0: none); m_bGPU: the pairs can draw it (rigid); m_bCut: it has masked submeshes, the CPU path draws those (found when a page renders)
        std::vector<draw_caster>                         Draws;
        std::vector<xmath::fbbox>                        Changed;
        std::vector<xmath::fbbox>                        SunStatic, SunMovers;  // the sun's pages: changes of the static casters (a static caster changed or
                                                                                 // went, a mover settled), the movers' bounds this frame
        std::unordered_map<std::uint64_t, prev_caster>   Now;
        Draws.reserve(Scene.m_Draws.size());
        m_DrawWorld.assign(Scene.m_Draws.size(), xmath::fbbox{});
        if( m_SlotCasters.empty() ) m_SlotCasters.resize(max_instances_v);
        const std::uint64_t Pass = ++m_ShadowPassCount;
        m_CasterSlots.clear();
        const auto MaterialsHash = [](const draw& D) { return Hash(1469598103934665603ull, D.m_Materials.data(), D.m_Materials.size() * sizeof(material_id)); };     // alpha test, displacement
        const auto WorldOf       = [](const mesh_slot& M, const draw& D) { return D.m_Bounds.m_Min.m_X <= D.m_Bounds.m_Max.m_X ? D.m_Bounds : (M.m_bCluster ? M.m_Geom.m_BBox : M.m_BBox).TransformCopy(D.m_L2W); };
        for( std::size_t iDraw = 0; iDraw < Scene.m_Draws.size(); ++iDraw )
        {
            const auto& D     = Scene.m_Draws[iDraw];
            const auto* pMesh = findMesh(D.m_Mesh);
            if( pMesh == nullptr ) continue;
            const std::uint32_t Slot = m_DrawSlot[iDraw];
            if( pMesh->m_bCluster && Slot == instance_slots::none_v ) continue;

            // The editor's view of huge meshes. Boxes only: a cluster mesh casts no shadow (gone from the casters, its pages re-render, and again when it is back).
            // A frame budget's boxes are the main view's only: every cluster casts. (They were part of the caster's state: the boxed set changes from frame to
            // frame, so every page under the caster rendered again - and the caster became a mover -, and a slower frame boxed more: in the Debug editor on
            // Bistro, 64 pages a frame that never settled and 85 ms of CPU a frame. The pages are cached: the budget's cost is the main view's.)
            if( pMesh->m_bCluster && m_bBoxesOnly ) { m_DrawWorld[iDraw] = WorldOf(*pMesh, D); continue; }
            constexpr std::uint32_t BoxedBase = ~0u;
            // (its shadow-relevant state besides the transform and materials: the geometry's resident finest LOD - cluster_geom::Refresh switches it when
            // LOD 0 arrives or goes. Without it, pages drawn while only the coarse LODs were loaded kept them: holes in a building's shadow, sunlit
            // rectangles in the street, Bistro Game 2026-11-07, Tomas)
            const std::uint64_t BoxedHash = pMesh->m_bCluster ? std::uint64_t(pMesh->m_Geom.m_FinestLOD) + 1 : 0;

            const bool   bSkin = pMesh->m_bCluster && pMesh->m_Geom.m_Kind == geom_kind::SKIN;
            xmath::fbbox World;
            bool         bMover = false;
            if( Slot < std::uint32_t(max_instances_v) )
            {
                auto&               C     = m_SlotCasters[Slot];
                const std::uint64_t MHash = MaterialsHash(D);
                const bool bChanged = m_DrawChanged[iDraw] || !C.m_bValid || bSkin || C.m_Mesh.m_Index != D.m_Mesh.m_Index
                                   || C.m_Mesh.m_Generation != D.m_Mesh.m_Generation || C.m_MaterialsHash != MHash || C.m_BoxedHash != BoxedHash;
                if( bChanged )
                {
                    m_ChangeWhy[0] += m_DrawChanged[iDraw] != 0; m_ChangeWhy[1] += !C.m_bValid; m_ChangeWhy[2] += C.m_Mesh.m_Index != D.m_Mesh.m_Index || C.m_Mesh.m_Generation != D.m_Mesh.m_Generation; m_ChangeWhy[3] += C.m_MaterialsHash != MHash;
                    if( C.m_bValid ) Changed.push_back(C.m_World);
                    if( C.m_bValid && Pass - C.m_ChangedPass >= static_frames_v ) SunStatic.push_back(C.m_World);    // (a static caster starts moving)
                    C = { WorldOf(*pMesh, D), D.m_Mesh, MHash, BoxedHash, Pass, true, Pass };
                    Changed.push_back(C.m_World);
                }
                C.m_Seen = Pass;
                World    = C.m_World;
                bMover   = bSkin || Pass - C.m_ChangedPass < static_frames_v;
                if( !bSkin && Pass - C.m_ChangedPass == static_frames_v ) SunStatic.push_back(World), ++m_ChangeWhy[5];             // (a mover settles)
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
                const auto It = m_PrevCasters.find(Key);
                const std::uint64_t First = It == m_PrevCasters.end() ? Pass : It->second.m_First;
                Now.emplace(Key, prev_caster{ World, First });
                if( It == m_PrevCasters.end() || bSkin ) Changed.push_back(World), ++m_ChangeWhy[7];
                if( It != m_PrevCasters.end() && bSkin ) Changed.push_back(It->second.m_World);
                bMover = bSkin || Pass - First < static_frames_v;
                if( !bSkin && Pass - First == static_frames_v ) SunStatic.push_back(World), ++m_ChangeWhy[6];                       // (a mover settles)
            }
            m_DrawWorld[iDraw] = World;
            if( !m_bShadowSplitMovers ) bMover = false;
            if( bMover ) SunMovers.push_back(World);
            Draws.push_back({ &D, pMesh, World, Slot, BoxedBase, bMover, pMesh->m_bCluster && !bSkin });
        }
        for( const auto& [Key, P] : m_PrevCasters ) if( !Now.contains(Key) )
        {
            Changed.push_back(P.m_World);
            if( Pass - 1 - P.m_First >= static_frames_v ) SunStatic.push_back(P.m_World);                         // (a static caster went)
        }
        m_PrevCasters = std::move(Now);
        for( auto Slot : m_CasterSlotsPrev )                                // persistent casters no longer drawn
        {
            auto& C = m_SlotCasters[Slot];
            if( C.m_bValid && C.m_Seen != Pass )
            {
                Changed.push_back(C.m_World); ++m_ChangeWhy[4];
                if( Pass - C.m_ChangedPass >= static_frames_v ) SunStatic.push_back(C.m_World);                   // (a static caster went)
                C.m_bValid = false;
            }
        }
        std::swap(m_CasterSlots, m_CasterSlotsPrev);
        // The probes capture again where the static casters changed (a mover once it settles, a static one moved, new or gone), not each frame a mover moves -
        // as Unreal's captures: Bistro Game's walking characters dirtied a probe every frame, a cube face of the whole level recorded each frame, 26 ms of the
        // main thread's 31 (2026-11-07). After the gone casters above, which it missed
        m_FrameChanged = m_bShadowSplitMovers ? SunStatic : Changed;

        const auto& Sun = Scene.m_Sun;
        // The sun: its virtual pages (a sun with a disc: the 3 coarser levels' pages asked for too - the contact-hardening's search and its
        // wide penumbrae read them)
        m_VSMSoftLevels = Sun.m_SourceAngle > 0 ? 3 : 0;
        m_VSM.m_ToRender.clear();
        static const bool s_bNoPairs = xlionrender::Env("XLION_SH_NO_PAIRS") != nullptr;      // (measurement: every page's casters by the CPU path)
        static const bool s_bNoShadowCache = xlionrender::Env("XLION_SH_NO_CACHE") != nullptr;      // (measurement: every shadow page rendered again each frame)
        if( s_bNoShadowCache ) m_bShadowCache = false;
        bool bPairs = m_pShadowPairs && m_bShadowGPUPairs && !s_bNoPairs;
        if( m_pShadowPairs )
        {
            m_pShadowPairs->Begin(m_Device.getFrameIndex());
            m_PairsReport.m_nPairs = m_pShadowPairs->m_Pairs; m_PairsReport.m_nDropped = m_pShadowPairs->m_Dropped; m_PairsReport.m_Triangles = m_pShadowPairs->m_Triangles;
            if( m_pShadowPairs->m_bDropped ) { m_VSM.Invalidate(); bPairs = false; }      // (its pages missed pairs: all again, a draw a cluster and page)
        }
        m_PairsReport.m_nCandidates = m_PairsReport.m_nRuns = 0;
        m_PairsReport.m_bGPU        = false;
        static const bool s_bNoSunShadows = xlionrender::Env("XLION_NO_SUN_SHADOWS") != nullptr;     // (measurement: the sun's pages' cost - an ablation)
        if( Sun.m_bCastShadows && !s_bNoSunShadows )
        {
            if( std::vector<std::uint64_t> Ns; m_VSMTimes.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull && Ns[1] >= Ns[0] )
                m_VSMReport.m_RequestGpuMs = double(Ns[1] - Ns[0]) * 1e-6;
            if( const int Slot = int(m_Device.getFrameIndex() % ring_frames_v); !m_VSMRenderTimed[Slot] ) m_VSMReport.m_RenderGpuMs = 0;
            else if( std::vector<std::uint64_t> Ns; m_VSMRenderTimes.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull && Ns[1] >= Ns[0] )
            {
                m_VSMReport.m_RenderGpuMs = double(Ns[1] - Ns[0]) * 1e-6;
                // a page's GPU time, smoothed (the frames that rendered a few: their fixed part would dominate)
                if( const int n = m_VSMRenderPages[std::size_t(m_Device.getFrameIndex() % ring_frames_v)]; n >= 8 ) m_VSMPageMs += (m_VSMReport.m_RenderGpuMs / n - m_VSMPageMs) * 0.1;
            }
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
            // This frame's epoch first, then the requests of this ring slot's last request pass (a ring ago: done) and its counters, decoded
            // with the windows that made them - moved into this epoch when they are of an older one (their depth ranges)
            m_VSM.OpenEpoch(Sun.m_Direction, Level0Texel, m_Device.getFrameIndex());
            {
                const int Slot = int(m_Device.getFrameIndex() % ring_frames_v), nBits = vsmBitUints(), nReq = vsmReqUints();
                m_VSMRequests.clear();
                if( m_VSMSnap[Slot].m_bValid )
                {
                    // (a ring ago: done - no wait for the frame in flight, which cost 2.3 ms of CPU; one copy out, then decoded: the mapped
                    // memory can be the GPU's, where every read crosses the bus - a word at a time it cost 1 ms). The bits and counters, then the
                    // depth ranges of the bit words with a request only (all of them, 100 KB, cost 0.12 ms a frame)
                    auto& Got = m_VSMGot;
                    Got.resize(static_cast<std::size_t>(nReq));
                    (void)m_VSMReqHost.MemoryMapFrameRegion(Slot * nReq, nReq, [&](void* p)
                    {
                        const auto* pSrc = static_cast<const std::uint32_t*>(p);
                        std::memcpy(Got.data(), pSrc, std::size_t(nBits + 32) * sizeof(std::uint32_t));
                        for( int w = 0; w < nBits; ++w ) if( Got[w] )
                        {
                            const std::size_t o = std::size_t(nBits + 32 + 64 * w);
                            std::memcpy(Got.data() + o, pSrc + o, std::min<std::size_t>(64, Got.size() - o) * sizeof(std::uint32_t));
                        }
                    });
                    const auto* pDepth = reinterpret_cast<const shadows::vsm::request_depth*>(Got.data() + nBits + 32);
                    m_VSM.DecodeRequests({ Got.data(), std::size_t(nBits) }, m_VSMSnap[Slot], m_VSMRequests, { pDepth, std::size_t(m_VSM.getSetup().TableEntries()) });
                    std::memcpy(m_VSMReport.m_Counters.data(), Got.data() + nBits, m_VSMReport.m_Counters.size() * sizeof(std::uint32_t));
                    m_VSMReport.m_bCounters = true;
                    m_VSMSnap[Slot].m_bValid = false;
                }
            }
            // (the static pages see the static casters' changes only - every change when the movers are not split; the movers have pages of their own)
            m_VSM.SetDetailDistance(Sun.m_ShadowDetailDistance);
            {   // (measurement: a frame whose pixels found no page at any level - lit where they should be shadowed: VSMNONE)
                static const bool s_bN = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                const auto& C = m_VSMReport.m_Counters;
                if( s_bN && C[0] && double(C[3]) > 0.005 * double(C[0]) )
                    std::printf("VSMNONE frame %llu: %.1f%% of the pixels without a page (coarser %.1f%%); pages requested %d rendered %d net %d rejected %d deferred %d resident %d, epoch %d, dynamic %.2f\n", (unsigned long long)m_Device.getFrameIndex(), 100.0 * C[3] / C[0], 100.0 * C[2] / C[0]
                               , m_VSM.m_Stats.m_Requested, m_VSM.m_Stats.m_Rendered, m_VSM.m_Stats.m_RenderedPinned, m_VSM.m_Stats.m_Rejected, m_VSM.m_Stats.m_Deferred, m_VSM.m_Stats.m_ResidentFine, int(m_VSM.HasPrevious()), m_VSM.m_Stats.m_DynamicBias), std::fflush(stdout);
            }
            { static const bool s_bT = xlionrender::Env("XLION_GPU_TIMES") != nullptr;      // (measurement: the sun's shadow detail reaching the renderer)
              if( s_bT && m_Device.getFrameIndex() % 240 == 0 ) std::printf("VSMDETAIL sun dir %.3f %.3f %.3f, distance %.2f blend %.2f: bias %.2f (dynamic %.2f); pages requested %d rendered %d rejected %d deferred %d resident %d\n", Sun.m_Direction.m_X, Sun.m_Direction.m_Y, Sun.m_Direction.m_Z, Sun.m_ShadowDetailDistance, Sun.m_ShadowLevelBlend, m_VSM.ResolutionBias(), m_VSM.m_Stats.m_DynamicBias, m_VSM.m_Stats.m_Requested, m_VSM.m_Stats.m_Rendered, m_VSM.m_Stats.m_Rejected, m_VSM.m_Stats.m_Deferred, m_VSM.m_Stats.m_ResidentFine), std::fflush(stdout); }
            m_VSMLevelBand = std::clamp(Sun.m_ShadowLevelBlend, 0.01f, 0.95f);
            ShCpu.Lap();
            // The frame's pages by their measured cost: what fits the pages' share of the shadows' slice (XLION_SH_PAGES_MS overrides; 0 off)
            static const float s_PagesMs = [] { const char* p = xlionrender::Env("XLION_SH_PAGES_MS"); return p ? float(std::atof(p)) : -1.0f; }();
            const float PagesMs = s_PagesMs >= 0 ? s_PagesMs : m_ShadowPagesMs;
            // a small scene needs no budget: every page within the frame costs less than the budget's own bookkeeping would allow for
            static const int s_InstantEnv = [] { const char* p = xlionrender::Env("XLION_SH_INSTANT_TRIS"); return p ? std::atoi(p) : -1; }();
            std::int64_t nCasterTris = 0;
            for( const auto& Dr : Draws ) nCasterTris += Dr.m_pMesh->m_bCluster ? std::int64_t(Dr.m_pMesh->m_Geom.m_Clusters.size()) * 256 : Dr.m_pMesh->m_nIndices / 3;      // (ponytail: a cluster taken as half its 512 triangle cap, near enough for this choice)
            const int  InstantTris = s_InstantEnv >= 0 ? s_InstantEnv : m_ShadowInstantTris;
            const bool bInstant = InstantTris > 0 && nCasterTris <= std::int64_t(InstantTris);
            m_VSM.SetInstant(bInstant);
            m_VSM.SetFrameBudget(!bInstant && PagesMs > 0 ? std::clamp(int(PagesMs / std::max(m_VSMPageMs, 1e-3)), 8, 1024) : 0);
            static const bool s_bMoverSameLevel = xlionrender::Env("XLION_VSM_MOVER_SAME_LEVEL") != nullptr;     // (measurement: the movers at the pages' own level)
            m_VSM.m_bMoversCoarse = m_bShadowMoversCoarse && !s_bMoverSameLevel;
            std::vector<xmath::fbbox> VSMMovers;      // (the mover maps: the skinned movers have no sun mover pages)
            if( m_bMoverMaps ) { for( const auto& D : Draws ) if( D.m_bMover && !(D.m_pMesh->m_bCluster && D.m_pMesh->m_Geom.m_Kind == geom_kind::SKIN) ) VSMMovers.push_back(D.m_World); }
            m_VSM.Update(Sun.m_Direction, Level0Texel, Eye, m_bShadowSplitMovers ? SunStatic : Changed, m_bShadowCache, m_VSMRequests, m_Device.getFrameIndex(), m_bMoverMaps ? VSMMovers : SunMovers);
            m_VSMReport.m_Pages = m_VSM.m_Stats;
            ShCpu.Lap();
            if( s_bShCpu )      // (measurement: vsm::Update by part, the largest of 120 frames - SHADOWCPU's line)
            {
                static float s_SecMax[7] = {}; static int s_SecN = 0;
                for( int k = 0; k < 7; ++k ) s_SecMax[k] = std::max(s_SecMax[k], m_VSM.m_Stats.m_SecMs[k]);
                if( ++s_SecN == 120 ) { std::printf("VSMUPDATE max of 120 frames: windows %.3f, out %.3f, dirty %.3f, net %.3f, requests %.3f, movers %.3f, rest %.3f ms\n", s_SecMax[0], s_SecMax[1], s_SecMax[2], s_SecMax[3], s_SecMax[4], s_SecMax[5], s_SecMax[6]); std::fflush(stdout); s_SecN = 0; for( auto& v : s_SecMax ) v = 0; }
            }
            {   static const bool s_bE = xlionrender::Env("XLION_VSM_EVENTS") != nullptr;     // (its own switch: a print and flush most frames while the sun turns, ms of them - not with the GPU times)      // (measurement: a frame that dropped many pages - the dropout hunt)
                const auto& St = m_VSM.m_Stats;
                if( s_bE && (St.m_Dirtied > 32 || St.m_Recentred || St.m_Epochs || St.m_Evicted > 32 || St.m_Deferred > 64) )
                    std::printf("VSMEVENT frame %llu: dirtied %d (changed boxes %zu), recentred %d, epochs %d, evicted %d, rendered %d, deferred %d, rejected %d, requested %d, resident %d, dynamic bias %.2f; why: slot moved %d new %d mesh %d materials %d gone %d settled %d hashed settled %d hashed new %d\n", (unsigned long long)m_Device.getFrameIndex(), St.m_Dirtied, (m_bShadowSplitMovers ? SunStatic : Changed).size(), St.m_Recentred, St.m_Epochs, St.m_Evicted, St.m_Rendered, St.m_Deferred, St.m_Rejected, St.m_Requested, St.m_ResidentFine, St.m_DynamicBias, m_ChangeWhy[0], m_ChangeWhy[1], m_ChangeWhy[2], m_ChangeWhy[3], m_ChangeWhy[4], m_ChangeWhy[5], m_ChangeWhy[6], m_ChangeWhy[7]), std::fflush(stdout);
                m_ChangeWhy = {};
            }
        }
        else m_VSM.Invalidate();            // the casters changing meanwhile reach no page: all of them when it casts again
        // The local lights' pages: this ring slot's last request pass (a ring ago: done; its light slots' generations), then the frame's pages
        {
            const int Slot = int(m_Device.getFrameIndex() % ring_frames_v), nBits = localBitUints();
            m_LocalRequests.clear();
            if( m_LocalReqValid[Slot] )
            {
                m_LocalGot.resize(std::size_t(nBits + 32));
                (void)m_LocalReqHost.MemoryMapFrameRegion(Slot * (nBits + 32), nBits + 32, [&](void* p) { std::memcpy(m_LocalGot.data(), p, m_LocalGot.size() * sizeof(std::uint32_t)); });
                shadows::local_pages::DecodeRequests({ m_LocalGot.data(), std::size_t(nBits) }, m_LocalReqGen[Slot], m_LocalPages, m_LocalRequests);
                std::memcpy(m_LocalReport.m_Counters.data(), m_LocalGot.data() + nBits, m_LocalReport.m_Counters.size() * sizeof(std::uint32_t));
                m_LocalReport.m_bCounters = true;
                m_LocalReqValid[Slot] = false;
            }
            m_LocalPages.setBudget(m_ShadowUpdatesPerFrame);
            // (movers apart, as the sun's: a page keeps the static casters, dirtied by their changes only; the movers draw into mover pages each frame)
            std::vector<xmath::fbbox> LampMovers;      // (the mover maps: the skinned movers have no lamp mover pages either)
            if( m_bMoverMaps && Scene.m_Sun.m_bCastShadows ) { for( const auto& D : Draws ) if( D.m_bMover && !(D.m_pMesh->m_bCluster && D.m_pMesh->m_Geom.m_Kind == geom_kind::SKIN) ) LampMovers.push_back(D.m_World); }
            if( m_bShadowSplitMovers ) m_LocalPages.Update(Scene.m_LocalLights, SunStatic, m_bShadowCache, m_LocalRequests, m_Device.getFrameIndex(), m_bMoverMaps && Scene.m_Sun.m_bCastShadows ? LampMovers : SunMovers);
            else                       m_LocalPages.Update(Scene.m_LocalLights, Changed, m_bShadowCache, m_LocalRequests, m_Device.getFrameIndex());
            m_LocalReport.m_Pages = m_LocalPages.m_Stats;
            static const bool s_bLocalLog = xlionrender::Env("XLION_LOCAL_LOG") != nullptr; static int s_LocalN = 0;      // (measurement: the local pages a frame, every 120 frames)
            if( s_bLocalLog && ++s_LocalN % 120 == 0 ) { const auto& St = m_LocalPages.m_Stats; std::printf("LOCALPAGES rendered %d pinned %d mover %d (rejected %d) dirtied %d requested %d resident %d stale %d movers %zu static changes %zu\n", St.m_Rendered, St.m_RenderedPinned, St.m_RenderedMover, St.m_RejectedMover, St.m_Dirtied, St.m_Requested, St.m_Resident, St.m_Stale, SunMovers.size(), SunStatic.size()); std::fflush(stdout); }
        }
        m_LightStats.m_nShadowed = m_LightStats.m_nShadowFallback = 0;
        for( std::size_t i = 0; i < Scene.m_LocalLights.size(); ++i ) if( Scene.m_LocalLights[i].m_bCastShadows )
        {
            ++m_LightStats.m_nShadowed;
            m_LightStats.m_nShadowFallback += m_LocalPages.SlotOfLight(int(i)) < 0;
        }
        m_LightStats.m_nShadowStale = m_LocalPages.m_Stats.m_Stale;

        m_nShadowTilesRendered = static_cast<int>(m_VSM.m_ToRender.size() + m_LocalPages.m_ToRender.size());
        m_VSMRenderTimed[m_Device.getFrameIndex() % ring_frames_v] = !m_VSM.m_ToRender.empty();
        m_VSMRenderPages[m_Device.getFrameIndex() % ring_frames_v] = int(m_VSM.m_ToRender.size());
        const bool bMoverMaps = m_bMoverMaps && Scene.m_Sun.m_bCastShadows && std::ranges::any_of(Draws, [](const draw_caster& D) { return D.m_pMesh->m_bCluster && D.m_pMesh->m_Geom.m_Kind == geom_kind::SKIN; });     // (the skinned movers' sun maps: drawn every frame)
        if( m_nShadowTilesRendered == 0 && !bMoverMaps ) { m_MoverMaps.clear(); return; }

        ShCpu.Lap();
        // The casters culled in light space (only when a page renders): their light-space boxes, kept from frame to frame - a draw's again when
        // its world box changes, all when the light's basis does (an epoch). The CPU path's casters (all of them without the pairs)
        auto&                       LightBounds = m_CasterLight;
        std::vector<std::uint32_t>  AllCasters;
        if( !m_VSM.m_ToRender.empty() )
        {
            const std::array<xmath::fvec3, 3> Basis = { m_VSM.m_X, m_VSM.m_Y, m_VSM.m_Z };
            if( std::memcmp(&Basis, &m_CasterLightBasis, sizeof(Basis)) != 0 ) { m_CasterLightWorld.clear(); m_CasterLightBasis = Basis; }
            m_CasterLight.resize(Draws.size());
            m_CasterLightWorld.resize(Draws.size(), xmath::fbbox{ xmath::fvec3(1, 1, 1), xmath::fvec3(-1, -1, -1) });
            AllCasters.reserve(Draws.size());
            for( std::uint32_t i = 0; i < Draws.size(); ++i )
            {
                if( std::memcmp(&m_CasterLightWorld[i], &Draws[i].m_World, sizeof(xmath::fbbox)) != 0 )
                {
                    m_CasterLight[i]      = m_VSM.Bounds(Draws[i].m_World);
                    m_CasterLightWorld[i] = Draws[i].m_World;
                }
            }
        }

        // The GPU page/caster pairs (P1c, features/shadows/xlionrender_shadow_pairs.h): the sun's pages and the local lights' pages, the
        // rigid opaque clusters of the casters that overlap a page (the sun's: of their kind, light-space rects; the local lights': their
        // light's sphere), a candidate each, a run a geometry and kind (the sun's and the local lights' draw in viewports of their own), paired
        // on the GPU - outside the passes
        struct pairs_run { const cluster_geom* m_pGeom; std::uint32_t m_First, m_Max; bool m_bLocal; material_slot* m_pCut = nullptr; };     // m_pCut: a masked material's run (its cutout through the records)
        std::vector<pairs_run> PairRuns;
        // The casters' LOD a sun page draws (cluster_geom::LODSubmeshes): texels up to XLION_SH_LOD_TEXEL (2 cm) the finest, each doubling a
        // LOD coarser - a coarse page drew every full-detail triangle (the shadow pass's cost was its triangles). XLION_SH_NO_LOD: LOD 0
        static const float s_LODTexel = [] { const char* p = xlionrender::Env("XLION_SH_LOD_TEXEL"); return p ? float(std::atof(p)) : 0.02f; }();
        static const bool  s_bNoLOD   = xlionrender::Env("XLION_SH_NO_LOD") != nullptr;
        const auto PageLOD = [&]( int Level ) -> int
        {
            if( s_bNoLOD || s_LODTexel <= 0.0f ) return 0;
            const float Texel = m_VSM.Level0Texel() * float(1 << Level);
            return std::clamp(int(std::ceil(std::log2(std::max(Texel / s_LODTexel, 1.0f)) - 1e-4f)), 0, 7);
        };
        // A page of LOD k draws submesh s (a LOD group) at the coarsest LOD up to k whose error stays within the page's texel (its finest at
        // LOD k: half the texel that asks for k) - a coarse LOD's shape change (a vault collapsed flat) never moves a shadow by more than a texel
        const auto PageSub = [&]( const cluster_geom& G, int k, std::size_t s ) -> const cluster_geom::submesh&
        {
            const float Texel = s_LODTexel * float(1 << std::max(k - 1, 0));
            int j = k;
            while( j > 0 && std::size_t(j) < G.m_LODErrors.size() && s < G.m_LODErrors[std::size_t(j)].size() && G.m_LODErrors[std::size_t(j)][s] > Texel ) --j;
            return G.LODSubmeshes(j)[s];
        };
        const int LocalW = m_LocalPages.getSetup().m_PoolTexels, LocalH = m_LocalPages.getSetup().PoolHeight();
        if( bPairs && (!m_VSM.m_ToRender.empty() || !m_LocalPages.m_ToRender.empty()) )
        {
            const float AW = float(sunAtlasWidth()), AH = float(sunAtlasHeight()), T = float(shadows::vsm::page_texels_v);
            std::vector<shadow_pairs::page> Pages;
            Pages.reserve(m_VSM.m_ToRender.size() + m_LocalPages.m_ToRender.size());
            bool bAny[2] = {};
            std::uint32_t LODsUsed = 0;         // (the LODs this frame's sun pages draw: a candidate list each)
            for( const auto& P : m_VSM.m_ToRender )
            {
                const int k = PageLOD(P.m_Key.m_Level);
                Pages.push_back({ P.m_W2C, { T / AW, T / AH, (T + 2.0f * float(P.m_AtlasX)) / AW - 1.0f, (T + 2.0f * float(P.m_AtlasY)) / AH - 1.0f }, { P.m_MinX, P.m_MinY, P.m_Size, (P.m_bMover ? 1.0f : 0.0f) + 4.0f * float(k) } });     // (w: its kind + 4 x its LOD)
                bAny[P.m_bMover] = true;
                LODsUsed |= 1u << k;
            }
            // (the local pages: in their part's viewport - LocalW x LocalH texels)
            std::vector<int> Lit;
            for( const auto& P : m_LocalPages.m_ToRender )
            {
                const float LW = float(LocalW), LH = float(LocalH), TL = float(shadows::local_pages::page_texels_v);
                Pages.push_back({ P.m_W2C, { TL / LW, TL / LH, (TL + 2.0f * float(P.m_AtlasX)) / LW - 1.0f, (TL + 2.0f * float(P.m_AtlasY)) / LH - 1.0f }, { 0, 0, 0, P.m_bMover ? 3.0f : 2.0f } });     // (2: a local page, 3: its mover page)
                if( std::find(Lit.begin(), Lit.end(), P.m_Light) == Lit.end() ) Lit.push_back(P.m_Light);
            }
            // ponytail: a caster against every page of its kind until one overlaps (the epoch's hundreds of static pages: a grid if it shows)
            const auto Touches = [&]( std::uint32_t i )
            {
                if( !bAny[Draws[i].m_bMover] ) return false;
                for( const auto& P : m_VSM.m_ToRender ) if( P.m_bMover == Draws[i].m_bMover && shadows::RectOverlaps(LightBounds[i], P.m_MinX, P.m_MinY, P.m_Size) ) return true;
                return false;
            };
            const auto TouchesLocal = [&]( std::uint32_t i )
            {
                for( int l : Lit ) if( shadows::Overlaps(Draws[i].m_World, Scene.m_LocalLights[std::size_t(l)].m_Position, Scene.m_LocalLights[std::size_t(l)].m_Radius) ) return true;
                return false;
            };
            std::vector<shadow_pairs::candidate>& Cands = m_PairCandidates;
            std::vector<xmath::fmat4>             L2L;
            std::map<std::tuple<const cluster_geom*, bool, const material_slot*>, std::uint32_t> RunOf;     // (a masked material's submeshes: a run of their own)
            Cands.clear();
            xmath::fmat4 Rot = xmath::fmat4::fromZero();
            Rot(0, 0) = m_VSM.m_X.m_X; Rot(0, 1) = m_VSM.m_X.m_Y; Rot(0, 2) = m_VSM.m_X.m_Z;
            Rot(1, 0) = m_VSM.m_Y.m_X; Rot(1, 1) = m_VSM.m_Y.m_Y; Rot(1, 2) = m_VSM.m_Y.m_Z;
            Rot(2, 0) = m_VSM.m_Z.m_X; Rot(2, 1) = m_VSM.m_Z.m_Y; Rot(2, 2) = m_VSM.m_Z.m_Z;
            Rot(3, 3) = 1.0f;
            const auto AddCandidates = [&]( std::uint32_t i, bool bLocal )
            {
                const auto& D    = *Draws[i].m_pDraw;
                const auto& G    = Draws[i].m_pMesh->m_Geom;
                const auto  Draw = std::uint32_t(L2L.size()) | (bLocal ? shadow_pairs::local_v : 0u) | (Draws[i].m_bMover ? shadow_pairs::mover_v : 0u);
                const auto RunFor = [&](material_slot* pCut)
                {
                    const auto [It, bNew] = RunOf.try_emplace({ &G, bLocal, pCut }, std::uint32_t(PairRuns.size()));
                    if( bNew ) PairRuns.push_back({ &G, 0, 0, bLocal, pCut });
                    return It->second;
                };
                L2L.push_back(bLocal ? D.m_L2W : Rot * D.m_L2W);
                // a candidate list per LOD the pages use (Center.w: its LOD - it pairs with the pages of that LOD only), the geometry's coarsest
                // once for that LOD and every coarser one (w -1 - k: a single-LOD geometry's candidates pair with every page, as before -
                // a list per LOD outgrew the candidates' room); a local light's: LOD 0, any page
                const int nGeomLODs = std::max(1, int(G.m_LODSubmeshes.size()));
                for( int k = 0; k < nGeomLODs; ++k )
                {
                    const bool     bLast = k == nGeomLODs - 1;
                    const uint32_t Want  = bLast ? ~((1u << k) - 1u) : (1u << k);       // (the pages' LODs this list serves)
                    if( bLocal ? k != 0 : (LODsUsed & Want) == 0 ) continue;
                    const float    Tag   = bLocal ? -1.0f : bLast ? -1.0f - float(k) : float(k);
                    for( std::size_t s = 0; s < G.LODSubmeshes(k).size(); ++s )
                    {
                        const auto& Sub  = PageSub(G, k, s);
                        auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                        if( pMat && !MaterialVisibilityOf(pMat->m_State).m_bCastsShadow ) continue;
                        // a masked one: its cutout's run (the material's fragment through the records); no pipeline: the CPU path draws it
                        material_slot* pCut = nullptr;
                        if( pMat && MaterialVisibilityOf(pMat->m_State).m_Opacity == material_visibility::opacity::MASKED )
                        {
                            if( getPairsCutInstance(*pMat) == nullptr ) { Draws[i].m_bCut = true; continue; }
                            pCut = pMat;
                        }
                        const std::uint32_t Run = RunFor(pCut);
                        for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                        {
                            if( Draws[i].m_BoxedBase != ~0u && m_ClusterSkip[Draws[i].m_BoxedBase + c] == cluster_boxed_v ) continue;     // drawn as its box: no shadow
                            const auto& C    = G.m_Clusters[c];
                            const auto  Ctr  = C.m_BBox.getCenter();
                            const auto  Half = (C.m_BBox.m_Max - C.m_BBox.m_Min) * 0.5f;
                            Cands.push_back({ Run, 0, c, Draws[i].m_Slot, C.m_nIndices, C.m_iIndex, std::int32_t(C.m_iVertex), Draw
                                            , { Ctr.m_X, Ctr.m_Y, Ctr.m_Z, Tag }, { Half.m_X, Half.m_Y, Half.m_Z, 0 } });
                            ++PairRuns[Run].m_Max;
                        }
                    }
                }
            };
            for( std::uint32_t i = 0; i < Draws.size(); ++i )
            {
                if( !Draws[i].m_bGPU ) continue;
                if( Touches(i) )                AddCandidates(i, false);
                if( !Lit.empty() && TouchesLocal(i) ) AddCandidates(i, true);
            }
            if( shadow_pairs::Fits(Cands.size(), L2L.size(), Pages.size(), PairRuns.size()) )
            {
                std::uint32_t First = 0;        // a run's commands: its candidates' room
                for( auto& Run : PairRuns ) { Run.m_First = First; First += Run.m_Max; }
                for( auto& K : Cands ) K.m_RunFirst = PairRuns[K.m_Run].m_First;
                m_pShadowPairs->Pair(Window, Cands, L2L, Pages, PairRuns.size());
                m_PairsReport.m_nCandidates = int(Cands.size()); m_PairsReport.m_nRuns = int(PairRuns.size());
                m_PairsReport.m_bGPU        = true;
            }
            else { bPairs = false; PairRuns.clear(); }
        }
        else bPairs = false;
        if( !m_VSM.m_ToRender.empty() )
            for( std::uint32_t i = 0; i < Draws.size(); ++i ) if( !(bPairs && Draws[i].m_bGPU) || Draws[i].m_bCut ) AllCasters.push_back(i);

        // A run of the pairs: its geometry, a draw for every page its candidates pair with (the viewport set)
        const auto DrawPairs = [&]( xgpu::cmd_buffer& Cmd, bool bLocal )
        {
            auto& ViewBlock = m_ShadowUBO.allocEntry<details::ubo_geom_view>();
            ViewBlock.m_w2C         = xmath::fmat4::fromIdentity();
            ViewBlock.m_WorldOffset = xmath::fvec4(m_Origin, 0);
            bool bCutLight = false;
            for( std::uint32_t r = 0; r < std::uint32_t(PairRuns.size()); ++r )
            {
                const auto& Run = PairRuns[r];
                if( Run.m_bLocal != bLocal ) continue;
                if( Run.m_pCut )
                {
                    auto* pCut = getPairsCutInstance(*Run.m_pCut);
                    if( pCut == nullptr ) continue;
                    if( !bCutLight ) { m_LightUBO.allocEntry<details::ubo_lighting>().m_LightParams = xmath::fvec4(0, 0, 1, 1); bCutLight = true; }     // (the fragment's "shadow pass" flag, once)
                    std::array<xgpu::buffer*, 12> Buffers{ Run.m_pGeom->m_pClusterData, &m_InstanceBuffer, &m_pShadowPairs->m_Records, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing
                                                         , &m_MaterialParams, &m_EntityParams, &m_pShadowPairs->m_Pages, TailBuffer() };
                    Cmd.setPipelineInstance(*pCut, std::span{ Buffers });
                    if( Run.m_pCut->m_bTable ) Cmd.setTextureTable(Run.m_pCut->m_Table);
                    Cmd.setDynamicUBO(m_LightUBO, 1);
                    Cmd.setPushConstants(details::push_static_geom{ 0u, 0u, 0u, 0xFFFFFFFFu, Run.m_pCut->m_ParamBase });     // (the fragment's MaterialBase at 16)
                }
                else
                {
                    std::array<xgpu::buffer*, 4> Buffers{ Run.m_pGeom->m_pClusterData, &m_InstanceBuffer, &m_pShadowPairs->m_Records, &m_pShadowPairs->m_Pages };
                    Cmd.setPipelineInstance(m_ShadowPairsInstance, std::span{ Buffers });
                }
                Cmd.setDynamicUBO(m_ShadowUBO, 0);
                BindClusterStreams(Cmd, *Run.m_pGeom);
                Cmd.DrawIndexedIndirectCount(m_pShadowPairs->m_Commands, int(Run.m_First), m_pShadowPairs->m_Counts, int(r), int(Run.m_Max));
                ++m_ShadowDrawStats.m_IndirectDraws; m_ShadowDrawStats.m_IndirectMax += Run.m_Max;
            }
        };

        // Draws the casters that pass Keep with W2C into the current viewport
        // Draws the casters of Indices that pass Keep with W2C into the current viewport; a sun page (pSunPage): each cluster against its
        // light-space rect too (a caster's box - a Bistro section's - passed every page: the cutout clusters were 52M triangles a frame)
        const shadows::vsm::page_to_render* pSunPage = nullptr;
        bool bOpaqueCut = false;        // (the GI's sun map: masked submeshes drawn whole - the GI's rays block on them alike; their cutout's fragment cost 49 ms a map in Bistro)
        std::vector<std::vector<decltype(m_VSM.Bounds(xmath::fbbox{}))>> ClusterLight(Draws.size());     // (DrawCasters: a draw's clusters in light space, this frame)
        const auto DrawCasters = [&]( xgpu::cmd_buffer& Cmd, const xmath::fmat4& W2C, std::span<const std::uint32_t> Indices, const auto& Keep, int Lod = 0 )
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
                for( std::size_t s = 0; s < G.LODSubmeshes(Lod).size(); ++s )          // (the page's LOD: PageLOD; each group's within its error: PageSub)
                {
                    const auto& Sub = PageSub(G, Lod, s);
                    // The material's settings: no shadow, or a masked one's cutout (its own caster: the material's fragment, which stops after the cut)
                    auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                    const auto Vis = pMat ? MaterialVisibilityOf(pMat->m_State) : material_visibility{};
                    if( !Vis.m_bCastsShadow ) continue;
                    if( bPairs && Draws[i].m_bGPU && (Vis.m_Opacity != material_visibility::opacity::MASKED || !Draws[i].m_bCut) ) continue;     // (the pairs drew it: a masked one too, unless its cutout had no pipeline - m_bCut)
                    static const bool s_bNoMaskedCasters = xlionrender::Env("XLION_SH_NO_MASKED") != nullptr;     // (measurement: the masked casters' share)
                    if( s_bNoMaskedCasters && Vis.m_Opacity == material_visibility::opacity::MASKED && !bOpaqueCut ) continue;
                    auto* pCut = Vis.m_Opacity == material_visibility::opacity::MASKED && !bOpaqueCut ? getMaterialInstance(*pMat, G.m_Kind, true) : nullptr;
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
                            std::array<xgpu::buffer*, 12> Buffers = bSkin ? std::array<xgpu::buffer*, 12>{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy, &m_MaterialParams, &m_EntityParams, TailBuffer() }
                                                                          : std::array<xgpu::buffer*, 12>{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, TailBuffer(), nullptr, nullptr };
                            Cmd.setPipelineInstance(*pCut, std::span{ Buffers.data(), bSkin ? std::size_t{12} : std::size_t{10} });
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
                        if( pSunPage && !bSkin )
                        {
                            // (its clusters' light-space boxes once a frame, not once a page: 2-4 ms of recording a frame while the sun turned, Bistro)
                            auto& CL = ClusterLight[i];
                            if( CL.size() != G.m_Clusters.size() ) { CL.resize(G.m_Clusters.size()); for( std::size_t k = 0; k < CL.size(); ++k ) CL[k] = m_VSM.Bounds(G.m_Clusters[k].m_BBox.TransformCopy(D.m_L2W)); }
                            if( !shadows::RectOverlaps(CL[c], pSunPage->m_MinX, pSunPage->m_MinY, pSunPage->m_Size) ) continue;
                        }
                        if( bSkin ) Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Draws[i].m_Slot, 0u, MaterialBase, G.m_PrevBones });
                        else        Cmd.setPushConstants(details::push_static_geom{ c, Draws[i].m_Slot, 0u, 0xFFFFFFFFu, MaterialBase });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                        ++m_ShadowDrawStats.m_CpuDraws; m_ShadowDrawStats.m_CpuTris += C.m_nIndices / 3;
                    }
                }
            }
        };

        const auto Viewport = [&]( xgpu::cmd_buffer& Cmd, int X, int Y, int S )
        {
            Cmd.setViewport(float(X), float(Y), float(S), float(S));
            Cmd.setScissor(X, Y, S, S);
        };

        // The GI's sun shadow map: the relight's sun visibility (xlionrender_gi_links_comp.glsl SunMapVis) - a shadow ray a hit was 98% of
        // the relight (Bistro: 8.3 ms a full relight, 0.19 without it). Lumen traces these rays (divergence ledger); a map over the whole
        // level (the camera's virtual pages miss what is off screen), 2048 x 2048 along the sun, drawn only when the sun turns or the GI's
        // geometry's box changes. The relight waits for it (gpu_trace: a sun change relights when the map is the new sun's)
        if( m_pGITrace && Scene.m_GI.m_bEnable && !m_bGISunRays && Scene.m_Sun.m_Direction.Dot(Scene.m_Sun.m_Direction) > 1e-12f )
        {
            xmath::fvec3 Lo{ 1e30f, 1e30f, 1e30f }, Hi{ -1e30f, -1e30f, -1e30f };
            std::vector<std::uint32_t> GICasters;
            for( std::uint32_t i = 0; i < Draws.size(); ++i )
            {
                const auto& D = *Draws[i].m_pDraw;
                if( D.m_Instance == 0 || getGIId(D.m_Instance) < 0 ) continue;      // (what the GI's BVH holds: the rays' casters)
                GICasters.push_back(i);
                Lo = xmath::fvec3::Min(Lo, Draws[i].m_World.m_Min); Hi = xmath::fvec3::Max(Hi, Draws[i].m_World.m_Max);
            }
            const xmath::fvec3 Z = Scene.m_Sun.m_Direction.NormalizeCopy();
            const std::array<float, 9> Key{ Z.m_X, Z.m_Y, Z.m_Z, Lo.m_X, Lo.m_Y, Lo.m_Z, Hi.m_X, Hi.m_Y, Hi.m_Z };
            if( !m_bGISunMapTried )
            {
                m_bGISunMapTried = true;
                m_bGISunMapMade = !m_Device.Create(m_GISunMap, { .m_Format = xgpu::texture::format::DEPTH_U16, .m_Width = gi_sun_map_v, .m_Height = gi_sun_map_v, .m_isGamma = false });
                if( m_bGISunMapMade )
                {
                    auto Attachments = std::array{ xgpu::renderpass::attachment{ m_GISunMap } };
                    m_bGISunMapMade = !m_Device.Create(m_GISunPass, { .m_Attachments = Attachments }) && !m_Device.Create(m_GISunTimes, { .m_Count = 2 });
                }
            }
            if( m_bGISunMapMade && !GICasters.empty() && Key != m_GISunKey )
            {
                // the light's basis (as the virtual pages': x, y across, z along the light) and the box's extent in it
                const xmath::fvec3 Up = std::abs(Z.m_Y) < 0.99f ? xmath::fvec3{ 0, 1, 0 } : xmath::fvec3{ 1, 0, 0 };
                const xmath::fvec3 X = Up.Cross(Z).NormalizeCopy(), Y = Z.Cross(X);
                float Mn[3] = { 1e30f, 1e30f, 1e30f }, Mx[3] = { -1e30f, -1e30f, -1e30f };
                for( int c = 0; c < 8; ++c )
                {
                    const xmath::fvec3 P{ (c & 1) ? Hi.m_X : Lo.m_X, (c & 2) ? Hi.m_Y : Lo.m_Y, (c & 4) ? Hi.m_Z : Lo.m_Z };
                    const float v[3] = { P.Dot(X), P.Dot(Y), P.Dot(Z) };
                    for( int k = 0; k < 3; ++k ) { Mn[k] = std::min(Mn[k], v[k]); Mx[k] = std::max(Mx[k], v[k]); }
                }
                const float Ext = std::max(Mx[0] - Mn[0], Mx[1] - Mn[1]) * 1.01f + 0.1f, Cx = 0.5f * (Mn[0] + Mx[0]), Cy = 0.5f * (Mn[1] + Mx[1]);
                const float ZMin = Mn[2] - 0.1f, ZRange = Mx[2] - Mn[2] + 0.2f;
                const float SX = 2.0f / Ext, SZ = 1.0f / ZRange;
                xmath::fmat4 M = xmath::fmat4::fromZero();
                M(0, 0) = X.m_X * SX; M(0, 1) = X.m_Y * SX; M(0, 2) = X.m_Z * SX; M(0, 3) = -Cx * SX;
                M(1, 0) = Y.m_X * SX; M(1, 1) = Y.m_Y * SX; M(1, 2) = Y.m_Z * SX; M(1, 3) = -Cy * SX;
                M(2, 0) = Z.m_X * SZ; M(2, 1) = Z.m_Y * SZ; M(2, 2) = Z.m_Z * SZ; M(2, 3) = -ZMin * SZ;
                M(3, 3) = 1.0f;
                static const bool s_bTimes = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                if( s_bTimes )
                    if( std::vector<std::uint64_t> Ns; m_GISunTimes.getResults(Ns) && Ns.size() >= 2 && Ns[1] > Ns[0] )
                        std::printf("GISUNMAP the last draw %.3f ms (%zu casters, %.1f m a texel)\n", double(Ns[1] - Ns[0]) * 1e-6, GICasters.size(), double(Ext) / gi_sun_map_v), std::fflush(stdout);
                Window.WriteTimestamp(m_GISunTimes, 0);
                {
                    auto SunCmd = Window.StartRenderPass(m_GISunPass);
                    SunCmd.setViewport(0.0f, 0.0f, float(gi_sun_map_v), float(gi_sun_map_v));
                    SunCmd.setScissor(0, 0, gi_sun_map_v, gi_sun_map_v);
                    const bool bPairsWas = bPairs; bPairs = false;      // (the pairs draw the camera's pages only: every caster by the CPU path here)
                    const bool bOpaqueCutWas = bOpaqueCut;
                    bOpaqueCut = true;
                    DrawCasters(SunCmd, M, GICasters, [](std::uint32_t) { return true; });
                    bOpaqueCut = bOpaqueCutWas;
                    bPairs = bPairsWas;
                }
                Window.WriteTimestamp(m_GISunTimes, 1);
                m_GISunW2C = M; m_GISunTexel = Ext / float(gi_sun_map_v); m_GISunDepthPerMeter = SZ; m_GISunKey = Key;
            }
            m_bGISunMapValid = m_bGISunMapMade && Key == m_GISunKey;
        }
        else m_bGISunMapValid = false;

        // The skinned casters (XLION_SH_NO_SKIN_PAGES: page by page, cluster by cluster, as before): each one's pages - the sun's of its kind its
        // light-space rect overlaps, the local lights' its box reaches in the page's frustum - in a list, the pages once, all of it uploaded before
        // the pass; drawn a cluster at a time, an instance a page (xlionrender_shadow_skin_pages_vert.glsl)
        static const bool s_bNoSkinPages = xlionrender::Env("XLION_SH_NO_SKIN_PAGES") != nullptr;
        const auto IsSkin = [&]( std::uint32_t i ) { return Draws[i].m_pMesh->m_bCluster && Draws[i].m_pMesh->m_Geom.m_Kind == geom_kind::SKIN; };
        struct skin_page { xmath::fmat4 m_W2C; xmath::fvec4 m_Atlas, m_Rect; };
        static_assert(sizeof(skin_page) == 96);
        struct skin_run { std::uint32_t m_Draw, m_First, m_Count; float m_Texel = 1e30f; };     // (m_Texel: its pages' finest texel at the skin, world units - its LOD)
        std::vector<skin_page> SkinPages;
        std::vector<std::uint32_t> SkinList;
        std::vector<skin_run> SkinSun, SkinLocal;
        const int SkinSlot = int(m_Device.getFrameIndex() % ring_frames_v);
        bool bSkinPages = !s_bNoSkinPages && m_ShadowSkinPagesInstance.m_Private;
        if( bSkinPages )
        {
            const float AW = float(sunAtlasWidth()), AH = float(sunAtlasHeight()), T = float(shadows::vsm::page_texels_v);
            const float LW = float(LocalW), LH = float(LocalH), TL = float(shadows::local_pages::page_texels_v);
            const std::uint32_t SunBase = std::uint32_t(SkinSlot * skin_pages_v);      // (the list holds the pages' places in the whole buffer: the frame's region first)
            for( const auto& P : m_VSM.m_ToRender ) SkinPages.push_back({ P.m_W2C, { T / AW, T / AH, (T + 2.0f * float(P.m_AtlasX)) / AW - 1.0f, (T + 2.0f * float(P.m_AtlasY)) / AH - 1.0f }, {} });
            // The mover maps (XLION_SH_MOVER_MAPS): a skinned mover's tile - its light-space rect, square, a texel's margin; its depth range along the light
            m_MoverMaps.clear();
            std::vector<std::uint32_t> MapOfDraw(Draws.size(), ~0u);
            std::vector<std::vector<std::uint32_t>> LampPagesOfDraw(Draws.size());      // (the lamps' maps of a skinned mover: their places in SkinPages)
            if( bMoverMaps )
            {
                const int nMaps = sunAtlasWidth() / mover_map_texels_v;
                const float TM = float(mover_map_texels_v);
                for( std::uint32_t i = 0; i < Draws.size() && int(m_MoverMaps.size()) < nMaps; ++i )
                {
                    if( !IsSkin(i) ) continue;
                    const auto B = m_VSM.Bounds(Draws[i].m_World);
                    const float Ext = std::max(B.m_Max[0] - B.m_Min[0], B.m_Max[1] - B.m_Min[1]);
                    const float S   = Ext * (TM + 8.0f) / TM + 1e-3f;                     // (4 texels each side: the filter's reach)
                    const float Cx  = 0.5f * (B.m_Min[0] + B.m_Max[0]), Cy = 0.5f * (B.m_Min[1] + B.m_Max[1]);
                    mover_map M{ Cx - 0.5f * S, Cy - 0.5f * S, S, B.m_Min[2] - 0.05f, B.m_Max[2] - B.m_Min[2] + 0.1f, float(int(m_MoverMaps.size()) * mover_map_texels_v), float(sunAtlasHeight() - mover_map_texels_v), 0 };
                    xmath::fmat4 W = xmath::fmat4::fromZero();
                    const float SX = 2.0f / M.m_Size, SZ = 1.0f / M.m_ZRange;
                    const auto& X = m_VSM.m_X; const auto& Y = m_VSM.m_Y; const auto& Z = m_VSM.m_Z;
                    W(0, 0) = X.m_X * SX; W(0, 1) = X.m_Y * SX; W(0, 2) = X.m_Z * SX; W(0, 3) = -M.m_X0 * SX - 1.0f;
                    W(1, 0) = Y.m_X * SX; W(1, 1) = Y.m_Y * SX; W(1, 2) = Y.m_Z * SX; W(1, 3) = -M.m_Y0 * SX - 1.0f;
                    W(2, 0) = Z.m_X * SZ; W(2, 1) = Z.m_Y * SZ; W(2, 2) = Z.m_Z * SZ; W(2, 3) = -M.m_Z0 * SZ;
                    W(3, 3) = 1.0f;
                    MapOfDraw[i] = std::uint32_t(m_MoverMaps.size());
                    SkinPages.push_back({ W, { TM / AW, TM / AH, (TM + 2.0f * M.m_AtlasX) / AW - 1.0f, (TM + 2.0f * M.m_AtlasY) / AH - 1.0f }, {} });
                    m_MoverMaps.push_back(M);
                }
                // The lamps' maps: each shadowed lamp with a light slot whose sphere reaches a skinned mover, a perspective tile toward the mover's box
                const int nLampTiles = 2 * (sunAtlasWidth() / lamp_map_texels_v);
                const float TLm = float(lamp_map_texels_v);
                for( std::uint32_t i = 0; i < Draws.size(); ++i )
                {
                    if( MapOfDraw[i] == ~0u ) continue;
                    const auto& Bx = Draws[i].m_World;
                    const xmath::fvec3 Cn = (Bx.m_Min + Bx.m_Max) * 0.5f;
                    const float Rr = (Bx.m_Max - Bx.m_Min).Length() * 0.5f * 1.05f;
                    for( std::size_t li = 0; li < Scene.m_LocalLights.size() && int(m_LampMaps.size()) < nLampTiles; ++li )
                    {
                        const auto& L = Scene.m_LocalLights[li];
                        const int Slot = m_LocalPages.SlotOfLight(int(li));
                        if( !L.m_bCastShadows || Slot < 0 || !shadows::Overlaps(Bx, L.m_Position, L.m_Radius) ) continue;
                        const xmath::fvec3 ToC = Cn - L.m_Position;
                        const float Dist = ToC.Length();
                        if( Dist <= Rr + 0.05f ) continue;                      // (the lamp inside its box: no map - ponytail: its mover pages' shadow is gone there)
                        const xmath::fvec3 Ax = ToC / Dist;
                        const xmath::fvec3 Up = std::abs(Ax.m_Y) < 0.99f ? xmath::fvec3{ 0, 1, 0 } : xmath::fvec3{ 1, 0, 0 };
                        const xmath::fvec3 Rt = Up.Cross(Ax).NormalizeCopy(), Uv = Ax.Cross(Rt);
                        const float Tan = Rr / std::sqrt(Dist * Dist - Rr * Rr), Nr = Dist - Rr, Fr = Dist + Rr;
                        const float A = Fr / (Fr - Nr), B = -Fr * Nr / (Fr - Nr);
                        const int k = int(m_LampMaps.size()), PerRow = sunAtlasWidth() / lamp_map_texels_v;
                        lamp_map LM{ { Ax.m_X, Ax.m_Y, Ax.m_Z }, { Rt.m_X, Rt.m_Y, Rt.m_Z }, { Uv.m_X, Uv.m_Y, Uv.m_Z }, Tan, A, B, Nr
                                   , float((k % PerRow) * lamp_map_texels_v), float(sunAtlasHeight() - MoverMapRows() + (k / PerRow) * lamp_map_texels_v), std::uint32_t(Slot) };
                        xmath::fmat4 W = xmath::fmat4::fromZero();
                        const float lR = L.m_Position.Dot(Rt), lU = L.m_Position.Dot(Uv), lA = L.m_Position.Dot(Ax);
                        W(0, 0) = Rt.m_X / Tan; W(0, 1) = Rt.m_Y / Tan; W(0, 2) = Rt.m_Z / Tan; W(0, 3) = -lR / Tan;
                        W(1, 0) = Uv.m_X / Tan; W(1, 1) = Uv.m_Y / Tan; W(1, 2) = Uv.m_Z / Tan; W(1, 3) = -lU / Tan;
                        W(2, 0) = Ax.m_X * A;   W(2, 1) = Ax.m_Y * A;   W(2, 2) = Ax.m_Z * A;   W(2, 3) = -lA * A + B;
                        W(3, 0) = Ax.m_X;       W(3, 1) = Ax.m_Y;       W(3, 2) = Ax.m_Z;       W(3, 3) = -lA;
                        LampPagesOfDraw[i].push_back(std::uint32_t(SkinPages.size()));
                        SkinPages.push_back({ W, { TLm / AW, TLm / AH, (TLm + 2.0f * LM.m_AtlasX) / AW - 1.0f, (TLm + 2.0f * LM.m_AtlasY) / AH - 1.0f }, {} });
                        m_LampMaps.push_back(LM);
                    }
                }
            }
            const std::uint32_t LocalBase = SunBase + std::uint32_t(SkinPages.size());
            std::vector<frustum> LocalFrustum;
            for( const auto& P : m_LocalPages.m_ToRender )
            {
                SkinPages.push_back({ P.m_W2C, { TL / LW, TL / LH, (TL + 2.0f * float(P.m_AtlasX)) / LW - 1.0f, (TL + 2.0f * float(P.m_AtlasY)) / LH - 1.0f }, {} });
                LocalFrustum.push_back(frustum::FromW2C(P.m_W2C));
            }
            for( std::uint32_t i = 0; i < Draws.size(); ++i )
            {
                if( !IsSkin(i) ) continue;
                skin_run Run{ i, std::uint32_t(SkinList.size()), 0 };
                if( MapOfDraw[i] != ~0u ) { SkinList.push_back(SunBase + std::uint32_t(m_VSM.m_ToRender.size()) + MapOfDraw[i]); Run.m_Texel = m_MoverMaps[MapOfDraw[i]].m_Size / float(mover_map_texels_v); }
                for( const auto p : LampPagesOfDraw[i] ) SkinList.push_back(SunBase + p);
                for( std::uint32_t p = 0; p < m_VSM.m_ToRender.size() && MapOfDraw[i] == ~0u; ++p )
                {
                    const auto& P = m_VSM.m_ToRender[p];
                    if( Draws[i].m_bMover == P.m_bMover && shadows::RectOverlaps(LightBounds[i], P.m_MinX, P.m_MinY, P.m_Size) )
                    {
                        SkinList.push_back(SunBase + p);
                        Run.m_Texel = std::min(Run.m_Texel, m_VSM.Level0Texel() * float(1 << P.m_Key.m_Level));
                    }
                }
                Run.m_Count = std::uint32_t(SkinList.size()) - Run.m_First;
                if( Run.m_Count ) SkinSun.push_back(Run);
                Run = { i, std::uint32_t(SkinList.size()), 0 };
                for( std::uint32_t p = 0; p < m_LocalPages.m_ToRender.size() && MapOfDraw[i] == ~0u; ++p )      // (a mover with maps: none - its lamps' maps are in its sun run)
                {
                    const auto& P = m_LocalPages.m_ToRender[p];
                    const auto& L = Scene.m_LocalLights[std::size_t(P.m_Light)];
                    if( (!m_bShadowSplitMovers || P.m_bWithMovers || Draws[i].m_bMover == P.m_bMover) && shadows::Overlaps(Draws[i].m_World, L.m_Position, L.m_Radius)
                        && (!m_bShadowFaceCull || LocalFrustum[p].Overlaps(Draws[i].m_World, false)) )
                    {
                        SkinList.push_back(LocalBase + p);
                        // (a face's texel at the skin: 2 d tan(half fov) over the level's texels across)
                        const float Dist = ((Draws[i].m_World.m_Min + Draws[i].m_World.m_Max) * 0.5f - L.m_Position).Length();
                        const float Tan  = L.m_Type == local_light::type::POINT ? 1.0f : shadows::SpotTanHalf(L);
                        Run.m_Texel = std::min(Run.m_Texel, 2.0f * Dist * Tan / float((shadows::local_pages::level0_pages_v >> P.m_Key.m_Level) * shadows::local_pages::page_texels_v));
                    }
                }
                Run.m_Count = std::uint32_t(SkinList.size()) - Run.m_First;
                if( Run.m_Count ) SkinLocal.push_back(Run);
            }
            if( SkinPages.size() > std::size_t(skin_pages_v) || SkinList.size() > std::size_t(skin_page_list_v) ) bSkinPages = false;      // (too many this frame: page by page, as before)
            else
            {
                if( !SkinPages.empty() ) (void)m_SkinPagesBuffer.MemoryMapFrameRegion(SkinSlot * skin_pages_v, int(SkinPages.size()), [&](void* p) { std::memcpy(p, SkinPages.data(), SkinPages.size() * sizeof(skin_page)); });
                if( !SkinList.empty() )  (void)m_SkinPagesList.MemoryMapFrameRegion(SkinSlot * skin_page_list_v, int(SkinList.size()), [&](void* p) { std::memcpy(p, SkinList.data(), SkinList.size() * sizeof(std::uint32_t)); });
            }
        }
        // A run's skinned draw, a cluster at a time, an instance a page of its list (its first instance: its list's start in the frame's region)
        const auto DrawSkinPages = [&]( xgpu::cmd_buffer& Cmd, std::span<const skin_run> Runs )
        {
            auto& ViewBlock = m_ShadowUBO.allocEntry<details::ubo_geom_view>();
            ViewBlock.m_w2C         = xmath::fmat4::fromIdentity();
            ViewBlock.m_WorldOffset = xmath::fvec4(m_Origin, 0);
            for( const auto& Run : Runs )
            {
                const auto& D = *Draws[Run.m_Draw].m_pDraw;
                const auto& G = Draws[Run.m_Draw].m_pMesh->m_Geom;
                std::array<xgpu::buffer*, 6> Buffers{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_SkinPagesBuffer, &m_SkinPagesList, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy };
                Cmd.setPipelineInstance(m_ShadowSkinPagesInstance, std::span{ Buffers });
                Cmd.setDynamicUBO(m_ShadowUBO, 0);
                BindClusterStreams(Cmd, G);
                // Each submesh at the coarsest LOD whose error (times the draw's scale) stays within its pages' finest texel - the static casters' rule
                // (PageSub), apart from the main view's pick: LOD 0 here drew every character's full mesh into every page (Bistro Game 2026-11-07:
                // 6.9 ms of shadow pass; the main view's LOD 3.2, every skin's coarsest 1.6)
                const float Scale = std::max({ (D.m_L2W * xmath::fvec3{ 1, 0, 0 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length(), (D.m_L2W * xmath::fvec3{ 0, 1, 0 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length()
                                             , (D.m_L2W * xmath::fvec3{ 0, 0, 1 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length() });
                static const int s_SkinLOD = [] { const char* p = xlionrender::Env("XLION_SKIN_LOD"); return p ? std::atoi(p) : -1; }();
                for( std::size_t s = 0; s < G.m_Submeshes.size(); ++s )
                {
                    int j = s_SkinLOD >= 0 ? s_SkinLOD : int(G.m_LODErrors.size()) - 1;
                    if( s_SkinLOD < 0 ) while( j > 0 && (std::size_t(j) >= G.m_LODErrors.size() || s >= G.m_LODErrors[std::size_t(j)].size() || G.m_LODErrors[std::size_t(j)][s] * Scale > Run.m_Texel) ) --j;
                    const auto& Sub = G.LODSubmeshes(j)[s];
                    auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                    const auto Vis = pMat ? MaterialVisibilityOf(pMat->m_State) : material_visibility{};
                    if( !Vis.m_bCastsShadow ) continue;         // (a masked skin: its cutout's not here - as an opaque one, ponytail: a skinned cutout caster if one shows)
                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
                        const auto& Cl = G.m_Clusters[c];
                        Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Draws[Run.m_Draw].m_Slot, 0u, 0u, G.m_PrevBones });
                        Cmd.DrawInstance(int(Run.m_Count), int(Cl.m_nIndices), int(SkinSlot * skin_page_list_v + Run.m_First), int(Cl.m_iIndex), int(Cl.m_iVertex));
                        ++m_ShadowDrawStats.m_IndirectDraws; m_ShadowDrawStats.m_IndirectMax += int(Run.m_Count);
                    }
                }
            }
        };

        ShCpu.Lap();
        auto Cmd = Window.StartRenderPass(m_ShadowPass);

            // The sun's virtual pages: each in its slot of the atlas, cleared, its casters (light-space rect test)
            // (XLION_SH_SPLIT, measurement - G0 of gpu_driven_plan_2026-11-07.md: the shadow pass by part, GPU ms, mean of 120 frames: the sun's pages
            // (clears + the CPU path), its skinned draws, its pairs; the lamps' likewise; the pages by kind)
            static const bool s_bShSplit = xlionrender::Env("XLION_SH_SPLIT") != nullptr;
            static xgpu::timestamps s_ShTS; static bool s_bShTS = false;
            if( s_bShSplit && !s_bShTS ) s_bShTS = m_Device.Create(s_ShTS, { .m_Count = 7 }) == nullptr;
            const bool bShTS = s_bShSplit && s_bShTS;
            if( bShTS )
            {
                static std::vector<std::uint64_t> s_Ns; static double s_Sum[6] = {}; static int s_N = 0, s_Pages[4] = {};
                if( s_ShTS.getResults(s_Ns) && s_Ns.size() >= 7 && s_Ns[6] >= s_Ns[0] )
                {
                    for( int k = 0; k < 6; ++k ) s_Sum[k] += s_Ns[k + 1] >= s_Ns[k] ? double(s_Ns[k + 1] - s_Ns[k]) * 1e-6 : 0.0;
                    for( const auto& P : m_VSM.m_ToRender ) ++s_Pages[P.m_bMover ? 1 : 0];
                    for( const auto& P : m_LocalPages.m_ToRender ) ++s_Pages[P.m_bMover ? 3 : 2];
                    if( ++s_N == 120 )
                    {
                        std::printf("SHSPLIT mean of 120 frames, GPU ms: sun pages (clears + CPU path) %.3f, sun skinned %.3f, sun pairs %.3f, lamp pages %.3f, lamp skinned %.3f, lamp pairs %.3f; pages a frame: sun static %.1f mover %.1f, lamp static %.1f mover %.1f\n"
                                   , s_Sum[0] / s_N, s_Sum[1] / s_N, s_Sum[2] / s_N, s_Sum[3] / s_N, s_Sum[4] / s_N, s_Sum[5] / s_N, s_Pages[0] / double(s_N), s_Pages[1] / double(s_N), s_Pages[2] / double(s_N), s_Pages[3] / double(s_N));
                        std::fflush(stdout); s_N = 0; for( auto& v : s_Sum ) v = 0; for( auto& v : s_Pages ) v = 0;
                    }
                }
                Cmd.WriteTimestamp(s_ShTS, 0);
            }
            if( !m_VSM.m_ToRender.empty() ) Cmd.WriteTimestamp(m_VSMRenderTimes, 0);
            m_ShadowDrawStats.m_PageClears += int(m_VSM.m_ToRender.size());      // (a viewport, a scissor and a clear draw each)
            m_ShadowDrawStats.m_CpuCasters  = int(AllCasters.size());
            for( const auto& P : m_VSM.m_ToRender )
            {
                Viewport(Cmd, P.m_AtlasX, P.m_AtlasY, shadows::vsm::page_texels_v);
                Cmd.setPipelineInstance(m_ShadowClearInstance);
                Cmd.setBuffer(m_FullscreenVB);
                Cmd.setBuffer(m_FullscreenIB);
                Cmd.Draw(3);
                pSunPage = &P;
                // (XLION_VSM_PAGE_LOG, measurement: each static sun page drawn - its key, the casters it kept, and the large casters it left out as movers:
                // the sunlit page-sized holes, Tomas 2026-11-07)
                static const bool s_bPageLog = xlionrender::Env("XLION_VSM_PAGE_LOG") != nullptr;
                if( s_bPageLog && !P.m_bMover )
                {
                    int nKept = 0, nBigMover = 0;
                    for( const auto i : AllCasters )
                    {
                        if( !shadows::RectOverlaps(LightBounds[i], P.m_MinX, P.m_MinY, P.m_Size) ) continue;
                        if( Draws[i].m_bMover ) { const auto E = Draws[i].m_World.m_Max - Draws[i].m_World.m_Min; nBigMover += (E.m_X + E.m_Y + E.m_Z) > 20.0f; }
                        else nKept += !(bSkinPages && IsSkin(i));
                    }
                    std::printf("VSMPAGE frame %llu: L%d (%d,%d) kept %d casters, left out %d large movers\n", (unsigned long long)m_Device.getFrameIndex(), P.m_Key.m_Level, P.m_Key.m_X, P.m_Key.m_Y, nKept, nBigMover);
                }
                DrawCasters(Cmd, P.m_W2C, AllCasters, [&](std::uint32_t i) { return !(bSkinPages && IsSkin(i)) && Draws[i].m_bMover == P.m_bMover && shadows::RectOverlaps(LightBounds[i], P.m_MinX, P.m_MinY, P.m_Size); }, PageLOD(P.m_Key.m_Level));
                pSunPage = nullptr;
            }
            if( bShTS ) Cmd.WriteTimestamp(s_ShTS, 1);
            for( const auto& M : m_LampMaps )
            {
                Viewport(Cmd, int(M.m_AtlasX), int(M.m_AtlasY), lamp_map_texels_v);
                Cmd.setPipelineInstance(m_ShadowClearInstance);
                Cmd.setBuffer(m_FullscreenVB);
                Cmd.setBuffer(m_FullscreenIB);
                Cmd.Draw(3);
            }
            for( const auto& M : m_MoverMaps )      // (the mover maps' tiles: cleared, then drawn with the skinned runs)
            {
                Viewport(Cmd, int(M.m_AtlasX), int(M.m_AtlasY), mover_map_texels_v);
                Cmd.setPipelineInstance(m_ShadowClearInstance);
                Cmd.setBuffer(m_FullscreenVB);
                Cmd.setBuffer(m_FullscreenIB);
                Cmd.Draw(3);
            }
            if( bSkinPages && (!m_VSM.m_ToRender.empty() || !m_MoverMaps.empty()) )        // (the skinned casters into all their sun pages at once: one viewport over the sun's atlas)
            {
                Cmd.setViewport(0.0f, 0.0f, float(sunAtlasWidth()), float(sunAtlasHeight()));
                Cmd.setScissor(0, 0, sunAtlasWidth(), sunAtlasHeight());
                DrawSkinPages(Cmd, SkinSun);
            }
            if( bShTS ) Cmd.WriteTimestamp(s_ShTS, 2);
            if( !m_VSM.m_ToRender.empty() && bPairs )      // the pairs: a draw a geometry, one viewport over the sun's atlas (each instance into its page)
            {
                const int AW = sunAtlasWidth(), AH = sunAtlasHeight();
                Cmd.setViewport(0.0f, 0.0f, float(AW), float(AH));
                Cmd.setScissor(0, 0, AW, AH);
                DrawPairs(Cmd, false);
            }
            if( !m_VSM.m_ToRender.empty() ) Cmd.WriteTimestamp(m_VSMRenderTimes, 1);
            if( bShTS ) Cmd.WriteTimestamp(s_ShTS, 3);

        // The local lights' pages (features/shadows/xlionrender_shadow_local_pages.h): each in its slot of the atlas's local part, cleared,
        // the casters its light's sphere reaches (once a light) in the page's frustum
        const int LocalX = sunAtlasWidth();
        std::vector<std::vector<std::uint32_t>> LightCasters(Scene.m_LocalLights.size());
        std::vector<std::uint8_t>               bLightCasters(Scene.m_LocalLights.size(), 0);
        for( const auto& P : m_LocalPages.m_ToRender )
        {
            const auto& L = Scene.m_LocalLights[std::size_t(P.m_Light)];
            auto&       C = LightCasters[std::size_t(P.m_Light)];
            if( !bLightCasters[std::size_t(P.m_Light)] )
            {
                bLightCasters[std::size_t(P.m_Light)] = 1;
                for( std::uint32_t i = 0; i < Draws.size(); ++i ) if( (!(bPairs && Draws[i].m_bGPU) || Draws[i].m_bCut) && shadows::Overlaps(Draws[i].m_World, L.m_Position, L.m_Radius) ) C.push_back(i);
            }
            Viewport(Cmd, LocalX + P.m_AtlasX, P.m_AtlasY, shadows::local_pages::page_texels_v);
            Cmd.setPipelineInstance(m_ShadowClearInstance);
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
            const frustum Page = frustum::FromW2C(P.m_W2C);
            DrawCasters(Cmd, P.m_W2C, C, [&](std::uint32_t i) { return !(bSkinPages && IsSkin(i)) && (!m_bShadowSplitMovers || P.m_bWithMovers || Draws[i].m_bMover == P.m_bMover) && (!m_bShadowFaceCull || Page.Overlaps(Draws[i].m_World, false)); });
        }
        if( bShTS ) Cmd.WriteTimestamp(s_ShTS, 4);
        if( bSkinPages && !m_LocalPages.m_ToRender.empty() )        // (the skinned casters into all their local pages at once: one viewport over the local part)
        {
            Cmd.setViewport(float(LocalX), 0.0f, float(LocalW), float(LocalH));
            Cmd.setScissor(LocalX, 0, LocalW, LocalH);
            DrawSkinPages(Cmd, SkinLocal);
        }
        if( bShTS ) Cmd.WriteTimestamp(s_ShTS, 5);
        if( !m_LocalPages.m_ToRender.empty() && bPairs )      // the pairs' local runs: one viewport over the local part (each instance into its page)
        {
            Cmd.setViewport(float(LocalX), 0.0f, float(LocalW), float(LocalH));
            Cmd.setScissor(LocalX, 0, LocalW, LocalH);
            DrawPairs(Cmd, true);
        }
        if( bShTS ) Cmd.WriteTimestamp(s_ShTS, 6);
    }

    //------------------------------------------------------------------------------------------------

    // The sun's pages rendered this frame, each reduced to its depth bounds in its slot's entry (S1b; the sun mask's early-out reads them)
    void instance::vsmBounds( xgpu::window& Window ) noexcept
    {
        if( !m_VSMBoundsCompute.m_Private || m_SunMaskMode < 4 || m_VSM.m_ToRender.empty() ) return;
        const int Slots = m_VSM.getSetup().Slots();
        const int n     = std::min(int(m_VSM.m_ToRender.size()), Slots);
        const int Base  = int(m_Device.getFrameIndex() % ring_frames_v) * Slots;
        (void)m_VSMBoundsList.MemoryMapFrameRegion(Base, n, [&](void* p)
        {
            auto* pO = static_cast<std::int32_t*>(p);
            for( int i = 0; i < n; ++i ) { pO[2 * i] = m_VSM.m_ToRender[i].m_AtlasX; pO[2 * i + 1] = m_VSM.m_ToRender[i].m_AtlasY; }
        });
        const std::uint32_t Push[8] = { std::uint32_t(Base), std::uint32_t(n), std::uint32_t(m_VSM.getSetup().SlotsPerRow()), std::uint32_t(m_Device.getFrameIndex()), std::uint32_t(Slots), 0u, 0u, 0u };
        auto Buffers  = std::array<xgpu::buffer*, 2>{ &m_VSMBoundsList, &m_VSMBounds };
        auto Textures = std::array<const xgpu::texture*, 1>{ &m_ShadowAtlas };
        Window.Dispatch(m_VSMBoundsCompute, Buffers, Textures, Push, sizeof(Push), n, 1);
    }

    //------------------------------------------------------------------------------------------------

    // The scene into View's HDR target (and depth), lit, from a camera-centered W2C at Eye: the main view, and the reflection
    // probes' capture faces (bMain false: no timestamps). Shadows, instances and cells of this frame are already up to date.
    // The screen tail (light_mode::SCREEN, the main view; documentation/lighting_p3_design.md "The decision"), before the scene pass: the thin
    // G-buffer at half size - the visible draws that are not blended, through their materials' G-buffer variants (a material without one is
    // left out: its pixels find no texel of theirs and shade their tail inline) -, the tail pass over it (demodulated diffuse and specular),
    // the denoiser (temporal, spatial) into the records the scene pass's materials read (binding 14). Sets the lighting block's mode 3 and
    // the tail's parameters; anything missing: mode 2 (the inline sampled tail)
    void instance::renderTail( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye, details::ubo_lighting& Light, bool bPlan ) noexcept
    {
        if( createTailTargets(View) ) { Light.m_LocalCount.m_Y = float(int(light_mode::CELLS)); return; }      // (no targets: the split lists exactly)
        const bool bLights = m_LightModeUsed == light_mode::SCREEN;      // (else only the G-buffer, for the reflections)
        const int W = View.m_TailWidth, H = View.m_TailHeight;
        if( std::vector<std::uint64_t> Ns; m_TailTimes.getResults(Ns) && Ns.size() >= 5 && Ns[1] >= Ns[0] && Ns[2] >= Ns[1] && Ns[4] >= Ns[2] && Ns[3] >= Ns[4] && std::ranges::none_of(Ns, [](std::uint64_t t) { return t == ~0ull; }) )     // (the tail lights off: 2 to 4 never written - they read ~0 and passed, 'tail lights 1.6e13 ms')
        {
            m_TailReport.m_GBufferMs = double(Ns[1] - Ns[0]) * 1e-6; m_TailReport.m_TailMs = double(Ns[2] - Ns[1]) * 1e-6; m_TailReport.m_DenoiseMs = double(Ns[3] - Ns[2]) * 1e-6;
            m_TailReport.m_TemporalMs = double(Ns[4] - Ns[2]) * 1e-6;
            static const bool s_bTailLog = xlionrender::Env("XLION_GPU_TIMES") != nullptr; static int s_TailN = 0;
            if( s_bTailLog && ++s_TailN % 120 == 0 ) { std::printf("TAILSPLIT thin G-buffers (half size + sharp full size) %.3f, tail lights %.3f, denoiser %.3f ms\n", m_TailReport.m_GBufferMs, m_TailReport.m_TailMs, m_TailReport.m_DenoiseMs); std::fflush(stdout); }
        }
        m_TailReport.m_bOn = bLights; m_TailReport.m_Width = W; m_TailReport.m_Height = H;
        m_TailReport.m_MemoryMB = double(W) * double(H) * (4 + 8 + 4 + 4 + 4 + 2 * 16) / (1024.0 * 1024.0);     // (G-buffer, motion, depth, the tail's two, the denoiser's two)

        // A camera cut (a jump past 2 m, or the view turned past 45 degrees): no history (the GI screen pass's rule)
        const xmath::fmat4 C2W = W2C.Inverse();
        if( View.m_bTailPrev )
        {
            const xmath::fmat4 PrevW2C = View.m_TailPrevC2W.Inverse();
            const auto Fwd = []( const xmath::fmat4& M ) { const float* p = reinterpret_cast<const float*>(&M); const xmath::fvec3 F{ p[3], p[7], p[11] }; return F / std::max(F.Length(), 1e-12f); };
            const xmath::fvec3 Jump = Eye - View.m_TailPrevEye;
            if( Jump.Dot(Jump) > 4.0f || Fwd(W2C).Dot(Fwd(PrevW2C)) < 0.7071f ) View.m_bTailPrev = false;
        }

        Light.m_TailParams = { 1.0f, float(W), float(H), 0.0f };
        Light.m_TailC2W    = C2W;
        Light.m_TailW2C    = W2C;
        Light.m_TailTrace  = { m_TailTraceLength, float(m_TailTraceSteps), 0.05f, 0.0f };
        // P4: the GI's BVH (gi::gpu_trace::getBvh, after this frame's updateGI), never walked when it is a placeholder (bEmpty: its root walks
        // off the buffers and loses the device); without it the GI dummy is bound and the screen trace alone shadows those lights
        const auto Bvh = m_pGITrace && m_bTailTraceBvh ? m_pGITrace->getBvh() : gi::gpu_trace::bvh_view{ nullptr, nullptr, nullptr, nullptr, 0, 0, true };
        Light.m_TailBvh    = { Bvh.m_bEmpty ? 0.0f : 1.0f, std::bit_cast<float>(Bvh.m_TlasRoot), std::bit_cast<float>(Bvh.m_PrimOffset), 0.0f };
        if( bLights ) Light.m_LocalCount.m_Y = float(int(light_mode::SCREEN));

        // Reflections' sharp receivers: the G-buffer at full size too (createSharpTargets: made when first needed; its rays a pixel each)
        static const float s_SharpEnv = [] { const char* p = xlionrender::Env("XLION_REFL_SHARP"); return p ? float(std::atof(p)) : -1.0f; }();      // (measurement)
        if( s_SharpEnv >= 0.0f ) m_ReflSharpMax = s_SharpEnv;
        const bool bSharp = xlionrender::Env("XLION_REFL_PT") == nullptr && m_ReflTraceBench == 0 && Scene.m_ReflectionMaxRoughness > 0 && m_ReflSharpMax > 0 && m_bReflSharpSeen && !Scene.m_bDebugReflectionMirror && xlionrender::Env("XLION_SDF") == nullptr && createSharpTargets(View) == nullptr;
        Light.m_TailParams.m_X = bSharp ? float(View.m_SharpWidth) : 0.0f;      // (the materials: the full-size records' row, 0: none)
        Window.WriteTimestamp(m_TailTimes, 0);
        const auto DrawGBuffer = [&]( xgpu::renderpass& Pass )
        {
            const xmath::fmat4 ToEye = xmath::fmat4::fromTranslation(-Eye);
            auto Cmd = Window.StartRenderPass(Pass);
            const void* pBound = nullptr; const cluster_geom* pBoundGeom = nullptr;
            int nDraws = 0, nNoVariant = 0;
            for( const auto& Vis : m_Visible )
            {
                const std::size_t iDraw = Vis.m_iDraw;
                const auto&       D     = Scene.m_Draws[iDraw];
                const auto*       pMesh = Vis.m_pMesh;
                if( pMesh->m_bCluster == false )
                {
                    std::array<xgpu::buffer*, 3> LightBuffers{ &m_LightRing, &m_CellRing, TailBuffer() };
                    Cmd.setPipelineInstance(m_ForwardGBInstance, LightBuffers);
                    Cmd.setDynamicUBO(m_LightUBO, 1);
                    auto& Mat = m_MaterialUBO.allocEntry<details::ubo_material>();
                    Mat.m_BaseColor = xmath::fvec4(D.m_Material.m_BaseColor, 1);
                    Mat.m_Params    = { D.m_Material.m_Metallic, D.m_Material.m_Roughness, D.m_Material.m_AO, D.m_Material.m_F0 };
                    Mat.m_Emissive  = xmath::fvec4(D.m_Material.m_Emissive, D.m_Material.m_bEditorGrid ? 1.0f : 0.0f);
                    Cmd.setDynamicUBO(m_MaterialUBO, 2);
                    Cmd.setPushConstants(details::push_forward{ ToEye * D.m_L2W, W2C });
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_VertexBuffer));
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_IndexBuffer));
                    Cmd.Draw(pMesh->m_nIndices);
                    pBound = nullptr; ++nDraws;
                    continue;
                }
                const std::uint32_t Slot = m_DrawSlot[iDraw];
                if( Slot == instance_slots::none_v || (bPlan && m_bBoxesOnly) ) continue;
                const auto& G     = pMesh->m_Geom;
                const bool  bSkin = G.m_Kind == geom_kind::SKIN;
                const std::uint32_t SkipBase = bPlan ? m_ClusterBase[iDraw] : ~0u;
                for( const auto& Sub : G.m_Submeshes )
                {
                    auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                    if( pMat && pMat->m_State.isBlended() ) continue;
                    auto* pInstance = pMat ? getGBufferInstance(*pMat, G.m_Kind) : (bSkin ? nullptr : &m_StaticGeomGBInstance);
                    if( pInstance == nullptr ) { ++nNoVariant; continue; }
                    if( pInstance->m_Private.get() != pBound || &G != pBoundGeom )
                    {
                        if( !bSkin )
                        {
                            std::array<xgpu::buffer*, 10> Buffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, TailBuffer() };
                            Cmd.setPipelineInstance(*pInstance, std::span{ Buffers });
                        }
                        else
                        {
                            std::array<xgpu::buffer*, 12> Buffers{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy, &m_MaterialParams, &m_EntityParams, TailBuffer() };
                            Cmd.setPipelineInstance(*pInstance, std::span{ Buffers });
                        }
                        Cmd.setDynamicUBO(m_LightUBO, 1);
                        Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                        BindClusterStreams(Cmd, G);
                        pBound = pInstance->m_Private.get(); pBoundGeom = &G;
                    }
                    if( pMat && pMat->m_bTable ) Cmd.setTextureTable(pMat->m_Table);
                    const std::uint32_t MaterialBase = pMat ? pMat->m_ParamBase : 0u;
                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
                        if( SkipBase != ~0u && m_ClusterSkip[SkipBase + c] ) continue;
                        const auto& C = G.m_Clusters[c];
                        if( bSkin ) Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Slot, 0u, MaterialBase, G.m_PrevBones });
                        else        Cmd.setPushConstants(details::push_static_geom{ c, Slot, 0u, 0xFFFFFFFFu, MaterialBase });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                        ++nDraws;
                    }
                }
            }
            m_TailReport.m_nGBufferDraws = nDraws; m_TailReport.m_nNoVariant = nNoVariant;
        };
        DrawGBuffer(View.m_TailGPass);
        if( bSharp ) DrawGBuffer(View.m_SharpGPass);
        Window.WriteTimestamp(m_TailTimes, 1);
        if( bLights )
        {
        {
            auto Cmd = Window.StartRenderPass(View.m_TailPass);
            const auto B = [&](const xgpu::buffer* p) { return Bvh.m_bEmpty ? &m_GIDummy : const_cast<xgpu::buffer*>(p); };
            std::array<xgpu::buffer*, 6> Buffers{ &m_LightRing, &m_CellRing, B(Bvh.m_pNodes), B(Bvh.m_pTri4), B(Bvh.m_pInstances), B(Bvh.m_pTlas) };
            Cmd.setPipelineInstance(View.m_TailInstance, Buffers);
            Cmd.setDynamicUBO(m_LightUBO, 1);
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
        }
        Window.WriteTimestamp(m_TailTimes, 2);
        {
            // the denoiser's frame (shaders/xlionrender_tail_denoise_comp.glsl frame_t): this frame's clip -> world, this frame's clip -> last
            // frame's (the eye's move between: the camera's reprojection, where the G-buffer has no motion), the size, whether last frame's
            // is there, the frame's number
            const int Slot = int(m_Device.getFrameIndex() % ring_frames_v);
            const xmath::fmat4 Reproject = View.m_bTailPrev ? View.m_TailPrevW2C * xmath::fmat4::fromTranslation(Eye - View.m_TailPrevEye) * C2W : xmath::fmat4::fromIdentity();
            const xmath::fvec4 Params{ float(W), float(H), View.m_bTailPrev ? 1.0f : 0.0f, float(View.m_TailFrame & 1023u) };
            (void)m_TailFrames.MemoryMapFrameRegion(Slot, 1, [&](void* p)
            {
                auto* pB = static_cast<std::byte*>(p);
                std::memcpy(pB, &C2W, 64); std::memcpy(pB + 64, &Reproject, 64); std::memcpy(pB + 128, &Params, 16);
            });
            auto Buffers  = std::array<xgpu::buffer*, 3>{ &m_TailFrames, &View.m_TailResult, &View.m_TailTemporal };
            auto Textures = std::array<const xgpu::texture*, 5>{ &View.m_TailDiffuse, &View.m_TailSpecular, &View.m_TailGColor, &View.m_TailGDepth, &View.m_TailGMotion };
            for( std::uint32_t Pass = 0; Pass < 2; ++Pass )
            {
                const std::uint32_t Push[8] = { Pass, std::uint32_t(Slot), 0u, 0u, 0u, 0u, 0u, 0u };
                Window.Dispatch(m_TailDenoise, Buffers, Textures, Push, sizeof(Push), (W + 7) / 8, (H + 7) / 8);
                if( Pass == 0 ) Window.WriteTimestamp(m_TailTimes, 4);
            }
        }
        Window.WriteTimestamp(m_TailTimes, 3);
        }
        // Reflections (shaders/xlionrender_refl_trace_comp.glsl; documentation/reflections_strategy.md milestone B): a ray a G-buffer texel
        // through the GI's BVH, its hit lit with the GI's model, into the records' second region - the materials' EnvLighting reads it
        // (UBOLight.EnvParams.w: the roughness up to which, 0 none this frame). Its counters of this ring slot (written ring_frames_v frames
        // ago, done) into the report, cleared. m_ReflTraceBench > 0: the benchmark's rays only (mirror, roughness 0.5, the materials')
        const auto BvhR = m_pGITrace ? m_pGITrace->getBvh() : gi::gpu_trace::bvh_view{ nullptr, nullptr, nullptr, nullptr, 0, 0, true };
        Light.m_EnvParams.m_W = 0.0f;
        static const bool s_bGradeRays = xlionrender::Env("XLION_REFL_GRADE_RAYS") != nullptr;      // (rays graded too, at XLION_REFL_GRADE's roughness: half size, one frame, no reconstruction: its noise counts against them)
        static const int  s_GradeRough = [] { const char* p = xlionrender::Env("XLION_REFL_GRADE"); return p ? int(std::atof(p) * 1000.0 + 0.5) : 0; }();
        static const bool s_bGrading = xlionrender::Env("XLION_REFL_GRADE") != nullptr && !s_bGradeRays;      // (what's graded is the read at a roughness; the mirror view's rays are mirror rays, painted over it)
        const bool bTraced = !s_bGrading && !Scene.m_bDebugReflectionPathTraced && (m_ReflTraceBench > 0 || Scene.m_ReflectionMaxRoughness > 0) && !BvhR.m_bEmpty && BvhR.m_pData;
        const bool bScreen = ReflectionScreen(Scene) && m_ReflTraceBench == 0 && !Scene.m_bDebugReflection && xlionrender::Env("XLION_REFL_PT") == nullptr;      // (its colors are last frame's image: in the debug view, reflections of reflections)
        const int  Slot    = int(m_Device.getFrameIndex() % ring_frames_v);
        static const bool s_bReflPT = xlionrender::Env("XLION_REFL_PT") != nullptr;      // (A12: the reflections' ground truth, real materials)
        static const bool s_bNoRecon = xlionrender::Env("XLION_REFL_NO_RECON") != nullptr;      // (measurement: the records as traced, no reconstruction)
        const bool bReconstruct = !s_bReflPT && !s_bNoRecon && Scene.m_bReflectionReconstruct && m_ReflTraceBench == 0 && (!Scene.m_bDebugReflectionMirror || xlionrender::Env("XLION_REFL_GRADE_RECONSTRUCT") != nullptr);      // (its kernel follows the real roughness)
        if( bTraced || bScreen )
        {
            if( bScreen || bReconstruct )
            {
                const xmath::fmat4 ToPrev = View.m_bTailPrev ? View.m_TailPrevW2C * xmath::fmat4::fromTranslation(Eye - View.m_TailPrevEye) : xmath::fmat4::fromIdentity();
                const xmath::fvec4 RP{ float(W), float(H), View.m_bTailPrev ? 1.0f : 0.0f, float(View.m_TailFrame & 1023u) };
                (void)m_ReflFrames.MemoryMapFrameRegion(Slot, 1, [&](void* p)
                {
                    auto* pB = static_cast<std::byte*>(p);
                    std::memcpy(pB, &C2W, 64); std::memcpy(pB + 64, &ToPrev, 64); std::memcpy(pB + 128, &RP, 16);
                });
            }
            // (up to roughness 0.5, the voxels past it: at every roughness it cost 0.3 to 0.8 ms more on Bistro for no better image)
            // the screen trace (SSR: xlionrender_refl_ssr_comp.glsl) before the BVH trace, which leaves its certain answers: last frame's lit image
            // (View.m_HDR: the scene pass has not drawn yet) at the ray's hit in this frame's thin G-buffer. Every roughness: the probes stay
            // the base where it misses (off screen, hidden)
            if( bScreen )
            {
                if( std::vector<std::uint64_t> Ns; m_ReflSSRTimes.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] > Ns[0] )
                {
                    m_ReflSSRSum += double(Ns[1] - Ns[0]) * 1e-6;
                    static const bool s_bTimes = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                    if( ++m_ReflSSRSumN == 120 )
                    {
                        if( s_bTimes ) std::printf("SSRGPU %d x %d: %.3f ms (the screen trace and the reconstruction)\n", W, H, m_ReflSSRSum / m_ReflSSRSumN), std::fflush(stdout);
                        m_ReflSSRSum = 0; m_ReflSSRSumN = 0;
                    }
                }
                constexpr float ssr_max_rough_v = 0.5f;
                const auto SSRSteps = [&] { static const int s = [] { const char* p = xlionrender::Env("XLION_SSR_STEPS"); return p ? std::atoi(p) : 0; }(); return s > 0 ? s : m_SSRSteps; };      // (measurement: the screen trace's steps)
                const struct { std::uint32_t m_Slot, m_bKeepTraced, m_Frame, m_Steps; float m_MaxDist, m_Thickness, m_MaxRough, m_Pad; } SPush
                    { std::uint32_t(Slot), 0u, bReconstruct || Scene.m_bReflectionJitter ? View.m_TailFrame : 0u, std::uint32_t(std::max(SSRSteps(), 1)), m_SSRMaxDist, m_SSRThickness, ssr_max_rough_v, 0.0f };
                auto SBuffers  = std::array<xgpu::buffer*, 2>{ &m_ReflFrames, &View.m_TailResult };
                auto STextures = std::array<const xgpu::texture*, 3>{ &View.m_TailGColor, &View.m_TailGDepth, View.m_bHDRPrev ? &View.m_HDRPrev : &View.m_HDR };      // (m_HDRPrev: m_HDR itself is cleared by the prepass)
                Window.WriteTimestamp(m_ReflSSRTimes, 0);
                Window.Dispatch(m_ReflSSR, SBuffers, STextures, &SPush, sizeof(SPush), (W + 7) / 8, (H + 7) / 8);
                Light.m_EnvParams.m_W = std::max(Light.m_EnvParams.m_W, ssr_max_rough_v);      // (the materials read the records up to the screen trace's roughness: 1e-3 here read none - EnvLighting fades them out by .7 w)
            }
        }
        if( bTraced )
        {
            if( !bLights )      // (the denoiser writes this slot's frame when it runs)
            {
                const xmath::fvec4 Params{ float(W), float(H), 0.0f, float(View.m_TailFrame & 1023u) };
                const xmath::fmat4 Identity = xmath::fmat4::fromIdentity();
                (void)m_TailFrames.MemoryMapFrameRegion(Slot, 1, [&](void* p)
                {
                    auto* pB = static_cast<std::byte*>(p);
                    std::memcpy(pB, &C2W, 64); std::memcpy(pB + 64, &Identity, 64); std::memcpy(pB + 128, &Params, 16);
                });
            }
            std::uint32_t Got[8] = {};
            (void)m_ReflStats.MemoryMap(Slot * 8, 8, [&](void* p) { std::memcpy(Got, p, sizeof(Got)); std::memset(p, 0, sizeof(Got)); });
            if( Got[0] ) { m_bReflSharpSeen = (Got[7] >> 31) != 0u; Got[7] &= 0x7FFFFFFFu; }
            if( std::vector<std::uint64_t> Ns; Got[0] && m_ReflTimes.getResults(Ns) && Ns.size() >= 5 && Ns[0] != ~0ull && Ns[4] > Ns[0] )
            {
                m_ReflSum[0] += double(Ns[4] - Ns[0]) * 1e-6;
                m_ReflSum[1] += Got[0]; m_ReflSum[2] += Got[1]; m_ReflSum[3] += Got[2]; m_ReflPlanarSum += Got[3]; m_ReflIncompleteSum += Got[5]; if( Got[3] ) m_ReflPlanarRough = double(Got[4]) * 0.01 / double(Got[3]);
                for( int k = 0; k < 4; ++k ) m_ReflSplit[k] += Ns[k + 1] >= Ns[k] ? double(Ns[k + 1] - Ns[k]) * 1e-6 : 0.0;
                m_ReflTextured += Got[6]; m_ReflTexturedFull += Got[7];
                if( Ns.size() >= 7 && Ns[5] >= Ns[2] && Ns[6] >= Ns[5] && Ns[3] >= Ns[6] ) { m_ReflHitSplit[0] += double(Ns[5] - Ns[2]) * 1e-6; m_ReflHitSplit[1] += double(Ns[6] - Ns[5]) * 1e-6; m_ReflHitSplit[2] += double(Ns[3] - Ns[6]) * 1e-6; }
                if( ++m_ReflSumN == 120 )
                {
                    const double n = m_ReflSumN;
                    m_ReflReport = { m_ReflTraceBench, m_ReflSum[0] / n, m_ReflSum[1] / n, m_ReflSum[2] / std::max(m_ReflSum[1], 1.0), m_ReflSum[3] / std::max(m_ReflSum[1], 1.0), m_ReflSumN };
                    static const bool s_bTimes = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                    if( m_ReflTraceBench > 0 || s_bTimes ) std::printf("REFLBENCH mode %d, %d x %d: %.3f ms, %.0f rays (%.0f Mrays/s), hits %.1f%%, BVH visits %.1f a ray, %.0f planar texels of roughness %.2f, %.0f rays PREFIX_INCOMPLETE (planar view %s, plane %.3f %.3f %.3f %.3f)\n", m_ReflReport.m_Mode, W, H, m_ReflReport.m_Ms
                        , m_ReflReport.m_Rays, m_ReflReport.m_Rays / std::max(m_ReflReport.m_Ms, 1e-6) * 1e-3, 100 * m_ReflReport.m_HitRate, m_ReflReport.m_VisitsPerRay, m_ReflPlanarSum / n, m_ReflPlanarRough, m_ReflIncompleteSum / n
                        , m_bPlanarOn ? "on" : m_PlanarWhy == 4 ? "off: no floor under the eye" : m_PlanarWhy == 5 ? "off: oblique" : m_PlanarWhy == 2 ? "off: view" : "off", m_PlanarPlane.m_X, m_PlanarPlane.m_Y, m_PlanarPlane.m_Z, m_PlanarPlane.m_W);
                    if( m_ReflTraceBench > 0 || s_bTimes ) std::printf("REFLSPLIT sharp trace %.3f, trace %.3f, hits' shading %.3f, reconstruction %.3f ms; %.0f hits textured a frame (%.0f at full size; bins %.3f, draws %.3f, relight %.3f)\n", m_ReflSplit[0] / n, m_ReflSplit[1] / n, m_ReflSplit[2] / n, m_ReflSplit[3] / n, m_ReflTextured / n + m_ReflTexturedFull / n, m_ReflTexturedFull / n, m_ReflHitSplit[0] / n, m_ReflHitSplit[1] / n, m_ReflHitSplit[2] / n);
                    std::fflush(stdout);
                    m_ReflSumN = 0; m_ReflPlanarSum = 0; m_ReflIncompleteSum = 0; for( auto& v : m_ReflSum ) v = 0; for( auto& v : m_ReflSplit ) v = 0; m_ReflTextured = 0; m_ReflTexturedFull = 0; for( auto& v : m_ReflHitSplit ) v = 0;
                    if( m_bReflTraceCycle ) m_ReflTraceBench = m_ReflTraceBench % 3 + 1;      // (XLION_REFL_BENCH=9: the three modes in turn)
                }
            }
            const xmath::fvec3 E = Eye;
            const float MaxRough = m_ReflTraceBench >= 4 ? 1.0f : m_ReflTraceBench > 0 ? 0.0f : Scene.m_ReflectionMaxRoughness;    // (4: its records shown)
            const bool  bPlanar  = m_bPlanarOn && m_PlanarView.m_Width == W && m_PlanarView.m_Height == H;
            // the boxes of the drawn things the GI's BVH does not hold (skinned, without GI; world space): a ray's prefix crossing one is
            // PREFIX_INCOMPLETE (documentation/reflections_strategy.md sec. 4: a label - its confidence, not its light)
            {
                std::vector<xmath::fvec4> Boxes;
                for( const auto& D : Scene.m_Draws )
                {
                    if( int(Boxes.size()) >= 2 * refl_boxes_v ) break;
                    if( D.m_Instance != 0 && getGIId(D.m_Instance) >= 0 ) continue;
                    const auto* pM = findMesh(D.m_Mesh);
                    if( pM == nullptr ) continue;
                    const auto& B = pM->m_bCluster ? pM->m_Geom.m_BBox : pM->m_BBox;
                    xmath::fvec3 Lo{ 1e30f, 1e30f, 1e30f }, Hi{ -1e30f, -1e30f, -1e30f };
                    for( int c = 0; c < 8; ++c )
                    {
                        const xmath::fvec3 P = D.m_L2W * xmath::fvec3{ (c & 1) ? B.m_Max.m_X : B.m_Min.m_X, (c & 2) ? B.m_Max.m_Y : B.m_Min.m_Y, (c & 4) ? B.m_Max.m_Z : B.m_Min.m_Z };
                        Lo = xmath::fvec3::Min(Lo, P); Hi = xmath::fvec3::Max(Hi, P);
                    }
                    Boxes.push_back(xmath::fvec4(Lo, 0)); Boxes.push_back(xmath::fvec4(Hi, 0));
                }
                (void)m_PlanarParams.MemoryMapFrameRegion(Slot, 1, [&](void* p)
                {
                    auto* pB = static_cast<std::byte*>(p) + 64 + 16 + 8 * 16;
                    const std::uint32_t n[4] = { std::uint32_t(Boxes.size() / 2), 0, 0, 0 };
                    std::memcpy(pB, n, 16);
                    if( !Boxes.empty() ) std::memcpy(pB + 16, Boxes.data(), Boxes.size() * 16);
                });
            }
            const auto& Pl = m_PlanarPlane;
            // the hits to texture: the trace appends them to m_HitList (their keys: the half size's texels, then the full size's sharp ones)
            const std::uint32_t HalfKeys = std::uint32_t(W * H);
            const bool bHits = m_ReflTraceBench == 0 && Scene.m_ReflectionTextureMaxRoughness > 0 && createHitTargets() == nullptr;
            // (the voxels on: they are the hits' indirect light; the surfels hold nothing then)
            static const bool s_bReflNoVox = xlionrender::Env("XLION_REFL_NO_VOX") != nullptr;      // (measurement: the reflections without the voxels' light at their hits - compute_overlap_plan_2026-11-07.md option b)
            xgpu::buffer* pVox = m_bVoxOn && m_pGITrace && !s_bReflNoVox ? m_pGITrace->VoxBuffer() : nullptr;
            const std::array<float, 3> VoxAt = pVox ? m_pGITrace->VoxAnchor() : std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
            struct { std::uint32_t m_Mode, m_Slot; int m_TlasRoot, m_PrimOffset; float m_Eye[3]; std::uint32_t m_Frame; float m_EnvIntensity, m_MaxRough, m_bPlanar, m_SharpMax; float m_Plane[4]; int m_Full[4]; float m_Tex[4]; float m_Vox[4]; float m_SharpSeen, m_FlatR, m_FlatG, m_FlatB; } Push
                { std::uint32_t(m_ReflTraceBench), std::uint32_t(Slot), BvhR.m_TlasRoot, BvhR.m_PrimOffset, { E.m_X, E.m_Y, E.m_Z }, Scene.m_bReflectionJitter || Scene.m_bReflectionReconstruct ? View.m_TailFrame : 0u, m_bEnvOn ? m_EnvIntensity : 0.0f, MaxRough, bPlanar ? 1.0f : 0.0f, bSharp ? m_ReflSharpMax : 0.0f
                , { bPlanar ? Pl.m_X : 0.0f, bPlanar ? Pl.m_Y : 0.0f, bPlanar ? Pl.m_Z : 0.0f, bPlanar ? Pl.m_W : 0.0f }, { 0, 0, bScreen ? 3 : 0, s_bGradeRays ? (Scene.m_bReflectionFixedRays ? 3000 : 1000) + s_GradeRough : Scene.m_bDebugReflectionMirror ? 1 : Scene.m_bReflectionFixedRays ? 2 : 0 }
                , { bHits ? Scene.m_ReflectionTextureMaxRoughness : 0.0f, Scene.m_ReflectionTextureMinReflectance, float((View.m_TailFrame & 1023u) & 1u), float(hit_cap_v) }, { VoxAt[0], VoxAt[1], VoxAt[2], pVox ? gi::gpu_trace::VoxConfig()[0] : 0.0f }, m_ReflSharpMax
                , m_bEnvOn ? 0.0f : Scene.m_Ambient.m_X / gi::pi_v, m_bEnvOn ? 0.0f : Scene.m_Ambient.m_Y / gi::pi_v, m_bEnvOn ? 0.0f : Scene.m_Ambient.m_Z / gi::pi_v };
            const auto Bc = [](const xgpu::buffer* p) { return const_cast<xgpu::buffer*>(p); };
            // (XLION_SDF, prototype, reflections_sdf_plan_2026-11.md's gate: 1 hit or miss by the world's distance field, built here over some frames; 2 by the BVH)
            static const int s_Sdf = [] { const char* p = xlionrender::Env("XLION_SDF"); return p ? std::atoi(p) : 0; }();
            const bool bSdf = s_Sdf == 1 || (s_Sdf >= 3 && s_Sdf < 11);
            if( bSdf && (E - m_WorldSdfCenter).Length() > 3.0f ) { m_WorldSdfCascade = 0; m_WorldSdfSlice = 0; m_WorldSdfAge = 0; }      // ponytail: the whole field again when the eye leaves; scrolling cascades when it moves in a game
            if( bSdf && ++m_WorldSdfAge == 300 ) { m_WorldSdfCascade = 0; m_WorldSdfSlice = 0; }      // ponytail: and once more when the materials' capture is in: its albedo was 0 before
            if( bSdf && m_WorldSdfCascade < 4 )
            {
                if( m_WorldSdfCascade == 0 && m_WorldSdfSlice == 0 ) m_WorldSdfCenter = E;
                struct { float m_Center[3], m_Cell0; int m_Cascade, m_Slice, m_Mode, m_TlasRoot, m_PrimOffset; float m_MaxCells; } SPush{ { m_WorldSdfCenter.m_X, m_WorldSdfCenter.m_Y, m_WorldSdfCenter.m_Z }, 0.1f, m_WorldSdfCascade, m_WorldSdfSlice, 0, BvhR.m_TlasRoot, BvhR.m_PrimOffset, 8.0f };
                auto SB = std::array<xgpu::buffer*, 7>{ const_cast<xgpu::buffer*>(BvhR.m_pNodes), const_cast<xgpu::buffer*>(BvhR.m_pTri4), const_cast<xgpu::buffer*>(BvhR.m_pInstances), const_cast<xgpu::buffer*>(BvhR.m_pTlas), const_cast<xgpu::buffer*>(BvhR.m_pData), &m_WorldSdfB, &m_WorldSdfSurfB };
                Window.Dispatch(m_WorldSdf, SB, &SPush, sizeof(SPush), 16, 16, 16);
                if( (m_WorldSdfSlice += 16) >= 128 )
                {
                    SPush.m_Slice = 0; SPush.m_Mode = 1;
                    for( int f = 0; f < 3; ++f ) Window.Dispatch(m_WorldSdf, SB, &SPush, sizeof(SPush), 16, 16, 128);
                    m_WorldSdfSlice = 0; ++m_WorldSdfCascade;
                }
            }
            Push.m_Full[1] = s_Sdf;
            float MaxRoughOut = 0.0f;
            // XLION_REFL_PT (A12): the reflections' ground truth with the real materials (refl_trace_comp.glsl m_Full.w 5): no planar view, no store, no screen trace
            if( s_bReflPT && !m_ReflPTAccum.m_Private ) { const std::vector<float> Z(8 * std::size_t(W) * std::size_t(H), 0.0f); (void)m_Device.Create(m_ReflPTAccum, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = 2 * W * H, .m_pData = const_cast<float*>(Z.data()), .m_pName = "Reflections: the ground truth's sums" }); }
            static const bool s_bHitPT = xlionrender::Env("XLION_REFL_HIT_PT") != nullptr; if( s_bHitPT && Push.m_Full[3] == 0 ) Push.m_Full[3] = 6;      // (measurement: the hits lit by a path, not the store)
            if( s_bReflPT ) { Push.m_Full[3] = 5; Push.m_Full[2] = 0; for( auto& v : Push.m_Plane ) v = 0.0f; Push.m_bPlanar = 0.0f; Push.m_Frame = View.m_TailFrame; }
            static const bool s_bBands = xlionrender::Env("XLION_REFL_BANDS") != nullptr; if( s_bBands && !bPlanar ) Push.m_Plane[0] = -1.0f;
            static const std::array<float, 4> s_DirsTrace = [] { std::array<float, 4> D{ 0.0f, 3.0f, 0.15f, 0.8f }; float K = 8.0f; const char* t = xlionrender::Env("XLION_VOX_DIRS_TRACE"); const char* d = xlionrender::Env("XLION_VOX_DIRS"); if( !t || !d ) return std::array<float, 4>{}; xlionrender::Scan(d, "%f,%f", &D[1], &K); xlionrender::Scan(t, "%f,%f", &D[2], &D[3]); if( D[1] < 0.0f ) return std::array<float, 4>{}; D[0] = K; return D; }();      // (rays below D[2], the store's read to D[3]: the shader)
            if( pVox && s_DirsTrace[0] > 0.0f && !s_bReflPT ) { if( !bPlanar ) std::memcpy(Push.m_Plane, s_DirsTrace.data(), sizeof(Push.m_Plane)); MaxRoughOut = std::max(MaxRoughOut, s_DirsTrace[3]); }
            auto Buffers  = std::array<xgpu::buffer*, 16>{ Bc(BvhR.m_pNodes), Bc(BvhR.m_pTri4), Bc(BvhR.m_pInstances), Bc(BvhR.m_pTlas), Bc(BvhR.m_pData), &m_TailFrames, &m_ReflStats, &View.m_TailResult, &m_PlanarMipsBuffer, &m_PlanarParams, bHits ? &m_HitList : &m_ReflStats, pVox ? pVox : &m_ReflStats, bSdf ? &m_WorldSdfB : &m_ReflStats, bSdf ? &m_WorldSdfSurfB : &m_ReflStats, m_bBakedSdf ? &m_BakedSdfB : &m_ReflStats, s_bReflPT && m_ReflPTAccum.m_Private ? &m_ReflPTAccum : &m_ReflSharpList };
            auto Textures = std::array<const xgpu::texture*, 6>{ &View.m_TailGColor, &View.m_TailGDepth, &m_EnvAtlas, bPlanar ? &m_PlanarView.m_HDR : bSdf ? &View.m_HDR : &View.m_TailGColor, bPlanar ? &m_PlanarView.m_Depth : &View.m_TailGDepth, &View.m_TailGMotion };
            // the sharp receivers at full size first (their records after the half-size ones, 2 pixels a record): mirrors take the hits
            // list's places before the glossy ones
            Window.WriteTimestamp(m_ReflTimes, 0);
            if( bSharp )
            {
                auto P2 = Push;
                P2.m_Full[0] = int(View.m_SharpWidth | (View.m_SharpHeight << 16)); P2.m_Full[2] = 2 * W * H; P2.m_SharpMax = m_ReflSharpMax;
                P2.m_Frame = View.m_TailFrame;     // (the sharp ones are not reconstructed: a new ray each frame, TAA averages them; one fixed ray a pixel was static sparkle on Bistro's glass. Same cost, noise over 8 .020 to .028 %)
                auto T2 = std::array<const xgpu::texture*, 6>{ &View.m_SharpGColor, &View.m_SharpGDepth, &m_EnvAtlas, Textures[3], Textures[4], &View.m_SharpGMotion };
                P2.m_Frame = (P2.m_Frame & ~1u) | (m_ReflSharpRuns++ & 1u);      // (the list's parity: alternates each run, gaps or not)
                static const bool s_bSharpSplit = xlionrender::Env("XLION_GPU_TIMES") != nullptr;      // (measurement: the sharp pass's classification and its list's rays apart - SHARPSPLIT)
                static xgpu::timestamps s_SharpTS; static bool s_bSharpTS = false;
                if( s_bSharpSplit && !s_bSharpTS ) s_bSharpTS = m_Device.Create(s_SharpTS, { .m_Count = 3 }) == nullptr;
                if( s_bSharpSplit && s_bSharpTS )
                {
                    static std::vector<std::uint64_t> s_Ns; static double s_A = 0, s_B = 0; static int s_N = 0;
                    if( s_SharpTS.getResults(s_Ns) && s_Ns.size() >= 3 && s_Ns[2] >= s_Ns[1] && s_Ns[1] >= s_Ns[0] ) { s_A += (s_Ns[1] - s_Ns[0]) * 1e-6; s_B += (s_Ns[2] - s_Ns[1]) * 1e-6;
                        if( ++s_N == 120 ) { std::printf("SHARPSPLIT classification (full size) %.3f, the list's rays %.3f ms\n", s_A / s_N, s_B / s_N); std::fflush(stdout); s_A = s_B = 0; s_N = 0; } }
                    Window.WriteTimestamp(s_SharpTS, 0);
                }
                Window.Dispatch(m_ReflTrace, Buffers, T2, &P2, sizeof(P2), (View.m_SharpWidth + 7) / 8, (View.m_SharpHeight + 7) / 8);
                if( s_bSharpSplit && s_bSharpTS ) Window.WriteTimestamp(s_SharpTS, 1);
                P2.m_Full[0] = -P2.m_Full[0];      // (the list: 64 a group, every lane busy)
                Window.Dispatch(m_ReflTrace, Buffers, T2, &P2, sizeof(P2), 262144 / 64, 1);
                if( s_bSharpSplit && s_bSharpTS ) Window.WriteTimestamp(s_SharpTS, 2);
            }
            Window.WriteTimestamp(m_ReflTimes, 1);
            Window.Dispatch(m_ReflTrace, Buffers, Textures, &Push, sizeof(Push), (W + 7) / 8, (H + 7) / 8);
            Window.WriteTimestamp(m_ReflTimes, 2);
            if( bHits ) shadeReflectionHits(Window, Scene, View, BvhR, Slot, HalfKeys, std::uint32_t(2 * W * H));
            Window.WriteTimestamp(m_ReflTimes, 3);
            Light.m_EnvParams.m_W = std::max(Light.m_EnvParams.m_W, std::max(MaxRough, MaxRoughOut));
            static const float s_GradeCut = [] { const char* p = xlionrender::Env("XLION_REFL_GRADE_CUT"); return p ? float(std::atof(p)) : 0.0f; }();      /* (grading: the mirror view asks rays of everything; the cutoff their fade follows) */
            if( s_GradeCut > 0.0f ) Light.m_EnvParams.m_W = s_GradeCut;
        }
        if( bTraced || bScreen )
        {
            // the reconstruction (milestone D: xlionrender_refl_denoise_comp.glsl): temporal by the reflection's virtual motion, then spatial;
            // the half-size records it leaves are what the materials read (REFLSPLIT's reconstruction includes the screen trace when both run)
            if( bReconstruct )
            {
                auto DBuffers  = std::array<xgpu::buffer*, 4>{ &m_ReflFrames, &View.m_TailResult, &View.m_ReflTemporal, &View.m_ReflHistory };
                auto DTextures = std::array<const xgpu::texture*, 3>{ &View.m_TailGColor, &View.m_TailGDepth, &View.m_TailGMotion };
                for( std::uint32_t Pass = 0; Pass < 2; ++Pass )
                {
                    static const bool s_bDiskFresh = xlionrender::Env("XLION_REFL_DISK_FRESH") != nullptr;      // (measurement: the spatial disk only for fresh histories)
                    const std::uint32_t DPush[4] = { Pass, std::uint32_t(Slot), s_bDiskFresh ? 1u : 0u, 0 };
                    Window.Dispatch(m_ReflDenoise, DBuffers, DTextures, DPush, sizeof(DPush), (W + 7) / 8, (H + 7) / 8);
                }
            }
            if( bScreen ) Window.WriteTimestamp(m_ReflSSRTimes, 1);     // (SSRGPU: the screen trace and the reconstruction)
            if( bTraced ) Window.WriteTimestamp(m_ReflTimes, 4);
        }
        View.m_TailPrevC2W = C2W; View.m_TailPrevW2C = W2C; View.m_TailPrevEye = Eye; View.m_bTailPrev = true; ++View.m_TailFrame;
        m_pTailRecords = &View.m_TailResult;
    }

    // Reflections, the hits' textures (shaders/xlionrender_refl_hits_comp.glsl): the bins - per GI instance drawn this frame, per its
    // material slot: its (material, draw slot) - and the bins counted the frame this ring slot last ran (their hits shaded now: a bin
    // first hit shows its capture's albedo a frame), then count, scan and scatter on the GPU, each drawn bin's hits through its
    // material's capture variant into the shade target (albedo, emission), and the relight into the records
    void instance::shadeReflectionHits( xgpu::window& Window, const scene& Scene, render_view& View, const gi::gpu_trace::bvh_view& BvhR, int Slot, std::uint32_t HalfKeys, std::uint32_t FullBase ) noexcept
    {
        // this frame's bins
        std::unordered_map<std::uint64_t, std::uint32_t> BinOf;
        const auto KeyOf = [](material_id M, std::uint32_t DrawSlot) { std::uint64_t k = 0; std::memcpy(&k, &M, std::min(sizeof(M), sizeof(k))); return (k << 20) ^ DrawSlot; };
        m_HitBinKeys.clear();
        std::vector<std::uint32_t> T(m_GITrack.size(), ~0u);
        for( std::size_t Id = 0; Id < m_GITrack.size(); ++Id )
        {
            const auto& G = m_GITrack[Id];
            if( G.m_Render != m_GIRender || G.m_Materials.empty() ) continue;
            T[Id] = std::uint32_t(T.size());
            for( const auto& M : G.m_Materials )
            {
                const auto [It, bNew] = BinOf.try_emplace(KeyOf(M, G.m_DrawSlot), std::uint32_t(m_HitBinKeys.size()));
                if( bNew )
                {
                    if( m_HitBinKeys.size() >= hit_max_bins_v ) { BinOf.erase(It); T.push_back(~0u); continue; }
                    m_HitBinKeys.push_back({ M, G.m_DrawSlot });
                }
                T.push_back(It->second);
            }
        }
        const std::uint32_t nBins = std::uint32_t(m_HitBinKeys.size());
        if( nBins == 0 ) return;
        // the bins this ring slot counted (its frame is done) that are bins again now: drawn, if their material has a capture pipeline
        const std::uint32_t DrawnAt = std::uint32_t(T.size());
        T.resize(T.size() + (hit_max_bins_v + 31) / 32, 0u);
        std::vector<std::uint32_t> Drawn;
        {
            if( m_HitBinKeysAt.size() < ring_frames_v ) m_HitBinKeysAt.resize(ring_frames_v);
            const auto& Was = m_HitBinKeysAt[std::size_t(Slot)];
            (void)m_HitCounts.MemoryMap(std::size_t(Slot) * hit_max_bins_v, hit_max_bins_v, [&](void* p)
            {
                const auto* C = static_cast<const std::uint32_t*>(p);
                for( std::size_t b = 0; b < Was.size(); ++b )
                {
                    if( C[b] == 0 ) continue;
                    const auto It = BinOf.find(KeyOf(Was[b].first, Was[b].second));
                    if( It == BinOf.end() ) continue;
                    auto* pMat = findMaterial(m_HitBinKeys[It->second].first);
                    if( pMat == nullptr || getCaptureInstance(*pMat) == nullptr ) continue;
                    T[DrawnAt + It->second / 32] |= 1u << (It->second % 32);
                    Drawn.push_back(It->second);
                }
            });
            m_HitBinKeysAt[std::size_t(Slot)] = m_HitBinKeys;
        }
        // the tables, in this ring slot's region
        if( m_HitTableCap < T.size() )
        {
            if( m_HitTables.m_Private ) { xgpu::buffer Old; Old.m_Private.swap(m_HitTables.m_Private); }     // (its destructor queues the deferred destroy)
            m_HitTableCap = std::max<std::uint32_t>(std::uint32_t(T.size() * 3 / 2), 1u << 16);
            if( m_Device.Create(m_HitTables, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = int(4 * m_HitTableCap), .m_EntryCount = ring_frames_v }) ) { m_HitTableCap = 0; return; }
        }
        (void)m_HitTables.MemoryMapFrameRegion(Slot, 1, [&](void* p) { std::memcpy(p, T.data(), T.size() * 4); });

        const auto Bc = [](const xgpu::buffer* p) { return const_cast<xgpu::buffer*>(p); };
        auto Buffers  = std::array<xgpu::buffer*, 12>{ Bc(BvhR.m_pNodes), Bc(BvhR.m_pTri4), Bc(BvhR.m_pInstances), Bc(BvhR.m_pTlas), Bc(BvhR.m_pData), &m_HitList, &m_HitTables, &m_HitBins, &m_HitCounts, &m_HitDraws, &m_HitCapture, &View.m_TailResult };
        auto Textures = std::array<const xgpu::texture*, 1>{ &m_HitShadeTex };
        struct { std::uint32_t m_Pass, m_Keys, m_Half, m_FullBase, m_Bins, m_Cap, m_Drawn, m_Slot, m_Width, m_TableBase; float m_TexMax; std::uint32_t m_Parity, m_Debug; } Push
            { 0, hit_cap_v, HalfKeys, FullBase, nBins, hit_cap_v, DrawnAt, std::uint32_t(Slot), hit_width_v, std::uint32_t(Slot) * m_HitTableCap, Scene.m_ReflectionTextureMinReflectance, (View.m_TailFrame & 1023u) & 1u, xlionrender::Env("XLION_REFL_HITS_ALBEDO") ? std::uint32_t(std::atoi(xlionrender::Env("XLION_REFL_HITS_ALBEDO"))) : 0u };
        for( std::uint32_t Pass = 0; Pass < 3; ++Pass )
        {
            Push.m_Pass = Pass;
            Window.Dispatch(m_ReflHits, Buffers, Textures, &Push, sizeof(Push), Pass == 1 ? 1 : int((hit_cap_v + 255) / 256));
        }
        Window.WriteTimestamp(m_ReflTimes, 5);
        // the drawn bins' hits, a pixel each, through their materials' capture variants (xlionrender_gi_capture_vert.glsl reads m_HitCapture)
        if( !Drawn.empty() )
        {
            // (the UBOs as they are: the main pass's entries - the capture variant reads neither; a new entry here would be the one the
            // main pass binds, its lighting zeroed)
            auto  Cmd = Window.StartRenderPass(m_HitShadePass);
            std::array<xgpu::buffer*, 10> CBuffers{ &m_HitCapture, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, TailBuffer() };
            for( const std::uint32_t b : Drawn )
            {
                auto* pMat  = findMaterial(m_HitBinKeys[b].first);
                auto* pInst = getCaptureInstance(*pMat);
                Cmd.setPipelineInstance(*pInst, std::span{ CBuffers });
                if( pMat->m_bTable ) Cmd.setTextureTable(pMat->m_Table);
                Cmd.setDynamicUBO(m_LightUBO, 1);
                Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                Cmd.setBuffer(m_HitVB);
                Cmd.setBuffer(m_HitIB);
                for( std::uint32_t Pass = 0; Pass < 2; ++Pass )      // (albedo x (1 - metal), emission: xlionrender_surfel_gi.glsl's capture flags)
                {
                    Cmd.setPushConstants(details::push_static_geom{ 0u, m_HitBinKeys[b].second, (hit_width_v << 16) | (hit_cap_v / hit_width_v), (3u << 9) | (Pass ? 0x800u : 0u), pMat->m_ParamBase });
                    Cmd.DrawIndexedIndirect(m_HitDraws, int(Pass * nBins + b), 1);
                }
            }
        }
        Window.WriteTimestamp(m_ReflTimes, 6);
        Push.m_Pass = 3;
        Window.Dispatch(m_ReflHits, Buffers, Textures, &Push, sizeof(Push), int((hit_cap_v + 255) / 256));
    }

    // Reflections, the planar view (documentation/reflections_strategy.md sec. 8): the scene mirrored in scene::m_ReflectionPlane, at the
    // screen tail's size - the reflection pass resolves its records from it 1:1 (xlionrender_refl_trace_comp.glsl). The main projection
    // times the plane's reflection, x flipped (two reflections: the triangles keep their winding; a point of the plane lands on the
    // mirrored texel), its depth row oblique (Lengyel 2005): the near plane is the mirror plane, nothing behind it is drawn. A pass as a
    // probe's face (bMain false: no screen tail, no reflections of its own - no feedback)
    void instance::renderPlanar( xgpu::window& Window, const scene& Scene, const render_view& Main, const xgpu::tools::view& Camera ) noexcept
    {
        m_bPlanarOn = false; m_PlanarWhy = 1;
        const auto& Pl = Scene.m_ReflectionPlane;
        const float Len = std::sqrt(Pl.m_X * Pl.m_X + Pl.m_Y * Pl.m_Y + Pl.m_Z * Pl.m_Z);
        if( Scene.m_ReflectionMaxRoughness <= 0 || Len < 1e-6f ) return;
        const int W = (Main.m_Width + 1) / 2, H = (Main.m_Height + 1) / 2;
        if( m_PlanarView.m_Width != W || m_PlanarView.m_Height != H )
        {
            if( m_PlanarView.m_Width ) DestroyView(m_PlanarView);
            if( CreateView(m_PlanarView, { .m_Width = W, .m_Height = H }) ) { m_PlanarView = {}; m_PlanarWhy = 2; return; }
        }
        const xmath::fvec3 n{ Pl.m_X / Len, Pl.m_Y / Len, Pl.m_Z / Len };
        float              d = Pl.m_W / Len;
        if( std::isnan(d) )         // the floor under the eye: the GI's geometry straight down, a surface facing n
        {
            if( !m_pGIWorld ) { m_PlanarWhy = 3; return; }
            // (straight down from where the view's centre ray lands - the floor of what is looked at -, else from the eye)
            const auto C = m_pGIWorld->Cast(Camera.getPosition(), Camera.getForward(), 1000.0f);
            const xmath::fvec3 From = C.m_Inst >= 0 ? Camera.getPosition() + Camera.getForward() * (C.m_T - 0.05f) : Camera.getPosition();
            auto H = m_pGIWorld->Cast(From, n * -1.0f, 1000.0f);
            if( (H.m_Inst < 0 || m_pGIWorld->HitNormal(H).Dot(n) < 0.95f) && C.m_Inst >= 0 ) H = m_pGIWorld->Cast(Camera.getPosition(), n * -1.0f, 1000.0f);
            if( H.m_Inst < 0 || m_pGIWorld->HitNormal(H).Dot(n) < 0.95f ) { m_PlanarWhy = 4; return; }
            d = -n.Dot(m_pGIWorld->HitPosition(H));
        }
        m_PlanarPlane = { n.m_X, n.m_Y, n.m_Z, d };
        xmath::fmat4 Reflect = xmath::fmat4::fromIdentity();
        for( int r = 0; r < 3; ++r )
        {
            for( int c = 0; c < 3; ++c ) Reflect(r, c) = (r == c ? 1.0f : 0.0f) - 2.0f * n[r] * n[c];
            Reflect(r, 3) = -2.0f * d * n[r];
        }
        const xmath::fvec3 Eye  = Camera.getPosition();
        const xmath::fvec3 EyeM = Eye - n * (2.0f * (n.Dot(Eye) + d));
        xgpu::tools::view Cam = Camera;
        Cam.setViewport({ 0, 0, W, H });
        xmath::fmat4 M = Cam.getW2C() * Reflect;
        {   // the oblique near plane: the depth row a multiple of the plane, scaled so the far plane still passes the frustum's far corner
            const xmath::fmat4 Inv = M.Inverse();
            const xmath::fvec4 C{ n.m_X, n.m_Y, n.m_Z, d };
            xmath::fvec4 Cc;
            for( int j = 0; j < 4; ++j ) Cc[j] = C[0] * Inv(0, j) + C[1] * Inv(1, j) + C[2] * Inv(2, j) + C[3] * Inv(3, j);
            const xmath::fvec4 q = Inv * xmath::fvec4{ Cc[0] >= 0 ? 1.0f : -1.0f, Cc[1] >= 0 ? 1.0f : -1.0f, 1.0f, 1.0f };
            const float Wq = M(3, 0) * q[0] + M(3, 1) * q[1] + M(3, 2) * q[2] + M(3, 3) * q[3];
            const float Cq = C[0] * q[0] + C[1] * q[1] + C[2] * q[2] + C[3] * q[3];
            if( std::abs(Cq) < 1e-12f ) { m_PlanarWhy = 5; return; }
            for( int c = 0; c < 4; ++c ) M(2, c) = C[c] * (Wq / Cq);
        }
        for( int c = 0; c < 4; ++c ) M(0, c) = -M(0, c);
        const xmath::fmat4 W2CM = M * xmath::fmat4::fromTranslation(EyeM);
        renderScenePass(Window, Scene, m_PlanarView, W2CM, EyeM, false);

        // its pyramid (rough receivers: xlionrender_refl_planar_mips_comp.glsl) and its parameters for the reflection pass (this ring slot)
        std::array<std::uint32_t, 8 * 4> Levels{};
        std::uint32_t Base = 0;
        int Lw = W, Lh = H, nLevels = 0;
        while( nLevels < 7 && (Lw > 1 || Lh > 1) )
        {
            Lw = std::max(1, (Lw + 1) / 2); Lh = std::max(1, (Lh + 1) / 2); ++nLevels;
            Levels[4 * nLevels] = Base; Levels[4 * nLevels + 1] = std::uint32_t(Lw); Levels[4 * nLevels + 2] = std::uint32_t(Lh);
            Base += std::uint32_t(Lw * Lh);
        }
        Levels[1] = std::uint32_t(W); Levels[2] = std::uint32_t(H);
        if( m_PlanarLevels != nLevels || m_PlanarMipsBuffer.getEntryCount() < int(Base) )
        {
            m_Device.Destroy(std::move(m_PlanarMipsBuffer));
            const std::vector<std::uint64_t> Zeros(std::max<std::uint32_t>(Base, 1), 0);      // (a device-local buffer is made from its first data: without it the handle stayed the staging buffer, bound as storage - a validation error each bind)
            if( m_Device.Create(m_PlanarMipsBuffer, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 8, .m_EntryCount = int(Zeros.size()), .m_pData = const_cast<std::uint64_t*>(Zeros.data()) }) ) { m_PlanarLevels = 0; return; }
            m_PlanarLevels = nLevels;
        }
        for( int L = 1; L <= nLevels; ++L )
        {
            const std::uint32_t Push[8] = { L == 1 ? std::uint32_t(W) : Levels[4 * (L - 1) + 1], L == 1 ? std::uint32_t(H) : Levels[4 * (L - 1) + 2], Levels[4 * L + 1], Levels[4 * L + 2]
                                          , L == 1 ? 0u : Levels[4 * (L - 1)], Levels[4 * L], L == 1 ? 1u : 0u, 0u };
            auto Buffers  = std::array<xgpu::buffer*, 1>{ &m_PlanarMipsBuffer };
            auto Textures = std::array<const xgpu::texture*, 1>{ &m_PlanarView.m_HDR };
            Window.Dispatch(m_PlanarMips, Buffers, Textures, Push, sizeof(Push), (int(Levels[4 * L + 1]) + 7) / 8, (int(Levels[4 * L + 2]) + 7) / 8);
        }
        const float ProjScale = float(H) / (2.0f * xmath::Tan(xmath::radian{ Camera.getFov().m_Value * 0.5f }));
        const xmath::fmat4 C2WM = W2CM.Inverse();
        const int Slot = int(m_Device.getFrameIndex() % ring_frames_v);
        (void)m_PlanarParams.MemoryMapFrameRegion(Slot, 1, [&](void* p)
        {
            auto* pB = static_cast<std::byte*>(p);
            const float Info[4] = { ProjScale, float(nLevels), 0, 0 };
            std::memcpy(pB, &C2WM, 64); std::memcpy(pB + 64, Info, 16); std::memcpy(pB + 80, Levels.data(), sizeof(Levels));
        });
        m_bPlanarOn = true; m_PlanarWhy = 0;
    }

    void instance::renderScenePass( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye, bool bMain ) noexcept
    {
        m_pTailRecords = nullptr;
        // Measurement only (XLION_GPU_TIMES): what runs before the scene pass - the GI's half resolution read, GPU culling, the screen tail
        // (PRESCENE: the largest of 120 frames each; the sun turning added ~3.8 ms there, 2026-10-29)
        static std::unordered_map<const void*, xgpu::timestamps> s_Pre;      // ponytail: per renderer, never erased (a measurement hook)
        xgpu::timestamps* pPre = nullptr;
        static double s_PreCpuMs = 0;
        if( bMain && xlionrender::Env("XLION_GPU_TIMES") )
        {
            auto [It, bNew] = s_Pre.try_emplace(this);
            if( bNew && m_Device.Create(It->second, { .m_Count = 5 }) ) s_Pre.erase(It);
            else pPre = &It->second;
        }
        if( pPre )
        {
            static std::vector<std::uint64_t> s_Ns; static double s_Max[4] = {}; static int s_n = 0;
            if( pPre->getResults(s_Ns) && s_Ns.size() >= 5 && s_Ns[4] >= s_Ns[3] && s_Ns[3] >= s_Ns[2] && s_Ns[2] >= s_Ns[1] && s_Ns[1] >= s_Ns[0] )
                for( int k = 0; k < 4; ++k ) s_Max[k] += double(s_Ns[k + 1] - s_Ns[k]) * 1e-6;
            if( ++s_n == 120 ) { std::printf("PRESCENE mean of 120 frames: setup (lights, cells, GI) %.3f, GI half-res %.3f, GPU cull %.3f, screen tail %.3f ms; half-res pixels fresh %d reprojected %d rejected %d aged %d; its CPU %.3f ms\n", s_Max[0] / 120, s_Max[1] / 120, s_Max[2] / 120, s_Max[3] / 120
                                             , m_GIStats.m_nScreenFresh, m_GIStats.m_nScreenReprojected, m_GIStats.m_nScreenRejected, m_GIStats.m_nScreenAged, s_PreCpuMs / 120); std::fflush(stdout); s_n = 0; s_PreCpuMs = 0; for( auto& v : s_Max ) v = 0; }
            Window.WriteTimestamp(*pPre, 0);
        }
        details::ubo_lighting* pLight = nullptr;       // (this render's lighting block: the screen tail sets its part, renderTail)
        const xmath::fmat4 ToEye = xmath::fmat4::fromTranslation(-Eye);
        // Scene -> HDR
        {
            {
                const auto& L     = Scene.m_Sun;
                auto&       Light = m_LightUBO.allocEntry<details::ubo_lighting>();
                pLight = &Light;
                Light.m_LightColor        = xmath::fvec4(L.m_Color * L.m_Intensity, 1);
                Light.m_AmbientLightColor = xmath::fvec4(Scene.m_Ambient, 0);
                Light.m_wSpaceLightPos    = xmath::fvec4(-L.m_Direction.NormalizeCopy(), 0);
                Light.m_wSpaceEyePos      = xmath::fvec4(0, 0, 0, 1);       // camera-centered
                Light.m_LightParams       = { 0, L.m_Temperature, 1, 0 };

                // The sun's shadow: the virtual pages (its basis, levels and this frame's table) or the clipmap. With the pages on, the
                // clipmap's flag is off: materials compiled before the pages read only it, and sample no sun shadow rather than pages as tiles
                const bool bVSM = L.m_bCastShadows;
                const xmath::fvec3 BX = m_VSM.m_X, BY = m_VSM.m_Y, BZ = m_VSM.m_Z;
                Light.m_W2Light = xmath::fmat4::fromIdentity();
                for( int c = 0; c < 3; ++c )
                {
                    const float Axis[3][3] = { { BX.m_X, BX.m_Y, BX.m_Z }, { BY.m_X, BY.m_Y, BY.m_Z }, { BZ.m_X, BZ.m_Y, BZ.m_Z } };
                    for( int r = 0; r < 3; ++r ) Light.m_W2Light(r, c) = Axis[r][c];
                }
                Light.m_CameraPos = xmath::fvec4(Eye, 1);
                Light.m_ShadowLevels = {};           // (the clipmap's: reserved, zero - the compiled materials' layout keeps them)
                static const bool s_bNoLocalShadows = xlionrender::Env("XLION_NO_LOCAL_SHADOWS") != nullptr;     // (measurement: the lamps' shadow taps' share of the shading)
                Light.m_ShadowParams = { s_bNoLocalShadows ? 0.0f : float(sunAtlasWidth()), float(localTableBase()), float(m_LocalPages.getSetup().SlotsPerRow()), m_LocalPages.ResolutionBias() };     // the local lights' pages
                const auto& VS = m_VSM.getSetup();
                static const int s_DiscTaps = [] { const char* p = xlionrender::Env("XLION_VSM_DISC_TAPS"); return p ? std::atoi(p) : 0; }();      // (measurement: overrides the sun's)
                // (one tap while the sun turns: its pages rebuild every epoch then and the disc's taps cost 1 to 2.6 ms; a turning sun's soft edge is a blur anyway. XLION_VSM_MOVING_TAPS=<n>, 0: off)
                static const int s_MovingTaps = [] { const char* p = xlionrender::Env("XLION_VSM_MOVING_TAPS"); return p ? std::atoi(p) : 1; }();
                { const xmath::fvec3 Sd = Scene.m_Sun.m_Direction; if( (Sd - m_SunDirSeen).Length() > 1e-5f ) { m_SunDirSeen = Sd; m_SunMoved = std::chrono::steady_clock::now(); } }
                const bool bSunTurning = s_MovingTaps > 0 && std::chrono::duration<float>(std::chrono::steady_clock::now() - m_SunMoved).count() < 0.5f;
                Light.m_VSMParams  = bVSM ? xmath::fvec4{ float(std::clamp(s_DiscTaps > 0 ? s_DiscTaps : bSunTurning ? std::min(s_MovingTaps, Scene.m_Sun.m_ShadowDiscTaps) : Scene.m_Sun.m_ShadowDiscTaps, 1, 16)), m_VSM.Level0Texel(), float(VS.m_Levels), float(vsmTableBase()) } : xmath::fvec4{ 0, 0, 0, 0 };     // (x: the disc's taps; 0: no sun shadow)
                static const float s_Band = [] { const char* p = xlionrender::Env("XLION_VSM_BAND"); return p ? float(std::atof(p)) : -1.0f; }();       // (measurement: the levels' blend band)
                static const bool  s_bNoDisc = xlionrender::Env("XLION_VSM_NO_DISC") != nullptr;                                                         // (measurement: no contact-hardening)
                Light.m_VSMParams2 = { float(VS.m_Window), s_Band > 0.0f ? s_Band : m_VSMLevelBand, float(VS.SlotsPerRow()), float(m_VSMDebugView > 0 ? m_VSMDebugView : m_bVSMTapTranslation ? 1 : 0) };
                for( int i = 0; i < shadows::vsm::max_levels_v; ++i )
                {
                    const auto& Lv = m_VSM.m_Levels[i];
                    Light.m_VSMLevels[i] = { float(Lv.m_OriginX), float(Lv.m_OriginY), Lv.m_PageSize, Lv.m_DepthCenter };
                }
                Light.m_VSMParams3 = { m_VSM.ResolutionBias(), s_bNoDisc ? 0.0f : std::tan(std::max(L.m_SourceAngle, 0.0f) * 0.5f * 3.14159265f / 180.0f), float(m_VSMSoftLevels)
                                     , m_VSM.HasMovers() ? (m_VSM.m_bMoversCoarse ? -1.0f : 1.0f) * float(vsmTableBase() + 2 * VS.TableUints() + 1) : 0.0f };     // w: the movers' table + 1 (0: none; negative: a level coarser)
                Light.m_VSMPrevW2Light = xmath::fmat4::fromIdentity();
                {
                    const float Axis[3][3] = { { m_VSM.m_PX.m_X, m_VSM.m_PX.m_Y, m_VSM.m_PX.m_Z }, { m_VSM.m_PY.m_X, m_VSM.m_PY.m_Y, m_VSM.m_PY.m_Z }, { m_VSM.m_PZ.m_X, m_VSM.m_PZ.m_Y, m_VSM.m_PZ.m_Z } };
                    for( int c = 0; c < 3; ++c ) for( int r = 0; r < 3; ++r ) Light.m_VSMPrevW2Light(r, c) = Axis[r][c];
                }
                for( int i = 0; i < shadows::vsm::max_levels_v; ++i )
                {
                    const auto& Lv = m_VSM.m_PrevLevels[i];
                    Light.m_VSMPrevLevels[i] = { float(Lv.m_OriginX), float(Lv.m_OriginY), Lv.m_PageSize, Lv.m_DepthCenter };
                }
                Light.m_VSMPrevParams = { bVSM && m_VSM.HasPrevious() ? 1.0f : 0.0f, m_VSM.Fade(), m_VSM.m_PrevLevel0Texel, float(vsmTableBase() + VS.TableUints()) };

                writeLocalLights(Window, Scene, Eye, Light);
                if( m_LightModeUsed == light_mode::SCREEN && !bMain ) Light.m_LocalCount.m_Y = float(int(light_mode::CELLS));     // (the screen tail is the main view's: the others' every light of the list)
                // XLION_REFL_GRADE=<roughness below 1>: the mirror view at that roughness on every surface, for grading against the ground truth (with XLION_GI_REF: it is what shows: 4 +)
                static const float s_Grade = [] { const char* p = xlionrender::Env("XLION_REFL_GRADE"); return p ? float(std::atof(p)) : -1.0f; }();
                static const float s_GradeBase = s_Grade >= 0.0f ? (xlionrender::Env("XLION_GI_REF") ? 4.0f : xlionrender::Env("XLION_REFL_GRADE_RAW_PROBE") ? 6.0f : 2.0f) + s_Grade : 2.0f;
                Light.m_EnvParams = { m_bEnvOn ? 1.0f : 0.0f, m_EnvIntensity, bMain && Scene.m_bDebugReflection ? (Scene.m_bDebugReflectionMirror ? (Scene.m_bDebugReflectionPathTraced ? 4.01f : s_GradeBase) : 1.0f) : 0.0f, 0 };     // (the main view only: a probe captures the lit scene)

            }

            auto& ViewBlock = m_StaticGeomUBO.allocEntry<details::ubo_geom_view>();      // camera-centered
            ViewBlock.m_w2C         = W2C;
            ViewBlock.m_WorldOffset = xmath::fvec4(m_Origin - Eye, 0);
            if( bMain && View.m_bTailPrev )     // (the screen tail's motion: last frame's camera)
            {
                ViewBlock.m_PrevW2C         = View.m_TailPrevW2C;
                ViewBlock.m_PrevWorldOffset = xmath::fvec4(m_Origin - View.m_TailPrevEye, 1);
            }

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

            static const bool s_bNoPrepass = xlionrender::Env("XLION_NO_PREPASS") != nullptr;
            // The G-buffer prepass (gi_gbuffer_gather_plan.md): the main view's prepass in a pass of its own writing each pixel's normal and
            // submesh instance too (a masked one's through its cutout); the scene pass loads its depth
            const bool bGPre = bMain && !s_bNoPrepass && !(bPlan && m_bBoxesOnly) && createGPreTargets(View) == nullptr;
            // A submesh the half-resolution GI pass reads (and the depth prepass lays): opaque, single sided, depth written and tested, filled
            const auto Opaque = [&]( const material_slot* pMat )
            {
                if( pMat == nullptr ) return true;
                const auto& St = pMat->m_State;
                return !(St.m_Blend != material_state::blend::OFF || St.m_bDoubleSided || !St.m_bDepthWrite || !St.m_bDepthTest || St.m_bWireframe);
            };
            // ... and a masked one too (the ivy): on the half-resolution read as if solid - its cutout is the full-resolution pass's, whose
            // upsample keeps the most alike of the 4 (never the per-pixel walk again: 5.9 of Bistro's 13.7 ms). XLION_GI_SCREEN_MASKED=0: off
            static const bool s_bScreenMasked = [] { const char* p = xlionrender::Env("XLION_GI_SCREEN_MASKED"); return !(p && p[0] == '0'); }();
            const auto ScreenOk = [&]( const material_slot* pMat )
            {
                if( Opaque(pMat) ) return true;
                const auto& St = pMat->m_State;
                // (double sided: in the G-buffer prepass too - its pipeline that culls nothing)
                if( bGPre && St.m_Blend == material_state::blend::OFF && St.m_bDepthWrite && St.m_bDepthTest && !St.m_bWireframe ) return true;
                return s_bScreenMasked && St.m_Blend == material_state::blend::MASKED && (bGPre || !St.m_bDoubleSided) && St.m_bDepthWrite && St.m_bDepthTest && !St.m_bWireframe;
            };
            // The GI read at half resolution (scene::gi_settings::m_bHalfRes; XLION_GI_HALFRES=0/1 overrides): the main view, no GI debug view,
            // gathered from the G-buffer prepass (GatherGI, below: no geometry drawn for it) into the view's buffer; the scene pass's GI draws
            // upsample it (SurfelGIScreen) instead of reading the surfels each pixel. With the per-pixel read off (the vertex read alone) none:
            // the gather has no vertices - the scene pass's own vertex read
            static const int s_HalfResEnv = [] { const char* p = xlionrender::Env("XLION_GI_HALFRES"); return p ? std::atoi(p) : -1; }();     // (measurement: 0 off, 1 on)
            // (the GI off: the same pass, no GI in it - the ambient occlusion's way to the materials)
            // (the voxels' light view: the surfels hold nothing to show, the gather's read does)
            const bool bVoxLightView = Scene.m_GI.m_Debug == gi_debug::LIGHT && gi::gpu_trace::VoxConfig()[0] > 0.0f;
            if( pLight ) pLight->m_ShadowLevels[1].m_X = bVoxLightView ? 2.0f : 0.0f;
            const bool bScreen = bGPre && (s_HalfResEnv >= 0 ? s_HalfResEnv != 0 : Scene.m_GI.m_bHalfRes) && (Scene.m_GI.m_Debug == gi_debug::NONE || bVoxLightView)
                              && (Scene.m_GI.m_bPerPixel || !Scene.m_GI.m_bEnable) && createGIScreenTargets(View) == nullptr;
            static const bool s_bTile = [] { const char* p = xlionrender::Env("XLION_TILE"); return p && p[0] != '2' && p[0] != '0'; }();        // (2: the attributes written, no tile pass: the prepass's cost)
            const bool bTile = s_bTile && bGPre && bScreen && View.m_TilePass.m_Private;
            const auto TileMat = [&]( const material_slot* pMat )       // (opaque or masked, one or two sided: the prepass has them all)
            {
                if( !bTile || !pMat || pMat->m_TileGraph < 0 ) return false;
                const auto& St = pMat->m_State;
                return (St.m_Blend == material_state::blend::OFF || St.m_Blend == material_state::blend::MASKED) && St.m_bDepthWrite && St.m_bDepthTest && !St.m_bWireframe;
            };
            // What the prepass keeps for the tile pass: the material's base, or nothing (+ 1 in the id's w: 0, the raster path)
            const auto TileBase = [&]( const material_slot* pMat ) { return TileMat(pMat) ? pMat->m_ParamBase : 0xFFFFFFFFu; };
            // The ambient occlusion (scene::gi_settings::m_bAO; XLION_AO=0/1 overrides): with the half-resolution pass, the GI on or off
            static const int s_AOEnv = [] { const char* p = xlionrender::Env("XLION_AO"); return p ? std::atoi(p) : -1; }();
            const bool bAO = bScreen && (s_AOEnv >= 0 ? s_AOEnv != 0 : Scene.m_GI.m_bAO);
            if( pPre ) Window.WriteTimestamp(*pPre, 1);
            const auto PreCpu0 = std::chrono::steady_clock::now();     // (measurement: the half-res block's CPU, PRESCENE)
            if( bMain ) m_GIStats.m_nScreenDraws = m_GIStats.m_nScreenCalls = 0;
            if( bMain && !bScreen ) View.m_bGIScreenPrev = false;      // (a frame without it: nothing to reproject from)
            if( bScreen )
            {
                // Temporal (XLION_GI_TEMPORAL=0: off): last frame's buffer becomes the one reprojected from; this frame's is cleared (a pixel no
                // GI surface covers keeps nothing of an older frame) and gets the header: the size, the frame's 2x2 pattern, whether last
                // frame's is there, its world to clip and the eye's move (xlionrender_gi_gather_frag.glsl)
                static const int s_TemporalEnv = [] { const char* p = xlionrender::Env("XLION_GI_TEMPORAL"); return p ? std::atoi(p) : -1; }();     // (0 off, 1/2/4 the rate)
                const bool bTemporal = s_TemporalEnv >= 0 ? s_TemporalEnv != 0 : Scene.m_GI.m_bTemporal;
                const int  Rate      = s_TemporalEnv > 0 ? s_TemporalEnv : Scene.m_GI.m_TemporalRate;
                const std::uint32_t RateLog2 = Rate >= 4 ? 2u : Rate >= 2 ? 1u : 0u;
                const int  CSlot = int(m_Device.getFrameIndex() % ring_frames_v);
                (void)View.m_GIScreenCounters.MemoryMap(CSlot, 1, [&](void* p)         // (this slot's frame is done: its counters)
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
                        pH[28] = std::uint32_t(GIScreenDivisor());      // (entry 7.x: the read's resolution divisor - the upsample's, mb_standard_pbr.frag)
                        std::memcpy(pH + 4, &View.m_GIScreenPrevW2C, sizeof(float) * 16);        // (entries 1-4: its columns, as the UBO's mat4)
                        const float M[3] = { Move[0], Move[1], Move[2] };
                        std::memcpy(pH + 20, M, sizeof(M));
                    });
                    const xgpu::buffer::copy C{ HBase, 0, screen_header_v };
                    Window.CopyBuffer(View.m_GIScreenHeader, View.m_GIScreenBuffer, std::span{ &C, 1 });
                }
                View.m_GIScreenPrevW2C = W2C; View.m_GIScreenPrevEye = Eye; View.m_bGIScreenPrev = true; ++View.m_GIScreenFrame;
                {
                    // The gather's table (xlionrender_gi_gather_common.glsl): every visible rigid GI instance's clusters - their GI reference
                    // and GI buffer, the passes' list of buffers (GatherBuffers); the passes run after the G-buffer prepass, below
                    m_GatherTable.assign(40, 0u);
                    // (XLION_VOX_PUBLISHED: word 37 the published read on, 38 the frame - the gather's VoxReadPublished, the resolve's stamp)
                    static const bool s_bVoxPub = xlionrender::Env("XLION_VOX_PUBLISHED") != nullptr;
                    { static const char* s_pPub = xlionrender::Env("XLION_VOX_PUBLISHED"); m_GatherTable[37] = s_pPub ? (s_pPub[0] == '2' ? 2u : 1u) : 0u; }      // (2, measurement: no level read)
                    m_GatherTable[38] = (m_VoxPubFrame + 1u) & 0x7FFFFFFFu;      // (its own count of gathers: the device's frame index steps by every window's frames - the lists' parity broke)
                    static const bool s_bNoGatherRefl = xlionrender::Env("XLION_GI_NO_REFL") != nullptr;      // (measurement: the gather's reflection term off - the path traced view has none: read against it fairly)
                    m_GatherTable[29] = s_bNoGatherRefl ? 1u : 0u;
                    static const bool s_bBinPdf = xlionrender::Env("XLION_VOX_NO_BIN_PDF") == nullptr;
                    m_GatherTable[31] = s_bBinPdf ? 1u : 0u;
                    {   // (XLION_VOX_DIRS=<level>,<k>[,<roughness>]: a prototype, the reflections' light in a coarse level by k x k directions: words 32, 33, xlionrender_gi_gather_frag.glsl VoxDirsRead. The read knows one roughness, XLION_REFL_GRADE's)
                    { static const float s_BucketAO = [] { const char* p = xlionrender::Env("XLION_VOX_AO_BUCKET"); return p ? float(std::atof(p)) : 0.0f; }(); std::memcpy(&m_GatherTable[34], &s_BucketAO, 4); }      // (XLION_VOX_AO_BUCKET: the gather's read)
                    { static const float s_Reach = [] { const char* p = xlionrender::Env("XLION_VOX_REACH"); return p ? float(std::atof(p)) : 0.0f; }(); std::memcpy(&m_GatherTable[36], &s_Reach, 4); }     // (B1: corners the point can't reach on screen weigh ~0; gi_gather_frag VoxReach)
                    { static const float s_Trust = [] { const char* p = xlionrender::Env("XLION_VOX_TRUST"); return p ? float(std::atof(p)) : 0.0f; }(); std::memcpy(&m_GatherTable[35], &s_Trust, 4); }      // (XLION_VOX_TRUST: the samples a level's record needs to be read alone, 256 when 0: xlionrender_gi_gather_frag.glsl ClipLevel)      // (XLION_VOX_AO_BUCKET: the gather's read)
                        static const std::array<float, 3> s_Dirs = [] { std::array<float, 3> D{ -1.0f, 8.0f, 0.3f }; if( const char* p = xlionrender::Env("XLION_VOX_DIRS") ) xlionrender::Scan(p, "%f,%f,%f", &D[0], &D[1], &D[2]); if( const char* g = xlionrender::Env("XLION_REFL_GRADE"); g && !xlionrender::Env("XLION_VOX_DIRS_BAND") ) D[2] = float(std::atof(g)); return D; }();      /* (XLION_VOX_DIRS_BAND: the read's roughness stays the third number, whatever is graded) */
                        static const bool s_bDirsFixed = xlionrender::Env("XLION_VOX_DIRS_FIXED") != nullptr;
                        static const bool s_bDirsInTrace = xlionrender::Env("XLION_VOX_DIRS_TRACE") != nullptr;      /* (the read in the reflections' pass instead: refl_trace_comp) */
                        if( s_Dirs[0] >= 0.0f && !s_bDirsInTrace ) { m_GatherTable[32] = 8u | (s_bDirsFixed ? 16u : 0u) | (std::uint32_t(s_Dirs[0]) & 3u) | (s_Dirs[1] > 8.5f ? 4u : 0u) | (std::uint32_t(std::clamp(s_Dirs[2], 0.0f, 1.0f) * 255.0f + 0.5f) << 8); static const bool s_bDirsStill = xlionrender::Env("XLION_VOX_DIRS_STILL") != nullptr; m_GatherTable[33] = s_bDirsStill ? 0u : std::uint32_t(m_Device.getFrameIndex()); }      /* (STILL: the same samples each frame: the store's own flicker, not the read's) */
                    }
                    {   // X1: the clipmap the gather reads instead (words 20 on, xlionrender_gi_gather_frag.glsl); 20: 0 none
                        static const std::array<float, 2> s_ClipCell = [] { std::array<float, 2> A{ 0.0f, 0.5f }; if( const char* p = xlionrender::Env("XLION_GI_CLIP") ) (void)xlionrender::Scan(p, "%f,%f", &A[0], &A[1]); return A; }();
                        m_pGatherClip = s_ClipCell[0] > 0.0f && m_pGITrace ? m_pGITrace->ClipBuffer() : nullptr;
                        if( m_pGatherClip )
                        {
                            const auto E = m_pGITrace->ClipEye();
                            const float C[7] = { s_ClipCell[1], E[0], E[1], E[2], Eye[0], Eye[1], Eye[2] };
                            m_GatherTable[20] = 1u;
                            std::memcpy(m_GatherTable.data() + 21, C, sizeof(C));
                        }
                        // The ground truth instead (scene::gi_settings::m_bPathTraced, or XLION_GI_REF=<paths a frame>[,<bounces>]): words 20 = 2, 28 the half-size width; the gather writes each pixel's point, reads its mean
                        static const std::array<int, 2> s_Ref = [] { std::array<int, 2> A{ 0, 4 }; if( const char* p = xlionrender::Env("XLION_GI_REF") ) (void)xlionrender::Scan(p, "%d,%d", &A[0], &A[1]); return A; }();
                        m_pGatherRefS = m_pGatherRefA = nullptr;
                        // Experiment X1c (gpu_trace::VoxConfig): the gather reads voxels of light on the surfaces (xlionrender_gi_ref_comp.glsl VoxMain), the rays go out in updateGI
                        const auto& s_Vox = gi::gpu_trace::VoxConfig();
                        std::uint32_t LightsKey = 2166136261u;         // (the lights' version: a change starts the tracer over)
                        const auto KeyOf = [&](const void* p, std::size_t n) { const auto* b = static_cast<const std::uint8_t*>(p); for( std::size_t k = 0; k < n; ++k ) LightsKey = (LightsKey ^ b[k]) * 16777619u; };
                        { const float Sn[7] = { Scene.m_Sun.m_Direction[0], Scene.m_Sun.m_Direction[1], Scene.m_Sun.m_Direction[2], Scene.m_Sun.m_Color[0], Scene.m_Sun.m_Color[1], Scene.m_Sun.m_Color[2], Scene.m_Sun.m_Intensity }; KeyOf(Sn, sizeof(Sn)); }
                        for( const auto& LL : Scene.m_LocalLights ) { const float Ls[5] = { LL.m_Position[0], LL.m_Position[1], LL.m_Position[2], LL.m_Intensity, LL.m_Radius }; KeyOf(Ls, sizeof(Ls)); }
                        if( (s_Ref[0] > 0 || Scene.m_GI.m_bPathTraced || Scene.m_bDebugReflectionPathTraced) && m_pGITrace && m_pGITrace->RefStep(Window, std::size_t(View.m_GIScreenWidth) * View.m_GIScreenHeight, m_Device.getFrameIndex(), s_Ref[0] > 0 ? s_Ref[0] : 1 << 18, s_Ref[1], LightsKey) )
                        {
                            m_pGatherRefS = m_pGITrace->RefSamples(); m_pGatherRefA = m_pGITrace->RefAccum();
                            const float Ey[3] = { Eye[0], Eye[1], Eye[2] };
                            m_GatherTable[20] = 2u; m_GatherTable[28] = std::uint32_t(View.m_GIScreenWidth);
                            static const float s_GradeRef = [] { const char* p = xlionrender::Env("XLION_REFL_GRADE"); return p ? float(std::atof(p)) + (xlionrender::Env("XLION_REFL_REF_NO_WEIGHT") ? 2.0f : 0.0f) : -1.0f; }();      // (the reflections' ground truth at this roughness; none: the irradiance)
                            const float GradeRef = Scene.m_bDebugReflectionPathTraced ? 0.01f : s_GradeRef;      // (the mirror view's path traced option: a mirror's ground truth)
                            std::memcpy(m_GatherTable.data() + 30, &GradeRef, sizeof(GradeRef));
                            std::memcpy(m_GatherTable.data() + 25, Ey, sizeof(Ey));
                        }
                        if( s_Vox[0] > 0.0f ) { m_VoxPixels = std::size_t(View.m_GIScreenWidth) * View.m_GIScreenHeight; m_VoxKey = LightsKey; }
                        // Experiment X1c: the voxels' rays (here, not in updateGI: at the frame's start they measured 3.4 ms whatever their number), as many as fit their ms a frame by what they measured
                        // (64 rays cost about as much as 32768 and long rays are what cost - why is not known yet (gi_next_level_analysis.md) -, so with nothing changing, none on most frames, not fewer on each)
            static const double s_VoxStop = [] { const char* p = xlionrender::Env("XLION_VOX_STOP"); return p ? std::atof(p) : 0.0; }();      /* (measurement: the voxels' rays stop this many seconds after the first: what still moves is not the store) */
            static const auto s_VoxT0 = std::chrono::steady_clock::now();
            // Settled (XLION_VOX_SETTLE seconds, 0: never): lights, eye and view all still that long: no more rays until one of them moves. With the rays stopped a still moto's whole surfaces stop shifting
            // (mean change 0.6 to 0.2 of 255, Tomas: rough reflections should settle). ponytail: moving objects don't wake it, they aren't in the lights' key; wake on a changed caster when they matter
            static const double s_VoxSettle = [] { const char* p = xlionrender::Env("XLION_VOX_SETTLE"); return p ? std::atof(p) : 0.0; }();
            const float* pFw = reinterpret_cast<const float*>(&W2C);
            const xmath::fvec3 VoxFw = xmath::fvec3{ pFw[3], pFw[7], pFw[11] } / std::max(xmath::fvec3{ pFw[3], pFw[7], pFw[11] }.Length(), 1e-12f);
            if( VoxFw.Dot(m_VoxFwd) < 0.999999f ) { m_VoxFwd = VoxFw; m_VoxTurned = std::chrono::steady_clock::now(); }
            const bool bVoxSettled = s_VoxSettle > 0.0 && m_pGITrace && m_pGITrace->VoxQuiet() && std::chrono::duration<double>(std::chrono::steady_clock::now() - m_VoxTurned).count() > s_VoxSettle;
            if( const auto& Vox = s_Vox; Vox[0] > 0.0f && !m_pGatherRefS && m_pGITrace && m_VoxPixels > 0 && !bVoxSettled && !(s_VoxStop > 0.0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - s_VoxT0).count() > s_VoxStop) && !(m_pGITrace->VoxQuiet() && m_Device.getFrameIndex() % 8 != 0) )
                        {
                            if( !m_bVoxTSTried ) { m_bVoxTSTried = true; m_bVoxTS = m_Device.Create(m_VoxTS, { .m_Count = 2 }) == nullptr; }
                            if( std::vector<std::uint64_t> Ns; m_bVoxTS && m_VoxTS.getResults(Ns) && Ns.size() >= 2 && Ns[1] > Ns[0] )
                            {
                                const double Ms = double(Ns[1] - Ns[0]) * 1e-6, Budget = double(Vox[1]);
                                m_VoxRays = std::clamp(m_VoxRays * std::clamp(Budget / std::max(Ms, 1e-3), 0.7, 1.3), 2048.0, 65536.0);     // (groups of 4 lanes: the cost follows the rays - in groups of 64 a dispatch cost 2 ms and more with any rays at all, and the budget could only starve them. The top: past it many small groups cost more than the rays)
                                static const char* s_pFixed = xlionrender::Env("XLION_VOX_RAYS");      // (measurement: the rays' cost against their number)
                                if( s_pFixed ) m_VoxRays = std::atof(s_pFixed);
                                static const bool s_bVoxLog = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                                if( s_bVoxLog && m_Device.getFrameIndex() % 120 == 0 ) { std::printf("VOXRAYS %.2f ms, %.0f rays, budget %.2f\n", Ms, m_VoxRays, Budget); std::fflush(stdout); }
                            }
                            if( m_bVoxTS ) Window.WriteTimestamp(m_VoxTS, 0);
                            m_bVoxOn = m_pGITrace->VoxStep(Window, m_VoxPixels, m_Device.getFrameIndex(), int(m_VoxRays), int(Vox[2]), Vox[0], { Eye[0], Eye[1], Eye[2] }, m_VoxKey, View.m_bHDRPrev ? View.m_HDRPrev : View.m_HDR, View.m_GIScreenWidth, !Scene.m_bDebugReflection && Scene.m_GI.m_Debug == gi_debug::NONE);
                            if( m_bVoxTS ) Window.WriteTimestamp(m_VoxTS, 1);
                        }
                        if( s_Vox[0] > 0.0f && !m_pGatherRefS && m_bVoxOn && m_pGITrace && m_pGITrace->RefSamples() && m_pGITrace->VoxBuffer() )     // (the ground truth above stays: the voxels overwrote it, a black or voxel image in its place)
                        {
                            m_pGatherRefS = m_pGITrace->RefSamples(); m_pGatherRefA = m_pGITrace->RefAccum(); m_pGatherClip = m_pGITrace->VoxBuffer();
                            const auto A = m_pGITrace->VoxAnchor();
                            const float C[7] = { s_Vox[0], A[0], A[1], A[2], Eye[0], Eye[1], Eye[2] };
                            m_GatherTable[20] = 3u; m_GatherTable[28] = std::uint32_t(View.m_GIScreenWidth);
                            std::memcpy(m_GatherTable.data() + 21, C, sizeof(C));
                        }
                    }
                    {
                        const xmath::fmat4 C2W = W2C.Inverse();
                        std::memcpy(m_GatherTable.data(), &C2W, sizeof(float) * 16);
                        const xmath::fvec3 Off = m_Origin - Eye;
                        const float O[3] = { Off[0], Off[1], Off[2] };
                        std::memcpy(m_GatherTable.data() + 16, O, sizeof(O));
                    }
                    m_GatherBuffers.clear();
                    std::uint32_t nSlots = 0;
                    for( const auto& Vis : m_Visible ) if( Vis.m_pMesh->m_bCluster && m_DrawSlot[Vis.m_iDraw] != instance_slots::none_v ) nSlots = std::max(nSlots, m_DrawSlot[Vis.m_iDraw] + 1u);
                    m_GatherTable[19] = nSlots;
                    m_GatherTable.resize(40 + std::size_t(nSlots), 0u);
                    const auto BufferIndex = [&]( const xgpu::buffer* p ) -> std::uint32_t
                    {
                        for( std::size_t k = 0; k < m_GatherBuffers.size(); ++k ) if( m_GatherBuffers[k] == p ) return std::uint32_t(k);     // ponytail: linear, a few buffers (Bistro's sections ~30); a map if a scene has hundreds
                        m_GatherBuffers.push_back(p);
                        return std::uint32_t(m_GatherBuffers.size() - 1);
                    };
                    bool bFull = false;
                    for( const auto& Vis : m_Visible )
                    {
                        const std::size_t iDraw = Vis.m_iDraw;
                        const auto*       pMesh = Vis.m_pMesh;
                        const std::uint32_t Slot = m_DrawSlot[iDraw];
                        if( !pMesh->m_bCluster || Slot == instance_slots::none_v ) continue;
                        if( pMesh->m_Geom.m_Kind == geom_kind::SKIN )         // (a skinned one: the probe grid - its GI read is the gather's too)
                        {
                            const xgpu::buffer* pProbes = Scene.m_GI.m_bEnable && Scene.m_GI.m_bProbes && m_pGITrace ? m_pGITrace->ProbeBuffer() : nullptr;
                            // (B8, XLION_SKIN_VOX: with the voxel GI - the editor's default, the probe grid off - a skinned surface had no entry, so the gather
                            // skipped its pixels: no indirect light on the characters. The voxel read is the gather's for every pixel with an entry (the
                            // entry's source unread), so the skin gets one: the scene's own store at its points)
                            static const bool s_bSkinVox = [] { const char* p = xlionrender::Env("XLION_SKIN_VOX"); return p == nullptr || p[0] != '0'; }();     // (on again 2026-11-07: its pixels read the store but start no rays - feeding it blackened the whole voxel GI. Graded at 35 deg sun: characters mean 12.88 to 9.76, whole image 40.78 to 40.40 %; XLION_SKIN_VOX=0 off)
                            // (the path traced ground truth, mode 2, likewise: its rays start from every pixel with an entry - without one the reference's
                            // characters had no indirect light either, and a character without GI graded as right. Always: the reference must light them)
                            const bool bVoxRead = m_GatherTable.size() > 20 && ((s_bSkinVox && m_GatherTable[20] == 3u) || m_GatherTable[20] == 2u);
                            if( (pProbes == nullptr && !bVoxRead) || m_GatherTable.size() + 1 + 4 * pMesh->m_Geom.m_Clusters.size() > std::size_t(gather_table_uints_v) ) continue;
                            m_GatherTable.push_back(std::uint32_t(pMesh->m_Geom.m_Clusters.size()));     // (its clusters' count before them: the gather reads no other instance's)
                            m_GatherTable[40 + Slot] = std::uint32_t(m_GatherTable.size());
                            const std::uint32_t k = pProbes ? BufferIndex(pProbes) : 0u;
                            for( std::size_t c = 0; c < pMesh->m_Geom.m_Clusters.size(); ++c ) m_GatherTable.insert(m_GatherTable.end(), { 0u, 0xFFFFFFFEu, k, 0u });     // (xlionrender_surfel_gi.glsl gi_probes_v)
                            continue;
                        }
                        const auto& G = pMesh->m_Geom;
                        const bool bSections = !pMesh->m_GISections.empty() && pMesh->m_GIClusterSection.size() == G.m_Clusters.size();
                        const auto GI = iDraw < m_DrawGI.size() ? m_DrawGI[iDraw] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };
                        if( !(bSections ? iDraw < m_DrawGISec.size() && !m_DrawGISec[iDraw].empty() : pMesh->m_pGI && GI.second != 0xFFFFFFFFu) ) continue;
                        if( m_GatherTable.size() + 1 + 4 * G.m_Clusters.size() > std::size_t(gather_table_uints_v) ) { bFull = true; continue; }
                        m_GatherTable.push_back(std::uint32_t(G.m_Clusters.size()));
                        m_GatherTable[40 + Slot] = std::uint32_t(m_GatherTable.size());
                        for( std::uint32_t c = 0; c < std::uint32_t(G.m_Clusters.size()); ++c )
                        {
                            std::pair<std::uint32_t, std::uint32_t> R = GI;
                            const xgpu::buffer* pBuf = &pMesh->m_GIBuffer;
                            if( bSections )
                            {
                                const std::uint16_t k = pMesh->m_GIClusterSection[c];
                                R    = k < m_DrawGISec[iDraw].size() ? m_DrawGISec[iDraw][k] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };
                                pBuf = &pMesh->m_GISections[k].m_Buffer;
                            }
                            // Experiment X1b, its cheapest step (XLION_GI_READ_PROBES=<spacing> [<probes a frame> [<cascades> [1: the rays' hits read them too, not the surfels]]]): every surface reads the ray-traced probe grid
                            static const bool s_bReadProbes = xlionrender::Env("XLION_GI_READ_PROBES") != nullptr;
                            if( const xgpu::buffer* pP = s_bReadProbes && m_pGITrace ? m_pGITrace->ProbeBuffer() : nullptr ) { R = { 0u, 0xFFFFFFFEu }; pBuf = pP; }
                            m_GatherTable.insert(m_GatherTable.end(), { R.first, R.second, R.second == 0xFFFFFFFFu ? 0u : BufferIndex(pBuf), 0u });
                        }
                    }
                    if( bFull && m_Device.getFrameIndex() % 240 == 0 ) { std::printf("GATHERFULL the GI gather's table is full (%d uints): instances left without GI\n", gather_table_uints_v); std::fflush(stdout); }
                    m_GatherTableBase = std::uint32_t(m_Device.getFrameIndex() % ring_frames_v) * std::uint32_t(gather_table_uints_v);
                    (void)View.m_GIGatherTable.MemoryMapFrameRegion(int(m_GatherTableBase), int(m_GatherTable.size()), [&](void* p) { std::memcpy(p, m_GatherTable.data(), m_GatherTable.size() * sizeof(std::uint32_t)); });
                    m_GIStats.m_nScreenDraws = int(m_GatherBuffers.size());          // (its passes: a full-screen draw a GI buffer)
                    m_GIStats.m_nScreenCalls = int(m_GatherBuffers.size()) + 1;      // (+ the classification)
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
                static const bool s_bBlendNone = [] { const char* p = xlionrender::Env("XLION_GI_ABLATE"); return p && std::string_view(p).find("blend_none") != std::string_view::npos; }();     // (measurement: the blended draws without GI)
                const bool  bNoGI  = s_bBlendNone && pMat && pMat->m_State.m_Blend != material_state::blend::OFF;
                const auto  GI     = iDraw < m_DrawGI.size() && !bNoGI ? m_DrawGI[iDraw] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };
                const bool  bSections0 = !bNoGI && bRigid && !pMesh->m_GISections.empty() && pMesh->m_GIClusterSection.size() == pMesh->m_Geom.m_Clusters.size();
                // read at half resolution (the pass above): the buffer as its GI buffer, gi_screen_v as its level - one buffer for every section
                const bool  bScreenSub = bScreen && bRigid && ScreenOk(pMat) && (!Scene.m_GI.m_bEnable || (bSections0 ? iDraw < m_DrawGISec.size() && !m_DrawGISec[iDraw].empty() : pMesh->m_pGI && GI.second != 0xFFFFFFFFu));     // (the GI off: every one - the screen buffer brings the ambient occlusion)
                // a blended draw (glass) reads the probe grid - 8 probes, O(1) - not the surfels a pixel: no prepass under it, each layer paid
                // the walk (Bistro: 22 of its 33 ms scene pass). Frostbite's GIBS lights what is outside its deferred path from a probe volume,
                // Lumen's translucency from a volume too. XLION_GI_BLEND_PIXEL=1: the per-pixel read, as before (measurement)
                static const bool s_bBlendPixel = xlionrender::Env("XLION_GI_BLEND_PIXEL") != nullptr;
                // (BLENDED only: a masked one - the ivy, its alpha test - read the street's probes, flat pale patches on the walls, Tomas)
                const bool  bBlendProbes = !s_bBlendPixel && bRigid && Probes && pMat && MaterialVisibilityOf(pMat->m_State).m_Opacity == material_visibility::opacity::BLENDED;
                const bool  bSections = bSections0 && !bScreenSub && !bBlendProbes;
                // measurement (XLION_GPU_TIMES): why a main-view GI draw misses the half-resolution read - GIHALFMISS every 240 frames
                static const bool s_bHalfMiss = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                if( s_bHalfMiss && bMain && bScreen )
                {
                    static int s_n[8] = {}; static std::uint64_t s_Frame = 0;
                    const auto* pSt = pMat ? &pMat->m_State : nullptr;
                    const int Why = bScreenSub ? 0 : !bRigid ? 1 : (pSt && pSt->m_Blend != material_state::blend::OFF) ? 2 : (pSt && pSt->m_bDoubleSided) ? 3
                                  : (pSt && (!pSt->m_bDepthWrite || !pSt->m_bDepthTest || pSt->m_bWireframe)) ? 4 : (bSections0 ? 5 : 6);
                    ++s_n[Why];
                    if( m_Device.getFrameIndex() != s_Frame && m_Device.getFrameIndex() % 240 == 0 )
                    {
                        std::printf("GIHALFMISS draws: half-res %d; missed: skinned %d, blended %d, double-sided %d, depth/wire %d, sections without GI %d, no GI buffer %d\n", s_n[0], s_n[1], s_n[2], s_n[3], s_n[4], s_n[5], s_n[6]);
                        std::fflush(stdout); for( auto& v : s_n ) v = 0;
                    }
                    s_Frame = m_Device.getFrameIndex();
                }
                // a rigid mesh without surfels of its own reads the probe grid as well (its GI buffer: the grid; its level: gi_probes_v)
                const bool  bRigidProbes = bRigid && Probes && (!pMesh->m_pGI || bBlendProbes);
                return [=, this, &View]( std::uint32_t c, std::uint32_t Slot, std::uint32_t MaterialBase ) -> std::pair<xgpu::buffer*, details::push_static_geom>
                {
                    if( bScreenSub ) return { &View.m_GIScreenBuffer, { c, Slot, 0u, 0xFFFFFFFDu, MaterialBase } };      // (xlionrender_surfel_gi.glsl gi_screen_v)
                    if( bSections )         // GI sections: the cluster's section - its buffer, its place in the light ring, its vertices
                    {
                        const std::uint16_t k = pMesh->m_GIClusterSection[c];
                        const auto SGI = k < m_DrawGISec[iDraw].size() ? m_DrawGISec[iDraw][k] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };
                        return { const_cast<xgpu::buffer*>(&pMesh->m_GISections[k].m_Buffer), { c, Slot, SGI.first, SGI.second, MaterialBase, pMesh->m_GIClusterVertexDelta[c] } };
                    }
                    xgpu::buffer* const pBuf = bRigidProbes ? Probes : pMesh->m_pGI ? const_cast<xgpu::buffer*>(&pMesh->m_GIBuffer) : const_cast<xgpu::buffer*>(&m_GIDummy);
                    return { pBuf, { c, Slot, GI.first, bRigidProbes ? 0xFFFFFFFEu : GI.second, MaterialBase } };     // (0xFFFFFFFE: xlionrender_surfel_gi.glsl gi_probes_v)
                };
            };

            if( pPre ) { Window.WriteTimestamp(*pPre, 2); s_PreCpuMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - PreCpu0).count(); }
            // GPU culling (features/cull, documentation/gpu_cluster_culling.md): the main view's opaque rigid clusters with a material, a
            // candidate each, in runs (a bind state each; a prepass run a geometry), culled here - outside the pass - into their runs' commands
            static const bool s_bNoGPUCull     = xlionrender::Env("XLION_NO_GPU_CULL") != nullptr;
            static const bool s_bNoGPUOcclusion = xlionrender::Env("XLION_NO_GPU_OCCLUSION") != nullptr;
            m_CullCandidates.clear(); m_CullMatrices.clear(); m_CullRuns.clear(); m_CullRunOf.clear(); m_CullSub.clear();
            if( bMain ) m_LODReport = {};
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
                    // Measurement only (XLION_MAIN_LOD=k): every rigid geometry's LOD k in the main view (its GI not mapped: the upper bound of
                    // the main view's LODs, shadows_gpu_plan.md)
                    static const int s_MainLOD = [] { const char* p = xlionrender::Env("XLION_MAIN_LOD"); return p ? std::atoi(p) : 0; }();
                    // The LOD a submesh (a LOD group) draws: the coarsest whose error projects under XLION_LOD_PIXELS (1) pixels at the group's
                    // box (its nearest point; inside it: LOD 0) - groups meet crack-free at any mix (xgeom_static's locked group borders).
                    // XLION_LOD_PIXELS=0: LOD 0
                    static const float s_LODPixels = [] { const char* p = xlionrender::Env("XLION_LOD_PIXELS"); return p ? float(std::atof(p)) : 1.0f; }();
                    const bool  bPick    = s_MainLOD == 0 && s_LODPixels > 0 && G.m_LODSubmeshes.size() > 1 && G.m_SubBoxes.size() == G.m_Submeshes.size() && m_LODProjScale > 0;
                    const float L2WScale = bPick ? std::max({ (D.m_L2W * xmath::fvec3{ 1, 0, 0 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length(), (D.m_L2W * xmath::fvec3{ 0, 1, 0 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length()
                                                            , (D.m_L2W * xmath::fvec3{ 0, 0, 1 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length() }) : 1.0f;
                    for( std::size_t s = 0; s < G.m_Submeshes.size(); ++s )
                    {
                        int Lod = s_MainLOD;
                        if( bPick )
                        {
                            const auto B = G.m_SubBoxes[s].TransformCopy(D.m_L2W);
                            const auto Near = xmath::fvec3::Max(B.m_Min, xmath::fvec3::Min(Eye, B.m_Max));
                            Lod = G.PickLOD(s, (Near - Eye).Length(), L2WScale, m_LODProjScale, s_LODPixels);
                            ++m_LODReport.m_nGroups[std::size_t(std::min(Lod, 3))];
                            m_LODReport.m_nLODs = std::max(m_LODReport.m_nLODs, int(G.m_LODErrors.size()));
                            for( std::size_t k = 1; k < 4 && k < G.m_LODErrors.size(); ++k ) if( s < G.m_LODErrors[k].size() ) m_LODReport.m_MinError[k] = std::min(m_LODReport.m_MinError[k], G.m_LODErrors[k][s]);
                        }
                        const auto& Sub  = Lod > 0 ? G.LODSubmeshes(Lod)[s] : G.m_Submeshes[s];
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
                            const auto [pBuf, Rec] = GIOf(c, Slot, bTile ? TileBase(pMat) : pMat->m_ParamBase);
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
                    static const bool s_bCullTimes = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                    if( s_bCullTimes && m_Device.getFrameIndex() % 120 == 0 )
                        std::printf("CULLGPU pass 1 cull %.3f, pyramid + pass 2 cull %.3f ms; occlusion %s (it hid %.0f%% of the triangles it tested); %d clusters, %lld triangles shaded\n"
                                   , m_pCull->m_CullGpuMs, m_pCull->m_OccludeGpuMs, m_pCull->isOcclusionSkipped() ? "skipped" : "on", 100.0 * m_pCull->m_OccludedShare, m_pCull->m_Drawn, (long long)m_pCull->m_Triangles), std::fflush(stdout);
                    if( s_bCullTimes && m_Device.getFrameIndex() % 120 == 0 ) std::printf("LODGROUPS the main view's LOD groups by their pick: LOD 0 %d, 1 %d, 2 %d, 3+ %d; %d LODs, the least error at LOD 1 %g, 2 %g, 3 %g\n", m_LODReport.m_nGroups[0], m_LODReport.m_nGroups[1], m_LODReport.m_nGroups[2], m_LODReport.m_nGroups[3], m_LODReport.m_nLODs, m_LODReport.m_MinError[1], m_LODReport.m_MinError[2], m_LODReport.m_MinError[3]), std::fflush(stdout);
                    m_CullFrozen = { true, m_CullRuns.size(), CullSecond, bOcclude };
                }
            }
            const auto CulledOnGPU = [&]( std::size_t iDraw, const cluster_geom& G, const cluster_geom::submesh& Sub )
            {
                return m_CullSubBase[iDraw] != ~0u && m_CullSub[m_CullSubBase[iDraw] + std::size_t(&Sub - G.m_Submeshes.data())] != 0;
            };

            if( pPre ) Window.WriteTimestamp(*pPre, 3);
            // The screen tail (light_mode::SCREEN): the thin G-buffer, the tail pass and the denoiser, before the scene pass reads them
            if( bMain && pLight && (m_LightModeUsed == light_mode::SCREEN || Scene.m_ReflectionMaxRoughness > 0 || m_ReflTraceBench > 0 || ReflectionScreen(Scene)) ) renderTail(Window, Scene, View, W2C, Eye, *pLight, bPlan);     // (reflections: its G-buffer)
            else if( bMain ) m_TailReport.m_bOn = false;
            if( pPre ) Window.WriteTimestamp(*pPre, 4);

            // The GI gather (bScreen), between the G-buffer prepass and the scene pass: at half size, the classification (each pixel's GI
            // buffer as its depth), then a full-screen pass a GI buffer tested EQUAL - into the buffer the scene pass upsamples
            const auto GatherGI = [&]
            {
                // Measurement only (XLION_GPU_TIMES): its start, the classification's end, its end - SCREENSPLIT as the geometry read's (the
                // tools wait for its "GI read")
                static const bool s_bGatherSplit = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                static std::unordered_map<const void*, xgpu::timestamps> s_GatherSplit;      // ponytail: per renderer, never erased (a measurement hook)
                xgpu::timestamps* pGS = nullptr;
                if( s_bGatherSplit )
                {
                    auto [It, bNew] = s_GatherSplit.try_emplace(this);
                    if( bNew && m_Device.Create(It->second, { .m_Count = 4 }) ) s_GatherSplit.erase(It);
                    else pGS = &It->second;
                }
                if( pGS )
                {
                    static std::vector<std::uint64_t> s_Ns;
                    static double s_AOms = 0, s_Cls = 0, s_Read = 0; static int s_n = 0;
                    if( pGS->getResults(s_Ns) && s_Ns.size() >= 4 && s_Ns[3] >= s_Ns[2] && s_Ns[2] >= s_Ns[1] && s_Ns[1] >= s_Ns[0] )
                    {
                        s_AOms += double(s_Ns[1] - s_Ns[0]) * 1e-6; s_Cls += double(s_Ns[2] - s_Ns[1]) * 1e-6; s_Read += double(s_Ns[3] - s_Ns[2]) * 1e-6;
                        if( ++s_n == 120 ) { std::printf("SCREENSPLIT gather: AO %.3f, classification %.3f, GI read %.3f ms (%d passes)\n", s_AOms / s_n, s_Cls / s_n, s_Read / s_n, int(m_GatherBuffers.size())); std::fflush(stdout); s_n = 0; s_AOms = s_Cls = s_Read = 0; }
                    }
                }
                // The ambient occlusion first (xlionrender_ao_comp.glsl): from the G-buffer, on the gather's grid, an "AO only" entry into
                // each pixel of the GI screen buffer - the gather multiplies its irradiance by the AO where it has GI, the materials' upsample
                // takes the AO entries' (mb_standard_pbr.frag SurfelGIScreen): the GI on or off, no extra read. Last frame's: the history
                if( pGS ) Window.WriteTimestamp(*pGS, 0);
                static const bool s_bSSGI = xlionrender::Env("XLION_SSGI") != nullptr;
                static const float s_AORadius = [] { const char* p = xlionrender::Env("XLION_AO_RADIUS"); return p ? float(std::atof(p)) : 0.0f; }();      // (measurement: the AO's and the screen GI's range)      // (prototype, A32: screen GI on the AO's bitmask - xLION_Docs ssgi_bitmask_plan_2026-11-06.md)
                if( bAO )
                {
                    std::swap(View.m_AO, View.m_AOPrev);
                    const float* pM = reinterpret_cast<const float*>(&W2C);      // (column major: the clip y row's scale, P11 - the view's rotation keeps it)
                    const float  P11 = std::sqrt(pM[1] * pM[1] + pM[5] * pM[5] + pM[9] * pM[9]);
                    // (two dispatches: the raw AO, then its 4 x 4 filter - the pattern's rotations summed: each frame good on its own - and a third
                    // of the way from the history)
                    details::push_ao Push{ std::uint32_t(View.m_GIScreenWidth), std::uint32_t(View.m_GIScreenHeight), std::uint32_t(GIScreenDivisor()), 0u
                                         , s_AORadius > 0.0f ? s_AORadius : Scene.m_GI.m_AORadius, Scene.m_GI.m_AOThickness, 0.5f * float(View.m_GIScreenHeight) * P11, m_GatherTableBase, 0u, 0.33f };     // (no history: each frame good on its own - HypeHype's; moving looks as still)
                    auto Buffers  = std::array<xgpu::buffer*, 5>{ &View.m_GIScreenBuffer, &View.m_GIGatherTable, &View.m_AOPrev, &View.m_AO, &View.m_AORaw };
                    auto Textures = std::array<const xgpu::texture*, 4>{ &View.m_Depth, &View.m_GPreNormal, &View.m_GPreId, View.m_bHDRPrev ? &View.m_HDRPrev : &View.m_HDR };     // (last frame's lit image, XLION_SSGI's light)
                    const int GX = (View.m_GIScreenWidth + 7) / 8, GY = (View.m_GIScreenHeight + 7) / 8;
                    if( s_bSSGI ) Push.m_Mode = 4u;
                    Window.Dispatch(m_AOCompute, Buffers, Textures, &Push, sizeof(Push), GX, GY);
                    Push.m_Mode = s_bSSGI ? 5u : 1u;
                    Window.Dispatch(m_AOCompute, Buffers, Textures, &Push, sizeof(Push), GX, GY);
                }
                View.m_bAOPrev = bAO;
                // The published light (XLION_VOX_PUBLISHED, gi_published_store_plan_2026-11-08.md): last frame's requests resolved, then this
                // frame's lists, bits and statistics cleared for the gather - only in the voxels' mode (word 20 3), never with the ground truth
                const bool bVoxPub = m_GatherTable.size() > 38 && m_GatherTable[37] != 0u && m_GatherTable[20] == 3u && m_pGatherClip && bMain && View.m_GIScreenWidth * View.m_GIScreenHeight >= 16384;      // (the main view only: one set of lists)
                if( bVoxPub )
                {
                    constexpr int pub_head_v = 4, pub_bits_v = 16384, pub_cap_v = 1 << 19, pub_slots_v = 1 << 21;
                    constexpr int pub_entries_v = pub_head_v + pub_bits_v + pub_slots_v + 2 * pub_cap_v;
                    if( !m_VoxPub.m_Private )
                    {
                        std::vector<std::uint32_t> Z(std::size_t(pub_entries_v) * 4, 0u);
                        (void)m_Device.Create(m_VoxPub, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = pub_entries_v, .m_pData = Z.data(), .m_pName = "GI: the published voxel light" });
                        (void)m_Device.Create(m_VoxPubStats, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = 4 * ring_frames_v, .m_pName = "GI: the published light's statistics" });
                        (void)m_Device.Create(m_VoxPubTimes, { .m_Count = 2 });
                        (void)m_Device.Create(m_VoxPubFrameB, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = ring_frames_v, .m_pName = "GI: the published light's frame" });
                        std::printf("VOXPUB buffer %.1f MB\n", double(pub_entries_v) * 16.0 / (1024.0 * 1024.0)); std::fflush(stdout);
                    }
                    if( m_VoxPub.m_Private )
                    {
                        const std::uint32_t Frame = m_GatherTable[38], Prev = (Frame - 1u) & 1u;
                        m_VoxPubFrame = Frame;
                        static const bool s_bPubLog = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                        if( s_bPubLog )
                        {
                            const int Slot = int(m_Device.getFrameIndex() % ring_frames_v);
                            std::uint32_t St[4] = {}, H[16] = {};
                            (void)m_VoxPubStats.MemoryMap(4 * Slot, 4, [&](void* p) { std::memcpy(H, p, sizeof(H)); std::memcpy(St, H + 8, sizeof(St)); });
                            if( std::vector<std::uint64_t> Ns; m_VoxPubTimes.getResults(Ns) && Ns.size() >= 2 && Ns[1] > Ns[0] ) { m_VoxPubMs += double(Ns[1] - Ns[0]) * 1e-6; ++m_VoxPubN; }
                            if( m_VoxPubN >= 120 ) { std::printf("VOXPUB resolve %.3f ms; a frame: pixels published %u, fallen back %u, requests dropped %u\n", m_VoxPubMs / m_VoxPubN, St[0], St[1], St[2]); std::printf("VOXPUBHEAD counts %u %u frame %u parity %u (cpu frame %u)\n", H[0], H[4], H[12], H[13], Frame); std::fflush(stdout); m_VoxPubMs = 0; m_VoxPubN = 0; }
                            Window.WriteTimestamp(m_VoxPubTimes, 0);
                        }
                        const std::uint32_t PPush[4] = { m_GatherTableBase, Frame, Prev, 0u };
                        auto PB = std::array<xgpu::buffer*, 3>{ m_pGatherClip, &View.m_GIGatherTable, &m_VoxPub };
                        Window.Dispatch(m_GIPublishCompute, PB, PPush, sizeof(PPush), pub_cap_v / 64);
                        if( s_bPubLog ) Window.WriteTimestamp(m_VoxPubTimes, 1);
                        const std::array<xgpu::buffer::copy, 3> Zc{ xgpu::buffer::copy{ 0, int(Prev), 1 }, xgpu::buffer::copy{ 0, 2, 1 }, xgpu::buffer::copy{ 0, pub_head_v, pub_bits_v } };
                        Window.CopyBuffer(View.m_GIScreenZeros, m_VoxPub, std::span{ Zc });
                        // (the gather's frame and parity: in the buffer itself, [3] - the table's word reached the gather a frame late or early: the
                        // lists' parity never matched the resolve's, every request dropped)
                        const int FSlot = int(m_Device.getFrameIndex() % ring_frames_v);
                        (void)m_VoxPubFrameB.MemoryMap(FSlot, 1, [&](void* p) { const std::uint32_t F[4] = { Frame, Frame & 1u, 0u, 0u }; std::memcpy(p, F, sizeof(F)); });
                        const xgpu::buffer::copy Fc{ FSlot, 3, 1 };
                        Window.CopyBuffer(m_VoxPubFrameB, m_VoxPub, std::span{ &Fc, 1 });
                    }
                }
                // XLION_HEAT (measurement, 2026-11-08): the gather's clock a pixel, shown as colours over a grey image (xlionrender_heat_comp.glsl), a HEAT
                // line with its mean every 120 frames; its buffers made the first time (the gather built timed: instance.cpp)
                static const bool s_bHeat = xlionrender::Env("XLION_HEAT") != nullptr;
                const int HeatW = View.m_GIScreenWidth, HeatH = View.m_GIScreenHeight;
                if( s_bHeat && !View.m_HeatBuffer.m_Private )
                {
                    const std::vector<std::uint32_t> Z(4 + std::size_t(HeatW) * HeatH + 3, 0u);
                    bool bOk = !m_Device.Create(View.m_HeatBuffer, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = int(Z.size() / 4), .m_pData = const_cast<std::uint32_t*>(Z.data()), .m_pName = "Heat: the gather's clock a pixel" })
                            && !m_Device.Create(View.m_HeatStats, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = ring_frames_v, .m_pName = "Heat: the header a frame" })
                            && !m_Device.Create(View.m_HeatTex, { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = HeatW, .m_Height = HeatH, .m_isGamma = false, .m_bStorage = true, .m_pName = "Heat: colours" });
                    if( bOk ) { auto Bindings = std::array{ xgpu::pipeline_instance::sampler_binding{ View.m_HDR }, xgpu::pipeline_instance::sampler_binding{ View.m_HeatTex }, xgpu::pipeline_instance::sampler_binding{ View.m_ExposureTex } };
                                bOk = !m_Device.Create(View.m_HeatToneMap, { .m_PipeLine = m_ToneMapPipeline, .m_SamplersBindings = Bindings }); }
                    if( !bOk ) std::printf("HEAT: its buffers could not be made\n");
                }
                const bool bHeat = s_bHeat && View.m_HeatToneMap.m_Private;
                if( bHeat ) { const xgpu::buffer::copy Hc{ 0, 0, 1 }; Window.CopyBuffer(View.m_GIScreenZeros, View.m_HeatBuffer, std::span{ &Hc, 1 }); }
                {
                    auto GCmd = Window.StartRenderPass(View.m_GIScreenPass);
                    if( pGS ) GCmd.WriteTimestamp(*pGS, 1);
                    const details::push_gi_gather Push{ m_GatherTableBase, 0u, std::uint32_t(GIScreenDivisor()), bAO ? (s_bSSGI ? 2u : 1u) : 0u };
                    std::array<xgpu::buffer*, 13> Buffers{ &m_InstanceBuffer, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &View.m_GIScreenBuffer, &View.m_GIScreenPrev, &View.m_GIGatherTable, &View.m_AO, m_pGatherClip ? m_pGatherClip : &m_GIDummy
                                                         , m_pGatherRefS ? m_pGatherRefS : &m_GIDummy, m_pGatherRefA ? m_pGatherRefA : &m_GIDummy, bVoxPub && m_VoxPub.m_Private ? &m_VoxPub : &m_GIDummy
                                                         , bHeat ? &View.m_HeatBuffer : &m_GIDummy };
                    GCmd.setPipelineInstance(View.m_GIClassifyInstance, std::span{ Buffers });
                    GCmd.setPushConstants(Push);
                    GCmd.setBuffer(m_FullscreenVB);
                    GCmd.setBuffer(m_FullscreenIB);
                    GCmd.Draw(3);
                    if( pGS ) GCmd.WriteTimestamp(*pGS, 2);
                    for( std::size_t k = 0; k < m_GatherBuffers.size(); ++k )
                    {
                        Buffers[1] = const_cast<xgpu::buffer*>(m_GatherBuffers[k]);
                        GCmd.setPipelineInstance(View.m_GIGatherInstance, std::span{ Buffers });
                        GCmd.setPushConstants(details::push_gi_gather{ m_GatherTableBase, std::uint32_t(k), Push.m_Div, Push.m_AO });
                        GCmd.setBuffer(m_FullscreenVB);
                        GCmd.setBuffer(m_FullscreenIB);
                        GCmd.Draw(3);
                    }
                    if( pGS ) GCmd.WriteTimestamp(*pGS, 3);
                }
                if( bHeat )
                {
                    const int Slot = int(m_Device.getFrameIndex() % ring_frames_v);
                    std::uint32_t Hd[4] = {};
                    (void)View.m_HeatStats.MemoryMap(Slot, 1, [&](void* p) { std::memcpy(Hd, p, sizeof(Hd)); });      // (ring_frames_v frames ago's)
                    if( Hd[1] ) { View.m_HeatSum += double(Hd[0]) * 256.0 / Hd[1]; View.m_HeatN += Hd[1]; }
                    if( Hd[2] ) { View.m_HeatLanes += double(Hd[3]) / Hd[2]; View.m_HeatGroups += Hd[2]; }
                    if( ++View.m_HeatFrames == 120 ) { std::printf("HEAT gather: mean %.0f cycles a pixel (its subgroup's), %.0f pixels a frame; %.0f subgroups a frame, %.1f pixels each\n", View.m_HeatSum / 120.0, View.m_HeatN / 120.0, View.m_HeatGroups / 120.0, View.m_HeatLanes / 120.0); std::fflush(stdout); View.m_HeatSum = View.m_HeatN = View.m_HeatLanes = View.m_HeatGroups = 0.0; View.m_HeatFrames = 0; }
                    const std::uint32_t HPush[4] = { std::uint32_t(HeatW), std::uint32_t(HeatH), 0u, 0u };
                    auto HB = std::array<xgpu::buffer*, 1>{ &View.m_HeatBuffer }; const xgpu::texture* HI[] = { &View.m_HeatTex };
                    Window.Dispatch(m_HeatCompute, HB, std::span<const xgpu::texture* const>{}, HI, HPush, sizeof(HPush), (HeatW + 7) / 8, (HeatH + 7) / 8);
                    const xgpu::buffer::copy Hc{ 0, Slot, 1 };
                    Window.CopyBuffer(View.m_HeatBuffer, View.m_HeatStats, std::span{ &Hc, 1 });
                }
                // The voxels' read cleaned on the screen (XLION_GI_DENOISE="<strides as cells> <largest stride>"; two passes, the second on the first)
                static const std::array<float, 2> s_Denoise = [] { std::array<float, 2> A{ 0.0f, 32.0f }; if( const char* p = xlionrender::Env("XLION_GI_DENOISE") ) (void)xlionrender::Scan(p, "%f %f", &A[0], &A[1]); return A; }();
                if( s_Denoise[0] > 0.0f && m_bVoxOn && !m_pGatherRefS && View.m_GIScreenFilter.m_Private )
                {
                    const float* pM = reinterpret_cast<const float*>(&W2C);
                    const float  P11 = std::sqrt(pM[1] * pM[1] + pM[5] * pM[5] + pM[9] * pM[9]);
                    struct { int m_W, m_H, m_TableBase, m_bAO; float m_CellPixels, m_MaxStride, m_Mult, m_Pad; } DPush{ View.m_GIScreenWidth, View.m_GIScreenHeight, int(m_GatherTableBase), bAO ? 1 : 0
                        , gi::gpu_trace::VoxConfig()[0] * s_Denoise[0] * 0.5f * float(View.m_GIScreenHeight) * P11, s_Denoise[1], 0.5f, 0.0f };
                    const int GX = (View.m_GIScreenWidth + 7) / 8, GY = (View.m_GIScreenHeight + 7) / 8;
                    { auto Bf = std::array<xgpu::buffer*, 4>{ &View.m_GIScreenBuffer, &View.m_GIScreenFilter, &View.m_GIGatherTable, &View.m_AO }; Window.Dispatch(m_GIDenoiseCompute, Bf, std::span<const xgpu::texture* const>{}, std::span<const xgpu::texture* const>{}, &DPush, sizeof(DPush), GX, GY); }
                    DPush.m_Mult = 1.0f;
                    { auto Bf = std::array<xgpu::buffer*, 4>{ &View.m_GIScreenFilter, &View.m_GIScreenBuffer, &View.m_GIGatherTable, &View.m_AO }; Window.Dispatch(m_GIDenoiseCompute, Bf, std::span<const xgpu::texture* const>{}, std::span<const xgpu::texture* const>{}, &DPush, sizeof(DPush), GX, GY); }
                }
                if( bVoxPub && m_VoxPubStats.m_Private ) { const xgpu::buffer::copy Cs{ 0, 4 * int(m_Device.getFrameIndex() % ring_frames_v), 4 }; Window.CopyBuffer(m_VoxPub, m_VoxPubStats, std::span{ &Cs, 1 }); }
                const xgpu::buffer::copy C{ 6, int(m_Device.getFrameIndex() % ring_frames_v), 1 };     // (its counters into this frame's slot, as the geometry read's)
                Window.CopyBuffer(View.m_GIScreenBuffer, View.m_GIScreenCounters, std::span{ &C, 1 });
            };

            // The sun's shadow mask (XLION_SUN_MASK; S1a measurement): between the G-buffer prepass and the scene pass, the sun's visibility
            // per pixel from the prepass (shaders/xlionrender_sun_mask_frag.glsl); mode 2: the materials skip their own lookup
            // Mode 6 (the reuse) runs it only while the view holds still - the sun's parameters (the key) and the eye as last frame's: the
            // first such frame computes it all, the next ones keep what still holds; a moving frame shades inline (the pass costs more than
            // it saves when nothing can be kept) and leaves no mask to keep
            const bool bSunMaskOk = bGPre && m_SunMaskMode > 0 && pLight && pLight->m_VSMParams.m_X != 0.0f && View.m_SunMaskInstance.m_Private;
            std::vector<float> SunKey;
            if( bSunMaskOk && m_SunMaskMode >= 6 )
            {
                const auto Add = [&](const float* p, int n) { SunKey.insert(SunKey.end(), p, p + n); };
                const auto& L = *pLight;
                Add(reinterpret_cast<const float*>(&L.m_W2Light), 16);
                Add(&L.m_VSMParams.m_X, 3); Add(&L.m_VSMParams2.m_X, 3); Add(&L.m_VSMParams3.m_X, 3); Add(&L.m_VSMPrevParams.m_X, 1);
                for( const auto& Lv : L.m_VSMLevels ) Add(&Lv.m_Z, 2);         // (the windows' origins move with the eye: the pages stay)
            }
            const bool bSunStill = SunKey == View.m_SunMaskPrevKey && View.m_SunMaskPrevW == View.m_Width && View.m_SunMaskPrevH == View.m_Height
                                && Eye.m_X == View.m_SunMaskPrevEye.m_X && Eye.m_Y == View.m_SunMaskPrevEye.m_Y && Eye.m_Z == View.m_SunMaskPrevEye.m_Z;
            const bool bSunMask = bSunMaskOk && (m_SunMaskMode < 6 || bSunStill);
            if( m_SunMaskMode >= 6 )
            {
                if( !bSunMask ) View.m_bSunMaskValid = false;                   // (nothing kept next frame: it computes it all)
                View.m_SunMaskPrevKey = SunKey; View.m_SunMaskPrevEye = Eye; View.m_SunMaskPrevW = View.m_Width; View.m_SunMaskPrevH = View.m_Height;
            }
            if( bSunMask )
            {
                pLight->m_TailC2W = W2C.Inverse();
                pLight->m_ShadowLevels[0].m_Y = m_SunMaskMode == 5 ? 2.0f : m_SunMaskMode == 4 ? 1.0f : 0.0f;       // (the mask pass's early-out; 5: the test's view of it)
                if( m_SunMaskMode == 2 ) pLight->m_VSMParams.m_X = 0.0f;
            }
            const auto SunMask = [&]
            {
                m_LightUBO.allocEntry<details::ubo_lighting>() = *pLight;     // (the scene's block: other passes took entries since - the pass binds the latest)
                {
                    auto MCmd = Window.StartRenderPass(View.m_SunMaskPass);
                    MCmd.WriteTimestamp(m_SunMaskTimes, 0);          // (in the pass's own command buffer: the window's wrote them around nothing)
                    MCmd.setViewport(0.0f, 0.0f, float(View.m_Width), float(View.m_Height));
                    MCmd.setScissor(0, 0, View.m_Width, View.m_Height);
                    // the reuse (mode 6, a still view with last frame's mask): how much may be kept - 3 nothing about the pages changed (none
                    // rendered, evicted, dirtied or re-centred: every pixel at the same surface), 2 none gone (a page not in the table was not
                    // last frame either: only the ones rendered now change), 0 nothing (the first still frame)
                    const auto ReuseLevel = [&]() -> std::uint32_t
                    {
                        if( m_SunMaskMode < 6 || !View.m_bSunMaskValid ) return 0u;
                        const auto& St = m_VSM.m_Stats;
                        if( St.m_Evicted || St.m_Dirtied || St.m_Recentred ) return 1u;
                        return m_VSM.m_ToRender.empty() ? 3u : 2u;
                    };
                    struct { xmath::fmat4 m_Reproject; xmath::fvec4 m_Delta; std::uint32_t m_Params[4]; } Push
                    { View.m_SunMaskPrevW2C * xmath::fmat4::fromTranslation(Eye - View.m_SunMaskPrevEye), xmath::fvec4(Eye - View.m_SunMaskPrevEye, 0.0f)
                    , { ReuseLevel(), std::uint32_t(m_Device.getFrameIndex()), std::uint32_t(m_VSM.getSetup().Slots()), m_SunMaskMode == 7 ? 1u : 0u } };
                    std::array<xgpu::buffer*, 3> Buffers{ &m_LightRing, &m_CellRing, &m_VSMBounds };
                    MCmd.setPipelineInstance(View.m_SunMaskInstance, Buffers);
                    MCmd.setDynamicUBO(m_LightUBO, 1);
                    MCmd.setBuffer(m_FullscreenVB);
                    MCmd.setBuffer(m_FullscreenIB);
                    MCmd.setPushConstants(Push);          // (after the pipeline's bind)
                    MCmd.Draw(3);
                    MCmd.WriteTimestamp(m_SunMaskTimes, 1);
                }
                Window.CopyTexture(m_SunMaskNew, m_SunMaskTex, View.m_Width, View.m_Height);      // (the materials read it there; next frame's pass reads it as last frame's)
                View.m_SunMaskPrevW2C = W2C; View.m_bSunMaskValid = true;
                // mode 3: the materials read it - the flag in the scene's block itself (the light cuts' per draw entries are copies of it) and
                // that block again (the next binds take the latest entry). A lit pass before this one in the frame sees the flag too: its
                // fragments are not at the mask's distances (the check) or are last frame's (a still camera). A debug view of the pages
                // shades inline: ShadowVSM's colours
                if( m_SunMaskMode >= 3 && pLight->m_VSMParams2.m_W < 2.0f )
                {
                    pLight->m_ShadowLevels[0].m_X = 1.0f;
                    m_LightUBO.allocEntry<details::ubo_lighting>() = *pLight;
                }
                if( std::vector<std::uint64_t> Ns; m_SunMaskTimes.getResults(Ns) && Ns.size() >= 2 && Ns[1] >= Ns[0] && Ns[0] != ~0ull ) { m_SunMaskSum += double(Ns[1] - Ns[0]) * 1e-6; ++m_SunMaskSumN; }
                static const bool s_bTimes = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                if( s_bTimes && m_SunMaskSumN >= 120 ) { std::printf("SUNMASKGPU %d x %d: %.3f ms (mode %d)\n", View.m_Width, View.m_Height, m_SunMaskSum / m_SunMaskSumN, m_SunMaskMode); std::fflush(stdout); m_SunMaskSum = 0; m_SunMaskSumN = 0; }
            };

            std::optional<xgpu::cmd_buffer> CmdPass;     // (GPU culling's occlusion ends the pass for its Hi-Z and loads it again: the same Cmd)
            CmdPass.emplace(Window.StartRenderPass(bGPre ? View.m_GPrePass : View.m_ScenePass));
            auto& Cmd = *CmdPass;
            if( bMain ) Cmd.WriteTimestamp(m_Timestamps, 1);
            // Measurement only (XLION_GPU_TIMES): the main view's scene pass split - its start, the depth prepass's end, its end (SCENESPLIT)
            static const bool s_bSplit = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
            static std::unordered_map<const void*, xgpu::timestamps> s_Split;      // ponytail: per renderer, never erased (a measurement hook)
            xgpu::timestamps* pSplit = nullptr;
            if( s_bSplit && bMain )
            {
                auto [It, bNew] = s_Split.try_emplace(this);
                if( bNew && m_Device.Create(It->second, { .m_Count = 6 }) ) s_Split.erase(It);
                else pSplit = &It->second;
            }
            if( pSplit )
            {
                static std::vector<std::uint64_t> s_Ns;
                static double s_Pre = 0, s_Shade = 0, s_Blend = 0; static int s_n = 0;
                if( pSplit->getResults(s_Ns) && s_Ns.size() >= 4 && s_Ns[2] >= s_Ns[3] && s_Ns[3] >= s_Ns[1] && s_Ns[1] > s_Ns[0] )
                {
                    s_Pre += double(s_Ns[1] - s_Ns[0]) * 1e-6; s_Shade += double(s_Ns[3] - s_Ns[1]) * 1e-6; s_Blend += double(s_Ns[2] - s_Ns[3]) * 1e-6;
                    if( ++s_n == 120 ) { std::printf("SCENESPLIT prepass %.3f, shading + sky %.3f, blended %.3f ms\n", s_Pre / s_n, s_Shade / s_n, s_Blend / s_n); std::fflush(stdout); s_n = 0; s_Pre = s_Shade = s_Blend = 0; }
                    static double s_Depth = 0, s_Screen = 0, s_Tile = 0; static int s_m = 0;
                    if( s_Ns.size() >= 6 && s_Ns[4] > s_Ns[0] && s_Ns[5] >= s_Ns[4] && s_Ns[1] >= s_Ns[5] )
                    {
                        s_Depth += double(s_Ns[4] - s_Ns[0]) * 1e-6; s_Screen += double(s_Ns[5] - s_Ns[4]) * 1e-6; s_Tile += double(s_Ns[1] - s_Ns[5]) * 1e-6;
                        if( ++s_m == 120 ) { std::printf("PREPASSSPLIT depth and attributes %.3f, gather + mask + contact %.3f, tile pass %.3f ms\n", s_Depth / s_m, s_Screen / s_m, s_Tile / s_m); std::fflush(stdout); s_m = 0; s_Depth = s_Screen = s_Tile = 0; }
                    }
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
            // The material tile pass's prototype (XLION_TILE): the opaque texture-table materials shaded full-screen from the prepass, before the scene pass; that pass no longer draws them
            // The sun's contact shadow (XLION_CONTACT="length steps thickness fade div", world units; xlionrender_contact_comp.glsl): before the shading reads it
            static const std::array<float, 5> s_Contact = [] { std::array<float, 5> A{ 0.0f, 8.0f, 0.15f, 30.0f, 1.0f }; if( const char* p = xlionrender::Env("XLION_CONTACT") ) (void)xlionrender::Scan(p, "%f %f %f %f %f", &A[0], &A[1], &A[2], &A[3], &A[4]); return A; }();
            const bool bContact = s_Contact[0] > 0.0f && bGPre && bScreen && pLight && m_ContactTex.m_Private && View.m_Width <= sun_mask_w_v && View.m_Height <= sun_mask_h_v && Scene.m_Sun.m_bCastShadows;
            const auto ContactPass = [&]
            {
                static const bool s_bTimes = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                static double s_Sum = 0.0; static int s_n = 0;
                if( std::vector<std::uint64_t> Ns; s_bTimes && m_ContactTimes.getResults(Ns) && Ns.size() >= 2 && Ns[1] > Ns[0] )
                {
                    s_Sum += double(Ns[1] - Ns[0]) * 1e-6;
                    if( ++s_n == 120 ) { std::printf("CONTACTMS %.3f ms\n", s_Sum / 120.0); std::fflush(stdout); s_Sum = 0.0; s_n = 0; }
                }
                const std::uint32_t Div = std::uint32_t(s_Contact[4]);
                struct { xmath::fmat4 m_W2C; xmath::fvec4 m_ToSun, m_Params; std::uint32_t m_Size[4]; } Push{ W2C, xmath::fvec4(-Scene.m_Sun.m_Direction.NormalizeCopy(), -1.0f), { s_Contact[0], s_Contact[1], s_Contact[2], s_Contact[3] }, { std::uint32_t(View.m_Width), std::uint32_t(View.m_Height), Div, m_GatherTableBase } };
                auto Buffers  = std::array<xgpu::buffer*, 1>{ &View.m_GIGatherTable };
                auto Textures = std::array<const xgpu::texture*, 1>{ &View.m_Depth };
                auto Images   = std::array<const xgpu::texture*, 1>{ &m_ContactTex };
                const int GX = (View.m_Width / int(Div) + 7) / 8, GY = (View.m_Height / int(Div) + 7) / 8;
                if( s_bTimes ) Window.WriteTimestamp(m_ContactTimes, 0);
                Window.Dispatch(m_ContactCompute, Buffers, Textures, Images, &Push, sizeof(Push), GX, GY);
                if( s_bTimes ) Window.WriteTimestamp(m_ContactTimes, 1);
                pLight->m_ShadowLevels[0].m_Z = s_Contact[4];
                pLight->m_ShadowLevels[0].m_W = s_Contact[0];
                pLight->m_TailW2C = W2C; pLight->m_TailC2W = W2C.Inverse();
                m_LightUBO.allocEntry<details::ubo_lighting>() = *pLight;      // (the scene's block, with its flag)
            };
            const auto TilePass = [&]
            {
                int nTileDraws = 0;
                // A5.3 (XLION_LIGHT_TILES=0: off): the local lights each 8 x 8 tile can see, from the prepass's depth - the tile pass reads them
                // instead of its world cell's list (xlionrender_light_tiles_comp.glsl); the lights' count and base, the mode: this render's block
                static const bool s_bLightTiles = [] { const char* p = xlionrender::Env("XLION_LIGHT_TILES"); return !(p && p[0] == '0'); }();
                const int  nLocal = pLight ? int(pLight->m_LocalCount.m_X) : 0;
                bool bLightTiles = s_bLightTiles && pLight && nLocal > 0 && nLocal < 65536 && pLight->m_LocalCount.m_Y < 1.5f;
                if( bLightTiles )
                {
                    const int TX = (View.m_Width + 7) / 8, TY = (View.m_Height + 7) / 8, nUints = 16 + 16 * TX * TY;
                    if( View.m_LightTiles.m_Private && int(View.m_LightTiles.getEntryCount()) != nUints ) m_Device.Destroy(std::move(View.m_LightTiles));
                    if( !View.m_LightTiles.m_Private )
                    {
                        const std::vector<std::uint32_t> Z(std::size_t(nUints), 0u);
                        (void)m_Device.Create(View.m_LightTiles, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = nUints, .m_pData = const_cast<std::uint32_t*>(Z.data()), .m_pName = "Lights: the screen tiles' lists" });
                    }
                    bLightTiles = View.m_LightTiles.m_Private != nullptr;
                    if( bLightTiles )
                    {
                        struct { xmath::fmat4 m_C2W; std::uint32_t m_W, m_H, m_LightBase, m_nLights; } LPush{ W2C.Inverse(), std::uint32_t(View.m_Width), std::uint32_t(View.m_Height), std::uint32_t(pLight->m_LocalBase.m_X), std::uint32_t(nLocal) };
                        auto LB = std::array<xgpu::buffer*, 2>{ &m_LightRing, &View.m_LightTiles };
                        const xgpu::texture* LT[] = { &View.m_Depth };
                        Window.Dispatch(m_LightTilesCompute, LB, LT, std::span<const xgpu::texture* const>{}, &LPush, sizeof(LPush), TX, TY);
                    }
                }
                // A5.0's light funnel (XLION_TILE_ABLATE bit 8, measurement): the counters zeroed before the pass, copied into a ring after it;
                // a TILECOUNT line every 120 frames (xlionrender_local_lights.glsl LOCAL_COUNT)
                static const bool s_bTileCount = [] { const char* p = xlionrender::Env("XLION_TILE_ABLATE"); return p && (std::atoi(p) & 8) != 0; }();
                if( s_bTileCount && bMain && !m_TileCount.m_Private )
                {
                    const std::uint32_t Z[16] = {};
                    (void)m_Device.Create(m_TileCount, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = 4, .m_pData = const_cast<std::uint32_t*>(Z), .m_pName = "Tile pass: the light funnel's counters" });
                    (void)m_Device.Create(m_TileCountStats, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = 4 * ring_frames_v, .m_pName = "Tile pass: the light funnel a frame" });
                }
                if( m_TileCount.m_Private && bMain )
                {
                    const int Slot = int(m_Device.getFrameIndex() % ring_frames_v);
                    std::uint32_t C[16] = {};
                    (void)m_TileCountStats.MemoryMap(4 * Slot, 4, [&](void* p) { std::memcpy(C, p, sizeof(C)); });
                    if( C[0] ) { for( int k = 0; k < 9; ++k ) m_TileCountSum[k] += C[k]; ++m_TileCountN; }
                    if( m_TileCountN == 120 )
                    {
                        const double* S = m_TileCountSum; const double px = S[0] / 120.0;
                        std::printf("TILECOUNT a frame: %.0f pixels lit by locals (%.0f in a cell); a pixel: %.2f candidates, %.2f in range, %.2f in range and cone, %.2f lit side (shadow entered), %.2f fully shadowed, %.2f contributing; outside the cells %.1f lights walked\n",
                            px, S[1] / 120.0, S[2] / S[0], S[3] / S[0], S[4] / S[0], S[5] / S[0], S[6] / S[0], S[7] / S[0], S[8] / 120.0);
                        std::fflush(stdout); for( auto& v : m_TileCountSum ) v = 0; m_TileCountN = 0;
                    }
                    const xgpu::buffer::copy Zc{ 0, 0, 4 };
                    Window.CopyBuffer(View.m_GIScreenZeros, m_TileCount, std::span{ &Zc, 1 });
                }
                View.m_TileInstances.resize(m_TileGraphs.size());
                std::optional<xgpu::cmd_buffer> C;
                for( std::size_t g = 0; g < m_TileGraphs.size(); ++g )
                {
                    material_slot* pFirst = nullptr;
                    for( auto& M : m_Materials ) if( M.m_bAlive && M.m_TileGraph == int(g) && TileMat(&M) ) { pFirst = &M; break; }
                    if( !pFirst ) continue;
                    static const int s_TileSkip = [] { const char* p = xlionrender::Env("XLION_TILE_SKIP_GRAPH"); return p ? std::atoi(p) : -1; }();      // (measurement, A5 step 0: a graph's draw left out - its pixels unshaded)
                    if( int(g) == s_TileSkip ) continue;
                    auto& G = m_TileGraphs[g];
                    if( !G.m_Pipeline.m_Private )
                    {
                        m_pBuildTile = &G;
                        const auto* pErr = buildMaterialPipeline(G.m_Pipeline, *pFirst, geom_kind::RIGID, false);
                        m_pBuildTile = nullptr;
                        if( pErr ) { G.m_Pipeline = {}; continue; }
                    }
                    auto& Inst = View.m_TileInstances[g];
                    if( !Inst.m_Private )
                    {
                        auto Textures = std::array<xgpu::texture*, 7>{ nullptr, &View.m_Depth, &View.m_GPreNormal, &View.m_GPreId, &View.m_TileAttr0, &View.m_TileAttr1, &View.m_TileColor };
                        auto Bindings = LitBindings(Textures, true);
                        if( m_Device.Create(Inst, { .m_PipeLine = G.m_Pipeline, .m_SamplersBindings = Bindings }) ) continue;
                    }
                    if( !C ) C.emplace(Window.StartRenderPass(View.m_TilePass));
                    std::array<xgpu::buffer*, 12> Buffers{ &m_GIDummy, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &View.m_GIScreenBuffer, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, TailBuffer()
                                                         , m_TileCount.m_Private ? &m_TileCount : &m_GIDummy, bLightTiles ? &View.m_LightTiles : &m_GIDummy };
                    C->setPipelineInstance(Inst, std::span{ Buffers });
                    C->setTextureTable(pFirst->m_Table);      // (bindless: the heaps' set)
                    C->setDynamicUBO(m_LightUBO, 1);
                    C->setDynamicUBO(m_StaticGeomUBO, 0);
                    static const std::uint32_t s_TileAblate = [] { const char* p = xlionrender::Env("XLION_TILE_ABLATE"); return p ? std::uint32_t(std::atoi(p)) & 0xFFFFu : 0u; }();      // (measurement, A5 step 0: mb_standard_pbr.frag TILE_ABLATE)
                    struct { xmath::fmat4 m_C2W; std::uint32_t m_Graph, m_Pad[3]; } Push{ W2C.Inverse(), std::uint32_t(g) | (s_TileAblate << 16), {} };
                    C->setPushConstants(Push);
                    C->setBuffer(m_FullscreenVB);
                    C->setBuffer(m_FullscreenIB);
                    C->Draw(3);
                    ++nTileDraws;
                }
                if( m_TileCount.m_Private && bMain && C ) { C.reset(); const xgpu::buffer::copy Cc{ 0, 4 * int(m_Device.getFrameIndex() % ring_frames_v), 4 }; Window.CopyBuffer(m_TileCount, m_TileCountStats, std::span{ &Cc, 1 }); }
                static const bool s_bTileLog = xlionrender::Env("XLION_GPU_TIMES") != nullptr; static int s_TileN = 0;     // (measurement: the full-screen draws a frame - one a graph)
                if( s_bTileLog && ++s_TileN % 600 == 0 ) { std::printf("TILEPASS %d full-screen draws (graphs) of %zu\n", nTileDraws, m_TileGraphs.size()); std::fflush(stdout); }
            };
            const auto PrepassCulled = [&]( int Pass )
            {
                for( std::size_t r = 0; r < m_CullRuns.size(); ++r )
                {
                    const auto& R = m_CullRuns[r];
                    if( R.m_pInstance ) continue;
                    std::array<xgpu::buffer*, 10> Buffers{ R.m_pGeom->m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, &m_pCull->m_Records };
                    Cmd.setPipelineInstance(bGPre ? m_GPreRecordsInstance : m_GIScreenDepthInstance, std::span{ Buffers });     // (the depth prepass's pipeline reading the records: the G-buffer's, or depth only)
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
                const bool bSkinScreen = bSkin && bScreen && Opaque(pMat);      // (in the G-buffer: the screen buffer brings its GI - the gather's probe read - and its AO)
                xgpu::buffer* const SkinGI = bSkinScreen ? &View.m_GIScreenBuffer : bSkin && Probes ? Probes : &m_GIDummy;
                const auto GIOf = RigidGI(iDraw, pMat);         // (rigid: a cluster's GI buffer and push constants)
                const auto Bind = [&](xgpu::buffer* pGIBuffer)
                {
                    const auto TB = std::chrono::steady_clock::now();
                    struct bind_lap { std::chrono::steady_clock::time_point t; ~bind_lap() { g_BindMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); } } BindLap{ TB };
                    if( !bSkin )
                    {
                        std::array<xgpu::buffer*, 10> GIBuffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing
                                                              , pGIBuffer, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, TailBuffer() };
                        Cmd.setPipelineInstance(*pInstance, std::span{ GIBuffers });     // every rigid pipeline reads the GI
                    }
                    else
                    {
                        std::array<xgpu::buffer*, 12> Static{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_LightRing, &m_CellRing, SkinGI, &m_GILightRing, &m_GIMomentRing, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy, &m_MaterialParams, &m_EntityParams, TailBuffer() };
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
                    if( bSkin ) { Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Slot, bSkinScreen ? 2u : SkinGI != &m_GIDummy ? 1u : 0u, MaterialBase, G.m_PrevBones }); }
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
                // A masked material (the ivy, the leaves): its own pipeline in its cutout mode (the shadow casters' flag, m_LightParams.w 1: the
                // fragment stops after the cut) - its depth alpha-tested, as its shading's, so it too shades once a pixel (its per-pixel GI read
                // ran for every leaf layer: 22 ms of Bistro's 32 ms scene pass). Its color is written over by its shading. XLION_PREPASS_MASKED=0: none
                static const bool s_bMaskedPrepass = [] { const char* p = xlionrender::Env("XLION_PREPASS_MASKED"); return !(p && p[0] == '0'); }();
                bool bCutLight = false;
                const void* pPrepassBound = nullptr;
                // (Mode 0: all of it; with the G-buffer prepass 1: all of it into the G-buffer - a masked one's cutout writes its normal and
                // instance (the material compiler's XLION_GPRE_OUT) -, then 2: the scene pass starts)
                for( int Mode = bGPre ? 1 : 0; Mode <= (bGPre ? 2 : 0); ++Mode )
                {
                if( Mode == 2 && bOcclude )        // (GPU occlusion with the G-buffer: its Hi-Z from the G-buffer's depth, pass 2's prepass into it too)
                {
                    CmdPass.reset();
                    m_pCull->Occlude(Window, View.m_Depth, View.m_HiZ, View.m_Width, View.m_Height);
                    CmdPass.emplace(Window.StartRenderPass(View.m_GPrePassLoad));
                    PrepassCulled(1);
                }
                if( Mode == 2 ) { CmdPass.reset(); if( pSplit ) Window.WriteTimestamp(*pSplit, 4); if( bScreen ) GatherGI(); if( bSunMask ) SunMask(); if( bContact ) ContactPass(); if( pSplit ) Window.WriteTimestamp(*pSplit, 5); if( bTile ) TilePass(); CmdPass.emplace(Window.StartRenderPass(View.m_ScenePassLoad)); pPrepassGeom = nullptr; pPrepassBound = nullptr; }
                for( const auto& Vis : m_Visible )
                {
                    const std::size_t iDraw = Vis.m_iDraw;
                    const auto&       D     = Scene.m_Draws[iDraw];
                    const auto*       pMesh = Vis.m_pMesh;
                    const std::uint32_t Slot = m_DrawSlot[iDraw];
                    if( !pMesh->m_bCluster || Slot == instance_slots::none_v || (bPlan && m_bBoxesOnly) ) continue;
                    const auto& G = pMesh->m_Geom;
                    if( G.m_Kind == geom_kind::SKIN )
                    {
                        // (with the G-buffer: its single-sided opaque submeshes into it - their AO and the gather's probe-grid read)
                        if( !(bGPre && Mode == 1) ) continue;
                        const std::uint32_t SkipBase = bPlan ? m_ClusterBase[iDraw] : ~0u;
                        for( const auto& Sub : G.m_Submeshes )
                        {
                            auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                            if( !Opaque(pMat) ) continue;
                            std::array<xgpu::buffer*, 12> Buffers{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, G.m_pBoneMap ? G.m_pBoneMap : &m_GIDummy, &m_MaterialParams, &m_EntityParams, TailBuffer() };
                            Cmd.setPipelineInstance(m_GPreSkinInstance, std::span{ Buffers });
                            Cmd.setDynamicUBO(m_LightUBO, 1);
                            Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                            BindClusterStreams(Cmd, G);
                            pPrepassGeom = nullptr; pPrepassBound = nullptr;
                            for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                            {
                                if( SkipBase != ~0u && m_ClusterSkip[SkipBase + c] ) continue;
                                const auto& C = G.m_Clusters[c];
                                Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences, Slot, 0u, 0u, G.m_PrevBones });
                                Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                            }
                        }
                        continue;
                    }
                    const std::uint32_t SkipBase = bPlan ? m_ClusterBase[iDraw] : ~0u;
                    for( const auto& Sub : G.m_Submeshes )
                    {
                        if( CulledOnGPU(iDraw, G, Sub) ) continue;
                        auto* pMat = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                        xgpu::pipeline_instance* pCut = nullptr;
                        bool b2S = false;           // (double sided, opaque: the G-buffer prepass's pipeline that culls nothing)
                        if( pMat )
                        {
                            const auto& St = pMat->m_State;
                            if( !St.m_bDepthWrite || !St.m_bDepthTest || St.m_bWireframe ) continue;
                            if( MaterialVisibilityOf(St).m_Opacity == material_visibility::opacity::MASKED )
                            {
                                if( !s_bMaskedPrepass || pLight == nullptr || (pCut = getCutoutInstance(*pMat)) == nullptr ) continue;
                            }
                            else if( St.m_Blend != material_state::blend::OFF || (St.m_bDoubleSided && !(bGPre && Mode == 1)) ) continue;
                            b2S = St.m_bDoubleSided;
                        }
                        if( Mode == 2 ) continue;        // (everything in the G-buffer: the masked ones' cutout writes its normal and instance too, mb_standard_pbr.frag XLION_GPRE_OUT)
                        if( pCut )
                        {
                            if( !bCutLight )            // (this render's lighting block, its cutout flag set: renderTail wrote its part already)
                            {
                                auto& E = m_LightUBO.allocEntry<details::ubo_lighting>();
                                E = *pLight; E.m_LightParams.m_W = 1.0f;
                                bCutLight = true;
                            }
                            if( pCut->m_Private.get() != pPrepassBound || &G != pPrepassGeom )
                            {
                                std::array<xgpu::buffer*, 10> Buffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, TailBuffer() };
                                Cmd.setPipelineInstance(*pCut, std::span{ Buffers });
                                Cmd.setDynamicUBO(m_LightUBO, 1);
                                Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                                BindClusterStreams(Cmd, G);
                                pPrepassGeom = &G; pPrepassBound = pCut->m_Private.get();
                            }
                            if( pMat->m_bTable ) Cmd.setTextureTable(pMat->m_Table);
                        }
                        else if( &G != pPrepassGeom || pPrepassBound != (b2S ? m_GPre2SInstance.m_Private.get() : nullptr) )
                        {
                            std::array<xgpu::buffer*, 9> Buffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams };
                            Cmd.setPipelineInstance(bGPre && Mode == 1 ? (b2S ? m_GPre2SInstance : m_GPreInstance) : m_DepthPrepassInstance, std::span{ Buffers });
                            Cmd.setDynamicUBO(m_LightUBO, 1);
                            Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                            BindClusterStreams(Cmd, G);
                            pPrepassGeom = &G; pPrepassBound = b2S ? m_GPre2SInstance.m_Private.get() : nullptr;
                        }
                        pBound = nullptr;           // (the shading's binds start again)
                        for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                        {
                            if( SkipBase != ~0u && m_ClusterSkip[SkipBase + c] ) continue;
                            const auto& C = G.m_Clusters[c];
                            Cmd.setPushConstants(details::push_static_geom{ c, Slot, 0u, 0xFFFFFFFFu, pCut ? pMat->m_ParamBase : TileBase(pMat) });       // (no GI: the vertex shader's read skipped)
                            Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                        }
                    }
                }
                }
                if( bCutLight ) m_LightUBO.allocEntry<details::ubo_lighting>() = *pLight;      // (the shading's block again: the next binds take the latest entry)
            }

            if( pSplit ) Cmd.WriteTimestamp(*pSplit, 1);

            // GPU culling's second pass: the depth laid so far (the clusters seen last frame, the CPU path's) into the Hi-Z pyramid, every other
            // cluster in the view tested against it; the visible ones' depth too, in the pass loaded again - then the shading of both passes
            if( bOcclude && !bGPre )        // (with the G-buffer prepass: done before the scene pass, above)
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
                    std::array<xgpu::buffer*, 3> LightBuffers{ &m_LightRing, &m_CellRing, TailBuffer() };
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
                    // (a skinned mesh is not in the G-buffer prepass - so not in the tile pass either, whatever its material: it shades here. It drew nothing and
                    // showed as a flat silhouette in every level with characters, Bistro Game 2026-11-06)
                    if( !CulledOnGPU(iDraw, pMesh->m_Geom, Sub) && (!TileMat(pMat) || pMesh->m_Geom.m_Kind == geom_kind::SKIN) ) DrawSubmesh(iDraw, Sub, pMat);
                }
            }

            // The GPU culled runs, by bind state: a state is bound once (texture-table materials of one pipeline share it: a run of another
            // material sets its table - bindless: nothing - and its material base; the rest: the records the cull wrote)
            {
                std::vector<std::uint32_t> Order;
                for( std::uint32_t r = 0; r < std::uint32_t(m_CullRuns.size()); ++r ) if( m_CullRuns[r].m_pInstance && !TileMat(m_CullRuns[r].m_pMat) ) Order.push_back(r);
                const auto Key = [&]( std::uint32_t r ) { const auto& R = m_CullRuns[r]; return std::tuple{ R.m_pInstance->m_Private.get(), R.m_pGeom, R.m_pGI }; };
                std::stable_sort(Order.begin(), Order.end(), [&]( std::uint32_t a, std::uint32_t b ) { return Key(a) < Key(b); });
                const void* pRunBound = nullptr; const cluster_geom* pRunGeom = nullptr; const xgpu::buffer* pRunGI = nullptr;
                for( const auto r : Order )
                {
                    const auto& R = m_CullRuns[r];
                    const auto& G = *R.m_pGeom;
                    if( R.m_pInstance->m_Private.get() != pRunBound || &G != pRunGeom || R.m_pGI != pRunGI )
                    {
                        std::array<xgpu::buffer*, 11> Buffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, R.m_pGI, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, &m_pCull->m_Records, TailBuffer() };
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

            if( pSplit ) Cmd.WriteTimestamp(*pSplit, 3);
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
        std::vector<gi::surfel_mesh::surfel_material> MaskMats;      // (a mask job: the UV square's grid, its slot)
        if( J.m_MaskSlot >= 0 )
        {
            constexpr int T = gi::world::mask_texels_v;
            MaskMats.resize(std::size_t(T) * T);
            for( int y = 0; y < T; ++y ) for( int x = 0; x < T; ++x ) MaskMats[std::size_t(y) * T + x] = { (float(x) + 0.5f) / T, (float(y) + 0.5f) / T, 0.25f / T, std::uint32_t(J.m_MaskSlot) };
        }
        const auto& Mats  = J.m_MaskSlot >= 0 ? MaskMats : J.m_bTriangles ? TriMats : G.m_SurfelMaterial;
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
                Data[2 * i + 1] = xmath::fvec4(J.m_MaskSlot >= 0 ? gi::v3{ 0, 0, 1 } : J.m_bTriangles ? TriN[J.m_Order[i]] : L0[J.m_Order[i]].m_Normal, 0);
            }
            if( m_Device.Create(J.m_Data, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = sizeof(xmath::fvec4), .m_EntryCount = 2 * n, .m_pData = Data.data() }) )
            { m_GICaptures.pop_front(); return; }

            (void)m_GICaptureVB.MemoryMap(0, 4 * gi_capture_passes_v * std::size_t(n), [&](void* p)
            {
                auto* V = static_cast<float*>(p);
                for( int Pass = 0; Pass < gi_capture_passes_v; ++Pass )
                    for( int i = 0; i < n; ++i )
                    {
                        const float x0 = float(2 * (i % (gi_capture_w_v / 2))), y0 = float(2 * (i / (gi_capture_w_v / 2)) + Pass * gi_capture_h_v / gi_capture_passes_v);
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
            std::array<xgpu::buffer*, 10> Buffers{ &J.m_Data, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing, &m_MaterialParams, &m_EntityParams, TailBuffer() };
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
                for( int Pass = 0; Pass < gi_capture_passes_v; ++Pass )      // (xlionrender_surfel_gi.glsl: 0x800 emission, 0x2000 the specular passes)
                {
                    Cmd.setPushConstants(details::push_static_geom{ std::uint32_t(-Pass * n), J.m_Slot, 0u, (3u << 9) | ((Pass & 1) ? 0x800u : 0u) | (Pass >= 2 ? 0x2000u : 0u), pInst == &m_GIConstCaptureInstance ? 0u : pMat->m_ParamBase });
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
            if( J.m_MaskSlot >= 0 )
            {
                std::vector<std::uint32_t> Bits(gi::world::mask_words_v, 0u);
                int nOpaque = 0;
                for( int i = 0; i < n; ++i )
                {
                    const int x0 = 2 * (i % (W / 2)), y0 = 2 * (i / (W / 2));
                    const auto A = [&](int x, int y) { return Half(P[4 * (std::size_t(y) * W + x) + 3]); };
                    if( (A(x0, y0) + A(x0 + 1, y0) + A(x0, y0 + 1) + A(x0 + 1, y0 + 1)) * 0.25f > 0.5f ) { const int k = J.m_Order[i]; Bits[std::size_t(k / 32)] |= 1u << (k % 32); ++nOpaque; }
                }
                static const bool s_bMaskLog = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
                if( s_bMaskLog ) { std::printf("GIMASK slot %d: %.1f%% opaque\n", J.m_MaskSlot, 100.0 * nOpaque / std::max(n, 1)); std::fflush(stdout); }
                const std::uint64_t Key = (std::uint64_t(J.m_Materials[std::size_t(J.m_MaskSlot)].m_Index) << 32) | J.m_Materials[std::size_t(J.m_MaskSlot)].m_Generation;
                for( const auto& M : m_GIMaskWaiting ) if( M.m_Key == Key ) m_pGIWorld->SetSlotMask(M.m_GIId, M.m_Slot, Bits);
                std::erase_if(m_GIMaskWaiting, [&](const gi_mask_wait& M) { return M.m_Key == Key; });
                m_GIMaskCache[Key] = std::move(Bits);
                m_GICaptures.pop_front();
                return;
            }
            std::vector<xmath::fvec3> Albedo(n), Emission(n);
            std::vector<std::array<float, 4>> Specular(n);       // (F0's color, roughness: reflections' secondary hits)
            for( int i = 0; i < n; ++i )
            {
                const int x0 = 2 * (i % (W / 2)), y0 = 2 * (i / (W / 2));
                const auto Block = [&](int y) { return (Texel(x0, y) + Texel(x0 + 1, y) + Texel(x0, y + 1) + Texel(x0 + 1, y + 1)) * 0.25f; };
                Albedo[J.m_Order[i]]   = Block(y0);
                Emission[J.m_Order[i]] = Block(y0 + H / gi_capture_passes_v);         // as captured (the world scales it: scene::gi_settings::m_EmissiveScale)
                const auto F0 = Block(y0 + 2 * H / gi_capture_passes_v), RM = Block(y0 + 3 * H / gi_capture_passes_v);
                Specular[J.m_Order[i]] = { F0.m_X, F0.m_Y, F0.m_Z, RM.m_X };
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
                m_pGIWorld->SetTriangleSpecular(J.m_GIId, J.m_First, Specular);
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
                    const auto& Ends = pSlot->m_Geom.m_LODIndexEnd;       // (the section's triangles are LOD 0's, the last in the index buffer: coarser LODs' clusters belong to sections too, and counting them left no table: SURFACE_COLOR black)
                    const std::uint32_t Lod0First = Ends.size() > 1 ? Ends[1] : 0u;
                    for( const auto c : Order )
                        if( c < pSlot->m_GIClusterSection.size() && pSlot->m_GIClusterSection[c] == std::uint16_t(J.m_Section) && Cl[c].m_iIndex >= Lod0First ) { First[c] = Off; Off += Cl[c].m_nIndices / 3; }
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
        m_pTailRecords = nullptr;              // (binding 14: none until this render's screen tail)
        if( m_SunMaskTex.m_Private && !m_bSunMaskCleared ) { (void)Window.StartRenderPass(m_SunMaskClear); m_bSunMaskCleared = true; }      // (slot 11 sampled from the first draw)
        static const bool s_bNoBudget = xlionrender::Env("XLION_NO_FRAME_BUDGET") != nullptr;    // measurement only (GI review): every cluster drawn, whatever the host's budget
        if( s_bNoBudget ) m_FrameBudgetMs = 0;

        // The render target's viewport is for this render: on a copy. The host's own view keeps the viewport it has (the editor's panel, in screen
        // coordinates): its click-pick of the same frame builds its ray from it (it used to get the target's: every click missed)
        xgpu::tools::view Camera = CameraIn;
        Camera.setViewport({ 0, 0, View.m_Width, View.m_Height });

        // Camera-centered world (same as the editors): precision stays near the eye in large worlds
        const xmath::fvec3 Eye     = Camera.getPosition();
        const xmath::fmat4 W2CNoJitter = Camera.getW2C() * xmath::fmat4::fromTranslation(Eye);
        static const int s_EnvTAA = [] { const char* p = xlionrender::Env("XLION_TAA"); return p ? (p[0] != '0' ? 1 : 0) : -1; }();      // (measurement: 0 off, else on)
        // (the path tracers accumulate while each texel's point holds: TAA's sub-pixel jitter moved it every frame and they never converged - Tomas, 2026-11-06: off with them, always)
        static const bool s_bPTRefl = xlionrender::Env("XLION_REFL_PT") != nullptr || xlionrender::Env("XLION_GI_REF") != nullptr;
        const bool bPathTracing = s_bPTRefl || Scene.m_GI.m_bPathTraced || Scene.m_bDebugReflectionPathTraced;
        const bool bTAA = !bPathTracing && (s_EnvTAA >= 0 ? s_EnvTAA == 1 : Scene.m_Post.m_bTAA);
        const xmath::fmat4 W2C = [&]
        {
            if( !bTAA ) return W2CNoJitter;
            // A sub-pixel a frame: Halton 2, 3 over 8 frames
            const auto Halton = [](int i, int b) { float f = 1.0f, r = 0.0f; for( ; i > 0; i /= b ) { f /= float(b); r += f * float(i % b); } return r; };
            const int k = (View.m_TAAFrame % 8) + 1;
            return xmath::fmat4::fromTranslation(xmath::fvec3{ (Halton(k, 2) - 0.5f) * 2.0f / float(View.m_Width), (Halton(k, 3) - 0.5f) * 2.0f / float(View.m_Height), 0.0f }) * W2CNoJitter;
        }();

        const float ProjScale = float(View.m_Height) / (2.0f * xmath::Tan(xmath::radian{ Camera.getFov().m_Value * 0.5f }));    // pixels per unit at distance 1
        m_LODProjScale = ProjScale;                             // (the main view's LOD pick: renderScenePass)

        // A skinned mesh's LOD, this frame, for every pass (Tomas 2026-11-06: LODs by the area on the screen, not the distance): each submesh the
        // coarsest LOD whose error projects under XLION_LOD_PIXELS (1) pixels at the draw's box (its nearest point) - the rigid groups' rule
        // (renderScenePass), on its submeshes' list: the prepass, the scene pass, the masks and the shadow pages (the coarser of a page's and this)
        // read it. (A skin's mesh is its entity's own: setting it here touches no other draw)
        {
            static const float s_LODPixels = [] { const char* p = xlionrender::Env("XLION_LOD_PIXELS"); return p ? float(std::atof(p)) : 1.0f; }();
            static const int   s_SkinLOD   = [] { const char* p = xlionrender::Env("XLION_SKIN_LOD"); return p ? std::atoi(p) : -1; }();     // (to see them: every skin at LOD k)
            m_SkinLODReport = {};
            for( const auto& D : Scene.m_Draws )
            {
                const auto* pMesh = findMesh(D.m_Mesh);
                if( pMesh == nullptr || !pMesh->m_bCluster || pMesh->m_Geom.m_Kind != geom_kind::SKIN || pMesh->m_Geom.m_LODSubmeshes.size() < 2 ) continue;
                auto& G = const_cast<cluster_geom&>(pMesh->m_Geom);
                const auto  B     = G.m_BBox.TransformCopy(D.m_L2W);
                const float Dist  = (xmath::fvec3::Max(B.m_Min, xmath::fvec3::Min(Eye, B.m_Max)) - Eye).Length();
                const float Scale = std::max({ (D.m_L2W * xmath::fvec3{ 1, 0, 0 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length(), (D.m_L2W * xmath::fvec3{ 0, 1, 0 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length()
                                             , (D.m_L2W * xmath::fvec3{ 0, 0, 1 } - D.m_L2W * xmath::fvec3{ 0, 0, 0 }).Length() });
                const int   Finest = std::max(G.m_FinestLOD, G.m_DrawnLOD), nLODs = int(G.m_LODSubmeshes.size());
                for( std::size_t s = 0; s < G.m_Submeshes.size() && s < G.m_LODSubmeshes[0].size(); ++s )
                {
                    const int Lod = std::clamp(std::max(Finest, s_SkinLOD >= 0 ? s_SkinLOD : s_LODPixels > 0 ? G.PickLOD(s, Dist, Scale, ProjScale, s_LODPixels) : 0), 0, nLODs - 1);
                    G.m_Submeshes[s] = G.m_LODSubmeshes[std::size_t(Lod)][s];
                    ++m_SkinLODReport.m_nSubmeshes[std::size_t(std::min(Lod, 3))];
                }
                // (XLION_SKIN_LOD_LOG, measurement: each skinned draw's distance and its first submesh's LOD, every 120 frames)
                static const bool s_bSkinLODLog = xlionrender::Env("XLION_SKIN_LOD_LOG") != nullptr;
                if( s_bSkinLODLog && m_Device.getFrameIndex() % 120 == 0 && !G.m_Submeshes.empty() )
                {
                    int Lod0 = 0;
                    for( int k = 0; k < nLODs; ++k ) if( G.m_LODSubmeshes[std::size_t(k)][0].m_iCluster == G.m_Submeshes[0].m_iCluster ) Lod0 = k;
                    std::printf("SKINLOD entity %016llX: %.2f m, scale %.3f, LOD %d of %d (LOD 1 error %g)\n", (unsigned long long)D.m_UserID, Dist, Scale, Lod0, nLODs
                               , G.m_LODErrors.size() > 1 && !G.m_LODErrors[1].empty() ? G.m_LODErrors[1][0] : 0.0f);
                    std::fflush(stdout);
                }
            }
        }
        using clock = std::chrono::steady_clock;
        const auto Ms = [](clock::time_point A, clock::time_point B) { return std::chrono::duration<double, std::milli>(B - A).count(); };
        const auto T0 = clock::now();
        m_nSceneBinds = 0;
        if( m_ContactTex.m_Private && !m_bContactReady )       // (written once, white, before anything samples it: every view's shaders bind it)
        {
            m_bContactReady = true;
            struct { xmath::fmat4 m_W2C; xmath::fvec4 m_ToSun, m_Params; std::uint32_t m_Size[4]; } Fill{ W2C, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { std::uint32_t(sun_mask_w_v), std::uint32_t(sun_mask_h_v), 1u, 0u } };
            auto Buffers  = std::array<xgpu::buffer*, 1>{ &m_GIDummy };
            auto Textures = std::array<const xgpu::texture*, 1>{ &m_WhiteTexture };
            auto Images   = std::array<const xgpu::texture*, 1>{ &m_ContactTex };
            Window.Dispatch(m_ContactCompute, Buffers, Textures, Images, &Fill, sizeof(Fill), (sun_mask_w_v + 7) / 8, (sun_mask_h_v + 7) / 8);
        }
        // the geometries' residency (cluster_geom::Refresh): a geometry whose buffers dropped or regained LOD 0 draws what they hold
        {
            static const bool s_bResLog = xlionrender::Env("XLION_RESIDENCY_LOG") != nullptr;     // (measurement: a geometry's resident finest LOD changing)
            for( auto& M : m_Meshes ) if( M.m_bAlive && M.m_bCluster && M.m_Geom.Refresh() && s_bResLog )
                std::printf("RESIDENCY frame %llu: a geometry of %zu clusters now draws from LOD %d (of %zu)\n", (unsigned long long)m_Device.getFrameIndex(), M.m_Geom.m_Clusters.size(), M.m_Geom.m_FinestLOD, M.m_Geom.m_LODSubmeshes.size()), std::fflush(stdout);
        }
        // Measurement only (XLION_GPU_TIMES): the whole Render on the GPU (its start, after the GI and the captures, after the tone map, its end)
        static const bool s_bTimes = xlionrender::Env("XLION_GPU_TIMES") != nullptr;
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
                // measurement (XLION_FRAME_CSV=<file>, the scorecard's game sequence): every frame's parts, for p95 / max
                static FILE* s_pCsv = [] { const char* p = xlionrender::Env("XLION_FRAME_CSV"); FILE* f = p ? xlionrender::Open(p, "w") : nullptr; if (f) std::fprintf(f, "frame,environment,instances,gi_update,captures,shadows_scene_tonemap,overlays,total,shadows,scene,tonemap,vsm_pages,vsm_net,vsm_epoch,sh_cpu_draws,sh_cpu_tris,sh_cpu_casters,sh_indirect,sh_indirect_max,sh_page_clears,sh_pair_cands,shadow_pass,probes_planar,wall_ms,sh_pair_tris\n"); return f; }();
                if( s_pCsv )
                {
                    std::fprintf(s_pCsv, "%llu", (unsigned long long)m_Device.getFrameIndex());
                    for( int k = 0; k < 6; ++k ) std::fprintf(s_pCsv, ",%.4f", (s_Ns[k + 1] - s_Ns[k]) * 1e-6);
                    { gpu_times G; (void)getGPUTimes(G); }       // (their split: the latest the scene pass' own timers have - read by whoever came first)
                    const auto& G = getLastGPUTimes();
                    const auto& Pg = m_VSMReport.m_Pages;     // (the sun's pages rendered this frame: the fine ones, the net's; a new epoch)
                    const auto& Sd = m_ShadowDrawStats;     // (the shadow pass's commands this frame: the CPU path's draws, the pairs' indirect ones)
                    // (wall_ms: the time since the last row - a stall of the main thread, a load on it, shows here and not in the GPU parts)
                    static auto s_Last = std::chrono::steady_clock::now(); const auto Now = std::chrono::steady_clock::now();
                    const double WallMs = std::chrono::duration<double, std::milli>(Now - s_Last).count(); s_Last = Now;
                    std::fprintf(s_pCsv, ",%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%lld,%d,%d,%d,%d,%d,%.4f,%.4f,%.3f,%lld\n", (s_Ns[6] - s_Ns[0]) * 1e-6, G.m_ShadowMs, G.m_SceneMs, G.m_ToneMapMs, Pg.m_Rendered, Pg.m_RenderedPinned, Pg.m_Epochs
                                , Sd.m_CpuDraws, (long long)Sd.m_CpuTris, Sd.m_CpuCasters, Sd.m_IndirectDraws, Sd.m_IndirectMax, Sd.m_PageClears, m_PairsReport.m_nCandidates, m_SplitShadowMs, m_SplitProbesMs, WallMs, (long long)m_PairsReport.m_Triangles);     // (sh_pair_tris: the pairs' triangles, a ring late)
                    if( m_Device.getFrameIndex() % 60 == 0 ) std::fflush(s_pCsv);
                }
                if( ++s_n == 120 ) { std::printf("RENDERGPU environment %.2f, instances %.2f, GI update %.2f, captures %.2f, shadows+scene+tonemap %.2f, overlays+outline %.2f ms\n", s_Sum[0] / s_n, s_Sum[1] / s_n, s_Sum[2] / s_n, s_Sum[3] / s_n, s_Sum[4] / s_n, s_Sum[5] / s_n); std::fflush(stdout); s_n = 0; for( auto& v : s_Sum ) v = 0; }
            }
            Window.WriteTimestamp(*pTS, 0);
        }
        // The probe atlas and the shadow atlas are bound by every lit pipeline, the GI capture's below included: cleared before anything samples them
        // (the first frame's capture used to read them undefined: a Vulkan validation error)
        if( !m_bProbeAtlasReady ) { auto Cmd = Window.StartRenderPass(m_ProbeAtlasClearPass); m_bProbeAtlasReady = true; }
        if( !m_bShadowAtlasReady ) { auto Cmd = Window.StartRenderPass(m_ShadowClearPass); m_bShadowAtlasReady = true; }
        { XPROFILE_SCOPE("Render: environment"); XPROFILE_GPU_SCOPE(Window, "Render: environment"); updateEnvironment(Window, Scene); }
        if( pTS ) Window.WriteTimestamp(*pTS, 1);
        { XPROFILE_SCOPE("Render: instances"); XPROFILE_GPU_SCOPE(Window, "Render: instances"); updateInstances(Window, Scene, Eye); }
        if( pTS ) Window.WriteTimestamp(*pTS, 2);
        { XPROFILE_SCOPE("Render: GI update"); XPROFILE_GPU_SCOPE(Window, "Render: GI update"); updateGI(Window, Scene, Eye, ProjScale); }
        if( pTS ) Window.WriteTimestamp(*pTS, 3);                                   // CPU (getGIStats); the probes' captures read it too
        {
            XPROFILE_SCOPE("Render: GI capture"); XPROFILE_GPU_SCOPE(Window, "Render: GI capture");
            const auto TCapture = std::chrono::steady_clock::now();
            processGICapture(Window);                           // the surfels' material colors (one mesh at a time)
            if( !m_GIFrames.empty() ) m_GIFrames.back().m_Stats.m_CaptureMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - TCapture).count();
        }
        const auto T1 = clock::now();
        if( pTS ) Window.WriteTimestamp(*pTS, 4);
        Window.WriteTimestamp(m_Timestamps, 0);                 // GPU: the shadow passes start here (after the instance copy)
        // measurement (XLION_FRAME_CSV): the shadows alone, the probes and planar after them ("shadows" above runs to the scene pass: the
        // screen tail too - the frame budget's categories, methodology 3)
        static std::unordered_map<const void*, xgpu::timestamps> s_SplitTS;     // ponytail: per renderer, never erased (a measurement hook)
        xgpu::timestamps* pSplit = nullptr;
        if( pTS ) { auto [It, bNew] = s_SplitTS.try_emplace(this); if( bNew && m_Device.Create(It->second, { .m_Count = 3 }) ) s_SplitTS.erase(It); else pSplit = &It->second; }
        if( pSplit ) Window.WriteTimestamp(*pSplit, 0);
        { XPROFILE_SCOPE("Render: shadows"); XPROFILE_GPU_SCOPE(Window, "Render: shadows"); renderShadows(Window, Scene, Eye, ProjScale); vsmBounds(Window); }
        if( pSplit ) Window.WriteTimestamp(*pSplit, 1);
        { XPROFILE_SCOPE("Render: probes"); XPROFILE_GPU_SCOPE(Window, "Render: probes"); updateProbes(Window, Scene, Eye); }
        { XPROFILE_SCOPE("Render: planar"); XPROFILE_GPU_SCOPE(Window, "Render: planar"); renderPlanar(Window, Scene, View, Camera); }
        if( pSplit ) { Window.WriteTimestamp(*pSplit, 2); static std::vector<std::uint64_t> s_SplitNs; if( pSplit->getResults(s_SplitNs) && s_SplitNs.size() >= 3 && s_SplitNs[2] >= s_SplitNs[0] ) { m_SplitShadowMs = (s_SplitNs[1] - s_SplitNs[0]) * 1e-6; m_SplitProbesMs = (s_SplitNs[2] - s_SplitNs[1]) * 1e-6; } }
        const auto T2 = clock::now();

        updateBudget();
        const auto T2b = clock::now();
        { XPROFILE_SCOPE("Render: scene pass"); XPROFILE_GPU_SCOPE(Window, "Render: scene pass"); renderScenePass(Window, Scene, View, W2C, Eye, true); }
        {
            XPROFILE_SCOPE("Render: shadow requests"); XPROFILE_GPU_SCOPE(Window, "Render: shadow requests");
            if( Scene.m_Sun.m_bCastShadows && !xlionrender::Env("XLION_NO_SUN_SHADOWS") ) vsmRequestPass(Window, View, W2C, Eye);      // (the main view's depth: the pages it asks for)
            localRequestPass(Window, Scene, View, W2C, Eye);
        }
        const auto T2c = clock::now();

        // HDR -> bloom, exposure -> LionMapper -> sRGB output (alpha = coverage)
        {
            XPROFILE_SCOPE("Render: tone map"); XPROFILE_GPU_SCOPE(Window, "Render: tone map");
            static const float s_EnvBloom = [] { const char* p = xlionrender::Env("XLION_BLOOM"); return p ? float(std::atof(p)) : -1.0f; }();       // (measurement)
            static const int   s_EnvAuto = [] { const char* p = xlionrender::Env("XLION_AUTO_EXPOSURE"); return p ? (p[0] != '0' ? 1 : 0) : -1; }();     // (measurement: 0 off, else on)
            const float Bloom = s_EnvBloom >= 0.0f ? s_EnvBloom : Scene.m_Post.m_Bloom;
            const bool  bAuto = s_EnvAuto >= 0 ? s_EnvAuto == 1 : Scene.m_Post.m_bAutoExposure;
            // (last frame's lit image for the next frame's screen reads: the prepass clears m_HDR - they had read black)
            static const bool s_bSSGIPrev = xlionrender::Env("XLION_SSGI") != nullptr || xlionrender::Env("XLION_VOX_SEEN") != nullptr;      // (and the voxels' lit feed)
            if( s_bSSGIPrev || ReflectionScreen(Scene) )
            {
                if( !View.m_HDRPrev.m_Private ) (void)m_Device.Create(View.m_HDRPrev, { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = View.m_Width, .m_Height = View.m_Height, .m_isGamma = false, .m_pName = "Render target: last frame's HDR" });
                if( View.m_HDRPrev.m_Private ) { Window.CopyTexture(View.m_HDR, View.m_HDRPrev, View.m_Width, View.m_Height); View.m_bHDRPrev = true; }
            }
            static const float s_EnvExposure = [] { const char* p = xlionrender::Env("XLION_EXPOSURE"); return p ? float(std::atof(p)) : 0.0f; }();      // (measurement: a fixed exposure with auto exposure off - graded lit images)
            if( s_EnvExposure > 0.0f ) View.m_Exposure = s_EnvExposure;
            const xgpu::texture* pHDR = &View.m_HDR;
            xgpu::pipeline_instance* pToneMap = &View.m_ToneMapInstance;
            if( bTAA )
            {
                if( !View.m_TAA[0].m_Private )
                {
                    bool bOk = true;
                    for( int i = 0; i < 2; ++i )
                    {
                        bOk = bOk && !m_Device.Create(View.m_TAA[i], { .m_Format = xgpu::texture::format::R16G16B16A16_SFLOAT, .m_Width = View.m_Width, .m_Height = View.m_Height, .m_isGamma = false, .m_bStorage = true, .m_pName = "Render target: TAA" });
                        if( !bOk ) break;
                        auto Bindings = std::array{ xgpu::pipeline_instance::sampler_binding{ View.m_TAA[i] }, xgpu::pipeline_instance::sampler_binding{ View.m_Bloom[0] }, xgpu::pipeline_instance::sampler_binding{ View.m_ExposureTex } };
                        bOk = !m_Device.Create(View.m_ToneMapTAA[i], { .m_PipeLine = m_ToneMapPipeline, .m_SamplersBindings = Bindings });
                        if( !bOk ) break;
                        struct { int m_W, m_H, m_Mode; float m_A, m_B, m_C, m_D, m_E; } Fill{ View.m_Width, View.m_Height, 3, 0.0f };
                        const xgpu::texture* T[] = { &View.m_HDR, &View.m_ExposureTex }; const xgpu::texture* I[] = { &View.m_TAA[i] };
                        Window.Dispatch(m_PostCompute, std::span<xgpu::buffer* const>{}, T, I, &Fill, sizeof(Fill), (View.m_Width + 7) / 8, (View.m_Height + 7) / 8);
                    }
                    View.m_bTAAHistory = false;
                }
                const int Cur = View.m_TAAFrame & 1;
                static const float s_TAABlendEnv = [] { const char* p = xlionrender::Env("XLION_TAA_BLEND"); return p ? float(std::atof(p)) : 0.0f; }();      // (measurement: this frame's share)
                struct { xmath::fmat4 m_Reproject; int m_W, m_H; float m_Blend, m_bHistory; } Push{ View.m_TAAPrevW2C * xmath::fmat4::fromTranslation(Eye - View.m_TAAPrevEye) * W2CNoJitter.Inverse(), View.m_Width, View.m_Height, s_TAABlendEnv > 0.0f ? s_TAABlendEnv : Scene.m_Post.m_TAABlend, View.m_bTAAHistory ? 1.0f : 0.0f };
                const xgpu::texture* T[] = { &View.m_HDR, &View.m_Depth, &View.m_TAA[1 - Cur] }; const xgpu::texture* I[] = { &View.m_TAA[Cur] };
                Window.Dispatch(m_TAACompute, std::span<xgpu::buffer* const>{}, T, I, &Push, sizeof(Push), (View.m_Width + 7) / 8, (View.m_Height + 7) / 8);
                pHDR = &View.m_TAA[Cur]; pToneMap = &View.m_ToneMapTAA[Cur];
                View.m_TAAPrevW2C = W2CNoJitter; View.m_TAAPrevEye = Eye; View.m_bTAAHistory = true; ++View.m_TAAFrame;
            }
            else View.m_bTAAHistory = false;
            {
                struct { int m_W, m_H, m_Mode; float m_A, m_B, m_C, m_D, m_E; } Push{};
                const auto Run = [&](const xgpu::texture& Src, const xgpu::texture& Dst, int W, int H)
                {
                    const xgpu::texture* T[] = { &Src, &View.m_HDR }; const xgpu::texture* I[] = { &Dst };
                    if( &Dst != &View.m_ExposureTex ) T[1] = &View.m_ExposureTex;      // (read: the threshold; the exposure's own pass writes it)
                    Window.Dispatch(m_PostCompute, std::span<xgpu::buffer* const>{}, T, I, &Push, sizeof(Push), (W + 7) / 8, (H + 7) / 8);
                };
                const auto Size = [&](int i) { return std::array<int, 2>{ std::max(View.m_Width >> (i + 1), 1), std::max(View.m_Height >> (i + 1), 1) }; };
                if( !View.m_bPostReady )
                {
                    Push.m_Mode = 3; Push.m_A = 0.0f;
                    for( int i = 0; i < int(View.m_Bloom.size()); ++i ) { const auto S = Size(i); Push.m_W = S[0]; Push.m_H = S[1]; Run(View.m_HDR, View.m_Bloom[i], S[0], S[1]); }
                    Push.m_W = Push.m_H = 1; Run(View.m_HDR, View.m_ExposureTex, 1, 1);
                    View.m_bPostReady = true;
                }
                // The levels by the view's height: the bloom's width a share of the screen at any resolution (the smallest level about 48 pixels high at the view's full size)
                const int nBloom = std::clamp(int(std::round(std::log2(float(View.m_Height) / 48.0f))), 3, int(View.m_Bloom.size()));
                const int nLevels = Bloom > 0.0f ? nBloom : 0;
                for( int i = 0; i < nLevels; ++i )
                {
                    const auto S = Size(i);
                    Push = {}; Push.m_W = S[0]; Push.m_H = S[1]; Push.m_Mode = 0; Push.m_A = i == 0 ? 1.0f : 0.0f;
                    Push.m_B = Scene.m_Post.m_BloomThreshold; Push.m_C = Scene.m_Post.m_BloomKnee; Push.m_D = View.m_Exposure; Push.m_E = bAuto ? 1.0f : 0.0f;
                    Run(i == 0 ? *pHDR : View.m_Bloom[i - 1], View.m_Bloom[i], S[0], S[1]);
                }
                if( bAuto )
                {
                    const auto Now = clock::now();
                    const float Dt = View.m_PostLast == clock::time_point{} ? 0.0f : std::min(std::chrono::duration<float>(Now - View.m_PostLast).count(), 0.25f);
                    View.m_PostLast = Now;
                    Push = { 1, 1, 2, Dt, Scene.m_Post.m_ExposureRate, Scene.m_Post.m_ExposureKey, Scene.m_Post.m_ExposureMin, Scene.m_Post.m_ExposureMax };
                    Run(*pHDR, View.m_ExposureTex, 1, 1);
                }
                if( Bloom > 0.0f ) for( int i = nBloom - 2; i >= 0; --i )
                {
                    const auto S = Size(i);
                    Push = {}; Push.m_W = S[0]; Push.m_H = S[1]; Push.m_Mode = 1; Push.m_A = Scene.m_Post.m_BloomSpread;
                    Run(View.m_Bloom[i + 1], View.m_Bloom[i], S[0], S[1]);
                }
            }
            auto Cmd = Window.StartRenderPass(View.m_OutputPass);
            const bool bHeatView = View.m_HeatToneMap.m_Private != nullptr;       // (XLION_HEAT: the gather's clock over a grey image)
            Cmd.setPipelineInstance(bHeatView ? View.m_HeatToneMap : *pToneMap);
            Cmd.setPushConstants(details::push_tonemap{ { View.m_Exposure, Bloom, bAuto ? 1.0f : 0.0f, bHeatView ? 1.0f : 0.0f }, View.m_Background });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
            Cmd.WriteTimestamp(m_Timestamps, 3);
            if( pTS ) Cmd.WriteTimestamp(*pTS, 5);
        }
        // XLION_NOISE_LAGS="<seconds to wait> <frames>" (measurement, Tomas 2026-11-06): the noise of the tone-mapped image by how far apart in time, at the
        // real frame rate - every 4th pixel's luminance a frame into a CPU-readable ring (xlionrender_noise_lags_comp.glsl), compared on the CPU at lags
        // 1 to 60 frames once the GPU is done with them; a NOISELAGS line when it ends
        static const std::array<float, 2> s_NoiseLags = [] { std::array<float, 2> A{ 0.0f, 0.0f }; if( const char* p = xlionrender::Env("XLION_NOISE_LAGS") ) (void)xlionrender::Scan(p, "%f %f", &A[0], &A[1]); return A; }();
        if( s_NoiseLags[1] > 0.0f )
        {
            static const auto s_T0 = std::chrono::steady_clock::now();
            constexpr int lags_v[7] = { 1, 2, 4, 8, 15, 30, 60 }, hist_v = 61;
            const int SubW = View.m_Width / 4, SubH = View.m_Height / 4, nSub = SubW * SubH, nWords = (nSub + 3) / 4, nFrames = int(s_NoiseLags[1]);
            const int Slot = int(m_Device.getFrameIndex() % ring_frames_v);
            if( std::chrono::duration<float>(std::chrono::steady_clock::now() - s_T0).count() > s_NoiseLags[0] && m_NoiseLagsN < nFrames + int(ring_frames_v) )
            {
                if( !m_NoiseLagsRing.m_Private ) { (void)m_Device.Create(m_NoiseLagsRing, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = int(ring_frames_v) * nWords, .m_pName = "Debug: noise lags" }); m_NoiseLagsHist.assign(std::size_t(hist_v) * nSub, 0); }
                if( m_NoiseLagsN >= int(ring_frames_v) )      // (this slot's frame, ring_frames_v ago, is done: its image in the history)
                {
                    const int f = m_NoiseLagsN - int(ring_frames_v);
                    if( f == 0 ) m_NoiseLagsT0 = std::chrono::steady_clock::now();
                    m_NoiseLagsT1 = std::chrono::steady_clock::now();
                    std::uint8_t* pCur = m_NoiseLagsHist.data() + std::size_t(f % hist_v) * nSub;
                    (void)m_NoiseLagsRing.MemoryMap(Slot * nWords, nWords, [&](void* p) { std::memcpy(pCur, p, std::size_t(nSub)); });
                    for( int l = 0; l < 7; ++l )
                    {
                        if( f < lags_v[l] ) continue;
                        const std::uint8_t* pOld = m_NoiseLagsHist.data() + std::size_t((f - lags_v[l]) % hist_v) * nSub;
                        double S = 0.0, O2 = 0.0, O4 = 0.0, O8 = 0.0;
                        for( int i = 0; i < nSub; ++i ) { const float d = float(int(pCur[i]) - int(pOld[i])), s = std::abs(d) * 0.70710678f; S += d * d * 0.5; O2 += s > 2.0f; O4 += s > 4.0f; O8 += s > 8.0f; }
                        m_NoiseLagsSum[l] += S; m_NoiseLagsCnt[l] += nSub; m_NoiseLagsOver[l][0] += O2; m_NoiseLagsOver[l][1] += O4; m_NoiseLagsOver[l][2] += O8;
                    }
                }
                if( m_NoiseLagsN < nFrames )
                {
                    const std::uint32_t Push[4] = { std::uint32_t(SubW), std::uint32_t(SubH), std::uint32_t(Slot * nWords), 0u };
                    auto Buffers = std::array<xgpu::buffer*, 1>{ &m_NoiseLagsRing };
                    auto Textures = std::array<const xgpu::texture*, 1>{ &View.m_Output };
                    Window.Dispatch(m_NoiseLagsCompute, Buffers, Textures, Push, sizeof(Push), (nWords + 63) / 64, 1);
                }
                if( ++m_NoiseLagsN == nFrames + int(ring_frames_v) )
                {
                    const int nF = nFrames; const double Ms = nF > 1 ? std::chrono::duration<double, std::milli>(m_NoiseLagsT1 - m_NoiseLagsT0).count() / double(nF - 1) : 0.0;
                    std::printf("NOISELAGS %d frames, %.2f ms apart, every 4th pixel: lag, ms, rms, over 2 %%, over 4 %%, over 8 %%\n", nF, Ms);
                    for( int l = 0; l < 7; ++l ) if( m_NoiseLagsCnt[l] > 0.0 ) std::printf("NOISELAG %d %.0f ms: %.3f %.2f %.3f %.4f\n", lags_v[l], Ms * lags_v[l], std::sqrt(m_NoiseLagsSum[l] / m_NoiseLagsCnt[l]), 100.0 * m_NoiseLagsOver[l][0] / m_NoiseLagsCnt[l], 100.0 * m_NoiseLagsOver[l][1] / m_NoiseLagsCnt[l], 100.0 * m_NoiseLagsOver[l][2] / m_NoiseLagsCnt[l]);
                    std::fflush(stdout);
                }
            }
        }

        // Editor overlays over the tone-mapped image, depth-tested against the scene
        if( Overlay )
        {
            auto Cmd = Window.StartRenderPass(View.m_OverlayPass);
            Overlay(Cmd);
        }

        // The selection's outline last, over the overlays too: the host's grid is opaque floor and painted over the outline of an object standing on it
        { XPROFILE_SCOPE("Render: outline"); XPROFILE_GPU_SCOPE(Window, "Render: outline"); renderOutline(Window, Scene, View, W2C, Eye); }
        if( pTS ) Window.WriteTimestamp(*pTS, 6);

        const auto T3 = clock::now();
        m_CPUTimes = { Ms(T0, T1), Ms(T1, T2), m_LightBuildMs, Ms(T0, T3) };
        // Measurement only (XLION_LEDGER): the lighting ledger (documentation/lighting_strategy.md sec. 10), median, p95 and p99 of 120 frames
        // per row, per renderer (its view's size names it). Rows that do not exist yet (reconstruction, rays, cache maintenance) print 0;
        // direct shading is fused in the scene pass, so the scene pass is the row (paired variants attribute it). The GPU times are the
        // previous frames': each timing set once, whoever read it (getGPUTimes: the GI's log or the host may have)
        static const bool s_bLedger = xlionrender::Env("XLION_LEDGER") != nullptr;
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
                           , Scene.m_Sun.m_bCastShadows ? "on" : "off", m_VSMReport.m_RequestGpuMs, m_VSMReport.m_Pages.m_Rendered, m_VSMReport.m_Pages.m_RenderedPinned, m_VSMReport.m_Pages.m_Rejected, m_VSMReport.m_Pages.m_ResidentFine, m_VSMReport.m_Pages.m_ResidentPinned, m_VSM.getSetup().Slots(), m_VSMReport.m_Pages.m_PoolLoad, m_VSMReport.m_Pages.m_DynamicBias
                           , m_VSMReport.m_Counters[0] ? 100.0 * m_VSMReport.m_Counters[1] / m_VSMReport.m_Counters[0] : 0.0, m_VSMReport.m_Counters[0] ? 100.0 * m_VSMReport.m_Counters[2] / m_VSMReport.m_Counters[0] : 0.0
                           , m_VSMReport.m_Counters[0] ? 100.0 * m_VSMReport.m_Counters[3] / m_VSMReport.m_Counters[0] : 0.0);
                static const char* s_ModeNames[] = { "EXACT", "CELLS", "SAMPLED", "SCREEN", "AUTO" };
                const auto& TR = m_TailReport;           // the screen tail (P3: lighting_p3_design.md)
                std::printf("LEDGER screen tail: mode %s, %s %dx%d, G-buffer draws %d (left out, no variant: %d), G-buffer %.3f + tail %.3f + denoiser %.3f ms, %.1f MB\n"
                           , s_ModeNames[int(m_LightModeUsed)], TR.m_bOn ? "on" : "off", TR.m_Width, TR.m_Height, TR.m_nGBufferDraws, TR.m_nNoVariant, TR.m_GBufferMs, TR.m_TailMs, TR.m_DenoiseMs, TR.m_MemoryMB);
                const auto& LP = m_LocalReport;          // the local lights' pages (P1d)
                std::printf("LEDGER local pages: lights %d, requested %d, rendered %d + pinned %d, rejected %d, resident %d + pinned %d, stale %d; pixel-light pairs by their level %.1f%% / coarser %.1f%% / none %.1f%%\n"
                           , LP.m_Pages.m_Lights, LP.m_Pages.m_Requested, LP.m_Pages.m_Rendered, LP.m_Pages.m_RenderedPinned, LP.m_Pages.m_Rejected, LP.m_Pages.m_Resident, LP.m_Pages.m_ResidentPinned, LP.m_Pages.m_Stale
                           , LP.m_Counters[0] ? 100.0 * LP.m_Counters[1] / LP.m_Counters[0] : 0.0, LP.m_Counters[0] ? 100.0 * LP.m_Counters[2] / LP.m_Counters[0] : 0.0, LP.m_Counters[0] ? 100.0 * LP.m_Counters[3] / LP.m_Counters[0] : 0.0);
                std::fflush(stdout);
            }
        }
        // Measurement only (XLION_FRAME_TIMES): Render's CPU parts, the mean of 120 frames
        static const bool s_bFrameTimes = xlionrender::Env("XLION_FRAME_TIMES") != nullptr;
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
