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
            m_DrawChanged.assign(Scene.m_Draws.size(), 0);
            for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
            {
                const auto&         D  = Scene.m_Draws[i];
                const std::uint32_t ID = static_cast<std::uint32_t>(D.m_Instance);
                bool bChanged = true;
                std::uint32_t Slot = ID ? m_InstanceSlots.Acquire(ID, D.m_L2W, Frame, D.m_Material, &bChanged) : instance_slots::none_v;
                m_DrawChanged[i] = bChanged;
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

    // Surfel GI (features/gi/xlionrender_surfel_gi.h): the draws of meshes with surfels and a persistent id become world instances
    // (added, moved, removed when no longer drawn), the scene's lights its lights, one Update on the ray budget; then every drawn
    // instance's active-level light into this frame's region of the GI light ring (m_DrawGI: where, which level)
    void instance::updateGI( const scene& Scene, const xmath::fvec3& Eye, float ProjScale, float ViewPixels ) noexcept
    {
        m_DrawGI.assign(Scene.m_Draws.size(), { 0u, 0xFFFFFFFFu });
        if( !Scene.m_GI.m_bEnable ) return;
        const auto TGI = std::chrono::steady_clock::now();
        const auto MsSince = [](std::chrono::steady_clock::time_point T) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T).count(); };
        if( !m_pGIWorld ) m_pGIWorld = std::make_unique<gi::world>();
        auto& W = *m_pGIWorld;
        ++m_GIRender;

        std::vector<int> DrawId(Scene.m_Draws.size(), -1);
        for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
        {
            const auto&         D  = Scene.m_Draws[i];
            const std::uint32_t ID = std::uint32_t(D.m_Instance);
            const auto*         pM = findMesh(D.m_Mesh);
            if( ID == 0 || pM == nullptr || !pM->m_pGI ) continue;
            auto [It, bNew] = m_GIIds.try_emplace(ID, -1);
            // New surfels (a recompile): by the stamp, not the address - the new ones can be allocated where the old ones were
            if( It->second >= 0 && m_GITrack[It->second].m_GIStamp != pM->m_GIStamp ) { W.Remove(It->second); It->second = -1; }
            // What the GI takes from the draw's materials: which slots it sees through (transparent ones), and its colors (captured)
            const auto TakeMaterials = [&](int Id)
            {
                std::vector<bool> Alpha(D.m_Materials.size(), false);
                for( std::size_t k = 0; k < D.m_Materials.size(); ++k ) if( const auto* pMat = findMaterial(D.m_Materials[k]); pMat && pMat->m_bAlpha ) Alpha[k] = true;
                W.SetSeeThrough(Id, Alpha);
                const_cast<mesh_slot*>(pM)->m_bTriColors = false;              // the SURFACE_COLOR view's table: made again from the new capture
                // a capture still waiting for this instance is of materials it no longer has (one in flight does not publish: processGICapture)
                std::erase_if(m_GICaptures, [&](const gi_capture_job& J) { return J.m_GIId == Id && !J.m_bRecorded; });
                if( !pM->m_pGI->m_SurfelMaterial.empty() )      // its surfels' and its triangles' colors: captured from its materials (processGICapture)
                {
                    m_GICaptures.push_back({ .m_GIId = Id, .m_Mesh = D.m_Mesh, .m_Materials = D.m_Materials, .m_Slot = i < m_DrawSlot.size() ? m_DrawSlot[i] : 0u });
                    for( std::size_t t = 0; t < pM->m_pGI->m_Tris.size(); t += gi_capture_max_v )
                        m_GICaptures.push_back({ .m_GIId = Id, .m_Mesh = D.m_Mesh, .m_Materials = D.m_Materials, .m_Slot = i < m_DrawSlot.size() ? m_DrawSlot[i] : 0u
                                               , .m_bTriangles = true, .m_First = std::uint32_t(t) });
                }
            };
            if( It->second < 0 )
            {
                const auto& Mat = D.m_Material;
                It->second = W.Add(*pM->m_pGI, D.m_L2W, Mat.m_BaseColor * (1 - Mat.m_Metallic));
                m_GITrack.resize(std::max<std::size_t>(m_GITrack.size(), It->second + 1));
                m_GITrack[It->second] = { D.m_L2W, pM->m_pGI.get(), pM->m_GIStamp, 0, D.m_Materials };
                TakeMaterials(It->second);
            }
            else
            {
                if( m_GITrack[It->second].m_Materials != D.m_Materials )      // its materials were made again (a material compiled again) or changed
                {
                    m_GITrack[It->second].m_Materials = D.m_Materials;
                    TakeMaterials(It->second);
                }
                if( std::memcmp(&m_GITrack[It->second].m_L2W, &D.m_L2W, sizeof(xmath::fmat4)) != 0 )
                {
                    W.SetTransform(It->second, D.m_L2W);
                    m_GITrack[It->second].m_L2W = D.m_L2W;
                }
            }
            m_GITrack[It->second].m_Render = m_GIRender;
            DrawId[i] = It->second;
        }
        for( auto It = m_GIIds.begin(); It != m_GIIds.end(); )
        {
            if( m_GITrack[It->second].m_Render != m_GIRender ) { W.Remove(It->second); It = m_GIIds.erase(It); }
            else ++It;
        }

        // The lights: the sun, the sky (or the flat ambient as a sky), the local lights as points (ponytail: spots as points)
        gi::lights L;
        L.m_SunDirection = Scene.m_Sun.m_Direction.NormalizeCopy();
        L.m_SunColor     = Scene.m_Sun.m_Color * Scene.m_Sun.m_Intensity;
        L.m_Sky          = Scene.m_Sky.m_Intensity <= 0 ? Scene.m_Ambient * (1.0f / gi::pi_v)
                         : Scene.m_Sky.m_pCubemap      ? m_EnvMean * Scene.m_Sky.m_Intensity                                  // (updateEnvironment)
                         :                               (Scene.m_Sky.m_Zenith + Scene.m_Sky.m_Horizon) * (0.5f * Scene.m_Sky.m_Intensity);
        if( Scene.m_Sky.m_Intensity > 0 && Scene.m_Sky.m_pCubemap && m_bEnvOct )
        {
            for( std::size_t k = 0; k < L.m_SkyOct.size(); ++k ) L.m_SkyOct[k] = m_EnvOct[k] * Scene.m_Sky.m_Intensity;
            L.m_bSkyOct = true;
        }
        L.m_Points.reserve(Scene.m_LocalLights.size());
        for( const auto& LL : Scene.m_LocalLights ) L.m_Points.push_back({ LL.m_Position, LL.m_Color * LL.m_Intensity, LL.m_Radius });
        W.m_Setup.m_FilterPasses  = std::clamp(Scene.m_GI.m_FilterPasses, 0, 16);   // what is shown only: no re-solve
        W.m_Setup.m_RaysPerSurfel = std::clamp(Scene.m_GI.m_RaysPerSurfel, 1, 256);
        W.m_Setup.m_Saturation    = std::clamp(Scene.m_GI.m_Saturation, 0.0f, 4.0f);     // what is shown only: no re-solve
        W.m_Setup.m_ProjScale     = ProjScale;
        W.m_Setup.m_ViewPixels    = ViewPixels;
        W.m_Setup.m_PixelsPerSurfel = std::max(Scene.m_GI.m_PixelsPerSurfel, 0.0f);
        if( W.m_Setup.m_EmissiveScale != Scene.m_GI.m_EmissiveScale )          // a new scale: every instance re-solves
        {
            W.m_Setup.m_EmissiveScale = Scene.m_GI.m_EmissiveScale;
            W.DirtyAll();
        }
        m_GIStats = W.Update(L, Eye, Scene.m_GI.m_RayBudget, std::uint32_t(m_GIRender));
        const auto TUpload = std::chrono::steady_clock::now();

        // The packed moments (gi_track::m_Moments): the surfels this update changed (each a mean and a mean^2 per texel in two 16-bit floats)
        const auto PackMoments = [&](std::uint32_t* pOut, const gi::moments& M)
        {
            for( int t = 0; t < gi::moment_texels_v; ++t ) pOut[t] = details::Half(M[2 * t]) | (std::uint32_t(details::Half(M[2 * t + 1])) << 16);
        };
        const auto TMoments = std::chrono::steady_clock::now();
        for( const auto& [i, s] : W.LastUpdated() )
        {
            if( std::size_t(i) >= m_GITrack.size() ) continue;
            auto& T = m_GITrack[i];
            const auto& M = W.InstanceMoments(i);
            if( T.m_MomentsReset == W.MomentsReset(i) && T.m_Moments.size() == M.size() * gi::moment_texels_v ) PackMoments(T.m_Moments.data() + std::size_t(s) * gi::moment_texels_v, M[s]);
        }

        // This frame's light region (several renders per frame append)
        const std::uint64_t Frame = m_Device.getFrameIndex();
        if( Frame != m_GIFrame ) { m_GIFrame = Frame; m_GIRingUsed = 0; m_GIFrameRegion.clear(); }
        const int Base = int(Frame % ring_frames_v) * gi_light_region_v;
        std::vector<std::pair<int, int>> Written;          // (first surfel of the region, instance): their moments next
        const int Start = m_GIRingUsed;
        m_GIStats.m_MomentsMs += MsSince(TMoments);
        double FilterMs = 0;
        const auto TRing = std::chrono::steady_clock::now();
        (void)m_GILightRing.MemoryMap(std::size_t(Base + m_GIRingUsed) * 3, std::size_t(gi_light_region_v - m_GIRingUsed) * 3, [&](void* pMemory)     // 3 vec4 a surfel
        {
            auto* pOut = static_cast<xmath::fvec4*>(pMemory);
            int   Used = 0;
            // scene::gi_debug in the shaders' bits (xlionrender_surfel_gi.glsl GIDebugCode): 1..3 at bit 9, 4 and 8 at bits 12 and 13
            const std::uint32_t GIDebugBits = ((std::uint32_t(Scene.m_GI.m_Debug) & 3u) << 9) | ((std::uint32_t(Scene.m_GI.m_Debug) & 0xCu) << 10)
                                            | (Scene.m_GI.m_bMoments ? 0u : 0x4000u);                   // gi_no_moments_v
            for( std::size_t i = 0; i < Scene.m_Draws.size(); ++i )
            {
                const int Id = DrawId[i];
                if( Id < 0 ) continue;
                // One upload a frame: the frame's other renders (the probes' faces, other views) read the same region. Each used to
                // upload its own copy and the frame's region ran out: a render without GI, a different one every frame (flashing)
                if( auto It = m_GIFrameRegion.find(Id); It != m_GIFrameRegion.end() )
                {
                    m_DrawGI[i] = { std::uint32_t(Base + It->second), std::uint32_t(W.Level(Id)) | (Scene.m_GI.m_bPerPixel ? 0x100u : 0u) | GIDebugBits };
                    continue;
                }
                const auto TFilter = std::chrono::steady_clock::now();
                const auto& E = W.BasisLight(Id);              // shown: the HL2 basis, spatially filtered (the solver keeps its own)
                FilterMs += MsSince(TFilter);
                const auto& V = W.InstanceValid(Id);
                if( m_GIRingUsed + Used + 1 + int(E.size()) > gi_light_region_v ) { ++m_GIStats.m_nDropped; continue; }  // ponytail: no GI for what does not fit
                // The slot before its surfels: its mean light, what a pixel no read supports shows (SurfelGIMean: the solver's policy too,
                // gi::world::Unsupported). Then its surfels
                const auto& Mean = W.MeanLight(Id);
                for( int r = 0; r < 3; ++r ) pOut[3 * Used + r] = xmath::fvec4(Mean, 1);
                ++Used;
                for( std::size_t k = 0; k < E.size(); ++k )
                {
                    pOut[3 * (Used + k) + 0] = xmath::fvec4(E[k][0], V[k]);
                    pOut[3 * (Used + k) + 1] = xmath::fvec4(E[k][1], 0);
                    pOut[3 * (Used + k) + 2] = xmath::fvec4(E[k][2], 0);
                }
                Written.push_back({ Used, Id });
                m_GIFrameRegion[Id] = m_GIRingUsed + Used;
                m_DrawGI[i] = { std::uint32_t(Base + m_GIRingUsed + Used), std::uint32_t(W.Level(Id)) | (Scene.m_GI.m_bPerPixel ? 0x100u : 0u) | GIDebugBits };     // xlionrender_surfel_gi.glsl gi_per_pixel_v, gi_debug_shift_v
                Used += int(E.size());
            }
            m_GIRingUsed += Used;
            m_GIStats.m_UploadBytes += std::size_t(Used) * 3 * sizeof(xmath::fvec4);
        });
        m_GIStats.m_FilterMs   = FilterMs;
        m_GIStats.m_RingCopyMs = MsSince(TRing) - FilterMs;
        const auto TMomentsCopy = std::chrono::steady_clock::now();

        // The surfels' depth moments at the same index (gi::moment_texels_v per surfel, each a mean and a mean^2 in two 16-bit floats)
        if( m_GIRingUsed > Start )
            (void)m_GIMomentRing.MemoryMap(std::size_t(Base + Start) * gi::moment_texels_v, std::size_t(m_GIRingUsed - Start) * gi::moment_texels_v, [&](void* pMemory)
            {
                auto* pOut = static_cast<std::uint32_t*>(pMemory);
                for( const auto& [First, Id] : Written )
                {
                    auto& T = m_GITrack[Id];
                    const auto& M = W.InstanceMoments(Id);
                    if( T.m_MomentsReset != W.MomentsReset(Id) || T.m_Moments.size() != M.size() * gi::moment_texels_v )     // all of them again
                    {
                        T.m_Moments.resize(M.size() * gi::moment_texels_v);
                        for( std::size_t k = 0; k < M.size(); ++k ) PackMoments(T.m_Moments.data() + k * gi::moment_texels_v, M[k]);
                        T.m_MomentsReset = W.MomentsReset(Id);
                    }
                    std::memcpy(pOut + std::size_t(First) * gi::moment_texels_v, T.m_Moments.data(), T.m_Moments.size() * sizeof(std::uint32_t));
                }
            });
        m_GIStats.m_MomentsMs += MsSince(TMomentsCopy);
        m_GIStats.m_UploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - TUpload).count();
        m_GIStats.m_UpdateGIMs = MsSince(TGI);
        m_GIStats.m_LoadMs = m_GILoadMsAcc; m_GILoadMsAcc = 0;
        m_GIFrames.push_back({ m_Device.getFrameIndex(), m_GIStats });
        if( m_GIFrames.size() > gi_frame_log_v ) m_GIFrames.erase(m_GIFrames.begin(), m_GIFrames.begin() + std::ptrdiff_t(m_GIFrames.size() - gi_frame_log_v));
    }

    //------------------------------------------------------------------------------------------------

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
                    Out.m_Shadow      = { float(shadows::atlas_tiles_v * m_ShadowSetup.m_ClipmapTileSize + Map.m_X), float(Map.m_Y), float(Map.m_Size), bSpot ? shadows::SpotTanHalf(LL) : 1.0f };
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
        struct draw_caster { const draw* m_pDraw; const mesh_slot* m_pMesh; xmath::fbbox m_World; std::uint32_t m_Slot; };
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

            const bool   bSkin = pMesh->m_bCluster && pMesh->m_Geom.m_Kind == geom_kind::SKIN;
            xmath::fbbox World;
            if( Slot < std::uint32_t(max_instances_v) )
            {
                auto&               C     = m_SlotCasters[Slot];
                const std::uint64_t MHash = MaterialsHash(D);
                const bool bChanged = m_DrawChanged[iDraw] || !C.m_bValid || bSkin || C.m_Mesh.m_Index != D.m_Mesh.m_Index
                                   || C.m_Mesh.m_Generation != D.m_Mesh.m_Generation || C.m_MaterialsHash != MHash;
                if( bChanged )
                {
                    if( C.m_bValid ) Changed.push_back(C.m_World);
                    C = { WorldOf(*pMesh, D), D.m_Mesh, MHash, Pass, true };
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
                World = WorldOf(*pMesh, D);
                Now.emplace(Key, World);
                const auto It = m_PrevCasters.find(Key);
                if( It == m_PrevCasters.end() || bSkin ) Changed.push_back(World);
                if( It != m_PrevCasters.end() && bSkin ) Changed.push_back(It->second);
            }
            m_DrawWorld[iDraw] = World;
            Draws.push_back({ &D, pMesh, World, Slot });
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
        if( Sun.m_bCastShadows ) m_Clipmap.Update(Sun.m_Direction, Sun.m_ShadowTileSize, Eye, Changed, m_bShadowCache, m_ShadowTiles);
        else                     m_ShadowTiles.clear();
        m_LocalShadows.Update(Scene.m_LocalLights, Changed, m_bShadowCache, Eye, ProjScale, m_ShadowUpdatesPerFrame, m_LocalMaps, m_LocalToRender);

        m_nShadowTilesRendered = static_cast<int>(m_ShadowTiles.size() + m_LocalToRender.size());
        if( m_nShadowTilesRendered == 0 ) return;

        // Draws the casters that pass Keep with W2C into the current viewport
        // Draws the casters of Indices that pass Keep with W2C into the current viewport
        const auto DrawCasters = [&]( xgpu::cmd_buffer& Cmd, const xmath::fmat4& W2C, std::span<const std::uint32_t> Indices, const auto& Keep )
        {
            auto& ViewBlock = m_ShadowUBO.allocEntry<details::ubo_geom_view>();     // absolute world: the instances' origin as is
            ViewBlock.m_w2C         = W2C;
            ViewBlock.m_WorldOffset = xmath::fvec4(m_Origin, 0);
            const cluster_geom* pBoundGeom = nullptr;       // binds only when the mesh changes

            for( const std::uint32_t i : Indices )
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
        std::vector<std::uint32_t> LightCasters;        // a local light's casters (reused per light)
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
                std::vector<shadows::lbox>  LightBounds;        // the clipmap culls in light space (only when a tile renders)
                std::vector<std::uint32_t>  AllCasters;
                if( !m_ShadowTiles.empty() )
                {
                    LightBounds.reserve(Draws.size());
                    AllCasters.reserve(Draws.size());
                    for( std::uint32_t i = 0; i < Draws.size(); ++i ) { LightBounds.push_back(m_Clipmap.Bounds(Draws[i].m_World)); AllCasters.push_back(i); }
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
            auto Cmd = Window.StartRenderPass(View.m_ScenePass);
            if( bMain ) Cmd.WriteTimestamp(m_Timestamps, 1);

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
                std::array<xgpu::buffer*, 8> StaticBuffers = bSkin ? std::array<xgpu::buffer*, 8>{ G.m_pClusterData, G.m_pBones, &m_InstanceBuffer, &m_LightRing, &m_CellRing, &m_GIDummy, &m_GILightRing, &m_GIMomentRing }
                                                                   : std::array<xgpu::buffer*, 8>{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing, nullptr, nullptr, nullptr, nullptr };
                const auto  Static = std::span{ StaticBuffers.data(), bSkin ? std::size_t{8} : std::size_t{4} };       // SKIN: the GI bindings of the material fragment (no surfels)
                std::array<xgpu::buffer*, 7> GIBuffers{ G.m_pClusterData, &m_InstanceBuffer, &m_LightRing, &m_CellRing
                                                      , pMesh->m_pGI ? const_cast<xgpu::buffer*>(&pMesh->m_GIBuffer) : &m_GIDummy, &m_GILightRing, &m_GIMomentRing };
                const auto  GI = iDraw < m_DrawGI.size() ? m_DrawGI[iDraw] : std::pair<std::uint32_t, std::uint32_t>{ 0u, 0xFFFFFFFFu };

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
                        if( !bSkin ) Cmd.setPipelineInstance(*pInstance, std::span{ GIBuffers });     // every rigid pipeline reads the GI
                        else         Cmd.setPipelineInstance(*pInstance, Static);
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
                        else        Cmd.setPushConstants(details::push_static_geom{ c, Slot, GI.first, GI.second });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                    }
                }
            }

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
        if( pM == nullptr || !pM->m_pGI ) { m_GICaptures.pop_front(); return; }     // (its buffer goes with it)
        const auto& G     = *pM->m_pGI;
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
            const auto& L0 = pM->m_pGI->m_Levels[0];
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
            std::array<xgpu::buffer*, 7> Buffers{ &J.m_Data, &m_InstanceBuffer, &m_LightRing, &m_CellRing, const_cast<xgpu::buffer*>(&pM->m_GIBuffer), &m_GILightRing, &m_GIMomentRing };
            for( int a = 0; a < n; )
            {
                const std::uint32_t Slot = Mats[J.m_Order[a]].m_iMaterial;
                int b = a;
                while( b < n && Mats[J.m_Order[b]].m_iMaterial == Slot ) ++b;
                auto* pMat  = Slot < J.m_Materials.size() ? findMaterial(J.m_Materials[Slot]) : nullptr;
                auto* pInst = pMat ? getCaptureInstance(*pMat) : &m_GIConstCaptureInstance;
                if( pInst == nullptr ) pInst = &m_GIConstCaptureInstance;
                Cmd.setPipelineInstance(*pInst, std::span{ Buffers });
                Cmd.setDynamicUBO(m_LightUBO, 1);
                Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                Cmd.setBuffer(m_GICaptureVB);
                Cmd.setBuffer(m_GICaptureIB);
                for( int Pass = 0; Pass < 2; ++Pass )
                {
                    Cmd.setPushConstants(details::push_static_geom{ Pass ? std::uint32_t(-n) : 0u, J.m_Slot, 0u, (3u << 9) | (Pass ? 0x800u : 0u) });
                    Cmd.Draw(6 * (b - a), 6 * a, Pass * 4 * n);
                }
                a = b;
            }
            J.m_bRecorded = true;
            J.m_Frame     = Frame;
            return;
        }

        if( Frame < J.m_Frame + ring_frames_v + 1 ) return;         // the GPU is done with it after the frames in flight
        std::vector<std::byte> Raw;
        int W = 0, H = 0;
        if( m_Device.ReadTexture(m_GICaptureTex, Raw, W, H) == nullptr && W == gi_capture_w_v && H == gi_capture_h_v )
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
            // Captured from materials the instance no longer has (they were made again while this was in flight): not published - its new capture is queued
            const bool bCurrent = std::size_t(J.m_GIId) < m_GITrack.size() && m_GITrack[J.m_GIId].m_Materials == J.m_Materials;
            if( bCurrent && !J.m_bTriangles ) m_pGIWorld->SetSurfelMaterial(J.m_GIId, std::move(Albedo), std::move(Emission));
            else if( bCurrent )
            {
                m_pGIWorld->SetTriangleAlbedo(J.m_GIId, J.m_First, Albedo);
                m_pGIWorld->SetTriangleEmission(J.m_GIId, J.m_First, Emission);
                if( J.m_First + std::size_t(n) >= G.m_Tris.size() )         // its last batch: the SURFACE_COLOR view's table (the mesh's first instance)
                {
                    const auto& All = m_pGIWorld->TriangleAlbedo(J.m_GIId);
                    if( auto* pSlot = const_cast<mesh_slot*>(pM); pSlot->m_GIBuffer.m_Private && pSlot->m_bTriColors == false )
                    {
                        std::vector<xmath::fvec3> C(All.begin(), All.end());
                        (void)buildMeshGI(*pSlot, C);
                        pSlot->m_bTriColors = true;
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
                std::array<xgpu::buffer*, 3> StaticBuffers{ G.m_pClusterData, bSkin ? G.m_pBones : &m_InstanceBuffer, &m_InstanceBuffer };
                Cmd.setPipelineInstance(View.m_MaskInstance[bSkin ? 2 : 1], std::span{ StaticBuffers.data(), bSkin ? std::size_t{3} : std::size_t{2} });
                Cmd.setDynamicUBO(m_ShadowUBO, 0);
                Cmd.setStreamingBuffers({ G.m_pStreams, 3 });
                const std::uint32_t Slot = m_DrawSlot[iDraw];
                for( const auto& Sub : G.m_Submeshes )
                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
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
        // The probe atlas and the shadow atlas are bound by every lit pipeline, the GI capture's below included: cleared before anything samples them
        // (the first frame's capture used to read them undefined: a Vulkan validation error)
        if( !m_bProbeAtlasReady ) { auto Cmd = Window.StartRenderPass(m_ProbeAtlasClearPass); m_bProbeAtlasReady = true; }
        if( !m_bShadowAtlasReady ) { auto Cmd = Window.StartRenderPass(m_ShadowClearPass); m_bShadowAtlasReady = true; }
        updateEnvironment(Window, Scene);
        updateInstances(Window, Scene, Eye);
        updateGI(Scene, Eye, ProjScale, float(View.m_Width) * float(View.m_Height));                                   // CPU (getGIStats); the probes' captures read it too
        {
            const auto TCapture = std::chrono::steady_clock::now();
            processGICapture(Window);                           // the surfels' material colors (one mesh at a time)
            if( !m_GIFrames.empty() ) m_GIFrames.back().m_Stats.m_CaptureMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - TCapture).count();
        }
        const auto T1 = clock::now();
        Window.WriteTimestamp(m_Timestamps, 0);                 // GPU: the shadow passes start here (after the instance copy)
        renderShadows(Window, Scene, Eye, ProjScale);
        updateProbes(Window, Scene, Eye);
        const auto T2 = clock::now();

        renderScenePass(Window, Scene, View, W2C, Eye, true);

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

        // The selection's outline last, over the overlays too: the host's grid is opaque floor and painted over the outline of an object standing on it
        renderOutline(Window, Scene, View, W2C, Eye);

        const auto T3 = clock::now();
        m_CPUTimes = { Ms(T0, T1), Ms(T1, T2), m_LightBuildMs, Ms(T0, T3) };
    }
}
