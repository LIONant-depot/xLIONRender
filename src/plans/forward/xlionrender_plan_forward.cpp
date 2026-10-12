// Plan A - forward. instance::Render lives here so each plan is one file to read.
#include "xlionrender_instance.h"
#include "plans/forward/xlionrender_plan_forward.h"

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

    // The directional light's shadow clipmap: which casters changed, which tiles must be rendered, then those tiles (each cleared and
    // drawn with the casters over it) into the atlas. Before the scene pass, which samples the atlas.
    void instance::renderShadows( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye ) noexcept
    {
        if( !m_bShadowAtlasReady )
        {
            auto Cmd = Window.StartRenderPass(m_ShadowClearPass);      // the atlas's first use: cleared to far, so it can be loaded and sampled
            m_bShadowAtlasReady = true;
        }

        m_nShadowTilesRendered = 0;
        const auto& L = Scene.m_Light;
        if( !L.m_bCastShadows || L.m_Radius > 0 ) return;

        struct draw_caster { const draw* m_pDraw; const mesh_slot* m_pMesh; };
        std::vector<shadows::caster> Casters;
        std::vector<draw_caster>     Draws;
        Casters.reserve(Scene.m_Draws.size());
        Draws.reserve(Scene.m_Draws.size());
        for( const auto& D : Scene.m_Draws )
        {
            const auto* pMesh = findMesh(D.m_Mesh);
            if( pMesh == nullptr ) continue;

            const bool bSkin = pMesh->m_bCluster && pMesh->m_Geom.m_Kind == geom_kind::SKIN;
            std::uint64_t Key = 1469598103934665603ull;
            Key = Hash(Key, &D.m_Mesh, sizeof(D.m_Mesh));
            Key = Hash(Key, &D.m_L2W, sizeof(D.m_L2W));
            Casters.push_back({ Key, m_Clipmap.Bounds(pMesh->m_bCluster ? pMesh->m_Geom.m_BBox : pMesh->m_BBox, D.m_L2W), bSkin });
            Draws.push_back({ &D, pMesh });
        }

        m_Clipmap.Update(L.m_Direction, L.m_ShadowTileSize, Eye, Casters, m_bShadowCache, m_ShadowTiles);
        if( m_ShadowTiles.empty() ) return;
        m_nShadowTilesRendered = static_cast<int>(m_ShadowTiles.size());

        static const xmath::fmat4 NoShadow = []{ xmath::fmat4 M = xmath::fmat4::fromZero(); M(2, 3) = 2; M(3, 3) = 1; return M; }();

        auto Cmd = Window.StartRenderPass(m_ShadowPass);
        for( const auto& T : m_ShadowTiles )
        {
            const float X = float(T.m_AtlasX * shadows::tile_size_v), Y = float(T.m_AtlasY * shadows::tile_size_v), S = float(shadows::tile_size_v);
            Cmd.setViewport(X, Y, S, S);
            Cmd.setScissor(int(X), int(Y), int(S), int(S));

            Cmd.setPipelineInstance(m_ShadowClearInstance);
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);

            for( std::size_t i = 0; i < Draws.size(); ++i )
            {
                if( !shadows::clipmap::Overlaps(Casters[i].m_Bounds, T.m_MinX, T.m_MinY, T.m_Size) ) continue;
                const auto& D = *Draws[i].m_pDraw;
                const auto& M = *Draws[i].m_pMesh;

                if( M.m_bCluster == false )
                {
                    Cmd.setPipelineInstance(m_ShadowInstance[0]);
                    Cmd.setPushConstants(details::push_forward{ D.m_L2W, T.m_W2C });
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(M.m_VertexBuffer));
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(M.m_IndexBuffer));
                    Cmd.Draw(M.m_nIndices);
                    continue;
                }

                const auto& G     = M.m_Geom;
                const bool  bSkin = G.m_Kind == geom_kind::SKIN;
                std::array<xgpu::buffer*, 2> StaticBuffers{ G.m_pClusterData, G.m_pBones };
                Cmd.setPipelineInstance(m_ShadowInstance[bSkin ? 2 : 1], std::span{ StaticBuffers.data(), bSkin ? std::size_t{2} : std::size_t{1} });

                auto& Mesh = m_ShadowUBO.allocEntry<details::ubo_static_geom_mesh>();
                Mesh.m_L2w        = D.m_L2W;
                Mesh.m_w2C        = T.m_W2C;
                Mesh.m_L2CShadowT = NoShadow;
                Cmd.setDynamicUBO(m_ShadowUBO, 0);
                Cmd.setStreamingBuffers({ G.m_pStreams, 3 });

                for( const auto& Sub : G.m_Submeshes )
                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
                        const auto& C = G.m_Clusters[c];
                        if( bSkin ) Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences });
                        else        Cmd.setPushConstants(details::push_static_geom{ c });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                    }
            }
        }
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

        // The vertex shaders' single-shadow-map output is unused (the clipmap is looked up from the world position): z = 2, outside (-1,1)
        static const xmath::fmat4 NoShadow = []{ xmath::fmat4 M = xmath::fmat4::fromZero(); M(2, 3) = 2; M(3, 3) = 1; return M; }();

        renderShadows(Window, Scene, Eye);

        // Scene -> HDR
        {
            auto Cmd = Window.StartRenderPass(View.m_ScenePass);
            Cmd.WriteTimestamp(m_Timestamps, 0);

            {
                const auto& L     = Scene.m_Light;
                auto&       Light = m_LightUBO.allocEntry<details::ubo_lighting>();
                Light.m_LightColor        = xmath::fvec4(L.m_Color * L.m_Intensity, 1);
                Light.m_AmbientLightColor = xmath::fvec4(Scene.m_Ambient, 0);
                Light.m_wSpaceLightPos    = L.m_Radius > 0 ? xmath::fvec4(L.m_Position - Eye, L.m_Radius)
                                                           : xmath::fvec4(-L.m_Direction.NormalizeCopy(), 0);
                Light.m_wSpaceEyePos      = xmath::fvec4(0, 0, 0, 1);       // camera-centered
                Light.m_LightParams       = { L.m_AreaRadius, L.m_Temperature, 1, 0 };

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
                Light.m_ShadowParams = { (L.m_bCastShadows && L.m_Radius == 0) ? 1.0f : 0.0f, m_Clipmap.DepthHalfRange(0), 0, 0 };
            }

            for( const auto& D : Scene.m_Draws )
            {
                const auto* pMesh = findMesh(D.m_Mesh);
                if( pMesh == nullptr ) continue;

                const xmath::fmat4 L2w = ToEye * D.m_L2W;

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
                    BindStandard(m_ForwardInstance, {});
                    Cmd.setPushConstants(details::push_forward{ L2w, W2C });
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_VertexBuffer));
                    Cmd.setBuffer(const_cast<xgpu::buffer&>(pMesh->m_IndexBuffer));
                    Cmd.Draw(pMesh->m_nIndices);
                    continue;
                }

                const auto& G     = pMesh->m_Geom;
                const bool  bSkin = G.m_Kind == geom_kind::SKIN;
                std::array<xgpu::buffer*, 2> StaticBuffers{ G.m_pClusterData, G.m_pBones };
                const auto  Static = std::span{ StaticBuffers.data(), bSkin ? std::size_t{2} : std::size_t{1} };

                auto& Mesh = m_StaticGeomUBO.allocEntry<details::ubo_static_geom_mesh>();
                Mesh.m_L2w        = L2w;
                Mesh.m_w2C        = W2C;
                Mesh.m_L2CShadowT = NoShadow;

                for( const auto& Sub : G.m_Submeshes )
                {
                    auto* pMat      = Sub.m_iMaterial < D.m_Materials.size() ? findMaterial(D.m_Materials[Sub.m_iMaterial]) : nullptr;
                    auto* pInstance = pMat ? getMaterialInstance(*pMat, G.m_Kind) : nullptr;
                    if( pInstance )
                    {
                        Cmd.setPipelineInstance(*pInstance, Static);
                        Cmd.setDynamicUBO(m_LightUBO, 1);
                    }
                    else if( bSkin )
                    {
                        continue;       // ponytail: no constant-material fallback on the skin path; the editors give every slot a material
                    }
                    else
                    {
                        BindStandard(m_StaticGeomInstance, Static);
                    }
                    Cmd.setDynamicUBO(m_StaticGeomUBO, 0);
                    Cmd.setStreamingBuffers({ G.m_pStreams, 3 });

                    for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters; ++c )
                    {
                        const auto& C = G.m_Clusters[c];
                        if( bSkin ) Cmd.setPushConstants(details::push_skin_geom{ c, G.m_MaxInfluences });
                        else        Cmd.setPushConstants(details::push_static_geom{ c });
                        Cmd.Draw(static_cast<int>(C.m_nIndices), static_cast<int>(C.m_iIndex), static_cast<int>(C.m_iVertex));
                    }
                }
            }

            Cmd.WriteTimestamp(m_Timestamps, 1);
        }

        // HDR -> exposure -> LionMapper -> sRGB output (alpha = coverage)
        {
            auto Cmd = Window.StartRenderPass(View.m_OutputPass);
            Cmd.setPipelineInstance(View.m_ToneMapInstance);
            Cmd.setPushConstants(details::push_tonemap{ { View.m_Exposure, 0, 0, 0 }, View.m_Background });
            Cmd.setBuffer(m_FullscreenVB);
            Cmd.setBuffer(m_FullscreenIB);
            Cmd.Draw(3);
            Cmd.WriteTimestamp(m_Timestamps, 2);
        }

        // Editor overlays over the tone-mapped image, depth-tested against the scene
        if( Overlay )
        {
            auto Cmd = Window.StartRenderPass(View.m_OverlayPass);
            Overlay(Cmd);
        }
    }
}
