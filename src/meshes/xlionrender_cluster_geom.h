#ifndef XLIONRENDER_CLUSTER_GEOM_H
#define XLIONRENDER_CLUSTER_GEOM_H
#pragma once

// The xgeom_* clustered, quantized meshes, drawn straight from the buffers their resource loaders already created - the
// renderer does not own or copy them. Two vertex paths (the geometry decides, the material is the fragment shader):
//  RIGID: xgeom_static, decoded by shaders/xlionrender_static_geom_vert.glsl
//  SKIN:  xgeom_skin,   skinned by shaders/xlionrender_skin_geom_vert.glsl with the bone matrices the caller uploads
// ponytail: skinning per pass in the vertex shader; a compute pre-skin (one rigid path for every pass, previous positions for
// motion vectors, RT refits) once xGPU has compute - see documentation/xlion_renderer_research.md.

#include "dependencies/xGPU/source/xGPU.h"
#include "dependencies/xmath/source/xmath.h"
#include <algorithm>
#include <vector>

namespace xlionrender
{
    enum class geom_kind : std::uint8_t { RIGID, SKIN, COUNT };

    struct cluster_geom
    {
        struct cluster
        {
            std::uint32_t   m_iIndex;
            std::uint32_t   m_nIndices;
            std::uint32_t   m_iVertex;
            xmath::fbbox    m_BBox;                 // local bounds (the resource's cluster bounds): the editor's frame budget draws a cluster not drawn as its box
        };

        struct submesh
        {
            std::uint32_t   m_iCluster;             // first entry in m_Clusters, also its index in the cluster SSBO
            std::uint32_t   m_nClusters;
            std::uint32_t   m_iMaterial;            // the geom's material slot
        };

        geom_kind                   m_Kind          = geom_kind::RIGID;
        xgpu::buffer*               m_pStreams      = nullptr;  // 3 contiguous: index (uint16), positions, extras - the core streams
        xgpu::buffer*               m_pColors       = nullptr;  // the optional COLOR0 stream (RGBA8 sRGB per vertex); null: the renderer binds constant white
        xgpu::buffer*               m_pUV1          = nullptr;  // the optional UV1 stream (float2 per vertex); null: the renderer binds a constant, the shaders read UV0
        xgpu::buffer*               m_pClusterData  = nullptr;  // SSBO of the per-cluster dequantization data
        xgpu::buffer*               m_pBones        = nullptr;  // SKIN: SSBO of the bone matrices (the caller fills it each frame), every bone of the skeleton
        xgpu::buffer*               m_pBoneMap      = nullptr;  // SKIN: SSBO of the palettes (xgeom_skin BoneMap); null: the renderer binds a dummy
        std::uint32_t               m_MaxInfluences = 4;        // SKIN: 1..4 bone weights used
        std::uint32_t               m_PrevBones     = 0;        // SKIN: last frame's pose in m_pBones from this matrix on (the screen tail's motion); 0: none (this frame's)
        xmath::fbbox                m_BBox;                     // local bounds (shadow culling and invalidation); SKIN: covering every pose
        std::vector<cluster>        m_Clusters;                 // every LOD's (the cluster ids are the geometry's, across its LODs)
        std::vector<submesh>        m_Submeshes;                // the drawn LOD's (MakeClusterGeom's iLOD)
        // Per LOD k (0 the finest): every mesh's submeshes at k (a mesh with fewer repeats its coarsest) - the same count and order at each
        // k (submesh s is the same material slot): a pass may draw a coarser LOD (the shadow pages by their texel size)
        std::vector<std::vector<submesh>> m_LODSubmeshes;
        const std::vector<submesh>& LODSubmeshes( int k ) const noexcept { return m_LODSubmeshes.empty() ? m_Submeshes : m_LODSubmeshes[std::size_t(std::clamp(k, 0, int(m_LODSubmeshes.size()) - 1))]; }
        // The main view's LOD pick (instance: by projected error): per LOD k each submesh's error, object space (its mesh's LOD error: the
        // geometry's lod::m_Error, 0 without one), and per submesh s the box of its LOD 0 clusters (a LOD group: xgeom_static's compiler)
        std::vector<std::vector<float>>   m_LODErrors;
        std::vector<xmath::fbbox>         m_SubBoxes;
        // the coarsest LOD of submesh s whose error, Scale times larger (the draw's), stays under MaxPixels at Distance (ProjScale: pixels per
        // unit at distance 1)
        int PickLOD( std::size_t s, float Distance, float Scale, float ProjScale, float MaxPixels ) const noexcept
        {
            int k = 0;
            while( std::size_t(k + 1) < m_LODErrors.size() && s < m_LODErrors[std::size_t(k + 1)].size()
                && m_LODErrors[std::size_t(k + 1)][s] * Scale * ProjScale <= MaxPixels * Distance ) ++k;
            return k;
        }
    };

    // From an xgeom_static::xgpu::geom or an xgeom_skin::xgpu::geom (templated: no xgeom / resource includes here), each mesh at
    // LOD iLOD (clamped). A skin geom also needs its bone buffer: MakeSkinGeom.
    template<typename T_GEOM>
    cluster_geom MakeClusterGeom( T_GEOM& Geom, int iLOD = 0 ) noexcept
    {
        cluster_geom G{ .m_pStreams = &Geom.IndexBuffer(), .m_pColors = Geom.getVertexColors().empty() ? nullptr : &Geom.ColorBuffer(), .m_pUV1 = Geom.getUV1s().empty() ? nullptr : &Geom.UV1Buffer(), .m_pClusterData = &Geom.ClusterBuffer(), .m_BBox = Geom.m_BBox };

        for( auto& C : Geom.getClusters() ) G.m_Clusters.push_back({ C.m_iIndex, C.m_nIndices, C.m_iVertex, C.m_BBox });
        int nLODs = 1;
        for( auto& M : Geom.getMeshes() ) nLODs = std::max(nLODs, int(M.m_nLODs));
        G.m_LODSubmeshes.resize(std::size_t(nLODs));
        G.m_LODErrors.resize(std::size_t(nLODs));
        for( int k = 0; k < nLODs; ++k )
            for( auto& M : Geom.getMeshes() )
            {
                if( M.m_nLODs == 0 ) continue;
                const auto& L = Geom.getLODs()[M.m_iLOD + std::clamp(k, 0, int(M.m_nLODs) - 1)];
                float Error = 0;
                if constexpr( requires { L.m_Error; } ) Error = L.m_Error;      // (xgeom_static's; another geometry's LODs: 0, never picked by error)
                for( auto& Sub : Geom.getSubmeshes().subspan(L.m_iSubmesh, L.m_nSubmesh) )
                {
                    G.m_LODSubmeshes[std::size_t(k)].push_back({ Sub.m_iCluster, Sub.m_nCluster, Sub.m_iMaterial });
                    G.m_LODErrors[std::size_t(k)].push_back(k == 0 ? 0.0f : Error > 0 ? Error : 1e30f);     // (no error recorded: never picked)
                }
            }
        for( const auto& Sub : G.m_LODSubmeshes[0] )
        {
            xmath::fbbox B{ xmath::fvec3(1e30f), xmath::fvec3(-1e30f) };
            for( std::uint32_t c = Sub.m_iCluster; c < Sub.m_iCluster + Sub.m_nClusters && c < G.m_Clusters.size(); ++c )
            {
                const auto& CB = G.m_Clusters[c].m_BBox;
                B.m_Min = xmath::fvec3::Min(B.m_Min, CB.m_Min); B.m_Max = xmath::fvec3::Max(B.m_Max, CB.m_Max);
            }
            G.m_SubBoxes.push_back(B);
        }
        G.m_Submeshes = G.m_LODSubmeshes[std::size_t(std::clamp(iLOD, 0, nLODs - 1))];
        return G;
    }

    template<typename T_GEOM>
    cluster_geom MakeSkinGeom( T_GEOM& Geom, xgpu::buffer& Bones, int iLOD = 0, int MaxInfluences = 4 ) noexcept
    {
        cluster_geom G = MakeClusterGeom(Geom, iLOD);
        G.m_Kind          = geom_kind::SKIN;
        G.m_pBones        = &Bones;
        G.m_pBoneMap      = &Geom.BoneMapBuffer();
        G.m_MaxInfluences = static_cast<std::uint32_t>(std::clamp(MaxInfluences, 1, 4));
        // ponytail: the bind-pose bounds grown 2x around their center to cover the poses; exact bounds want the skinned positions (compute skinning)
        const auto C = G.m_BBox.getCenter(), E = (G.m_BBox.m_Max - G.m_BBox.m_Min);
        G.m_BBox.m_Min = C - E;
        G.m_BBox.m_Max = C + E;
        return G;
    }
}

#endif
