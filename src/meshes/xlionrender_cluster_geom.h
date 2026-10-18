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
        xgpu::buffer*               m_pStreams      = nullptr;  // 3 contiguous: index (uint16), positions, extras - as setStreamingBuffers takes them
        xgpu::buffer*               m_pClusterData  = nullptr;  // SSBO of the per-cluster dequantization data
        xgpu::buffer*               m_pBones        = nullptr;  // SKIN: SSBO of the bone matrices (the caller fills it each frame)
        std::uint32_t               m_MaxInfluences = 4;        // SKIN: 1..4 bone weights used
        xmath::fbbox                m_BBox;                     // local bounds (shadow culling and invalidation); SKIN: covering every pose
        std::vector<cluster>        m_Clusters;
        std::vector<submesh>        m_Submeshes;
    };

    // From an xgeom_static::xgpu::geom or an xgeom_skin::xgpu::geom (templated: no xgeom / resource includes here), each mesh at
    // LOD iLOD (clamped). A skin geom also needs its bone buffer: MakeSkinGeom.
    template<typename T_GEOM>
    cluster_geom MakeClusterGeom( T_GEOM& Geom, int iLOD = 0 ) noexcept
    {
        cluster_geom G{ .m_pStreams = &Geom.IndexBuffer(), .m_pClusterData = &Geom.ClusterBuffer(), .m_BBox = Geom.m_BBox };

        for( auto& C : Geom.getClusters() ) G.m_Clusters.push_back({ C.m_iIndex, C.m_nIndices, C.m_iVertex, C.m_BBox });
        for( auto& M : Geom.getMeshes() )
        {
            if( M.m_nLODs == 0 ) continue;
            const auto& L = Geom.getLODs()[M.m_iLOD + std::clamp(iLOD, 0, int(M.m_nLODs) - 1)];
            for( auto& Sub : Geom.getSubmeshes().subspan(L.m_iSubmesh, L.m_nSubmesh) )
                G.m_Submeshes.push_back({ Sub.m_iCluster, Sub.m_nCluster, Sub.m_iMaterial });
        }
        return G;
    }

    template<typename T_GEOM>
    cluster_geom MakeSkinGeom( T_GEOM& Geom, xgpu::buffer& Bones, int iLOD = 0, int MaxInfluences = 4 ) noexcept
    {
        cluster_geom G = MakeClusterGeom(Geom, iLOD);
        G.m_Kind          = geom_kind::SKIN;
        G.m_pBones        = &Bones;
        G.m_MaxInfluences = static_cast<std::uint32_t>(std::clamp(MaxInfluences, 1, 4));
        // ponytail: the bind-pose bounds grown 2x around their center to cover the poses; exact bounds want the skinned positions (compute skinning)
        const auto C = G.m_BBox.getCenter(), E = (G.m_BBox.m_Max - G.m_BBox.m_Min);
        G.m_BBox.m_Min = C - E;
        G.m_BBox.m_Max = C + E;
        return G;
    }
}

#endif
