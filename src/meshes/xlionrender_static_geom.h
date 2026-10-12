#ifndef XLIONRENDER_STATIC_GEOM_H
#define XLIONRENDER_STATIC_GEOM_H
#pragma once

// xgeom_static meshes (quantized, clustered) drawn by the renderer straight from the buffers the xgeom_static
// resource loader already created - the renderer does not own or copy them. Decoded by the same vertex shader the
// static geom editor uses (shaders/GeomStaticBasicShader_vert.glsl).

#include "dependencies/xGPU/source/xGPU.h"
#include <span>
#include <vector>

namespace xlionrender
{
    struct static_geom
    {
        struct cluster
        {
            std::uint32_t   m_iIndex;
            std::uint32_t   m_nIndices;
            std::uint32_t   m_iVertex;
        };

        struct submesh
        {
            std::uint32_t   m_iCluster;             // first entry in m_Clusters, also its index in the cluster SSBO
            std::uint32_t   m_nClusters;
            std::uint32_t   m_iMaterial;            // the geom's material slot
        };

        xgpu::buffer*               m_pStreams      = nullptr;  // 3 contiguous: index (uint16), positions, extras - as setStreamingBuffers takes them
        xgpu::buffer*               m_pClusterData  = nullptr;  // SSBO of the per-cluster dequantization data
        std::vector<cluster>        m_Clusters;
        std::vector<submesh>        m_Submeshes;
    };

    // From an xgeom_static::xgpu::geom (templated so this header needs no xgeom_static / resource includes).
    // ponytail: takes each mesh's first LOD; LOD selection comes with the view/culling work.
    template<typename T_GEOM>
    static_geom MakeStaticGeom( T_GEOM& Geom ) noexcept
    {
        static_geom S{ .m_pStreams = &Geom.IndexBuffer(), .m_pClusterData = &Geom.ClusterBuffer() };

        for( auto& C : Geom.getClusters() ) S.m_Clusters.push_back({ C.m_iIndex, C.m_nIndices, C.m_iVertex });
        for( auto& M : Geom.getMeshes() )
        {
            if( M.m_nLODs == 0 ) continue;
            const auto& L = Geom.getLODs()[M.m_iLOD];
            for( auto& Sub : Geom.getSubmeshes().subspan(L.m_iSubmesh, L.m_nSubmesh) )
                S.m_Submeshes.push_back({ Sub.m_iCluster, Sub.m_nCluster, Sub.m_iMaterial });
        }
        return S;
    }
}

#endif
