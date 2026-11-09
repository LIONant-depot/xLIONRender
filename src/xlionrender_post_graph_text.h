#ifndef XLIONRENDER_POST_GRAPH_TEXT_H
#define XLIONRENDER_POST_GRAPH_TEXT_H
#pragma once

// The post chain's description as text (render graph P2; xLION_Docs/render_graph_p2_note_2026-11-09.md): one xtextfile table, the
// authored nodes in order by their semantic names. The renderer never reads files - a view holds a post::desc; this reader is for
// the tests now and the render graph resource's compiler later (one reader).
//      [ PostGraph : 4 ]
//      { Type:s }
//        "post.taa"
//        "post.bloom"
//        "post.exposure"
//        "post.tonemap"
#include "xlionrender_post_graph.h"
#include "dependencies/xtextfile/source/xtextfile.h"

namespace xlionrender::post
{
    inline constexpr std::pair<std::string_view, node_type> type_names_v[] = { { "post.taa", node_type::TAA }, { "post.bloom", node_type::BLOOM }
                                                                             , { "post.exposure", node_type::EXPOSURE }, { "post.tonemap", node_type::TONE_MAP } };

    // Reads or writes D. Problem: why a row was refused (an unknown type names its row); the file's own errors come back as xerr
    inline xerr Serialize( xtextfile::stream& F, desc& D, std::string& Problem ) noexcept
    {
        std::vector<std::string> Names;
        if( !F.isReading() ) for( const auto& N : D.m_Nodes ) for( const auto& [Name, T] : type_names_v ) if( T == N.m_Type ) Names.emplace_back(Name);
        if( auto Err = F.Record("PostGraph"
            , [&](std::size_t& Count, xerr&) { if( F.isReading() ) Names.resize(Count); else Count = Names.size(); }
            , [&](std::size_t i, xerr& E) { E = F.Field("Type", Names[i]); }); Err ) return Err;
        if( F.isReading() )
        {
            D.m_Nodes.clear();
            for( std::size_t i = 0; i < Names.size(); ++i )
            {
                const auto It = std::find_if(std::begin(type_names_v), std::end(type_names_v), [&](const auto& P) { return P.first == Names[i]; });
                if( It == std::end(type_names_v) ) { Problem = std::format("row {}: no node type '{}' (the types: post.taa, post.bloom, post.exposure, post.tonemap)", i, Names[i]); D.m_Nodes.clear(); return {}; }
                D.m_Nodes.push_back({ It->second });
            }
        }
        return {};
    }
}
#endif
