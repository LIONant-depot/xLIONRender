#ifndef XLIONRENDER_POST_GRAPH_TEXT_H
#define XLIONRENDER_POST_GRAPH_TEXT_H
#pragma once

// The post chain's description as text (render graph P2; xLION_Docs/render_graph_p2_note_2026-11-09.md): one xtextfile table, the
// authored nodes in order by their semantic names. The renderer never reads files - a view holds a post::desc; this reader is for
// the tests now and the render graph resource's compiler later (one reader).
//      [ PostGraph : 4 ]
//      { Type:s  On:c  Custom:d  Params:s }      (Custom: a post.custom node's shader index, -1 for the others; Params: the parameters it pushes, comma separated)
//        "post.taa"       1
//        "post.bloom"     1          (0: bypassed - kept in the recipe, left out of the plan)
//        "post.exposure"  1
//        "post.tonemap"   1
//      [ Params : 10 ]
//      { Name:s  Source:s  Type:s  Default:f  Min:f  Max:f }
//        "BloomThreshold"  "scene.post.bloom_threshold"  "float"  0.99  0  16
//        ...
#include "xlionrender_post_graph.h"
#include "dependencies/xtextfile/source/xtextfile.h"

namespace xlionrender::post
{
    inline constexpr std::pair<std::string_view, node_type> type_names_v[] = { { "post.taa", node_type::TAA }, { "post.bloom", node_type::BLOOM }, { "post.custom", node_type::CUSTOM }
                                                                             , { "post.exposure", node_type::EXPOSURE }, { "post.tonemap", node_type::TONE_MAP } };

    // Reads or writes D. Problem: why a row was refused (an unknown type names its row); the file's own errors come back as xerr
    inline xerr Serialize( xtextfile::stream& F, desc& D, std::string& Problem ) noexcept
    {
        std::vector<std::string> Names;
        std::vector<char>        On;        // (bool as the table's c column)
        std::vector<int>         Custom;    // (a custom node's shader index; -1 for the others)
        std::vector<std::string> Params;    // (a custom node's parameters, comma separated)
        if( !F.isReading() ) for( const auto& N : D.m_Nodes )
        {
            for( const auto& [Name, T] : type_names_v ) if( T == N.m_Type ) Names.emplace_back(Name);
            On.push_back(N.m_bOn ? 1 : 0); Custom.push_back(N.m_Custom);
            std::string P; for( const auto& s : N.m_Params ) { if( !P.empty() ) P += ","; P += s; } Params.push_back(P);
        }
        if( auto Err = F.Record("PostGraph"
            , [&](std::size_t& Count, xerr&) { if( F.isReading() ) { Names.resize(Count); On.resize(Count, 1); Custom.resize(Count, -1); Params.resize(Count); } else Count = Names.size(); }
            , [&](std::size_t i, xerr& E) { bool b = On[i] != 0; false || (E = F.Field("Type", Names[i])) || (E = F.Field("On", b)) || (E = F.Field("Custom", Custom[i])) || (E = F.Field("Params", Params[i])); On[i] = b ? 1 : 0; }); Err ) return Err;
        if( F.isReading() )
        {
            D.m_Nodes.clear();
            for( std::size_t i = 0; i < Names.size(); ++i )
            {
                const auto It = std::find_if(std::begin(type_names_v), std::end(type_names_v), [&](const auto& P) { return P.first == Names[i]; });
                if( It == std::end(type_names_v) ) { Problem = std::format("row {}: no node type '{}' (the types: post.taa, post.bloom, post.exposure, post.tonemap, post.custom)", i, Names[i]); D.m_Nodes.clear(); return {}; }
                node N{ It->second, On[i] != 0, Custom[i] };
                for( std::size_t b = 0; b < Params[i].size(); )
                {
                    std::size_t e = Params[i].find(',', b); if( e == std::string::npos ) e = Params[i].size();
                    if( e > b ) N.m_Params.push_back(Params[i].substr(b, e - b));
                    b = e + 1;
                }
                D.m_Nodes.push_back(std::move(N));
            }
        }
        // The parameters (P3): what the nodes push, where a frame reads each, its range
        if( auto Err = F.Record("Params"
            , [&](std::size_t& Count, xerr&) { if( F.isReading() ) D.m_Params.resize(Count); else Count = D.m_Params.size(); }
            , [&](std::size_t i, xerr& E)
            {
                auto& Pa = D.m_Params[i];
                std::string Type = Pa.m_Type == param_type::BOOL ? "bool" : "float";
                false || (E = F.Field("Name", Pa.m_Name)) || (E = F.Field("Source", Pa.m_Source)) || (E = F.Field("Type", Type))
                      || (E = F.Field("Default", Pa.m_Default)) || (E = F.Field("Min", Pa.m_Min)) || (E = F.Field("Max", Pa.m_Max));
                if( F.isReading() ) Pa.m_Type = Type == "bool" ? param_type::BOOL : param_type::FLOAT;
            }); Err ) return Err;
        if( F.isReading() ) for( std::size_t i = 0; i < D.m_Params.size(); ++i )
            if( D.m_Params[i].m_Type == param_type::FLOAT && std::isnan(D.m_Params[i].m_Default) ) { Problem = std::format("parameter row {}: no default", i); D = {}; return {}; }
        return {};
    }
}
#endif
