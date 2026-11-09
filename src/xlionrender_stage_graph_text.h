#ifndef XLIONRENDER_STAGE_GRAPH_TEXT_H
#define XLIONRENDER_STAGE_GRAPH_TEXT_H
#pragma once

// A frame stage's description as text (render graph R3; xlionrender_stage_graph.h): two xtextfile tables - its passes, then every pass's
// bindings. The renderer never reads files - a view holds stage::desc overrides; this reader is the tests' and the RenderGraph
// compiler's (one reader).
//      [ StagePasses : 1 ]
//      { Name:s  Kernel:s  Size:s  Push:s }      (Push: the push words in order, comma separated - a frame value's name, or a constant:
//        "light tiles"  "light_tiles"  "view.size"  "frame.c2w,view.width,view.height,lights.local_base,lights.local_count"
//                                                    a number is one uint word, f<number> one float word: "0", "f0.5")
//      [ StageBindings : 3 ]
//      { Pass:d  Kind:s  Name:s  Life:s  Usage:s  Write:c  Policy:s  If:s }     (Kind: buffer, texture, image - each in its kernel's order;
//        0  "buffer"   "Lights: light ring"  "retained"  "storage"  0  "none"  ""      If: a frame value whose first word 0 leaves the
//        ...                                                                            binding undeclared that frame - "" always)
#include "xlionrender_stage_graph.h"
#include "dependencies/xtextfile/source/xtextfile.h"
#include <format>
#include <bit>
#include <charconv>

namespace xlionrender::stage
{
    namespace text
    {
        inline constexpr std::string_view life_v[]   = { "frame", "history", "retained", "static", "external" };
        inline constexpr std::string_view usage_v[]  = { "sampled", "storage", "color", "depth", "transfer_src", "transfer_dst" };
        inline constexpr std::string_view policy_v[] = { "none", "clear", "full_overwrite", "preserve", "partial", "discard" };
        inline constexpr std::string_view kind_v[]   = { "buffer", "texture", "image" };

        template< class T, std::size_t N >
        bool Find( const std::string_view (&Names)[N], std::string_view S, T& Out ) noexcept
        {
            for( std::size_t i = 0; i < N; ++i ) if( Names[i] == S ) { Out = T(i); return true; }
            return false;
        }
        inline std::string Word( const word& W )
        {
            if( !W.m_Frame.empty() ) return W.m_Frame;
            return std::to_string(W.m_Constant);
        }
        inline bool Word( std::string_view S, word& W )
        {
            W = {};
            if( S.empty() ) return false;
            if( S[0] == 'f' && S.size() > 1 && (std::isdigit(static_cast<unsigned char>(S[1])) || S[1] == '-' || S[1] == '.') )
            {
                float f = 0; const auto R = std::from_chars(S.data() + 1, S.data() + S.size(), f);
                if( R.ec != std::errc{} || R.ptr != S.data() + S.size() ) return false;
                W.m_Constant = std::bit_cast<std::uint32_t>(f); return true;
            }
            if( std::isdigit(static_cast<unsigned char>(S[0])) )
            {
                const auto R = std::from_chars(S.data(), S.data() + S.size(), W.m_Constant);
                return R.ec == std::errc{} && R.ptr == S.data() + S.size();
            }
            W.m_Frame = std::string(S); return true;
        }
    }

    // Reads or writes D. Problem: why a row was refused; the file's own errors come back as xerr
    inline xerr Serialize( xtextfile::stream& F, desc& D, std::string& Problem ) noexcept
    {
        std::vector<std::string> Name, Kernel, Size, Push;
        if( !F.isReading() ) for( const auto& P : D.m_Passes )
        {
            Name.push_back(P.m_Name); Kernel.push_back(P.m_Kernel); Size.push_back(P.m_Size);
            std::string S; for( const auto& W : P.m_Push ) { if( !S.empty() ) S += ","; S += text::Word(W); } Push.push_back(S);
        }
        if( auto Err = F.Record("StagePasses"
            , [&](std::size_t& Count, xerr&) { if( F.isReading() ) { Name.resize(Count); Kernel.resize(Count); Size.resize(Count); Push.resize(Count); } else Count = Name.size(); }
            , [&](std::size_t i, xerr& E) { false || (E = F.Field("Name", Name[i])) || (E = F.Field("Kernel", Kernel[i])) || (E = F.Field("Size", Size[i])) || (E = F.Field("Push", Push[i])); }); Err ) return Err;

        struct row { int m_Pass = 0; std::string m_Kind, m_Name, m_Life, m_Usage, m_Policy, m_If; char m_Write = 0; };
        std::vector<row> Rows;
        if( !F.isReading() ) for( std::size_t p = 0; p < D.m_Passes.size(); ++p )
        {
            const auto& P = D.m_Passes[p];
            int k = 0;
            for( const auto* pList : { &P.m_Buffers, &P.m_Textures, &P.m_Images } )
            {
                for( const auto& B : *pList ) Rows.push_back({ int(p), std::string(text::kind_v[k]), B.m_Name, std::string(text::life_v[int(B.m_Life)]), std::string(text::usage_v[int(B.m_Usage)]), std::string(text::policy_v[int(B.m_Policy)]), B.m_DeclareIf, char(B.m_bWrite ? 1 : 0) });
                ++k;
            }
        }
        if( auto Err = F.Record("StageBindings"
            , [&](std::size_t& Count, xerr&) { if( F.isReading() ) Rows.resize(Count); else Count = Rows.size(); }
            , [&](std::size_t i, xerr& E)
            {
                auto& R = Rows[i]; bool b = R.m_Write != 0;
                false || (E = F.Field("Pass", R.m_Pass)) || (E = F.Field("Kind", R.m_Kind)) || (E = F.Field("Name", R.m_Name)) || (E = F.Field("Life", R.m_Life))
                      || (E = F.Field("Usage", R.m_Usage)) || (E = F.Field("Write", b)) || (E = F.Field("Policy", R.m_Policy)) || (E = F.Field("If", R.m_If));
                R.m_Write = b ? 1 : 0;
            }); Err ) return Err;
        if( !F.isReading() ) return {};

        desc Out;
        for( std::size_t i = 0; i < Name.size(); ++i )
        {
            pass P{ .m_Name = Name[i], .m_Kernel = Kernel[i], .m_Size = Size[i] };
            for( std::size_t b = 0; b <= Push[i].size() && !Push[i].empty(); )
            {
                std::size_t e = Push[i].find(',', b); if( e == std::string::npos ) e = Push[i].size();
                word W; if( !text::Word(std::string_view(Push[i]).substr(b, e - b), W) ) { Problem = std::format("pass {} ({}): push word {} '{}' is not a frame value, a number or f<number>", i, Name[i], P.m_Push.size(), Push[i].substr(b, e - b)); return {}; }
                P.m_Push.push_back(std::move(W));
                b = e + 1;
            }
            Out.m_Passes.push_back(std::move(P));
        }
        for( std::size_t i = 0; i < Rows.size(); ++i )
        {
            const auto& R = Rows[i];
            if( R.m_Pass < 0 || R.m_Pass >= int(Out.m_Passes.size()) ) { Problem = std::format("binding {}: no pass {}", i, R.m_Pass); return {}; }
            binding B{ .m_Name = R.m_Name, .m_bWrite = R.m_Write != 0, .m_DeclareIf = R.m_If };
            int Kind = 0;
            if( !text::Find(text::kind_v, R.m_Kind, Kind) )         { Problem = std::format("binding {} ({}): no kind '{}' (buffer, texture, image)", i, R.m_Name, R.m_Kind); return {}; }
            if( !text::Find(text::life_v, R.m_Life, B.m_Life) )     { Problem = std::format("binding {} ({}): no lifetime '{}'", i, R.m_Name, R.m_Life); return {}; }
            if( !text::Find(text::usage_v, R.m_Usage, B.m_Usage) )  { Problem = std::format("binding {} ({}): no usage '{}'", i, R.m_Name, R.m_Usage); return {}; }
            if( !text::Find(text::policy_v, R.m_Policy, B.m_Policy) ) { Problem = std::format("binding {} ({}): no policy '{}'", i, R.m_Name, R.m_Policy); return {}; }
            auto& P = Out.m_Passes[std::size_t(R.m_Pass)];
            (Kind == 0 ? P.m_Buffers : Kind == 1 ? P.m_Textures : P.m_Images).push_back(std::move(B));
        }
        D = std::move(Out);
        return {};
    }
}
#endif
