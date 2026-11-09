#ifndef XLIONRENDER_STAGE_GRAPH_TEXT_H
#define XLIONRENDER_STAGE_GRAPH_TEXT_H
#pragma once

// Frame stages' descriptions as text (render graph R3; xlionrender_stage_graph.h): a library of named stages in three xtextfile tables -
// the stages, their passes, every pass's bindings. The renderer never reads files - a view holds stage::desc overrides by name; this
// reader is the tests', the RenderGraph compiler's (the plugin's Stages/*.stage.txt) and its bundle's (one reader, one format).
//      [ Stages : 1 ]
//      { Name:s }                                (the name a view's override goes by: light_tiles, refl_ssr, refl_recon)
//        "light_tiles"
//      [ StagePasses : 1 ]
//      { Stage:d  Name:s  Kernel:s  Size:s  Push:s  RenderPass:s  Repeat:s  RepeatSlot:d }   (a draw pass: its render pass, and a buffer list it repeats over; Push: the push words in order, comma separated - a frame value's name, or a
//        0  "light tiles"  "light_tiles"  "view.size"  "frame.c2w,view.width,..."    constant: a number one uint word, f<number> one float word)
//      [ StageBindings : 3 ]
//      { Stage:d  Pass:d  Kind:s  Name:s  Life:s  Usage:s  Write:c  Policy:s  If:s }     (Pass: within its stage; Kind: buffer, texture,
//        0  0  "buffer"  "Lights: light ring"  "retained"  "storage"  0  "none"  ""      image - each in its kernel's order; If: a frame
//        ...                                                                              value whose first word 0 leaves it undeclared; "never": bound only)
#include "xlionrender_stage_graph.h"
#include "dependencies/xtextfile/source/xtextfile.h"
#include <format>
#include <bit>
#include <charconv>
#include <cctype>

namespace xlionrender::stage
{
    using library = std::vector<std::pair<std::string, desc>>;     // named stages

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

    // Reads or writes L. Problem: why a row was refused (L left empty); the file's own errors come back as xerr
    inline xerr Serialize( xtextfile::stream& F, library& L, std::string& Problem ) noexcept
    {
        std::vector<std::string> Stages;
        if( !F.isReading() ) for( const auto& [N, D] : L ) Stages.push_back(N);
        if( auto Err = F.Record("Stages"
            , [&](std::size_t& Count, xerr&) { if( F.isReading() ) Stages.resize(Count); else Count = Stages.size(); }
            , [&](std::size_t i, xerr& E) { E = F.Field("Name", Stages[i]); }); Err ) return Err;

        struct pass_row { int m_Stage = 0; std::string m_Name, m_Kernel, m_Size, m_Push, m_RenderPass, m_Repeat; int m_RepeatSlot = -1; std::string m_Items, m_UBOs, m_Recorder; };
        std::vector<pass_row> Passes;
        if( !F.isReading() ) for( std::size_t s = 0; s < L.size(); ++s ) for( const auto& P : L[s].second.m_Passes )
        {
            std::string S; for( const auto& W : P.m_Push ) { if( !S.empty() ) S += ","; S += text::Word(W); }
            std::string U; for( const auto& u : P.m_UBOs ) { if( !U.empty() ) U += ","; U += u; }
            Passes.push_back({ int(s), P.m_Name, P.m_Kernel, P.m_Size, S, P.m_RenderPass, P.m_Repeat, P.m_RepeatSlot, P.m_Items, U, P.m_Recorder });
        }
        if( auto Err = F.Record("StagePasses"
            , [&](std::size_t& Count, xerr&) { if( F.isReading() ) Passes.resize(Count); else Count = Passes.size(); }
            , [&](std::size_t i, xerr& E) { auto& R = Passes[i]; false || (E = F.Field("Stage", R.m_Stage)) || (E = F.Field("Name", R.m_Name)) || (E = F.Field("Kernel", R.m_Kernel)) || (E = F.Field("Size", R.m_Size)) || (E = F.Field("Push", R.m_Push)) || (E = F.Field("RenderPass", R.m_RenderPass)) || (E = F.Field("Repeat", R.m_Repeat)) || (E = F.Field("RepeatSlot", R.m_RepeatSlot)) || (E = F.Field("Items", R.m_Items)) || (E = F.Field("UBOs", R.m_UBOs)) || (E = F.Field("Recorder", R.m_Recorder)); }); Err ) return Err;

        struct binding_row { int m_Stage = 0, m_Pass = 0; std::string m_Kind, m_Name, m_Life, m_Usage, m_Policy, m_If; char m_Write = 0; };
        std::vector<binding_row> Rows;
        if( !F.isReading() ) for( std::size_t s = 0; s < L.size(); ++s ) for( std::size_t p = 0; p < L[s].second.m_Passes.size(); ++p )
        {
            const auto& P = L[s].second.m_Passes[p];
            int k = 0;
            for( const auto* pList : { &P.m_Buffers, &P.m_Textures, &P.m_Images } )
            {
                for( const auto& B : *pList ) Rows.push_back({ int(s), int(p), std::string(text::kind_v[k]), B.m_Name, std::string(text::life_v[int(B.m_Life)]), std::string(text::usage_v[int(B.m_Usage)]), std::string(text::policy_v[int(B.m_Policy)]), B.m_DeclareIf, char(B.m_bWrite ? 1 : 0) });
                ++k;
            }
        }
        if( auto Err = F.Record("StageBindings"
            , [&](std::size_t& Count, xerr&) { if( F.isReading() ) Rows.resize(Count); else Count = Rows.size(); }
            , [&](std::size_t i, xerr& E)
            {
                auto& R = Rows[i]; bool b = R.m_Write != 0;
                false || (E = F.Field("Stage", R.m_Stage)) || (E = F.Field("Pass", R.m_Pass)) || (E = F.Field("Kind", R.m_Kind)) || (E = F.Field("Name", R.m_Name)) || (E = F.Field("Life", R.m_Life))
                      || (E = F.Field("Usage", R.m_Usage)) || (E = F.Field("Write", b)) || (E = F.Field("Policy", R.m_Policy)) || (E = F.Field("If", R.m_If));
                R.m_Write = b ? 1 : 0;
            }); Err ) return Err;
        if( !F.isReading() ) return {};

        library Out;
        for( auto& N : Stages ) Out.emplace_back(std::move(N), desc{});
        const auto Fail = [&]( std::string Why ) { Problem = std::move(Why); L.clear(); return xerr{}; };
        for( std::size_t i = 0; i < Passes.size(); ++i )
        {
            const auto& R = Passes[i];
            if( R.m_Stage < 0 || R.m_Stage >= int(Out.size()) ) return Fail(std::format("pass row {} ({}): no stage {}", i, R.m_Name, R.m_Stage));
            pass P{ .m_Name = R.m_Name, .m_Kernel = R.m_Kernel, .m_Size = R.m_Size, .m_RenderPass = R.m_RenderPass, .m_Repeat = R.m_Repeat, .m_RepeatSlot = R.m_RepeatSlot, .m_Items = R.m_Items, .m_Recorder = R.m_Recorder };
            for( std::size_t b = 0; b < R.m_UBOs.size(); ) { std::size_t e = R.m_UBOs.find(',', b); if( e == std::string::npos ) e = R.m_UBOs.size(); if( e > b ) P.m_UBOs.push_back(R.m_UBOs.substr(b, e - b)); b = e + 1; }
            for( std::size_t b = 0; b <= R.m_Push.size() && !R.m_Push.empty(); )
            {
                std::size_t e = R.m_Push.find(',', b); if( e == std::string::npos ) e = R.m_Push.size();
                word W; if( !text::Word(std::string_view(R.m_Push).substr(b, e - b), W) ) return Fail(std::format("pass row {} ({}): push word {} '{}' is not a frame value, a number or f<number>", i, R.m_Name, P.m_Push.size(), R.m_Push.substr(b, e - b)));
                P.m_Push.push_back(std::move(W));
                b = e + 1;
            }
            Out[std::size_t(R.m_Stage)].second.m_Passes.push_back(std::move(P));
        }
        for( std::size_t i = 0; i < Rows.size(); ++i )
        {
            const auto& R = Rows[i];
            if( R.m_Stage < 0 || R.m_Stage >= int(Out.size()) ) return Fail(std::format("binding row {} ({}): no stage {}", i, R.m_Name, R.m_Stage));
            auto& D = Out[std::size_t(R.m_Stage)].second;
            if( R.m_Pass < 0 || R.m_Pass >= int(D.m_Passes.size()) ) return Fail(std::format("binding row {} ({}): stage {} has no pass {}", i, R.m_Name, Out[std::size_t(R.m_Stage)].first, R.m_Pass));
            binding B{ .m_Name = R.m_Name, .m_bWrite = R.m_Write != 0, .m_DeclareIf = R.m_If };
            int Kind = 0;
            if( !text::Find(text::kind_v, R.m_Kind, Kind) )           return Fail(std::format("binding row {} ({}): no kind '{}' (buffer, texture, image)", i, R.m_Name, R.m_Kind));
            if( !text::Find(text::life_v, R.m_Life, B.m_Life) )       return Fail(std::format("binding row {} ({}): no lifetime '{}' (frame, history, retained, static, external)", i, R.m_Name, R.m_Life));
            if( !text::Find(text::usage_v, R.m_Usage, B.m_Usage) )    return Fail(std::format("binding row {} ({}): no usage '{}' (sampled, storage, color, depth, transfer_src, transfer_dst)", i, R.m_Name, R.m_Usage));
            if( !text::Find(text::policy_v, R.m_Policy, B.m_Policy) ) return Fail(std::format("binding row {} ({}): no policy '{}' (none, clear, full_overwrite, preserve, partial, discard)", i, R.m_Name, R.m_Policy));
            auto& P = D.m_Passes[std::size_t(R.m_Pass)];
            (Kind == 0 ? P.m_Buffers : Kind == 1 ? P.m_Textures : P.m_Images).push_back(std::move(B));
        }
        L = std::move(Out);
        return {};
    }
}
#endif
