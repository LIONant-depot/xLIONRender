#ifndef XLIONRENDER_POST_GRAPH_H
#define XLIONRENDER_POST_GRAPH_H
#pragma once

// The post chain as data (render graph P1a; xLION_Docs/render_graph_design_v3_2026-11-09.md 11.2-11.3, render_graph_p1_plan_2026-11-09.md):
// an authored description (the effects, in order) is validated and lowered, for a view's size, into an immutable plan - a flat list of
// passes with their kernel, images, size, dispatch and the render graph's reads / writes - that a frame walks with no name lookups.
// The plan matches what the hand-written chain records (xLION_Docs/render_graph_post_chain_inventory_2026-11-09.md): same kernels,
// sizes, bindings, order. P3: the values the passes push are data too - the description declares its parameters (typed, ranged, with
// a source such as "scene.post.bloom_threshold"), each pass says which word of its push takes which; lowering resolves the names, a
// frame takes one snapshot by index (Snapshot) - no string lookups per frame.
#include "xlionrender_render_graph.h"
#include "scene/xlionrender_scene.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <string>
#include <vector>

namespace xlionrender::post
{
    // The authored effects. Bloom is one node: lowering expands it into its levels. The initialisations (TAA history, bloom levels and
    // exposure zeroed) are the nodes' lifecycle, never wired by a user
    // CUSTOM (render graph stage I): a node with its own compute shader - the chain image in, a new chain image out (an image effect); its
    // SPIR-V the view's m_PostCustomSpirv[m_Custom] (the RenderGraph bundle's shader link). Any number, after TAA, before the tone map
    enum class node_type : std::uint8_t { TAA, BLOOM, EXPOSURE, TONE_MAP, CUSTOM };
    struct node { node_type m_Type; bool m_bOn = true; int m_Custom = -1; bool operator==( const node& ) const = default; };     // (off: bypassed - kept in the recipe, left out of the plan; Custom: the shader's index)
    inline constexpr std::string_view node_names_v[] = { "taa", "bloom", "exposure", "tonemap" };     // (the edit commands' names; the text's are post.<name>)
    inline constexpr int max_custom_nodes_v = 4;

    // A parameter of the chain: what the nodes push. Source: where a frame reads it ("scene.post.bloom_threshold"; empty: the default
    // only); a value outside Min..Max is clamped (and reported once a change: SnapshotParams). Bools are 0 / 1
    enum class param_type : std::uint8_t { FLOAT, BOOL };
    struct param
    {
        std::string     m_Name, m_Source;
        param_type      m_Type = param_type::FLOAT;
        float           m_Default = 0, m_Min = 0, m_Max = 0;
        bool operator==( const param& ) const = default;
    };

    // The scene's settings a parameter can come from (stable names: the description refers to them; the Scene is read each frame,
    // never kept - Play / Stop replace it)
    struct source { std::string_view m_Name; float post_settings::* m_pFloat; bool post_settings::* m_pBool; };
    inline constexpr source sources_v[] = {
        { "scene.post.bloom",           &post_settings::m_Bloom,          nullptr },
        { "scene.post.bloom_threshold", &post_settings::m_BloomThreshold, nullptr },
        { "scene.post.bloom_knee",      &post_settings::m_BloomKnee,      nullptr },
        { "scene.post.bloom_spread",    &post_settings::m_BloomSpread,    nullptr },
        { "scene.post.auto_exposure",   nullptr,                          &post_settings::m_bAutoExposure },
        { "scene.post.taa_blend",       &post_settings::m_TAABlend,       nullptr },
        { "scene.post.exposure_key",    &post_settings::m_ExposureKey,    nullptr },
        { "scene.post.exposure_rate",   &post_settings::m_ExposureRate,   nullptr },
        { "scene.post.exposure_min",    &post_settings::m_ExposureMin,    nullptr },
        { "scene.post.exposure_max",    &post_settings::m_ExposureMax,    nullptr },
    };

    struct desc
    {
        std::vector<node>  m_Nodes;
        std::vector<param> m_Params;
        static desc Default( void )
        {
            return { { { node_type::TAA }, { node_type::BLOOM }, { node_type::EXPOSURE }, { node_type::TONE_MAP } }
                   , { { "Bloom",          "scene.post.bloom",           param_type::FLOAT, 0.0f,   0.0f,    1.0f }
                     , { "BloomThreshold", "scene.post.bloom_threshold", param_type::FLOAT, 0.99f,  0.0f,    16.0f }
                     , { "BloomKnee",      "scene.post.bloom_knee",      param_type::FLOAT, 0.5f,   0.0f,    4.0f }
                     , { "BloomSpread",    "scene.post.bloom_spread",    param_type::FLOAT, 0.7f,   0.0f,    1.0f }
                     , { "AutoExposure",   "scene.post.auto_exposure",   param_type::BOOL,  0.0f,   0.0f,    1.0f }
                     , { "TAABlend",       "scene.post.taa_blend",       param_type::FLOAT, 0.1f,   0.01f,   1.0f }
                     , { "ExposureKey",    "scene.post.exposure_key",    param_type::FLOAT, 0.115f, 0.001f,  10.0f }
                     , { "ExposureRate",   "scene.post.exposure_rate",   param_type::FLOAT, 1.5f,   0.0f,    20.0f }
                     , { "ExposureMin",    "scene.post.exposure_min",    param_type::FLOAT, 0.03f,  0.0001f, 1000.0f }
                     , { "ExposureMax",    "scene.post.exposure_max",    param_type::FLOAT, 30.0f,  0.0001f, 1000.0f } } };
        }
        bool operator==( const desc& ) const = default;
    };

    // The images a pass touches, resolved by the view each frame. CHAIN: the TAA's output when it ran this frame, else the lit image
    enum class image : std::uint8_t { NONE, HDR, DEPTH, CHAIN, TAA_PREV, TAA_CUR, TAA0, TAA1, EXPOSURE, BLOOM, OUTPUT, CUSTOM };     // (CUSTOM: m_Level is the custom node's index)
    struct image_ref { image m_Image = image::NONE; int m_Level = 0; bool operator==( const image_ref& ) const = default; };     // (Level: the bloom's)

    enum class kernel : std::uint8_t { TAA, POST, TONE_MAP, CUSTOM };     // (CUSTOM: the node's own shader, post_comp's bindings and push - xlionrender_post_node.glsl)       // xlionrender_taa_comp, xlionrender_post_comp, xlionrender_tonemap_frag
    enum class when   : std::uint8_t { ALWAYS, TAA_ON, BLOOM_ON, AUTO_EXPOSURE };      // (settings, not topology: a slider at 0 skips, it rebuilds nothing)
    enum class push   : std::uint8_t { FILL, TAA, BLOOM_DOWN, EXPOSURE, BLOOM_UP, TONE_MAP };

    // A word of a pass's push constants (4 bytes, counted from the push's start) and what fills it: a constant, a parameter of the
    // frame's snapshot, or a value only the frame knows (the exposure step's seconds, the view's manual exposure)
    enum class slot_from : std::uint8_t { CONSTANT, PARAM, FRAME_DT, VIEW_EXPOSURE };
    struct push_slot { int m_Word; slot_from m_From; int m_Param = -1; float m_Constant = 0; };

    struct pass
    {
        std::string             m_Scope;        // the render graph's scope name ("bloom down": the audit and the frame map match it)
        std::string             m_Source;       // where it came from, for people: "Bloom, down 3, 201 x 94"
        int                     m_Node = -1;    // the authored node (-1: a lifecycle initialisation)
        kernel                  m_Kernel = kernel::POST;
        int                     m_Mode = 0;     // post_comp's mode: 0 down, 1 up, 2 exposure, 3 fill
        push                    m_Push = push::FILL;
        when                    m_When = when::ALWAYS;
        int                     m_Level = 0;    // the bloom level it works on (push: the first down step is the thresholded one)
        image_ref               m_Bind[3];      // the sampled bindings, in the kernel's order
        image_ref               m_Dst;          // the storage / colour target
        int                     m_W = 0, m_H = 0, m_GroupsX = 0, m_GroupsY = 0;
        struct access { image_ref m_Ref; rg::usage m_Usage; rg::policy m_Policy; bool m_bWrite; };
        std::vector<access>     m_Accesses;     // what it declares to the render graph (R0b), in the legacy order
        std::vector<push_slot>  m_Slots;        // the push words the plan fills (the rest - sizes, mode, TAA's matrix and history flag - the frame's)
    };

    // A parameter resolved: its source by index (-1: the default only), its range
    struct resolved_param { int m_Source = -1; param_type m_Type = param_type::FLOAT; float m_Default = 0, m_Min = 0, m_Max = 0; std::string m_Name; };

    struct plan
    {
        std::vector<pass>       m_Inits;        // once: TAA history (both images; when the TAA's images are first made), bloom levels + exposure (once a view)
        std::vector<pass>       m_Passes;       // a frame, in order
        int                     m_nBloom = 0;   // the levels the bloom uses
        std::vector<resolved_param> m_Params;   // the snapshot's layout (SnapshotParams)
        int                     m_iBloom = -1, m_iAutoExposure = -1, m_iTAABlend = -1;     // (what decides whether passes run, and the measurement overrides)
        std::string             m_Error;        // not empty: the description is not valid (and the plan is empty)
    };

    inline constexpr int max_bloom_levels_v = 6;        // (render_view::m_Bloom)

    // The bloom's level i: half the view, and half again, at least 1 x 1 (floor)
    inline std::array<int, 2> BloomSize( int W, int H, int i ) noexcept { return { std::max(W >> (i + 1), 1), std::max(H >> (i + 1), 1) }; }
    // The levels by the view's height: the smallest about 48 pixels high at any resolution
    inline int BloomLevels( int H ) noexcept { return std::clamp(int(std::round(std::log2(float(H) / 48.0f))), 3, max_bloom_levels_v); }

    // The description, checked and expanded for a W x H view. The bloom's down steps run where the node is, its up steps right before
    // the tone map (they only feed it; the exposure and the up steps touch different images, so this is the hand-written order).
    // (LowerOn: the nodes that are on; Lower below leaves the bypassed ones out and maps the indices back)
    inline plan LowerOn( const desc& D, int W, int H ) noexcept
    {
        plan P;
        const auto Fail = [&](int i, std::string Why) { P = {}; P.m_Error = i >= 0 ? std::format("node {}: {}", i, Why) : Why; return P; };
        if( W <= 0 || H <= 0 ) return Fail(-1, std::format("the view is {} x {}: nothing to lower", W, H));
        int nType[5] = {};
        for( int i = 0; i < int(D.m_Nodes.size()); ++i )
        {
            const auto T = D.m_Nodes[std::size_t(i)].m_Type;
            if( std::size_t(T) >= 5 ) return Fail(i, "unknown node type");
            if( T == node_type::CUSTOM )
            {
                const int c = D.m_Nodes[std::size_t(i)].m_Custom;
                if( c < 0 || c >= max_custom_nodes_v ) return Fail(i, std::format("a custom node's shader index {} is not 0 .. {}", c, max_custom_nodes_v - 1));
                for( int j = 0; j < i; ++j ) if( D.m_Nodes[std::size_t(j)].m_Type == node_type::CUSTOM && D.m_Nodes[std::size_t(j)].m_Custom == c ) return Fail(i, std::format("shader {} is used by node {} already (each custom node its own image)", c, j));
                ++nType[4]; continue;
            }
            if( ++nType[std::size_t(T)] > 1 ) return Fail(i, "a second one of a node that keeps state (one each)");
            if( T == node_type::TAA && i != 0 ) return Fail(i, "TAA must come first: everything after it reads its output");
            if( T == node_type::TONE_MAP && i != int(D.m_Nodes.size()) - 1 ) return Fail(i, "the tone map must be last: it makes the output");
        }
        if( !nType[std::size_t(node_type::TONE_MAP)] ) return Fail(-1, "no tone map: the chain makes no output");

        // The parameters: names unique, sources known, ranges sane - resolved to indices once
        for( const auto& Pa : D.m_Params )
        {
            resolved_param R{ -1, Pa.m_Type, Pa.m_Default, Pa.m_Min, Pa.m_Max, Pa.m_Name };
            if( std::count_if(D.m_Params.begin(), D.m_Params.end(), [&](const param& Q) { return Q.m_Name == Pa.m_Name; }) > 1 ) return Fail(-1, std::format("parameter '{}' is declared twice", Pa.m_Name));
            if( !(Pa.m_Min <= Pa.m_Max) ) return Fail(-1, std::format("parameter '{}': its range {} .. {} is empty", Pa.m_Name, Pa.m_Min, Pa.m_Max));
            if( !Pa.m_Source.empty() )
            {
                const auto It = std::find_if(std::begin(sources_v), std::end(sources_v), [&](const source& S) { return S.m_Name == Pa.m_Source; });
                if( It == std::end(sources_v) ) return Fail(-1, std::format("parameter '{}': no source '{}'", Pa.m_Name, Pa.m_Source));
                if( (It->m_pBool != nullptr) != (Pa.m_Type == param_type::BOOL) ) return Fail(-1, std::format("parameter '{}': its source '{}' is a {}", Pa.m_Name, Pa.m_Source, It->m_pBool ? "bool" : "float"));
                R.m_Source = int(It - std::begin(sources_v));
            }
            P.m_Params.push_back(std::move(R));
        }
        const auto Param = [&](std::string_view Name) { for( int i = 0; i < int(P.m_Params.size()); ++i ) if( P.m_Params[std::size_t(i)].m_Name == Name ) return i; return -1; };
        // Each node's parameters (a node that needs one the description lacks is refused, naming both)
        static constexpr std::string_view s_Needs[5][5] = { { "TAABlend" }, { "Bloom", "BloomThreshold", "BloomKnee", "BloomSpread", "AutoExposure" }
                                                          , { "AutoExposure", "ExposureRate", "ExposureKey", "ExposureMin", "ExposureMax" }, { "Bloom", "AutoExposure" }, {} };
        static constexpr std::string_view s_TypeNames[5] = { "taa", "bloom", "exposure", "tone map", "custom" };
        for( int i = 0; i < int(D.m_Nodes.size()); ++i )
            for( const auto Need : s_Needs[std::size_t(D.m_Nodes[std::size_t(i)].m_Type)] )
                if( !Need.empty() && Param(Need) < 0 ) return Fail(i, std::format("the {} needs the parameter '{}'", s_TypeNames[std::size_t(D.m_Nodes[std::size_t(i)].m_Type)], Need));
        P.m_iBloom = Param("Bloom"); P.m_iAutoExposure = Param("AutoExposure"); P.m_iTAABlend = Param("TAABlend");
        const auto Pm = [&](int Word, std::string_view Name) { return push_slot{ Word, slot_from::PARAM, Param(Name) }; };
        const auto Ct = [](int Word, float V) { return push_slot{ Word, slot_from::CONSTANT, -1, V }; };

        using A = pass::access;
        const auto Groups = [](pass& p) { p.m_GroupsX = (p.m_W + 7) / 8; p.m_GroupsY = (p.m_H + 7) / 8; };
        const image_ref HDR{ image::HDR }, DEPTH{ image::DEPTH }, CHAIN{ image::CHAIN }, EXPO{ image::EXPOSURE };
        const auto Bloom = [](int i) { return image_ref{ image::BLOOM, i }; };
        const auto R = [](image_ref r, rg::usage U = rg::usage::SAMPLED) { return A{ r, U, rg::policy::NONE, false }; };
        const auto Wr = [](image_ref r, rg::policy Pol, rg::usage U = rg::usage::STORAGE) { return A{ r, U, Pol, true }; };
        const bool bBloom = nType[std::size_t(node_type::BLOOM)] > 0;
        P.m_nBloom = bBloom ? BloomLevels(H) : 0;

        // Lifecycle: the TAA's two history images zeroed when made (the history is not read until a frame wrote it)
        if( nType[std::size_t(node_type::TAA)] ) for( int i = 0; i < 2; ++i )
        {
            pass p{ .m_Scope = "TAA history init", .m_Source = std::format("TAA, history {} zeroed", i), .m_Mode = 3, .m_When = when::TAA_ON,
                    .m_Bind = { HDR, EXPO }, .m_Dst = { i ? image::TAA1 : image::TAA0 }, .m_W = W, .m_H = H };
            p.m_Accesses = { R(HDR), Wr(p.m_Dst, rg::policy::FULL_OVERWRITE) };
            Groups(p); P.m_Inits.push_back(std::move(p));
        }
        // Lifecycle: every bloom level and the exposure zeroed once (sampled before any frame wrote them: the tone map binds level 0 always)
        for( int i = 0; i <= max_bloom_levels_v; ++i )
        {
            const bool bExpo = i == max_bloom_levels_v;
            const auto S = bExpo ? std::array<int, 2>{ 1, 1 } : BloomSize(W, H, i);
            pass p{ .m_Scope = bExpo ? "exposure" : "post init", .m_Source = bExpo ? std::string("Exposure, zeroed") : std::format("Bloom, level {} zeroed, {} x {}", i, S[0], S[1]),
                    .m_Mode = 3, .m_Bind = { HDR, bExpo ? HDR : EXPO }, .m_Dst = bExpo ? EXPO : Bloom(i), .m_W = S[0], .m_H = S[1] };
            p.m_Accesses = { R(HDR), R(bExpo ? HDR : EXPO), Wr(p.m_Dst, rg::policy::FULL_OVERWRITE) };
            Groups(p); P.m_Inits.push_back(std::move(p));
        }

        std::vector<pass> Ups;
        for( int n = 0; n < int(D.m_Nodes.size()); ++n ) switch( D.m_Nodes[std::size_t(n)].m_Type )
        {
            case node_type::TAA:
            {
                pass p{ .m_Scope = "TAA", .m_Source = std::format("TAA, {} x {}", W, H), .m_Node = n, .m_Kernel = kernel::TAA, .m_Push = push::TAA, .m_When = when::TAA_ON,
                        .m_Bind = { HDR, DEPTH, { image::TAA_PREV } }, .m_Dst = { image::TAA_CUR }, .m_W = W, .m_H = H };
                p.m_Accesses = { R(HDR), R(DEPTH), R({ image::TAA_PREV }), Wr(p.m_Dst, rg::policy::FULL_OVERWRITE) };
                p.m_Slots = { Pm(18, "TAABlend") };
                Groups(p); P.m_Passes.push_back(std::move(p));
                break;
            }
            case node_type::BLOOM:
                for( int i = 0; i < P.m_nBloom; ++i )
                {
                    const auto S = BloomSize(W, H, i);
                    const image_ref Src = i ? Bloom(i - 1) : CHAIN;
                    pass p{ .m_Scope = "bloom down", .m_Source = std::format("Bloom, down {}, {} x {}", i, S[0], S[1]), .m_Node = n, .m_Mode = 0, .m_Push = push::BLOOM_DOWN,
                            .m_When = when::BLOOM_ON, .m_Level = i, .m_Bind = { Src, EXPO }, .m_Dst = Bloom(i), .m_W = S[0], .m_H = S[1] };
                    p.m_Accesses = { R(Src), R(EXPO), Wr(p.m_Dst, rg::policy::FULL_OVERWRITE) };
                    p.m_Slots = { Ct(3, i == 0 ? 1.0f : 0.0f), Pm(4, "BloomThreshold"), Pm(5, "BloomKnee"), { 6, slot_from::VIEW_EXPOSURE }, Pm(7, "AutoExposure") };
                    Groups(p); P.m_Passes.push_back(std::move(p));
                }
                for( int i = P.m_nBloom - 2; i >= 0; --i )
                {
                    const auto S = BloomSize(W, H, i);
                    pass p{ .m_Scope = "bloom up", .m_Source = std::format("Bloom, up {}, {} x {}", i, S[0], S[1]), .m_Node = n, .m_Mode = 1, .m_Push = push::BLOOM_UP,
                            .m_When = when::BLOOM_ON, .m_Level = i, .m_Bind = { Bloom(i + 1), EXPO }, .m_Dst = Bloom(i), .m_W = S[0], .m_H = S[1] };
                    p.m_Accesses = { R(Bloom(i + 1)), R(EXPO), R(Bloom(i)), Wr(p.m_Dst, rg::policy::PRESERVE) };
                    p.m_Slots = { Pm(3, "BloomSpread") };
                    Groups(p); Ups.push_back(std::move(p));
                }
                break;
            case node_type::EXPOSURE:
            {
                // (the kernel's second slot holds the lit image here: bound, unused - the hand-written chain declares it, so this does)
                pass p{ .m_Scope = "exposure", .m_Source = "Exposure, adapt", .m_Node = n, .m_Mode = 2, .m_Push = push::EXPOSURE, .m_When = when::AUTO_EXPOSURE,
                        .m_Bind = { CHAIN, HDR }, .m_Dst = EXPO, .m_W = 1, .m_H = 1 };
                p.m_Accesses = { R(CHAIN), R(HDR), Wr(EXPO, rg::policy::PRESERVE) };
                p.m_Slots = { { 3, slot_from::FRAME_DT }, Pm(4, "ExposureRate"), Pm(5, "ExposureKey"), Pm(6, "ExposureMin"), Pm(7, "ExposureMax") };
                Groups(p); P.m_Passes.push_back(std::move(p));
                break;
            }
            case node_type::CUSTOM:
            {
                const int c = D.m_Nodes[std::size_t(n)].m_Custom;
                pass p{ .m_Scope = "custom", .m_Source = std::format("Custom {}, {} x {}", c, W, H), .m_Node = n, .m_Kernel = kernel::CUSTOM, .m_Push = push::FILL,
                        .m_Level = c, .m_Bind = { CHAIN, EXPO }, .m_Dst = { image::CUSTOM, c }, .m_W = W, .m_H = H };
                p.m_Accesses = { R(CHAIN), R(EXPO), Wr(p.m_Dst, rg::policy::FULL_OVERWRITE) };
                Groups(p); P.m_Passes.push_back(std::move(p));
                break;
            }
            case node_type::TONE_MAP:
            {
                for( auto& u : Ups ) P.m_Passes.push_back(std::move(u));
                pass p{ .m_Scope = "tone map", .m_Source = std::format("Tone map, {} x {}", W, H), .m_Node = n, .m_Kernel = kernel::TONE_MAP, .m_Push = push::TONE_MAP,
                        .m_Bind = { CHAIN, Bloom(0), EXPO }, .m_Dst = { image::OUTPUT }, .m_W = W, .m_H = H };
                p.m_Accesses = { R(CHAIN), R(Bloom(0)), R(EXPO), Wr(p.m_Dst, rg::policy::FULL_OVERWRITE, rg::usage::COLOR_ATTACHMENT) };
                p.m_Slots = { { 0, slot_from::VIEW_EXPOSURE }, Pm(1, "Bloom"), Pm(2, "AutoExposure") };
                P.m_Passes.push_back(std::move(p));
                break;
            }
        }
        return P;
    }

    inline plan Lower( const desc& D, int W, int H ) noexcept
    {
        desc On{ {}, D.m_Params };
        std::vector<int> Map;
        for( int i = 0; i < int(D.m_Nodes.size()); ++i ) if( D.m_Nodes[std::size_t(i)].m_bOn ) { On.m_Nodes.push_back(D.m_Nodes[std::size_t(i)]); Map.push_back(i); }
        if( Map.size() == D.m_Nodes.size() ) return LowerOn(D, W, H);
        plan P = LowerOn(On, W, H);
        for( auto* pV : { &P.m_Inits, &P.m_Passes } ) for( auto& Pa : *pV ) if( Pa.m_Node >= 0 ) Pa.m_Node = Map[std::size_t(Pa.m_Node)];
        if( P.m_Error.starts_with("node ") )        // ("node <i>: ..." in the on nodes' numbering: the description's)
        {
            const std::size_t e = P.m_Error.find(':');
            const int i = std::atoi(P.m_Error.c_str() + 5);
            if( e != std::string::npos && i >= 0 && i < int(Map.size()) ) P.m_Error = std::format("node {}{}", Map[std::size_t(i)], P.m_Error.substr(e));
        }
        return P;
    }

    // An edit of the description, all or nothing (render graph P6): the candidate must lower (at any size: 64 x 64 checks the structure),
    // else D is untouched and Message says why. Commands (the console's and the editor's, the same):
    //      bypass <node> <0|1>         a node off (kept in the recipe, left out of the plan) or on
    //      insert <gap> <type>         a node before the gap-th (gap = the node count: at the end); type: taa, bloom, exposure, tonemap
    //      remove <node>
    //      move <node> <to>            to the gap-th position (before the node there)
    inline bool Edit( desc& D, std::string_view Command, std::string& Message ) noexcept
    {
        std::vector<std::string> W;
        for( std::size_t b = 0; b < Command.size(); )
        {
            while( b < Command.size() && Command[b] == ' ' ) ++b;
            std::size_t e = Command.find(' ', b); if( e == std::string_view::npos ) e = Command.size();
            if( e > b ) W.emplace_back(Command.substr(b, e - b));
            b = e;
        }
        const auto Int = [&](std::size_t i, int& Out) { if( i >= W.size() ) return false; char* pEnd = nullptr; const long v = std::strtol(W[i].c_str(), &pEnd, 10); Out = int(v); return pEnd && *pEnd == 0; };
        desc C = D;
        const int n = int(C.m_Nodes.size());
        int a = 0, b = 0;
        if( W.size() == 3 && W[0] == "bypass" && Int(1, a) && Int(2, b) && a >= 0 && a < n ) C.m_Nodes[std::size_t(a)].m_bOn = b != 0;
        else if( W.size() == 3 && W[0] == "insert" && Int(1, a) && a >= 0 && a <= n )
        {
            const auto It = std::find(std::begin(node_names_v), std::end(node_names_v), W[2]);
            if( It == std::end(node_names_v) ) { Message = std::format("no node type '{}' (taa, bloom, exposure, tonemap)", W[2]); return false; }
            C.m_Nodes.insert(C.m_Nodes.begin() + a, node{ node_type(It - std::begin(node_names_v)) });
        }
        else if( W.size() == 2 && W[0] == "remove" && Int(1, a) && a >= 0 && a < n ) C.m_Nodes.erase(C.m_Nodes.begin() + a);
        else if( W.size() == 3 && W[0] == "move" && Int(1, a) && Int(2, b) && a >= 0 && a < n && b >= 0 && b <= n )
        {
            const node N = C.m_Nodes[std::size_t(a)];
            C.m_Nodes.erase(C.m_Nodes.begin() + a);
            C.m_Nodes.insert(C.m_Nodes.begin() + (b > a ? b - 1 : b), N);
        }
        else { Message = std::format("not an edit: '{}' (bypass <node> <0|1>, insert <gap> <type>, remove <node>, move <node> <to>; nodes 0 .. {})", Command, n - 1); return false; }
        if( const auto P = Lower(C, 64, 64); !P.m_Error.empty() ) { Message = "refused: " + P.m_Error; return false; }
        D = std::move(C);
        Message = "ok";
        return true;
    }

    // The node types an "insert <gap>" would accept (the editor offers only these)
    inline std::vector<node_type> Insertable( const desc& D, int Gap ) noexcept
    {
        std::vector<node_type> Out;
        for( std::size_t t = 0; t < std::size(node_names_v); ++t )
        {
            desc C = D; std::string M;
            if( Edit(C, std::format("insert {} {}", Gap, node_names_v[t]), M) ) Out.push_back(node_type(t));
        }
        return Out;
    }

    // The frame's values, one read of the Scene by index: Out[i] for plan.m_Params[i], clamped to its range. A clamp is reported (Report:
    // a line) once a change of the raw value - Noted keeps the last reported raw value a parameter (NaN: none)
    template< typename T_REPORT >
    void SnapshotParams( const plan& P, const post_settings& S, std::vector<float>& Out, std::vector<float>& Noted, T_REPORT&& Report ) noexcept
    {
        Out.resize(P.m_Params.size()); Noted.resize(P.m_Params.size(), std::numeric_limits<float>::quiet_NaN());
        for( std::size_t i = 0; i < P.m_Params.size(); ++i )
        {
            const auto& R = P.m_Params[i];
            float V = R.m_Default;
            if( R.m_Source >= 0 ) { const auto& Src = sources_v[std::size_t(R.m_Source)]; V = Src.m_pBool ? (S.*Src.m_pBool ? 1.0f : 0.0f) : S.*Src.m_pFloat; }
            const float C = std::isnan(V) ? R.m_Default : std::clamp(V, R.m_Min, R.m_Max);
            if( C != V ) { if( !(Noted[i] == V) ) { Noted[i] = V; Report(std::format("{} = {} is outside {} .. {}: {} is used", R.m_Name, V, R.m_Min, R.m_Max, C)); } }
            else Noted[i] = std::numeric_limits<float>::quiet_NaN();
            Out[i] = C;
        }
    }

    // The chain as the editor shows it (xRenderEditor::DescribePost), a line each, fields split by tabs:
    //      E <the description's error>
    //      N <index> <type> <on 0|1>                                           the authored nodes, in order (0: bypassed)
    //      P <name> <value> <source> <min> <max> <the Scene's value when clamped, else empty>
    //      X <node> <ran 0|1> <source map> <scope> <kernel> <W> <H> <groups x> <groups y> <runs when>     the frame's passes
    //      I <node> 0 <source map> <scope> <kernel> <W> <H> <groups x> <groups y> <runs when>             the initialisations
    inline std::string Describe( const desc& D, const plan& P, const std::string& Error, const std::vector<float>& Snapshot, const std::vector<float>& Noted, const std::vector<std::uint8_t>& Ran ) noexcept
    {
        static constexpr const char* s_Types[]  = { "TAA", "Bloom", "Exposure", "Tone map", "Custom" };
        static constexpr const char* s_Kernel[] = { "taa (compute)", "post (compute)", "tone map (raster)", "custom (compute)" };
        static constexpr const char* s_When[]   = { "always", "TAA on", "Bloom > 0", "AutoExposure on" };
        std::string T;
        if( !Error.empty() ) T += "E\t" + Error + "\n";
        for( std::size_t i = 0; i < D.m_Nodes.size(); ++i ) T += std::format("N\t{}\t{}\t{}\n", i, s_Types[std::size_t(D.m_Nodes[i].m_Type)], D.m_Nodes[i].m_bOn ? 1 : 0);
        for( std::size_t i = 0; i < P.m_Params.size(); ++i )
        {
            const auto& R = P.m_Params[i];
            const std::string Src = R.m_Source >= 0 ? std::string(sources_v[std::size_t(R.m_Source)].m_Name) : std::string("(default)");
            const float V = i < Snapshot.size() ? Snapshot[i] : R.m_Default;
            const std::string Raw = i < Noted.size() && !std::isnan(Noted[i]) ? std::format("{}", Noted[i]) : std::string();
            T += std::format("P\t{}\t{}\t{}\t{}\t{}\t{}\n", R.m_Name, R.m_Type == param_type::BOOL ? (V > 0.5f ? "on" : "off") : std::format("{}", V), Src, R.m_Min, R.m_Max, Raw);
        }
        const auto Pass = [&](char K, const pass& Pa, bool bRan)
        {
            T += std::format("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", K, Pa.m_Node, bRan ? 1 : 0, Pa.m_Source, Pa.m_Scope, s_Kernel[std::size_t(Pa.m_Kernel)], Pa.m_W, Pa.m_H, Pa.m_GroupsX, Pa.m_GroupsY, s_When[std::size_t(Pa.m_When)]);
        };
        for( std::size_t i = 0; i < P.m_Passes.size(); ++i ) Pass('X', P.m_Passes[i], i < Ran.size() && Ran[i]);
        for( const auto& Pa : P.m_Inits ) Pass('I', Pa, false);
        return T;
    }

    // A pass's push words from the snapshot (the frame's own words - sizes, mode, matrices - written by the caller)
    inline void FillSlots( const pass& Pa, const std::vector<float>& Snapshot, float Dt, float ViewExposure, void* pPush ) noexcept
    {
        for( const auto& s : Pa.m_Slots )
        {
            float V = s.m_Constant;
            if( s.m_From == slot_from::PARAM ) V = Snapshot[std::size_t(s.m_Param)];
            else if( s.m_From == slot_from::FRAME_DT ) V = Dt;
            else if( s.m_From == slot_from::VIEW_EXPOSURE ) V = ViewExposure;
            std::memcpy(static_cast<std::byte*>(pPush) + std::size_t(s.m_Word) * 4, &V, 4);
        }
    }
}
#endif
