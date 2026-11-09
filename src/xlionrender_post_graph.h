#ifndef XLIONRENDER_POST_GRAPH_H
#define XLIONRENDER_POST_GRAPH_H
#pragma once

// The post chain as data (render graph P1a; xLION_Docs/render_graph_design_v3_2026-11-09.md 11.2-11.3, render_graph_p1_plan_2026-11-09.md):
// an authored description (the effects, in order) is validated and lowered, for a view's size, into an immutable plan - a flat list of
// passes with their kernel, images, size, dispatch and the render graph's reads / writes - that a frame walks with no name lookups.
// The plan matches what the hand-written chain records (xLION_Docs/render_graph_post_chain_inventory_2026-11-09.md): same kernels,
// sizes, bindings, order. Nothing runs it yet (P1b).
#include "xlionrender_render_graph.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string>
#include <vector>

namespace xlionrender::post
{
    // The authored effects. Bloom is one node: lowering expands it into its levels. The initialisations (TAA history, bloom levels and
    // exposure zeroed) are the nodes' lifecycle, never wired by a user
    enum class node_type : std::uint8_t { TAA, BLOOM, EXPOSURE, TONE_MAP };
    struct node { node_type m_Type; };
    struct desc
    {
        std::vector<node> m_Nodes;
        static desc Default( void ) { return { { { node_type::TAA }, { node_type::BLOOM }, { node_type::EXPOSURE }, { node_type::TONE_MAP } } }; }
    };

    // The images a pass touches, resolved by the view each frame. CHAIN: the TAA's output when it ran this frame, else the lit image
    enum class image : std::uint8_t { NONE, HDR, DEPTH, CHAIN, TAA_PREV, TAA_CUR, TAA0, TAA1, EXPOSURE, BLOOM, OUTPUT };
    struct image_ref { image m_Image = image::NONE; int m_Level = 0; bool operator==( const image_ref& ) const = default; };     // (Level: the bloom's)

    enum class kernel : std::uint8_t { TAA, POST, TONE_MAP };       // xlionrender_taa_comp, xlionrender_post_comp, xlionrender_tonemap_frag
    enum class when   : std::uint8_t { ALWAYS, TAA_ON, BLOOM_ON, AUTO_EXPOSURE };      // (settings, not topology: a slider at 0 skips, it rebuilds nothing)
    enum class push   : std::uint8_t { FILL, TAA, BLOOM_DOWN, EXPOSURE, BLOOM_UP, TONE_MAP };

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
    };

    struct plan
    {
        std::vector<pass>       m_Inits;        // once: TAA history (both images; when the TAA's images are first made), bloom levels + exposure (once a view)
        std::vector<pass>       m_Passes;       // a frame, in order
        int                     m_nBloom = 0;   // the levels the bloom uses
        std::string             m_Error;        // not empty: the description is not valid (and the plan is empty)
    };

    inline constexpr int max_bloom_levels_v = 6;        // (render_view::m_Bloom)

    // The bloom's level i: half the view, and half again, at least 1 x 1 (floor)
    inline std::array<int, 2> BloomSize( int W, int H, int i ) noexcept { return { std::max(W >> (i + 1), 1), std::max(H >> (i + 1), 1) }; }
    // The levels by the view's height: the smallest about 48 pixels high at any resolution
    inline int BloomLevels( int H ) noexcept { return std::clamp(int(std::round(std::log2(float(H) / 48.0f))), 3, max_bloom_levels_v); }

    // The description, checked and expanded for a W x H view. The bloom's down steps run where the node is, its up steps right before
    // the tone map (they only feed it; the exposure and the up steps touch different images, so this is the hand-written order)
    inline plan Lower( const desc& D, int W, int H ) noexcept
    {
        plan P;
        const auto Fail = [&](int i, std::string Why) { P = {}; P.m_Error = i >= 0 ? std::format("node {}: {}", i, Why) : Why; return P; };
        if( W <= 0 || H <= 0 ) return Fail(-1, std::format("the view is {} x {}: nothing to lower", W, H));
        int nType[4] = {};
        for( int i = 0; i < int(D.m_Nodes.size()); ++i )
        {
            const auto T = D.m_Nodes[std::size_t(i)].m_Type;
            if( std::size_t(T) >= 4 ) return Fail(i, "unknown node type");
            if( ++nType[std::size_t(T)] > 1 ) return Fail(i, "a second one of a node that keeps state (one each)");
            if( T == node_type::TAA && i != 0 ) return Fail(i, "TAA must come first: everything after it reads its output");
            if( T == node_type::TONE_MAP && i != int(D.m_Nodes.size()) - 1 ) return Fail(i, "the tone map must be last: it makes the output");
        }
        if( !nType[std::size_t(node_type::TONE_MAP)] ) return Fail(-1, "no tone map: the chain makes no output");

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
                    Groups(p); P.m_Passes.push_back(std::move(p));
                }
                for( int i = P.m_nBloom - 2; i >= 0; --i )
                {
                    const auto S = BloomSize(W, H, i);
                    pass p{ .m_Scope = "bloom up", .m_Source = std::format("Bloom, up {}, {} x {}", i, S[0], S[1]), .m_Node = n, .m_Mode = 1, .m_Push = push::BLOOM_UP,
                            .m_When = when::BLOOM_ON, .m_Level = i, .m_Bind = { Bloom(i + 1), EXPO }, .m_Dst = Bloom(i), .m_W = S[0], .m_H = S[1] };
                    p.m_Accesses = { R(Bloom(i + 1)), R(EXPO), R(Bloom(i)), Wr(p.m_Dst, rg::policy::PRESERVE) };
                    Groups(p); Ups.push_back(std::move(p));
                }
                break;
            case node_type::EXPOSURE:
            {
                // (the kernel's second slot holds the lit image here: bound, unused - the hand-written chain declares it, so this does)
                pass p{ .m_Scope = "exposure", .m_Source = "Exposure, adapt", .m_Node = n, .m_Mode = 2, .m_Push = push::EXPOSURE, .m_When = when::AUTO_EXPOSURE,
                        .m_Bind = { CHAIN, HDR }, .m_Dst = EXPO, .m_W = 1, .m_H = 1 };
                p.m_Accesses = { R(CHAIN), R(HDR), Wr(EXPO, rg::policy::PRESERVE) };
                Groups(p); P.m_Passes.push_back(std::move(p));
                break;
            }
            case node_type::TONE_MAP:
            {
                for( auto& u : Ups ) P.m_Passes.push_back(std::move(u));
                pass p{ .m_Scope = "tone map", .m_Source = std::format("Tone map, {} x {}", W, H), .m_Node = n, .m_Kernel = kernel::TONE_MAP, .m_Push = push::TONE_MAP,
                        .m_Bind = { CHAIN, Bloom(0), EXPO }, .m_Dst = { image::OUTPUT }, .m_W = W, .m_H = H };
                p.m_Accesses = { R(CHAIN), R(Bloom(0)), R(EXPO), Wr(p.m_Dst, rg::policy::FULL_OVERWRITE, rg::usage::COLOR_ATTACHMENT) };
                P.m_Passes.push_back(std::move(p));
                break;
            }
        }
        return P;
    }
}
#endif
