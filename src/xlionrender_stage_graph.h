#pragma once
// Frame stages as data (render graph R1; xLION_Docs render_graph_stage_nodes_plan_2026-11-09.md): a stage's passes described by names -
// the kernel, the buffers / textures / images it binds, its push words, its dispatch size - lowered once against a registry (names to
// indices: no strings per frame) and run by one executor, which also declares each pass to the render graph (the Frame graph shows the
// stage's real passes). The registry: the kernels the instance registers once; the resources and frame values each view fills per frame.
#include "dependencies/xGPU/source/xGPU.h"
#include "xlionrender_render_graph.h"
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace xlionrender::stage
{
    // the dispatch's size: the view's pixels in 8 x 8 tiles
    enum class size_rule : std::uint8_t { TILES8 };
    enum class kind      : std::uint8_t { BUFFER, TEXTURE, IMAGE };

    // a bound resource by its render graph name (what the Frame graph calls it), and how the pass uses it
    struct binding
    {
        std::string     m_Name;
        rg::lifetime    m_Life      = rg::lifetime::FRAME;
        rg::usage       m_Usage     = rg::usage::STORAGE;
        bool            m_bWrite    = false;
        rg::policy      m_Policy    = rg::policy::NONE;
    };
    // a push word source: a frame value by name (all its words), or one constant word when m_Frame is empty
    struct word { std::string m_Frame; std::uint32_t m_Constant = 0; };

    struct pass
    {
        std::string             m_Name, m_Kernel;
        std::vector<binding>    m_Buffers, m_Textures, m_Images;
        std::vector<word>       m_Push;
        size_rule               m_Size = size_rule::TILES8;
    };
    struct desc { std::vector<pass> m_Passes; };

    // names to objects: Add once (the index stays), Set per frame
    template< class T >
    struct table
    {
        std::vector<std::string> m_Names;
        std::vector<T>           m_Items;
        int  Add ( std::string Name, T Item = {} ) { if( int i = Find(Name); i >= 0 ) { m_Items[std::size_t(i)] = Item; return i; } m_Names.push_back(std::move(Name)); m_Items.push_back(Item); return int(m_Items.size()) - 1; }
        int  Find( std::string_view Name ) const noexcept { for( std::size_t i = 0; i < m_Names.size(); ++i ) if( m_Names[i] == Name ) return int(i); return -1; }
    };
    struct value { std::array<std::uint32_t, 16> m_Words{}; int m_n = 0; };     // (a matrix: 16 words)
    struct kernel_info { xgpu::compute* m_pCompute = nullptr; int m_nBuffers = 0, m_nTextures = 0, m_nImages = 0, m_PushBytes = 0; std::string m_Recorded; };

    struct registry
    {
        table<kernel_info>              m_Kernels;
        table<xgpu::buffer*>            m_Buffers;
        table<const xgpu::texture*>     m_Textures;     // (sampled textures and storage images alike)
        table<value>                    m_Frame;
        int                             m_Width = 0, m_Height = 0;

        void SetFloats( int i, const float* p, int n ) noexcept { auto& V = m_Frame.m_Items[std::size_t(i)]; V.m_n = n; std::memcpy(V.m_Words.data(), p, std::size_t(n) * 4); }
        void SetU32   ( int i, std::uint32_t u ) noexcept { auto& V = m_Frame.m_Items[std::size_t(i)]; V.m_n = 1; V.m_Words[0] = u; }
    };

    // lowered: indices into the registry
    struct lowered_pass
    {
        const pass*         m_pPass = nullptr;
        int                 m_Kernel = -1;
        std::vector<int>    m_Buffers, m_Textures, m_Images;
        std::vector<int>    m_Push;         // a frame value's index, or -1 - k - 1 for the pass's k-th push word as a constant
    };
    struct plan { std::vector<lowered_pass> m_Passes; std::string m_Error; };

    // Every name must exist and every pass must match its kernel's layout (the push's size from the frame values registered now)
    inline plan Lower( const desc& D, const registry& R ) noexcept
    {
        plan P;
        const auto Fail = [&]( const pass& Pa, const std::string& Why ) { P.m_Passes.clear(); P.m_Error = Pa.m_Name + ": " + Why; return P; };
        for( const auto& Pa : D.m_Passes )
        {
            lowered_pass L{ &Pa, R.m_Kernels.Find(Pa.m_Kernel) };
            if( L.m_Kernel < 0 ) return Fail(Pa, "no kernel '" + Pa.m_Kernel + "'");
            const auto& K = R.m_Kernels.m_Items[std::size_t(L.m_Kernel)];
            for( const auto& B : Pa.m_Buffers )  { const int i = R.m_Buffers.Find(B.m_Name);  if( i < 0 ) return Fail(Pa, "no buffer '" + B.m_Name + "'");  L.m_Buffers.push_back(i); }
            for( const auto& B : Pa.m_Textures ) { const int i = R.m_Textures.Find(B.m_Name); if( i < 0 ) return Fail(Pa, "no texture '" + B.m_Name + "'"); L.m_Textures.push_back(i); }
            for( const auto& B : Pa.m_Images )   { const int i = R.m_Textures.Find(B.m_Name); if( i < 0 ) return Fail(Pa, "no image '" + B.m_Name + "'");   L.m_Images.push_back(i); }
            if( int(L.m_Buffers.size()) != K.m_nBuffers || int(L.m_Textures.size()) != K.m_nTextures || int(L.m_Images.size()) != K.m_nImages )
                return Fail(Pa, "the bindings do not match the kernel '" + Pa.m_Kernel + "'");
            int Bytes = 0;
            for( std::size_t k = 0; k < Pa.m_Push.size(); ++k )
            {
                const auto& W = Pa.m_Push[k];
                if( W.m_Frame.empty() ) { L.m_Push.push_back(-int(k) - 1); Bytes += 4; continue; }
                const int i = R.m_Frame.Find(W.m_Frame);
                if( i < 0 ) return Fail(Pa, "no frame value '" + W.m_Frame + "'");
                L.m_Push.push_back(i); Bytes += 4 * R.m_Frame.m_Items[std::size_t(i)].m_n;
            }
            if( Bytes != K.m_PushBytes ) return Fail(Pa, "pushes " + std::to_string(Bytes) + " bytes, the kernel '" + Pa.m_Kernel + "' takes " + std::to_string(K.m_PushBytes));
            P.m_Passes.push_back(std::move(L));
        }
        return P;
    }

    // One executor for every stage: the dispatches, then their declarations (the reads before the writes, as the hand-written ones)
    template< class T_WINDOW >
    void Run( const plan& P, registry& R, T_WINDOW& Window, rg::graph& G ) noexcept
    {
        for( const auto& L : P.m_Passes )
        {
            const auto& Pa = *L.m_pPass;
            const auto& K  = R.m_Kernels.m_Items[std::size_t(L.m_Kernel)];
            std::array<xgpu::buffer*, 16> B{}; std::array<const xgpu::texture*, 16> T{}, I{};
            for( std::size_t i = 0; i < L.m_Buffers.size();  ++i ) B[i] = R.m_Buffers.m_Items[std::size_t(L.m_Buffers[i])];
            for( std::size_t i = 0; i < L.m_Textures.size(); ++i ) T[i] = R.m_Textures.m_Items[std::size_t(L.m_Textures[i])];
            for( std::size_t i = 0; i < L.m_Images.size();   ++i ) I[i] = R.m_Textures.m_Items[std::size_t(L.m_Images[i])];
            std::array<std::uint32_t, 64> Push{}; int n = 0;
            for( const int s : L.m_Push )
            {
                if( s < 0 ) { Push[std::size_t(n++)] = Pa.m_Push[std::size_t(-s - 1)].m_Constant; continue; }
                const auto& V = R.m_Frame.m_Items[std::size_t(s)];
                std::copy_n(V.m_Words.begin(), V.m_n, Push.begin() + n); n += V.m_n;
            }
            const int X = (R.m_Width + 7) / 8, Y = (R.m_Height + 7) / 8;      // (size_rule::TILES8, the only one so far)
            Window.Dispatch(*K.m_pCompute, std::span<xgpu::buffer* const>(B.data(), L.m_Buffers.size()), std::span<const xgpu::texture* const>(T.data(), L.m_Textures.size())
                           , std::span<const xgpu::texture* const>(I.data(), L.m_Images.size()), Push.data(), std::size_t(n) * 4, X, Y, 1);
            const int S = G.Begin(Pa.m_Name, K.m_Recorded);
            for( const auto* pList : { &Pa.m_Textures, &Pa.m_Buffers, &Pa.m_Images } ) for( const auto& Bd : *pList ) if( !Bd.m_bWrite ) G.Read(S, G.Resource(Bd.m_Name, Bd.m_Life), Bd.m_Usage);
            for( const auto* pList : { &Pa.m_Textures, &Pa.m_Buffers, &Pa.m_Images } ) for( const auto& Bd : *pList ) if(  Bd.m_bWrite ) G.Write(S, G.Resource(Bd.m_Name, Bd.m_Life), Bd.m_Usage, Bd.m_Policy);
        }
    }
}
