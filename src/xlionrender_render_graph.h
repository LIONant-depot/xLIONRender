#ifndef XLIONRENDER_RENDER_GRAPH_H
#define XLIONRENDER_RENDER_GRAPH_H
#pragma once

// The render graph's first step (R0b, xLION_Docs/render_graph_design_2026-11-09.md section 6, render_graph_r0a_contracts_2026-11-09.md):
// the frame's scopes declare what they consume and produce - resource, logical version, usage, content policy - in the order they record.
// Nothing is scheduled from it yet: the declarations are dumped (XLION_RG_DUMP=<file>, one JSON line a frame) and audited against what
// xGPU recorded (XGPU_FRAME_MAP; tools/rg_audit.py). A write makes the resource's next version; a read names the version it consumes.
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace xlionrender::rg
{
    enum class usage    : std::uint8_t { SAMPLED, STORAGE, COLOR_ATTACHMENT, DEPTH_ATTACHMENT, TRANSFER_SRC, TRANSFER_DST };
    enum class policy   : std::uint8_t { NONE, CLEAR, FULL_OVERWRITE, PRESERVE, PARTIAL, DISCARD };     // (a write's content policy; a read: NONE)
    enum class lifetime : std::uint8_t { FRAME, HISTORY, RETAINED, STATIC, EXTERNAL };

    struct access   { int m_Res; int m_Version; usage m_Usage; policy m_Policy; bool m_bWrite; };
    struct scope    { std::string m_Name, m_Recorded; bool m_bManaged = true; std::vector<access> m_Accesses; };
    struct resource { std::string m_Name; lifetime m_Life = lifetime::FRAME; int m_Version = 0; bool m_bValid = false; };

    class graph
    {
    public:

        // The frame's declarations start again (the resources and their versions stay: histories cross frames)
        void BeginFrame( void ) noexcept { m_Scopes.clear(); m_Errors.clear(); }

        // A resource by its name (the xGPU object's m_pName: the audit matches them); made the first time
        int Resource( std::string_view Name, lifetime Life ) noexcept
        {
            auto [It, bNew] = m_ByName.try_emplace(std::string(Name), int(m_Resources.size()));
            if( bNew ) m_Resources.push_back({ std::string(Name), Life });
            return It->second;
        }
        // Some scope of this frame wrote it already
        bool WrittenThisFrame( int Res ) const noexcept
        {
            for( const auto& S : m_Scopes ) for( const auto& A : S.m_Accesses ) if( A.m_Res == Res && A.m_bWrite ) return true;
            return false;
        }
        // A history that cannot be read as valid (made again, resized, reset): its next read is an initialisation path
        void Invalidate( int Res ) noexcept { m_Resources[std::size_t(Res)].m_bValid = false; }

        // A scope: Recorded is the name xGPU's frame map gives its work (a compute's m_pName, a render pass's first target, "copy ...")
        int Begin( std::string_view Name, std::string_view Recorded, bool bManaged = true ) noexcept
        {
            m_Scopes.push_back({ std::string(Name), std::string(Recorded), bManaged, {} });
            return int(m_Scopes.size()) - 1;
        }
        // A read of the resource's current version; a history read before any valid write is an error (the scope must take an init path)
        void Read( int Scope, int Res, usage U ) noexcept
        {
            auto& R = m_Resources[std::size_t(Res)];
            if( R.m_Life == lifetime::HISTORY && !R.m_bValid ) m_Errors.push_back(m_Scopes[std::size_t(Scope)].m_Name + " reads " + R.m_Name + " with no valid version");
            m_Scopes[std::size_t(Scope)].m_Accesses.push_back({ Res, R.m_Version, U, policy::NONE, false });
        }
        // A write: the resource's next version (PRESERVE and PARTIAL need a valid previous one - else an error)
        int Write( int Scope, int Res, usage U, policy P ) noexcept
        {
            auto& R = m_Resources[std::size_t(Res)];
            if( (P == policy::PRESERVE || P == policy::PARTIAL) && !R.m_bValid ) m_Errors.push_back(m_Scopes[std::size_t(Scope)].m_Name + " keeps the contents of " + R.m_Name + " that are not valid");
            ++R.m_Version; R.m_bValid = true;
            m_Scopes[std::size_t(Scope)].m_Accesses.push_back({ Res, R.m_Version, U, P, true });
            return R.m_Version;
        }

        // XLION_RG_DUMP=<file>: the frame's scopes as one JSON line (overwritten every frame: the last frame stays), errors first
        void Dump( void ) const noexcept
        {
            static const char* s_pPath = std::getenv("XLION_RG_DUMP");
            if( !s_pPath ) return;
            std::FILE* f = std::fopen(s_pPath, "w");
            if( !f ) return;
            static constexpr const char* s_Usage[]  = { "sampled", "storage", "colour attachment", "depth attachment", "transfer src", "transfer dst" };
            static constexpr const char* s_Policy[] = { "none", "clear", "full overwrite", "preserve", "partial", "discard" };
            std::fprintf(f, "{\"errors\":[");
            for( std::size_t i = 0; i < m_Errors.size(); ++i ) std::fprintf(f, "%s\"%s\"", i ? "," : "", m_Errors[i].c_str());
            std::fprintf(f, "],\"scopes\":[");
            for( std::size_t s = 0; s < m_Scopes.size(); ++s )
            {
                const auto& S = m_Scopes[s];
                std::fprintf(f, "%s{\"name\":\"%s\",\"recorded\":\"%s\",\"managed\":%s,\"accesses\":[", s ? "," : "", S.m_Name.c_str(), S.m_Recorded.c_str(), S.m_bManaged ? "true" : "false");
                for( std::size_t a = 0; a < S.m_Accesses.size(); ++a )
                {
                    const auto& A = S.m_Accesses[a];
                    std::fprintf(f, "%s{\"res\":\"%s\",\"version\":%d,\"write\":%s,\"usage\":\"%s\",\"policy\":\"%s\"}", a ? "," : "", m_Resources[std::size_t(A.m_Res)].m_Name.c_str(), A.m_Version, A.m_bWrite ? "true" : "false", s_Usage[int(A.m_Usage)], s_Policy[int(A.m_Policy)]);
                }
                std::fprintf(f, "]}");
            }
            std::fprintf(f, "]}\n");
            std::fclose(f);
        }

        // The last frame's scopes as text for the editor's view (xRenderEditor::DescribeGraph), a line each, fields split by tabs:
        //      E <error>      S <managed 0|1> <name> <recorded>      A <write 0|1> <version> <usage> <policy> <lifetime> <resource>
        // (BeginFrame clears the scopes at the next frame's start: read between frames)
        std::string Text( void ) const noexcept
        {
            static constexpr const char* s_Usage[]  = { "sampled", "storage", "colour attachment", "depth attachment", "transfer src", "transfer dst" };
            static constexpr const char* s_Policy[] = { "", "clear", "full overwrite", "preserve", "partial", "discard" };
            static constexpr const char* s_Life[]   = { "frame", "history", "retained", "static", "external" };
            std::string T;
            for( const auto& E : m_Errors ) T += "E\t" + E + "\n";
            for( const auto& S : m_Scopes )
            {
                T += std::string("S\t") + (S.m_bManaged ? "1" : "0") + "\t" + S.m_Name + "\t" + S.m_Recorded + "\n";
                for( const auto& A : S.m_Accesses )
                {
                    const auto& R = m_Resources[std::size_t(A.m_Res)];
                    T += std::string("A\t") + (A.m_bWrite ? "1" : "0") + "\t" + std::to_string(A.m_Version) + "\t" + s_Usage[int(A.m_Usage)] + "\t" + s_Policy[int(A.m_Policy)] + "\t" + s_Life[int(R.m_Life)] + "\t" + R.m_Name + "\n";
                }
            }
            return T;
        }

        // (R1a) The barriers of a managed section: SyncStart at its start (the section's own full barrier: nothing pending), then Sync(S) for
        // each scope before it records - true: it needs a barrier first (it reads what an earlier scope since the last barrier wrote, or
        // writes what one read or wrote). The scope's accesses are then pending. ponytail: one global barrier kind (compute to compute); the
        // stages from the usages when a section holds more than dispatches
        void SyncStart( void ) noexcept { m_PendingW.assign(m_Resources.size(), 0); m_PendingR.assign(m_Resources.size(), 0); }
        bool Sync( int Scope ) noexcept
        {
            m_PendingW.resize(m_Resources.size(), 0); m_PendingR.resize(m_Resources.size(), 0);
            const auto& S = m_Scopes[std::size_t(Scope)];
            bool bHazard = false;
            for( const auto& A : S.m_Accesses ) bHazard |= m_PendingW[std::size_t(A.m_Res)] || (A.m_bWrite && m_PendingR[std::size_t(A.m_Res)]);
            if( bHazard ) SyncStart();
            for( const auto& A : S.m_Accesses ) (A.m_bWrite ? m_PendingW : m_PendingR)[std::size_t(A.m_Res)] = 1;
            return bHazard;
        }

        const std::vector<std::string>& getErrors( void ) const noexcept { return m_Errors; }

    private:

        std::vector<resource>                   m_Resources;
        std::unordered_map<std::string, int>    m_ByName;
        std::vector<scope>                      m_Scopes;
        std::vector<std::string>                m_Errors;
        std::vector<std::uint8_t>               m_PendingW, m_PendingR;      // (R1a: written / read since the last barrier, by resource)
    };
}

#endif
