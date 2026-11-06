#pragma once
// The environment's switches (XLION_...): one place reads them. std::getenv is the standard's and safe here (read-only, at startup or once into a static);
// MSVC deprecates it (C4996, getenv_s instead), so its warning is answered once, here, instead of at every switch
#include <cstdlib>
#include <cstdio>

namespace xlionrender
{
    inline const char* Env( const char* pName ) noexcept
    {
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4996)
#endif
        return std::getenv(pName);
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif
    }

    // (the switches' values read: std::sscanf, which MSVC deprecates the same way - its _s form is MSVC's own; numbers only here, no %s)
    template< typename... T_ARGS >
    int Scan( const char* pText, const char* pFormat, T_ARGS... Args ) noexcept
    {
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4996)
#endif
        return std::sscanf(pText, pFormat, Args...);
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif
    }

    // (a measurement's file: std::fopen, the same)
    inline std::FILE* Open( const char* pPath, const char* pMode ) noexcept
    {
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4996)
#endif
        return std::fopen(pPath, pMode);
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif
    }
}
