#pragma once
// The tests never stop on a dialog: an assert, an abort or a crash writes to stderr and the process ends with a non-zero code (ctest reports
// the failure) - no CRT assert box, no abort box, no Windows Error Reporting box on the screen of whoever is at the machine
#if defined(_WIN32)
#include <crtdbg.h>
#include <cstdlib>

extern "C" __declspec(dllimport) unsigned int __stdcall SetErrorMode(unsigned int uMode);

namespace xlionrender::test
{
    inline const bool g_bNoDialogs = []
    {
        _set_error_mode(_OUT_TO_STDERR);                                  // assert(): its message to stderr, then abort
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);     // abort(): no box, no WER report
        for (const int Type : { _CRT_WARN, _CRT_ERROR, _CRT_ASSERT })     // the debug CRT's _ASSERT / _RPT reports
        {
            _CrtSetReportMode(Type, _CRTDBG_MODE_FILE);
            _CrtSetReportFile(Type, _CRTDBG_FILE_STDERR);
        }
        SetErrorMode(0x0001u | 0x0002u);                                  // SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX: a crash ends the process
        return true;
    }();
}
#endif
