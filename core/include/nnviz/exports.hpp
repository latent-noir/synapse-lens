#pragma once
// ─── Shared Export Macros ───────────────────────────────────────────────────
// Defines DLL export/import macros for cross-platform shared library builds.

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef NNVIZ_CORE_EXPORTS
        #define NNVIZ_API __declspec(dllexport)
    #else
        #define NNVIZ_API __declspec(dllimport)
    #endif
#else
    #define NNVIZ_API __attribute__((visibility("default")))
#endif
