#ifndef CORE_H
#define CORE_H

namespace cvcore {

// Enable dll imports/exports in case of windows support
#ifdef _WIN32
#ifdef CVCORE_EXPORT_SYMBOLS             // Needs to be enabled in case of compiling dll
#define CVCORE_API __declspec(dllexport) // Exports symbols when compiling the library.
#else
#define CVCORE_API __declspec(dllimport) // Imports the symbols when linked with library.
#endif
#else
#define CVCORE_API
#endif

} // namespace cvcore
#endif // CORE_H
