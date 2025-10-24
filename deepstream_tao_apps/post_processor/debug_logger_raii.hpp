#ifndef DEBUG_LOGGER_RAII_H
#define DEBUG_LOGGER_RAII_H

#include <chrono>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#define ENABLE_ENV_NAME "DEBUG"

/**
 * @brief Debug logger class using RAII (Resource Acquisition Is Initialization)
 * to automatically manage the log file.
 *
 * This class is used to log debug messages to a file.
 * The log file is automatically closed when the object is destroyed.
 * With the help of the DEBUG_DUMP_SECTION macro, we can start and end a debug section.
 * Advantages of using this class:
 * - Automatically closes the log file even if there's an early return or exception.
 * - clearly defines the scope of debug logging using {} block
 * - No need to remember to close anything
 * - Thread-safe as each instance has its own file handle
 * - Object construction and destruction has minimal overhead when it is disabled
 *
 * @param func The function name
 * @param line The line number
 * @param is_enabled Whether the logger is enabled
 *
 * @note The log file is created in the /tmp directory.
 */
class DebugLoggerRAII {
public:
    DebugLoggerRAII(const char *func, int line, bool is_enabled);
    ~DebugLoggerRAII();
    void log(const char *format, ...);

private:
    std::string filename;
    std::ofstream log_file;
    const char *func_name;
    int start_line;
    bool enabled;

    // Helper function to get formatted timestamp
    static std::string GetTimestamp();
    // Helper function to get log header with timestamp and file info
    std::string GetLogHeader();
};

class DebugConfig {
private:
    static const bool enabled;

public:
    static bool IsEnabled() { return enabled; }
};

// Debug macros
#define DEBUG_DUMP_SECTION() \
    DebugLoggerRAII debug_logger(__func__, __LINE__, DebugConfig::IsEnabled())

#define DEBUG_DUMP(...) debug_logger.log(__VA_ARGS__)

#endif // DEBUG_LOGGER_RAII_H