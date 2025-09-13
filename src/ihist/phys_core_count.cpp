/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include "phys_core_count.hpp"

#include <cstddef>

#ifdef __linux__
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>

#include <dirent.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#ifdef _WIN32
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace ihist::internal {

#ifdef __linux__
auto get_physical_core_count_linux() -> int {
    auto dir_closer = [](DIR *d) { closedir(d); };
    std::unique_ptr<DIR, decltype(dir_closer)> dir(
        opendir("/sys/devices/system/cpu"), dir_closer);
    if (not dir)
        return -1;

    std::set<int> core_ids;
    struct dirent *entry;
    while ((entry = readdir(dir.get())) != nullptr) {
        if (std::strncmp(entry->d_name, "cpu", 3) == 0 &&
            std::isdigit(entry->d_name[3])) {
            auto const path = "/sys/devices/system/cpu/" +
                              std::string(entry->d_name) + "/topology/core_id";
            auto file_closer = [](FILE *f) { std::fclose(f); };
            std::unique_ptr<FILE, decltype(file_closer)> file(
                std::fopen(path.c_str(), "r"), file_closer);
            if (not file) {
                return -1;
            }
            int core_id;
            if (std::fscanf(file.get(), "%d", &core_id) != 1) {
                return -1;
            }
            core_ids.insert(core_id);
        }
    }
    return core_ids.size();
}
#endif

#ifdef __APPLE__
auto get_physical_core_count_macos() -> int {
    int phys_cores{};
    std::size_t size = sizeof(phys_cores);
    if (sysctlbyname("hw.physicalcpu", &phys_cores, &size, nullptr, 0) != 0) {
        return -1;
    }
    return phys_cores;
}
#endif

#ifdef _WIN32
auto get_physical_core_count_win32() -> int {
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr,
                                     &buffer_size);
    std::vector<char> buffer(buffer_size);
    if (not GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
                buffer.data()),
            &buffer_size)) {
        return -1;
    }

    int phys_cores = 0;
    for (std::size_t offset = 0; offset < buffer_size;) {
        auto const *entry =
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
                buffer.data() + offset);
        if (entry->Relationship == RelationProcessorCore) {
            ++phys_cores;
        }
        offset += entry->Size;
    }
    return phys_cores;
}
#endif

IHIST_PUBLIC auto get_physical_core_count() -> int {
    static int count =
#ifdef __linux__
        get_physical_core_count_linux();
#elif defined(__APPLE__)
        get_physical_core_count_macos();
#elif defined(_WIN32)
        get_physical_core_count_win32();
#else
        -1;
#endif
    return count;
}

} // namespace ihist::internal