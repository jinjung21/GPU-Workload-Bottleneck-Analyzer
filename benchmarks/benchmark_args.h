#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct BenchmarkArgs {
    bool csv = false;
    int n = -1;
    int rows = -1;
    int cols = -1;
    int iterations = -1;
};

inline int parse_positive_int(const char* value, const char* name) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 2147483647L) {
        std::fprintf(stderr, "Invalid positive integer for %s: %s\n", name, value);
        std::exit(1);
    }
    return static_cast<int>(parsed);
}

inline BenchmarkArgs parse_benchmark_args(int argc, char** argv) {
    BenchmarkArgs args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0) {
            args.csv = true;
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            args.n = parse_positive_int(argv[++i], "--n");
        } else if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            args.rows = parse_positive_int(argv[++i], "--rows");
        } else if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            args.cols = parse_positive_int(argv[++i], "--cols");
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            args.iterations = parse_positive_int(argv[++i], "--iterations");
        } else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            std::exit(1);
        }
    }
    return args;
}

inline int choose_arg(int provided, int fallback) {
    return provided > 0 ? provided : fallback;
}
