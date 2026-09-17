// Minimal C++ probe for the SpacemiT NPU execution provider.
//
// The Python binding (`spacemi_ort` wheel) segfaults intermittently, while SpacemiT's
// own C++ binaries work, so all NPU verdicts have to be taken through this channel.
//
// Build:
//   g++ -O2 -std=c++17 npu_probe.cc -o npu_probe \
//       -I$PKG/include -L$PKG/lib -lonnxruntime \
//       -Wl,-rpath,$PKG/lib -Wl,-rpath-link,$PKG/lib
// Run:
//   ./npu_probe model.onnx [runs]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "spacemit_ort_env.h"

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.onnx [runs] [cpu_only]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int runs = argc > 2 ? atoi(argv[2]) : 3;
    bool cpu_only = argc > 3 && strcmp(argv[3], "cpu_only") == 0;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "npu_probe");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    std::unordered_map<std::string, std::string> popts;
    if (!cpu_only) {
        Ort::SessionOptionsSpaceMITEnvInit(so, popts);
    }

    printf("[1] model  : %s%s\n", path, cpu_only ? "  (CPU only)" : "");
    fflush(stdout);

    auto t0 = Clock::now();
    Ort::Session session(env, path, so);
    printf("[2] init   : %.1f ms\n", ms_since(t0));
    fflush(stdout);

    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<std::string> in_names_s, out_names_s;
    for (size_t i = 0; i < session.GetInputCount(); i++)
        in_names_s.emplace_back(session.GetInputNameAllocated(i, alloc).get());
    for (size_t i = 0; i < session.GetOutputCount(); i++)
        out_names_s.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
    std::vector<const char *> in_names, out_names;
    for (auto &s : in_names_s) in_names.push_back(s.c_str());
    for (auto &s : out_names_s) out_names.push_back(s.c_str());

    // Keep the backing buffers alive: Ort::Value does not copy.
    std::vector<std::vector<uint8_t>> bufs;
    std::vector<Ort::Value> inputs;
    bufs.reserve(session.GetInputCount());
    inputs.reserve(session.GetInputCount());
    for (size_t i = 0; i < session.GetInputCount(); i++) {
        // Hold the TypeInfo: TensorTypeAndShapeInfo keeps a raw pointer into it.
        auto type_info = session.GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        for (auto &d : shape)
            if (d < 0) d = 1;
        size_t n = 1;
        for (auto d : shape) n *= (size_t)d;
        auto et = info.GetElementType();
        size_t esz = (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)   ? 8
                     : (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 2
                                                                     : 4;
        bufs.emplace_back(n * esz, 0);
        inputs.push_back(Ort::Value::CreateTensor(alloc.GetInfo(), bufs.back().data(),
                                                  bufs.back().size(), shape.data(), shape.size(),
                                                  et));
        printf("[3] input  : %s shape=[", in_names_s[i].c_str());
        for (auto d : shape) printf("%lld,", (long long)d);
        printf("] elem_type=%d\n", (int)et);
        fflush(stdout);
    }

    auto run_once = [&]() {
        auto a = Clock::now();
        auto outs = session.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                                inputs.size(), out_names.data(), out_names.size());
        return ms_since(a);
    };

    run_once();  // warm-up + triggers provider subgraph compile
    double best = 1e18, sum = 0;
    for (int i = 0; i < runs; i++) {
        double d = run_once();
        best = d < best ? d : best;
        sum += d;
        printf("[4] run %d  : %.2f ms\n", i + 1, d);
        fflush(stdout);
    }
    printf("[5] RESULT : best %.2f ms  avg %.2f ms  OK\n", best, sum / runs);
    return 0;
}
