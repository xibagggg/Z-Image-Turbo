#!/usr/bin/env python3
"""Bridge SpacemiT's ggml (0.16.0 + A100/IME backend) to stable-diffusion.cpp.

sd.cpp master is written against its own `leejet/ggml` fork, which adds exactly five
things upstream ggml 0.16 lacks. Measured by building sd.cpp with -k 0 against the
system ggml and collecting every distinct error:

    ggml_mul_mat_i8_tensorwise      int8 tensorwise matmul   (private op)
    ggml_quantize_i8_convrot        int8 convrot packing     (private op)
    GGML_TYPE_F8_E4M3 / F8_E5M2     fp8 tensor types
    GGML_MAX_NAME >= 160            handled via -DGGML_MAX_NAME=160

The two private ops are only reached on sd.cpp's int8-tensorwise quant path, which
SD1.5/SDXL (Q8_0, fp16) never take, so aborting stubs are sufficient -- and they fail
loudly instead of silently producing wrong pixels if that assumption ever breaks.

Idempotent: safe to re-run.
"""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
HDR = os.path.join(ROOT, "ggml/include/ggml.h")
SRC = os.path.join(ROOT, "ggml/src/ggml.c")

MARK = "/* === leejet/ggml compatibility shims (added for stable-diffusion.cpp) === */"


def patch_header():
    s = open(HDR, encoding="utf-8").read()
    if MARK in s:
        print("header already patched")
        return

    # 1) fp8 tensor types. ggml 0.16 ends at GGML_TYPE_COUNT = 43, so 43/44 are free.
    old = "        GGML_TYPE_COUNT   = 43,"
    assert old in s, "GGML_TYPE_COUNT = 43 not found"
    s = s.replace(old, """        GGML_TYPE_COUNT   = 43,

        // private types from leejet/ggml, required by stable-diffusion.cpp
        GGML_TYPE_F8_E4M3 = 43,
        GGML_TYPE_F8_E5M2 = 44,
        GGML_TYPE_COUNT_SD = 45,""", 1)

    # 2) the two private ops, declared right after ggml_mul_mat
    anchor = "    // change the precision of a matrix multiplication"
    assert anchor in s, "anchor for ggml_mul_mat_set_prec not found"
    s = s.replace(anchor, """    %s
    GGML_API struct ggml_tensor * ggml_mul_mat_i8_tensorwise(
            struct ggml_context * ctx,
            struct ggml_tensor  * weight,
            struct ggml_tensor  * input,
            struct ggml_tensor  * weight_scale,
            struct ggml_tensor  * bias,
            int                   convrot_group_size);

    GGML_API struct ggml_tensor * ggml_quantize_i8_convrot(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   group_size);

%s
""" % (MARK, anchor), 1)

    open(HDR, "w", encoding="utf-8").write(s)
    print("ggml.h patched")


def patch_source():
    s = open(SRC, encoding="utf-8").read()
    if MARK in s:
        print("source already patched")
        return
    s += """

%s
struct ggml_tensor * ggml_mul_mat_i8_tensorwise(
        struct ggml_context * ctx,
        struct ggml_tensor  * weight,
        struct ggml_tensor  * input,
        struct ggml_tensor  * weight_scale,
        struct ggml_tensor  * bias,
        int                   convrot_group_size) {
    GGML_UNUSED(ctx); GGML_UNUSED(weight); GGML_UNUSED(input);
    GGML_UNUSED(weight_scale); GGML_UNUSED(bias); GGML_UNUSED(convrot_group_size);
    GGML_ABORT("ggml_mul_mat_i8_tensorwise is not implemented in this ggml build "
               "(SpacemiT ggml + leejet shims). Use an fp16 or Q8_0 model.");
}

struct ggml_tensor * ggml_quantize_i8_convrot(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   group_size) {
    GGML_UNUSED(ctx); GGML_UNUSED(a); GGML_UNUSED(group_size);
    GGML_ABORT("ggml_quantize_i8_convrot is not implemented in this ggml build "
               "(SpacemiT ggml + leejet shims). Use an fp16 or Q8_0 model.");
}
""" % MARK
    open(SRC, "w", encoding="utf-8").write(s)
    print("ggml.c patched")


if __name__ == "__main__":
    patch_header()
    patch_source()
