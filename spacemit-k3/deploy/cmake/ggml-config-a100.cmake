# Point SD_USE_SYSTEM_GGML at the SpacemiT ggml + leejet shims build.
#
#   cmake -B build-a100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DSD_BUILD_EXAMPLES=ON -DSD_USE_SYSTEM_GGML=ON \
#         -Dggml_DIR=/home/bianbu/image-generation/src/ggml-a100 \
#         -DCMAKE_C_FLAGS=-DGGML_MAX_NAME=160 -DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=160
#
# GGML_MAX_NAME must be forced on sd.cpp's side too: the ggml library was compiled with
# -DGGML_MAX_NAME=160 (sd.cpp requires >= 160 for its tensor naming), while the installed
# ggml.h still defaults to 64. Both sides must agree or ggml_tensor layouts differ.

set(ggml_FOUND TRUE)
set(GGML_FOUND TRUE)

set(_ggml_root "/home/bianbu/image-generation/src/llama-spacemit")
set(_ggml_inc  "${_ggml_root}/ggml/include")
set(_ggml_lib  "${_ggml_root}/build-ggml/bin")

if(NOT TARGET ggml::ggml)
    add_library(ggml::ggml INTERFACE IMPORTED)
    set_target_properties(ggml::ggml PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ggml_inc}"
        INTERFACE_LINK_DIRECTORIES "${_ggml_lib}"
        INTERFACE_LINK_LIBRARIES "${_ggml_lib}/libggml.so;${_ggml_lib}/libggml-base.so;${_ggml_lib}/libggml-cpu.so"
        INTERFACE_LINK_OPTIONS "-Wl,-rpath-link,${_ggml_lib};-Wl,-rpath,${_ggml_lib}"
    )
endif()
