# Minimal CMake package config so stable-diffusion.cpp's
#   -DSD_USE_SYSTEM_GGML=ON  ->  find_package(ggml REQUIRED)
# resolves against the SpacemiT-provided ggml (0.16.0) that ships the
# CPU_RISCV64_SPACEMIT backend (A100 AI cores: IME1/IME2 kernels + TCM).
#
# Ubuntu/Bianbu does not install a ggmlConfig.cmake, hence this shim.

set(ggml_FOUND TRUE)
set(GGML_FOUND TRUE)

if(NOT TARGET ggml::ggml)
    add_library(ggml::ggml INTERFACE IMPORTED)
    set_target_properties(ggml::ggml PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
        INTERFACE_LINK_LIBRARIES "/usr/lib/libggml.so"
    )
endif()

if(NOT TARGET ggml::ggml-base)
    add_library(ggml::ggml-base INTERFACE IMPORTED)
    set_target_properties(ggml::ggml-base PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
        INTERFACE_LINK_LIBRARIES "/usr/lib/libggml-base.so"
    )
endif()

if(NOT TARGET ggml::ggml-cpu)
    add_library(ggml::ggml-cpu INTERFACE IMPORTED)
    set_target_properties(ggml::ggml-cpu PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
        INTERFACE_LINK_LIBRARIES "/usr/lib/libggml-cpu.so"
    )
endif()
