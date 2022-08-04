/*
 * Each of the architectures has a separate header.
 */
#if defined(__x86_64__) || defined(_M_X64)
#include "tinysimd_sse4.hpp"
// #else
// #if defined(__aarch64__) || defined(_M_ARM64)
// #include "tinysimd_neon.hpp"
#else
#if defined(__wasm_simd128__)
#include "tinysimd_wasm.hpp"
// #else
// #if defined(__VSX__) || defined(__ALTIVEC__)
// #include "tinysimd_vsx.hpp"
// #else
// include scalar fallback
// #endif
// #endif
#endif
#endif
