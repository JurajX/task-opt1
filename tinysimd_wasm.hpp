#pragma once

#include <wasm_simd128.h>

#define int128_t v128_t

#define ts_set_si16(a0, a1, a2, a3, a4, a5, a6, a7) wasm_u16x8_make(a0, a1, a2, a3, a4, a5, a6, a7)
#define ts_set1_si8(a) wasm_i8x16_splat(a)
#define ts_set1_si16(a) wasm_i16x8_splat(a)
#define ts_setzero_si128() wasm_u64x2_const_splat(0)

#define ts_or_si128(a, b) wasm_v128_or(a, b)

#define ts_add_si8(a, b) wasm_i8x16_add(a, b)
#define ts_add_si16(a, b) wasm_i16x8_add(a, b)
#define ts_add_sat_ui16(a, b) wasm_u16x8_add_sat(a, b)

static int128_t ts_sub_abs_ui8(int128_t &a, int128_t &b)
{
    return ts_or_si128( wasm_u8x16_sub_sat(a, b), wasm_u8x16_sub_sat(b, a) );
}

// static int128_t ts_slli_si8(int128_t &a, uint8_t imm)
// {
//     int128_t tmp1 = _mm_srli_epi16(_mm_slli_epi16(a, 8+imm), 8);
//     int128_t tmp2 = _mm_slli_epi16(_mm_srli_epi16(a, 8), 8+imm);
//     return ts_or_si128(tmp1, tmp2);
// }
// static int128_t ts_srli_si8(int128_t &a, uint8_t imm)
// {
//     int128_t tmp1 = _mm_slli_epi16(_mm_srli_epi16(a, 8+imm), 8);
//     int128_t tmp2 = _mm_srli_epi16(_mm_slli_epi16(a, 8), 8+imm);
//     return ts_or_si128(tmp1, tmp2);
// }

#define ts_slli_si16(a, imm) wasm_i16x8_shl(a, imm)
#define ts_srli_si16(a, imm) wasm_u16x8_shr(a, imm)


#define ts_narrow_ui8_from_si16(a, b) wasm_u8x16_narrow_i16x8(a, b)

#define ts_extend_lo_ui8_to_ui16(a) wasm_u16x8_extend_low_u8x16(a)
#define ts_extend_hi_ui8_to_ui16(a) wasm_u16x8_extend_high_u8x16(a)

static int128_t ts_mul_hi_ui16(const int128_t &&a, const int128_t &b)
{
    int128_t tmp1 = wasm_u32x4_extmul_low_u16x8(a, b);
    int128_t tmp2 = wasm_u32x4_extmul_high_u16x8(a, b);
    tmp1 = wasm_u32x4_shr(tmp1, 16);
    tmp2 = wasm_u32x4_shr(tmp2, 16);
    return wasm_u16x8_narrow_i32x4(tmp1, tmp2);
}

static int128_t ts_mul_lo_si16(const int128_t &a, const int128_t &b)
{
    const int128_t MASK = wasm_u32x4_const(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
    int128_t tmp1 = wasm_u32x4_extmul_low_u16x8(a, b);
    int128_t tmp2 = wasm_u32x4_extmul_high_u16x8(a, b);
    tmp1 = wasm_v128_and(tmp1, MASK);
    tmp2 = wasm_v128_and(tmp2, MASK);
    return wasm_u16x8_narrow_i32x4(tmp1, tmp2);
}

#define ts_store_i128(mem, b) wasm_v128_store((void *)mem, b)

// static int ts_extract_si16(int128_t &a, const int imm)
// {
//     int16_t elem;
//     wasm_v128_store16_lane((void *)(&elem), a, imm);
//     return elem;
// }

static void ts_min_pos_ui16(uint16_t &min, uint16_t &pos, int128_t &a)
{
    int128_t vec = a;
    vec = wasm_u16x8_min(vec, wasm_i16x8_shuffle(vec, vec, 7, 0, 1, 2, 3, 4, 5, 6));
    vec = wasm_u16x8_min(vec, wasm_i16x8_shuffle(vec, vec, 6, 7, 0, 1, 2, 3, 4, 5));
    vec = wasm_u16x8_min(vec, wasm_i16x8_shuffle(vec, vec, 4, 5, 6, 7, 0, 1, 2, 3));
    wasm_v128_store16_lane(&min, vec, 0);
    int128_t mask = wasm_i16x8_eq(vec, a);
    pos = __builtin_ctz(wasm_i16x8_bitmask(mask));
}