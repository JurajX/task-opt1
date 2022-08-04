#pragma once

#include <immintrin.h>

#define int128_t __m128i

#define ts_set_si16(a0, a1, a2, a3, a4, a5, a6, a7) _mm_set_epi16(a7, a6, a5, a4, a3, a2, a1, a0)
#define ts_set1_si8(a) _mm_set1_epi8(a)
#define ts_set1_si16(a) _mm_set1_epi16(a)
#define ts_setzero_si128() _mm_setzero_si128()

#define ts_or_si128(a, b) _mm_or_si128(a, b)

#define ts_add_si8(a, b) _mm_add_epi8(a, b)
#define ts_add_si16(a, b) _mm_add_epi16(a, b)
#define ts_add_sat_ui16(a, b) _mm_adds_epu16(a, b)

static int128_t ts_sub_abs_ui8(int128_t &a, int128_t &b)
{
    return ts_or_si128( _mm_subs_epu8(a, b), _mm_subs_epu8(b, a) );
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

#define ts_slli_si16(a, imm) _mm_slli_epi16(a, imm)
#define ts_srli_si16(a, imm) _mm_srli_epi16(a, imm)


#define ts_narrow_ui8_from_si16(a, b) _mm_packus_epi16(a, b)

#define ts_extend_lo_ui8_to_ui16(a) _mm_unpacklo_epi8(a, _mm_setzero_si128())
#define ts_extend_hi_ui8_to_ui16(a) _mm_unpackhi_epi8(a, _mm_setzero_si128())

#define ts_mul_hi_ui16(a, b) _mm_mulhi_epu16(a, b)
#define ts_mul_lo_si16(a, b) _mm_mullo_epi16(a, b)

#define ts_store_i128(mem, b) _mm_store_si128((int128_t *)mem, b)
#define ts_extract_si16(a, imm) _mm_extract_epi16(a, imm)

static void ts_min_pos_ui16(uint16_t &min, uint16_t &pos, int128_t &a)
{
    int128_t tmp = _mm_minpos_epu16(a);
    min = _mm_extract_epi16(tmp, 0);
    pos = _mm_extract_epi16(tmp, 1);
}