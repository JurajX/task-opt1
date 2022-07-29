// Extract from basisu_transcoder.cpp
// Copyright (C) 2019-2021 Binomial LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <immintrin.h>

//************************** Helpers and Boilerplate **************************/

#include "basisu_headers.h"

/**
 * Helper to return the current time in milliseconds.
 */
static unsigned millis() {
    return static_cast<unsigned>((clock() * 1000LL) / CLOCKS_PER_SEC);
}

/**
 * Prebuilt table with known results.
 */
static const etc1_to_dxt1_56_solution known[32 * 8 * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS * NUM_ETC1_TO_DXT1_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_dxt1_6.inc"
};

/**
 * Helper to compare two tables to see if they match.
 */
static bool verifyTable(const etc1_to_dxt1_56_solution* a, const etc1_to_dxt1_56_solution* b) {
    for (unsigned n = 0; n < 32 * 8 * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS * NUM_ETC1_TO_DXT1_SELECTOR_RANGES; n++) {
        if (a->m_hi != b->m_hi || a->m_lo != b->m_lo || a->m_err != b->m_err) {
            printf("Failed with n = %d\n", n);
            return false;
        }
        a++;
        b++;
    }
    return true;
}

//************************ Optimisation Task Goes Here ************************/

/**
 * Results stored here.
 */
static etc1_to_dxt1_56_solution result[32 * 8 * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS * NUM_ETC1_TO_DXT1_SELECTOR_RANGES];

/**
 * Function to optimise.
 */

// some constants to declare for convenience - this has negligible effect on performance
const __m128i VAR_0_7 = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
const __m128i VAR_8_15 = _mm_set_epi16(15, 14, 13, 12, 11, 10, 9, 8);
const __m128i DIV_3 = _mm_set1_epi16(0x5556);
const __m128i ZEROS = _mm_setzero_si128();

static void extract_green_from_block_colours(__m128i *block_green, int inten, uint32_t green)
{
    color32 block_colors[4];
    decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(green, green, green, 255), false), inten);
    for (uint32_t idx = 0; idx < 4; idx += 1) {
        block_green[idx] = _mm_set1_epi8((uint8_t)(block_colors[idx].g));;
    }
}

static void make_colours(__m128i *colors, uint8_t high, __m128i &high16, __m128i &lows_lo, __m128i &lows_hi)
{
    __m128i hlf_lo = _mm_or_si128(_mm_slli_epi16(lows_lo, 2), _mm_srli_epi16(lows_lo, 4));
    __m128i hlf_hi = _mm_or_si128(_mm_slli_epi16(lows_hi, 2), _mm_srli_epi16(lows_hi, 4));
    colors[3] = _mm_set1_epi8(high);
    colors[0] = _mm_packus_epi16(hlf_lo, hlf_hi);
    colors[2] = _mm_packus_epi16(
        _mm_mulhi_epu16(_mm_add_epi16(_mm_slli_epi16(high16, 1), hlf_lo), DIV_3),
        _mm_mulhi_epu16(_mm_add_epi16(_mm_slli_epi16(high16, 1), hlf_hi), DIV_3)
    );
    colors[1] = _mm_packus_epi16(
        _mm_mulhi_epu16(_mm_add_epi16(_mm_slli_epi16(hlf_lo, 1), high16), DIV_3),
        _mm_mulhi_epu16(_mm_add_epi16(_mm_slli_epi16(hlf_hi, 1), high16), DIV_3)
    );
}

static void accumulate_errors(__m128i &total_err_lo, __m128i &total_err_hi, __m128i &block_green, __m128i &colors)
{
    __m128i err = _mm_or_si128(
        _mm_subs_epu8(block_green, colors),
        _mm_subs_epu8(colors, block_green)
    );
    __m128i tmp_lo = _mm_unpacklo_epi8(err, ZEROS);
    __m128i err2_lo = _mm_mullo_epi16(tmp_lo, tmp_lo);
    __m128i tmp_hi = _mm_unpackhi_epi8(err, ZEROS);
    __m128i err2_hi = _mm_mullo_epi16(tmp_hi, tmp_hi);
    total_err_lo = _mm_adds_epu16(total_err_lo, err2_lo);
    total_err_hi = _mm_adds_epu16(total_err_hi, err2_hi);
}

static void adjust_bests(
    uint16_t &best_err, uint8_t &best_lo, uint8_t &best_hi,
    __m128i &total_err_lo, __m128i &total_err_hi,
    __m128i &lows_lo, __m128i &lows_hi, uint16_t hi)
{
    __m128i min_and_pos_lo = _mm_minpos_epu16(total_err_lo);
    __m128i min_and_pos_hi = _mm_minpos_epu16(total_err_hi);
    uint16_t min_lo = _mm_extract_epi16(min_and_pos_lo, 0);
    uint16_t min_hi = _mm_extract_epi16(min_and_pos_hi, 0);

    if (min_lo < best_err) {
        best_err = min_lo;
        uint16_t pos = _mm_extract_epi16(min_and_pos_lo, 1);
        best_lo = ((uint16_t*)(&lows_lo))[pos];
        best_hi = hi;
    }
    if (min_hi < best_err) {
        best_err = min_hi;
        uint16_t pos = _mm_extract_epi16(min_and_pos_hi, 1);
        best_lo = ((uint16_t*)(&lows_hi))[pos];
        best_hi = hi;
    }
}

static void create_etc1_to_dxt1_6_conversion_table()
{
    uint32_t n = 0;
    __m128i block_green[4];
    __m128i colors[4];

    for (int inten = 0; inten < 8; inten += 1) {
        for (uint32_t g = 0; g < 32; g += 1) {
            extract_green_from_block_colours(block_green, inten, g);
            for (uint32_t sr = 0; sr < NUM_ETC1_TO_DXT1_SELECTOR_RANGES; sr += 1) {
                const uint16_t low_selector = g_etc1_to_dxt1_selector_ranges[sr].m_low;
                const uint16_t high_selector = g_etc1_to_dxt1_selector_ranges[sr].m_high;
                for (uint32_t m = 0; m < NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS; m += 1) {
                    uint8_t best_lo = 0;
                    uint8_t best_hi = 0;
                    uint16_t best_err = UINT16_MAX;

                    for (uint16_t hi = 0; hi < 64; hi += 1) {
                        uint16_t high = (hi << 2) | (hi >> 4);
                        __m128i high16 = _mm_set1_epi16(high);
                        for (uint16_t lo = 0; lo < 64; lo += 16) {
                            __m128i offset = _mm_set1_epi16(lo);
                            __m128i lows_lo = _mm_add_epi16(VAR_0_7, offset);
                            __m128i lows_hi = _mm_add_epi16(VAR_8_15, offset);
                            make_colours(colors, high, high16, lows_lo, lows_hi);

                            __m128i total_err_lo = _mm_setzero_si128();
                            __m128i total_err_hi = _mm_setzero_si128();
                            for (uint16_t s = low_selector; s <= high_selector; s += 1) {
                                uint8_t idx = g_etc1_to_dxt1_selector_mappings[m][s];
                                accumulate_errors(total_err_lo, total_err_hi, block_green[s], colors[idx]);
                            }
                            adjust_bests(best_err, best_lo, best_hi, total_err_lo, total_err_hi, lows_lo, lows_hi, hi);
                        }
                    }
                    assert(best_err <= 0xFFFF);
                    result[n] = (etc1_to_dxt1_56_solution){ best_lo, best_hi, best_err };
                    n += 1;
                } // m
            } // sr
        } // g
    } // inten
}

//******************************** Entry Point ********************************/

/**
 * Tests the generation and benchmarks it.
 */
int main(int /*argc*/, char* /*argv*/[]) {
    // Run this once and compare the result to the known table
    create_etc1_to_dxt1_6_conversion_table();
    if (!verifyTable(result, known)) {
        printf("Generated results don't match known values\n");
    }

    // Perform multiple runs and take the best time
    unsigned best = UINT32_MAX;
    for (int n = 10; n > 0; n--) {
        unsigned time = millis();
        create_etc1_to_dxt1_6_conversion_table();
        time = millis() - time;
        if (time < best) {
            best = time;
        }
    }

    printf("Best run took %dms\n", best);
    return 0;
}
