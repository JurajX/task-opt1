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

#include "tinysimd.hpp"

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
const int128_t VAR_0_7 = ts_set_si16(0, 1, 2, 3, 4, 5, 6, 7);
const int128_t VAR_8_15 = ts_set_si16(8, 9, 10, 11, 12, 13, 14, 15);
const int128_t DIV3 = ts_set1_si16(0x5556);
const int128_t ZEROS = ts_setzero_si128();

static void extract_green_from_block_colours(int128_t *block_green, int inten, uint32_t green)
{
    color32 block_colors[4];
    decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(green, green, green, 255), false), inten);
    for (uint32_t idx = 0; idx < 4; idx += 1) {
        block_green[idx] = ts_set1_si8((uint8_t)(block_colors[idx].g));;
    }
}

static void precompute_colours(int128_t *color_table)
{
    int128_t lows[4];
    int128_t lows_lo[4];
    int128_t lows_hi[4];
    int128_t lows_lo2[4];
    int128_t lows_hi2[4];
    for (uint8_t idx = 0; idx < 64/16; idx += 1) {
        int128_t offset = ts_set1_si16(idx<<4);
        lows_lo[idx] = ts_add_si16(VAR_0_7, offset);
        lows_hi[idx] = ts_add_si16(VAR_8_15, offset);
        lows_lo[idx] = ts_or_si128(ts_slli_si16(lows_lo[idx], 2), ts_srli_si16(lows_lo[idx], 4));
        lows_hi[idx] = ts_or_si128(ts_slli_si16(lows_hi[idx], 2), ts_srli_si16(lows_hi[idx], 4));
        lows[idx] = ts_narrow_ui8_from_si16(lows_lo[idx], lows_hi[idx]);
        lows_lo2[idx] = ts_slli_si16(lows_lo[idx], 1);
        lows_hi2[idx] = ts_slli_si16(lows_hi[idx], 1);
    }

    int128_t *colors;
    for (uint8_t hi = 0; hi < 64; hi += 1) {
        uint8_t high = ((uint8_t *)lows)[hi];
        int128_t high16 = ts_set1_si16(high);
        int128_t high16_2 = ts_set1_si16(high<<1);
        for (uint8_t idx = 0; idx < 64/16; idx += 1) {
            colors = &(color_table[4*((hi<<2)+(idx))]);
            colors[3] = ts_set1_si8(high);
            colors[0] = lows[idx];
            colors[2] = ts_narrow_ui8_from_si16(
                ts_mul_hi_ui16(ts_add_si16(lows_lo[idx], high16_2), DIV3),
                ts_mul_hi_ui16(ts_add_si16(lows_hi[idx], high16_2), DIV3)
            );
            colors[1] = ts_narrow_ui8_from_si16(
                ts_mul_hi_ui16(ts_add_si16(lows_lo2[idx], high16), DIV3),
                ts_mul_hi_ui16(ts_add_si16(lows_hi2[idx], high16), DIV3)
            );
        }
    }
}

static void accumulate_errors(int128_t &total_err_lo, int128_t &total_err_hi, int128_t &block_green, int128_t &colors)
{
    int128_t err = ts_sub_abs_ui8(block_green, colors);
    int128_t tmp_lo = ts_extend_lo_ui8_to_ui16(err);
    int128_t err2_lo = ts_mul_lo_si16(tmp_lo, tmp_lo);
    int128_t tmp_hi = ts_extend_hi_ui8_to_ui16(err);
    int128_t err2_hi = ts_mul_lo_si16(tmp_hi, tmp_hi);
    total_err_lo = ts_add_sat_ui16(total_err_lo, err2_lo);
    total_err_hi = ts_add_sat_ui16(total_err_hi, err2_hi);
}

static void adjust_bests(
    uint16_t &best_err, uint8_t &best_lo, uint8_t &best_hi,
    int128_t &total_err_lo, int128_t &total_err_hi,
    uint16_t lo, uint16_t hi)
{
    uint16_t min_lo;
    uint16_t pos_lo;
    ts_min_pos_ui16(min_lo, pos_lo, total_err_lo);
    uint16_t min_hi;
    uint16_t pos_hi;
    ts_min_pos_ui16(min_hi, pos_hi, total_err_hi);

    if (min_lo < best_err) {
        best_err = min_lo;
        uint16_t b_lo[8];
        ts_store_i128((int128_t *)b_lo, ts_add_si16(VAR_0_7, ts_set1_si16(lo)));
        best_lo = b_lo[pos_lo];
        best_hi = hi;
    }
    if (min_hi < best_err) {
        best_err = min_hi;
        uint16_t b_lo[8];
        ts_store_i128((int128_t *)b_lo, ts_add_si16(VAR_8_15, ts_set1_si16(lo)));
        best_lo = b_lo[pos_hi];
        best_hi = hi;
    }
}

static void create_etc1_to_dxt1_6_conversion_table()
{
    uint32_t n = 0;
    int128_t block_green[4];
    int128_t color_table[4*64*4];
    precompute_colours(color_table);
    int128_t *colors;

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
                        for (uint16_t lo = 0; lo < 64; lo += 16) {
                            colors = &(color_table[4*((hi<<2)+(lo>>4))]);
                            int128_t total_err_lo = ZEROS;
                            int128_t total_err_hi = ZEROS;
                            for (uint16_t s = low_selector; s <= high_selector; s += 1) {
                                uint8_t idx = g_etc1_to_dxt1_selector_mappings[m][s];
                                accumulate_errors(total_err_lo, total_err_hi, block_green[s], colors[idx]);
                            }
                            adjust_bests(best_err, best_lo, best_hi, total_err_lo, total_err_hi, lo, hi);
                        }
                    }
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
