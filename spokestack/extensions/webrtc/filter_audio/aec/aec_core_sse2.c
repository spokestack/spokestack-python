/*
 *  Copyright (c) 2011 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/*
 * The core AEC algorithm, SSE2 version of speed-critical functions.
 */

#ifdef __SSE2__

#include <emmintrin.h>
#include <math.h>
#include <string.h>  // memset

#include "../other/signal_processing_library.h"
#include "aec_common.h"
#include "aec_core_internal.h"
#include "aec_rdft.h"

__inline static float MulRe(float aRe, float aIm, float bRe, float bIm) {
  return aRe * bRe - aIm * bIm;
}

__inline static float MulIm(float aRe, float aIm, float bRe, float bIm) {
  return aRe * bIm + aIm * bRe;
}

static void FilterFarSSE2(AecCore* aec, float yf[2][PART_LEN1]) {
  int i;
  const int num_partitions = aec->num_partitions;
  for (i = 0; i < num_partitions; i++) {
    int j;
    int xPos = (i + aec->xfBufBlockPos) * PART_LEN1;
    int pos = i * PART_LEN1;
    // Check for wrap
    if (i + aec->xfBufBlockPos >= num_partitions) {
      xPos -= num_partitions * (PART_LEN1);
    }

    // vectorized code (four at once)
    for (j = 0; j + 3 < PART_LEN1; j += 4) {
      const __m128 xfBuf_re = _mm_loadu_ps(&aec->xfBuf[0][xPos + j]);
      const __m128 xfBuf_im = _mm_loadu_ps(&aec->xfBuf[1][xPos + j]);
      const __m128 wfBuf_re = _mm_loadu_ps(&aec->wfBuf[0][pos + j]);
      const __m128 wfBuf_im = _mm_loadu_ps(&aec->wfBuf[1][pos + j]);
      const __m128 yf_re = _mm_loadu_ps(&yf[0][j]);
      const __m128 yf_im = _mm_loadu_ps(&yf[1][j]);
      const __m128 a = _mm_mul_ps(xfBuf_re, wfBuf_re);
      const __m128 b = _mm_mul_ps(xfBuf_im, wfBuf_im);
      const __m128 c = _mm_mul_ps(xfBuf_re, wfBuf_im);
      const __m128 d = _mm_mul_ps(xfBuf_im, wfBuf_re);
      const __m128 e = _mm_sub_ps(a, b);
      const __m128 f = _mm_add_ps(c, d);
      const __m128 g = _mm_add_ps(yf_re, e);
      const __m128 h = _mm_add_ps(yf_im, f);
      _mm_storeu_ps(&yf[0][j], g);
      _mm_storeu_ps(&yf[1][j], h);
    }
    // scalar code for the remaining items.
    for (; j < PART_LEN1; j++) {
      yf[0][j] += MulRe(aec->xfBuf[0][xPos + j],
                        aec->xfBuf[1][xPos + j],
                        aec->wfBuf[0][pos + j],
                        aec->wfBuf[1][pos + j]);
      yf[1][j] += MulIm(aec->xfBuf[0][xPos + j],
                        aec->xfBuf[1][xPos + j],
                        aec->wfBuf[0][pos + j],
                        aec->wfBuf[1][pos + j]);
    }
  }
}

static void ScaleErrorSignalSSE2(AecCore* aec, float ef[2][PART_LEN1]) {
  const __m128 k1e_10f = _mm_set1_ps(1e-10f);
  const __m128 kMu = aec->extended_filter_enabled ? _mm_set1_ps(kExtendedMu)
                                                  : _mm_set1_ps(aec->normal_mu);
  const __m128 kThresh = aec->extended_filter_enabled
                             ? _mm_set1_ps(kExtendedErrorThreshold)
                             : _mm_set1_ps(aec->normal_error_threshold);

  int i;
  // vectorized code (four at once)
  for (i = 0; i + 3 < PART_LEN1; i += 4) {
    const __m128 xPow = _mm_loadu_ps(&aec->xPow[i]);
    const __m128 ef_re_base = _mm_loadu_ps(&ef[0][i]);
    const __m128 ef_im_base = _mm_loadu_ps(&ef[1][i]);

    const __m128 xPowPlus = _mm_add_ps(xPow, k1e_10f);
    __m128 ef_re = _mm_div_ps(ef_re_base, xPowPlus);
    __m128 ef_im = _mm_div_ps(ef_im_base, xPowPlus);
    const __m128 ef_re2 = _mm_mul_ps(ef_re, ef_re);
    const __m128 ef_im2 = _mm_mul_ps(ef_im, ef_im);
    const __m128 ef_sum2 = _mm_add_ps(ef_re2, ef_im2);
    const __m128 absEf = _mm_sqrt_ps(ef_sum2);
    const __m128 bigger = _mm_cmpgt_ps(absEf, kThresh);
    __m128 absEfPlus = _mm_add_ps(absEf, k1e_10f);
    const __m128 absEfInv = _mm_div_ps(kThresh, absEfPlus);
    __m128 ef_re_if = _mm_mul_ps(ef_re, absEfInv);
    __m128 ef_im_if = _mm_mul_ps(ef_im, absEfInv);
    ef_re_if = _mm_and_ps(bigger, ef_re_if);
    ef_im_if = _mm_and_ps(bigger, ef_im_if);
    ef_re = _mm_andnot_ps(bigger, ef_re);
    ef_im = _mm_andnot_ps(bigger, ef_im);
    ef_re = _mm_or_ps(ef_re, ef_re_if);
    ef_im = _mm_or_ps(ef_im, ef_im_if);
    ef_re = _mm_mul_ps(ef_re, kMu);
    ef_im = _mm_mul_ps(ef_im, kMu);

    _mm_storeu_ps(&ef[0][i], ef_re);
    _mm_storeu_ps(&ef[1][i], ef_im);
  }
  // scalar code for the remaining items.
  {
    const float mu =
        aec->extended_filter_enabled ? kExtendedMu : aec->normal_mu;
    const float error_threshold = aec->extended_filter_enabled
                                      ? kExtendedErrorThreshold
                                      : aec->normal_error_threshold;
    for (; i < (PART_LEN1); i++) {
      float abs_ef;
      ef[0][i] /= (aec->xPow[i] + 1e-10f);
      ef[1][i] /= (aec->xPow[i] + 1e-10f);
      abs_ef = sqrtf(ef[0][i] * ef[0][i] + ef[1][i] * ef[1][i]);

      if (abs_ef > error_threshold) {
        abs_ef = error_threshold / (abs_ef + 1e-10f);
        ef[0][i] *= abs_ef;
        ef[1][i] *= abs_ef;
      }

      // Stepsize factor
      ef[0][i] *= mu;
      ef[1][i] *= mu;
    }
  }
}

static void FilterAdaptationSSE2(AecCore* aec,
                                 float* fft,
                                 float ef[2][PART_LEN1]) {
  int i, j;
  const int num_partitions = aec->num_partitions;
  for (i = 0; i < num_partitions; i++) {
    int xPos = (i + aec->xfBufBlockPos) * (PART_LEN1);
    int pos = i * PART_LEN1;
    // Check for wrap
    if (i + aec->xfBufBlockPos >= num_partitions) {
      xPos -= num_partitions * PART_LEN1;
    }

    // Process the whole array...
    for (j = 0; j < PART_LEN; j += 4) {
      // Load xfBuf and ef.
      const __m128 xfBuf_re = _mm_loadu_ps(&aec->xfBuf[0][xPos + j]);
      const __m128 xfBuf_im = _mm_loadu_ps(&aec->xfBuf[1][xPos + j]);
      const __m128 ef_re = _mm_loadu_ps(&ef[0][j]);
      const __m128 ef_im = _mm_loadu_ps(&ef[1][j]);
      // Calculate the product of conjugate(xfBuf) by ef.
      //   re(conjugate(a) * b) = aRe * bRe + aIm * bIm
      //   im(conjugate(a) * b)=  aRe * bIm - aIm * bRe
      const __m128 a = _mm_mul_ps(xfBuf_re, ef_re);
      const __m128 b = _mm_mul_ps(xfBuf_im, ef_im);
      const __m128 c = _mm_mul_ps(xfBuf_re, ef_im);
      const __m128 d = _mm_mul_ps(xfBuf_im, ef_re);
      const __m128 e = _mm_add_ps(a, b);
      const __m128 f = _mm_sub_ps(c, d);
      // Interleave real and imaginary parts.
      const __m128 g = _mm_unpacklo_ps(e, f);
      const __m128 h = _mm_unpackhi_ps(e, f);
      // Store
      _mm_storeu_ps(&fft[2 * j + 0], g);
      _mm_storeu_ps(&fft[2 * j + 4], h);
    }
    // ... and fixup the first imaginary entry.
    fft[1] = MulRe(aec->xfBuf[0][xPos + PART_LEN],
                   -aec->xfBuf[1][xPos + PART_LEN],
                   ef[0][PART_LEN],
                   ef[1][PART_LEN]);

    aec_rdft_inverse_128(fft);
    memset(fft + PART_LEN, 0, sizeof(float) * PART_LEN);

    // fft scaling
    {
      float scale = 2.0f / PART_LEN2;
      const __m128 scale_ps = _mm_load_ps1(&scale);
      for (j = 0; j < PART_LEN; j += 4) {
        const __m128 fft_ps = _mm_loadu_ps(&fft[j]);
        const __m128 fft_scale = _mm_mul_ps(fft_ps, scale_ps);
        _mm_storeu_ps(&fft[j], fft_scale);
      }
    }
    aec_rdft_forward_128(fft);

    {
      float wt1 = aec->wfBuf[1][pos];
      aec->wfBuf[0][pos + PART_LEN] += fft[1];
      for (j = 0; j < PART_LEN; j += 4) {
        __m128 wtBuf_re = _mm_loadu_ps(&aec->wfBuf[0][pos + j]);
        __m128 wtBuf_im = _mm_loadu_ps(&aec->wfBuf[1][pos + j]);
        const __m128 fft0 = _mm_loadu_ps(&fft[2 * j + 0]);
        const __m128 fft4 = _mm_loadu_ps(&fft[2 * j + 4]);
        const __m128 fft_re =
            _mm_shuffle_ps(fft0, fft4, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 fft_im =
            _mm_shuffle_ps(fft0, fft4, _MM_SHUFFLE(3, 1, 3, 1));
        wtBuf_re = _mm_add_ps(wtBuf_re, fft_re);
        wtBuf_im = _mm_add_ps(wtBuf_im, fft_im);
        _mm_storeu_ps(&aec->wfBuf[0][pos + j], wtBuf_re);
        _mm_storeu_ps(&aec->wfBuf[1][pos + j], wtBuf_im);
      }
      aec->wfBuf[1][pos] = wt1;
    }
  }
}

static __m128 mm_pow_ps(__m128 a, __m128 b) {
  // a^b = exp2(b * log2(a))
  //   exp2(x) and log2(x) are calculated using polynomial approximations.
  __m128 log2_a, b_log2_a, a_exp_b;

  // Calculate log2(x), x = a.
  {
    // To calculate log2(x), we decompose x like this:
    //   x = y * 2^n
    //     n is an integer
    //     y is in the [1.0, 2.0) range
    //
    //   log2(x) = log2(y) + n
    //     n       can be evaluated by playing with float representation.
    //     log2(y) in a small range can be approximated, this code uses an order
    //             five polynomial approximation. The coefficients have been
    //             estimated with the Remez algorithm and the resulting
    //             polynomial has a maximum relative error of 0.00086%.

    // Compute n.
    //    This is done by masking the exponent, shifting it into the top bit of
    //    the mantissa, putting eight into the biased exponent (to shift/
    //    compensate the fact that the exponent has been shifted in the top/
    //    fractional part and finally getting rid of the implicit leading one
    //    from the mantissa by substracting it out.
    static const ALIGN16_BEG int float_exponent_mask[4] ALIGN16_END = {
        0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000};
    static const ALIGN16_BEG int eight_biased_exponent[4] ALIGN16_END = {
        0x43800000, 0x43800000, 0x43800000, 0x43800000};
    static const ALIGN16_BEG int implicit_leading_one[4] ALIGN16_END = {
        0x43BF8000, 0x43BF8000, 0x43BF8000, 0x43BF8000};
    static const int shift_exponent_into_top_mantissa = 8;
    const __m128 two_n = _mm_and_ps(a, *((__m128*)float_exponent_mask));
    const __m128 n_1 = _mm_castsi128_ps(_mm_srli_epi32(
        _mm_castps_si128(two_n), shift_exponent_into_top_mantissa));
    const __m128 n_0 = _mm_or_ps(n_1, *((__m128*)eight_biased_exponent));
    const __m128 n = _mm_sub_ps(n_0, *((__m128*)implicit_leading_one));

    // Compute y.
    static const ALIGN16_BEG int mantissa_mask[4] ALIGN16_END = {
        0x007FFFFF, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF};
    static const ALIGN16_BEG int zero_biased_exponent_is_one[4] ALIGN16_END = {
        0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000};
    const __m128 mantissa = _mm_and_ps(a, *((__m128*)mantissa_mask));
    const __m128 y =
        _mm_or_ps(mantissa, *((__m128*)zero_biased_exponent_is_one));

    // Approximate log2(y) ~= (y - 1) * pol5(y).
    //    pol5(y) = C5 * y^5 + C4 * y^4 + C3 * y^3 + C2 * y^2 + C1 * y + C0
    static const ALIGN16_BEG float ALIGN16_END C5[4] = {
        -3.4436006e-2f, -3.4436006e-2f, -3.4436006e-2f, -3.4436006e-2f};
    static const ALIGN16_BEG float ALIGN16_END
        C4[4] = {3.1821337e-1f, 3.1821337e-1f, 3.1821337e-1f, 3.1821337e-1f};
    static const ALIGN16_BEG float ALIGN16_END
        C3[4] = {-1.2315303f, -1.2315303f, -1.2315303f, -1.2315303f};
    static const ALIGN16_BEG float ALIGN16_END
        C2[4] = {2.5988452f, 2.5988452f, 2.5988452f, 2.5988452f};
    static const ALIGN16_BEG float ALIGN16_END
        C1[4] = {-3.3241990f, -3.3241990f, -3.3241990f, -3.3241990f};
    static const ALIGN16_BEG float ALIGN16_END
        C0[4] = {3.1157899f, 3.1157899f, 3.1157899f, 3.1157899f};
    const __m128 pol5_y_0 = _mm_mul_ps(y, *((__m128*)C5));
    const __m128 pol5_y_1 = _mm_add_ps(pol5_y_0, *((__m128*)C4));
    const __m128 pol5_y_2 = _mm_mul_ps(pol5_y_1, y);
    const __m128 pol5_y_3 = _mm_add_ps(pol5_y_2, *((__m128*)C3));
    const __m128 pol5_y_4 = _mm_mul_ps(pol5_y_3, y);
    const __m128 pol5_y_5 = _mm_add_ps(pol5_y_4, *((__m128*)C2));
    const __m128 pol5_y_6 = _mm_mul_ps(pol5_y_5, y);
    const __m128 pol5_y_7 = _mm_add_ps(pol5_y_6, *((__m128*)C1));
    const __m128 pol5_y_8 = _mm_mul_ps(pol5_y_7, y);
    const __m128 pol5_y = _mm_add_ps(pol5_y_8, *((__m128*)C0));
    const __m128 y_minus_one =
        _mm_sub_ps(y, *((__m128*)zero_biased_exponent_is_one));
    const __m128 log2_y = _mm_mul_ps(y_minus_one, pol5_y);

    // Combine parts.
    log2_a = _mm_add_ps(n, log2_y);
  }

  // b * log2(a)
  b_log2_a = _mm_mul_ps(b, log2_a);

  // Calculate exp2(x), x = b * log2(a).
  {
    // To calculate 2^x, we decompose x like this:
    //   x = n + y
    //     n is an integer, the value of x - 0.5 rounded down, therefore
    //     y is in the [0.5, 1.5) range
    //
    //   2^x = 2^n * 2^y
    //     2^n can be evaluated by playing with float representation.
    //     2^y in a small range can be approximated, this code uses an order two
    //         polynomial approximation. The coefficients have been estimated
    //         with the Remez algorithm and the resulting polynomial has a
    //         maximum relative error of 0.17%.

    // To avoid over/underflow, we reduce the range of input to ]-127, 129].
    static const ALIGN16_BEG float max_input[4] ALIGN16_END = {129.f, 129.f,
                                                               129.f, 129.f};
    static const ALIGN16_BEG float min_input[4] ALIGN16_END = {
        -126.99999f, -126.99999f, -126.99999f, -126.99999f};
    const __m128 x_min = _mm_min_ps(b_log2_a, *((__m128*)max_input));
    const __m128 x_max = _mm_max_ps(x_min, *((__m128*)min_input));
    // Compute n.
    static const ALIGN16_BEG float half[4] ALIGN16_END = {0.5f, 0.5f,
                                                          0.5f, 0.5f};
    const __m128 x_minus_half = _mm_sub_ps(x_max, *((__m128*)half));
    const __m128i x_minus_half_floor = _mm_cvtps_epi32(x_minus_half);
    // Compute 2^n.
    static const ALIGN16_BEG int float_exponent_bias[4] ALIGN16_END = {
        127, 127, 127, 127};
    static const int float_exponent_shift = 23;
    const __m128i two_n_exponent =
        _mm_add_epi32(x_minus_half_floor, *((__m128i*)float_exponent_bias));
    const __m128 two_n =
        _mm_castsi128_ps(_mm_slli_epi32(two_n_exponent, float_exponent_shift));
    // Compute y.
    const __m128 y = _mm_sub_ps(x_max, _mm_cvtepi32_ps(x_minus_half_floor));
    // Approximate 2^y ~= C2 * y^2 + C1 * y + C0.
    static const ALIGN16_BEG float C2[4] ALIGN16_END = {
        3.3718944e-1f, 3.3718944e-1f, 3.3718944e-1f, 3.3718944e-1f};
    static const ALIGN16_BEG float C1[4] ALIGN16_END = {
        6.5763628e-1f, 6.5763628e-1f, 6.5763628e-1f, 6.5763628e-1f};
    static const ALIGN16_BEG float C0[4] ALIGN16_END = {1.0017247f, 1.0017247f,
                                                        1.0017247f, 1.0017247f};
    const __m128 exp2_y_0 = _mm_mul_ps(y, *((__m128*)C2));
    const __m128 exp2_y_1 = _mm_add_ps(exp2_y_0, *((__m128*)C1));
    const __m128 exp2_y_2 = _mm_mul_ps(exp2_y_1, y);
    const __m128 exp2_y = _mm_add_ps(exp2_y_2, *((__m128*)C0));

    // Combine parts.
    a_exp_b = _mm_mul_ps(exp2_y, two_n);
  }
  return a_exp_b;
}

static void OverdriveAndSuppressSSE2(AecCore* aec,
                                     float hNl[PART_LEN1],
                                     const float hNlFb,
                                     float efw[2][PART_LEN1]) {
  int i;
  const __m128 vec_hNlFb = _mm_set1_ps(hNlFb);
  const __m128 vec_one = _mm_set1_ps(1.0f);
  const __m128 vec_minus_one = _mm_set1_ps(-1.0f);
  const __m128 vec_overDriveSm = _mm_set1_ps(aec->overDriveSm);
  // vectorized code (four at once)
  for (i = 0; i + 3 < PART_LEN1; i += 4) {
    // Weight subbands
    __m128 vec_hNl = _mm_loadu_ps(&hNl[i]);
    const __m128 vec_weightCurve = _mm_loadu_ps(&WebRtcAec_weightCurve[i]);
    const __m128 bigger = _mm_cmpgt_ps(vec_hNl, vec_hNlFb);
    const __m128 vec_weightCurve_hNlFb = _mm_mul_ps(vec_weightCurve, vec_hNlFb);
    const __m128 vec_one_weightCurve = _mm_sub_ps(vec_one, vec_weightCurve);
    const __m128 vec_one_weightCurve_hNl =
        _mm_mul_ps(vec_one_weightCurve, vec_hNl);
    const __m128 vec_if0 = _mm_andnot_ps(bigger, vec_hNl);
    const __m128 vec_if1 = _mm_and_ps(
        bigger, _mm_add_ps(vec_weightCurve_hNlFb, vec_one_weightCurve_hNl));
    vec_hNl = _mm_or_ps(vec_if0, vec_if1);

    {
      const __m128 vec_overDriveCurve =
          _mm_loadu_ps(&WebRtcAec_overDriveCurve[i]);
      const __m128 vec_overDriveSm_overDriveCurve =
          _mm_mul_ps(vec_overDriveSm, vec_overDriveCurve);
      vec_hNl = mm_pow_ps(vec_hNl, vec_overDriveSm_overDriveCurve);
      _mm_storeu_ps(&hNl[i], vec_hNl);
    }

    // Suppress error signal
    {
      __m128 vec_efw_re = _mm_loadu_ps(&efw[0][i]);
      __m128 vec_efw_im = _mm_loadu_ps(&efw[1][i]);
      vec_efw_re = _mm_mul_ps(vec_efw_re, vec_hNl);
      vec_efw_im = _mm_mul_ps(vec_efw_im, vec_hNl);

      // Ooura fft returns incorrect sign on imaginary component. It matters
      // here because we are making an additive change with comfort noise.
      vec_efw_im = _mm_mul_ps(vec_efw_im, vec_minus_one);
      _mm_storeu_ps(&efw[0][i], vec_efw_re);
      _mm_storeu_ps(&efw[1][i], vec_efw_im);
    }
  }
  // scalar code for the remaining items.
  for (; i < PART_LEN1; i++) {
    // Weight subbands
    if (hNl[i] > hNlFb) {
      hNl[i] = WebRtcAec_weightCurve[i] * hNlFb +
               (1 - WebRtcAec_weightCurve[i]) * hNl[i];
    }
    hNl[i] = powf(hNl[i], aec->overDriveSm * WebRtcAec_overDriveCurve[i]);

    // Suppress error signal
    efw[0][i] *= hNl[i];
    efw[1][i] *= hNl[i];

    // Ooura fft returns incorrect sign on imaginary component. It matters
    // here because we are making an additive change with comfort noise.
    efw[1][i] *= -1;
  }
}

__inline static void _mm_add_ps_4x1(__m128 sum, float *dst) {
  // A+B C+D
  sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2)));
  // A+B+C+D A+B+C+D
  sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1)));
  _mm_store_ss(dst, sum);
}
static int PartitionDelay(const AecCore* aec) {
  // Measures the energy in each filter partition and returns the partition with
  // highest energy.
  // TODO(bjornv): Spread computational cost by computing one partition per
  // block?
  float wfEnMax = 0;
  int i;
  int delay = 0;

  for (i = 0; i < aec->num_partitions; i++) {
    int j;
    int pos = i * PART_LEN1;
    float wfEn = 0;
    __m128 vec_wfEn = _mm_set1_ps(0.0f);
    // vectorized code (four at once)
    for (j = 0; j + 3 < PART_LEN1; j += 4) {
      const __m128 vec_wfBuf0 = _mm_loadu_ps(&aec->wfBuf[0][pos + j]);
      const __m128 vec_wfBuf1 = _mm_loadu_ps(&aec->wfBuf[1][pos + j]);
      vec_wfEn = _mm_add_ps(vec_wfEn, _mm_mul_ps(vec_wfBuf0, vec_wfBuf0));
      vec_wfEn = _mm_add_ps(vec_wfEn, _mm_mul_ps(vec_wfBuf1, vec_wfBuf1));
    }
    _mm_add_ps_4x1(vec_wfEn, &wfEn);

    // scalar code for the remaining items.
    for (; j < PART_LEN1; j++) {
      wfEn += aec->wfBuf[0][pos + j] * aec->wfBuf[0][pos + j] +
              aec->wfBuf[1][pos + j] * aec->wfBuf[1][pos + j];
    }

    if (wfEn > wfEnMax) {
      wfEnMax = wfEn;
      delay = i;
    }
  }
  return delay;
}

// Updates the following smoothed  Power Spectral Densities (PSD):
//  - sd  : near-end
//  - se  : residual echo
//  - sx  : far-end
//  - sde : cross-PSD of near-end and residual echo
//  - sxd : cross-PSD of near-end and far-end
//
// In addition to updating the PSDs, also the filter diverge state is determined
// upon actions are taken.
static void SmoothedPSD(AecCore* aec,
                        float efw[2][PART_LEN1],
                        float dfw[2][PART_LEN1],
                        float xfw[2][PART_LEN1]) {
  // Power estimate smoothing coefficients.
  const float* ptrGCoh = aec->extended_filter_enabled
      ? WebRtcAec_kExtendedSmoothingCoefficients[aec->mult - 1]
      : WebRtcAec_kNormalSmoothingCoefficients[aec->mult - 1];
  int i;
  float sdSum = 0, seSum = 0;
  const __m128 vec_15 =  _mm_set1_ps(WebRtcAec_kMinFarendPSD);
  const __m128 vec_GCoh0 = _mm_set1_ps(ptrGCoh[0]);
  const __m128 vec_GCoh1 = _mm_set1_ps(ptrGCoh[1]);
  __m128 vec_sdSum = _mm_set1_ps(0.0f);
  __m128 vec_seSum = _mm_set1_ps(0.0f);

  for (i = 0; i + 3 < PART_LEN1; i += 4) {
    const __m128 vec_dfw0 = _mm_loadu_ps(&dfw[0][i]);
    const __m128 vec_dfw1 = _mm_loadu_ps(&dfw[1][i]);
    const __m128 vec_efw0 = _mm_loadu_ps(&efw[0][i]);
    const __m128 vec_efw1 = _mm_loadu_ps(&efw[1][i]);
    const __m128 vec_xfw0 = _mm_loadu_ps(&xfw[0][i]);
    const __m128 vec_xfw1 = _mm_loadu_ps(&xfw[1][i]);
    __m128 vec_sd = _mm_mul_ps(_mm_loadu_ps(&aec->sd[i]), vec_GCoh0);
    __m128 vec_se = _mm_mul_ps(_mm_loadu_ps(&aec->se[i]), vec_GCoh0);
    __m128 vec_sx = _mm_mul_ps(_mm_loadu_ps(&aec->sx[i]), vec_GCoh0);
    __m128 vec_dfw_sumsq = _mm_mul_ps(vec_dfw0, vec_dfw0);
    __m128 vec_efw_sumsq = _mm_mul_ps(vec_efw0, vec_efw0);
    __m128 vec_xfw_sumsq = _mm_mul_ps(vec_xfw0, vec_xfw0);
    vec_dfw_sumsq = _mm_add_ps(vec_dfw_sumsq, _mm_mul_ps(vec_dfw1, vec_dfw1));
    vec_efw_sumsq = _mm_add_ps(vec_efw_sumsq, _mm_mul_ps(vec_efw1, vec_efw1));
    vec_xfw_sumsq = _mm_add_ps(vec_xfw_sumsq, _mm_mul_ps(vec_xfw1, vec_xfw1));
    vec_xfw_sumsq = _mm_max_ps(vec_xfw_sumsq, vec_15);
    vec_sd = _mm_add_ps(vec_sd, _mm_mul_ps(vec_dfw_sumsq, vec_GCoh1));
    vec_se = _mm_add_ps(vec_se, _mm_mul_ps(vec_efw_sumsq, vec_GCoh1));
    vec_sx = _mm_add_ps(vec_sx, _mm_mul_ps(vec_xfw_sumsq, vec_GCoh1));
    _mm_storeu_ps(&aec->sd[i], vec_sd);
    _mm_storeu_ps(&aec->se[i], vec_se);
    _mm_storeu_ps(&aec->sx[i], vec_sx);

    {
      const __m128 vec_3210 = _mm_loadu_ps(&aec->sde[i][0]);
      const __m128 vec_7654 = _mm_loadu_ps(&aec->sde[i + 2][0]);
      __m128 vec_a = _mm_shuffle_ps(vec_3210, vec_7654,
                                    _MM_SHUFFLE(2, 0, 2, 0));
      __m128 vec_b = _mm_shuffle_ps(vec_3210, vec_7654,
                                    _MM_SHUFFLE(3, 1, 3, 1));
      __m128 vec_dfwefw0011 = _mm_mul_ps(vec_dfw0, vec_efw0);
      __m128 vec_dfwefw0110 = _mm_mul_ps(vec_dfw0, vec_efw1);
      vec_a = _mm_mul_ps(vec_a, vec_GCoh0);
      vec_b = _mm_mul_ps(vec_b, vec_GCoh0);
      vec_dfwefw0011 = _mm_add_ps(vec_dfwefw0011,
                                  _mm_mul_ps(vec_dfw1, vec_efw1));
      vec_dfwefw0110 = _mm_sub_ps(vec_dfwefw0110,
                                  _mm_mul_ps(vec_dfw1, vec_efw0));
      vec_a = _mm_add_ps(vec_a, _mm_mul_ps(vec_dfwefw0011, vec_GCoh1));
      vec_b = _mm_add_ps(vec_b, _mm_mul_ps(vec_dfwefw0110, vec_GCoh1));
      _mm_storeu_ps(&aec->sde[i][0], _mm_unpacklo_ps(vec_a, vec_b));
      _mm_storeu_ps(&aec->sde[i + 2][0], _mm_unpackhi_ps(vec_a, vec_b));
    }

    {
      const __m128 vec_3210 = _mm_loadu_ps(&aec->sxd[i][0]);
      const __m128 vec_7654 = _mm_loadu_ps(&aec->sxd[i + 2][0]);
      __m128 vec_a = _mm_shuffle_ps(vec_3210, vec_7654,
                                    _MM_SHUFFLE(2, 0, 2, 0));
      __m128 vec_b = _mm_shuffle_ps(vec_3210, vec_7654,
                                    _MM_SHUFFLE(3, 1, 3, 1));
      __m128 vec_dfwxfw0011 = _mm_mul_ps(vec_dfw0, vec_xfw0);
      __m128 vec_dfwxfw0110 = _mm_mul_ps(vec_dfw0, vec_xfw1);
      vec_a = _mm_mul_ps(vec_a, vec_GCoh0);
      vec_b = _mm_mul_ps(vec_b, vec_GCoh0);
      vec_dfwxfw0011 = _mm_add_ps(vec_dfwxfw0011,
                                  _mm_mul_ps(vec_dfw1, vec_xfw1));
      vec_dfwxfw0110 = _mm_sub_ps(vec_dfwxfw0110,
                                  _mm_mul_ps(vec_dfw1, vec_xfw0));
      vec_a = _mm_add_ps(vec_a, _mm_mul_ps(vec_dfwxfw0011, vec_GCoh1));
      vec_b = _mm_add_ps(vec_b, _mm_mul_ps(vec_dfwxfw0110, vec_GCoh1));
      _mm_storeu_ps(&aec->sxd[i][0], _mm_unpacklo_ps(vec_a, vec_b));
      _mm_storeu_ps(&aec->sxd[i + 2][0], _mm_unpackhi_ps(vec_a, vec_b));
    }

    vec_sdSum = _mm_add_ps(vec_sdSum, vec_sd);
    vec_seSum = _mm_add_ps(vec_seSum, vec_se);
  }

  _mm_add_ps_4x1(vec_sdSum, &sdSum);
  _mm_add_ps_4x1(vec_seSum, &seSum);

  for (; i < PART_LEN1; i++) {
    aec->sd[i] = ptrGCoh[0] * aec->sd[i] +
                 ptrGCoh[1] * (dfw[0][i] * dfw[0][i] + dfw[1][i] * dfw[1][i]);
    aec->se[i] = ptrGCoh[0] * aec->se[i] +
                 ptrGCoh[1] * (efw[0][i] * efw[0][i] + efw[1][i] * efw[1][i]);
    // We threshold here to protect against the ill-effects of a zero farend.
    // The threshold is not arbitrarily chosen, but balances protection and
    // adverse interaction with the algorithm's tuning.
    // TODO(bjornv): investigate further why this is so sensitive.
    aec->sx[i] =
        ptrGCoh[0] * aec->sx[i] +
        ptrGCoh[1] * WEBRTC_SPL_MAX(
            xfw[0][i] * xfw[0][i] + xfw[1][i] * xfw[1][i],
            WebRtcAec_kMinFarendPSD);

    aec->sde[i][0] =
        ptrGCoh[0] * aec->sde[i][0] +
        ptrGCoh[1] * (dfw[0][i] * efw[0][i] + dfw[1][i] * efw[1][i]);
    aec->sde[i][1] =
        ptrGCoh[0] * aec->sde[i][1] +
        ptrGCoh[1] * (dfw[0][i] * efw[1][i] - dfw[1][i] * efw[0][i]);

    aec->sxd[i][0] =
        ptrGCoh[0] * aec->sxd[i][0] +
        ptrGCoh[1] * (dfw[0][i] * xfw[0][i] + dfw[1][i] * xfw[1][i]);
    aec->sxd[i][1] =
        ptrGCoh[0] * aec->sxd[i][1] +
        ptrGCoh[1] * (dfw[0][i] * xfw[1][i] - dfw[1][i] * xfw[0][i]);

    sdSum += aec->sd[i];
    seSum += aec->se[i];
  }

  // Divergent filter safeguard.
  aec->divergeState = (aec->divergeState ? 1.05f : 1.0f) * seSum > sdSum;

  if (aec->divergeState)
    memcpy(efw, dfw, sizeof(efw[0][0]) * 2 * PART_LEN1);

  // Reset if error is significantly larger than nearend (13 dB).
  if (!aec->extended_filter_enabled && seSum > (19.95f * sdSum))
    memset(aec->wfBuf, 0, sizeof(aec->wfBuf));
}

// Window time domain data to be used by the fft.
__inline static void WindowData(float* x_windowed, const float* x) {
  int i;
  for (i = 0; i < PART_LEN; i += 4) {
    const __m128 vec_Buf1 = _mm_loadu_ps(&x[i]);
    const __m128 vec_Buf2 = _mm_loadu_ps(&x[PART_LEN + i]);
    const __m128 vec_sqrtHanning = _mm_load_ps(&WebRtcAec_sqrtHanning[i]);
    // A B C D
    __m128 vec_sqrtHanning_rev =
        _mm_loadu_ps(&WebRtcAec_sqrtHanning[PART_LEN - i - 3]);
    // D C B A
    vec_sqrtHanning_rev =
        _mm_shuffle_ps(vec_sqrtHanning_rev, vec_sqrtHanning_rev,
                       _MM_SHUFFLE(0, 1, 2, 3));
    _mm_storeu_ps(&x_windowed[i], _mm_mul_ps(vec_Buf1, vec_sqrtHanning));
    _mm_storeu_ps(&x_windowed[PART_LEN + i],
                  _mm_mul_ps(vec_Buf2, vec_sqrtHanning_rev));
  }
}

// Puts fft output data into a complex valued array.
__inline static void StoreAsComplex(const float* data,
                                    float data_complex[2][PART_LEN1]) {
  int i;
  for (i = 0; i < PART_LEN; i += 4) {
    const __m128 vec_fft0 = _mm_loadu_ps(&data[2 * i]);
    const __m128 vec_fft4 = _mm_loadu_ps(&data[2 * i + 4]);
    const __m128 vec_a = _mm_shuffle_ps(vec_fft0, vec_fft4,
                                        _MM_SHUFFLE(2, 0, 2, 0));
    const __m128 vec_b = _mm_shuffle_ps(vec_fft0, vec_fft4,
                                        _MM_SHUFFLE(3, 1, 3, 1));
    _mm_storeu_ps(&data_complex[0][i], vec_a);
    _mm_storeu_ps(&data_complex[1][i], vec_b);
  }
  // fix beginning/end values
  data_complex[1][0] = 0;
  data_complex[1][PART_LEN] = 0;
  data_complex[0][0] = data[0];
  data_complex[0][PART_LEN] = data[1];
}

static void SubbandCoherenceSSE2(AecCore* aec,
                                 float efw[2][PART_LEN1],
                                 float xfw[2][PART_LEN1],
                                 float* fft,
                                 float* cohde,
                                 float* cohxd) {
  float dfw[2][PART_LEN1];
  int i;

  if (aec->delayEstCtr == 0)
    aec->delayIdx = PartitionDelay(aec);

  // Use delayed far.
  memcpy(xfw,
         aec->xfwBuf + aec->delayIdx * PART_LEN1,
         sizeof(xfw[0][0]) * 2 * PART_LEN1);

  // Windowed near fft
  WindowData(fft, aec->dBuf);
  aec_rdft_forward_128(fft);
  StoreAsComplex(fft, dfw);

  // Windowed error fft
  WindowData(fft, aec->eBuf);
  aec_rdft_forward_128(fft);
  StoreAsComplex(fft, efw);

  SmoothedPSD(aec, efw, dfw, xfw);

  {
    const __m128 vec_1eminus10 =  _mm_set1_ps(1e-10f);

    // Subband coherence
    for (i = 0; i + 3 < PART_LEN1; i += 4) {
      const __m128 vec_sd = _mm_loadu_ps(&aec->sd[i]);
      const __m128 vec_se = _mm_loadu_ps(&aec->se[i]);
      const __m128 vec_sx = _mm_loadu_ps(&aec->sx[i]);
      const __m128 vec_sdse = _mm_add_ps(vec_1eminus10,
                                         _mm_mul_ps(vec_sd, vec_se));
      const __m128 vec_sdsx = _mm_add_ps(vec_1eminus10,
                                         _mm_mul_ps(vec_sd, vec_sx));
      const __m128 vec_sde_3210 = _mm_loadu_ps(&aec->sde[i][0]);
      const __m128 vec_sde_7654 = _mm_loadu_ps(&aec->sde[i + 2][0]);
      const __m128 vec_sxd_3210 = _mm_loadu_ps(&aec->sxd[i][0]);
      const __m128 vec_sxd_7654 = _mm_loadu_ps(&aec->sxd[i + 2][0]);
      const __m128 vec_sde_0 = _mm_shuffle_ps(vec_sde_3210, vec_sde_7654,
                                              _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vec_sde_1 = _mm_shuffle_ps(vec_sde_3210, vec_sde_7654,
                                              _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vec_sxd_0 = _mm_shuffle_ps(vec_sxd_3210, vec_sxd_7654,
                                              _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vec_sxd_1 = _mm_shuffle_ps(vec_sxd_3210, vec_sxd_7654,
                                              _MM_SHUFFLE(3, 1, 3, 1));
      __m128 vec_cohde = _mm_mul_ps(vec_sde_0, vec_sde_0);
      __m128 vec_cohxd = _mm_mul_ps(vec_sxd_0, vec_sxd_0);
      vec_cohde = _mm_add_ps(vec_cohde, _mm_mul_ps(vec_sde_1, vec_sde_1));
      vec_cohde = _mm_div_ps(vec_cohde, vec_sdse);
      vec_cohxd = _mm_add_ps(vec_cohxd, _mm_mul_ps(vec_sxd_1, vec_sxd_1));
      vec_cohxd = _mm_div_ps(vec_cohxd, vec_sdsx);
      _mm_storeu_ps(&cohde[i], vec_cohde);
      _mm_storeu_ps(&cohxd[i], vec_cohxd);
    }

    // scalar code for the remaining items.
    for (; i < PART_LEN1; i++) {
      cohde[i] =
          (aec->sde[i][0] * aec->sde[i][0] + aec->sde[i][1] * aec->sde[i][1]) /
          (aec->sd[i] * aec->se[i] + 1e-10f);
      cohxd[i] =
          (aec->sxd[i][0] * aec->sxd[i][0] + aec->sxd[i][1] * aec->sxd[i][1]) /
          (aec->sx[i] * aec->sd[i] + 1e-10f);
    }
  }
}

void WebRtcAec_InitAec_SSE2(void) {
  WebRtcAec_FilterFar = FilterFarSSE2;
  WebRtcAec_ScaleErrorSignal = ScaleErrorSignalSSE2;
  WebRtcAec_FilterAdaptation = FilterAdaptationSSE2;
  WebRtcAec_OverdriveAndSuppress = OverdriveAndSuppressSSE2;
  WebRtcAec_SubbandCoherence = SubbandCoherenceSSE2;
}

#endif
