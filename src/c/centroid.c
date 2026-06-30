/*
 * centroid.c — SH-WFS Spot Centroiding Implementation
 * =====================================================
 * See centroid.h for full documentation.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -ffast-math -shared -fPIC \
 *       -o libcentroid.so centroid.c -lm
 *
 * For non-OpenMP systems:
 *   gcc -O3 -march=native -ffast-math -shared -fPIC \
 *       -o libcentroid.so centroid.c -lm
 */

#include "centroid.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

/* ─── Internal helpers ──────────────────────────────────── */

/**
 * row_ptr: returns pointer to the start of pixel (x_start, row) in the image.
 * The image is row-major: pixel(col, row) = image[row * img_width + col].
 */
static inline const float*
row_ptr(const float* image, int img_width, int row, int x_start)
{
    return image + row * img_width + x_start;
}

/* ─── Background estimation ─────────────────────────────── */

void estimate_background(const float*   image,
                         int             img_width,
                         const SubAperture* sa,
                         int             border_px,
                         float*          mean_out,
                         float*          std_out)
{
    if (border_px <= 0) {
        *mean_out = 0.0f;
        *std_out  = 1.0f;
        return;
    }

    double sum   = 0.0;
    double sum_sq= 0.0;
    int    count = 0;

    int x0 = sa->x_start;
    int y0 = sa->y_start;
    int w  = sa->width;
    int h  = sa->height;
    int bp = border_px;

    /* Iterate over border ring: top, bottom, left, right strips */
    for (int row = y0; row < y0 + h; row++) {
        const float* rp = row_ptr(image, img_width, row, x0);
        int in_top    = (row < y0 + bp);
        int in_bottom = (row >= y0 + h - bp);
        for (int col = 0; col < w; col++) {
            int in_left  = (col < bp);
            int in_right = (col >= w - bp);
            if (in_top || in_bottom || in_left || in_right) {
                double v = (double)rp[col];
                sum    += v;
                sum_sq += v * v;
                count++;
            }
        }
    }

    if (count < 2) {
        *mean_out = 0.0f;
        *std_out  = 1.0f;
        return;
    }

    double mean = sum / count;
    double var  = sum_sq / count - mean * mean;
    if (var < 0.0) var = 0.0;

    *mean_out = (float)mean;
    *std_out  = (float)sqrt(var);
}

/* ─── Center of Gravity ─────────────────────────────────── */

void centroid_cog(const float*   image,
                  int             img_width,
                  const SubAperture* sa,
                  const CentroidConfig* cfg,
                  Centroid*       result)
{
    if (!sa->valid) {
        result->valid = 0;
        return;
    }

    double sum_I  = 0.0;
    double sum_Ix = 0.0;
    double sum_Iy = 0.0;

    int x0 = sa->x_start;
    int y0 = sa->y_start;
    int w  = sa->width;
    int h  = sa->height;

    for (int row = y0; row < y0 + h; row++) {
        const float* rp = row_ptr(image, img_width, row, x0);
        double fy = (double)row;
        for (int col = 0; col < w; col++) {
            double val = (double)rp[col];
            double fx  = (double)(x0 + col);
            sum_I  += val;
            sum_Ix += val * fx;
            sum_Iy += val * fy;
        }
    }

    result->total_flux = sum_I;
    if (sum_I > (double)cfg->min_flux) {
        result->cx    = sum_Ix / sum_I;
        result->cy    = sum_Iy / sum_I;
        result->valid = 1;
    } else {
        /* Fall back to geometric centre */
        result->cx    = x0 + 0.5 * (w - 1);
        result->cy    = y0 + 0.5 * (h - 1);
        result->valid = 0;
    }
}

/* ─── Thresholded Center of Gravity ─────────────────────── */

void centroid_threshold_cog(const float*   image,
                             int             img_width,
                             const SubAperture* sa,
                             const CentroidConfig* cfg,
                             Centroid*       result)
{
    if (!sa->valid) {
        result->valid = 0;
        return;
    }

    /* Estimate background from border pixels */
    float bg_mean = 0.0f, bg_std = 1.0f;
    estimate_background(image, img_width, sa, cfg->border_px,
                        &bg_mean, &bg_std);

    /* Threshold: bg_mean + sigma * bg_std, or use cfg->threshold directly
     * if border_px == 0 (caller provides absolute threshold). */
    float threshold;
    if (cfg->border_px > 0) {
        threshold = bg_mean + (float)cfg->threshold * bg_std;
    } else {
        threshold = cfg->threshold;
    }

    double sum_I  = 0.0;
    double sum_Ix = 0.0;
    double sum_Iy = 0.0;

    int x0 = sa->x_start;
    int y0 = sa->y_start;
    int w  = sa->width;
    int h  = sa->height;

    for (int row = y0; row < y0 + h; row++) {
        const float* rp = row_ptr(image, img_width, row, x0);
        double fy = (double)row;
        for (int col = 0; col < w; col++) {
            /* Subtract threshold and clamp to zero */
            double val = (double)rp[col] - (double)threshold;
            if (val < 0.0) val = 0.0;

            double fx = (double)(x0 + col);
            sum_I  += val;
            sum_Ix += val * fx;
            sum_Iy += val * fy;
        }
    }

    result->total_flux = sum_I;
    if (sum_I > (double)cfg->min_flux) {
        result->cx    = sum_Ix / sum_I;
        result->cy    = sum_Iy / sum_I;
        result->valid = 1;
    } else {
        result->cx    = x0 + 0.5 * (w - 1);
        result->cy    = y0 + 0.5 * (h - 1);
        result->valid = 0;
    }
}

/* ─── Windowed Center of Gravity ────────────────────────── */

void centroid_windowed_cog(const float*   image,
                            int             img_width,
                            const SubAperture* sa,
                            const CentroidConfig* cfg,
                            Centroid*       result)
{
    if (!sa->valid) {
        result->valid = 0;
        return;
    }

    /* Pass 1: thresholded CoG to get initial estimate */
    Centroid initial;
    centroid_threshold_cog(image, img_width, sa, cfg, &initial);

    /* If pass-1 failed, return the failure */
    if (!initial.valid) {
        *result = initial;
        return;
    }

    double x0_est = initial.cx;
    double y0_est = initial.cy;
    double sigma  = cfg->window_sigma;
    double inv2s2 = 0.5 / (sigma * sigma);

    /* Pass 2: Gaussian-windowed CoG */
    float bg_mean = 0.0f, bg_std = 1.0f;
    estimate_background(image, img_width, sa, cfg->border_px,
                        &bg_mean, &bg_std);
    float threshold = bg_mean + (float)cfg->threshold * bg_std;

    double sum_wI  = 0.0;
    double sum_wIx = 0.0;
    double sum_wIy = 0.0;

    int x0 = sa->x_start;
    int y0 = sa->y_start;
    int w  = sa->width;
    int h  = sa->height;

    for (int row = y0; row < y0 + h; row++) {
        const float* rp = row_ptr(image, img_width, row, x0);
        double fy  = (double)row;
        double dy  = fy - y0_est;
        double dy2 = dy * dy;

        for (int col = 0; col < w; col++) {
            double fx  = (double)(x0 + col);
            double dx  = fx - x0_est;
            double r2  = dx * dx + dy2;

            double weight = exp(-r2 * inv2s2);
            double val    = (double)rp[col] - (double)threshold;
            if (val < 0.0) val = 0.0;

            double wval = weight * val;
            sum_wI  += wval;
            sum_wIx += wval * fx;
            sum_wIy += wval * fy;
        }
    }

    result->total_flux = sum_wI;
    if (sum_wI > (double)cfg->min_flux) {
        result->cx    = sum_wIx / sum_wI;
        result->cy    = sum_wIy / sum_wI;
        result->valid = 1;
    } else {
        /* Fall back to pass-1 result */
        *result = initial;
    }
}

/* ─── Batch centroiding ─────────────────────────────────── */

void batch_centroid(const float*   image,
                    int             img_width,
                    int             img_height,
                    const SubAperture* sas,
                    int             n_sa,
                    const CentroidConfig* cfg,
                    Centroid*       results)
{
    (void)img_height; /* unused but part of the API for bounds-checking extensions */

    /* Dispatch to the right per-SA function */
    typedef void (*centroid_fn)(const float*, int, const SubAperture*,
                                const CentroidConfig*, Centroid*);

    centroid_fn fn;
    switch (cfg->method) {
        case CENTROID_COG:  fn = centroid_cog;            break;
        case CENTROID_TCOG: fn = centroid_threshold_cog;  break;
        case CENTROID_WCOG: fn = centroid_windowed_cog;   break;
        default:            fn = centroid_threshold_cog;  break;
    }

#ifdef _OPENMP
    if (cfg->use_openmp) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_sa; i++) {
            fn(image, img_width, &sas[i], cfg, &results[i]);
        }
        return;
    }
#endif

    /* Serial fallback */
    for (int i = 0; i < n_sa; i++) {
        fn(image, img_width, &sas[i], cfg, &results[i]);
    }
}

/* ─── Sub-aperture grid builder ─────────────────────────── */

int build_subaperture_grid(int    n_sa_x,
                            int    n_sa_y,
                            int    pix_per_sa,
                            double pupil_cx,
                            double pupil_cy,
                            double pupil_radius,
                            SubAperture* sas)
{
    int n_valid = 0;
    int idx = 0;

    for (int j = 0; j < n_sa_y; j++) {
        for (int i = 0; i < n_sa_x; i++) {
            SubAperture* sa = &sas[idx++];

            sa->x_start = i * pix_per_sa;
            sa->y_start = j * pix_per_sa;
            sa->width   = pix_per_sa;
            sa->height  = pix_per_sa;

            /* Centre of this sub-aperture in full-frame pixels */
            double cx = sa->x_start + 0.5 * (pix_per_sa - 1);
            double cy = sa->y_start + 0.5 * (pix_per_sa - 1);

            double dx = cx - pupil_cx;
            double dy = cy - pupil_cy;
            double r  = sqrt(dx*dx + dy*dy);

            if (r <= pupil_radius) {
                sa->valid = 1;
                n_valid++;
            } else {
                sa->valid = 0;
            }
        }
    }

    return n_valid;
}
