/*
 * centroid.h — SH-WFS Spot Centroiding (C Implementation)
 * =========================================================
 * Fast centroiding algorithms for Shack-Hartmann wavefront sensor
 * sub-aperture spots.  Designed to be the inner-loop of a real-time
 * AO control system: no heap allocation, branch-free inner loop,
 * OpenMP-parallelisable across sub-apertures.
 *
 * Compilation:
 *   gcc -O3 -march=native -fopenmp -ffast-math -shared -fPIC \
 *       -o libcentroid.so centroid.c
 *
 * Usage from Python via ctypes — see centroiding.py
 */

#ifndef CENTROID_H
#define CENTROID_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Data structures ──────────────────────────────────────── */

/**
 * SubAperture: pixel window on the detector corresponding to one lenslet.
 * All values are in pixel units (0-indexed from top-left of detector).
 */
typedef struct {
    int x_start;    /* leftmost pixel column (inclusive) */
    int y_start;    /* topmost pixel row    (inclusive)  */
    int width;      /* sub-aperture width  [pixels]      */
    int height;     /* sub-aperture height [pixels]      */
    int valid;      /* 1 = inside pupil mask, 0 = ignore */
} SubAperture;

/**
 * Centroid: result for one sub-aperture.
 * Coordinates are in full-frame pixel space (not sub-aperture-local).
 */
typedef struct {
    double cx;          /* x centroid [pixels, full-frame] */
    double cy;          /* y centroid [pixels, full-frame] */
    double total_flux;  /* sum of thresholded/weighted intensities */
    int    valid;       /* 1 = centroid computed, 0 = insufficient flux */
} Centroid;

/**
 * CentroidConfig: parameters controlling algorithm behaviour.
 */
typedef struct {
    int    method;              /* 0=CoG | 1=T-CoG | 2=W-CoG */
    float  threshold;           /* absolute threshold for T-CoG (ADU) */
    float  min_flux;            /* minimum total flux to declare valid */
    double window_sigma;        /* Gaussian sigma for W-CoG [pixels]  */
    int    border_px;           /* border width for background estimate */
    int    use_openmp;          /* 1 = parallelise via OpenMP          */
} CentroidConfig;

/* Centroid method identifiers */
#define CENTROID_COG     0
#define CENTROID_TCOG    1
#define CENTROID_WCOG    2

/* ── Per-sub-aperture functions ───────────────────────────── */

/**
 * centroid_cog:
 *   Pure centre-of-gravity.  Simple, fast, background-biased.
 *
 * @param image      Pointer to flat float32 image buffer (row-major).
 * @param img_width  Full detector width in pixels.
 * @param sa         Sub-aperture descriptor.
 * @param cfg        Algorithm configuration.
 * @param result     Output centroid (written in place).
 */
void centroid_cog(const float* image,
                  int           img_width,
                  const SubAperture* sa,
                  const CentroidConfig* cfg,
                  Centroid*     result);

/**
 * centroid_threshold_cog:
 *   Background-subtracted, thresholded CoG.  The threshold is either
 *   taken directly from cfg->threshold, or estimated from border pixels
 *   if cfg->border_px > 0.
 */
void centroid_threshold_cog(const float* image,
                             int           img_width,
                             const SubAperture* sa,
                             const CentroidConfig* cfg,
                             Centroid*     result);

/**
 * centroid_windowed_cog:
 *   Two-pass: first computes a T-CoG estimate x0,y0, then applies a
 *   Gaussian window exp(-r²/2σ²) and recomputes.
 */
void centroid_windowed_cog(const float* image,
                            int           img_width,
                            const SubAperture* sa,
                            const CentroidConfig* cfg,
                            Centroid*     result);

/* ── Batch function (main entry point) ───────────────────── */

/**
 * batch_centroid:
 *   Compute centroids for ALL sub-apertures in a single detector frame.
 *   Iterates over sub-apertures, skipping those with sa.valid == 0.
 *   If cfg->use_openmp is set, parallelises across sub-apertures.
 *
 * @param image       Full detector frame as float32 row-major [H × W].
 * @param img_width   Frame width in pixels.
 * @param img_height  Frame height in pixels.
 * @param sas         Array of SubAperture descriptors [n_sa].
 * @param n_sa        Number of sub-apertures.
 * @param cfg         Algorithm configuration.
 * @param results     Output array of Centroid structs [n_sa].
 */
void batch_centroid(const float*  image,
                    int            img_width,
                    int            img_height,
                    const SubAperture* sas,
                    int            n_sa,
                    const CentroidConfig* cfg,
                    Centroid*      results);

/* ── Utility ─────────────────────────────────────────────── */

/**
 * estimate_background:
 *   Estimate background mean and std from a ring of border_px pixels
 *   around the sub-aperture boundary.  Used internally by T-CoG.
 *
 * @param mean_out  Output: estimated background mean.
 * @param std_out   Output: estimated background standard deviation.
 */
void estimate_background(const float* image,
                         int           img_width,
                         const SubAperture* sa,
                         int           border_px,
                         float*        mean_out,
                         float*        std_out);

/**
 * build_subaperture_grid:
 *   Construct the sub-aperture grid from MLA parameters and pupil mask.
 *
 * @param n_sa_x        Sub-apertures in x direction.
 * @param n_sa_y        Sub-apertures in y direction.
 * @param pix_per_sa    Pixels per sub-aperture (square assumed).
 * @param pupil_cx      Pupil centre x in full-frame pixels.
 * @param pupil_cy      Pupil centre y in full-frame pixels.
 * @param pupil_radius  Pupil radius in full-frame pixels.
 * @param sas           Output array [n_sa_x * n_sa_y] (caller-allocated).
 * @return              Number of valid (in-pupil) sub-apertures.
 */
int build_subaperture_grid(int    n_sa_x,
                            int    n_sa_y,
                            int    pix_per_sa,
                            double pupil_cx,
                            double pupil_cy,
                            double pupil_radius,
                            SubAperture* sas);

#ifdef __cplusplus
}
#endif

#endif /* CENTROID_H */
