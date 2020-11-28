/*******************************************************************************
 ********************************* BLUEBOTTLE **********************************
 *******************************************************************************
 *
 *  Copyright 2012 - 2018 Adam Sierakowski and Daniel Willen, 
 *                         The Johns Hopkins University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Please contact the Johns Hopkins University to use Bluebottle for
 *  commercial and/or for-profit applications.
 ******************************************************************************/

/****h* Bluebottle/cuda_physalis
 * NAME
 *  cuda_physalis
 * FUNCTION
 *  Physalis-related operations
 */

#ifndef _CUDA_PHYSALIS_H
#define _CUDA_PHYSALIS_H

extern "C"
{
#include "bluebottle.h"
#include "bluebottle.cuh"
}

/****d* cuda_physalis/NSP
 * NAME
 *  NSP
 * USAGE
 */
#define NSP 6
/* FUNCTION
 * Defines number of scalar products
 ******
 */

/****d* cuda_physalis/SP_YP_RE
 * NAME
 *  SP_YP_RE
 * USAGE
 */
#define SP_YP_RE 0
/* FUNCTION
 *  Defines scalar product stride for Re(Ylm, p) for indexing in packing array
 ******
 */

/****d* cuda_physalis/SP_YP_IM
 * NAME
 *  SP_YP_IM
 * USAGE
 */
#define SP_YP_IM 1
/* FUNCTION
 *  Defines scalar product strides for Im(Ylm, p) for indexing in packing array
 ******
 */

/****d* cuda_physalis/SP_RDYU_RE
 * NAME
 *  SP_RDYU_RE
 * USAGE
 */
#define SP_RDYU_RE 2
/* FUNCTION
 *  Defines scalar product strides for Re(rDYlm, u) for indexing in packing array
 ******
 */

/****d* cuda_physalis/SP_RDYU_IM
 * NAME
 *  SP_RDYU_IM
 * USAGE
 */
#define SP_RDYU_IM 3
/* FUNCTION
 *  Defines scalar product strides for Im(rDYlm, u) indexing in packing array
 ******
 */

/****d* cuda_physalis/SP_XXDYU_RE
 * NAME
 *  SP_XXDYU_RE
 * USAGE
 */
#define SP_XXDYU_RE 4
/* FUNCTION
 *  Defines scalar product strides for Re(x X DYlm, u) for indexing in packing
 *  array
 ******
 */

/****d* cuda_physalis/SP_XXDYU_IM
 * NAME
 *  SP_XXDYU_IM
 * USAGE
 */
#define SP_XXDYU_IM 5
/* FUNCTION
 *  Defines scalar product strides for Im(x X DYlm, u) for indexing in packing
 *  array
 ******
 */

/** VARIABLES **/
/****v* cuda_physalis/_int_Yp_re
 * NAME
 *  _int_Yp_re
 * USAGE
 */
extern real *_int_Yp_re;
/* FUNCTION
 * Scalar product Re(Ylm, p) for each particle, coefficient, and node
 ******
 */

/****v* cuda_physalis/_int_Yp_im
 * NAME
 *  _int_Yp_im
 * USAGE
 */
extern real *_int_Yp_im;
/* FUNCTION
 * Scalar product Im(Ylm, p) for each particle, coefficient, and node
 ******
 */

/****v* cuda_physalis/_int_rDYu_re
 * NAME
 *  _int_rDYu_re
 * USAGE
 */
extern real *_int_rDYu_re;
/* FUNCTION
 * Scalar product Re(rDYlm, u) for each particle, coefficient, and node
 ******
 */

/****v* cuda_physalis/_int_rDYu_im
 * NAME
 *  _int_rDYu_im
 * USAGE
 */
extern real *_int_rDYu_im;
/* FUNCTION
 * Scalar product Im(rDYlm, u) for each particle, coefficient, and node
 ******
 */

/****v* cuda_physalis/_int_xXDYu_re
 * NAME
 *  _int_xXDYu_re
 * USAGE
 */
extern real *_int_xXDYu_re;
/* FUNCTION
 * Scalar product Re(x X DYlm, u) for each particle, coefficient, and node
 ******
 */

/****v* cuda_physalis/_int_xXDYu_im
 * NAME
 *  _int_xXDYu_im
 * USAGE
 */
extern real *_int_xXDYu_im;
/* FUNCTION
 * Scalar product Im(x X DYlm, u) for each particle, coefficient, and node
 ******
 */


/** FUNCTIONS **/

/****f* cuda_physalis/rtp2xyz<<<>>>()
 * NAME
 *  rtp2xyz<<<>>>()
 * USAGE
 */
__device__ void rtp2xyz(real r, real t, real p, real *x, real *y, real *z);
/*
 * PURPOSE
 *  Compute (x, y, z) from (r, theta, phi).
 * ARGUMENTS
 *  r -- r-component in spherical basis
 *  t -- theta-component in spherical basis
 *  p -- phi-component in spherical basis
 *  x -- x-component in Cartesian basis
 *  y -- y-component in Cartesian basis
 *  z -- z-component in Cartesian basis
 ******
 */

/****f* cuda_physalis/nnm<<<>>>()
 * NAME
 *  nnm<<<>>>()
 * USAGE
 */
__device__ real nnm(int n, int m);
/*
 * PURPOSE
 *  Compute spherical harmonic normalization N_nm.
 * ARGUMENTS
 *  * n -- degree
 *  * m -- order
 ******
 */

/****f* cuda_physalis/pnm<<<>>>()
 * NAME
 *  pnm<<<>>>()
 * USAGE
 */
__device__ real pnm(int n, int m, real t);
/*
 * PURPOSE
 *  Compute associated Legendre function P_nm(theta).
 * ARGUMENTS
 *  * n -- degree
 *  * m -- order
 *  * t -- theta
 ******
 */

/****f* cuda_physalis/cart2sphere<<<>>>()
 * NAME
 *  cart2sphere<<<>>>()
 * USAGE
 */
__device__ void cart2sphere(real u, real v, real w, real t, real p,
  real *ur, real *ut, real *up);
/*
 * PURPOSE
 *  Compute (u_r, u_theta, u_phi) from (u, v, w).
 * ARGUMENTS
 *  u -- the x-component of velocity in Cartesian basis
 *  v -- the y-component of velocity in Cartesian basis
 *  w -- the z-component of velocity in Cartesian basis
 *  t -- theta-component in spherical basis
 *  p -- phi-component in spherical basis
 *  ur -- r-component of velocity in spherical basis
 *  ut -- theta-component of velocity in spherical basis
 *  up -- phi-component of velocity in spherical basis
 ******
 */

/****f* cuda_physalis/check_nodes<<<>>>()
 * NAME
 *  check_nodes<<<>>>()
 * USAGE
 */
__global__ void check_nodes(int nparts, part_struct *parts, BC *bc,
  dom_struct *DOM);
/*
 * PURPOSE
 *  CUDA kernel to interplate field variables to Lebedev quadrature nodes.
 * ARGUMENTS
 *  * nparts -- the number of particles
 *  * phi -- phi component of list of Lebedev quadrature nodes
 *  * bc -- domain boundary conditions
 *  * DOM -- global domain struct
 */

/****f* cuda_physalis/interpolate_nodes<<<>>>()
 * NAME
 *  interpolate_nodes<<<>>>()
 * USAGE
 */
__global__ void interpolate_nodes(real *p, real *u, real *v, real *w,
  real rho_f, real nu, gradP_struct gradP, part_struct *parts, real *pp,
  real *ur, real *ut, real *up, BC *bc,
  real s_beta, real s_ref, g_struct g);
/*
 * PURPOSE
 *  CUDA kernel to interpolate field variables to Lebedev quadrature nodes.
 * ARGUMENTS
 *  * p -- device pressure field
 *  * u -- device x-component velocity field
 *  * v -- device y-component velocity field
 *  * w -- device z-component velocity field
 *  * rho_f -- fluid density
 *  * nu -- fluid kinematic viscosity
 *  * gradP -- body force
 *  * parts -- list of particles on this device
 *  * pp -- the interpolated pressure field
 *  * ur -- the interpolated r-component of velocity field
 *  * ut -- the interpolated theta-component of velocity field
 *  * up -- the interpolated phi-component of velocity field
 *  * bc -- boundary condition structure
 ******
 */

/****f* cuda_physalis/lebedev_quadrature<<<>>>()
 * NAME
 *  lebedev_quadrature<<<>>>()
 * USAGE
 */
__global__ void lebedev_quadrature(part_struct *parts, int ncoeffs_max,
  real *pp, real *ur, real *ut, real *up,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Compute the partial sums of the Lebedev quadrature on all particles
 * ARGUMENTS
 *  * parts -- device particle structure
 *  * ncoeffs_max -- maximum number of coefficients for any particle
 *  * pp -- interpolated pressure at quadrature nodes
 *  * ur -- interpolated radial velocity at quadrature nodes
 *  * ut -- interpolated theta velocity at quadrature nodes
 *  * up -- interpolated phi velocity at quadrature nodes
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/compute_lambs_coeffs<<<>>>()
 * NAME
 *  compute_lambs_coeffs<<<>>>()
 * USAGE
 */
__global__ void compute_lambs_coeffs(part_struct *parts, real relax,
  real mu, real nu, int ncoeffs_max, int nparts,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Compute the partial sums of the Lebedev quadrature on all particles
 * ARGUMENTS
 *  * parts -- device particle structure
 *  * mu -- fluid kinematic viscosity
 *  * nu -- fliud dymanic viscosity
 *  * ncoeffs_max -- maximum number of coefficients for any particle
 *  * nparts -- number of parts in domain
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/calc_forces<<<>>>()
 * NAME
 *  calc_forces<<<>>>()
 * USAGE
 */
__global__ void calc_forces(part_struct *parts, int nparts,
  real gradPx, real gradPy, real gradPz, real rho_f, real mu, real nu,
  real s_beta, real s_ref, g_struct g);
/*
 * PURPOSE
 *  Calculate the hydrodynamic forces on each particle
 * ARGUMENTS
 *  * parts -- device particle structure
 *  * nparts -- number of particles in subdomain
 *  * gradPx -- x body force
 *  * gradPy -- y body force
 *  * gradPx -- z body force
 *  * rho_f -- fluid density
 *  * mu -- fluid dynamic viscosity
 *  * nu -- fluid kinematic viscosity
 ******
 */

/****f* cuda_physalis/pack_sums_e<<<>>>()
 * NAME
 *  pack_sums_e<<<>>>()
 * USAGE
 */
__global__ void pack_sums_e(real *sum_send_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * sum_send_e -- partial sums to be communicattd
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/pack_sums_w<<<>>>()
 * NAME
 *  pack_sums_w<<<>>>()
 * USAGE
 */
__global__ void pack_sums_w(real *sum_send_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * sum_send_e -- partial sums to be communicattd
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/pack_sums_n<<<>>>()
 * NAME
 *  pack_sums_n<<<>>>()
 * USAGE
 */
__global__ void pack_sums_n(real *sum_send_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * sum_send_n -- partial sums to be communicattd
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/pack_sums_s<<<>>>()
 * NAME
 *  pack_sums_s<<<>>>()
 * USAGE
 */
__global__ void pack_sums_s(real *sum_send_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * sum_send_s -- partial sums to be communicattd
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/pack_sums_t<<<>>>()
 * NAME
 *  pack_sums_t<<<>>>()
 * USAGE
 */
__global__ void pack_sums_t(real *sum_send_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * sum_send_t -- partial sums to be communicattd
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/pack_sums_b<<<>>>()
 * NAME
 *  pack_sums_b<<<>>>()
 * USAGE
 */
__global__ void pack_sums_b(real *sum_send_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * sum_send_b -- partial sums to be communicattd
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/unpack_sums_e<<<>>>()
 * NAME
 *  unpack_sums_e<<<>>>()
 * USAGE
 */
__global__ void unpack_sums_e(real *sum_recv_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Unpack received partial sums and complete summation
 * ARGUMENTS
 *  * sum_recv_e -- received partial sums
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/unpack_sums_w<<<>>>()
 * NAME
 *  unpack_sums_w<<<>>>()
 * USAGE
 */
__global__ void unpack_sums_w(real *sum_recv_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Unpack received partial sums and complete summation
 * ARGUMENTS
 *  * sum_recv_w -- received partial sums
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/unpack_sums_n<<<>>>()
 * NAME
 *  unpack_sums_n<<<>>>()
 * USAGE
 */
__global__ void unpack_sums_n(real *sum_recv_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Unpack received partial sums and complete summation
 * ARGUMENTS
 *  * sum_recv_n -- received partial sums
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/unpack_sums_s<<<>>>()
 * NAME
 *  unpack_sums_s<<<>>>()
 * USAGE
 */
__global__ void unpack_sums_s(real *sum_recv_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Unpack received partial sums and complete summation
 * ARGUMENTS
 *  * sum_recv_s -- received partial sums
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/unpack_sums_t<<<>>>()
 * NAME
 *  unpack_sums_t<<<>>>()
 * USAGE
 */
__global__ void unpack_sums_t(real *sum_recv_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Unpack received partial sums and complete summation
 * ARGUMENTS
 *  * sum_recv_t -- received partial sums
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* cuda_physalis/unpack_sums_b<<<>>>()
 * NAME
 *  unpack_sums_b<<<>>>()
 * USAGE
 */
__global__ void unpack_sums_b(real *sum_recv_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im);
/*
 * PURPOSE
 *  Unpack received partial sums and complete summation
 * ARGUMENTS
 *  * sum_recv_b -- received partial sums
 *  * offset -- offset of each bin in sum_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * ncoeffs_max -- maximum number of lambs coeffs
 *  * int_Yp_re -- real part of (Y_lm, p)
 *  * int_Yp_im -- imag part of (Y_lm, p)
 *  * int_rDYu_re -- real part of (rDY_lm, u)
 *  * int_rDYu_im -- imag part of (rDY_lm, u)
 *  * int_xXDYu_re -- real part of (r X DY_lm, u)
 *  * int_xXDYu_im -- real part of (r X DY_lm, u)
 ******
 */

/****f* quadrature_kernel/compute_error<<<>>>()
 * NAME
 *  compute_error<<<>>>()
 * USAGE
 */
__global__ void compute_error(real lamb_cut, int ncoeffs_max, int nparts,
  part_struct *parts, real *part_errors, int *part_nums);
/*
 * PURPOSE
 *  Compute the error between the current and previous iteration Lamb's
 *  coefficient.
 * ARGUMENTS
 *  * lamb_cut -- the magnitude of errors below which to ignore, referenced to
 *    the error with the greatest magnitude
 *  * ncoeffs_max -- the Lamb's coefficient array access stride length
 *  * nparts -- the total number of particles
 *  * parts -- particle info structure
 *  * part_errors -- the maximum error for each particle
 *  * part_nums -- particle global number
 ******
*/

/****f* quadrature_kernel/store_coeffs<<<>>>()
 * NAME
 *  store_coeffs<<<>>>()
 * USAGE
 */
__global__ void store_coeffs(part_struct *parts, int nparts, int ncoeffs_max);
/*
 * PURPOSE
 *  Compute the error between the current and previous iteration Lamb's
 *  coefficient.
 * ARGUMENTS
 *  * parts -- particle info structure
 *  * nparts -- the total number of particles
 *  * ncoeffs_max -- the Lamb's coefficient array access stride length
 ******
*/

#endif // _CUDA_PHYSALIS_H
