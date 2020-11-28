/*******************************************************************************
 ********************************* BLUEBOTTLE **********************************
 *******************************************************************************
 *
 *  Copyright 2015 - 2016 Yayun Wang, The Johns Hopkins University
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

#ifndef _SCALAR_H
#define _SCALAR_H

#include "bluebottle.h"

/****d* scalar/SNSP
 * NAME
 *  SNSP
 * USAGE
 */
#define SNSP 2
/* PURPOSE
 * Defines number of scalar products used for scalar field.
 ******
 */

/****d* scalar/SP_YS_RE
 * NAME
 *  SP_YS_RE
 * USAGE
 */
#define SP_YS_RE 0
/* FUNCTION
 *  Defines scalar product stride for Re(Ylm, s) for indexing in packing array
 ******
 */

/****d* scalar/SP_YS_IM
 * NAME
 *  SP_YS_IM
 * USAGE
 */
#define SP_YS_IM 1
/* FUNCTION
 *  Defines scalar product stride for Im(Ylm, s) for indexing in packing array
 ******
 */

/* VARIABLES */

/****s* scalar/BC_s
 * NAME
 *  BC_s
 * TYPE
 */
typedef struct BC_s {
  int sW;
  real sWD;
  real sWN;
  int sE;
  real sED;
  real sEN;
  int sN;
  real sND;
  real sNN;
  int sS;
  real sSD;
  real sSN;
  int sT;
  real sTD;
  real sTN;
  int sB;
  real sBD;
  real sBN;
} BC_s;
/*
 * PURPOSE
 *  Carry the type of boundary condition on each side of the domain.  Possible
 *  types include:
 *  * PERIODIC
 *  * DIRICHLET
 *  * NEUMANN
 * MEMBERS
 * * sW -- the scalar boundary condition type
 * * sWD -- the DIRICHLET boundary conditon value
 * * sWN -- the NEUMANN boundary condition value
 * * sE -- the scalar boundary condition type
 * * sED -- the DIRICHLET boundary conditon value
 * * sEN -- the NEUMANN boundary condition value
 * * sN -- the scalar boundary condition type
 * * sND -- the DIRICHLET boundary conditon value
 * * sNN -- the NEUMANN boundary condition value
 * * sS -- the scalar boundary condition type
 * * sSD -- the DIRICHLET boundary conditon value
 * * sSN -- the NEUMANN boundary condition value
 * * sT -- the scalar boundary condition type
 * * sTD -- the DIRICHLET boundary conditon value
 * * sTN -- the NEUMANN boundary condition value
 * * sB -- the scalar boundary condition type
 * * sBD -- the DIRICHLET boundary conditon value
 * * sBN -- the NEUMANN boundary condition value
 */

/****v* scalar/bc_s
 * NAME
 *  bc_s
 * TYPE
 */
extern BC_s bc_s;
/*
 * PURPOSE
 *  Create an instance of the struct BC to carry boundary condition types.
 ******
 */

/****v* scalar/_bc_s
 * NAME
 *  _bc_s
 * TYPE
 */
extern BC_s *_bc_s;
/*
 * PURPOSE
 *  CUDA device analog for BC bc_s
 ******
 */

/****v* scalar/s_D
 * NAME
 *  s_D
 * TYPE
 */
extern real s_D;
/*
 * PURPOSE
 *  The fluid thermal diffusivity.
 ******
 */

/****v* scalar/s_k
 * NAME
 *  s_k
 * TYPE
 */
extern real s_k;
/*
 * PURPOSE
 *  The fluid thermal conductivity.
 ******
 */

/****v* scalar/s_beta
 * NAME
 *  s_beta
 * TYPE
 */
extern real s_beta;
/*
 * PURPOSE
 *  The fluid thermal expansion coefficient.
 ******
 */

/****v* scalar/SCALAR
 * NAME
 *  SCALAR
 * TYPE
 */
extern int SCALAR;
/*
 * PURPOSE
 *  Used to determine if scalar is turned on(SCALAR >= 1) or not.
 ******
 */

/****v* scalar/lamb_cut_scalar
 * NAME
 *  lamb_cut_scalar
 * TYPE
 */
extern real lamb_cut_scalar;
/*
 * PURPOSE
 *  The magnitude below which errors in Lamb's coefficients are ignored,
 *  compared to the coefficient with greates magnitude. The lower this number,
 *  the more coefficients will be considered important when computing the error.
 *  To improve convergence rate, decrease this number. It should never be
 *  greater than 1e-2.
 ******
 */

/****v* scalar/s_init
 * NAME
 *  s_init
 * TYPE
 */
extern real s_init;
/*
 * PURPOSE
 *  Initial fluid temperature.
 ******
 */

/****v* scalar/s_init_rand
 * NAME
 *  s_init_rand
 * TYPE
 */
extern real s_init_rand;
/*
 * PURPOSE
 *  Fluctuation magnitude of initial fluid temperature.
 ******
 */

/****v* scalar/s_ref
 * NAME
 *  s_ref
 * TYPE
 */
extern real s_ref;
/*
 * PURPOSE
 *  Reference fluid temperature, used in calculation of modified density.
 ******
 */

/****v* scalar/s_ncoeffs_max
 * NAME
 *  s_ncoeffs_max
 * TYPE
 */
extern int s_ncoeffs_max;
/*
 * PURPOSE
 *  Maximum particle coefficient size
 ******
 */

/****v* scalar/s
 * NAME
 *  s
 * TYPE
 */
extern real *s;
/*
 * PURPOSE
 *   *  Scalar field (grid type Gcc; x-component varies first, then
 *  y-component, then z-component).
 ******
 */

/****v* scalar/_s
 * NAME
 *  _s
 * TYPE
 */
extern real *_s;
/*
 * PURPOSE
 *  CUDA device analog for s. It is an array on each processor that contains the
 *  subdomain's field
 ******
 */

/****v* scalar/s0
 * NAME
 *  s0
 * TYPE
 */
extern real *s0;
/*
 * PURPOSE
 * Host s stored from the previous timestep.
 ******
 */

/****v* scalar/_s0
 * NAME
 *  _s0
 * TYPE
 */
extern real *_s0;
/*
 * PURPOSE
 * CUDA device analog for s stored from the previous timestep.
 ******
 */

/****v* scalar/s_conv
 * NAME
 *  s_conv
 * TYPE
 */
extern real *s_conv;
/*
 * PURPOSE
 *  Scalar convection field.
 ******
 */

/****v* scalar/_s_conv
 * NAME
 *  _s_conv
 * TYPE
 */
extern real *_s_conv;
/*
 * PURPOSE
 *  CUDA device analog for s_conv. It is an array on each processor that
 *  contains the subdomain's field
 ******
 */

/****v* scalar/s_conv0
 * NAME
 *  s_conv0
 * TYPE
 */
extern real *s_conv0;
/*
 * PURPOSE
 *  Host array to store the previous scalar convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* scalar/_s_conv0
 * NAME
 *  _s_conv0
 * TYPE
 */
extern real *_s_conv0;
/*
 * PURPOSE
 * CUDA device analog for s_conv0.
 ******
 */

/****v* scalar/s_diff
 * NAME
 *  s_diff
 * TYPE
 */
extern real *s_diff;
/*
 * PURPOSE
 *  Scalar diffusion field.
 ******
 */

/****v* scalar/_s_diff
 * NAME
 *  _s_diff
 * TYPE
 */
extern real *_s_diff;
/*
 * PURPOSE
 *  CUDA device analog for s_diff. It is an array on each processor that
 *  contains the subdomain's field
 ******
 */

/****v* scalar/s_diff0
 * NAME
 *  s_diff0
 * TYPE
 */
extern real *s_diff0;
/*
 * PURPOSE
 *  Host array to store the previous scalar diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* scalar/_s_diff0
 * NAME
 *  _s_diff0
 * TYPE
 */
extern real *_s_diff0;
/*
 * PURPOSE
 * CUDA device analog for s_diff0.
 ******
 */

/****v* scalar/_int_Ys_re
 * NAME
 *  _int_Ys_re
 * TYPE
 */
extern real *_int_Ys_re;
/* FUNCTION
 * Scalar product Re(Ylm, s) for each particle, coefficient, and node
 ******
 */

/****v* scalar/_int_Ys_im
 * NAME
 *  _int_Ys_im
 * TYPE
 */
extern real *_int_Ys_im;
/* FUNCTION
 * Scalar product Im(Ylm, s) for each particle, coefficient, and node
 ******
 */

/* FUNCTIONS */

/****f* scalar/scalar_init_fields()
 * NAME
 *  scalar_init_fields()
 * USAGE
 */
void scalar_init_fields(void);
/*
 * FUNCTION
 *  Initialize scalar field with given boundary and initial conditions
 ******
 */

/****f* scalar/scalar_part_init()
 * NAME
 *  scalar_part_init()
 * USAGE
 */
void scalar_part_init(void);
/*
 * FUNCTION
 *  Initialize part_struct for scalar
 * ARGUMENTS
 ******
 */

/****f* scalar/mpi_send_s_psums_i()
 * NAME
 *  mpi_send_s_psums_i()
 * USAGE
 */
void mpi_send_s_psums_i(void);
/*
 * FUNCTION
 *  Send scalar partial sums in the east/west directions to the appropriate domain
 ******
 */

/****f* scalar/mpi_send_s_psums_j()
 * NAME
 *  mpi_send_s_psums_j()
 * USAGE
 */
void mpi_send_s_psums_j(void);
/*
 * FUNCTION
 *  Send scalar partial sums in the north/south directions to the appropriate domain
 ******
 */

/****f* scalar/mpi_send_s_psums_k()
 * NAME
 *  mpi_send_s_psums_k()
 * USAGE
 */
void mpi_send_s_psums_k(void);
/*
 * FUNCTION
 *  Send scalar partial sums in the top/bottom directions to the appropriate domain
 ******
 */

/****f* scalar/recorder_scalar_init()
 * NAME
 *  recorder_scalar_init()
 * USAGE
 */
void recorder_scalar_init(char *name);
/*
 * FUNCTION
 *  Initialize a scalar recorder
 ******
 */

/****f* scalar/recorder_scalar()
 * NAME
 *  recorder_scalar()
 * USAGE
 */
void recorder_scalar(char *name, real ttime, int mniter, real mtimeiter,
  real merr, int sniter, real stimeiter, real serr);
/*
 * FUNCTION
 *  Record momentum and scalar BC iteration information
 ******
 */

/****f* scalar/cuda_scalar_malloc_host()
 * NAME
 *  cuda_scalar_malloc_host()
 * USAGE
 */
void cuda_scalar_malloc_host(void);
/*
 * FUNCTION
 *  Allocate scalar memory on host
 ******
 */

/****f* scalar/cuda_scalar_malloc_dev()
 * NAME
 *  cuda_scalar_malloc_dev()
 * USAGE
 */
void cuda_scalar_malloc_dev(void);
/*
 * FUNCTION
 *  Allocate scalar memory on device
 ******
 */

/****f* scalar/cuda_scalar_push()
 * NAME
 *  cuda_scalar_push()
 * USAGE
 */
void cuda_scalar_push(void);
/*
 * FUNCTION
 *  Copy scalar fields from host to device
 ******
 */

/****f* scalar/cuda_scalar_pull()
 * NAME
 *  cuda_scalar_pull()
 * USAGE
 */
void cuda_scalar_pull(void);
/*
 * FUNCTION
 *  Copy s from device to host
 ******
 */

/****f* scalar/cuda_scalar_pull_debug()
 * NAME
 *  cuda_scalar_pull_debug()
 * USAGE
 */
void cuda_scalar_pull_debug(void);
/*
 * FUNCTION
 *  Copy s_conv, s_diff from device to host
 ******
 */

/****f* scalar/cuda_scalar_pull_restart()
 * NAME
 *  cuda_scalar_pull_restart()
 * USAGE
 */
void cuda_scalar_pull_restart(void);
/*
 * FUNCTION
 *  Copy s, s_conv, s_diff, s0, s_conv0, s_diff0 from device to host
 ******
 */

/****f* scalar/cuda_scalar_free()
 * NAME
 *  cuda_scalar_free()
 * USAGE
 */
void cuda_scalar_free(void);
/*
 * FUNCTION
 *  Free device memory for scalar on device and device memory reference
 *  pointers on host.
 ******
 */

/****f* scalar/cuda_compute_boussinesq()
 * NAME
 *  cuda_compute_boussinesq()
 * USAGE
 */
void cuda_compute_boussinesq(void);
/*
 * FUNCTION
 *  Set up the boussinesq forcing array for this time step
 ******
 */

/****f* scalar/cuda_scalar_BC()
 * NAME
 *  cuda_scalar_BC()
 * USAGE
 */
void cuda_scalar_BC(real *array);
/*
 * FUNCTION
 *  Enforce scalar boundary conditions on *array on domain boundaries.
 ******
 */

/****f* scalar/cuda_scalar_part_BC()
 * NAME
 *  cuda_scalar_part_BC()
 * USAGE
 */
void cuda_scalar_part_BC(real *array);
/*
 * FUNCTION
 *  Enforce scalar boundary conditions on *array on particle boundaries.
 ******
 */

/****f* scalar/cuda_scalar_part_fill()
 * NAME
 *  cuda_scalar_part_fill()
 * USAGE
 */
void cuda_scalar_part_fill(void);
/*
 * FUNCTION
 *  Fill part cells with particle temperature value.
 ******
 */

/****f* scalar/cuda_scalar_solve()
 * NAME
 *  cuda_scalar_solve()
 * USAGE
 */
void cuda_scalar_solve(void);
/*
 * FUNCTION
 *  Integrate temperature equation.
 ******
 */

/****f* scalar/cuda_scalar_partial_sum_i()
 * NAME
 *  cuda_scalar_partial_sum_i()
 * USAGE
 */
void cuda_scalar_partial_sum_i(void);
/* FUNCTION
 *  Communicate scalar partial sums in the i direction
 ******
 */

/****f* scalar/cuda_scalar_partial_sum_j()
 * NAME
 *  cuda_scalar_partial_sum_j()
 * USAGE
 */
void cuda_scalar_partial_sum_j(void);
/* FUNCTION
 *  Communicate scalar partial sums in the j direction
 ******
 */

/****f* scalar/cuda_scalar_partial_sum_k()
 * NAME
 *  cuda_scalar_partial_sum_k()
 * USAGE
 */
void cuda_scalar_partial_sum_k(void);
/* FUNCTION
 *  Communicate scalar partial sums in the k direction
 ******
 */

/****f* scalar/cuda_scalar_lamb()
 * NAME
 *  cuda_scalar_lamb()
 * USAGE
 */
void cuda_scalar_lamb(void);
/* FUNCTION
 *  Compute the scalar Lamb's coefficients
 ******
 */

/****f* scalar/cuda_scalar_lamb_err()
 * NAME
 *  cuda_scalar_lamb_err()
 * USAGE
 */
real cuda_scalar_lamb_err(void);
/*
 * FUNCTION
 *  Compute the error between the current and previous sets of scalar Lamb's
 *  coefficients.
 ******
 */

/****f* scalar/cuda_store_s()
 * NAME
 *  cuda_store_s()
 * USAGE
 */
void cuda_store_s(void);
/*
 * FUNCTION
 *  Store the _s, _s_conv, _s_diff to _s0, _s_conv0, _s_diff0.
 ******
 */

/****f* scalar/cuda_scalar_update_part()
 * NAME
 *  cuda_scalar_update_part()
 * USAGE
 */
void cuda_scalar_update_part(void);
/*
 * FUNCTION
 *  Update particle temperature.
 ******
 */

/****f* scalar/printMemInfo()
 * NAME
 *  printMemInfo()
 * USAGE
 */
void printMemInfo(void);
/*
 * FUNCTION
 *  Print current total device memory usage.
 ******
 */

#endif
