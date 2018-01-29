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
/****h* Bluebottle/cuda_solver
 * NAME
 *  cuda_solver
 * FUNCTION
 *  Bluebottle Poisson solver CUDA host functions.
 ******
 */

#ifndef _CUDA_SOLVER_H
#define _CUDA_SOLVER_H

extern "C"
{
#include "bluebottle.h"
#include "bluebottle.cuh"
#include "recorder.h"
}

/****f* cuda_solver/PP_jacobi_init<<<>>>()
  * NAME
  *  PP_jacobi_init<<<>>>()
  * USAGE
  */
__global__ void PP_jacobi_init(int *flag_u, int *flag_v, int *flag_w,
  real *invM);
/*
 * FUNCTION
 *  Compute the right-hand side of the pressure Poisson problem. 
 * ARGUMENTS
 *  * _flag_u -- Gfx field containing boundary condition flags
 *  * _flag_v -- Gfy field containing boundary condition flags
 *  * _flag_w -- Gfz field containing boundary condition flags
 *  * _invM -- diagonal jacobi preconditioner
 ******
 */

/****f* cuda_solver/PP_rhs<<<>>>()
  * NAME
  *  PP_rhs<<<>>>()
  * USAGE
  */
__global__ void PP_rhs(real rho_f, real *u_star, real *v_star, real *w_star,
  real *rhs, real dt);
/*
 * FUNCTION
 *  Compute the right-hand side of the pressure Poisson problem. 
 * ARGUMENTS
 *  * rho_f -- fluid density
 *  * u_star -- device subdomain u-component intermediate velocity field
 *  * v_star -- device subdomain v-component intermediate velocity field
 *  * w_star -- device subdomain w-component intermediate velocity field
 *  * rhs -- the right-hand side array to build
 *  * dt -- the current timestep
 ******
 */

/****f* cuda_solver/zero_rhs_ghost_i<<<>>>()
  * NAME
  *  zero_rhs_ghost_i<<<>>>()
  * USAGE
  */
__global__ void zero_rhs_ghost_i(real *rhs_p);
/*
 * FUNCTION
 *  Zero the i-direction ghost cells of the rhs.
 * ARGUMENTS
 *  * rhs_p -- the right-hand side array to build
 ******
 */

/****f* cuda_solver/zero_rhs_ghost_j<<<>>>()
  * NAME
  *  zero_rhs_ghost_j<<<>>>()
  * USAGE
  */
__global__ void zero_rhs_ghost_j(real *rhs_p);
/*
 * FUNCTION
 *  Zero the j-direction ghost cells of the rhs.
 * ARGUMENTS
 *  * rhs_p -- the right-hand side array to build
 ******
 */

/****f* cuda_solver/zero_rhs_ghost_k<<<>>>()
  * NAME
  *  zero_rhs_ghost_k<<<>>>()
  * USAGE
  */
__global__ void zero_rhs_ghost_k(real *rhs_p);
/*
 * FUNCTION
 *  Zero the k-direction ghost cells of the rhs.
 * ARGUMENTS
 *  * rhs_p -- the right-hand side array to build
 ******
 */


/****f* cuda_solver/coeffs_refine<<<>>>()
  * NAME
  *  coeffs_refine<<<>>>()
  * USAGE
  */
__global__ void coeffs_refine(real *rhs, int *phase, int *flag_u, int *flag_v,
  int *flag_w);
/*
 * FUNCTION
 *  Compute the right-hand side of the pressure Poisson problem. 
 * ARGUMENTS
 *  * rhs -- the right-hand side array to build
 *  * phase -- phase field
 *  * _flag_u -- Gfx field containing boundary condition flags
 *  * _flag_v -- Gfy field containing boundary condition flags
 *  * _flag_w -- Gfz field containing boundary condition flags
 ******
 */

/****f* cuda_solver/PP_cg_init<<<>>>()
  * NAME
  *  PP_cg_init<<<>>>()
  * USAGE
  */
__global__ void PP_cg_init(real *r_q, real *p_q, real *pb_q,
  real *phi, real *rhs, real *z_q, real *invM);
/*
 * FUNCTION
 *  Initialize the residual and search direction for cg. These exist on each
 *  subdoms portion of the global soln
 * ARGUMENTS
 *  * r_q -- initial residual at subdom soln points
 *  * p_q -- initial search direction at subdom soln points
 *  * pb_q -- initial search direction at global soln points
 *  * phi -- initial guess
 *  * rhs -- rhs of PP equation
 *  * z_q -- auxiliary vector for preconditioner
 *  * invM -- Jacobi preconditioner
 ******
 */

/****f* cuda_solver/PP_init_search_ghost<<<>>>()
  * NAME
  *  PP_init_search_ghost<<<>>>()
  * USAGE
  */
__global__ void PP_init_search_ghost(real *pb_q, real *rhs, 
  dom_struct *DOM, BC *bc);
/*
 * FUNCTION
 *  Initialize the search direction for cg. This exists on the global soln 
 *  cells in each subdom
 * ARGUMENTS
 *  * pb_q -- initial search direction at global soln points
 *  * rhs -- rhs of PP equation
 *  * DOM -- The device structure containing global domain info
 ******
 */

/****f* cuda_solver/PP_spmv<<<>>>()
 * NAME
 *  PP_spmv<<<>>>()
 * USAGE
 */
__global__ void PP_spmv(int *flag_u, int *flag_v, int *flag_w, 
  real *pb_q, real *Apb_q);
/*
 * FUNCTION
 *  Calculates the sparse matrix vector product A*pb_q for the subdomain. This
 *  uses a matrix-free implementation of A, where the matrix is calculated
 *  on-the-fly.
 * ARGUMENTS
 *  * _flag_u -- Gfx field containing boundary condition flags
 *  * _flag_v -- Gfy field containing boundary condition flags
 *  * _flag_w -- Gfz field containing boundary condition flags
 *  * _pb_q  -- The search direction at global soln points in subdom
 *  * _Apb_q -- Vector containing the result of A*pb_q at each point in the
 *    solution vector in the current subdomain
 ******
 */

/****f* cuda_solver/PP_spmv_shared<<<>>>()
 * NAME
 *  PP_spmv_shared<<<>>>()
 * USAGE
 */
__global__ void PP_spmv_shared(int *flag_u, int *flag_v, int *flag_w, 
  real *pb_q, real *Apb_q);
/*
 * FUNCTION
 *  Shared memory version of PP_spmv. Calculates the sparse matrix 
 *  vector product A*pb_q for the subdomain. This uses a matrix-free 
 *  implementation of A, where the matrix is calculated on-the-fly.
 * ARGUMENTS
 *  * _flag_u -- Gfx field containing boundary condition flags
 *  * _flag_v -- Gfy field containing boundary condition flags
 *  * _flag_w -- Gfz field containing boundary condition flags
 *  * _pb_q  -- The search direction at global soln points in subdom
 *  * _Apb_q -- Vector containing the result of A*pb_q at each point in the
 *    solution vector in the current subdomain
 ******
 */

/****f* cuda_solver/PP_spmv_shared_load<<<>>>()
 * NAME
 *  PP_spmv_shared_load<<<>>>()
 * USAGE
 */
__global__ void PP_spmv_shared_load(int *flag_u, int *flag_v, int *flag_w, 
  real *pb_q, real *Apb_q, int *phase);
/*
 * FUNCTION
 *  Smarter load version of shared PP_spmv. Calculates the sparse matrix 
 *  vector product A*pb_q for the subdomain. This uses a matrix-free 
 *  implementation of A, where the matrix is calculated on-the-fly.
 * ARGUMENTS
 *  * flag_u -- Gfx field containing boundary condition flags
 *  * flag_v -- Gfy field containing boundary condition flags
 *  * flag_w -- Gfz field containing boundary condition flags
 *  * pb_q  -- The search direction at global soln points in subdom
 *  * Apb_q -- Vector containing the result of A*pb_q at each point in the
 *    solution vector in the current subdomain
 *  * phase -- phase indicator field
 ******
 */

/****f* cuda_solver/PP_spmv_shared_load_noparts<<<>>>()
 * NAME
 *  PP_spmv_shared_load_noparts<<<>>>()
 * USAGE
 */
__global__ void PP_spmv_shared_load_noparts(int *flag_u, int *flag_v,
  int *flag_w, real *pb_q, real *Apb_q);
/*
 * FUNCTION
 *  Smarter load version of shared PP_spmv. Calculates the sparse matrix 
 *  vector product A*pb_q for the subdomain. This uses a matrix-free 
 *  implementation of A, where the matrix is calculated on-the-fly.
 *  Also has no particle capabilities.
 * ARGUMENTS
 *  * _flag_u -- Gfx field containing boundary condition flags
 *  * _flag_v -- Gfy field containing boundary condition flags
 *  * _flag_w -- Gfz field containing boundary condition flags
 *  * _pb_q  -- The search direction at global soln points in subdom
 *  * _Apb_q -- Vector containing the result of A*pb_q at each point in the
 *    solution vector in the current subdomain
 ******
 */

/****f* cuda_solver/PP_update_soln_resid<<<>>>()
  * NAME
  *  PP_update_soln_resid<<<>>>()
  * USAGE
  */
__global__ void PP_update_soln_resid(real *phi, real *p_q, real *r_q,
  real *Apb_q, real *z_q, real alpha, real *invM);
/*
 * FUNCTION
 *  Updates the solution vector and residual for conjugate gradient
 * ARGUMENTS
 *  * phi -- solution vector
 *  * p_q -- search direction
 *  * r_q -- residual
 *  * Apb_q -- result of A*pb_q
 *  * z_q -- aux vector
 *  * alpha -- step size
 *  * invM -- jacobi precond
 ******
 */

/****f* cuda_solver/PP_update_solution<<<>>>()
  * NAME
  *  PP_update_solution<<<>>>()
  * USAGE
  */
__global__ void PP_update_solution(real *phi, real *p_q, real alpha); 
/*
 * FUNCTION
 *  Updates the solution vector for conjugate gradient. Special
 *  function for when we update the residual explicitly.
 * ARGUMENTS
 *  * phi -- solution vector
 *  * p_q -- search direction
 *  * alpha -- step size
 ******
 */

/****f* cuda_solver/PP_update_residual<<<>>>()
  * NAME
  *  PP_update_residual<<<>>>()
  * USAGE
  */
__global__ void PP_update_residual(real *r_q, real *rhs, real *Apb_q, real *z_q,
  real *invM);
/*
 * FUNCTION
 *  Updates the residual for conjugate gradient. Special function for when we
 *  update the residual explicitly.
 * ARGUMENTS
 *  * r_q -- residual
 *  * Apb_q -- result of A*pb_q
 *  * z_q -- aux vector
 *  * invM -- jacobi precond
 ******
 */

/****f* cuda_solver/PP_update_search<<<>>>()
  * NAME
  *  PP_update_search<<<>>>()
  * USAGE
  */
__global__ void PP_update_search(real *p_q, real *pb_q, real *z_q,
  real beta);
/*
 * FUNCTION
 *  Updates the search direction for conjugate gradient
 * ARGUMENTS
 *  * p_q -- search direction
 *  * pb_q -- search direction (incl exchange cells)
 *  * z_q -- preconditioned residual
 *  * beta -- gradient correctio factor
 *  * invM -- joacobi preconditioner
 ******
 */

/****f* cuda_solver/PP_bcgs_init<<<>>>()
  * NAME
  *  PP_bcgs_init<<<>>>()
  * USAGE
  */
__global__ void PP_bcgs_init(real *r_q, real *rs_0, real *p_q, real *pb_q, 
  real *phi, real *rhs);
/*
 * FUNCTION
 *  Initialize the residual and search direction for BiCGstab. These exist on 
 *  each subdoms portion of the global soln. The initializations are:
 *  r_q = rhs_p
 *  rs_0 = rhs_p
 *  p_q = rhs_p
 *  pb_q = rhs_p
 *  phi = 0
 * ARGUMENTS
 *  * r_q -- initial residual at subdom soln points
 *  * p_q -- initial search direction at subdom soln points
 *  * pb_q -- initial search direction at global soln points
 *  * phi -- initial guess
 *  * rhs -- rhs of PP equation
 ******
 */

/****f* cuda_solver/PP_bcgs_update_search<<<>>>()
  * NAME
  *  PP_bcgs_update_search<<<>>>()
  * USAGE
  */
__global__ void PP_bcgs_update_search(real *p_q, real *pb_q, real *r_q,
  real beta, real omega, real *Apb_q);
/*
 * FUNCTION
 *  Updates the search direction for conjugate gradient
 * ARGUMENTS
 *  * p_q -- search direction (local soln points)
 *  * pb_q -- search direction (global soln points)
 *  * r_q -- residual (local soln points)
 *  * beta -- step size in search direction
 *  * omega -- step size in second search direction
 *  * Apb_q -- result of spmv A*pb_q
 ******
 */

/****f* cuda_solver/PP_find_second_search<<<>>>()
  * NAME
  *  PP_find_second_search<<<>>>()
  * USAGE
  */
__global__ void PP_find_second_search(real *s_q, real *sb_q, real *r_q, 
  real alpha, real *Apb_q);
/*
 * FUNCTION
 *  Updates the solution vector and residual for biconjugate gradient
 * ARGUMENTS
 *  * s_q -- second search direction (local soln cells)
 *  * sb_q -- second search direction (local + exchange cells)
 *  * r_q -- residual
 *  * alpha -- how far to move in new search direction
 *  * Apb_q -- result of A*pb_q
 ******
 */

/****f* cuda_solver/PP_bcgs_update_soln_resid<<<>>>()
  * NAME
  *  PP_update_soln_resid<<<>>>()
  * USAGE
  */
__global__ void PP_bcgs_update_soln_resid(real *phi, real alpha, real *p_q,
  real omega, real *s_q, real *r_q, real *Asb_q);
/*
 * FUNCTION
 *  Updates the solution vector and residual for conjugate gradient
 * ARGUMENTS
 *  * phi -- solution vector
 *  * alpha -- how far to move in new search direction
 *  * p_q -- search direction
 *  * omega -- step in second direction
 *  * s_q -- second direction
 *  * r_q -- residual
 *  * Asb_q -- result of A*sb_q
 ******
 */

/****f* cuda_solver/PP_subtract_mean<<<>>>()
  * NAME
  *  PP_subtract_mean<<<>>>()
  * USAGE
  */
__global__ void PP_subtract_mean(real *phi, real phi_mean);
/*
 * FUNCTION
 *  Subtracts the mean from all elements of the poisson solution
 * ARGUMENTS
 *  * phi -- poisson solution
 *  * phi_mean -- global mean of solution
 ******
 */

#endif
