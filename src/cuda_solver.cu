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

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include "cuda_solver.h"
#include "cuda_bluebottle.h"

extern "C"
void cuda_PP_init_jacobi_preconditioner(void)
{
  PP_jacobi_init<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_flag_u,
    _flag_v, _flag_w, _invM);
}

extern "C"
void cuda_PP_cg(void)
{
  // Timing
  struct timeval ts, te;
  gettimeofday(&ts, 0);

  /* TODO LIST
   *  Use a strided reduction to find the phi_mean
   *  Init search direction pb_q (s3b) without MPI. rhs is known at s3b
   *  PP_rhs -- only load one plane of w per loop
   *  Non-blocking mpi, e.g. overlap communication and computation (async cuda)
   *  quit solver if (rhs, rhs) < some tolerance.
   *  Initialzie jacobi with shared memory
   *  PP_rhs -- do final calculation into global, not shared, rather than copy?
   *  Yuhang's stuff (refine coefficients for particles and CG)
   */

 // Comp/comm overlap: see end of http://on-demand.gputechconf.com/gtc/2016/
 //   presentation/s6142-jiri-kraus-multi-gpu-programming-mpi.pdf for stream
 //   overlapping

  /* NOTE:
   *  Each element on the diagonal of the finite difference matrix A is <= 0.
   *  Since the jacobi preconditioner uses this, we end up where the residual
   *  sp_rq = (r_q, invM*r_q) = (r_q, z_q) is actually negative. This is tricky,
   *  since the residual should be positive. To get around this, we will
   *  actually solve -A*phi = -b, such that diag(A) >= 0 and sp_rq >= 0.
   *  This change is effected in several different kernels; a note has been made
   *  in these kernels where the changes are made.
   *  * PP_jacobi_init: use diag(-A)
   *  * PP_rhs: use -b
   *  * PP_spmv*: use -A*phi
   *  * resid = (-b) - (-A)*phi
   *  * part_BC_p and refine_coeffs also have modifications
   */

   /* Preconditioned conjugate gradient for inv(M)*(-A)*phi = inv(M)*(-b)
    * Y. Saad, Iterative Methods for Sparse Linear Systems
    * invM = diag(-A)
    * phi_0 = 0 (solving for pressure correction, should be close to zero)
    * r0 = b
    * z0 = invM*r0
    * p0 = z0
    * while (r_q, z_q) >= pp_residual^2 * (b, b)
    *   num   = (r_q, z_q)
    *   denom = (p_q, A*p_q)
    *   alpha = num/denum
    *   phi_q+1 = phi_q + alpha * p_q
    *   r_q+1 = r_q + alpha*A*p_q
    *     sometimes do r_q+1 = -b + A*phi_q+1
    *   z_q+1 = invM*r_q
    *   beta  = (r_q+1, z_q+1) / (r_q, z_q)
    *   p_q+1 = z_q+1 + beta*p_q
    *   q++
    */

  /* Variables (all Gcc)
   *  * _rhs_p      -- s3b -- RHS at all global soln points in each subdom
   *  * _phi        -- s3b -- soln at all global soln pts
   *  * _phinoghost -- s3  -- soln at subdoms soln pts
   *  * _r_q        -- s3  -- residual at subdom soln pts
   *  * _z_q        -- s3  -- extra array for jacobi
   *  * _p_q        -- s3  -- search direction at subdom's soln points
   *  * _pb_q       -- s3b -- search direction at dom's soln points
   *  * _Apb_q      -- s3  -- result of SpMV at subdom's soln points
   */
  real sp_rhs;          // (rhs, rhs)
  real sp_rq;           // (r_q, r_q)
  real sp_rq1;          // (r_q+1, r_q+1)
  real alpha;           // step length
  real beta;            // gradient correction factor
  real DENOM;           // alpha = sp_rq/DENOM
  real etime;           // elapsed time

  /* Device pointers for thrust reductions */
  thrust::device_ptr<real> t_r_q(_r_q);
  thrust::device_ptr<real> t_p_q(_p_q);
  thrust::device_ptr<real> t_Apb_q(_Apb_q);
  thrust::device_ptr<real> t_rhs(_rhs_p);
  thrust::device_ptr<real> t_z_q(_z_q);

  /* Reset _rhs_p to zero so scalar product isn't affected by data in the ghost
      cells -- we don't want to double-count when we MPI-Reduce */
  cudaMemset(_rhs_p, 0., dom[rank].Gcc.s3b * sizeof(real));

  /* Calculate _rhs_p at each subdomain's soln points */
  PP_rhs<<<blocks.Gcc.num_kn_s, blocks.Gcc.dim_kn_s>>>(rho_f, _u_star, _v_star, _w_star,
  _rhs_p, dt);

  if (NPARTS > 0) {
    /* Modify the right hand side for particle boundary conditions */
    if (nparts > 0) {
      cuda_part_BC_p();
    }

    /* Exchange rhs for refinement */
    mpi_cuda_exchange_Gcc(_rhs_p);

    /* Refine right hand side to be symmetric */
    // Need to exchange rhs to get refinement correct, then zero out ghost cells again.
    if (nparts > 0) {
      coeffs_refine<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_rhs_p, _phase, 
        _flag_u, _flag_v, _flag_w);
    }

    /* Zero out ghost cells so initial scalar product is correct */
    zero_rhs_ghost_i<<<blocks.Gcc.num_inb, blocks.Gcc.dim_inb>>>(_rhs_p);
    zero_rhs_ghost_j<<<blocks.Gcc.num_jnb, blocks.Gcc.dim_jnb>>>(_rhs_p);
    zero_rhs_ghost_k<<<blocks.Gcc.num_knb, blocks.Gcc.dim_knb>>>(_rhs_p);
  }

  /* Calculate initial norm ||rhs|| -- works bc internal ghost cells are zero */
  sp_rhs = thrust::inner_product(t_rhs, t_rhs + dom[rank].Gcc.s3b, t_rhs, 0.);
  MPI_Allreduce(MPI_IN_PLACE, &sp_rhs, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  /* Exchange RHS between subdomains */
  mpi_cuda_exchange_Gcc(_rhs_p);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Initial residual, search direction, and initial guess */
  PP_cg_init<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_r_q, _p_q, _pb_q, 
    _phi, _rhs_p, _z_q, _invM);

  /* Exchange search direction */
  mpi_cuda_exchange_Gcc(_pb_q);
  MPI_Barrier(MPI_COMM_WORLD);
  //PP_init_search_ghost<<<blocks.Gcc.num_knb, blocks.Gcc.dim_knb>>>(_pb_q,
  //  _rhs_p, _DOM, _bc);

  /* Initial residual scalar product */
  sp_rq = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_z_q, 0.);
  MPI_Allreduce(MPI_IN_PLACE, &sp_rq, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  /* Main CG loop */
  int q = 0;
  char rname[FILE_NAME_SIZE] = "solver_expd.rec";

  // Guess for tolerance. If it doesn't iterate, error will accumulate and
  // solver will run again
  real RHS_TOL = 1.e-8;

  if (sp_rhs < RHS_TOL*RHS_TOL) {         // Quit if RHS is small -- this is a guess!
    gettimeofday(&te, 0);
    etime = (te.tv_sec - ts.tv_sec);
    etime += (te.tv_usec - ts.tv_usec)*1.e-6;  // seconds

    recorder_PP(rname, q, 0., etime);

    if (rank == 0) 
      printf("N%d >> Norm of the rhs is less than %.1e, exiting solver\n", rank,
        RHS_TOL);

  } else {                        // Iterate
    while (q <= pp_max_iter) {
      ++q;

      /* alpha = (r_q, z_q) / (p_q, A*pb_q)
       * NUM:   sp_rq
       * DENOM: (p_q, A*pb_q) (SpMV)
       */

      /* Sparse Matrix-Vector Product */
      PP_spmv_shared_load<<<blocks.Gcc.num_kn_s,blocks.Gcc.dim_kn_s>>>(_flag_u,
        _flag_v, _flag_w, _pb_q, _Apb_q, _phase);

      DENOM=thrust::inner_product(t_p_q, t_p_q + dom[rank].Gcc.s3, t_Apb_q, 0.);
      MPI_Allreduce(MPI_IN_PLACE, &DENOM, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
      alpha = sp_rq/DENOM;

      /* Update solution and residual */
      if (q % 50 == 0) { // recalculate every 50 iterations
        // update solution
        PP_update_solution<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phi,
          _p_q, alpha);

        // exchange solution (needed for SpMv)
        mpi_cuda_exchange_Gcc(_phi);

        // SpMV, A*phi
        PP_spmv_shared_load<<<blocks.Gcc.num_kn_s,blocks.Gcc.dim_kn_s>>>(_flag_u,
          _flag_v, _flag_w, _phi, _Apb_q, _phase);

        // update residual (r_q+1 = -b - (-A)*phi) and precond array
        PP_update_residual<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_r_q,
          _rhs_p, _Apb_q, _z_q, _invM);

      } else { // update normally
        PP_update_soln_resid<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(
          _phi, _p_q, _r_q, _Apb_q, _z_q, alpha, _invM);
      }

      /* Scalar product of new residual */
      sp_rq1 = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_z_q, 0.);
      MPI_Allreduce(MPI_IN_PLACE, &sp_rq1, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

      /* Check convergence */
      if (sp_rq1 <= pp_residual*pp_residual*sp_rhs) {
        gettimeofday(&te, 0);
        etime = (te.tv_sec - ts.tv_sec);
        etime += (te.tv_usec - ts.tv_usec)*1e-6;  // seconds

        recorder_PP(rname, q, sqrt(sp_rq1)/sqrt(sp_rhs), etime);
        //printf("N%d >> The PP equation converged in %d iterations\n", rank, q);
        break;

      /* Check if sp_rq1 is nan */
      } else if (isnan(sp_rq1)) {
        if (rank == 0) {
          printf("N%d >> The PP equation did not converge.\n", rank);
          printf("N%d >> The residual at iteration %d is nan (%lf).\n",
            rank, q, sqrt(sp_rq1));
        }
        exit(EXIT_FAILURE);

      /* Continue with CG iterations */
      } else {
        /* Beta = (r_q1, r_q1)/(r_q, r_q) */
        beta = sp_rq1/sp_rq;

        /* Update search direction */
        PP_update_search<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p_q, 
          _pb_q, _z_q, beta);

        /* Communicate _pb_q */
        mpi_cuda_exchange_Gcc(_pb_q);

        /* change sp_rq1 -> sq_rq */
        sp_rq = sp_rq1;
      }
    }

    /* Exit if q == pp_max_iter */
    if (q > pp_max_iter) {
      printf("N%d >> The pressure-Poisson equation did not converge.\n",
        rank);
      printf("N%d >> (rhs, rhs) is %e\n", rank, sp_rhs);
      printf("N%d >> Residual at iteration %d is %lf\n", rank, q,
        sqrt(sp_rq1)/sqrt(sp_rhs));

      exit(EXIT_FAILURE);
    }

    //if (NPARTS == 0) {
    //  /* Copy phi -> phinoghost */
    //  copy_p_p_noghost<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, _phi);

    //  /* Subtract mean from solution */
    //  thrust::device_ptr<real> t_phing(_phinoghost);
    //  real phi_mean = thrust::reduce(t_phing, t_phing + dom[rank].Gcc.s3, 0.,
    //    thrust::plus<real>());
    //  MPI_Allreduce(MPI_IN_PLACE, &phi_mean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
    //  phi_mean /= (real) DOM.Gcc.s3;

    //  PP_subtract_mean<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, 
    //    phi_mean);

    //  /* Copy phinoghost -> phi */
    //  copy_p_noghost_p<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, _phi);
    //}

  }
}

extern "C"
void cuda_PP_cg_timed(void)
{
  // Timing
  struct timeval ts, te;
  gettimeofday(&ts, 0);

  /* Variables (all Gcc)
   *  * _rhs_p      -- s3b -- RHS at all global soln points in each subdom
   *  * _phi        -- s3b -- soln at all global soln pts
   *  * _phinoghost -- s3  -- soln at subdoms soln pts
   *  * _r_q        -- s3  -- residual at subdom soln pts
   *  * _z_q        -- s3  -- extra array for jacobi
   *  * _p_q        -- s3  -- search direction at subdom's soln points
   *  * _pb_q       -- s3b -- search direction at dom's soln points
   *  * _Apb_q      -- s3  -- result of SpMV at subdom's soln points
   */
  real sp_rhs;          // (rhs, rhs)
  real sp_rq;           // (r_q, r_q)
  real sp_rq1;          // (r_q+1, r_q+1)
  real alpha;           // step length
  real beta;            // gradient correction factor
  real DENOM;           // alpha = sp_rq/DENOM
  real etime;           // elapsed time

  /* Device pointers for thrust reductions */
  thrust::device_ptr<real> t_r_q(_r_q);
  thrust::device_ptr<real> t_p_q(_p_q);
  thrust::device_ptr<real> t_Apb_q(_Apb_q);
  thrust::device_ptr<real> t_rhs(_rhs_p);
  thrust::device_ptr<real> t_z_q(_z_q);

  /* Reset _rhs_p to zero so scalar product isn't affected by data in the ghost
      cells -- we don't want to double-count when we MPI-Reduce */
  cudaMemset(_rhs_p, 0., dom[rank].Gcc.s3b * sizeof(real));

  /* Calculate _rhs_p at each subdomain's soln points */
  PP_rhs<<<blocks.Gcc.num_kn_s, blocks.Gcc.dim_kn_s>>>(rho_f, _u_star, _v_star, _w_star,
  _rhs_p, dt);

  /* Calculate initial norm ||rhs|| */
  sp_rhs = thrust::inner_product(t_rhs, t_rhs + dom[rank].Gcc.s3b, t_rhs, 0.);

  MPI_Allreduce(MPI_IN_PLACE, &sp_rhs, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  /* Exchange RHS between subdomains */
  mpi_cuda_exchange_Gcc(_rhs_p);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Initial residual, search direction, and initial guess */
  PP_cg_init<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_r_q, _p_q, _pb_q, 
    _phi, _rhs_p, _z_q, _invM);

  /* Exchange search direction */
  mpi_cuda_exchange_Gcc(_pb_q);
  MPI_Barrier(MPI_COMM_WORLD);
  //PP_init_search_ghost<<<blocks.Gcc.num_knb, blocks.Gcc.dim_knb>>>(_pb_q,
  //  _rhs_p, _DOM, _bc);

  /* Initial residual scalar product */
  sp_rq = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_z_q, 0.);
  MPI_Allreduce(MPI_IN_PLACE, &sp_rq, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  /* Main CG loop */
  int q = 0;
  char rname[FILE_NAME_SIZE] = "solver_expd_timed.rec";

  // Guess for tolerance. If it doesn't iterate, error will accumulate and
  // solver will run again
  real RHS_TOL = 1.e-8;

  // for timing (put in an ifdef?)
  struct timeval ts_spmv, te_spmv;  // PP_spmv_shared_load
  struct timeval ts_ip1, te_ip1;    // thrust::inner_product
  struct timeval ts_AR1, te_AR1;    // MPI_Allreduce
  struct timeval ts_up1, te_up1;    // PP_update_soln_resid
  struct timeval ts_ip2, te_ip2;    // thurst::inner_product
  struct timeval ts_AR2, te_AR2;    // MPI_Allreduce
  struct timeval ts_up2, te_up2;    // PP_update_search
  struct timeval ts_mpi, te_mpi;    // mpi_cuda_exchange_Gcc

  real etime_spmv = 0.;
  real etime_ip1 = 0.;
  real etime_AR1 = 0.;
  real etime_up1 = 0.;
  real etime_ip2 = 0.;
  real etime_AR2 = 0.;
  real etime_up2 = 0.;
  real etime_mpi = 0.;

  if (stepnum == 1) {
    recorder_PP_init_timed(rname);
  }

  if (sp_rhs < RHS_TOL*RHS_TOL) {         // Quit if RHS is small -- this is a guess!
    gettimeofday(&te, 0);
    etime = (te.tv_sec - ts.tv_sec);
    etime += (te.tv_usec - ts.tv_usec)*1.e-6;  // seconds

    recorder_PP_timed(rname, q, 0., etime,
      etime_spmv, etime_ip1, etime_AR1, etime_up1, etime_ip2,
      etime_AR2, etime_up2, etime_mpi);

    printf("N%d >> Norm of the rhs is less than %e, exiting solver\n", rank,
      RHS_TOL);

  } else {                        // Iterate
    while (q <= pp_max_iter) {
      ++q;

      /* alpha = (r_q, z_q) / (p_q, A*pb_q)
       * NUM:   sp_rq
       * DENOM: (p_q, A*pb_q) (SpMV)
       */

      /* Sparse Matrix-Vector Product */
gettimeofday(&ts_spmv, 0);
      PP_spmv_shared_load_noparts<<<blocks.Gcc.num_kn_s,blocks.Gcc.dim_kn_s>>>(
        _flag_u, _flag_v, _flag_w, _pb_q, _Apb_q);
cudaDeviceSynchronize();
gettimeofday(&te_spmv, 0);
etime_spmv += (te_spmv.tv_sec  - ts_spmv.tv_sec);
etime_spmv += (te_spmv.tv_usec - ts_spmv.tv_usec) * 1.e-6;

      /* Inner product */
gettimeofday(&ts_ip1, 0);
      DENOM=thrust::inner_product(t_p_q, t_p_q + dom[rank].Gcc.s3, t_Apb_q, 0.);
gettimeofday(&te_ip1, 0);
etime_ip1 += (te_ip1.tv_sec  - ts_ip1.tv_sec);
etime_ip1 += (te_ip1.tv_usec - ts_ip1.tv_usec) * 1.e-6;

      /*   MPI_Allreduce */
gettimeofday(&ts_AR1, 0);
      MPI_Allreduce(MPI_IN_PLACE, &DENOM, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
gettimeofday(&te_AR1, 0);
etime_AR1 += (te_AR1.tv_sec  - ts_AR1.tv_sec);
etime_AR1 += (te_AR1.tv_usec - ts_AR1.tv_usec) * 1.e-6;

      alpha = sp_rq/DENOM;

      /* Update solution and residual */
      if (q % 50 == 0) { // recalculate every 50 iterations
        // update solution
gettimeofday(&ts_up1, 0);
        PP_update_solution<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phi,
          _p_q, alpha);
cudaDeviceSynchronize();
gettimeofday(&te_up1, 0);
etime_up1 += (te_up1.tv_sec  - ts_up1.tv_sec);
etime_up1 += (te_up1.tv_usec - ts_up1.tv_usec) * 1.e-6;

        // exchange solution (needed for SpMv)
gettimeofday(&ts_mpi, 0);
        mpi_cuda_exchange_Gcc(_phi);
gettimeofday(&te_mpi, 0);
etime_mpi += (te_mpi.tv_sec  - ts_mpi.tv_sec);
etime_mpi += (te_mpi.tv_usec - ts_mpi.tv_usec) * 1.e-6;

        // SpMV, A*phi
gettimeofday(&ts_up1, 0);
        PP_spmv_shared_load_noparts<<<blocks.Gcc.num_kn_s,blocks.Gcc.dim_kn_s>>>(_flag_u,
          _flag_v, _flag_w, _phi, _Apb_q);

        // update residual (r_q+1 = -b - (-A)*phi) and precond array
        PP_update_residual<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_r_q,
          _rhs_p, _Apb_q, _z_q, _invM);
cudaDeviceSynchronize();
gettimeofday(&te_up1, 0);
etime_up1 += (te_up1.tv_sec  - ts_up1.tv_sec);
etime_up1 += (te_up1.tv_usec - ts_up1.tv_usec) * 1.e-6;

      } else { // update normally
gettimeofday(&ts_up1, 0);
        PP_update_soln_resid<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(
          _phi, _p_q, _r_q, _Apb_q, _z_q, alpha, _invM);
cudaDeviceSynchronize();
gettimeofday(&te_up1, 0);
etime_up1 += (te_up1.tv_sec  - ts_up1.tv_sec);
etime_up1 += (te_up1.tv_usec - ts_up1.tv_usec) * 1.e-6;
      }

      /* Scalar product of new residual */
gettimeofday(&ts_ip2, 0);
      sp_rq1 = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_z_q, 0.);
gettimeofday(&te_ip2, 0);
etime_ip2 += (te_ip2.tv_sec  - ts_ip2.tv_sec);
etime_ip2 += (te_ip2.tv_usec - ts_ip2.tv_usec) * 1.e-6;

gettimeofday(&ts_AR2, 0);
      MPI_Allreduce(MPI_IN_PLACE, &sp_rq1, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
gettimeofday(&te_AR2, 0);
etime_AR2 += (te_AR2.tv_sec  - ts_AR2.tv_sec);
etime_AR2 += (te_AR2.tv_usec - ts_AR2.tv_usec) * 1.e-6;

      /* Check convergence */
      if (sp_rq1 <= pp_residual*pp_residual*sp_rhs) {
        gettimeofday(&te, 0);
        etime = (te.tv_sec - ts.tv_sec);
        etime += (te.tv_usec - ts.tv_usec)*1e-6;  // seconds

        recorder_PP_timed(rname, q, sqrt(sp_rq1)/sqrt(sp_rhs), etime,
          etime_spmv, etime_ip1, etime_AR1, etime_up1, etime_ip2,
          etime_AR2, etime_up2, etime_mpi);
        //printf("N%d >> The PP equation converged in %d iterations\n", rank, q);
        break;

      /* Check if sp_rq1 is nan */
      } else if (isnan(sp_rq1)) {
        if (rank == 0) {
          printf("N%d >> The PP equation did not converge.\n", rank);
          printf("N%d >> The residual at iteration %d is nan (%lf).\n",
            rank, q, sqrt(sp_rq1));
        }
        exit(EXIT_FAILURE);

      /* Continue with CG iterations */
      } else {
        /* Beta = (r_q1, r_q1)/(r_q, r_q) */
        beta = sp_rq1/sp_rq;

        /* Update search direction */
gettimeofday(&ts_up2, 0);
        PP_update_search<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p_q, 
          _pb_q, _z_q, beta);
cudaDeviceSynchronize();
gettimeofday(&te_up2, 0);
etime_up2 += (te_up2.tv_sec  - ts_up2.tv_sec);
etime_up2 += (te_up2.tv_usec - ts_up2.tv_usec) * 1.e-6;

        /* Communicate _pb_q */
gettimeofday(&ts_mpi, 0);
        mpi_cuda_exchange_Gcc(_pb_q);
gettimeofday(&te_mpi, 0);
etime_mpi += (te_mpi.tv_sec  - ts_mpi.tv_sec);
etime_mpi += (te_mpi.tv_usec - ts_mpi.tv_usec) * 1.e-6;

        /* change sp_rq1 -> sq_rq */
        sp_rq = sp_rq1;
      }
    }

    /* Exit if q == pp_max_iter */
    if (q > pp_max_iter) {
      printf("N%d >> The pressure-Poisson equation did not converge.\n",
        rank);
      printf("N%d >> (rhs, rhs) is %e\n", rank, sp_rhs);
      printf("N%d >> Residual at iteration %d is %lf\n", rank, q,
        sqrt(sp_rq1)/sqrt(sp_rhs));

      exit(EXIT_FAILURE);
    }

    /* Copy phi -> phinoghost */
    //copy_p_p_noghost<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, _phi);

    ///* Subtract mean from solution */
    //thrust::device_ptr<real> t_phing(_phinoghost);
    //real phi_mean = thrust::reduce(t_phing, t_phing + dom[rank].Gcc.s3, 0.,
    //  thrust::plus<real>());
    //MPI_Allreduce(MPI_IN_PLACE, &phi_mean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
    //phi_mean /= (real) DOM.Gcc.s3;

    //PP_subtract_mean<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, 
    //  phi_mean);

    ///* Copy phinoghost -> phi */
    //copy_p_noghost_p<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, _phi);

  }
}

extern "C"
void cuda_PP_cg_noparts(void)
{
  // Timing
  struct timeval ts, te;
  gettimeofday(&ts, 0);

  /* Variables (all Gcc)
   *  * _rhs_p      -- s3b -- RHS at all global soln points in each subdom
   *  * _phi        -- s3b -- soln at all global soln pts
   *  * _phinoghost -- s3  -- soln at subdoms soln pts
   *  * _r_q        -- s3  -- residual at subdom soln pts
   *  * _z_q        -- s3  -- extra array for jacobi
   *  * _p_q        -- s3  -- search direction at subdom's soln points
   *  * _pb_q       -- s3b -- search direction at dom's soln points
   *  * _Apb_q      -- s3  -- result of SpMV at subdom's soln points
   */
  real sp_rhs;          // (rhs, rhs)
  real sp_rq;           // (r_q, r_q)
  real sp_rq1;          // (r_q+1, r_q+1)
  real alpha;           // step length
  real beta;            // gradient correction factor
  real DENOM;           // alpha = sp_rq/DENOM
  real etime;           // elapsed time

  /* Device pointers for thrust reductions */
  thrust::device_ptr<real> t_r_q(_r_q);
  thrust::device_ptr<real> t_p_q(_p_q);
  thrust::device_ptr<real> t_Apb_q(_Apb_q);
  thrust::device_ptr<real> t_rhs(_rhs_p);
  thrust::device_ptr<real> t_z_q(_z_q);

  /* Reset _rhs_p to zero so scalar product isn't affected by data in the ghost
      cells -- we don't want to double-count when we MPI-Reduce */
  cudaMemset(_rhs_p, 0., dom[rank].Gcc.s3b * sizeof(real));

  /* Calculate _rhs_p at each subdomain's soln points */
  PP_rhs<<<blocks.Gcc.num_kn_s, blocks.Gcc.dim_kn_s>>>(rho_f, _u_star, _v_star, _w_star,
  _rhs_p, dt);

  /* Calculate initial norm ||rhs|| */
  sp_rhs = thrust::inner_product(t_rhs, t_rhs + dom[rank].Gcc.s3b, t_rhs, 0.);
  MPI_Allreduce(MPI_IN_PLACE, &sp_rhs, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  /* Exchange RHS between subdomains */
  mpi_cuda_exchange_Gcc(_rhs_p);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Initial residual, search direction, and initial guess */
  PP_cg_init<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_r_q, _p_q, _pb_q, 
    _phi, _rhs_p, _z_q, _invM);

  /* Exchange search direction */
  mpi_cuda_exchange_Gcc(_pb_q);
  MPI_Barrier(MPI_COMM_WORLD);
  //PP_init_search_ghost<<<blocks.Gcc.num_knb, blocks.Gcc.dim_knb>>>(_pb_q,
  //  _rhs_p, _DOM, _bc);

  /* Initial residual scalar product */
  sp_rq = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_z_q, 0.);
  MPI_Allreduce(MPI_IN_PLACE, &sp_rq, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  /* Main CG loop */
  int q = 0;
  char rname[FILE_NAME_SIZE] = "solver_expd.rec";

  // Guess for tolerance. If it doesn't iterate, error will accumulate and
  // solver will run again
  real RHS_TOL = 1.e-8;

  if (sp_rhs < RHS_TOL*RHS_TOL) {         // Quit if RHS is small -- this is a guess!
    gettimeofday(&te, 0);
    etime = (te.tv_sec - ts.tv_sec);
    etime += (te.tv_usec - ts.tv_usec)*1.e-6;  // seconds

    recorder_PP(rname, q, 0., etime);
    if (rank == 0) 
      printf("N%d >> Norm of the rhs is less than %.1e, exiting solver\n", rank,
        RHS_TOL);

  } else {                        // Iterate
    while (q <= pp_max_iter) {
      ++q;

      /* alpha = (r_q, z_q) / (p_q, A*pb_q)
       * NUM:   sp_rq
       * DENOM: (p_q, A*pb_q) (SpMV)
       */

      /* Sparse Matrix-Vector Product */
      PP_spmv_shared_load_noparts<<<blocks.Gcc.num_kn_s,blocks.Gcc.dim_kn_s>>>(
        _flag_u, _flag_v, _flag_w, _pb_q, _Apb_q);

      DENOM=thrust::inner_product(t_p_q, t_p_q + dom[rank].Gcc.s3, t_Apb_q, 0.);
      MPI_Allreduce(MPI_IN_PLACE, &DENOM, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
      alpha = sp_rq/DENOM;

      /* Update solution and residual */
      if (q % 50 == 0) { // recalculate every 50 iterations
        // update solution
        PP_update_solution<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phi,
          _p_q, alpha);

        // exchange solution (needed for SpMv)
        mpi_cuda_exchange_Gcc(_phi);

        // SpMV, A*phi
        PP_spmv_shared_load_noparts<<<blocks.Gcc.num_kn_s,blocks.Gcc.dim_kn_s>>>(_flag_u,
          _flag_v, _flag_w, _phi, _Apb_q);

        // update residual (r_q+1 = -b - (-A)*phi) and precond array
        PP_update_residual<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_r_q,
          _rhs_p, _Apb_q, _z_q, _invM);

      } else { // update normally
        PP_update_soln_resid<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(
          _phi, _p_q, _r_q, _Apb_q, _z_q, alpha, _invM);
      }

      /* Scalar product of new residual */
      sp_rq1 = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_z_q, 0.);
      MPI_Allreduce(MPI_IN_PLACE, &sp_rq1, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

      /* Check convergence */
      if (sp_rq1 <= pp_residual*pp_residual*sp_rhs) {
        gettimeofday(&te, 0);
        etime = (te.tv_sec - ts.tv_sec);
        etime += (te.tv_usec - ts.tv_usec)*1e-6;  // seconds

        recorder_PP(rname, q, sqrt(sp_rq1)/sqrt(sp_rhs), etime);
        //if (rank == 0)
        //  printf("N%d >> The PP equation converged in %d iterations\n", rank, q);
        break;

      /* Check if sp_rq1 is nan */
      } else if (isnan(sp_rq1)) {
        if (rank == 0) {
          printf("N%d >> The PP equation did not converge.\n", rank);
          printf("N%d >> The residual at iteration %d is nan (%lf).\n",
            rank, q, sqrt(sp_rq1));
        }
        exit(EXIT_FAILURE);

      /* Continue with CG iterations */
      } else {
        /* Beta = (r_q1, r_q1)/(r_q, r_q) */
        beta = sp_rq1/sp_rq;

        /* Update search direction */
        PP_update_search<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p_q, 
          _pb_q, _z_q, beta);

        /* Communicate _pb_q */
        mpi_cuda_exchange_Gcc(_pb_q);

        /* change sp_rq1 -> sq_rq */
        sp_rq = sp_rq1;
      }
    }

    /* Exit if q == pp_max_iter */
    if (q > pp_max_iter) {
      printf("N%d >> The pressure-Poisson equation did not converge.\n",
        rank);
      printf("N%d >> (rhs, rhs) is %e\n", rank, sp_rhs);
      printf("N%d >> Residual at iteration %d is %lf\n", rank, q,
        sqrt(sp_rq1)/sqrt(sp_rhs));

      exit(EXIT_FAILURE);
    }

    ///* Copy phi -> phinoghost */
    //copy_p_p_noghost<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, _phi);

    ///* Subtract mean from solution */
    //thrust::device_ptr<real> t_phing(_phinoghost);
    //real phi_mean = thrust::reduce(t_phing, t_phing + dom[rank].Gcc.s3, 0.,
    //  thrust::plus<real>());
    //MPI_Allreduce(MPI_IN_PLACE, &phi_mean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
    //phi_mean /= (real) DOM.Gcc.s3;

    //PP_subtract_mean<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, 
    //  phi_mean);

    ///* Copy phinoghost -> phi */
    //copy_p_noghost_p<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phinoghost, _phi);

  }
}


// extern "C"
// void cuda_PP_bicgstab()
// {
//   /* BiCGstab Method for A*phi_0 = b
//    * r_ = rhs - A*phi_0   -- Initial residual
//    * rs_0 = r             -- Arbitrary
//    * p_q = r0             -- Initial search direction
//    * while (!converged)
//    *  alpha_j = (r_j, rs_0)/(A*p_j, rs_0)
//    *  s_j = r_j - alpha*(A*p_j)
//    *  w_j = (A*s_j, s_j)/(A*sj, A*sj)
//    *  phi_j+1 = phi_j + alpha*p_j + w_j*s_j
//    *  r_j+1 = s_j - w_j*(A*s_j)
//    *  beta = (r_j+1, rs_0)/(r_j, rs_0) * alpha/w_j
//    *  p_j+1 = r_j+1 + beta*(p_j - w_j*(A*p_j))
//    */
// 
//   /* Variables:
//    *  * _rhs_p -- dom.Gcc.s3b -- rhs at all global soln points in each subdom
//    *  * _phi   -- dom.Gcc.s3  -- solution at each subdoms' soln points
//    *  * _r_q   -- dom.Gcc.s3  -- residual at each subdoms' soln points
//    *  * _rs_0  -- dom.Gcc.s3  -- initial residual at each subdom's soln points
//    *  * _p_q   -- dom.Gcc.s3  -- search dir at each subdom's soln points
//    *  * _pb_q  -- dom.Gcc.s3b -- search dir at all subdom points
//    *  * _s_q   -- dom.Gcc.s3  -- 2nd search dir at each subdom's soln points
//    *  * _sb_q  -- dom.Gcc.s3b -- 2nd search dir at all subdom points
//    *  * _Apb_q -- dom.Gcc.s3  -- result of SpMV at each subdom's soln points
//    *  * _Asb_q -- dom.Gcc.s3  -- result of SpMv at each subdom's soln points
//    */
//   real sp_rhs;                  // (_rhs_p, _rhs_p)
//   real norm_r0;                 // norm of initial residual
//   real sp_rq_rs0;               // (rs_0, r_q)
//   real sp_rqp1_rs0;             // (rs_0, r_qp1)
//   real sp_rqp1_rqp1;            // (r_qp1, r_qp1)
//   real alpha, beta, omega;      // Descent magnitudes
//   real sp_Apbq_rs0;             // (Apb_q, rs_0)
//   real sp_Asbq_sq;              // (Asb_q, s_q)
//   real sp_Asbq_Asbq;            // (Asb_q, Asb_q)
//   thrust::device_ptr<real> t_phi(_phi);
//   thrust::device_ptr<real> t_rhs(_rhs_p);
//   thrust::device_ptr<real> t_r_q(_r_q);
//   thrust::device_ptr<real> t_s_q(_s_q);
//   thrust::device_ptr<real> t_rs_0(_rs_0);
//   thrust::device_ptr<real> t_Apb_q(_Apb_q);
//   thrust::device_ptr<real> t_Asb_q(_Asb_q);
// 
// 
//   /* Calculate _rhs_p at each subdomain's solution points */
//   PP_rhs<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(rho_f, _u_star, _v_star, _w_star,
//   _rhs_p, dt);
// 
//   /* Calculate initial norm, ||rhs|| */
//   sp_rhs = thrust::inner_product(t_rhs, t_rhs + dom[rank].Gcc.s3b, t_rhs, 0.);
//   MPI_Allreduce(MPI_IN_PLACE, &sp_rhs, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
//   norm_r0 = sqrt(sp_rhs); 
// 
//   /* Exchange _rhs_p boundaries between subdomains (fill in exchange cells) */
//   mpi_cuda_exchange_Gcc(_rhs_p);
//   MPI_Barrier(MPI_COMM_WORLD);
// 
//   /* Init residual, search direction, and guess */
//   PP_bcgs_init<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_r_q, _rs_0, _p_q, 
//     _pb_q,  _phi, _rhs_p);
//   mpi_cuda_exchange_Gcc(_pb_q); 
//   MPI_Barrier(MPI_COMM_WORLD);
// 
//   /* Init ||residual|| */
//  sp_rq_rs0 = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_rs_0, 0.);
//  MPI_Allreduce(MPI_IN_PLACE, &sp_rq_rs0, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
// 
//   /* Main BiCGstab loop */
//   int q = 0;
//   while (q <= pp_max_iter) {
//     ++q;
// 
//     /* Step in search direction: alpha = (r_q, rs_0) / (A*pb_q, rs_0) */
//     PP_spmv_shared<<<blocks.Gcc.num_kn_s, blocks.Gcc.dim_kn_s>>>(_flag_u,
//       _flag_v, _flag_w, _pb_q, _Apb_q);
//     sp_Apbq_rs0 = thrust::inner_product(t_Apb_q, t_Apb_q + dom[rank].Gcc.s3, t_rs_0,
//       0.);
//     MPI_Allreduce(MPI_IN_PLACE, &sp_Apbq_rs0, 1, mpi_real, MPI_SUM, 
//       MPI_COMM_WORLD);
// 
//     alpha = sp_rq_rs0 / sp_Apbq_rs0;
// //printf("N%d >> q%02d >> alpha = %.25lf\n", rank, q, alpha);    
// 
//     /* Second search direction: s_q = r_q - alpha*(A*pb_q) */
// //real sq_sq = thrust::inner_product(t_s_q, t_s_q + dom[rank].Gcc.s3, t_s_q, 0.);
// //printf("N%d >> q%02d >> pre calc sq_sq = %.25lf\n", rank, q, sq_sq);
// 
//     PP_find_second_search<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_s_q, 
//       _sb_q, _r_q, alpha, _Apb_q);
// 
// //sq_sq = thrust::inner_product(t_s_q, t_s_q + dom[rank].Gcc.s3, t_s_q, 0.);
// //printf("N%d >> q%02d >> pre sq_sq = %.25lf\n", rank, q, sq_sq);
// //MPI_Allreduce(MPI_IN_PLACE, &sq_sq, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
// //printf("N%d >> q%02d >> sq_sq = %.25lf\n", rank, q, sq_sq);
// 
//     mpi_cuda_exchange_Gcc(_sb_q);
//     MPI_Barrier(MPI_COMM_WORLD);
// 
//     /* Step in 2nd search direction: omega= (A*sb_q, s_q) / (A*sb_q, A*sb_q) */
//     PP_spmv_shared<<<blocks.Gcc.num_kn_s, blocks.Gcc.dim_kn_s>>>(_flag_u, 
//       _flag_v, _flag_w, _sb_q, _Asb_q);
// 
//     sp_Asbq_sq = thrust::inner_product(t_Asb_q, t_Asb_q + dom[rank].Gcc.s3, t_s_q,
//       0.);
// //printf("N%d >> q%02d >> pre mpi sp_Asbq_sq = %.25lf\n", rank, q, sp_Asbq_sq);
//     MPI_Allreduce(MPI_IN_PLACE, &sp_Asbq_sq, 1, mpi_real, MPI_SUM, 
//       MPI_COMM_WORLD);
// 
//     sp_Asbq_Asbq = thrust::inner_product(t_Asb_q, t_Asb_q + dom[rank].Gcc.s3, 
//       t_Asb_q, 0.);
// //printf("N%d >> q%02d >> pre mpi sp_Asbq_Asbq = %.25lf\n", rank, q, sp_Asbq_Asbq);
//     MPI_Allreduce(MPI_IN_PLACE, &sp_Asbq_Asbq, 1, mpi_real, MPI_SUM, 
//       MPI_COMM_WORLD);
// 
// //printf("N%d >> q%02d >> sp_Asbq_sq = %.25lf\n", rank, q, sp_Asbq_sq);
// //printf("N%d >> q%02d >> sp_Asbq_Asbq = %.25lf\n", rank, q, sp_Asbq_Asbq);
// 
//     omega = sp_Asbq_sq / sp_Asbq_Asbq;
// //printf("N%d >> q%02d >> omega = %.25lf\n", rank, q, omega);    
// 
//     /* Update solution and residual:
//         phi_qp1 = phi_q + alpha*p_q + omega*s_q 
//         r_qp1 = s_q - omega*(A*sb_q)        */
//     PP_bcgs_update_soln_resid<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phi,
//       alpha, _p_q, omega, _s_q, _r_q, _Asb_q);
// 
//     /* Norm of new residual */
//     sp_rqp1_rqp1 = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_r_q, 0.);
//     MPI_Allreduce(MPI_IN_PLACE, &sp_rqp1_rqp1, 1, mpi_real, MPI_SUM, 
//       MPI_COMM_WORLD);
// 
// //printf("N%d >> q%02d >> sp_rqp1_rqp1 = %.25lf\n", rank, q, sp_rqp1_rqp1);    
// 
// //if (rank == 0) {
// //  printf("N%d >> q%02.d >> alpha = %.4e, omega = %.5e, sqrt(rqp1, rs0) = %.5e, resid = %.3e\n",
// //    rank, q, alpha, omega, sqrt(sp_rqp1_rqp1), sqrt(sp_rqp1_rqp1)/norm_r0); 
// //}
// 
//     /* if converged, break while */ 
//     if (sqrt(sp_rqp1_rqp1)/norm_r0 <= pp_residual) {
//       if (rank == 0) {
//         printf("N%d >> The pressure Poisson equation converged in %d "
//           "iterations\n", rank, q);
//       }
//       break;
// 
//     /* if q > pp_max_iter, exit */
//     } else if (q == pp_max_iter) {
//       fprintf(stderr, "N%d >> The pressure Poisson equation failed to converge"
//         "\n", rank);
//       exit(EXIT_FAILURE);
// 
//     } else {
//     /* else, continue with BiCGstab */
// 
//       /* beta = (r_q1, rs_0)/(r_q, rs_0) * alpha/omega */
//       sp_rqp1_rs0 = thrust::inner_product(t_r_q, t_r_q + dom[rank].Gcc.s3, t_rs_0, 
//         0.);
//       MPI_Allreduce(MPI_IN_PLACE, &sp_rqp1_rs0, 1, mpi_real, MPI_SUM, 
//         MPI_COMM_WORLD);
// 
// //printf("N%d >> q%02d >> sp_rqp1_rs0 = %.25lf\n", rank, q, sp_rqp1_rs0);    
// 
//       beta = (sp_rqp1_rs0 / sp_rq_rs0) * (alpha / omega);
// 
// //printf("N%d >> q%02d >> beta = %.25lf\n", rank, q, beta);    
//       
//       /* Update search direction: p_qp1 = r_qp1 + beta*(p_q - omega*(Apb_q)) */
//       PP_bcgs_update_search<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p_q,
//         _pb_q, _r_q, beta, omega, _Apb_q);
//       mpi_cuda_exchange_Gcc(_pb_q);
//       MPI_Barrier(MPI_COMM_WORLD);
// 
//       /* Change sp_rqp1_rs0 -> sp_rq_rs0 */
//       sp_rq_rs0 = sp_rqp1_rs0;
// //if (rank == 0) printf("\n");
//     }
//   }
// 
//   /* Subtract mean from solution */
//   real phi_mean = thrust::reduce(t_phi, t_phi + dom[rank].Gcc.s3);
//   MPI_Allreduce(MPI_IN_PLACE, &phi_mean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
//   phi_mean /= DOM.Gcc.s3;
// 
//   PP_subtract_mean<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_phi, phi_mean);
// }
