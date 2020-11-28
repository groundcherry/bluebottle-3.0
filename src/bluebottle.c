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

#include "bluebottle.h"

/* Global variables defined after main() */

int main(int argc, char *argv[])
{
  /* Start MPI and parse slurm devices */
  mpi_startup(argc, argv);
  if (rank == 0) printf("N%d >> Running Bluebottle_%s...\n", rank, VERSION);

  /* Parse inputs, read domain input file and fill structure */
  parse_cmdline_args(argc, argv);
  domain_read_input();
  recorder_read_config();
  domain_fill();

  /* Allocate domain memory on host and device */
  cuda_dom_malloc_host();
  cuda_dom_malloc_dev();
  if (SCALAR >= 1) {
    cuda_scalar_malloc_host();
    cuda_scalar_malloc_dev();
  }

  /* Initialize cuda threads/blocks and MPI structures */
  cuda_blocks_init();
  mpi_dom_comm_init();

  /* Initialize fields from boundary and initial conditions */
  domain_init_fields();
  if (SCALAR >= 1) {
    scalar_init_fields();
  }

  /* Initialize particles and bins */
  parts_read_input();
  cuda_part_malloc_host();
  if (NPARTS > 0) {
    init_bins();
    mpi_parts_init();
  }

  /* Initialize output writers */
  #ifndef CGNS_OUTPUT
    if (rank == 0) {
      printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
      printf("WARNING: cgns output is turned off in Makefile!!!\n");
      printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    }
  #endif
  if (use_restart == 0) {
    #ifdef CGNS_OUTPUT
      cgns_recorder_init();
      vtk_recorder_init();
    #endif
    recorder_PP_init("solver_expd.rec");
    recorder_scalar_init("iter_etime.rec");
  }

  /* Deal with restart input */
  if (use_restart == 1) {
    in_restart();
    if (ttime >= duration) {
      printf("\n...simulation completed");
      restart_stop = 1;
      exit(EXIT_SUCCESS);
    }
  } else {
    parts_init();
    scalar_part_init();
  }

  /* Allocate particles on device */
  cuda_part_malloc_dev();
  cuda_part_push();
  count_mem();

  /* Fill ghost bins for first time step */
  if (NPARTS > 0) {
    cuda_transfer_parts_i();
    cuda_transfer_parts_j();
    cuda_transfer_parts_k();
  }

  /* Push initial fields to device and exchange initial condition */
  cuda_dom_push();
  mpi_cuda_exchange_Gcc(_p);
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);
  if (SCALAR >= 1) {
    cuda_scalar_push();
    mpi_cuda_exchange_Gcc(_s);
  }

  /* Build phase, phase_shell, and flag variables */
  cuda_build_cages();

  /* get initial dt; this could be an extra check for the SHEAR init */
  cuda_find_dt();

  /* Apply boundary conditions to field variables */
  cuda_dom_BC();
  mpi_cuda_exchange_Gcc(_p);
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);
  if (SCALAR >= 1) {
    cuda_scalar_BC(_s);
    cuda_scalar_BC(_s0);
    mpi_cuda_exchange_Gcc(_s);
    mpi_cuda_exchange_Gcc(_s0);
  }

  /* Initialize jacobi preconditioner */
  cuda_PP_init_jacobi_preconditioner();

  /* Initialize physalis quadrature and coeffs */
  cuda_init_physalis();

#ifdef TEST
  run_test();
#else // TEST - Run normally

  /* Before entering main loop, print device memory usage status of each device,
   * this exclude the overhead and could include memory allocated from other program */
//  printMemInfo();

  if (rank == 0) {
    printf("\n=====BLUEBOTTLE============================");
    printf("======================================\n");
    fflush(stdout);
  }

  /* Output initial fields */
  #ifdef CGNS_OUTPUT
    if (use_restart == 0) {
      recorder_out();
    }
  #endif

  /*****************************************************************/
  /** Begin the main timestepping loop in the experimental domain **/
  /*****************************************************************/
  while (ttime <= duration) {
    ttime += dt;
    rec_cgns_flow_ttime_out += dt;
    rec_cgns_part_ttime_out += dt;
    rec_vtk_ttime_out += dt;
    //rec_restart_ttime_out += dt;
    stepnum++;
    int iter = 0;             // Lamb's iteration counter
    int lambflag = 1;         // To continue or not after max iters
    real iter_err = FLT_MAX;  // Lamb's error
    int iter_num = -1;
    real mitertimestart = 0.;
    real mitertimestop = 0.;

    if (rank == 0) {
      printf("\nEXPD: Time = %e of %e (dt = %e) (stepnum = %d).\n", ttime,
        duration, dt, stepnum);
      fflush(stdout);
    }

    printMemInfo();

    /* Compute forcing and velocity boundary conditions */
    cuda_compute_forcing();
    if (SCALAR >= 1) {
      cuda_compute_boussinesq();
    }
    cuda_compute_turb_forcing();
    compute_vel_BC();

    /* Iterate for Lamb's coefficients */
    mitertimestart = MPI_Wtime();
    while (iter_err > lamb_residual) {
      if (rank == 0) printf("  Iteration %d: ", iter);
      //if (rank == 0) printf("  Iteration %d:\n", iter);
      //if (rank == 0) printf("N%d >> Iteration %d:\n", rank, iter);
      fflush(stdout);

      /* Solve for U_star */
      cuda_U_star();
      mpi_cuda_exchange_Gfx(_u_star);
      mpi_cuda_exchange_Gfy(_v_star);
      mpi_cuda_exchange_Gfz(_w_star);

      /* Apply BC to U_star */
      if (nparts > 0) cuda_part_BC_star();
      cuda_dom_BC_star();
      mpi_cuda_exchange_Gfx(_u_star);
      mpi_cuda_exchange_Gfy(_v_star);
      mpi_cuda_exchange_Gfz(_w_star);

      /* Enforce solvability */
      cuda_solvability();
      if (nparts > 0) cuda_part_BC_star();
      cuda_dom_BC_star();
      mpi_cuda_exchange_Gfx(_u_star);
      mpi_cuda_exchange_Gfy(_v_star);
      mpi_cuda_exchange_Gfz(_w_star);

      /* Solve for pressure */
      if (NPARTS > 0) {
        cuda_PP_cg();
      } else {
        cuda_PP_cg_noparts(); // For nopart testing
      }
      mpi_cuda_exchange_Gcc(_phi);
      cuda_dom_BC_p(_phi);

      /* Solve for U */
      cuda_project();
      mpi_cuda_exchange_Gfx(_u);
      mpi_cuda_exchange_Gfy(_v);
      mpi_cuda_exchange_Gfz(_w);

      /* Apply boundary conditions to fields */
      if (nparts > 0) cuda_part_BC();
      cuda_dom_BC();
      mpi_cuda_exchange_Gfx(_u);
      mpi_cuda_exchange_Gfy(_v);
      mpi_cuda_exchange_Gfz(_w);

      /* update pressure */
      cuda_update_p();
      if (nparts > 0) {
        cuda_part_BC();
        cuda_part_p_fill();
      }
      cuda_dom_BC_p(_p);
      mpi_cuda_exchange_Gcc(_p);

      /* Update Lamb's coefficients */
      if (NPARTS > 0) {
        cuda_move_parts_sub();
        cuda_lamb();
      }

      /* Calculate iteration error */
      cuda_lamb_err(&iter_err, &iter_num);

      //if (rank == 0) printf("N%d >> Iteration %d error = %f\n\n", rank, iter, iter_err);
      if (rank == 0) printf("  Error = %f(%d)\r", iter_err, iter_num);
      //if (rank == 0) printf("  Iteration %d error = %f\n\n", iter, iter_err);

      iter++;
      if (iter == lamb_max_iter) {
        //  lambflag = 0;
        break;
      }
    } // End Lamb's iterations
    mitertimestop = MPI_Wtime();

    // Deal with lamb's iterations convergence (or not)
    if (rank == 0) {
      if (iter < lamb_max_iter) {
        printf("  The Lamb's coefficients converged in %d iterations\n", iter);
      } else if (iter == lamb_max_iter) {
        if (lambflag == 1) {
          printf("  Reached the max number of Lamb's iterations. Continuing...\n");
        } else {
          printf("  Reached the max number of Lamb's iterations. Exiting...\n");
          exit(EXIT_FAILURE);
        }
      }
    }

    /***** Scalar inner iteration *****/
    int scalar_iter = 0;
    real scalar_iter_err = FLT_MAX;
    real sitertimestart = 0.;
    real sitertimestop = 0.;
    if (SCALAR >= 1) {

      sitertimestart = MPI_Wtime();
      while (scalar_iter_err > lamb_residual) {
        if (rank == 0) printf("  Iteration %d: ", scalar_iter);
        fflush(stdout);

        // apply b.c. on s0, domain and part
        cuda_scalar_BC(_s0);
        cuda_scalar_part_BC(_s0);
        mpi_cuda_exchange_Gcc(_s0);

        // integrate scalar convection-diffusion equation
        cuda_scalar_solve();
        mpi_cuda_exchange_Gcc(_s);

        // apply b.c. on s, domain and part
        cuda_scalar_BC(_s);
        cuda_scalar_part_fill();
        mpi_cuda_exchange_Gcc(_s);

        // update lamb's coefficients using the new field variable s
        cuda_scalar_lamb();

        scalar_iter_err = cuda_scalar_lamb_err();
        if (rank == 0) printf("  Error = %f\r", scalar_iter_err);
        fflush(stdout);
        scalar_iter++;

        if (scalar_iter == lamb_max_iter) {
          // allow simulation continues even if it reaches the max number
          break;
        }
      }
      sitertimestop = MPI_Wtime();

      if (rank == 0) {
        if (scalar_iter < lamb_max_iter) {
          printf("  The Scalar's coefficients converged in %d iterations\n", scalar_iter);
        } else if (scalar_iter == lamb_max_iter) {
          if (lambflag == 1) {
            printf("  Reached the max number of Lamb's iterations. Continuing...\n");
          } else {
            printf("  Reached the max number of Lamb's iterations. Exiting...\n");
            exit(EXIT_FAILURE);
          }
        }
      }

      // store and update
      cuda_store_s();
      cuda_scalar_update_part();
    }

    if (NPARTS > 0) {
      /* Update particle forces and velocity */
      cuda_move_parts_sub();
      cuda_update_part_velocity();

      /* Stamp particle internals equal to rigid body velocity */
      cuda_parts_internal();
      cuda_dom_BC();
      mpi_cuda_exchange_Gfx(_u);
      mpi_cuda_exchange_Gfy(_v);
      mpi_cuda_exchange_Gfz(_w);
    }

    /* Store flow variables for next timestep */
    cuda_store_u();

    /* Compute next time step size */
    dt0 = dt;
    cuda_find_dt();

    /* Record info */
    recorder_scalar("iter_etime.rec", ttime, iter, mitertimestop-mitertimestart,
      iter_err, scalar_iter, sitertimestop-sitertimestart, scalar_iter_err);

    /* Output */
    #ifdef CGNS_OUTPUT
      recorder_out();
    #endif

    /* Move particles */
    if (NPARTS > 0) {
      cuda_update_part_position();
      cuda_transfer_parts_i();
      cuda_transfer_parts_j();
      cuda_transfer_parts_k();
    }

    /* Rebuild cages with new positions */
    cuda_build_cages();

    /* Re-init jacobi preconditioner given new particle positions */
    if (NPARTS > 0) {
      cuda_PP_init_jacobi_preconditioner();
    }

    /* Output restart file */
    if (restart_recorder_write()) break;

    /* Check for blow up */
    if (dt < 1.e-10 || dt > 1.e10) {
      printf("N%d >> The solution time step has diverged. Ending simulation.\n",
        rank);
      exit(EXIT_FAILURE);
    }
    fflush(stdout);
  }

  if (rank == 0) {
    if (ttime > duration) {
      printf("\nEXPD: The simulation has reached its specified duration\n");
    }
    printf("========================================");
    printf("========================================\n\n");
    fflush(stdout);
  }
#endif // TEST

  /* Free all cuda-allocated memory (device and pinned on host) */
  if (SCALAR >= 1) {
    cuda_scalar_free();
  }
  cuda_part_free();
  cuda_dom_free();

  /* Free MPI Memory */
  mpi_free();

  /* Free host memory */
  domain_free();
  part_free();

  /* End MPI */
  mpi_end();
  if (rank == 0) printf("N%d >> Bluebottle_%s Complete\n", rank, VERSION);
}

/* Define global variables declared in bluebottle.h */
// Structures
dom_struct DOM;
dom_struct *_DOM;
dom_struct *dom;
BC bc;
BC *_bc;
gradP_struct gradP;
g_struct g;

// Simulation parameters
real dt;
real dt0;
real CFL;
real duration;
real ttime;
int stepnum;

real rho_f;
real mu;
real nu;
real pp_residual;
int pp_max_iter;
int lamb_max_iter;
real lamb_residual;
real lamb_relax;
real lamb_cut;
real osci_f;

int init_cond;
int out_plane;
real v_bc_tdelay;
real p_bc_tdelay;
real g_bc_tdelay;
real pid_int;
real pid_back;
real Kp;
real Ki;
real Kd;

real turbA;
real turbl;
real turb_k0;

// Flow solver + projection method
real *p, *u, *v, *w;
real *p0, *u0, *v0, *w0;
real *phi;
real *u_star, *v_star, *w_star;
real *f_x, *f_y, *f_z;
real *conv_u, *conv_v, *conv_w;
real *conv0_u, *conv0_v,*conv0_w;
real *diff_u, *diff_v, *diff_w;
real *diff0_u, *diff0_v, *diff0_w;

real *_p, *_u, *_v, *_w;
real *_p0, *_u0, *_v0, *_w0;
real *_phi;
real *_phinoghost;
real *_invM;
real *_u_star, *_v_star, *_w_star;
real *_f_x, *_f_y, *_f_z;
real *_conv_u, *_conv_v, *_conv_w;
real *_conv0_u, *_conv0_v,*_conv0_w;
real *_diff_u, *_diff_v, *_diff_w;
real *_diff0_u, *_diff0_v, *_diff0_w;

// Poisson solver
real *_rhs_p;
real *_z_q;
real *_r_q;
real *_p_q;
real *_pb_q;
real *_Apb_q;
//real *_rs_0;  // only for BCG
//real *_s_q; // only for BCG
//real *_sb_q;  // only for BcG
//real *_Asb_q; // only for BCG

// MPI exchange arrays
real *_send_Gcc_e;  // send_Gcc_east
real *_send_Gcc_w;
real *_send_Gcc_n;
real *_send_Gcc_s;
real *_send_Gcc_t;
real *_send_Gcc_b;

real *_send_Gfx_e;  // send_Gfx_east
real *_send_Gfx_w;
real *_send_Gfx_n;
real *_send_Gfx_s;
real *_send_Gfx_t;
real *_send_Gfx_b;

real *_send_Gfy_e;  // send_Gfy_east
real *_send_Gfy_w;
real *_send_Gfy_n;
real *_send_Gfy_s;
real *_send_Gfy_t;
real *_send_Gfy_b;

real *_send_Gfz_e;  // send_Gfz_east
real *_send_Gfz_w;
real *_send_Gfz_n;
real *_send_Gfz_s;
real *_send_Gfz_t;
real *_send_Gfz_b;

real *_recv_Gcc_e;  // recv_Gcc_east
real *_recv_Gcc_w;
real *_recv_Gcc_n;
real *_recv_Gcc_s;
real *_recv_Gcc_t;
real *_recv_Gcc_b;

real *_recv_Gfx_e;  // recv_Gfx_east
real *_recv_Gfx_w;
real *_recv_Gfx_n;
real *_recv_Gfx_s;
real *_recv_Gfx_t;
real *_recv_Gfx_b;

real *_recv_Gfy_e;  // recv_Gfy_east
real *_recv_Gfy_w;
real *_recv_Gfy_n;
real *_recv_Gfy_s;
real *_recv_Gfy_t;
real *_recv_Gfy_b;

real *_recv_Gfz_e;  // recv_Gfz_east
real *_recv_Gfz_w;
real *_recv_Gfz_n;
real *_recv_Gfz_s;
real *_recv_Gfz_t;
real *_recv_Gfz_b;

// Extra 
long int cpumem;
long int gpumem;
