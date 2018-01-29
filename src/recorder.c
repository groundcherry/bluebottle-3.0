/*******************************************************************************
 ******************************** BLUEBOTTLE ***********************************
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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "recorder.h"

real rec_cgns_flow_dt;
real rec_cgns_part_dt;
real rec_vtk_dt;
real rec_cgns_flow_ttime_out;
real rec_cgns_part_ttime_out;
real rec_vtk_ttime_out;
int rec_vtk_stepnum_out;
real rec_restart_dt;
int rec_restart_complete;
//real rec_restart_ttime_out;
int restart_stop;
time_t startwalltime;

void recorder_read_config(void)
{
  int fret = 0;
  fret = fret;  // prevent compiler warning

  /* open config file for reading */
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/%s/record.config", ROOT_DIR, INPUT_DIR);
  FILE *infile = fopen(fname, "r");
  if (infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  rec_cgns_flow_dt = -1.;
  rec_cgns_part_dt = -1.;
  rec_vtk_dt = -1.;
  rec_restart_dt = -1.;
  rec_restart_complete = -1;
  restart_stop = 0;
  //rec_particle_dt = -1.;
  //rec_prec_dt = -1.;

  #ifdef DOUBLE
  fret = fscanf(infile, "=== OUTPUT FORMATS ===\n");
  fret = fscanf(infile, "CGNS_FLOW_FIELD %lf\n\n", &rec_cgns_flow_dt);
  fret = fscanf(infile, "CGNS_PARTICLES %lf\n\n", &rec_cgns_part_dt);
  fret = fscanf(infile, "VTK_FLOW_FIELD %lf\n", &rec_vtk_dt);
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "=== RESTART CAPABILITY ===\n");
  fret = fscanf(infile, "WRITE_RESTART_MINUTES %lf\n", &rec_restart_dt);
  fret = fscanf(infile, "WRITE_RESTART_COMPLETE %d\n", &rec_restart_complete);
  fret = fscanf(infile, "RESTART_STOP %d\n", &restart_stop);
  #else // single precision
  fret = fscanf(infile, "=== OUTPUT FORMAT ===\n");
  fret = fscanf(infile, "CGNS_FLOW_FIELD %f\n\n", &rec_cgns_flow_dt);
  fret = fscanf(infile, "CGNS_PARTICLES %f\n\n", &rec_cgns_part_dt);
  fret = fscanf(infile, "VTK_FLOW_FIELD %f\n", &rec_vtk_dt);
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "=== RESTART CAPABILITY ===\n");
  fret = fscanf(infile, "WRITE_RESTART_MINUTES %f\n", &rec_restart_dt);
  fret = fscanf(infile, "WRITE_RESTART_COMPLETE %d\n", &rec_restart_complete);
  fret = fscanf(infile, "RESTART_STOP %d\n", &restart_stop);
  #endif // DOUBLE

  fclose(infile);
}

void recorder_out(void)
{
  // Set up write booleans for clarity
  int write_cgns_flow = (rec_cgns_flow_dt > 0);
  int time_to_write_cgns_flow = (rec_cgns_flow_ttime_out >= rec_cgns_flow_dt ||
                                  stepnum == 0);

  int write_vtk = (rec_vtk_dt > 0);
  int time_to_write_vtk = (rec_vtk_ttime_out >= rec_vtk_dt || stepnum == 0);

  int write_cgns_part = (rec_cgns_part_dt > 0);
  int time_to_write_cgns_part = (rec_cgns_part_ttime_out >= rec_cgns_part_dt ||
                                  stepnum == 0);

  // Pull flow data if necessary
  if ((write_cgns_flow && time_to_write_cgns_flow) ||
      (write_vtk && time_to_write_vtk)) {

    // Pull domain
    cuda_dom_pull();
    cuda_dom_pull_phase();
    #ifdef DDEBUG // pull more information
      cuda_dom_pull_debug();
    #endif // DDEBUG
    if (NPARTS > 0) {
      cuda_part_pull(); // Because we need to map phase[p] to parts[p].N
    }
  
    // Write (more checks take place inside these functions)
    #ifdef CGNS_OUTPUT
      cgns_recorder_flow_write();
    #endif
    vtk_recorder_write(); // NOTE: particles are always written with vtk output
                          // and cuda_part_pull is called here
  }

  // Pull particle data if necessary
  if (write_cgns_part && time_to_write_cgns_part && (NPARTS > 0)) {
    #ifndef DDEBUG
      //cuda_dom_pull();
      //cuda_dom_pull_debug();
      //cuda_dom_pull_phase();
      cuda_part_pull();       // for central particles
    #else
      cuda_part_pull();
      //cuda_part_pull_debug(); // for all particles; not implemented
    #endif // DDEBUG

    #ifdef CGNS_OUTPUT
      cgns_recorder_part_write();
    #endif
  }

  // Output dudy at north and south walls if using RANDOM (turbulent channel)
  // input
  if (init_cond == TURB_CHANNEL) {
    cuda_wall_shear_stress();
  }

}

void recorder_PP_init(char *name)
{
  // create the file
  if (rank == 0) {
    // Create output directory if it doesn't exist
    struct stat st = {0};
    char buf[CHAR_BUF_SIZE];
    sprintf(buf, "%s/record", ROOT_DIR);
    if (stat(buf, &st) == -1) {
      mkdir(buf, 0700);
    }


    char path[FILE_NAME_SIZE] = "";
    sprintf(path, "%s/record/%s", ROOT_DIR, name);
    FILE *rec = fopen(path, "w");
    if(rec == NULL) {
      fprintf(stderr, "Could not open file %s\n", name);
      exit(EXIT_FAILURE);
    }

    fprintf(rec, "%-12s", "stepnum");
    fprintf(rec, "%-15s", "ttime");
    fprintf(rec, "%-15s", "dt");
    fprintf(rec, "%-8s", "niter");
    fprintf(rec, "%-15s", "resid");
    fprintf(rec, "%-15s", "time (s)");

    // close the file
    fclose(rec);
  }
}

void recorder_PP(char *name, int niter, real resid, real etime)
{
  // average over procs
  MPI_Allreduce(MPI_IN_PLACE, &etime, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  etime /= nprocs;

  // open the file
  if (rank == 0) {
    char path[FILE_NAME_SIZE] = "";
    sprintf(path, "%s/record/%s", ROOT_DIR, name);
    FILE *rec = fopen(path, "r+");
    if(rec == NULL) {
      recorder_PP_init(name);
      rec = fopen(path, "r+");
    }

    // move to the end of the file
    fseek(rec, 0, SEEK_END);

    fprintf(rec, "\n");
    fprintf(rec, "%-12d", stepnum);
    fprintf(rec, "%-15e", ttime);
    fprintf(rec, "%-15e", dt);
    fprintf(rec, "%-8d", niter);
    fprintf(rec, "%-15e", resid);

    fprintf(rec, "%-15e", etime);

    // close the file
    fclose(rec);
  }
}

void recorder_PP_init_timed(char *name)
{
  // create the file
  if (rank == 0) {
    // Create output directory if it doesn't exist
    struct stat st = {0};
    char buf[CHAR_BUF_SIZE];
    sprintf(buf, "%s/record", ROOT_DIR);
    if (stat(buf, &st) == -1) {
      mkdir(buf, 0700);
    }


    char path[FILE_NAME_SIZE] = "";
    sprintf(path, "%s/record/%s", ROOT_DIR, name);
    FILE *rec = fopen(path, "w");
    if(rec == NULL) {
      fprintf(stderr, "Could not open file %s\n", name);
      exit(EXIT_FAILURE);
    }

    fprintf(rec, "%-12s", "stepnum");
    fprintf(rec, "%-15s", "ttime");
    fprintf(rec, "%-15s", "dt");
    fprintf(rec, "%-8s", "niter");
    fprintf(rec, "%-15s", "resid");
    fprintf(rec, "%-16s", "Total time (s)");
    fprintf(rec, "%-16s", "spmv time (s)"); // PP_spmv_shared_load
    fprintf(rec, "%-16s", "ip1 time (s)");  // thrust::inner_product
    fprintf(rec, "%-16s", "ar1 time (s)");  // MPI_Allreduce
    fprintf(rec, "%-16s", "up1 time (s)");  // PP_update_soln_resid
    fprintf(rec, "%-16s", "ip2 time (s)");  // thurst::inner_product
    fprintf(rec, "%-16s", "ar2 time (s)");  // MPI_Allreduce
    fprintf(rec, "%-16s", "up2 time (s)");  // PP_update_search
    fprintf(rec, "%-16s", "mpi time (s)");  // mpi_cuda_exchange_Gcc

    // close the file
    fclose(rec);
  }
}

void recorder_PP_timed(char *name, int niter, real resid, real etime,
  real etime_spmv,    // PP_spmv_shared_load
  real etime_ip1,     // thrust::inner_product
  real etime_AR1,     // MPI_Allreduce
  real etime_up1,     // PP_update_soln_resid
  real etime_ip2,     // thurst::inner_product
  real etime_AR2,     // MPI_Allreduce
  real etime_up2,     // PP_update_search
  real etime_mpi)     // mpi_cuda_exchange_Gcc
{
  // average over procs
  real inprocs = 1. / nprocs;
  MPI_Allreduce(MPI_IN_PLACE, &etime, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &etime_spmv, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &etime_ip1, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &etime_AR1, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &etime_up1, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &etime_ip2, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &etime_AR2, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &etime_up2, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &etime_mpi, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  etime *= inprocs;
  etime_spmv *= inprocs;
  etime_ip1 *= inprocs;
  etime_AR1 *= inprocs;
  etime_up1 *= inprocs;
  etime_ip2 *= inprocs;
  etime_AR2 *= inprocs;
  etime_up2 *= inprocs;
  etime_mpi *= inprocs;

  // open the file
  if (rank == 0) {
    char path[FILE_NAME_SIZE] = "";
    sprintf(path, "%s/record/%s", ROOT_DIR, name);
    FILE *rec = fopen(path, "r+");
    if(rec == NULL) {
      recorder_PP_init_timed(name);
      rec = fopen(path, "r+");
    }

    // move to the end of the file
    fseek(rec, 0, SEEK_END);

    fprintf(rec, "\n");
    fprintf(rec, "%-12d", stepnum);
    fprintf(rec, "%-15e", ttime);
    fprintf(rec, "%-15e", dt);
    fprintf(rec, "%-8d", niter);
    fprintf(rec, "%-15e", resid);
    fprintf(rec, "%-16e", etime);
    fprintf(rec, "%-16e", etime_spmv); // PP_spmv_shared_load
    fprintf(rec, "%-16e", etime_ip1);  // thrust::inner_product
    fprintf(rec, "%-16e", etime_AR1);  // MPI_Allreduce
    fprintf(rec, "%-16e", etime_up1);  // PP_update_soln_resid
    fprintf(rec, "%-16e", etime_ip2);  // thurst::inner_product
    fprintf(rec, "%-16e", etime_AR2);  // MPI_Allreduce
    fprintf(rec, "%-16e", etime_up2);  // PP_update_search
    fprintf(rec, "%-16e", etime_mpi);  // mpi_cuda_exchange_Gcc

    //real comp = etime_spmv + etime_ip1 + etime_up1 + etime_ip2 + etime_up2;
    //real comm = etime_AR1 + etime_AR2 + etime_mpi;
    //real ratio = comp / (comp + comm);
    //printf("Total elapsed = %.2e (s); total comp = %.2e (s); total comm  = %.2e; rat = %.3f\n",
    //  etime,
    //  comp,
    //  comm,
    //  ratio);

    // close the file
    fclose(rec);
  }
}

int restart_recorder_write(void)
{
  // Get elapsed walltime since start
  real timestepwalltime = MPI_Wtime();
  real diffwalltime = difftime(timestepwalltime, startwalltime);

  // Average diff over all the proces
  MPI_Allreduce(MPI_IN_PLACE, &diffwalltime, 1, mpi_real, MPI_SUM,
    MPI_COMM_WORLD);
  diffwalltime /= nprocs;

  // Write restart file if necessary
  if ((rec_restart_dt > 0) && (diffwalltime/60. > rec_restart_dt)) {
    // 1) If the wall time has reached the time we specified in record.config
    cuda_dom_pull(); 
    cuda_dom_pull_phase();
    cuda_dom_pull_debug();
    cuda_dom_pull_restart();
    cuda_part_pull();

    printf("N%d >> Writing restart file (reached requested wall time) (t = %e)...\n",
      rank, ttime);
    out_restart();

    if (restart_stop) return 1; // Exit after write

  } else if ((rec_restart_complete > 0) && (ttime > duration)) {
    // 2) If the simulation has completed
    cuda_dom_pull(); 
    cuda_dom_pull_phase();
    cuda_dom_pull_debug();
    cuda_dom_pull_restart();
    cuda_part_pull();

    printf("N%d >> Writing restart file (sim completed) (t = %e)...\n", rank,
      ttime);
    out_restart();

    if (restart_stop) return 1; // Exit after write

  }
  return 0; // Nothing happened
}

void recorder_turb_init(char *name)
{
  // create the file
  if (rank == 0) {
    // Create output directory if it doesn't exist
    struct stat st = {0};
    char buf[CHAR_BUF_SIZE];
    sprintf(buf, "%s/record", ROOT_DIR);
    if (stat(buf, &st) == -1) {
      mkdir(buf, 0700);
    }


    char path[FILE_NAME_SIZE] = "";
    sprintf(path, "%s/record/%s", ROOT_DIR, name);
    FILE *rec = fopen(path, "w");
    if(rec == NULL) {
      fprintf(stderr, "Could not open file %s\n", name);
      exit(EXIT_FAILURE);
    }

    fprintf(rec, "turbA %e\n", turbA);
    fprintf(rec, "turbl %e\n", turbl);
    fprintf(rec, "nu %e\n", nu);

    fprintf(rec, "%-12s", "stepnum");
    fprintf(rec, "%-15s", "ttime");
    fprintf(rec, "%-15s", "tke");
    fprintf(rec, "%-15s", "dissipation");
    fprintf(rec, "%-15s", "eddy_time");
    fprintf(rec, "%-15s", "integral_length");

    // close the file
    fclose(rec);
  }
}

void recorder_turb(char *name, real tke, real dissipation)
{
  // Calculate eddy turnover time and integral length scale
  real eddy_time = tke / dissipation; 
  real integral_length = pow(2./3. * tke, 1.5) / dissipation;

  // open the file
  if (rank == 0) {
    char path[FILE_NAME_SIZE] = "";
    sprintf(path, "%s/record/%s", ROOT_DIR, name);
    FILE *rec = fopen(path, "r+");
    if(rec == NULL) {
      recorder_PP_init(name);
      rec = fopen(path, "r+");
    }

    // move to the end of the file
    fseek(rec, 0, SEEK_END);

    fprintf(rec, "\n");
    fprintf(rec, "%-12d", stepnum);
    fprintf(rec, "%-15e", ttime);
    fprintf(rec, "%-15e", tke);
    fprintf(rec, "%-15e", dissipation);
    fprintf(rec, "%-15e", eddy_time);
    fprintf(rec, "%-15e", integral_length);


    // close the file
    fclose(rec);
  }
}
