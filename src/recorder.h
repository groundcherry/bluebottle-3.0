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

/****h* Bluebottle/recorder
 * NAME
 *  recorder
 * FUNCTION
 *  A utility for recording simulation metrics.
 ******
 */

#ifndef _RECORDER_H
#define _RECORDER_H

#include "bluebottle.h"

/*** VARIABLES ***/
/****v* bluebottle/rec_cgns_flow_dt
 * NAME
 *  rec_cgns_flow_dt
 * TYPE
 */
extern real rec_cgns_flow_dt;
/*
 * PURPOSE
 *  Recorder cgns flow field output timestep size
 ******
 */

/****v* bluebottle/rec_cgns_part_dt
 * NAME
 *  rec_cgns_part_dt
 * TYPE
 */
extern real rec_cgns_part_dt;
/*
 * PURPOSE
 *  Recorder cgns particle output timestep size
 ******
 */

/****v* bluebottle/rec_vtk_dt
 * NAME
 *  rec_vtk_dt
 * TYPE
 */
extern real rec_vtk_dt;
/*
 * PURPOSE
 *  Recorder vtk output timestep size
 ******
 */

/****v* bluebottle/rec_cgns_flow_ttime_out
 * NAME
 *  rec_cgns_flow_ttime_out
 * TYPE
 */
extern real rec_cgns_flow_ttime_out;
/*
 * PURPOSE
 *  Recorder cgns flow field output time since last output
 ******
 */

/****v* bluebottle/rec_cgns_part_ttime_out
 * NAME
 *  rec_cgns_part_ttime_out
 * TYPE
 */
extern real rec_cgns_part_ttime_out;
/*
 * PURPOSE
 *  Recorder cgns part output time since last output
 ******
 */

/****v* bluebottle/rec_vtk_ttime_out
 * NAME
 *  rec_vtk_ttime_out
 * TYPE
 */
extern real rec_vtk_ttime_out;
/*
 * PURPOSE
 *  Recorder vtk output time since last output
 ******
 */

/****v* bluebottle/rec_vtk_stepnum_out
 * NAME
 *  rec_vtk_stepnum_out
 * TYPE
 */
extern int rec_vtk_stepnum_out;
/*
 * PURPOSE
 *  The output timestep number for the simulation.  The initial configuration
 *  is given by stepnum = 0.
 ******
 */

/****v* bluebottle/rec_restart_dt
 * NAME
 *  rec_restart_dt
 * TYPE
 */
extern real rec_restart_dt;
/*
 * PURPOSE
 *  Recorder restart output timestep size (minutes)
 ******
 */

/****v* bluebottle/rec_restart_complete
 * NAME
 *  rec_restart_complete
 * TYPE
 */
extern int rec_restart_complete;
/*
 * PURPOSE
 *  Recorder restart output if simulation has ended. Write if > 0
 ******
 */

/****v* bluebottle/rec_restart_ttime_out
 * NAME
 *  rec_restart_dt
 * TYPE
 */
//extern real rec_restart_ttime_out;
/*
 * PURPOSE
 *  Recorder restart time since last output
 ******
 */

/****v* bluebottle/restart_stop
 * NAME
 *  restart_stop
 * TYPE
 */
extern int restart_stop;
/*
 * PURPOSE
 *  Recorder restart and stop output boolean. 0 = continue, 1 = stop
 ******
 */

/****v* bluebottle/startwalltime
 * NAME
 *  startwalltime
 * TYPE
 */
extern time_t startwalltime;
/*
 * PURPOSE
 *  Actual time start of simulatoin
 ******
 */


/*** FUNCTIONS ***/
/****f* recorder/recorder_read_config()
 * NAME
 *  recorder_read_config()
 * TYPE
 */
void recorder_read_config(void);
/*
 * FUNCTION
 *  Read the record.config file to determine when to write output.
 ******
 */

/****f* recorder/recorder_out()
 * NAME
 *  recorder_out()
 * TYPE
 */
void recorder_out(void);
/*
 * FUNCTION
 *  Master function to deal with output. Only pull device memory if a restart
 *  is necessary
 ******
 */

/****f* recorder/recorder_PP_init()
 * NAME
 *  recorder_PP_init()
 * TYPE
 */
void recorder_PP_init(char *name);
/*
 * FUNCTION
 *  Create the file name for writing and summarize fields to be written for the
 *  Pressure Poisson solver
 * ARGUMENTS
 *  * name -- the name of the file to be written
 ******
 */

/****f* recorder/recorder_PP()
 * NAME
 *  recorder_PP()
 * TYPE
 */
void recorder_PP(char *name, int niter, real resid, real etime);
/*
 * FUNCTION 
 *  Write out PP solver information to file name.
 * ARGUMENTS
 *  * name -- the name of the file to which to write
 *  * niter -- the number of iterations to convergence
 *  * resid -- the residual at convergence
 *  * etime -- elapsed time
 ******
 */

/****f* recorder/recorder_PP_init_timed()
 * NAME
 *  recorder_PP_init_timed()
 * TYPE
 */
void recorder_PP_init_timed(char *name);
/*
 * FUNCTION
 *  Create the file name for writing and summarize fields to be written for the
 *  Pressure Poisson solver if we are looking to time and profile individual
 *  kernels
 * ARGUMENTS
 *  * name -- the name of the file to be written
 ******
 */

/****f* recorder/recorder_PP_timed()
 * NAME
 *  recorder_PP_timed()
 * TYPE
 */
void recorder_PP_timed(char *name, int niter, real resid, real etime,
  real etime_spmv,
  real etime_ip1, 
  real etime_AR1, 
  real etime_up1, 
  real etime_ip2, 
  real etime_AR2, 
  real etime_up2, 
  real etime_mpi);
/*
 * FUNCTION 
 *  Write out PP solver information to file name for profiling
 * ARGUMENTS
 *  * name -- the name of the file to which to write
 *  * niter -- the number of iterations to convergence
 *  * resid -- the residual at convergence
 *  * etime -- elapsed time
 *  * etime_spmv -- PP_spmv_shared_load
 *  * etime_ip1 --  thrust::inner_product
 *  * etime_AR1 --  MPI_Allreduce
 *  * etime_up1 --  PP_update_soln_resid
 *  * etime_ip2 --  thurst::inner_product
 *  * etime_AR2 --  MPI_Allreduce
 *  * etime_up2 --  PP_update_search
 *  * etime_mpi -- mpi_cuda_exchange_Gcc
 ******
 */

/****f* recorder/restart_recorder_write()
 * NAME
 *  restart_recorder_write()
 * TYPE
 */
int restart_recorder_write(void);
/*
 * FUNCTION 
 *  Write restart file if it's time to do so
 ******
 */

/****f* recorder/recorder_turb_init()
 * NAME
 *  recorder_turb_init()
 * TYPE
 */
void recorder_turb_init(char *name);
/*
 * FUNCTION
 *  Create the file name for writing information about the turbulent state at
 *  each timestep.
 * ARGUMENTS
 *  * name -- the name of the file to be written
 ******
 */

/****f* recorder/recorder_turb()
 * NAME
 *  recorder_turb()
 * TYPE
 */
void recorder_turb(char *name, real tke, real dissipation);
/*
 * FUNCTION 
 *  Write out turbulent info
 * ARGUMENTS
 *  * name -- the name of the file to which to write
 *  * tke -- turbulent kinetic energy
 *  * dissipation -- average dissipation
 ******
 */

#endif
