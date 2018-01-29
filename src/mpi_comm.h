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

/****h* Bluebottle/mpi_comm
 * NAME
 *  mpi_comm
 * FUNCTION
 *  Handle MPI Communications
 ******
 */

#ifndef _MPI_COMM_H
#define _MPI_COMM_H

#include "bluebottle.h"

#define WAIT() { fflush(stdout); \
                 int ret = time(0) + 1; \
                 while (time(0) < ret); } 

/* VARIABLES */

/****v* mpi_comm/mpi_part_struct
 * NAME
 *  mpi_part_struct
 * TYPE
 */
extern MPI_Datatype mpi_part_struct;
/*
 * PURPOSE
 *  Defines the mpi derived data type analog of part_struct
 */

/****v* mpi_comm/nparts_recv_win
 * NAME
 *  nparts_recv_win
 * TYPE
 */
extern MPI_Win nparts_recv_win;
/*
 * PURPOSE
 *  MPI Window into the nparts_recv array, which indicates to a process how many
 *  particles to expect from its 6 adjacent neighbors
 */

/****v* mpi_comm/parts_recv_win_e
 * NAME
 *  parts_recv_win_e
 * TYPE
 */
extern MPI_Win parts_recv_win_e;
/*
 * PURPOSE
 *  MPI Window into _recv_parts_e, a temporary storage location that contains
 *  incoming particles. The size changes every timestep, so it must be reallocd
 *  and a new window opened
 */

/****v* mpi_comm/parts_recv_win_w
 * NAME
 *  parts_recv_win_w
 * TYPE
 */
extern MPI_Win parts_recv_win_w;
/*
 * PURPOSE
 *  MPI Window into _recv_parts_w, a temporary storage location that contains
 *  incoming particles. The size changes every timestep, so it must be reallocd
 *  and a new window opened
 */

/****v* mpi_comm/parts_recv_win_n
 * NAME
 *  parts_recv_win_n
 * TYPE
 */
extern MPI_Win parts_recv_win_n;
/*
 * PURPOSE
 *  MPI Window into _recv_parts_n, a temporary storage location that contains
 *  incoming particles. The size changes every timestep, so it must be reallocd
 *  and a new window opened
 */

/****v* mpi_comm/parts_recv_win_s
 * NAME
 *  parts_recv_win_s
 * TYPE
 */
extern MPI_Win parts_recv_win_s;
/*
 * PURPOSE
 *  MPI Window into _recv_parts_b, a temporary storage location that contains
 *  incoming particles. The size changes every timestep, so it must be reallocd
 *  and a new window opened
 */

/****v* mpi_comm/parts_recv_win_t
 * NAME
 *  parts_recv_win_t
 * TYPE
 */
extern MPI_Win parts_recv_win_t;
/*
 * PURPOSE
 *  MPI Window into _recv_parts_t, a temporary storage location that contains
 *  incoming particles. The size changes every timestep, so it must be reallocd
 *  and a new window opened
 */

/****v* mpi_comm/parts_recv_win_b
 * NAME
 *  parts_recv_win_b
 * TYPE
 */
extern MPI_Win parts_recv_win_b;
/*
 * PURPOSE
 *  MPI Window into _recv_parts_b, a temporary storage location that contains
 *  incoming particles. The size changes every timestep, so it must be reallocd
 *  and a new window opened
 */

/****s* mpi_comm/mpi_field_windows
 * NAME
 *  mpi_field_windows
 * TYPE
 */
typedef struct mpi_field_windows {
  MPI_Win recv_e;
  MPI_Win recv_w;
  MPI_Win recv_n;
  MPI_Win recv_s;
  MPI_Win recv_t;
  MPI_Win recv_b;
} mpi_field_windows;
/*
 * PURPOSE
 *  Carry MPI windows related to different exchange directions
 * MEMBERS
 *  * recv_e -- mpi window into east recv array
 *  * recv_w -- mpi window into west recv array
 *  * recv_n -- mpi window into north recv array
 *  * recv_s -- mpi window into south recv array
 *  * recv_t -- mpi window into top recv array
 *  * recv_b -- mpi window into bottom recv array
 */

/****s* mpi_comm/comm_struct
 * NAME
 *  comm_struct
 * TYPE
 */
typedef struct comm_struct {
  mpi_field_windows Gcc;
  mpi_field_windows Gfx;
  mpi_field_windows Gfy;
  mpi_field_windows Gfz;
  //MPI_Group e; // for pscw
  //MPI_Group w;
  //MPI_Group n;
  //MPI_Group s;
  //MPI_Group t;
  //MPI_Group b;
} comm_struct;
/*
 * PURPOSE
 *  Carry information related to the different mpi communicating variables
 * MEMBERS
 *  * Gcc -- MPI recv windows for Gcc fields
 *  * Gfx -- MPI recv windows for Gfx fields
 *  * Gfy -- MPI recv windows for Gfy fields
 *  * Gfz -- MPI recv windows for Gfz fields
 *  * e -- MPI group containing east target
 *  * w -- MPI group containing west target
 *  * n -- MPI group containing north target
 *  * s -- MPI group containing south target
 *  * t -- MPI group containing top target
 *  * b -- MPI group containing bottom target
 */

/****v* mpi_comm/comm
 * NAME
 *  mpi_comm
 * TYPE
 */
extern comm_struct *comm;
/*
 * PURPOSE
 *  Contains mpi communication information
 */

/****v* mpi_comm/nprocs
 * NAME
 *  nprocs
 * TYPE
 */
extern int nprocs;
/*
 * PURPOSE 
 *  The number of MPI processes launched by mpiexec.
 ****** 
 */

/****v* mpi_comm/rank
 * NAME
 *  rank
 * TYPE
 */
extern int rank;
/*
 * PURPOSE
 *  The MPI process rank number.
 ******
 */

/****v* mpi_comm/device
 * NAME
 *  device
 * TYPE
 */
extern int device;
/*
 * PURPOSE
 *  The cuda device number associated to the current MPI process rank number.
 ******
 */

/****v* mpi_comm/world_group
 * NAME
 *  world_group
 * TYPE
 */
//extern MPI_Group world_group;
/*
 * PURPOSE
 *  The group of processes in MPI_COMM_WORLD
 ******
 */

/* FUNCTIONS */
/****f* mpi_comm/mpi_startup()
 * NAME
 *  mpi_startup()
 * USAGE
 */
void mpi_startup(int argc, char *argv[]);
/*
 * FUNCTION
 *  Call the MPI startup functions.
 * ARGUMENTS
 *  * argc -- command line argc
 *  * argv -- command line argv
 ******
 */

/****f* mpi_comm/mpi_dom_comm_init()
 * NAME
 *  mpi_dom_comm_init()
 * USAGE
 */
void mpi_dom_comm_init(void);
/*
 * FUNCTION
 *  Initialize MPI communication data types for the flow domain
 ******
 */

/****f* mpi_comm/mpi_dom_comm_init_groups()
 * NAME
 *  mpi_dom_comm_init_groups()
 * USAGE
 */
//void mpi_dom_comm_init_groups(void);
/*
 * FUNCTION
 *  Initialize MPI groups for use in general active target sync
 ******
 */

/****f* mpi_comm/mpi_dom_comm_create_windows()
 * NAME
 *  mpi_dom_comm_create_windows()
 * USAGE
 */
void mpi_dom_comm_create_windows(void);
/*
 * FUNCTION
 *  Create MPI windows into boundary data
 ******
 */

/****f* mpi_comm/mpi_cuda_exchange_Gcc()
 * NAME
 *  mpi_cuda_exchange_Gcc()
 * USAGE
 */
void mpi_cuda_exchange_Gcc(real *array);
/*
 * FUNCTION
 *  Exchange all Gcc boundaries between subdomains and their neighbors.
 *  1) Pack boundary planes into contiguous arrays
 *  2) Send to their neighbor
 *  3) Unpack
 * ARGUMENTS
 *  * array -- Gcc array to be exchanged
 ******
 */

/****f* mpi_comm/mpi_cuda_exchange_Gfx()
 * NAME
 *  mpi_cuda_exchange_Gfx()
 * USAGE
 */
void mpi_cuda_exchange_Gfx(real *array);
/*
 * FUNCTION
 *  Exchange all Gfx boundaries between subdomains and their neighbors.
 *  1) Pack boundary planes into contiguous arrays
 *  2) Send to their neighbor
 *  3) Unpack
 * ARGUMENTS
 *  * array -- Gfx array to be exchanged
 ******
 */

/****f* mpi_comm/mpi_cuda_exchange_Gfy()
 * NAME
 *  mpi_cuda_exchange_Gfy()
 * USAGE
 */
void mpi_cuda_exchange_Gfy(real *array);
/*
 * FUNCTION
 *  Exchange all Gfy boundaries between subdomains and their neighbors.
 *  1) Pack boundary planes into contiguous arrays
 *  2) Send to their neighbor
 *  3) Unpack
 * ARGUMENTS
 *  * array -- Gfy array to be exchanged
 ******
 */

/****f* mpi_comm/mpi_cuda_exchange_Gfz()
 * NAME
 *  mpi_cuda_exchange_Gfz()
 * USAGE
 */
void mpi_cuda_exchange_Gfz(real *array);
/*
 * FUNCTION
 *  Exchange all Gfz boundaries between subdomains and their neighbors.
 *  1) Pack boundary planes into contiguous arrays
 *  2) Send to their neighbor
 *  3) Unpack
 * ARGUMENTS
 *  * array -- Gfz array to be exchanged
 ******
 */

/****f* mpi_comm/mpi_cuda_fence_all()
 * NAME
 *  mpi_cuda_fence_all()
 * USAGE
 */
void mpi_cuda_fence_all(mpi_field_windows wins);
/*
 * FUNCTION
 *  MPI_Win_fence in all directions
 * ARGUMENTS
 *  * wins -- comm.G{cc,fx,fy,fz} type mpi communicator windows to be used
 ******
 */

/****f* mpi_comm/mpi_cuda_post_all()
 * NAME
 *  mpi_cuda_post_all()
 * USAGE
 */
//void mpi_cuda_post_all(mpi_field_windows wins);
/*
 * FUNCTION
 *  MPI_Win_post in all directions
 * ARGUMENTS
 *  * wins -- comm.G{cc,fx,fy,fz} type mpi communicator windows to be used
 ******
 */

/****f* mpi_comm/mpi_cuda_start_all()
 * NAME
 *  mpi_cuda_start_all()
 * USAGE
 */
//void mpi_cuda_start_all(mpi_field_windows wins);
/*
 * FUNCTION
 *  MPI_Win_start in all directions
 * ARGUMENTS
 *  * wins -- comm.G{cc,fx,fy,fz} type mpi communicator windows to be used
 ******
 */

/****f* mpi_comm/mpi_cuda_complete_all()
 * NAME
 *  mpi_cuda_complete_all()
 * USAGE
 */
//void mpi_cuda_complete_all(mpi_field_windows wins);
/*
 * FUNCTION
 *  MPI_Win_complete in all directions
 * ARGUMENTS
 *  * wins -- comm.G{cc,fx,fy,fz} type mpi communicator windows to be used
 ******
 */

/****f* mpi_comm/mpi_cuda_wait_all()
 * NAME
 *  mpi_cuda_wait_all()
 * USAGE
 */
//void mpi_cuda_wait_all(mpi_field_windows wins);
/*
 * FUNCTION
 *  MPI_Win_wait in all directions
 * ARGUMENTS
 *  * wins -- comm.G{cc,fx,fy,fz} type mpi communicator windows to be used
 ******
 */

/****f* mpi_comm/mpi_parts_init()
 * NAME
 *  mpi_parts_init()
 * USAGE
 */
void mpi_parts_init(void);
/*
 * FUNCTION
 *  Define the MPI derived data type analog for part_struct, as well as alloc
 *  MPI Windows
 ******
 */

/****f* mpi_comm/mpi_send_nparts_i()
 * NAME
 *  mpi_send_nparts_i()
 * USAGE
 */
void mpi_send_nparts_i(void);
/*
 * FUNCTION
 *  Send the number of particles to be transferred in the east/west direction to
 *  the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_nparts_j()
 * NAME
 *  mpi_send_nparts_j()
 * USAGE
 */
void mpi_send_nparts_j(void);
/*
 * FUNCTION
 *  Send the number of particles to be transferred in the north/south direction
 *  to the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_nparts_k()
 * NAME
 *  mpi_send_nparts_k()
 * USAGE
 */
void mpi_send_nparts_k(void);
/*
 * FUNCTION
 *  Send the number of particles to be transferred in the top/bot direction to
 *  the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_parts_i()
 * NAME
 *  mpi_send_parts_i()
 * USAGE
 */
void mpi_send_parts_i(void);
/*
 * FUNCTION
 *  Send particle data structures in the east/west directions to the appropriate
 *  domain
 ******
 */

/****f* mpi_comm/mpi_send_parts_j()
 * NAME
 *  mpi_send_parts_j()
 * USAGE
 */
void mpi_send_parts_j(void);
/*
 * FUNCTION
 *  Send particle data structures in the north/south directions to the
 *  appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_parts_k()
 * NAME
 *  mpi_send_parts_k()
 * USAGE
 */
void mpi_send_parts_k(void);
/*
 * FUNCTION
 *  Send particle data structures in the top/bottom directions to the
 *  appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_psums_i()
 * NAME
 *  mpi_send_psums_i()
 * USAGE
 */
void mpi_send_psums_i(void);
/*
 * FUNCTION
 *  Send partial sums in the east/west directions to the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_psums_j()
 * NAME
 *  mpi_send_psums_j()
 * USAGE
 */
void mpi_send_psums_j(void);
/*
 * FUNCTION
 *  Send partial sums in the north/south directions to the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_psums_k()
 * NAME
 *  mpi_send_psums_k()
 * USAGE
 */
void mpi_send_psums_k(void);
/*
 * FUNCTION
 *  Send partial sums in the top/bottom directions to the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_forces_i()
 * NAME
 *  mpi_send_forces_i()
 * USAGE
 */
void mpi_send_forces_i(void);
/*
 * FUNCTION
 *  Send particle forces in the east/west directions to the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_forces_j()
 * NAME
 *  mpi_send_forces_j()
 * USAGE
 */
void mpi_send_forces_j(void);
/*
 * FUNCTION
 *  Send particle forces in the north/south directions to the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_send_forces_k()
 * NAME
 *  mpi_send_forces_k()
 * USAGE
 */
void mpi_send_forces_k(void);
/*
 * FUNCTION
 *  Send particle forces in the top/bottom directions to the appropriate domain
 ******
 */

/****f* mpi_comm/mpi_free()
 * NAME
 *  mpi_free()
 * USAGE
 */
void mpi_free(void);
/*
 * FUNCTION
 *  Free MPI memory
 ******
 */

/****f* mpi_comm/mpi_end()
 * NAME
 *  mpi_end()
 * USAGE
 */
void mpi_end(void);
/*
 * FUNCTION
 *  End MPI using MPI_Finalize()
 ******
 */

#endif
