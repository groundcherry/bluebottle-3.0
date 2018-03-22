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

#include "mpi_comm.h"

int nprocs;
int rank;
comm_struct *comm;
//MPI_Group world_group; // For pscw
MPI_Datatype mpi_part_struct;
MPI_Win nparts_recv_win;
MPI_Win parts_recv_win_e;
MPI_Win parts_recv_win_w;
MPI_Win parts_recv_win_n;
MPI_Win parts_recv_win_s;
MPI_Win parts_recv_win_t;
MPI_Win parts_recv_win_b;

void mpi_startup(int argc, char *argv[])
{
  // Get local_rank of process to set device
  char *local_rank = NULL;
  local_rank = getenv(ENV_LOCAL_RANK);
   
  if (local_rank != NULL) {
    rank = atoi(local_rank);
  }

  // Number of devices -- from cuda function
  int dev_count = cuda_device_count(); // per node basis

  // Set device 
  // For slurm, dev0 maps to CUDA_VISIBLE_DEVICES[0]
  // cuda enumerates visible devices starting from zero
  // e.g., CUDA_VISIBLE_DEVICES=1,2 -> cudaSetDevice[0] maps to "1"
  //                                -> cudaSetDevice[1] maps to "2"
  int device = rank % dev_count;
  cuda_device_init(device);

  // Init MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Get world groups
  //MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // Enable peer access
  // cuda_enable_peer();

  // Print some info about each rank
  char node[FILE_NAME_SIZE];
  int namelen;
  MPI_Get_processor_name(node, &namelen);
  printf("N%d >> Local rank is %d on %s\n", rank, atoi(local_rank), node);

  // Get starting time
  startwalltime = MPI_Wtime();
}

void mpi_dom_comm_init(void)
{
  //printf("N%d >> Initializing MPI communication datatypes.\n", rank);
  // This needs to be after cuda_dom_malloc_dev so that the _cuda arrays
  // exist

  // Allocate comm_struct
  comm = (comm_struct*) malloc(sizeof(comm_struct));
    cpumem += sizeof(comm_struct);

  // Create MPI Groups -- for PSCW (GATS)
  //mpi_dom_comm_init_groups();

  // Create mpi windows
  mpi_dom_comm_create_windows();
}

// void mpi_dom_comm_init_groups(void)
// {
//   //printf("N%d >> Initializing MPI groups.\n", rank);
//   /* MPI Groups
//    *  Initialize a group on each process that contains transfer target
//    *  If no neighbor (boundary), make group MPI_GROUP_EMPTY:
//    *  * Call with this will return with no communication
//    */
//   int n_elems;
//   int target;
// 
//   // East
//   if (dom[rank].e == MPI_PROC_NULL) {
//     n_elems = 0;            // Group will be MPI_GROUP_EMPTY
//     target = MPI_PROC_NULL; // Just to be on the safe side
//   } else {
//     n_elems = 1;            // Group will contain
//     target = dom[rank].e;   // target
//   }
//   MPI_Group_incl(world_group, n_elems, &target, &comm->e);
// 
//   // West
//   if (dom[rank].w == MPI_PROC_NULL) {
//     n_elems = 0;
//     target = MPI_PROC_NULL;
//   } else {
//     n_elems = 1;
//     target = dom[rank].w;
//   }
//   MPI_Group_incl(world_group, n_elems, &target, &comm->w);
// 
//   // North
//   if (dom[rank].n == MPI_PROC_NULL) {
//     n_elems = 0;
//     target = MPI_PROC_NULL;
//   } else {
//     n_elems = 1;
//     target = dom[rank].n;
//   }
//   MPI_Group_incl(world_group, n_elems, &target, &comm->n);
// 
//   // South
//   if (dom[rank].s == MPI_PROC_NULL) {
//     n_elems = 0;
//     target = MPI_PROC_NULL;
//   } else {
//     n_elems = 1;
//     target = dom[rank].s;
//   }
//   MPI_Group_incl(world_group, n_elems, &target, &comm->s);
// 
//   // Top
//   if (dom[rank].t == MPI_PROC_NULL) {
//     n_elems = 0;
//     target = MPI_PROC_NULL;
//   } else {
//     n_elems = 1;
//     target = dom[rank].t;
//   }
//   MPI_Group_incl(world_group, n_elems, &target, &comm->t);
// 
//   // Bottom
//   if (dom[rank].b == MPI_PROC_NULL) {
//     n_elems = 0;
//     target = MPI_PROC_NULL;
//   } else {
//     n_elems = 1;
//     target = dom[rank].b;
//   }
//   MPI_Group_incl(world_group, n_elems, &target, &comm->b);
// }

void mpi_dom_comm_create_windows()
{
  //printf("N%d >> Creating transfer windows.\n", rank);
  /* MPI Windows
   *  MPI_Info info is:
   *  -- no_locks -- no passive target sync
   *  -- same_dist_unit -- all use sizeof(real)
   *  Create windows for transfers in both directions
   *  https://www.open-mpi.org/doc/current/man3/MPI_Win_create.3.php
   */
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // NOTE TO USERS:
  //  Unitialized memory access error during valgrind most likely an ompi
  //  internal issue

  // GCC
  MPI_Win_create(_recv_Gcc_e, dom[rank].Gcc.s2_i * sizeof(real), sizeof(real), 
    no_locks, MPI_COMM_WORLD, &comm->Gcc.recv_e);
  MPI_Win_create(_recv_Gcc_w, dom[rank].Gcc.s2_i * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gcc.recv_w);

  MPI_Win_create(_recv_Gcc_n, dom[rank].Gcc.s2_j * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gcc.recv_n);
  MPI_Win_create(_recv_Gcc_s, dom[rank].Gcc.s2_j * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gcc.recv_s);

  MPI_Win_create(_recv_Gcc_t, dom[rank].Gcc.s2_k * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gcc.recv_t);
  MPI_Win_create(_recv_Gcc_b, dom[rank].Gcc.s2_k * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gcc.recv_b);

  // GFX
  MPI_Win_create(_recv_Gfx_e, dom[rank].Gfx.s2_i * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfx.recv_e);
  MPI_Win_create(_recv_Gfx_w, dom[rank].Gfx.s2_i * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfx.recv_w);

  MPI_Win_create(_recv_Gfx_n, dom[rank].Gfx.s2_j * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfx.recv_n);
  MPI_Win_create(_recv_Gfx_s, dom[rank].Gfx.s2_j * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfx.recv_s);

  MPI_Win_create(_recv_Gfx_t, dom[rank].Gfx.s2_k * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfx.recv_t);
  MPI_Win_create(_recv_Gfx_b, dom[rank].Gfx.s2_k * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfx.recv_b);

  // GFY
  MPI_Win_create(_recv_Gfy_e, dom[rank].Gfy.s2_i * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfy.recv_e);
  MPI_Win_create(_recv_Gfy_w, dom[rank].Gfy.s2_i * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfy.recv_w);

  MPI_Win_create(_recv_Gfy_n, dom[rank].Gfy.s2_j * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfy.recv_n);
  MPI_Win_create(_recv_Gfy_s, dom[rank].Gfy.s2_j * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfy.recv_s);

  MPI_Win_create(_recv_Gfy_t, dom[rank].Gfy.s2_k * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfy.recv_t);
  MPI_Win_create(_recv_Gfy_b, dom[rank].Gfy.s2_k * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfy.recv_b);

  // GFZ
  MPI_Win_create(_recv_Gfz_e, dom[rank].Gfz.s2_i * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfz.recv_e);
  MPI_Win_create(_recv_Gfz_w, dom[rank].Gfz.s2_i * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfz.recv_w);

  MPI_Win_create(_recv_Gfz_n, dom[rank].Gfz.s2_j * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfz.recv_n);
  MPI_Win_create(_recv_Gfz_s, dom[rank].Gfz.s2_j * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfz.recv_s);

  MPI_Win_create(_recv_Gfz_t, dom[rank].Gfz.s2_k * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfz.recv_t);
  MPI_Win_create(_recv_Gfz_b, dom[rank].Gfz.s2_k * sizeof(real), sizeof(real),
    no_locks, MPI_COMM_WORLD, &comm->Gfz.recv_b);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_cuda_exchange_Gcc(real *array)
{
  /* TODO List
   *  self exchange through cuda, not mpi (could be faster than cuda mpi)
   *  - logic is complicated
   *  - should we pack/unpack, or do directly?
   *  pack planes
   *  - overlap each pack using cuda streams? async
   *  - or, use an MPI_Datatype (MPI derived data type) to allocate -- don't 
   *    need to pack, but need to use a cuda void** pointer on the device 
   *  cuda_block -- is this pretty harsh?
   *  look into having 1 receive array, and ewnstb are strided. might fix
   *    some hanging problems
   *  overlay computation with communication
   *  PSCW vs FENCE?
   */
  /*
   * _send_Gcc_e -> _recv_Gcc_w          |  _send_Gcc_w -> _recv_Gcc_e
   * if (dom[rank].e > -1) {             |  if (dom[rank].w -1) {
   *   pack _send_Gcc_e                  |    pack _send_Gcc_w
   *   POST _recv_Gcc_e                  |    POST _recv_Gcc_w
   *   START _send_Gcc_e -> _recv_Gcc_w  |    START _send_Gcc_w -> _recv_Gcc_e
   *   COMPLETE                          |    COMPLETE
   *   WAIT                              |    WAIT
   * }                                   |  }
   */

  /* Pack planes */
  cuda_pack_planes_Gcc(array);
  cuda_block();

  // Transfer planes
  //mpi_cuda_post_all(comm.Gcc);
  //mpi_cuda_start_all(comm.Gcc);
  mpi_cuda_fence_all(comm->Gcc);  

  MPI_Put(_send_Gcc_w, dom[rank].Gcc.s2_i, mpi_real,                  // w2e
    dom[rank].w, 0, dom[rank].Gcc.s2_i, mpi_real, comm->Gcc.recv_e);
  MPI_Put(_send_Gcc_e, dom[rank].Gcc.s2_i, mpi_real,                  // e2w
    dom[rank].e, 0, dom[rank].Gcc.s2_i, mpi_real, comm->Gcc.recv_w);

  MPI_Put(_send_Gcc_s, dom[rank].Gcc.s2_j, mpi_real,                  // s2n
    dom[rank].s, 0, dom[rank].Gcc.s2_j, mpi_real, comm->Gcc.recv_n);
  MPI_Put(_send_Gcc_n, dom[rank].Gcc.s2_j, mpi_real,                  // n2s
    dom[rank].n, 0, dom[rank].Gcc.s2_j, mpi_real, comm->Gcc.recv_s);

  MPI_Put(_send_Gcc_b, dom[rank].Gcc.s2_k, mpi_real,                  // b2t
    dom[rank].b, 0, dom[rank].Gcc.s2_k, mpi_real, comm->Gcc.recv_t);
  MPI_Put(_send_Gcc_t, dom[rank].Gcc.s2_k, mpi_real,                  // t2b
    dom[rank].t, 0, dom[rank].Gcc.s2_k, mpi_real, comm->Gcc.recv_b);

  mpi_cuda_fence_all(comm->Gcc);  
  //mpi_cuda_complete_all(comm.Gcc);
  //mpi_cuda_wait_all(comm.Gcc);

  /* Unpack planes */
  cuda_unpack_planes_Gcc(array);
  cuda_block();
}

void mpi_cuda_exchange_Gfx(real *array)
{
  // Pack planes
  cuda_pack_planes_Gfx(array);
  cuda_block();

  mpi_cuda_fence_all(comm->Gfx);

  MPI_Put(_send_Gfx_w, dom[rank].Gfx.s2_i, mpi_real,                   // w2e
    dom[rank].w, 0, dom[rank].Gfx.s2_i, mpi_real, comm->Gfx.recv_e);
  MPI_Put(_send_Gfx_e, dom[rank].Gfx.s2_i, mpi_real,                   // e2w
    dom[rank].e, 0, dom[rank].Gfx.s2_i, mpi_real, comm->Gfx.recv_w);

  MPI_Put(_send_Gfx_s, dom[rank].Gfx.s2_j, mpi_real,                   // s2n
    dom[rank].s, 0, dom[rank].Gfx.s2_j, mpi_real, comm->Gfx.recv_n);
  MPI_Put(_send_Gfx_n, dom[rank].Gfx.s2_j, mpi_real,                   // n2s
    dom[rank].n, 0, dom[rank].Gfx.s2_j, mpi_real, comm->Gfx.recv_s);

  MPI_Put(_send_Gfx_b, dom[rank].Gfx.s2_k, mpi_real,                   // b2t
    dom[rank].b, 0, dom[rank].Gfx.s2_k, mpi_real, comm->Gfx.recv_t);
  MPI_Put(_send_Gfx_t, dom[rank].Gfx.s2_k, mpi_real,                   // t2b
    dom[rank].t, 0, dom[rank].Gfx.s2_k, mpi_real, comm->Gfx.recv_b);

  mpi_cuda_fence_all(comm->Gfx);

  /* Unpack planes */
  cuda_unpack_planes_Gfx(array);
  cuda_block();
}

void mpi_cuda_exchange_Gfy(real *array)
{
  // Pack planes
  cuda_pack_planes_Gfy(array);
  cuda_block();

  mpi_cuda_fence_all(comm->Gfy);

  MPI_Put(_send_Gfy_w, dom[rank].Gfy.s2_i, mpi_real,                   // w2e
    dom[rank].w, 0, dom[rank].Gfy.s2_i, mpi_real, comm->Gfy.recv_e);
  MPI_Put(_send_Gfy_e, dom[rank].Gfy.s2_i, mpi_real,                   // e2w
    dom[rank].e, 0, dom[rank].Gfy.s2_i, mpi_real, comm->Gfy.recv_w);

  MPI_Put(_send_Gfy_s, dom[rank].Gfy.s2_j, mpi_real,                   // s2n
    dom[rank].s, 0, dom[rank].Gfy.s2_j, mpi_real, comm->Gfy.recv_n);
  MPI_Put(_send_Gfy_n, dom[rank].Gfy.s2_j, mpi_real,                   // n2s
    dom[rank].n, 0, dom[rank].Gfy.s2_j, mpi_real, comm->Gfy.recv_s);

  MPI_Put(_send_Gfy_b, dom[rank].Gfy.s2_k, mpi_real,                   // b2t
    dom[rank].b, 0, dom[rank].Gfy.s2_k, mpi_real, comm->Gfy.recv_t);
  MPI_Put(_send_Gfy_t, dom[rank].Gfy.s2_k, mpi_real,                   // t2b
    dom[rank].t, 0, dom[rank].Gfy.s2_k, mpi_real, comm->Gfy.recv_b);

  mpi_cuda_fence_all(comm->Gfy);

  /* Unpack planes */
  cuda_unpack_planes_Gfy(array);
  cuda_block();
}

void mpi_cuda_exchange_Gfz(real *array)
{
  // Pack planes
  cuda_pack_planes_Gfz(array);
  cuda_block();

  mpi_cuda_fence_all(comm->Gfz);

  MPI_Put(_send_Gfz_w, dom[rank].Gfz.s2_i, mpi_real,                   // w2e
    dom[rank].w, 0, dom[rank].Gfz.s2_i, mpi_real, comm->Gfz.recv_e);
  MPI_Put(_send_Gfz_e, dom[rank].Gfz.s2_i, mpi_real,                   // e2w
    dom[rank].e, 0, dom[rank].Gfz.s2_i, mpi_real, comm->Gfz.recv_w);

  MPI_Put(_send_Gfz_s, dom[rank].Gfz.s2_j, mpi_real,                   // s2n
    dom[rank].s, 0, dom[rank].Gfz.s2_j, mpi_real, comm->Gfz.recv_n);
  MPI_Put(_send_Gfz_n, dom[rank].Gfz.s2_j, mpi_real,                   // n2s
    dom[rank].n, 0, dom[rank].Gfz.s2_j, mpi_real, comm->Gfz.recv_s);

  MPI_Put(_send_Gfz_b, dom[rank].Gfz.s2_k, mpi_real,                   // b2t
    dom[rank].b, 0, dom[rank].Gfz.s2_k, mpi_real, comm->Gfz.recv_t);
  MPI_Put(_send_Gfz_t, dom[rank].Gfz.s2_k, mpi_real,                   // t2b
    dom[rank].t, 0, dom[rank].Gfz.s2_k, mpi_real, comm->Gfz.recv_b);

  mpi_cuda_fence_all(comm->Gfz);

  /* Unpack planes */
  cuda_unpack_planes_Gfz(array);
  cuda_block();
}

void mpi_cuda_fence_all(mpi_field_windows wins)
{
  int assert = 0; // Can use better ones

  MPI_Win_fence(assert, wins.recv_e);            // w2e
  MPI_Win_fence(assert, wins.recv_w);            // e2w
  MPI_Win_fence(assert, wins.recv_n);            // s2n
  MPI_Win_fence(assert, wins.recv_s);            // n2s
  MPI_Win_fence(assert, wins.recv_t);            // b2t
  MPI_Win_fence(assert, wins.recv_b);            // t2b
}

//void mpi_cuda_post_all(mpi_field_windows wins)
//{
//  /* Assert flags -- post
//   * MPI_MODE_NOCHECK -- on post and start if post is guaranteed to be before start
//   * MPI_MODE_NOSTORE -- window has not changed since last sync
//   * use && between ASSERT_1 and ASSERT_2 to include both
//   */
//  int assert = 0;
//
//  MPI_Win_post(comm->e, assert, wins.recv_e);      // w2e
//  MPI_Win_post(comm->w, assert, wins.recv_w);      // e2w
//  MPI_Win_post(comm->n, assert, wins.recv_n);      // s2n
//  MPI_Win_post(comm->s, assert, wins.recv_s);      // n2s
//  MPI_Win_post(comm->t, assert, wins.recv_t);      // b2t
//  MPI_Win_post(comm->b, assert, wins.recv_b);      // t2b
//}

//void mpi_cuda_start_all(mpi_field_windows wins)
//{
//  /* Assert flags -- start
//   * MPI_MODE_NOCHECK -- on post and start if post is guaranteed to be before start
//   */
//  int assert = 0;
//
//  MPI_Win_start(comm->e, assert, wins.recv_e);      // w2e
//  MPI_Win_start(comm->w, assert, wins.recv_w);      // e2w
//  MPI_Win_start(comm->n, assert, wins.recv_n);      // s2n
//  MPI_Win_start(comm->s, assert, wins.recv_s);      // n2s
//  MPI_Win_start(comm->t, assert, wins.recv_t);      // b2t
//  MPI_Win_start(comm->b, assert, wins.recv_b);      // t2b
//}

//void mpi_cuda_complete_all(mpi_field_windows wins)
//{
//  MPI_Win_complete(wins.recv_e);      // w2e
//  MPI_Win_complete(wins.recv_w);      // e2w
//  MPI_Win_complete(wins.recv_n);      // s2n
//  MPI_Win_complete(wins.recv_s);      // n2s
//  MPI_Win_complete(wins.recv_t);      // b2t
//  MPI_Win_complete(wins.recv_b);      // t2b
//}

//void mpi_cuda_wait_all(mpi_field_windows wins)
//{
//  MPI_Win_wait(wins.recv_e);      // w2e
//  MPI_Win_wait(wins.recv_w);      // e2w
//  MPI_Win_wait(wins.recv_n);      // s2n
//  MPI_Win_wait(wins.recv_s);      // n2s
//  MPI_Win_wait(wins.recv_t);      // b2t
//  MPI_Win_wait(wins.recv_b);      // t2b
//}

void mpi_parts_init(void)
{
  /* Define mpi part struct */
  // https://stackoverflow.com/questions/33618937/trouble-understanding-mpi-
  //  type-create-struct
  // https://stackoverflow.com/questions/9864510/struct-serialization-in-c-and-
  //  transfer-over-mpi

  // XXX Unfortunately, these need to be updated with part_struct
  #define NMEMS 91              // number of members in part_struct

  int block_lengths[NMEMS] = {1,                // int N
                              1,                // real r
                              1,                // real x
                              1,                // real y
                              1,                // real z
                              1,                // real u
                              1,                // real v
                              1,                // real w
                              1,                // real u0
                              1,                // real v0
                              1,                // real w0
                              1,                // real udot;
                              1,                // real vdot;
                              1,                // real wdot;
                              1,                // real udot0;
                              1,                // real vdot0;
                              1,                // real wdot0;
                              1,                // real axx;
                              1,                // real axy;
                              1,                // real axz;
                              1,                // real ayx;
                              1,                // real ayy;
                              1,                // real ayz;
                              1,                // real azx;
                              1,                // real azy;
                              1,                // real azz;
                              1,                // real ox;
                              1,                // real oy;
                              1,                // real oz;
                              1,                // real ox0;
                              1,                // real oy0;
                              1,                // real oz0;
                              1,                // real oxdot;
                              1,                // real oydot;
                              1,                // real ozdot;
                              1,                // real oxdot0;
                              1,                // real oydot0;
                              1,                // real ozdot0;
                              1,                // real Fx;
                              1,                // real Fy;
                              1,                // real Fz;
                              1,                // real Lx;
                              1,                // real Ly;
                              1,                // real Lz;
                              1,                // real aFx;
                              1,                // real aFy;
                              1,                // real aFz;
                              1,                // real aLx;
                              1,                // real aLy;
                              1,                // real aLz;
                              1,                // real kFx;
                              1,                // real kFy;
                              1,                // real kFz;
                              1,                // real iFx;
                              1,                // real iFy;
                              1,                // real iFz;
                              1,                // real iLx;
                              1,                // real iLy;
                              1,                // real iLz;
                              NNODES,           // int nodes[NNODES];
                              1,                // real rho;
                              1,                // real E;
                              1,                // real sigma;
                              1,                // int order;
                              1,                // real rs;
                              1,                // int ncoeff;
                              1,                // real spring_k;
                              1,                // real spring_x;
                              1,                // real spring_y;
                              1,                // real spring_z;
                              1,                // real spring_l;
                              1,                // int translating;
                              1,                // int rotating;
                              MAX_NEIGHBORS,    // real St[MAX_NEIGHBORS];
                              MAX_NEIGHBORS,    // int iSt[MAX_NEIGHBORS];
                              1,                // real e_dry;
                              1,                // real coeff_fric;
                              MAX_COEFFS,       // real pnm_re[MAX_COEFFS]
                              MAX_COEFFS,       // real pnm_im[MAX_COEFFS]
                              MAX_COEFFS,       // real phinm_re[MAX_COEFFS]
                              MAX_COEFFS,       // real phinm_im[MAX_COEFFS]
                              MAX_COEFFS,       // real chinm_re[MAX_COEFFS]
                              MAX_COEFFS,       // real chinm_im[MAX_COEFFS]
                              MAX_COEFFS,       // real pnm_re0[MAX_COEFFS]
                              MAX_COEFFS,       // real pnm_im0[MAX_COEFFS]
                              MAX_COEFFS,       // real phinm_re0[MAX_COEFFS]
                              MAX_COEFFS,       // real phinm_im0[MAX_COEFFS]
                              MAX_COEFFS,       // real chinm_re0[MAX_COEFFS]
                              MAX_COEFFS,       // real chinm_im0[MAX_COEFFS]
                              1,                // int ncoll_part
                              1};               // int ncoll_wall

  MPI_Datatype types[NMEMS] = {MPI_INT,           // int N
                               mpi_real,          // real r
                               mpi_real,          // real x
                               mpi_real,          // real y
                               mpi_real,          // real z
                               mpi_real,          // real u
                               mpi_real,          // real v
                               mpi_real,          // real w
                               mpi_real,          // real u0
                               mpi_real,          // real v0
                               mpi_real,          // real w0
                               mpi_real,          // real udot;
                               mpi_real,          // real vdot;
                               mpi_real,          // real wdot;
                               mpi_real,          // real udot0;
                               mpi_real,          // real vdot0;
                               mpi_real,          // real wdot0;
                               mpi_real,          // real axx;
                               mpi_real,          // real axy;
                               mpi_real,          // real axz;
                               mpi_real,          // real ayx;
                               mpi_real,          // real ayy;
                               mpi_real,          // real ayz;
                               mpi_real,          // real azx;
                               mpi_real,          // real azy;
                               mpi_real,          // real azz;
                               mpi_real,          // real ox;
                               mpi_real,          // real oy;
                               mpi_real,          // real oz;
                               mpi_real,          // real ox0;
                               mpi_real,          // real oy0;
                               mpi_real,          // real oz0;
                               mpi_real,          // real oxdot;
                               mpi_real,          // real oydot;
                               mpi_real,          // real ozdot;
                               mpi_real,          // real oxdot0;
                               mpi_real,          // real oydot0;
                               mpi_real,          // real ozdot0;
                               mpi_real,          // real Fx;
                               mpi_real,          // real Fy;
                               mpi_real,          // real Fz;
                               mpi_real,          // real Lx;
                               mpi_real,          // real Ly;
                               mpi_real,          // real Lz;
                               mpi_real,          // real aFx;
                               mpi_real,          // real aFy;
                               mpi_real,          // real aFz;
                               mpi_real,          // real aLx;
                               mpi_real,          // real aLy;
                               mpi_real,          // real aLz;
                               mpi_real,          // real kFx;
                               mpi_real,          // real kFy;
                               mpi_real,          // real kFz;
                               mpi_real,          // real iFx;
                               mpi_real,          // real iFy;
                               mpi_real,          // real iFz;
                               mpi_real,          // real iLx;
                               mpi_real,          // real iLy;
                               mpi_real,          // real iLz;
                               MPI_INT,           // int nodes[NNODES];
                               mpi_real,          // real rho;
                               mpi_real,          // real E;
                               mpi_real,          // real sigma;
                               MPI_INT,           // int order;
                               mpi_real,          // real rs;
                               MPI_INT,           // int ncoeff;
                               mpi_real,          // real spring_k;
                               mpi_real,          // real spring_x;
                               mpi_real,          // real spring_y;
                               mpi_real,          // real spring_z;
                               mpi_real,          // real spring_l;
                               MPI_INT,           // int translating;
                               MPI_INT,           // int rotating;
                               mpi_real,          // real St[MAX_NEIGHBORS];
                               MPI_INT,           // int iSt[MAX_NEIGHBORS];
                               mpi_real,          // real e_dry;
                               mpi_real,          // real coeff_fric;
                               mpi_real,          // real pnm_re[MAX_COEFFS]
                               mpi_real,          // real pnm_im[MAX_COEFFS]
                               mpi_real,          // real phinm_re[MAX_COEFFS]
                               mpi_real,          // real phinm_im[MAX_COEFFS]
                               mpi_real,          // real chinm_re[MAX_COEFFS]
                               mpi_real,          // real chinm_im[MAX_COEFFS]
                               mpi_real,          // real pnm_re0[MAX_COEFFS]
                               mpi_real,          // real pnm_im0[MAX_COEFFS]
                               mpi_real,          // real phinm_re0[MAX_COEFFS]
                               mpi_real,          // real phinm_im0[MAX_COEFFS]
                               mpi_real,          // real chinm_re0[MAX_COEFFS]
                               mpi_real,          // real chinm_im0[MAX_COEFFS]
                               MPI_INT,           // int ncoll_part
                               MPI_INT};          // int ncoll_wall

  MPI_Aint offsets[NMEMS];

  offsets[0] = offsetof(part_struct, N);
  offsets[1] = offsetof(part_struct, r);
  offsets[2] = offsetof(part_struct, x);
  offsets[3] = offsetof(part_struct, y);
  offsets[4] = offsetof(part_struct, z);
  offsets[5] = offsetof(part_struct, u);
  offsets[6] = offsetof(part_struct, v);
  offsets[7] = offsetof(part_struct, w);
  offsets[8] =  offsetof(part_struct, u0);
  offsets[9] = offsetof(part_struct, v0);
  offsets[10] = offsetof(part_struct, w0);
  offsets[11] = offsetof(part_struct, udot);
  offsets[12] = offsetof(part_struct, vdot);
  offsets[13] = offsetof(part_struct, wdot);
  offsets[14] = offsetof(part_struct, udot0);
  offsets[15] = offsetof(part_struct, vdot0);
  offsets[16] = offsetof(part_struct, wdot0);
  offsets[17] = offsetof(part_struct, axx);
  offsets[18] = offsetof(part_struct, axy);
  offsets[19] = offsetof(part_struct, axz);
  offsets[20] = offsetof(part_struct, ayx);
  offsets[21] = offsetof(part_struct, ayy);
  offsets[22] = offsetof(part_struct, ayz);
  offsets[23] = offsetof(part_struct, azx);
  offsets[24] = offsetof(part_struct, azy);
  offsets[25] = offsetof(part_struct, azz);
  offsets[26] = offsetof(part_struct, ox);
  offsets[27] = offsetof(part_struct, oy);
  offsets[28] = offsetof(part_struct, oz);
  offsets[29] = offsetof(part_struct, ox0);
  offsets[30] = offsetof(part_struct, oy0);
  offsets[31] = offsetof(part_struct, oz0);
  offsets[32] = offsetof(part_struct, oxdot);
  offsets[33] = offsetof(part_struct, oydot);
  offsets[34] = offsetof(part_struct, ozdot);
  offsets[35] = offsetof(part_struct, oxdot0);
  offsets[36] = offsetof(part_struct, oydot0);
  offsets[37] = offsetof(part_struct, ozdot0);
  offsets[38] = offsetof(part_struct, Fx);
  offsets[39] = offsetof(part_struct, Fy);
  offsets[40] = offsetof(part_struct, Fz);
  offsets[41] = offsetof(part_struct, Lx);
  offsets[42] = offsetof(part_struct, Ly);
  offsets[43] = offsetof(part_struct, Lz);
  offsets[44] = offsetof(part_struct, aFx);
  offsets[45] = offsetof(part_struct, aFy);
  offsets[46] = offsetof(part_struct, aFz);
  offsets[47] = offsetof(part_struct, aLx);
  offsets[48] = offsetof(part_struct, aLy);
  offsets[49] = offsetof(part_struct, aLz);
  offsets[50] = offsetof(part_struct, kFx);
  offsets[51] = offsetof(part_struct, kFy);
  offsets[52] = offsetof(part_struct, kFz);
  offsets[53] = offsetof(part_struct, iFx);
  offsets[54] = offsetof(part_struct, iFy);
  offsets[55] = offsetof(part_struct, iFz);
  offsets[56] = offsetof(part_struct, iLx);
  offsets[57] = offsetof(part_struct, iLy);
  offsets[58] = offsetof(part_struct, iLz);
  offsets[59] = offsetof(part_struct, nodes);
  offsets[60] = offsetof(part_struct, rho);
  offsets[61] = offsetof(part_struct, E);
  offsets[62] = offsetof(part_struct, sigma);
  offsets[63] = offsetof(part_struct, order);
  offsets[64] = offsetof(part_struct, rs);
  offsets[65] = offsetof(part_struct, ncoeff);
  offsets[66] = offsetof(part_struct, spring_k);
  offsets[67] = offsetof(part_struct, spring_x);
  offsets[68] = offsetof(part_struct, spring_y);
  offsets[69] = offsetof(part_struct, spring_z);
  offsets[70] = offsetof(part_struct, spring_l);
  offsets[71] = offsetof(part_struct, translating);
  offsets[72] = offsetof(part_struct, rotating);
  offsets[73] = offsetof(part_struct, St);
  offsets[74] = offsetof(part_struct, iSt);
  offsets[75] = offsetof(part_struct, e_dry);
  offsets[76] = offsetof(part_struct, coeff_fric);
  offsets[77] = offsetof(part_struct, pnm_re);
  offsets[78] = offsetof(part_struct, pnm_im);
  offsets[79] = offsetof(part_struct, phinm_re);
  offsets[80] = offsetof(part_struct, phinm_im);
  offsets[81] = offsetof(part_struct, chinm_re);
  offsets[82] = offsetof(part_struct, chinm_im);
  offsets[83] = offsetof(part_struct, pnm_re0);
  offsets[84] = offsetof(part_struct, pnm_im0);
  offsets[85] = offsetof(part_struct, phinm_re0);
  offsets[86] = offsetof(part_struct, phinm_im0);
  offsets[87] = offsetof(part_struct, chinm_re0);
  offsets[88] = offsetof(part_struct, chinm_im0);
  offsets[89] = offsetof(part_struct, ncoll_part);
  offsets[90] = offsetof(part_struct, ncoll_wall);

  int n_members = NMEMS;
  MPI_Type_create_struct(n_members, block_lengths, offsets, types, &mpi_part_struct);

/* Resize type for alignment
  MPI_Datatype tmp_type;
  MPI_Type_create_struct(n_members, block_lengths, offsets, types, &tmp_type);

  MPI_Aint lower_bound;
  MPI_Aint extent;
  MPI_Type_get_extent(tmp_type, &lower_bound, &extent);
  MPI_Type_create_resized(tmp_type, lower_bound, extent, &mpi_part_struct);
*/

  MPI_Type_commit(&mpi_part_struct);

  /* Create windows */
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");

  int n_card = 6; // number of cardinal directions to send/recv from
  MPI_Win_create(nparts_recv, n_card * sizeof(int), sizeof(int), no_locks,
    MPI_COMM_WORLD, &nparts_recv_win);

  // Free the info we provided
  MPI_Info_free(&no_locks);

}

void mpi_send_nparts_i(void)
{
  // Fence
  int fence_assert = 0;
  MPI_Win_fence(fence_assert, nparts_recv_win);

  int count = 1;  // number of elements to send

  MPI_Put(&nparts_send[EAST], // send nparts_send[EAST]
          count,              // ...with count 1...
          MPI_INT,            // ...and size MPI_INT...
          dom[rank].e,        // ...to the eastern rank...
          WEST,               // ...at an offset of WEST into nparts_recv...
          count,              // ...with count 1...
          MPI_INT,            // ...and size MPI_INT...
          nparts_recv_win);   // ...at the location spec'd by this window...
  MPI_Put(&nparts_send[WEST], count, MPI_INT,
          dom[rank].w, EAST, count, MPI_INT, nparts_recv_win);

  // Fence
  MPI_Win_fence(fence_assert, nparts_recv_win);
}

void mpi_send_nparts_j(void)
{
  // Fence
  int fence_assert = 0;
  MPI_Win_fence(fence_assert, nparts_recv_win);

  int count = 1;  // number of elements to send

  MPI_Put(&nparts_send[NORTH], // send nparts_send[NORTH]
          count,               // ...with count 1...
          MPI_INT,             // ...and size MPI_INT...
          dom[rank].n,         // ...to the northern rank...
          SOUTH,               // ...at an offset of SOUTH into nparts_recv...
          count,               // ...with count 1...
          MPI_INT,             // ...and size MPI_INT...
          nparts_recv_win);    // ...at the location spec'd by this window...
  MPI_Put(&nparts_send[SOUTH], count, MPI_INT,
          dom[rank].s, NORTH, count, MPI_INT, nparts_recv_win);

  // Fence
  MPI_Win_fence(fence_assert, nparts_recv_win);
}

void mpi_send_nparts_k(void)
{
  // Fence
  int fence_assert = 0;
  MPI_Win_fence(fence_assert, nparts_recv_win);

  int count = 1;  // number of elements to send

  MPI_Put(&nparts_send[TOP],  // send nparts_send[TOP]
          count,               // ...with count 1...
          MPI_INT,             // ...and size MPI_INT...
          dom[rank].t,         // ...to the northern rank...
          BOTTOM,              // ...at an offset of BOTTOM into nparts_recv...
          count,               // ...with count 1...
          MPI_INT,             // ...and size MPI_INT...
          nparts_recv_win);    // ...at the location spec'd by this window...
  MPI_Put(&nparts_send[BOTTOM], count, MPI_INT,
          dom[rank].b, TOP, count, MPI_INT, nparts_recv_win);

  // Fence
  MPI_Win_fence(fence_assert, nparts_recv_win);
}

void mpi_send_parts_i(void)
{
  // MPI Info hints
  //  - no_locks: window is never locked, e.g. only active sync
  //  - same_disp_unit: all use sizeof(part_struct)
  //  can also use SAME_DISP_UNIT
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_parts{e,w}
  MPI_Win_create(_recv_parts_e, nparts_recv[EAST] * sizeof(part_struct),
    sizeof(part_struct), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_e);
  MPI_Win_create(_recv_parts_w, nparts_recv[WEST] * sizeof(part_struct),
    sizeof(part_struct), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_w);

  // Fence and put _send_parts -> _recv_parts
  MPI_Win_fence(0, parts_recv_win_e);
  MPI_Win_fence(0, parts_recv_win_w);

  MPI_Put(_send_parts_e,       // send _send_parts_e
          nparts_send[EAST],    // ...of count nparts_send[EAST]...
          mpi_part_struct,      // ...of type mpi_part_struct...
          dom[rank].e,          // ...to the eastern rank...
          0,                    // ...with zero displacement...
          nparts_send[EAST],    // ...receiving rank count...
          mpi_part_struct,      // ...receiving rank datatype...
          parts_recv_win_w);    // ...open window
  MPI_Put(_send_parts_w, nparts_send[WEST], mpi_part_struct, dom[rank].w,
    0, nparts_send[WEST], mpi_part_struct, parts_recv_win_e);

  MPI_Win_fence(0, parts_recv_win_e);
  MPI_Win_fence(0, parts_recv_win_w);

  // Free
  MPI_Win_free(&parts_recv_win_e);
  MPI_Win_free(&parts_recv_win_w);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_parts_j(void)
{
  // MPI Info hints
  //  - no_locks: window is never locked, e.g. only active sync
  //  - same_disp_unit: all use sizeof(part_struct)
  //  can also use SAME_DISP_UNIT
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_parts{n,s}
  MPI_Win_create(_recv_parts_n, nparts_recv[NORTH] * sizeof(part_struct),
    sizeof(part_struct), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_n);
  MPI_Win_create(_recv_parts_s, nparts_recv[SOUTH] * sizeof(part_struct),
    sizeof(part_struct), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_s);

  // Fence and put _send_parts -> _recv_parts
  MPI_Win_fence(0, parts_recv_win_n);
  MPI_Win_fence(0, parts_recv_win_s);

  MPI_Put(_send_parts_n,        // send _send_parts_n
          nparts_send[NORTH],   // ...of count nparts_send[NORTH]...
          mpi_part_struct,      // ...of type mpi_part_struct...
          dom[rank].n,          // ...to the northen rank...
          0,                    // ...with zero displacement...
          nparts_send[NORTH],   // ...receiving rank count...
          mpi_part_struct,      // ...receiving rank datatype...
          parts_recv_win_s);    // ...open window
  MPI_Put(_send_parts_s, nparts_send[SOUTH], mpi_part_struct, dom[rank].s,
    0, nparts_send[SOUTH], mpi_part_struct, parts_recv_win_n);

  MPI_Win_fence(0, parts_recv_win_n);
  MPI_Win_fence(0, parts_recv_win_s);

  // Free
  MPI_Win_free(&parts_recv_win_n);
  MPI_Win_free(&parts_recv_win_s);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_parts_k(void)
{
  // MPI Info hints
  //  - no_locks: window is never locked, e.g. only active sync
  //  - same_disp_unit: all use sizeof(part_struct)
  //  can also use SAME_DISP_UNIT
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_parts{t,b}
  MPI_Win_create(_recv_parts_t, nparts_recv[TOP] * sizeof(part_struct),
    sizeof(part_struct), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_t);
  MPI_Win_create(_recv_parts_b, nparts_recv[BOTTOM] * sizeof(part_struct),
    sizeof(part_struct), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_b);

  // Fence and put _send_parts -> _recv_parts
  MPI_Win_fence(0, parts_recv_win_t);
  MPI_Win_fence(0, parts_recv_win_b);

  MPI_Put(_send_parts_t,      // send _send_parts_t
          nparts_send[TOP],   // ...of count nparts_send[TOP]...
          mpi_part_struct,    // ...of type mpi_part_struct...
          dom[rank].t,        // ...to the northen rank...
          0,                  // ...with zero displacement...
          nparts_send[TOP],   // ...receiving rank count...
          mpi_part_struct,    // ...receiving rank datatype...
          parts_recv_win_b);  // ...open window
  MPI_Put(_send_parts_b, nparts_send[BOTTOM], mpi_part_struct, dom[rank].b,
    0, nparts_send[BOTTOM], mpi_part_struct, parts_recv_win_t);

  MPI_Win_fence(0, parts_recv_win_t);
  MPI_Win_fence(0, parts_recv_win_b);

  // Free
  MPI_Win_free(&parts_recv_win_t);
  MPI_Win_free(&parts_recv_win_b);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_psums_i(void)
{
  // MPI Info
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_psums_{e,w}
  // Reuse windows from particle communication
  int n_scalar_prods = 6;
  int npsums = n_scalar_prods * ncoeffs_max;
  MPI_Win_create(_sum_recv_e, nparts_recv[EAST] * npsums * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_e);
  MPI_Win_create(_sum_recv_w, nparts_recv[WEST] * npsums * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_w);

  // Fence and put _sum_send -> _sum_recv
  MPI_Win_fence(0, parts_recv_win_e);
  MPI_Win_fence(0, parts_recv_win_w);

  MPI_Put(_sum_send_e,                   // send _sum_send_e
          nparts_send[EAST] * npsums,    // ...of count nparts_send[EAST]*npsums
          mpi_real,                      // ...of type mpi_real...
          dom[rank].e,                   // ...to the eastern rank...
          0,                             // ...with zero displacement...
          nparts_send[EAST] * npsums,    // ...receiving rank count...
          mpi_real,                      // ...receiving rank datatype...
          parts_recv_win_w);             // ...open window
  MPI_Put(_sum_send_w, nparts_send[WEST] * npsums, mpi_real, dom[rank].w,
    0, nparts_send[WEST] * npsums, mpi_real, parts_recv_win_e);

  MPI_Win_fence(0, parts_recv_win_e);
  MPI_Win_fence(0, parts_recv_win_w);

  // Free
  MPI_Win_free(&parts_recv_win_e);
  MPI_Win_free(&parts_recv_win_w);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_psums_j(void)
{
  // MPI Info
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_psums_{e,w}
  // Reuse windows from particle communication
  int n_scalar_prods = 6;
  int npsums = n_scalar_prods * ncoeffs_max;
  MPI_Win_create(_sum_recv_n, nparts_recv[NORTH] * npsums * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_n);
  MPI_Win_create(_sum_recv_s, nparts_recv[SOUTH] * npsums * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_s);

  // Fence and put _sum_send -> _sum_recv
  MPI_Win_fence(0, parts_recv_win_n);
  MPI_Win_fence(0, parts_recv_win_s);

  MPI_Put(_sum_send_n, nparts_send[NORTH] * npsums, mpi_real, dom[rank].n,
          0, nparts_send[NORTH] * npsums, mpi_real, parts_recv_win_s);
  MPI_Put(_sum_send_s, nparts_send[SOUTH] * npsums, mpi_real, dom[rank].s,
          0, nparts_send[SOUTH] * npsums, mpi_real, parts_recv_win_n);

  MPI_Win_fence(0, parts_recv_win_n);
  MPI_Win_fence(0, parts_recv_win_s);

  // Free
  MPI_Win_free(&parts_recv_win_n);
  MPI_Win_free(&parts_recv_win_s);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_psums_k(void)
{
  // MPI Info
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_psums_{e,w}
  // Reuse windows from particle communication
  int n_scalar_prods = 6;
  int npsums = n_scalar_prods * ncoeffs_max;
  MPI_Win_create(_sum_recv_t, nparts_recv[TOP] * npsums * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_t);
  MPI_Win_create(_sum_recv_b, nparts_recv[BOTTOM] * npsums * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_b);

  // Fence and put _sum_send -> _sum_recv
  MPI_Win_fence(0, parts_recv_win_t);
  MPI_Win_fence(0, parts_recv_win_b);

  MPI_Put(_sum_send_t, nparts_send[TOP] * npsums, mpi_real, dom[rank].t,
          0, nparts_send[TOP] * npsums, mpi_real, parts_recv_win_b);
  MPI_Put(_sum_send_b, nparts_send[BOTTOM] * npsums, mpi_real, dom[rank].b,
    0, nparts_send[BOTTOM] * npsums, mpi_real, parts_recv_win_t);

  MPI_Win_fence(0, parts_recv_win_t);
  MPI_Win_fence(0, parts_recv_win_b);

  // Free
  MPI_Win_free(&parts_recv_win_t);
  MPI_Win_free(&parts_recv_win_b);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_forces_i(void)
{
  // MPI Info
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_forces_{e,w}
  // Reuse windows from particle communication
  int n_send = 9;
  MPI_Win_create(_force_recv_e, nparts_recv[EAST] * n_send * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_e);
  MPI_Win_create(_force_recv_w, nparts_recv[WEST] * n_send * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_w);

  // Fence and put _force_send -> _force_recv
  MPI_Win_fence(0, parts_recv_win_e);
  MPI_Win_fence(0, parts_recv_win_w);

  MPI_Put(_force_send_e,                 // send _force_send_e
          nparts_send[EAST] * n_send,    // ...of count nparts_send[EAST]*n_send
          mpi_real,                      // ...of type mpi_real...
          dom[rank].e,                   // ...to the eastern rank...
          0,                             // ...with zero displacement...
          nparts_send[EAST] * n_send,    // ...receiving rank count...
          mpi_real,                      // ...receiving rank datatype...
          parts_recv_win_w);             // ...open window
  MPI_Put(_force_send_w, nparts_send[WEST] * n_send, mpi_real, dom[rank].w,
    0, nparts_send[WEST] * n_send, mpi_real, parts_recv_win_e);

  MPI_Win_fence(0, parts_recv_win_e);
  MPI_Win_fence(0, parts_recv_win_w);

  // Free
  MPI_Win_free(&parts_recv_win_e);
  MPI_Win_free(&parts_recv_win_w);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_forces_j(void)
{
  // MPI Info
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_forces_{e,w}
  // Reuse windows from particle communication
  int n_send = 9;
  MPI_Win_create(_force_recv_n, nparts_recv[NORTH] * n_send * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_n);
  MPI_Win_create(_force_recv_s, nparts_recv[SOUTH] * n_send * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_s);

  // Fence and put _force_send -> _force_recv
  MPI_Win_fence(0, parts_recv_win_n);
  MPI_Win_fence(0, parts_recv_win_s);

  MPI_Put(_force_send_n,                 // send _force_send_n
          nparts_send[NORTH] * n_send,    // ...of count nparts_send[NORTH]*n_send
          mpi_real,                      // ...of type mpi_real...
          dom[rank].n,                   // ...to the eastern rank...
          0,                             // ...with zero displacement...
          nparts_send[NORTH] * n_send,    // ...receiving rank count...
          mpi_real,                      // ...receiving rank datatype...
          parts_recv_win_s);             // ...open window
  MPI_Put(_force_send_s, nparts_send[SOUTH] * n_send, mpi_real, dom[rank].s,
    0, nparts_send[SOUTH] * n_send, mpi_real, parts_recv_win_n);

  MPI_Win_fence(0, parts_recv_win_n);
  MPI_Win_fence(0, parts_recv_win_s);

  // Free
  MPI_Win_free(&parts_recv_win_n);
  MPI_Win_free(&parts_recv_win_s);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_send_forces_k(void)
{
  // MPI Info
  MPI_Info no_locks;
  MPI_Info_create(&no_locks);
  MPI_Info_set(no_locks, "no_locks", "true");
  MPI_Info_set(no_locks, "same_disp_unit", "true");

  // Open MPI Windows for _recv_forces_{e,w}
  // Reuse windows from particle communication
  int n_send = 9;
  MPI_Win_create(_force_recv_t, nparts_recv[TOP] * n_send * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_t);
  MPI_Win_create(_force_recv_b, nparts_recv[BOTTOM] * n_send * sizeof(real),
    sizeof(real), MPI_INFO_NULL, MPI_COMM_WORLD, &parts_recv_win_b);

  // Fence and put _force_send -> _force_recv
  MPI_Win_fence(0, parts_recv_win_t);
  MPI_Win_fence(0, parts_recv_win_b);

  MPI_Put(_force_send_t,                 // send _force_send_t
          nparts_send[TOP] * n_send,    // ...of count nparts_send[TOP]*n_send
          mpi_real,                      // ...of type mpi_real...
          dom[rank].t,                   // ...to the eastern rank...
          0,                             // ...with zero displacement...
          nparts_send[TOP] * n_send,    // ...receiving rank count...
          mpi_real,                      // ...receiving rank datatype...
          parts_recv_win_b);             // ...open window
  MPI_Put(_force_send_b, nparts_send[BOTTOM] * n_send, mpi_real, dom[rank].b,
    0, nparts_send[BOTTOM] * n_send, mpi_real, parts_recv_win_t);

  MPI_Win_fence(0, parts_recv_win_t);
  MPI_Win_fence(0, parts_recv_win_b);

  // Free
  MPI_Win_free(&parts_recv_win_t);
  MPI_Win_free(&parts_recv_win_b);

  // Free the info we provided
  MPI_Info_free(&no_locks);
}

void mpi_free()
{
  //printf("N%d >> Freeing communication info... \n", rank);

  // Groups
  //MPI_Group_free(&comm->e);
  //MPI_Group_free(&comm->w);
  //MPI_Group_free(&comm->n);
  //MPI_Group_free(&comm->s);
  //MPI_Group_free(&comm->t);
  //MPI_Group_free(&comm->b);

  // Windows
  //cgnsorg.atlassian.net/browse/CGNS-116 - solved by using openMPI-2.1.0?
  MPI_Win_free(&comm->Gcc.recv_e);
  MPI_Win_free(&comm->Gcc.recv_w);
  MPI_Win_free(&comm->Gcc.recv_n);
  MPI_Win_free(&comm->Gcc.recv_s);
  MPI_Win_free(&comm->Gcc.recv_t);
  MPI_Win_free(&comm->Gcc.recv_b);
  MPI_Win_free(&comm->Gfx.recv_e);
  MPI_Win_free(&comm->Gfx.recv_w);
  MPI_Win_free(&comm->Gfx.recv_n);
  MPI_Win_free(&comm->Gfx.recv_s);
  MPI_Win_free(&comm->Gfx.recv_t);
  MPI_Win_free(&comm->Gfx.recv_b);
  MPI_Win_free(&comm->Gfy.recv_e);
  MPI_Win_free(&comm->Gfy.recv_w);
  MPI_Win_free(&comm->Gfy.recv_n);
  MPI_Win_free(&comm->Gfy.recv_s);
  MPI_Win_free(&comm->Gfy.recv_t);
  MPI_Win_free(&comm->Gfy.recv_b);
  MPI_Win_free(&comm->Gfz.recv_e);
  MPI_Win_free(&comm->Gfz.recv_w);
  MPI_Win_free(&comm->Gfz.recv_n);
  MPI_Win_free(&comm->Gfz.recv_s);
  MPI_Win_free(&comm->Gfz.recv_t);
  MPI_Win_free(&comm->Gfz.recv_b);

  free(comm);

  //MPI_Group_free(&world_group);

  if (NPARTS > 0) {
    MPI_Win_free(&nparts_recv_win);
    MPI_Type_free(&mpi_part_struct);
  }
}

void mpi_end() 
{
  if (rank == 0) printf("N%d >> Finalizing MPI... \n", rank);
  MPI_Finalize();
}
