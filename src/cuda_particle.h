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

/****h* Bluebottle/cuda_particle
 * NAME
 *  cuda_particle
 * FUNCTION
 *  Bluebottle cuda particle and cage kernel functions
 ******
 */

#ifndef _CUDA_PARTICLE_H
#define _CUDA_PARTICLE_H

extern "C"
{
#include "bluebottle.h"
#include "bluebottle.cuh"
}

// Unused but potentially useful helper functions
__global__ void print_parts(part_struct *parts, int nparts);
__global__ void print_bins(int *bin_start, int *bin_end);
__global__ void print_count(int *bin_count);

/* VARIABLES */

/* FUNCTIONS */
/****f* cuda_particle/print_int()
 * NAME
 *  print_int
 * TYPE
 */
__global__ void print_int(int *a, int n, int rank, int line);
/*
 * FUNCTION
 *  Print the first n values of the array containing numbers of int type
 * ARGUMENTS
 *  * a -- array to be printed
 *  * n -- length of the array
 *  * rank -- rank of the calling process
 *  * line -- line function was called from
 ******
 */

/****f* cuda_particle/reset_phases<<<>>>()
 * NAME
 *  reset_phases<<<>>>()
 * USAGE
 */
__global__ void reset_phases(int *phase, int *phase_shell);
/*
 * FUNCTION
 *  Set all phase array nodes to fluid (=-1) and phase shell nodes to fluid (=1)
 * ARGUMENTS
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 ******
 */

/****f* cuda_particle/reset_flag_u<<<>>>()
 * NAME
 *  reset_flag_u<<<>>>()
 * USAGE
 */
__global__ void reset_flag_u(int *flag_u);
/*
 * FUNCTION
 *  Set all flag_u nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_u -- the device x-direction velocity flag array subdomain
 ******
 */

/****f* cuda_particle/reset_flag_v<<<>>>()
 * NAME
 *  reset_flag_v<<<>>>()
 * USAGE
 */
__global__ void reset_flag_v(int *flag_v);
/*
 * FUNCTION
 *  Set all flag_v nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_v -- the device y-direction velocity flag array subdomain
 ******
 */

/****f* cuda_particle/reset_flag_w<<<>>>()
 * NAME
 *  reset_flag_w<<<>>>()
 * USAGE
 */
__global__ void reset_flag_w(int *flag_w);
/*
 * FUNCTION
 *  Set all flag_w nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_w -- the device z-direction velocity flag array subdomain
 ******
 */

/****f* cuda_particle/cage_setup<<<>>>()
 * NAME
 *  cage_setup<<<>>>()
 * USAGE
 */
__global__ void cage_setup(part_struct *parts, int n, int *cage_dim);
/*
 * FUNCTION
 *  Calculate cage size for the nth particle
 * ARGUMENTS
 *  * parts -- device particle structure
 *  * n -- current particle index
 *  * cage_dim -- cage dimensions in x,y,z directions
 ******
 */

/****f* cuda_particle/build_phase<<<>>>()
 * NAME
 *  build_phase<<<>>>()
 * USAGE
 */
__global__ void build_phase(part_struct *parts, int n, int *cage_dim, 
  int *phase, int *phase_shell, dom_struct *DOM, BC *bc);
/*
 * FUNCTION
 *  Build the cage for particle N.  Flag the particles in phase and build phase
 *  _shell
 * ARGUMENTS
 *  * parts -- device particle structure
 *  * n -- current particle index
 *  * cage_dim -- contains cage dimensins for current particle
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * DOM -- global domain struct
 *  * bc -- boundary condition structure
 ******
 */

/****f* cuda_particle/build_phase_shell<<<>>>()
 * NAME
 *  build_phase_shell<<<>>>()
 * USAGE
 */
__global__ void build_phase_shell(part_struct *parts, int n, int *cage_dim, 
  int *phase, int *phase_shell, dom_struct *DOM, BC *bc);
/*
 * FUNCTION
 *  Build the cage for particle N.  Flag the particles in phase and build phase
 *  _shell
 * ARGUMENTS
 *  * parts -- device particle structure
 *  * n -- current particle index
 *  * cage_dim -- contains cage dimensins for current particle
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * DOM -- global domain struct
 *  * bc -- boundary condition structure
 ******
 */

/****f* cuda_particle_kernel/phase_shell_x<<<>>>()
 * NAME
 *  phase_shell_x<<<>>>()
 * USAGE
 */
__global__ void phase_shell_x(part_struct *parts, int *phase, int *phase_shell);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the x-direction
 * ARGUMENTS
 *  * parts -- the device particle array subdomain
 *  * phase -- the phase array
 ******
 */

/****f* cuda_particle_kernel/phase_shell_y<<<>>>()
 * NAME
 *  phase_shell_y<<<>>>()
 * USAGE
 */
__global__ void phase_shell_y(part_struct *parts, int *phase, int *phase_shell);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the y-direction
 * ARGUMENTS
 *  * parts -- the device particle array subdomain
 *  * phase -- the phase array
 ******
 */

/****f* cuda_particle_kernel/phase_shell_z<<<>>>()
 * NAME
 *  phase_shell_z<<<>>>()
 * USAGE
 */
__global__ void phase_shell_z(part_struct *parts, int *phase, int *phase_shell);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the z-direction 
 *  flag outer cage nodes in each direction.
 * ARGUMENTS
 *  * parts -- the device particle array subdomain
 *  * phase -- the phase array
 ******
 */

/****f* cuda_particle/cage_flag_u<<<>>>()
 * NAME
 *  cage_flag_u<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u(int *flag_u, int *phase, int *phase_shell);
/*
 * FUNCTION
 *  Fill in flag_u for particle cages. flag = 1 (fluid), 0 (external dom bc),
 *  (-1) part bc
 * ARGUMENTS
 *  * flag_u -- the device flag_u array
 *  * phase -- the device phase array 
 *  * phase_shell -- the device phase_shell array 
 ******
 */

/****f* cuda_particle/cage_flag_v<<<>>>()
 * NAME
 *  cage_flag_v<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v(int *flag_v, int *phase, int *phase_shell);
/*
 * FUNCTION
 *  Fill in flag_v for particle cages. flag = 1 (fluid), 0 (external dom bc),
 *  (-1) part bc
 * ARGUMENTS
 *  * flag_v -- the device flag_u array
 *  * phase -- the device phase array 
 *  * phase_shell -- the device phase_shell array 
 ******
 */

/****f* cuda_particle/cage_flag_w<<<>>>()
 * NAME
 *  cage_flag_w<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w(int *flag_w, int *phase, int *phase_shell);
/*
 * FUNCTION
 *  Fill in flag_w for particle cages. flag = 1 (fluid), 0 (external dom bc),
 *  (-1) part bc
 * ARGUMENTS
 *  * flag_w -- the device flag_u array
 *  * phase -- the device phase array 
 *  * phase_shell -- the device phase_shell array 
 ******
 */

/****f* cuda_particle/flag_external_u<<<>>>()
  * NAME
  *  flag_external_u<<<>>>()
  * USAGE
  */
__global__ void flag_external_u(int *flag_u, int x_loc);
/*
 * FUNCTION
 * flags the flag_u based on the bc
 * ARGUMENTS
 *  * flag_u -- the device x-direction velocity flag array subdomain
 *  * x_loc -- the x location to set the flag at. This is _is or _ie depending
 *    on whether it is the east or west boundary
 ******
 */

/****f* cuda_particle/flag_external_v<<<>>>()
  * NAME
  *  flag_external_v<<<>>>()
  * USAGE
  */
__global__ void flag_external_v(int *flag_v, int y_loc);
/*
 * FUNCTION
 * flags the flag_v based on the bc
 * ARGUMENTS
 *  * flag_v -- the device y-direction velocity flag array subdomain
 *  * y_loc -- the y location to set the flag at. This is _js or _je depending
 *    on whether it is the south or north boundary
 ******
 */

/****f* cuda_particle/flag_external_w<<<>>>()
  * NAME
  *  flag_external_w<<<>>>()
  * USAGE
  */
__global__ void flag_external_w(int *flag_w, int z_loc);
/*
 * FUNCTION
 * flags the flag_w based on the bc
 * ARGUMENTS
 *  * flag_w -- the device x-direction velocity flag array subdomain
 *  * z_loc -- the z location to set the flag at. This is _ks or _ke depending
 *    on whether it is the bottom or top boundary
 ******
 */

/****f* cuda_particle/bin_fill_i<<<>>>()
  * NAME
  *  bin_fill_i<<<>>>()
  * USAGE
  */
__global__ void bin_fill_i(int *part_ind, int *part_bin, part_struct *parts,
  int nparts, dom_struct *DOM);
/*
 * FUNCTION
 *  Determines which bin each particle is in, for use when communicating in the
 *  i direction. Uses GFX_LOC.
 * ARGUMENTS
 *  * part_ind -- index of particle
 *  * part_bin -- index of bin
 *  * parts -- part_struct on subdom
 *  * nparts -- number of particles in subdom
 *  * DOM -- global domain information structure
 ******
 */

/****f* cuda_particle/bin_fill_j<<<>>>()
  * NAME
  *  bin_fill_j<<<>>>()
  * USAGE
  */
__global__ void bin_fill_j(int *part_ind, int *part_bin, part_struct *parts,
  int nparts, dom_struct *DOM);
/*
 * FUNCTION
 *  Determines which bin each particle is in, for use when communicating in the
 *  j direction. Uses GFY_LOC.
 * ARGUMENTS
 *  * part_ind -- index of particle
 *  * part_bin -- index of bin
 *  * parts -- part_struct on subdom
 *  * nparts -- number of particles in subdom
 *  * DOM -- global domain information structure
 ******
 */

/****f* cuda_particle/bin_fill_k<<<>>>()
  * NAME
  *  bin_fill_k<<<>>>()
  * USAGE
  */
__global__ void bin_fill_k(int *part_ind, int *part_bin, part_struct *parts,
  int nparts, dom_struct *DOM);
/*
 * FUNCTION
 *  Determines which bin each particle is in, for use when communicating in the
 *  k direction. Uses GFZ_LOC.
 * ARGUMENTS
 *  * part_ind -- index of particle
 *  * part_bin -- index of bin
 *  * parts -- part_struct on subdom
 *  * nparts -- number of particles in subdom
 *  * DOM -- global domain information structure
 ******
 */

/****f* cuda_particle/find_bin_start_end<<<>>>()
  * NAME
  *  find_bin_start_end<<<>>>()
  * USAGE
  */
__global__ void find_bin_start_end(int *bin_start, int *bin_end, int *part_bin,
  int nparts);
/*
 * FUNCTION
 *  Determines which bin each particle is in
 * ARGUMENTS
 *  * bin_start -- starting index of particle
 *  * bin_end -- ending index of particle
 *  * part_bin -- which bin each particle is located in
 *  * nparts -- number of particles in subdom
 ******
 */

/****f* cuda_particle/count_bin_parts_i<<<>>>()
  * NAME
  *  count_bin_parts_i<<<>>>()
  * USAGE
  */
__global__ void count_bin_parts_i(int *bin_start, int *bin_end, int *bin_count);
/*
 * FUNCTION
 *  Determines which bin each particle is in. Indexed with GFX
 * ARGUMENTS
 *  * bin_start -- starting index of particle
 *  * bin_end -- ending index of particle
 *  * bin_count -- number of particles in each bin
 ******
 */

/****f* cuda_particle/count_bin_parts_j<<<>>>()
  * NAME
  *  count_bin_parts_j<<<>>>()
  * USAGE
  */
__global__ void count_bin_parts_j(int *bin_start, int *bin_end, int *bin_count);
/*
 * FUNCTION
 *  Determines which bin each particle is in. Indexed with GFY
 * ARGUMENTS
 *  * bin_start -- starting index of particle
 *  * bin_end -- ending index of particle
 *  * bin_count -- number of particles in each bin
 ******
 */

/****f* cuda_particle/count_bin_parts_k<<<>>>()
  * NAME
  *  count_bin_parts_k<<<>>>()
  * USAGE
  */
__global__ void count_bin_parts_k(int *bin_start, int *bin_end, int *bin_count);
/*
 * FUNCTION
 *  Determines which bin each particle is in. Indexed with GFZ
 * ARGUMENTS
 *  * bin_start -- starting index of particle
 *  * bin_end -- ending index of particle
 *  * bin_count -- number of particles in each bin
 ******
 */

/****f* cuda_particle/pack_parts_e<<<>>>()
  * NAME
  *  pack_parts_e<<<>>>()
  * USAGE
  */
__global__ void pack_parts_e(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);
/*
 * FUNCTION
 *  Pack the particles in the east plane bins for sending
 * ARGUMENTS
 *  * send_parts -- structure to be filled and sent
 *  * parts -- subdom part_struct array
 *  * offset -- offset of each bin in send_parts
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 ******
 */

/****f* cuda_particle/pack_parts_w<<<>>>()
  * NAME
  *  pack_parts_w<<<>>>()
  * USAGE
  */
__global__ void pack_parts_w(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);
/*
 * FUNCTION
 *  pack the particles in the west plane bins for sending
 * ARGUMENTS
 *  * send_parts -- structure to be filled and sent
 *  * parts -- subdom part_struct array
 *  * offset -- offset of each bin in send_parts
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 ******
 */

/****f* cuda_particle/pack_parts_n<<<>>>()
  * NAME
  *  pack_parts_n<<<>>>()
  * USAGE
  */
__global__ void pack_parts_n(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);
/*
 * FUNCTION
 *  pack the particles in the north plane bins for sending
 * ARGUMENTS
 *  * send_parts -- structure to be filled and sent
 *  * parts -- subdom part_struct array
 *  * offset -- offset of each bin in send_parts
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 ******
 */

/****f* cuda_particle/pack_parts_s<<<>>>()
  * NAME
  *  pack_parts_s<<<>>>()
  * USAGE
  */
__global__ void pack_parts_s(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);
/*
 * FUNCTION
 *  pack the particles in the south plane bins for sending
 * ARGUMENTS
 *  * send_parts -- structure to be filled and sent
 *  * parts -- subdom part_struct array
 *  * offset -- offset of each bin in send_parts
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 ******
 */

/****f* cuda_particle/pack_parts_t<<<>>>()
  * NAME
  *  pack_parts_t<<<>>>()
  * USAGE
  */
__global__ void pack_parts_t(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);
/*
 * FUNCTION
 *  pack the particles in the top plane bins for sending
 * ARGUMENTS
 *  * send_parts -- structure to be filled and sent
 *  * parts -- subdom part_struct array
 *  * offset -- offset of each bin in send_parts
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 ******
 */

/****f* cuda_particle/pack_parts_b<<<>>>()
  * NAME
  *  pack_parts_b<<<>>>()
  * USAGE
  */
__global__ void pack_parts_b(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind);
/*
 * FUNCTION
 *  pack the particles in the bottom plane bins for sending
 * ARGUMENTS
 *  * send_parts -- structure to be filled and sent
 *  * parts -- subdom part_struct array
 *  * offset -- offset of each bin in send_parts
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 ******
 */

/****f* cuda_particle/copy_central_parts_i<<<>>>()
  * NAME
  *  copy_central_parts_i<<<>>>()
  * USAGE
  */
__global__ void copy_central_bin_parts_i(part_struct *tmp_parts,
  part_struct *parts, int *bin_start, int *bin_count, int *part_ind, int *offset);
/*
 * FUNCTION
 *  Copy the main particles into a temporary part structure
 * ARGUMENTS
 *  * tmp_parts -- temporary part struct while re-allocing
 *  * parts -- subdom part_struct array
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * offset -- destination offset of each bin in temp part storage
 ******
 */

/****f* cuda_particle/copy_central_parts_j<<<>>>()
  * NAME
  *  copy_central_parts_j<<<>>>()
  * USAGE
  */
__global__ void copy_central_bin_parts_j(part_struct *tmp_parts,
  part_struct *parts, int *bin_start, int *bin_count, int *part_ind, int *offset);
/*
 * FUNCTION
 *  Copy the main particles into a temporary part structure for j direction
 * ARGUMENTS
 *  * tmp_parts -- temporary part struct while re-allocing
 *  * parts -- subdom part_struct array
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * offset -- destination offset of each bin in temp part storage
 ******
 */

/****f* cuda_particle/copy_central_parts_k<<<>>>()
  * NAME
  *  copy_central_parts_k<<<>>>()
  * USAGE
  */
__global__ void copy_central_bin_parts_k(part_struct *tmp_parts,
  part_struct *parts, int *bin_start, int *bin_count, int *part_ind, int *offset);
/*
 * FUNCTION
 *  Copy the main particles into a temporary part structure for k direction
 * ARGUMENTS
 *  * tmp_parts -- temporary part struct while re-allocing
 *  * parts -- subdom part_struct array
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * offset -- destination offset of each bin in temp part storage
 ******
 */


/****f* cuda_particle/copy_ghost_bin_parts<<<>>>()
  * NAME
  *  copy_ghost_bin_parts<<<>>>()
  * USAGE
  */
__global__ void copy_ghost_bin_parts(part_struct *tmp_parts,
  part_struct *recv_parts, int nparts_recv, int offset, int plane,
  dom_struct *DOM);
/*
 * FUNCTION
 *  Add particles in ghost bins to temporary part struct
 * ARGUMENTS
 *  * tmp_parts -- temporary part struct while re-allocing
 *  * parts -- subdom part_struct array
 *  * nparts_recv -- number of ghost particles to add
 *  * offset -- offset into tmp_parts array to start adding
 *  * plane -- current plane being copied
 *  * DOM -- device side global domain struct
 ******
 */

/****f* cuda_particle/correct_periodic_boundaries_i<<<>>>()
  * NAME
  *  correct_periodic_boundaries_i<<<>>>()
  * USAGE
  */
__global__ void correct_periodic_boundaries_i(part_struct *parts,
  int offset, int nparts_added, BC *bc, dom_struct *DOM);
/*
 * FUNCTION
 *  Correct new particles for periodic boundaries (i)
 * ARGUMENTS
 *  * parts -- subdom part_struct array
 *  * offset -- beginning index of new (added) ghost particles
 *  * nparts_added -- total new (added) ghost particles
 *  * bc -- boundary condition structure
 *  * DOM -- global domain dom_struct
 ******
 */

/****f* cuda_particle/correct_periodic_boundaries_j<<<>>>()
  * NAME
  *  correct_periodic_boundaries_j<<<>>>()
  * USAGE
  */
__global__ void correct_periodic_boundaries_j(part_struct *parts,
  int offset, int nparts_added, BC *bc, dom_struct *DOM);
/*
 * FUNCTION
 *  Correct new particles for periodic boundaries (j)
 * ARGUMENTS
 *  * parts -- subdom part_struct array
 *  * offset -- beginning index of new (added) ghost particles
 *  * nparts_added -- total new (added) ghost particles
 *  * bc -- boundary condition structure
 *  * DOM -- global domain dom_struct
 ******
 */

/****f* cuda_particle/correct_periodic_boundaries_k<<<>>>()
  * NAME
  *  correct_periodic_boundaries_k<<<>>>()
  * USAGE
  */
__global__ void correct_periodic_boundaries_k(part_struct *parts,
  int offset, int nparts_added, BC *bc, dom_struct *DOM);
/*
 * FUNCTION
 *  Correct new particles for periodic boundaries (k)
 * ARGUMENTS
 *  * parts -- subdom part_struct array
 *  * offset -- beginning index of new (added) ghost particles
 *  * nparts_added -- total new (added) ghost particles
 *  * bc -- boundary condition structure
 *  * DOM -- global domain dom_struct
 ******
 */

/****f* cuda_particle/zero_ghost_bins_i<<<>>>()
  * NAME
  *  zero_ghost_bins_i<<<>>>()
  * USAGE
  */
__global__ void zero_ghost_bins_i(int *bin_count);
/*
 * FUNCTION
 *  Zero the ghost bin particle count on the EAST and WEST faces to prepare 
 *  for dev->host copy
 * ARGUMENTS
 *  * bin_count -- number of particles per bin
 ******
 */

/****f* cuda_particle/zero_ghost_bins_j<<<>>>()
  * NAME
  *  zero_ghost_bins_j<<<>>>()
  * USAGE
  */
__global__ void zero_ghost_bins_j(int *bin_count);
/*
 * FUNCTION
 *  Zero the ghost bin particle count on the NORTH and SOUTH faces to prepare 
 *  for dev->host copy
 * ARGUMENTS
 *  * bin_count -- number of particles per bin
 ******
 */

/****f* cuda_particle/zero_ghost_bins_k<<<>>>()
  * NAME
  *  zero_ghost_bins_k<<<>>>()
  * USAGE
  */
__global__ void zero_ghost_bins_k(int *bin_count);
/*
 * FUNCTION
 *  Zero the ghost bin particle count on the TOP and BOTTOM faces to prepare 
 *  for dev->host copy
 * ARGUMENTS
 *  * bin_count -- number of particles per bin
 ******
 */

/****f* cuda_particle/copy_subdom_parts<<<>>>()
  * NAME
  *  copy_subdom_parts<<<>>>()
  * USAGE
  */
__global__ void copy_subdom_parts(part_struct *tmp_parts,
  part_struct *parts, int *bin_start, int *bin_count, int *part_ind,
  int *bin_offset);
/*
 * FUNCTION
 *  Copy particles in subdomain (not ghost particles) to a temporary array for
 *  a device->host transfer. 
 * ARGUMENTS
 *  * tmp_parts -- temporary part struct while re-allocing
 *  * parts -- subdom part_struct array
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * bin_offset -- destination offset of each bin in temp part storage
 ******
 */

/****f* cuda_particle/move_parts_a<<<>>>()
  * NAME
  *  move_parts_a<<<>>>()
  * USAGE
  */
__global__ void move_parts_a(part_struct *parts, int nparts, real dt,
  g_struct g, gradP_struct gradP, real rho_f);
/*
 * FUNCTION
 *  Update the particle velocities and move the particles. Part A: does
 *  everything but move the particles. This way we can use the velocity
 *  expected at the end of the timestep to update the lubrication forces
 *  before updating the particle position.
 * ARGUMENTS
 *  * parts -- particle data structure
 *  * nparts -- number of particles in subdomain
 *  * dt -- current simulation timestep size
 *  * g -- structure containing gravity forces
 *  * gradP -- structure containing pressure gradients
 *  * rho_f -- fluid density
 ******
 */

/****f* cuda_particle/move_parts_b<<<>>>()
  * NAME
  *  move_parts_b<<<>>>()
  * USAGE
  */
__global__ void move_parts_b(part_struct *parts, int nparts, real dt,
  g_struct g, gradP_struct gradP, real rho_f);
/*
 * FUNCTION
 *  Update the particle velocities and move the particles. Part B: does
 *  everything includin move the particles. This way we can use the velocity
 *  expected at the end of the timestep to update the lubrication forces
 *  before updating the particle position.
 * ARGUMENTS
 *  * parts -- particle data structure
 *  * nparts -- number of particles in subdomain
 *  * dt -- current simulation timestep size
 *  * g -- structure containing gravity forces
 *  * gradP -- structure containing pressure gradients
 *  * rho_f -- fluid density
 ******
 */

/****f* cuda_particle/rotate()
 * NAME
 *  rotate
 * USAGE
 */
__device__ void rotate(real qr, real qi, real qj, real qk,
  real *pi, real *pj, real *pk);
/*
 * FUNCTION
 *  Apply quaternion conjugation p <-- q^-1 * p * q to rotate p according to
 *  rotation described by quaternion q.
 * ARGUMENTS
 *  * qr -- quaternion real component
 *  * qi -- quaternion imaginary component i
 *  * qj -- quaternion imaginary component j
 *  * qk -- quaternion imaginary component k
 *  * pi -- vector component i
 *  * pj -- vector component j
 *  * pk -- vector component k
 ******
 */

/****f* cuda_particle/part_BC_u<<<>>>()
 * NAME
 *  part_BC_u<<<>>>()
 * USAGE
 */
__global__ void part_BC_u(real *u, int *phase, int *flag_u,
  part_struct *parts, real nu, int nparts);
/*
 * FUNCTION
 *  Apply u-velocity boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * u -- the device flow velocity field
 *  * phase -- particle phase flag
 *  * flag_u -- the particle boundary flag on u-velocities
 *  * parts -- the device particle array subdomain
 *  * nu -- the fluid kinematic viscosity
 ******
 */

/****f* cuda_particle/part_BC_v<<<>>>()
 * NAME
 *  part_BC_v<<<>>>()
 * USAGE
 */
__global__ void part_BC_v(real *v, int *phase, int *flag_v,
  part_struct *parts, real nu, int nparts);
/*
 * FUNCTION
 *  Apply v-velocity boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * v -- the device flow velocity field
 *  * phase -- particle phase flag
 *  * flag_v -- the particle boundary flag on u-velocities
 *  * parts -- the device particle array subdomain
 *  * nu -- the fluid kinematic viscosity
 ******
 */

/****f* cuda_particle/part_BC_w<<<>>>()
 * NAME
 *  part_BC_w<<<>>>()
 * USAGE
 */
__global__ void part_BC_w(real *w, int *phase, int *flag_w,
  part_struct *parts, real nu, int nparts);
/*
 * FUNCTION
 *  Apply w-velocity boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * w -- the device flow velocity field
 *  * phase -- particle phase flag
 *  * flag_w -- the particle boundary flag on u-velocities
 *  * parts -- the device particle array subdomain
 *  * nu -- the fluid kinematic viscosity
 ******
 */

/****f* cuda_particle_kernel/part_BC_p<<<>>>()
 * NAME
 *  part_BC_p<<<>>>()
 * USAGE
 */
__global__ void part_BC_p(real *p, real *p_rhs, int *phase, int *phase_shell,
  part_struct *parts, real mu, real nu, real dt, real dt0, gradP_struct gradP, 
  real rho_f, int nparts);
/*
 * FUNCTION
 *  Apply pressure boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * p -- the device flow velocity field
 *  * p_rhs -- the Poisson problem right-hand side
 *  * phase_shell -- the particle shell boundary flag on pressure
 *  * phase -- the particle boundary flag on pressure
 *  * parts -- the device particle array subdomain
 *  * mu -- the fluid dynamic viscosity
 *  * nu -- the fluid kinematic viscosity
 *  * dt -- time step size
 *  * gradP -- the body force
 *  * rho_f -- fluid density
 ******
 */

/****f* cuda_particle_kernel/part_BC_p<<<>>>()
 * NAME
 *  part_BC_p<<<>>>()
 * USAGE
 */
__global__ void part_BC_p_fill(real *p, int *phase, part_struct *parts, real mu,
  real nu, real rho_f, gradP_struct gradP, int nparts);
/*
 * FUNCTION
 *  Apply pressure boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * p -- the device flow velocity field
 *  * p_rhs -- the Poisson problem right-hand side
 *  * phase_shell -- the particle shell boundary flag on pressure
 *  * phase -- the particle boundary flag on pressure
 *  * parts -- the device particle array subdomain
 *  * mu -- the fluid dynamic viscosity
 *  * nu -- the fluid kinematic viscosity
 *  * dt -- time step size
 *  * gradP -- the body force
 *  * rho_f -- fluid density
 ******
 */

/****f* cuda_particle/lamb_vel<<<>>>()
 * NAME
 *  lamb_vel<<<>>>()
 * USAGE
 */
__device__ void lamb_vel(int order, real a, real r, real theta, real phi,
  part_struct *parts, real nu, int p_ind, real *Ux, real *Uy, real *Uz);
/*
 * FUNCTION
 *  Calculate the velocities in the Cartesian directions from the
 *  analytic solution using the Lamb's coefficients.
 * ARGUMENTS
 *  * order -- Lamb's coefficient truncation order
 *  * a -- particle radius
 *  * r -- radial position
 *  * theta -- polar angle position
 *  * phi -- azimuthal angle position
 *  * parts -- particle information structure
 *  * nu -- the fluid kinematic viscosity
 *  * p_ind -- Lamb's coefficient access index helper value
 *  * Ux -- the x-direction pressure gradient
 *  * Uy -- the y-direction pressure gradient
 *  * Uz -- the z-direction pressure gradient
 ******
 */

/****f* cuda_particle/xyz2rtp<<<>>>()
 * NAME
 *  xyz2rtp<<<>>>()
 * USAGE
 */
__device__ void xyz2rtp(real x, real y, real z, real *r, real *t, real *p);
/*
 * PURPOSE
 *  Compute (r, theta, phi) from (x, y, z).
 * ARGUMENTS
 *  x -- x-component in Cartesian basis
 *  y -- y-component in Cartesian basis
 *  z -- z-component in Cartesian basis
 *  r -- r-component in spherical basis
 *  t -- theta-component in spherical basis
 *  p -- phi-component in spherical basis
 ******
 */

/****f* cuda_particle/Nnm<<<>>>()
 * NAME
 *  Nnm<<<>>>()
 * USAGE
 */
__device__ real Nnm(int n, int m);
/*
 * PURPOSE
 *  Compute spherical harmonic normalization N_nm.
 * ARGUMENTS
 *  * n -- degree
 *  * m -- order
 ******
 */

/****f* cuda_particle/Pnm<<<>>>()
 * NAME
 *  Pnm<<<>>>()
 * USAGE
 */
__device__ real Pnm(int n, int m, real t);
/*
 * PURPOSE
 *  Compute associated Legendre function P_nm(theta).
 * ARGUMENTS
 *  * n -- degree
 *  * m -- order
 *  * t -- theta
 ******
 */

/****f* cuda_particle_kernel/X_pn<<<>>>()
 * NAME
 *  X_pn<<<>>>()
 * USAGE
 */
__device__ real X_pn(int n, real theta, real phi, int pp, part_struct *parts);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pp -- Lamb's coefficient access index helper value
 *  * parts -- particle information structure
 ******
 */

/****f* cuda_particle_kernel/X_phin<<<>>>()
 * NAME
 *  X_phin<<<>>>()
 * USAGE
 */
__device__ real X_phin(int n, real theta, real phi, int pp, part_struct *parts);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pp -- Lamb's coefficient access index helper value
 *  * parts -- particle information structure
 ******
 */

/****f* cuda_particle/Y_pn<<<>>>()
 * NAME
 *  Y_pn<<<>>>()
 * USAGE
 */
__device__ real Y_pn(int n, real theta, real phi, int pp, part_struct *parts);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pp -- particle index
 *  * parts -- particle information structure
 ******
 */

/****f* cuda_particle_kernel/Y_phin<<<>>>()
 * NAME
 *  Y_phin<<<>>>()
 * USAGE
 */
__device__ real Y_phin(int n, real theta, real phi, int pp, part_struct *parts);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pp -- Lamb's coefficient access index helper value
 *  * parts -- particle information
 ******
 */

/****f* cuda_particle_kernel/Y_chin<<<>>>()
 * NAME
 *  Y_chin<<<>>>()
 * USAGE
 */
__device__ real Y_chin(int n, real theta, real phi, int pp, part_struct *parts);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pp -- Lamb's coefficient access index helper value
 *  * parts -- particle information
 ******
 */

/****f* cuda_particle/Z_pn<<<>>>()
 * NAME
 *  Z_pn<<<>>>()
 * USAGE
 */
__device__ real Z_pn(int n, real theta, real phi, int pp, part_struct *parts);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMETNS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pp -- particle index
 *  * parts -- particle information structure
 ******
 */

/****f* cuda_particle_kernel/Z_phin<<<>>>()
 * NAME
 *  Z_phin<<<>>>()
 * USAGE
 */
__device__ real Z_phin(int n, real theta, real phi, int pp, part_struct *parts);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pp -- Lamb's coefficient access index helper value
 *  * parts -- particle information
 ******
 */

/****f* cuda_particle_kernel/Z_chin<<<>>>()
 * NAME
 *  Z_chin<<<>>>()
 * USAGE
 */
__device__ real Z_chin(int n, real theta, real phi, int pp, part_struct *parts);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pp -- Lamb's coefficient access index helper value
 *  * parts -- particle information
 ******
 */

/****f* bluebottle_kernel/internal_u<<<>>>()
 * NAME
 *  internal_u<<<>>>()
 * TYPE
 */
__global__ void internal_u(real *u, part_struct *parts, int *flag_u, int *phase, int nparts);
/* PURPOSE
 *  CUDA device kernel to apply particle solid-body motion to internal
 *  velocity nodes.
 * ARGUMENTS
 *  * u -- device velocity field
 *  * parts -- device particle struct
 *  * flag_u -- device flag field
 *  * phase -- device phase mask field
 ******
 */

/****f* bluebottle_kernel/internal_v<<<>>>()
 * NAME
 *  internal_v<<<>>>()
 * TYPE
 */
__global__ void internal_v(real *v, part_struct *parts, int *flag_v, int *phase, int nparts);
/* PURPOSE
 *  CUDA device kernel to apply particle solid-body motion to internal
 *  velocity nodes.
 * ARGUMENTS
 *  * v -- device velocity field
 *  * parts -- device particle struct
 *  * flag_v -- device flag field
 *  * phase -- device phase mask field
 ******
 */

/****f* bluebottle_kernel/internal_w<<<>>>()
 * NAME
 *  internal_w<<<>>>()
 * TYPE
 */
__global__ void internal_w(real *w, part_struct *parts, int *flag_w, int *phase, int nparts);
/* PURPOSE
 *  CUDA device kernel to apply particle solid-body motion to internal
 *  velocity nodes.
 * ARGUMENTS
 *  * w -- device velocity field
 *  * parts -- device particle struct
 *  * flag_w -- device flag field
 *  * phase -- device phase mask field
 ******
 */

/****f* particle_kernel/collision_init<<<>>>()
 * NAME
 *  collision_init<<<>>>()
 * USAGE
 */
__global__ void collision_init(part_struct *parts, int nparts);
/*
 * FUNCTION
 *  Set interaction forces equal to zero to start.
 * ARGUMENTS
 *  * parts -- the device particle array subdomain
 *  * nparts -- the number of particles
 ******
 */

/****f* particle_kernel/spring_parts<<<>>>()
 * NAME
 *  spring_parts<<<>>>()
 * USAGE
 */
__global__ void spring_parts(part_struct *parts, int nparts, dom_struct *DOM);
/*
 * FUNCTION
 *  Calculate spring force pulling particle back to origin.
 * ARGUMENTS
 *  * parts -- the device particle array subdomain
 *  * nparts -- the number of particles
 *  * DOM -- global domain structure
 ******
 */

/****f* particle_kernel/collision_walls<<<>>>()
 * NAME
 *  collision_walls<<<>>>()
 * USAGE
 */
__global__ void collision_walls(part_struct *parts, int nparts, BC *bc, real eps,
  real mu, real rho_f, real nu, int interaction_length_ratio, real dt,
  dom_struct *DOM);
/*
 * FUNCTION
 *  Calculate collision forcing between particle i and all other particles.
 * ARGUMENTS
 *  * parts -- the device particle array subdomain
 *  * nparts -- the number of particles
 *  * bc -- boundary condition data
 *  * eps -- magnitude of forcing
 *  * mu -- fluid viscosity
 *  * interactionLengthRatio -- the compact support length for interactions
 *  * dt -- time step size
 *  * DOM -- global domain structure
 ******
 */

/****f* cuda_particle/collision_parts<<<>>>()
 * NAME
 *  collision_parts<<<>>>()
 * USAGE
 */
__global__ void collision_parts(part_struct *parts, int nparts,
  real eps, real mu, real rho_f, real nu, BC *bc,
  int *bin_start, int *bin_end, int *part_bin, int *part_ind,
  int interaction_length_ratio, real dt);
/*
 * FUNCTION
 *  Calculate collision forcing between particle i and all other particles.
 * ARGUMENTS
 *  * parts -- the device particle array subdomain
 *  * nparts -- the number of particles in the domain
 *  * dom -- the device domain array
 *  * eps -- magnitude of forcing
 *  * mu -- fluid viscosity
 *  * rho_f -- fluid density
 *  * nu -- fluid viscosity
 *  * bc -- boundary condition data
 *  * bin_start -- index of (sorted) partBin where each bin starts
 *  * bin_end -- index of (sorted) partBin where each bin ends
 *  * part_bin -- for each particle, give bin
 *  * part_ind -- corresponding particle index for partBin
 *  * interaction_length_ratio -- the compact support length for interactions
 *  * dt -- time step size
 ******
 */

/****f* cuda_particle/pack_forces_e<<<>>>()
 * NAME
 *  pack_forces_e<<<>>>()
 * USAGE
 */
__global__ void pack_forces_e(real *force_send_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_send_e -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/pack_forces_w<<<>>>()
 * NAME
 *  pack_forces_w<<<>>>()
 * USAGE
 */
__global__ void pack_forces_w(real *force_send_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_send_w -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/pack_forces_n<<<>>>()
 * NAME
 *  pack_forces_n<<<>>>()
 * USAGE
 */
__global__ void pack_forces_n(real *force_send_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_send_n -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/pack_forces_s<<<>>>()
 * NAME
 *  pack_forces_s<<<>>>()
 * USAGE
 */
__global__ void pack_forces_s(real *force_send_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_send_s -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/pack_forces_t<<<>>>()
 * NAME
 *  pack_forces_t<<<>>>()
 * USAGE
 */
__global__ void pack_forces_t(real *force_send_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_send_t -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/pack_forces_b<<<>>>()
 * NAME
 *  pack_forces_b<<<>>>()
 * USAGE
 */
__global__ void pack_forces_b(real *force_send_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Pack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_send_b -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_send
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/unpack_forces_e<<<>>>()
 * NAME
 *  unpack_forces_e<<<>>>()
 * USAGE
 */
__global__ void unpack_forces_e(real *force_recv_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Unpack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_recv_e -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_recv
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/unpack_forces_w<<<>>>()
 * NAME
 *  unpack_forces_w<<<>>>()
 * USAGE
 */
__global__ void unpack_forces_w(real *force_recv_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Unpack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_recv_w -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_recv
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/unpack_forces_n<<<>>>()
 * NAME
 *  unpack_forces_n<<<>>>()
 * USAGE
 */
__global__ void unpack_forces_n(real *force_recv_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Unpack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_recv_n -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_recv
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/unpack_forces_s<<<>>>()
 * NAME
 *  unpack_forces_s<<<>>>()
 * USAGE
 */
__global__ void unpack_forces_s(real *force_recv_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Unpack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_recv_s -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_recv
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/unpack_forces_t<<<>>>()
 * NAME
 *  unpack_forces_t<<<>>>()
 * USAGE
 */
__global__ void unpack_forces_t(real *force_recv_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Unpack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_recv_t -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_recv
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

/****f* cuda_particle/unpack_forces_b<<<>>>()
 * NAME
 *  unpack_forces_b<<<>>>()
 * USAGE
 */
__global__ void unpack_forces_b(real *force_recv_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts);
/*
 * PURPOSE
 *  Unpack partial sums into contigous array for communication
 * ARGUMENTS
 *  * force_recv_b -- partial sums to be communicattd
 *  * offset -- offset of each bin in force_recv
 *  * bin_start -- starting index of each bin in part_ind
 *  * bin_count -- number of particles per bin
 *  * part_ind -- sorted index of particles, by bin number
 *  * parts -- particle info structure
 ******
 */

#endif
