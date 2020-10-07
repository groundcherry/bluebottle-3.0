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

/****h* Bluebottle/particle
 * NAME
 *  domain
 * FUNCTION
 *  Host side particle functions
 ******
 */

#ifndef _PARTICLE_H
#define _PARTICLE_H

/* #DEFINE's */

/****d* particle/NNODES
 * NAME
 *  NNODES
 * TYPE
 */
#define NNODES 26
/*
 * PURPOSE
 *  Define the number of nodes used for the Lebedev quadrature scheme.
 *  This could also be defined in physalis.h, but is fine here.
 ******
 */

/****d* particle/MAX_COEFFS
 * NAME
 *  MAX_COEFFS
 * TYPE
 */
#define MAX_COEFFS 15
/*
 * PURPOSE
 *  Define the maximum possible number of coefficients to use given that the 
 *  quadrature can handle up to 4th order. Note that this is different from
 *  ncoeffs_max, which is the actual maximum number of coefficients that we've
 *  specified over all of the particles.
 ******
 */

/****d* particle/S_MAX_COEFFS
 * NAME
 *  S_MAX_COEFFS
 * TYPE
 */
#define S_MAX_COEFFS 25
/*
 * PURPOSE
 *  Define the maximum possible number of coefficients to use given that the
 *  quadrature can handle up to 4th order. This one is for scalar.
 ******
 */

/****d* particle/MAX_NEIGHBORS
 * NAME
 *  MAX_NEIGHBORS
 * TYPE
 */
#define MAX_NEIGHBORS 12
/*
 * PURPOSE
 *  Define the maximum number of neighbors a particle can have (close packed).
 ******
 */

/* VARIABLES */

/****v* particle/NPARTS
 * NAME
 *  NPARTS
 * USAGE
 */
extern int NPARTS;
/*
 * FUNCTION
 *  Define the total number of particles in the global domain
 ******
 */

/****v* particle/nparts
 * NAME
 *  nparts
 * USAGE
 */
extern int nparts;
/*
 * FUNCTION
 *  Define the total number of particles in the subdomain including ghost
 *  particles
 ******
 */

/****v* particle/nparts_subdom
 * NAME
 *  nparts_subdom
 * USAGE
 */
extern int nparts_subdom;
/*
 * FUNCTION
 *  Define the total number of particles in the subdomain, not including ghosts
 ******
 */

/****v* particle/volume_fraction
 * NAME
 *  volume_fraction
 * USAGE
 */
extern real volume_fraction;
/*
 * FUNCTION
 *  Mean bulk volume fraction
 ******
 */

/****v* particle/total_parts_mass;
 * NAME
 *  total_parts_mass;
 * USAGE
 */
extern real total_parts_mass;
/*
 * FUNCTION
 *  Total particle mass
 ******
 */

/****v* particle/total_parts_volume;
 * NAME
 *  total_parts_volume;
 * USAGE
 */
extern real total_parts_volume;
/*
 * FUNCTION
 *  Total particle volume
 ******
 */

/****v* particle/rho_avg;
 * NAME
 *  rho_avg;
 * USAGE
 */
extern real rho_avg;
/*
 * FUNCTION
 *  Average mixture density
 ******
 */

/****v* particle/interaction_length_ratio
 * NAME
 *  interaction_length_ratio
 * USAGE
 */
extern real interaction_length_ratio;
/*
 * FUNCTION
 *  Defines the particle-particle interaction length ratio (e.g. normalized by
 *  the radius)
 ******
 */

/****v* particle/bin_size
 * NAME
 *  bin_size
 * USAGE
 */
extern real bin_size;
/*
 * FUNCTION
 *  Size of a bin, based off particle radius and (lubrication) contact length
 ******
 */

/****v* particle/ncoeffs_max
 * NAME
 *  ncoeffs_max
 * USAGE
 */
extern int ncoeffs_max;
/*
 * FUNCTION
 *  Maximum particle coefficient size
 ******
 */

/****v* domain/flag_u
  * NAME
  *  flag_u
  * TYPE
  */
extern int *flag_u;
/*
 * PURPOSE
 *  Flag x-direction components of velocity field that are set as boundaries.
 *  1 is fluid, 0 is boundary
 ******
 */
 
/****v* domain/flag_v
  * NAME
  *  flag_v
  * TYPE
  */
extern int *flag_v;
/*
 * PURPOSE
 *  Flag y-direction components of velocity field that are set as boundaries
 ******
 */

/****v* domain/flag_w
  * NAME
  *  flag_w
  * TYPE
  */
extern int *flag_w;
/*
 * PURPOSE
 *  Flag z-direction components of velocity field that are set as boundaries
 ******
 */

/****v* domain/_flag_u
  * NAME
  *  _flag_u
  * TYPE
  */
extern int *_flag_u;
/*
 * PURPOSE
 *  CUDA device analog for flag_u.
 ******
 */
 
/****v* domain/_flag_v
  * NAME
  *  _flag_v
  * TYPE
  */
extern int *_flag_v;
/*
 * PURPOSE
 *  CUDA device analog for flag_v.
 ******
 */

/****v* domain/_flag_w
  * NAME
  *  _flag_w
  * TYPE
  */
extern int *_flag_w;
/*
 * PURPOSE
 *  CUDA device analog for flag_w.
 ******
 */

/****v* particle/phase
 * NAME
 *  phase
 * USAGE
 */
extern int *phase;
/*
 * FUNCTION
 *  The phase of a discretized cell (Gcc-type grid).  If cell C is not inside
 *  a particle, then phase[C] = -1.  Otherwise, phase[C] is equal to the index 
 *  assigned to the particle in which the cell resides.
 ******
 */

/****v* particle/phase_shell
 * NAME
 *  phase_shell
 * USAGE
 */
extern int *phase_shell;
/*
 * FUNCTION
 *  The outermost shell of phase, denoting the positions for Dirichlet pressure
 *  boundary conditions
 ******
 */

/****v* particle/_phase
 * NAME
 *  _phase
 * USAGE
 */
extern int *_phase;
/*
 * FUNCTION
 *  CUDA device analog for phase.
 ******
 */

/****v* particle/_phase_shell
 * NAME
 *  _phase_shell
 * USAGE
 */
extern int *_phase_shell;
/*
 * FUNCTION
 *  CUDA device analog for phase_shell.
 ******
 */

/****s* particle/part_struct
 * NAME
 *  part_struct
 * TYPE
 */
typedef struct part_struct {
  int N;
  real r;
  real x;
  real y;
  real z;
  real u;
  real v;
  real w;
  real u0;
  real v0;
  real w0;
  real udot;
  real vdot;
  real wdot;
  real udot0;
  real vdot0;
  real wdot0;
  real axx;
  real axy;
  real axz;
  real ayx;
  real ayy;
  real ayz;
  real azx;
  real azy;
  real azz;
  real ox;
  real oy;
  real oz;
  real ox0;
  real oy0;
  real oz0;
  real oxdot;
  real oydot;
  real ozdot;
  real oxdot0;
  real oydot0;
  real ozdot0;
  real Fx;
  real Fy;
  real Fz;
  real Lx;
  real Ly;
  real Lz;
  real aFx;
  real aFy;
  real aFz;
  real aLx;
  real aLy;
  real aLz;
  real kFx;
  real kFy;
  real kFz;
  real iFx;
  real iFy;
  real iFz;
  real iLx;
  real iLy;
  real iLz;
  int nodes[NNODES];
  real rho;
  real E;
  real sigma;
  int order;
  real rs;
  int ncoeff;
  real spring_k;
  real spring_x;
  real spring_y;
  real spring_z;
  real spring_l;
  int translating;
  int rotating;
  real St[MAX_NEIGHBORS];
  int iSt[MAX_NEIGHBORS];
  real e_dry;
  real coeff_fric;
  real pnm_re[MAX_COEFFS];
  real pnm_im[MAX_COEFFS];
  real phinm_re[MAX_COEFFS];
  real phinm_im[MAX_COEFFS];
  real chinm_re[MAX_COEFFS];
  real chinm_im[MAX_COEFFS];
  real pnm_re0[MAX_COEFFS];
  real pnm_im0[MAX_COEFFS];
  real phinm_re0[MAX_COEFFS];
  real phinm_im0[MAX_COEFFS];
  real chinm_re0[MAX_COEFFS];
  real chinm_im0[MAX_COEFFS];
  int ncoll_part;
  int ncoll_wall;
  real s;
  int update;
  real srs;
  real q;
  real cp;
  int sorder;
  int sncoeff;
  real anm_re[S_MAX_COEFFS];
  real anm_im[S_MAX_COEFFS];
  real anm_re0[S_MAX_COEFFS];
  real anm_im0[S_MAX_COEFFS];
} part_struct;
/*
 * PURPOSE
 *  Carry physical information regarding a particle.
 *  XXX NOTE: mpi_part_struct in mpi_comm.c:mpi_parts_init needs to reflect
 *    any changes to this part_struct
 *  XXX NOTE: cgns and vtk outputs need to reflect any changes as well
 * MEMBERS
 *  * N -- global particle number
 *  * r -- the particle radius
 *  * x -- particle x location
 *  * y -- particle y location
 *  * z -- particle z location
 *  * u -- particle x velocity
 *  * v -- particle y velocity
 *  * w -- particle z velocity
 *  * u0 -- particle x velocity at previous timestep
 *  * v0 -- particle y velocity at previous timestep
 *  * w0 -- particle z velocity at previous timestep
 *  * udot -- linear acceleration in x-direction
 *  * vdot -- linear acceleration in y-direction
 *  * wdot -- linear acceleration in z-direction
 *  * axx -- x-component of basis vector initially coincident with x-axis
 *  * axy -- y-component of basis vector initially coincident with x-axis
 *  * axz -- z-component of basis vector initially coincident with x-axis
 *  * ayx -- x-component of basis vector initially coincident with y-axis
 *  * ayy -- y-component of basis vector initially coincident with y-axis
 *  * ayz -- z-component of basis vector initially coincident with y-axis
 *  * azx -- x-component of basis vector initially coincident with z-axis
 *  * azy -- y-component of basis vector initially coincident with z-axis
 *  * azz -- z-component of basis vector initially coincident with z-axis
 *  * ox -- angular velocity in x-direction
 *  * oy -- angular velocity in y-direction
 *  * oz -- angular velocity in z-direction
 *  * ox0 -- angular velocity in x-direction (previous step)
 *  * oy0 -- angular velocity in y-direction (previous step)
 *  * oz0 -- angular velocity in z-direction (previous step)
 *  * oxdot -- angular acceleration in x-direction
 *  * oydot -- angular acceleration in y-direction
 *  * ozdot -- angular acceleration in z-direction
 *  * Fx -- hydrodynamic force in the x-direction
 *  * Fy -- hydrodynamic force in the y-direction
 *  * Fz -- hydrodynamic force in the z-direction
 *  * Lx -- hydrodynamic moment in the x-direction
 *  * Ly -- hydrodynamic moment in the y-direction
 *  * Lz -- hydrodynamic moment in the z-direction
 *  * aFx -- applied force in the x-direction
 *  * aFy -- applied force in the y-direction
 *  * aFz -- applied force in the z-direction
 *  * aLx -- applied moment in the x-direction
 *  * aLy -- applied moment in the y-direction
 *  * aLz -- applied moment in the z-direction
 *  * kFx -- applied spring force in the x-direction
 *  * kFy -- applied spring force in the y-direction
 *  * kFz -- applied spring force in the z-direction
 *  * iFx -- interaction force in the x-direction
 *  * iFy -- interaction force in the y-direction
 *  * iFz -- interaction force in the z-direction
 *  * iLx -- interaction moment in the x-direction
 *  * iLy -- interaction moment in the y-direction
 *  * iLz -- interaction moment in the z-direction
 *  * nodes -- the status of the nodes
 *  * rho -- particle density
 *  * E -- particle Young's modulus
 *  * sigma -- particle Poisson ratio (-1 < sigma <= 0.5)
 *  * order -- the order above which to truncate the Lamb's series solution
 *  * rs -- the radius of integration for scalar products
 *  * ncoeff -- the number of Lamb's coefficients required order truncation
 *  * spring_k -- strength of spring pulling particle back to origin
 *  * spring_x -- x location of spring connection
 *  * spring_y -- y location of spring connection
 *  * spring_z -- z location of spring connection
 *  * spring_l -- the relaxed length of the spring
 *  * translating -- 1 if allowed to translate, 0 if not
 *  * rotating -- 1 if allowed to rotate, 0 if not
 *  * e_dry -- dry coefficient of restitution
 *  * coeff_fric -- coefficient of friction
 *  * St -- particle contact Stokes number list
 *  * iSt -- particle contact Stokes number indices
 *  * pnm_re -- real parts of Lamb's coefficients p_nm
 *  * pnm_im -- imaginary parts of Lamb's coefficients p_nm
 *  * phinm_re -- real parts of Lamb's coefficients phi_nm
 *  * phinm_im -- imaginary parts of Lamb's coefficients phi_nm
 *  * chinm_re -- real parts of Lamb's coefficients chi_nm
 *  * chinm_im -- imaginary parts of Lamb's coefficients chi_nm
 *  * pnm_re0 -- real parts of Lamb's coefficients p_nm at previous step
 *  * pnm_im0 -- imaginary parts of Lamb's coefficients p_nm at previous step
 *  * phinm_re0 -- real parts of Lamb's coefficients phi_nm at previous step
 *  * phinm_im0 -- imaginary parts of Lamb's coefficients phi_nm at previous step
 *  * chinm_re0 -- real parts of Lamb's coefficients chi_nm at previous step
 *  * chinm_im0 -- imaginary parts of Lamb's coefficients chi_nm at previous step
 *  * ncoll_part -- cumulative number of particle collisions each particle has seen
 *  * ncoll_wall -- cumulative number of wall collisions each particle has seen
 ******
 */

/****s* particle/bin_grid_info
 * NAME
 *  bin_grid_info
 * TYPE
 */
typedef struct bin_grid_info {
  int _is;
  int _ie;
  int _isb;
  int _ieb;
  int in;
  int inb;
  int _js;
  int _je;
  int _jsb;
  int _jeb;
  int jn;
  int jnb;
  int _ks;
  int _ke;
  int _ksb;
  int _keb;
  int kn;
  int knb;
  int s1;
  int s2;
  int s3;
  int s1b;
  int s2b;
  int s3b;
  int s2_i;
  int s2_j;
  int s2_k;
  int s2b_i;
  int s2b_j;
  int s2b_k;
} bin_grid_info;
/*
 * PURPOSE
 *  Define bin grid. Essentially a stripped down grid_info
 * MEMBERS
 *  * _is -- the domain start index in the x-direction (local indexing)
 *  * _ie -- the domain end index in the x-direction (local indexing)
 *  * _isb -- the domain start index in the x-direction plus boundary ghost
 *    elements (local indexing)
 *  * _ieb -- the domain end index in the x-direction plus boundary ghost
 *  * in -- the number of elements in the domain in the x-direction
 *  * inb -- the number of elements in the domain in the x-direction plus
 *    the boundary ghost elements
 *    elements (local indexing)
 *  * _js -- the domain start index in the y-direction (local indexing)
 *  * _je -- the domain end index in the y-direction (local indexing)
 *  * _jsb -- the domain start index in the y-direction plus boundary ghost
 *    elements (local indexing)
 *  * _jeb -- the domain end index in the y-direction plus boundary ghost
 *    elements (local indexing)
 *  * jn -- the number of elements in the domain in the y-direction
 *  * jnb -- the number of elements in the domain in the y-direction plus
 *    the boundary ghost elements
 *  * _ks -- the domain start index in the z-direction (local indexing)
 *  * _ke -- the domain end index in the z-direction (local indexing)
 *  * _ksb -- the domain start index in the z-direction plus boundary ghost
 *    elements (local indexing)
 *  * _keb -- the domain end index in the z-direction plus boundary ghost
 *  * kn -- the number of elements in the domain in the z-direction
 *  * knb -- the number of elements in the domain in the z-direction plus
 *    the boundary ghost elements
 *    elements (local indexing)
 *  * s1 -- the looping stride length for the fastest-changing variable (x)
 *  * s1b -- the looping stride length for the fastest-changing variable (x)
 *    plus the boundary ghost elements
 *  * s2 -- the looping stride length for the second-changing variable (y)
 *  * s2b -- the looping stride length for the second-changing variable (y)
 *    plus the boundary ghost elements
 *  * s3 -- the looping stride length for the slowest-changing variable (z)
 *  * s3b -- the looping stride length for the slowest-changing variable (z)
 *    plus the boundary ghost elements
 *  * s2_i -- size of the outermost east/west computational plane
 *  * s2_j -- size of the outermost north/south computational plane
 *  * s2_k -- size of the outermost top/bottom computational plane
 *  * s2b_i -- size of the outermost east/west ghost cell plane
 *  * s2b_j -- size of the outermost north/south ghost cell plane
 *  * s2b_k -- size of the outermost top/bottom ghost cell plane
 ******
 */

/****s* particle/bin_struct
 * NAME
 *  bin_struct
 * TYPE
 */
typedef struct bin_struct {
  bin_grid_info Gcc;
  real xs;
  real xe;
  real xl;
  int xn;
  real dx;
  real ys;
  real ye;
  real yl;
  int yn;
  real dy;
  real zs;
  real ze;
  real zl;
  int zn;
  real dz;
} bin_struct;
/*
 * PURPOSE
 *  Carry physical information regarding bin domain. Essentially a stripped down
 *  dom_struct
 * MEMBERS
 *  * Gcc -- cell-centered bin grid information
 *  * xs -- physical start position in the x-direction
 *  * xe -- physical end position in the x-direction
 *  * xl -- physical length of the subdomain in the x-direction
 *  * xn -- number of discrete cells in the x-direction
 *  * dx -- cell size in the x-direction
 *  * ys -- physical start position in the y-direction
 *  * ye -- physical end position in the y-direction
 *  * yl -- physical length of the subdomain in the y-direction
 *  * yn -- number of discrete cells in the y-direction
 *  * dy -- cell size in the y-direction
 *  * zs -- physical start position in the z-direction
 *  * ze -- physical end position in the z-direction
 *  * zl -- physical length of the subdomain in the z-direction
 *  * zn -- number of discrete cells in the z-direction
 *  * dz -- cell size in the z-direction
 ******
 */

/****v* particle/parts
 * NAME
 *  parts
 * TYPE
 */
extern part_struct *parts;
/*
 * PURPOSE
 *  A list of all particle structures
 ******
 */

/****v* particle/_parts
 * NAME
 *  _parts
 * TYPE
 */
extern part_struct *_parts;
/*
 * PURPOSE
 *  CUDA device analog for parts.
 ******
 */

/****v* particle/bins
 * NAME
 *  bins;
 * TYPE
 */
extern bin_struct bins;
/*
 * PURPOSE
 *  Domain struct for particle contact bins. CUDA analog declared in 
 *  bluebottle.cuh
 ******
 */

/****v* particle/_bin_start
 * NAME
 *  _bin_start
 * TYPE
 */
extern int *_bin_start;
/*
 * PURPOSE
 *  CUDA device array that contains, for each bin, the starting index of
 *  particles belonging to the bin in _part_bin
 ******
 */

/****v* particle/_bin_end
 * NAME
 *  _bin_end
 * TYPE
 */
extern int *_bin_end;
/*
 * PURPOSE
 *  CUDA device array that contains, for each bin, the ending index of
 *  particles belonging to the bin in _part_bin. Inclusive, so loop should be
 *  over _bin_start[i] <= i <= _bin_end[i]
 ******
 */

/****v* particle/_part_ind
 * NAME
 *  _part_ind
 * TYPE
 */
extern int *_part_ind;
/*
 * PURPOSE
 *  CUDA device array that contains the index of each particle, sorted by its
 *  bin number
 ******
 */

/****v* particle/_part_bin
 * NAME
 *  _part_bin
 * TYPE
 */
extern int *_part_bin;
/*
 * PURPOSE
 *  CUDA device array that contains the bin of each particle, index-matched with
 *  part_ind
 ******
 */

/****v* particle/_bin_count
 * NAME
 *  _bin_count
 * TYPE
 */
extern int *_bin_count;
/*
 * PURPOSE
 *  CUDA device array that contains the number of particles in each bin
 ******
 */

/****v* particle/nparts_send
 * NAME
 *  nparts_send
 * TYPE
 */
extern int nparts_send[6];
/*
 * PURPOSE
 *  Host array that contains the number of particle to send to each of the
 *  cardinal directions. Indexed with the #DEFINE'd EAST/WEST/NORTH.. etc.
 ******
 */

/****v* particle/nparts_recv
 * NAME
 *  nparts_recv
 * TYPE
 */
extern int nparts_recv[6];
/*
 * PURPOSE
 *  Host array that contains the number of particle to recv from each of the
 *  cardinal directions. Indexed with the #DEFINE'd EAST/WEST/NORTH.. etc.
 ******
 */

/****v* particle/_send_parts_e
 * NAME
 *  _send_parts_e
 * TYPE
 */
extern part_struct *_send_parts_e;
/*
 * PURPOSE
 *  Holding array for particles to be sent east
 ******
 */

/****v* particle/_send_parts_w
 * NAME
 *  _send_parts_w
 * TYPE
 */
extern part_struct *_send_parts_w;
/*
 * PURPOSE
 *  Holding array for particles to be sent west
 ******
 */

/****v* particle/_send_parts_n
 * NAME
 *  _send_parts_n
 * TYPE
 */
extern part_struct *_send_parts_n;
/*
 * PURPOSE
 *  Holding array for particles to be sent north
 ******
 */

/****v* particle/_send_parts_s
 * NAME
 *  _send_parts_s
 * TYPE
 */
extern part_struct *_send_parts_s;
/*
 * PURPOSE
 *  Holding array for particles to be sent south
 ******
 */

/****v* particle/_send_parts_t
 * NAME
 *  _send_parts_t
 * TYPE
 */
extern part_struct *_send_parts_t;
/*
 * PURPOSE
 *  Holding array for particles to be sent top
 ******
 */

/****v* particle/_send_parts_b
 * NAME
 *  _send_parts_b
 * TYPE
 */
extern part_struct *_send_parts_b;
/*
 * PURPOSE
 *  Holding array for particles to be sent bottom
 ******
 */

/****v* particle/_recv_parts_e
 * NAME
 *  _recv_parts_e
 * TYPE
 */
extern part_struct *_recv_parts_e;
/*
 * PURPOSE
 *  Holding array for particles coming from the east
 ******
 */

/****v* particle/_recv_parts_w
 * NAME
 *  _recv_parts_w
 * TYPE
 */
extern part_struct *_recv_parts_w;
/*
 * PURPOSE
 *  Holding array for particles coming from the west
 *****
 */

/****v* particle/_recv_parts_n
 * NAME
 *  _recv_parts_n
 * TYPE
 */
extern part_struct *_recv_parts_n;
/*
 * PURPOSE
 *  Holding array for particles coming from the north
 ******
 */

/****v* particle/_recv_parts_s
 * NAME
 *  _recv_parts_s
 * TYPE
 */
extern part_struct *_recv_parts_s;
/*
 * PURPOSE
 *  Holding array for particles coming from the south
 *****
 */

/****v* particle/_recv_parts_t
 * NAME
 *  _recv_parts_t
 * TYPE
 */
extern part_struct *_recv_parts_t;
/*
 * PURPOSE
 *  Holding array for particles to be sent top
 ******
 */

/****v* particle/_recv_parts_b
 * NAME
 *  _recv_parts_b
 * TYPE
 */
extern part_struct *_recv_parts_b;
/*
 * PURPOSE
 *  Holding array for particles to be sent bottom
 ******
 */

/****v* cuda_physalis/_force_send_e
 * NAME
 *  _force_send_e
 * USAGE
 */
extern real *_force_send_e;
/* PURPOSE
 *  Contigous array of forces to be sent
 ******
 */

/****v* cuda_physalis/_force_send_w
 * NAME
 *  _force_send_w
 * USAGE
 */
extern real *_force_send_w;
/* PURPOSE
 *  Contigous array of forces to be sent
 ******
 */

/****v* cuda_physalis/_force_send_n
 * NAME
 *  _force_send_n
 * USAGE
 */
extern real *_force_send_n;
/* PURPOSE
 *  Contigous array of forces to be sent
 ******
 */

/****v* cuda_physalis/_force_send_s
 * NAME
 *  _force_send_s
 * USAGE
 */
extern real *_force_send_s;
/* PURPOSE
 *  Contigous array of forces to be sent
 ******
 */

/****v* cuda_physalis/_force_send_t
 * NAME
 *  _force_send_t
 * USAGE
 */
extern real *_force_send_t;
/* PURPOSE
 *  Contigous array of forces to be sent
 ******
 */

/****v* cuda_physalis/_force_send_b
 * NAME
 *  _force_send_b
 * USAGE
 */
extern real *_force_send_b;
/* PURPOSE
 *  Contigous array of forces to be sent
 ******
 */

/****v* cuda_physalis/_force_recv_e
 * NAME
 *  _force_recv_e
 * USAGE
 */
extern real *_force_recv_e;
/* PURPOSE
 *  Contigous array of forces to be received
 ******
 */

/****v* cuda_physalis/_force_recv_w
 * NAME
 *  _force_recv_w
 * USAGE
 */
extern real *_force_recv_w;
/* PURPOSE
 *  Contigous array of forces to be received
 ******
 */

/****v* cuda_physalis/_force_recv_n
 * NAME
 *  _force_recv_n
 * USAGE
 */
extern real *_force_recv_n;
/* PURPOSE
 *  Contigous array of forces to be received
 ******
 */

/****v* cuda_physalis/_force_recv_s
 * NAME
 *  _force_recv_s
 * USAGE
 */
extern real *_force_recv_s;
/* PURPOSE
 *  Contigous array of forces to be received
 ******
 */

/****v* cuda_physalis/_force_recv_t
 * NAME
 *  _force_recv_t
 * USAGE
 */
extern real *_force_recv_t;
/* PURPOSE
 *  Contigous array of forces to be received
 ******
 */

/****v* cuda_physalis/_force_recv_b
 * NAME
 *  _force_recv_b
 * USAGE
 */
extern real *_force_recv_b;
/* PURPOSE
 *  Contigous array of forces to be received
 ******
 */

/*** FUNCTIONS ***/

/****f* particle/parts_read_input()
 * NAME
 *  parts_read_input()
 * USAGE
 */
void parts_read_input(void);
/*
 * FUNCTION
 *  Read particle specificiations from part.config and fill part_struct for
 *  each domain
 * ARGUMENTS
 ******
 */

/****f* particle/parts_init()
 * NAME
 *  parts_init()
 * USAGE
 */
void parts_init(void);
/*
 * FUNCTION
 *  Initialize part_struct for various inputs
 * ARGUMENTS
 ******
 */

/****f* particle/flags_reset()
 * NAME
 *  flags_reset()
 * USAGE
 */
void flags_reset(void);
/*
 * FUNCTION
 *  Reinit the flag arrays to no boundaries (1)
 * ARGUMENTS
 ******
 */

/****f* particle/parts_print()
 * NAME
 *  parts_print()
 * USAGE
 */
void parts_print(void);
/*
 * FUNCTION
 *  Print particle info to file if running under debug
 * ARGUMENTS
 ******
 */

/****f* particle/init_bins()
 * NAME
 *  init_bins()
 * USAGE
 */
void init_bins(void);
/*
 * FUNCTION
 *  Initialize the bin struct
 * ARGUMENTS
 ******
 */

/****f* particle/bin_write_config()
 * NAME
 *  bin_write_config()
 * USAGE
 */
void bin_write_config(void);
/*
 * FUNCTION
 *  write bin config to file for debug output
 * ARGUMENTS
 ******
 */

/****f* particle/part_free()
 * NAME
 *  part_free()
 * USAGE
 */
void part_free(void);
/*
 * FUNCTION
 *  Free host side particle memory
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_part_malloc_host()
 * NAME
 *  cuda_part_malloc_host()
 * USAGE
 */
void cuda_part_malloc_host(void);
/*
 * FUNCTION
 *  Allocate the CUDA host-side particle memory
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_part_malloc_dev()
 * NAME
 *  cuda_part_malloc_dev()
 * USAGE
 */
void cuda_part_malloc_dev(void);
/*
 * FUNCTION
 *  Allocate the CUDA device-side particle memory
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_part_push()
 * NAME
 *  cuda_part_push()
 * USAGE
 */
void cuda_part_push(void);
/*
 * FUNCTION
 *  Copy host-side particle information to device memory.
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_part_pull()
 * NAME
 *  cuda_part_pull()
 * USAGE
 */
void cuda_part_pull(void);
/*
 * FUNCTION
 *  Copy device-side particle information to host memory. Does not include ghost
 *  particles.
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_part_pull_debug()
 * NAME
 *  cuda_part_pull_debug()
 * USAGE
 */
void cuda_part_pull_debug(void);
/*
 * FUNCTION
 *  Copy device-side particle information to host memory. Includes ghost
 *  particles
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_part_free()
 * NAME
 *  cuda_part_free()
 * USAGE
 */
void cuda_part_free(void);
/*
 * FUNCTION
 *  Free device side particle memory
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_transfer_parts_i()
 * NAME
 *  cuda_transfer_parts_i()
 * USAGE
 */
void cuda_transfer_parts_i(void);
/*
 * FUNCTION
 *  Transfer particles between subdomains in the i direction
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_transfer_parts_j()
 * NAME
 *  cuda_transfer_parts_j()
 * USAGE
 */
void cuda_transfer_parts_j(void);
/*
 * FUNCTION
 *  Transfer particles between subdomains in the j direction
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_transfer_parts_k()
 * NAME
 *  cuda_transfer_parts_k()
 * USAGE
 */
void cuda_transfer_parts_k(void);
/*
 * FUNCTION
 *  Transfer particles between subdomains in the k direction
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_move_parts()
 * NAME
 *  cuda_move_parts()
 * USAGE
 */
void cuda_move_parts(void);
/*
 * FUNCTION
 *  Integrate particle motion based on particle velocity
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_move_parts_sub()
 * NAME
 *  cuda_move_parts_sub()
 * USAGE
 */
void cuda_move_parts_sub(void);
/*
 * FUNCTION
 *  Integrate particle motion based on particle velocity subtimestep (do not
 *  move particles but calculate forces and update velocity)
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_update_part_velocity()
 * NAME
 *  cuda_update_part_velocity()
 * USAGE
 */
void cuda_update_part_velocity(void);
/*
 * FUNCTION
 *  Update the particle velocity at the end of the timestep.
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_update_part_position()
 * NAME
 *  cuda_update_part_position()
 * USAGE
 */
void cuda_update_part_position(void);
/*
 * FUNCTION
 *  Update the particle position at the end of the timestep and store.
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_part_BC()
 * NAME
 *  cuda_part_BC()
 * USAGE
 */
void cuda_part_BC(void);
/*
 * FUNCTION
 *  Apply the particle velocity boundary conditions to the flow domain.
 ******
 */

/****f* particle/cuda_part_BC_star()
 * NAME
 *  cuda_part_BC_star()
 * USAGE
 */
void cuda_part_BC_star(void);
/*
 * FUNCTION
 *  Enforce boundary conditions in intermediate velocity fields.  *See note
 *  in cuda_BC().
 ******
 */

/****f* particle/cuda_part_BC_p()
 * NAME
 *  cuda_part_BC_p()
 * USAGE
 */
void cuda_part_BC_p(void);
/*
 * FUNCTION
 *  Enforce boundary conditions in pressure-Poisson problem.
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_part_p_fill()
 * NAME
 *  cuda_part_p_fill()
 * USAGE
 */
void cuda_part_p_fill(void);
/*
 * FUNCTION
 *  Enforce boundary conditions in pressure-Poisson problem.
 * ARGUMENTS
 ******
 */

/****f* particle/cuda_parts_internal()
 * NAME
 *  cuda_parts_internal()
 * USAGE
 */
void cuda_parts_internal(void);
/*
 * FUNCTION
 *  Apply particle solid-body motion to internal velocity nodes.
 ******
 */

/****f* particle/cuda_update_part_forces_i()
 * NAME
 *  cuda_update_part_forces_i()
 * USAGE
 */
void cuda_update_part_forces_i(void);
/* FUNCTION
 *  Communicate updates to part forces in the i direction
 ******
 */

/****f* particle/cuda_update_part_forces_j()
 * NAME
 *  cuda_update_part_forces_j()
 * USAGE
 */
void cuda_update_part_forces_j(void);
/* FUNCTION
 *  Communicate updates to part forces in the j direction
 ******
 */

/****f* particle/cuda_update_part_forces_k()
 * NAME
 *  cuda_update_part_forces_k()
 * USAGE
 */
void cuda_update_part_forces_k(void);
/* FUNCTION
 *  Communicate updates to part forces in the k direction
 ******
 */


#endif // _PARTICLE_H
