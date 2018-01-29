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

/****h* Bluebottle/domain
 * NAME
 *  domain
 * FUNCTION
 *  Low-level domain and input functions
 ******
 */

#ifndef _DOMAIN_H
#define _DOMAIN_H 

/****v* domain/use_restart
 * NAME
 *  use_restart
 * TYPE
 */
extern int use_restart;
/*
 * PURPOSE
 *  Specify whether to use a restart file for input or not. "1" uses a restart,
 *  "0" reads the regular input files
 */

/****s* domain/grid_info
 * NAME
 *  grid_info
 * TYPE
 */
typedef struct grid_info {
  int is;
  int ie;
  int in;
  int isb;
  int ieb;
  int inb;
  int js;
  int je;
  int jn;
  int jsb;
  int jeb;
  int jnb;
  int ks;
  int ke;
  int kn;
  int ksb;
  int keb;
  int knb;
  int _is;
  int _ie;
  int _isb;
  int _ieb;
  int _js;
  int _je;
  int _jsb;
  int _jeb;
  int _ks;
  int _ke;
  int _ksb;
  int _keb;
  int s1;
  int s1b;
  int s2;
  int s2b;
  int s3;
  int s3b;
  int s2_i;
  int s2_j;
  int s2_k;
  int s2b_i;
  int s2b_j;
  int s2b_k;
} grid_info;
/*
 * PURPOSE
 *  Carry information related to the different discretization grids.
 * MEMBERS
 *  * is -- the domain start index in the x-direction (global indexing)
 *  * ie -- the domain end index in the x-direction (global indexing)
 *  * in -- the number of elements in the domain in the x-direction
 *  * isb -- the domain start index in the x-direction plus boundary ghost
 *    elements (global indexing)
 *  * ieb -- the domain end index in the x-direction plus boundary ghost
 *    elements (global indexing)
 *  * inb -- the number of elements in the domain in the x-direction plus
 *    the boundary ghost elements
 *  * js -- the domain start index in the y-direction (global indexing)
 *  * je -- the domain end index in the y-direction (global indexing)
 *  * jn -- the number of elements in the domain in the y-direction
 *  * jsb -- the domain start index in the y-direction plus boundary ghost
 *    elements (global indexing)
 *  * jeb -- the domain end index in the y-direction plus boundary ghost
 *    elements (global indexing)
 *  * jnb -- the number of elements in the domain in the y-direction plus
 *    the boundary ghost elements
 *  * ks -- the domain start index in the z-direction (global indexing)
 *  * ke -- the domain end index in the z-direction (global indexing)
 *  * kn -- the number of elements in the domain in the z-direction
 *  * ksb -- the domain start index in the z-direction plus boundary ghost
 *    elements (global indexing)
 *  * keb -- the domain end index in the z-direction plus boundary ghost
 *    elements (global indexing)
 *  * knb -- the number of elements in the domain in the z-direction plus
 *    the boundary ghost elements
 *  * _is -- the domain start index in the x-direction (local indexing)
 *  * _ie -- the domain end index in the x-direction (local indexing)
 *  * _isb -- the domain start index in the x-direction plus boundary ghost
 *    elements (local indexing)
 *  * _ieb -- the domain end index in the x-direction plus boundary ghost
 *    elements (local indexing)
 *  * _js -- the domain start index in the y-direction (local indexing)
 *  * _je -- the domain end index in the y-direction (local indexing)
 *  * _jsb -- the domain start index in the y-direction plus boundary ghost
 *    elements (local indexing)
 *  * _jeb -- the domain end index in the y-direction plus boundary ghost
 *    elements (local indexing)
 *  * _ks -- the domain start index in the z-direction (local indexing)
 *  * _ke -- the domain end index in the z-direction (local indexing)
 *  * _ksb -- the domain start index in the z-direction plus boundary ghost
 *    elements (local indexing)
 *  * _keb -- the domain end index in the z-direction plus boundary ghost
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

/****s* domain/dom_struct
 * NAME
 *  dom_struct
 * TYPE
 */
typedef struct dom_struct {
  grid_info Gcc;
  grid_info Gfx;
  grid_info Gfy;
  grid_info Gfz;
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
  int rank;
  int e;
  int w;
  int n;
  int s;
  int t;
  int b;
  int I;
  int Is;
  int Ie;
  int In;
  int J;
  int Js;
  int Je;
  int Jn;
  int K;
  int Ks;
  int Ke;
  int Kn;
  int S1;
  int S2;
  int S3;
} dom_struct;
/*
 * PURPOSE
 *  Carry information related to a subdomain.
 * MEMBERS
 *  * Gcc -- cell-centered grid information
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
 *  * rank -- rank of the current process
 *  * e -- the subdomain adjacent to the east face of the cell
 *    (e = -1 if the face is a domain boundary)
 *  * w -- the subdomain adjacent to the west face of the cell
 *    (w = -1 if the face is a domain boundary)
 *  * n -- the subdomain adjacent to the north face of the cell
 *    (n = -1 if the face is a domain boundary)
 *  * s -- the subdomain adjacent to the south face of the cell
 *    (s = -1 if the face is a domain boundary)
 *  * t -- the subdomain adjacent to the top face of the cell
 *    (t = -1 if the face is a domain boundary)
 *  * b -- the subdomain adjacent to the bottom face of the cell
 *    (b = -1 if the face is a domain boundary)
 *  * I -- for dom: index of MPI subdomain in x
 *  * Is -- for DOM: starting index of MPI subdomains in x direction
 *  * Ie -- for DOM: ending index of MPI subdomains in x direction
 *  * In -- for DOM: number of MPI subdomains in x direction.
 *  * J -- for dom: index of MPI subdomain in y
 *  * Js -- for DOM: starting index of MPI subdomains in y direction
 *  * Je -- for DOM: ending index of MPI subdomains in y direction
 *  * Jn -- for DOM: number of MPI subdomains in y direction.
 *  * K -- for dom: index of MPI subdomain in z
 *  * Ks -- for DOM: starting index of MPI subdomains in z direction
 *  * Ke -- for DOM: ending index of MPI subdomains in z direction
 *  * Kn -- for DOM: number of MPI subdomains in z direction.
 *  * S1 -- the looping stride length for the fastest-changing subdomain (x)
 *  * S2 -- the looping stride length for the fastest-changing subdomain (y)
 *  * S3 -- the looping stride length for the fastest-changing subdomain (z)
 ******
 */

/* FUNCTIONS */

/****f* domain/parse_cmdline_args()
 * NAME
 *  parse_cmdline_args()
 * USAGE
 */
void parse_cmdline_args(int argc, char *argv[]);
/*
 * FUNCTION
 *  Read command line arguments to bluebottle
 ******
 */

/****f* domain/domain_read_input()
 * NAME
 *  domain_read_input()
 * USAGE
 */
void domain_read_input(void);
/*
 * FUNCTION
 *  Read domain specifications and simulation parameters from flow.config.
 ******
 */

/****f* domain/domain_fill()
 * NAME
 *  domain_fill()
 * USAGE
 */
void domain_fill(void);
/*
 * FUNCTION
 *  Fill the domain on the host.
 ******
 */

/****f* domain/domain_write_config()
 * NAME
 *  domain_write_config()
 * USAGE
 */
void domain_write_config(void);
/*
 * FUNCTION
 *  Write domain specifications and simulation parameters to file.
 ******
 */

/****f* domain/compute_vel_BC()
 * NAME
 *  compute_vel_BC()
 * USAGE
 */
void compute_vel_BC(void);
/*
 * FUNCTION
 *  Set up the Dirichlet velocity boundary conditions given the maximum and
 *  acceleration
 ******
 */

/****f* domain/domain_init_fields()
 * NAME
 *  domain_init_fields()
 * USAGE
 */
void domain_init_fields(void);
/*
 * FUNCTION
 *  Initialize fields with given boundary and initial conditions
 ******
 */

/****f* domain/init_quiescent()
 * NAME
 *  init_quiescent()
 * USAGE
 */
void init_quiescent(void);
/*
 * FUNCTION
 *  Initialize fields with quiescent initial condition.
 ******
 */

/****f* domain/init_shear()
 * NAME
 *  init_shear()
 * USAGE
 */
void init_shear(void);
/*
 * FUNCTION
 *  Initialize fields with shear initial condition.
 ******
 */

/****f* domain/init_channel()
 * NAME
 *  init_channel()
 * USAGE
 */
void init_channel(void);
/*
 * FUNCTION
 *  Initialize fields with channel initial condition.
 ******
 */

/****f* domain/init_turb_channel()
 * NAME
 *  init_turb_channel()
 * USAGE
 */
void init_turb_channel(void);
/*
 * FUNCTION
 *  Initialize fields with turb_channel initial condition.
 ******
 */

/****f* domain/init_hit()
 * NAME
 *  init_hit()
 * USAGE
 */
void init_hit(void);
/*
 * FUNCTION
 *  Initialize fields with linearly forced HIT initial condition.
 ******
 */

/****f* domain/init_tg3()
 * NAME
 *  init_tg3()
 * USAGE
 */
void init_tg3(void);
/*
 * FUNCTION
 *  Initialize fields with tg3 initial condition.
 ******
 */
 
/****f* domain/init_tg()
 * NAME
 *  init_tg()
 * USAGE
 */
void init_tg(void);
/*
 * FUNCTION
 *  Initialize fields with tg initial condition.
 ******
 */

/****f* bluebottle/out_restart()
 * NAME
 *  out_restart()
 * USAGE
 */
void out_restart(void);
/*
 * FUNCTION
 *  Write the data required for restarting the simulation to file.
 ******
 */

/****f* bluebottle/in_restart()
 * NAME
 *  in_restart()
 * USAGE
 */
void in_restart(void);
/*
 * FUNCTION
 *  Read the data required for restarting the simulation to file.
 ******
 */

/****f* domain/count_mem()
 * NAME
 *  count_mem()
 * USAGE
 */
void count_mem(void);
/*
 * FUNCTION
 *  Displays total estimated CPU/GPU memory
 ******
 */

/****f* domain/domain_free()
 * NAME
 *  domain_free()
 * USAGE
 */
void domain_free(void);
/*
 * FUNCTION
 *  Free memory allocated for domain
 ******
 */

#endif // _DOMAIN_H
