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

/****h* Bluebottle/bluebottle
 * NAME
 *  bluebottle
 * FUNCTION
 *  Bluebottle main execution code and global variable declarations.
 ******
 */

#ifndef _BLUEBOTTLE_H
#define _BLUEBOTTLE_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <sys/time.h>

#ifdef DOUBLE
  typedef double real;
  #define mpi_real MPI_DOUBLE
#else
  typedef float real;
  #define mpi_real MPI_FLOAT
#endif

#include "domain.h"
#include "particle.h"
#include "mpi_comm.h"
#include "recorder.h"
#include "vtk.h"
#ifdef CGNS_OUTPUT
  #include "cgns.h"
#endif
#include "physalis.h"
#include "scalar.h"

/* MACROS */

/* Location macros
 * PURPOSE
 *  Find index in re-ordered strided arrays
 */
#define GCC_LOC(II, JJ, KK, S1, S2) ((II) + (JJ)*(S1) + (KK)*(S2))
#define GFX_LOC(II, JJ, KK, S1, S2) ((JJ) + (KK)*(S1) + (II)*(S2))
#define GFY_LOC(II, JJ, KK, S1, S2) ((KK) + (II)*(S1) + (JJ)*(S2))
#define GFZ_LOC(II, JJ, KK, S1, S2) ((II) + (JJ)*(S1) + (KK)*(S2))

/* #DEFINE VARIABLES */
/****d* Bluebottle/VERSION
 * NAME
 *  VERSION
 * TYPE
 */
#define VERSION "3.0"
/*
 * PURPOSE
 * Define the bluebottle version number
 * 0.1 -- Flow solver development
 * 0.5 -- Particle development
 * 2.0 -- Official release as Bluebottle-2.0
 */

/****d* Bluebottle/PI
 * NAME
 *  PI
 * TYPE
 */
#define PI 3.14159265358979323846
/*
 * PURPOSE
 * Define the constant pi
 */

/****d* Bluebottle/ROOT_DIR
 * NAME
 *  ROOT_DIR
 * TYPE
 */
#define ROOT_DIR "."
/*
 * PURPOSE
 * Define the root directory for the project.
 */

/****d* bluebottle/OUTPUT_DIR
 * NAME
 *  OUTPUT_DIR
 * TYPE
 */
#define OUTPUT_DIR "output"
/*
 * PURPOSE
 *  Define the output directory for the project.
 ******
 */

/****d* bluebottle/INPUT_DIR
 * NAME
 *  INPUT_DIR 
 * TYPE
 */
#define INPUT_DIR "input"
/*
 * PURPOSE
 *  Define the input directory for the project.
 ******
 */

/****d* bluebottle/DOM_BUF
 * NAME
 *  DOM_BUF
 * TYPE
 */
#define DOM_BUF 1
/*
 * PURPOSE
 *  Define the size of the domain boundary condition ghost cell buffer (the
 *  number of ghost cells on one side of a given domain direction).
 ******
 */

/****d* Bluebottle/ENV_LOCAL_RANK
 * NAME
 *  ENV_LOCAL_RANK
 * TYPE
 */
#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"
/*
 * PURPOSE
 * Define environment variable which reads the local rank of the current mpi 
 * process.
 * MVAPICH2: MV2_COMM_WORLD_LOCAL_RANK
 * OMPI: OMPI_COMM_WORLD_LOCAL_RANK
 */

/****d* bluebottle/CHAR_BUF_SIZE
 * NAME
 *  CHAR_BUF_SIZE
 * TYPE
 */
#define CHAR_BUF_SIZE 256
/*
 * PURPOSE
 *  Define the maximum length of a character buffer read.
 ******
 */

/****d* bluebottle/FILE_NAME_SIZE
 * NAME
 *  FILE_NAME_SIZE
 * TYPE
 */
#define FILE_NAME_SIZE 256
/*
 * PURPOSE
 *  Define the maximum length of a file name.
 ******
 */

/****d* bluebottle/MAX_THREADS_1D
 * NAME
 *  MAX_THREADS_1D
 * TYPE
 */
#define MAX_THREADS_1D 128
/*
 * PURPOSE
 *  Define the max number of threads per block on a CUDA device. Hardcoded but
 *  is device-specific 
 ******
 */

/****d* bluebottle/MAX_THREADS_DIM
 * NAME
 *  MAX_THREADS_DIM
 * TYPE
 */
#define MAX_THREADS_DIM 16
/*
 * PURPOSE
 *  Define the max number of threads per dimension per block on a CUDA device. 
 *  Hardcoded but is device-specific 
 ******
 */

/****d* bluebottle/PERIODIC
 * NAME
 *  PERIODIC
 * TYPE
 */
#define PERIODIC 0
/*
 * PURPOSE
 *  Define the periodic boundary condition type.
 ******
 */

/****d* bluebottle/DIRICHLET
 * NAME
 *  DIRICHLET
 * TYPE
 */
#define DIRICHLET 1
/*
 * PURPOSE
 *  Define the Dirichlet boundary condition type.
 ******
 */

/****d* bluebottle/NEUMANN
 * NAME
 *  NEUMANN
 * TYPE
 */
#define NEUMANN 2
/*
 * PURPOSE
 *  Define the Neumann boundary condition type.
 ******
 */

/****d* bluebottle/PRECURSOR
 * NAME
 *  PRECURSOR
 * TYPE
 */
#define PRECURSOR 3
/*
 * PURPOSE
 *  Define the turbulence precursor domain boundary condition type. This
 *  type of boundary is treated like a Dirichlet boundary, but takes its value
 *  from the precursor domain.
 ******
 */

/****d* bluebottle/QUIESCENT
 * NAME
 *  QUIESCENT
 * TYPE
 */
#define QUIESCENT 0
/*
 * PURPOSE
 *  Define the initial condition quiescent flow case.
 ******
 */

/****d* bluebottle/SHEAR
 * NAME
 *  SHEAR
 * TYPE
 */
#define SHEAR 1
/*
 * PURPOSE
 *  Define the initial condition shear case.
 ******
 */

/****d* bluebottle/CHANNEL
 * NAME
 *  CHANNEL
 * TYPE
 */
#define CHANNEL 2
/*
 * PURPOSE
 *  Define the initial condition channel case.
 ******
 */

/****d* bluebottle/TAYLOR_GREEN
 * NAME
 *  TAYLOR_GREEN
 * TYPE
 */
#define TAYLOR_GREEN 3
/*
 * PURPOSE
 *  Define the initial condition TAYLOR_GREEN case.
 ******
 */

/****d* bluebottle/TAYLOR_GREEN_3
 * NAME
 *  TAYLOR_GREEN_3
 * TYPE
 */
#define TAYLOR_GREEN_3 4
/*
 * PURPOSE
 *  Define the initial condition TAYLOR_GREEN_3 3d case.
 ******
 */

/****d* bluebottle/TURB_CHANNEL
 * NAME
 *  TURB_CHANNEL
 * TYPE
 */
#define TURB_CHANNEL 5
/*
 * PURPOSE
 *  Define the initial condition for turbulent channel flow
 ******
 */

/****d* bluebottle/TURBULENT
 * NAME
 *  TURBULENT
 * TYPE
 */
#define TURBULENT 6
/*
 * PURPOSE
 *  Define the initial condition for homogeneous isotropic turbulence
  * linearly-forced in physical space
 ******
 */

/****d* bluebottle/HOMOGENEOUS
 * NAME
 *  HOMOGENEOUS
 * TYPE
 */
#define HOMOGENEOUS 10
/*
 * PURPOSE
 *  Define the homogeneous outflow plane condition.
 ******
 */

/****d* bluebottle/WEST
 * NAME
 *  WEST
 * TYPE
 */
#define WEST 0
/*
 * PURPOSE
 *  Define the West boundary.
 ******
 */

/****d* bluebottle/EAST
 * NAME
 *  EAST
 * TYPE
 */
#define EAST 1
/*
 * PURPOSE
 *  Define the East boundary.
 ******
 */

/****d* bluebottle/SOUTH
 * NAME
 *  SOUTH
 * TYPE
 */
#define SOUTH 2
/*
 * PURPOSE
 *  Define the South boundary.
 ******
 */

/****d* bluebottle/NORTH
 * NAME
 *  NORTH
 * TYPE
 */
#define NORTH 3
/*
 * PURPOSE
 *  Define the North boundary.
 ******
 */

/****d* bluebottle/BOTTOM
 * NAME
 *  BOTTOM
 * TYPE
 */
#define BOTTOM 4
/*
 * PURPOSE
 *  Define the Bottom boundary.
 ******
 */

/****d* bluebottle/TOP
 * NAME
 *  TOP
 * TYPE
 */
#define TOP 5
/*
 * PURPOSE
 *  Define the Top boundary.
 ******
 */

/****d* bluebottle/GCC
 * NAME
 *  GCC
 * TYPE
 */
#define GCC 0
/*
 * PURPOSE
 *  Define the Gcc domain type
 ******
 */

/****d* bluebottle/GFX
 * NAME
 *  GFX
 * TYPE
 */
#define GFX 1
/*
 * PURPOSE
 *  Define the Gfx domain type
 ******
 */

/****d* bluebottle/GFY
 * NAME
 *  GFY
 * TYPE
 */
#define GFY 2
/*
 * PURPOSE
 *  Define the Gfy domain type
 ******
 */

/****d* bluebottle/GFZ
 * NAME
 *  GFZ
 * TYPE
 */
#define GFZ 3
/*
 * PURPOSE
 *  Define the Gfz domain type
 ******
 */

/* VARIABLES */

/****v* bluebottle/DOM
 * NAME
 *  DOM
 * TYPE
 */
extern dom_struct DOM;
/*
 * PURPOSE
 *  Contains global domain information
 ******
 */

/****v* bluebottle/_DOM
 * NAME
 *  _DOM
 * TYPE
 */
extern dom_struct *_DOM;
/*
 * PURPOSE
 *  CUDA device analog for DOM
 ******
 */

/****v* bluebottle/dom
 * NAME
 *  dom
 * TYPE
 */
extern dom_struct *dom;
/*
 * PURPOSE
 *  Contains domain information for each MPI process -- array of pointers to
 *  dom_structs
 ******
 */

/****v* bluebottle/rho_f
 * NAME
 *  rho_f
 * TYPE
 */
extern real rho_f;
/*
 * PURPOSE
 *  The fluid density.
 ******
 */

/****v* bluebottle/mu
 * NAME
 *  mu
 * TYPE
 */
extern real mu;
/*
 * PURPOSE
 *  The fluid dynamic viscosity.
 ******
 */

/****v* bluebottle/nu
 * NAME
 *  nu
 * TYPE
 */
extern real nu;
/*
 * PURPOSE
 *  The fluid kinematic viscosity.
 ******
 */

/****v* bluebottle/pp_residual
 * NAME
 *  pp_residual
 * TYPE
 */
extern real pp_residual;
/*
 * PURPOSE
 *  The maximum desired residual for the pressure-Poisson problem solver.
 ******
 */

/****v* bluebottle/pp_max_iter
 * NAME
 *  pp_max_iter
 * TYPE
 */
extern int pp_max_iter;
/*
 * PURPOSE
 *  The maximum number of iterations for the pressure-Poisson problem solver.
 ******
 */

/****v* bluebottle/lamb_max_iter
 * NAME
 *  lamb_max_iter
 * TYPE
 */
extern int lamb_max_iter;
/*
 * PURPOSE
 *  The maximum number of iterations for the Lamb's coefficient convergence
 ******
 */

/****v* bluebottle/lamb_residual
 * NAME
 *  lamb_residual
 * TYPE
 */
extern real lamb_residual;
/*
 * PURPOSE
 *  The maximum desired residual for the Lamb's coefficient iteration process
 ******
 */

/****v* bluebottle/lamb_relax
 * NAME
 *  lamb_relax
 * TYPE
 */
extern real lamb_relax;
/*
 * PURPOSE
 *  The underrelaxation factor for the Lamb's coefficient iteration process.
 *  Zero weights toward old value and unity weights toward new value
 ******
 */

/****v* bluebottle/lamb_cut
 * NAME
 *  lamb_cut
 * TYPE
 */
extern real lamb_cut;
/*
 * PURPOSE
 *  The magnitude below which errors in Lamb's coefficients are ignored,
 *  compared to the coefficient with greates magnitude. The lower this number,
 *  the more coefficients will be considered important when computing the error.
 *  To improve convergence rate, decrease this number. It should never be
 *  greater than 1e-2.
 ******
 */
extern real osci_f;
/****v* bluebottle/init_cond
 * NAME
 *  init_cond
 * TYPE
 */
extern int init_cond;
/*
 * PURPOSE
 *  Carries the initial condition type. For now, the only options are QUIESCENT,
 *  CHANNEL, and SHEAR.
 ******
 */

/****v* bluebottle/out_plane
 * NAME
 *  out_plane
 * TYPE
 */
extern int out_plane;
/*
 * PURPOSE
 *  Define which plane is the outlet plane.
 ******
 */

/****s* bluebottle/BC
 * NAME
 *  BC
 * TYPE
 */
typedef struct BC {
  int pW;
  int pE;
  int pS;
  int pN;
  int pB;
  int pT;
  int uW;
  real uWDm;
  real uWD;
  real uWDa;
  int uE;
  real uEDm;
  real uED;
  real uEDa;
  int uS;
  real uSDm;
  real uSD;
  real uSDa;
  int uN;
  real uNDm;
  real uND;
  real uNDa;
  int uB;
  real uBDm;
  real uBD;
  real uBDa;
  int uT;
  real uTDm;
  real uTD;
  real uTDa;
  int vW;
  real vWDm;
  real vWD;
  real vWDa;
  int vE;
  real vEDm;
  real vED;
  real vEDa;
  int vS;
  real vSDm;
  real vSD;
  real vSDa;
  int vN;
  real vNDm;
  real vND;
  real vNDa;
  int vB;
  real vBDm;
  real vBD;
  real vBDa;
  int vT;
  real vTDm;
  real vTD;
  real vTDa;
  int wW;
  real wWDm;
  real wWD;
  real wWDa;
  int wE;
  real wEDm;
  real wED;
  real wEDa;
  int wS;
  real wSDm;
  real wSD;
  real wSDa;
  int wN;
  real wNDm;
  real wND;
  real wNDa;
  int wB;
  real wBDm;
  real wBD;
  real wBDa;
  int wT;
  real wTDm;
  real wTD;
  real wTDa;
  real dsW;
  real dsE;
  real dsS;
  real dsN;
  real dsB;
  real dsT;
} BC;
/*
 * PURPOSE
 *  Carry the type of boundary condition on each side of the domain.  Possible
 *  types include:
 *  * PERIODIC
 *  * DIRICHLET
 *  * NEUMANN
 *  * PRECURSOR
 *  If the boundary type is DIRICHLET or PRECURSOR, the value of the field
 *  variable on the boundary must be defined.
 * MEMBERS
 *  * pW -- the pressure boundary condition type
 *  * pE -- the pressure boundary condition type
 *  * pS -- the pressure boundary condition type
 *  * pN -- the pressure boundary condition type
 *  * pB -- the pressure boundary condition type
 *  * pT -- the pressure boundary condition type
 *  * uW -- the boundary condition type
 *  * uWDm -- the maximum DIRICHLET boundary condition value
 *  * uWD -- the current DIRICHLET boundary conditon value
 *  * uWDa -- the DIRICHLET boundary condition value acceleration
 *  * uE -- the boundary condition type
 *  * uEDm -- the maximum DIRICHLET boundary condition value
 *  * uED -- the current DIRICHLET boundary conditon value
 *  * uEDa -- the DIRICHLET boundary condition value acceleration
 *  * uS -- the boundary condition type
 *  * uSDm -- the maximum DIRICHLET boundary condition value
 *  * uSD -- the current DIRICHLET boundary conditon value
 *  * uSDa -- the DIRICHLET boundary condition value acceleration
 *  * uN -- the boundary condition type
 *  * uND -- the maximum DIRICHLET boundary condition valuem
 *  * uND -- the current DIRICHLET boundary conditon value
 *  * uNDa -- the DIRICHLET boundary condition value acceleration
 *  * uB -- the boundary condition type
 *  * uBDm -- the maximum DIRICHLET boundary condition value
 *  * uBD -- the current DIRICHLET boundary conditon value
 *  * uBDa -- the DIRICHLET boundary condition value acceleration
 *  * uT -- the boundary condition type
 *  * uTDm -- the maximum DIRICHLET boundary condition value
 *  * uTD -- the current DIRICHLET boundary conditon value
 *  * uTDa -- the DIRICHLET boundary condition value acceleration
 *  * vW -- the boundary condition type
 *  * vWDm -- the maximum DIRICHLET boundary condition value
 *  * vWD -- the current DIRICHLET boundary conditon value
 *  * vWDa -- the DIRICHLET boundary condition value acceleration
 *  * vE -- the boundary condition type
 *  * vEDm -- the maximum DIRICHLET boundary condition value
 *  * vED -- the current DIRICHLET boundary conditon value
 *  * vEDa -- the DIRICHLET boundary condition value acceleration
 *  * vS -- the boundary condition type
 *  * vSDm -- the maximum DIRICHLET boundary condition value
 *  * vSD -- the current DIRICHLET boundary conditon value
 *  * vSDa -- the DIRICHLET boundary condition value acceleration
 *  * vN -- the boundary condition type
 *  * vNDm -- the maximum DIRICHLET boundary condition value
 *  * vND -- the current DIRICHLET boundary conditon value
 *  * vNDa -- the DIRICHLET boundary condition value acceleration
 *  * vB -- the boundary condition type
 *  * vBDm -- the maximum DIRICHLET boundary condition value
 *  * vBD -- the current DIRICHLET boundary conditon value
 *  * vBDa -- the DIRICHLET boundary condition value acceleration
 *  * vT -- the boundary condition type
 *  * vTDm -- the maximum DIRICHLET boundary condition value
 *  * vTD -- the current DIRICHLET boundary conditon value
 *  * vTDa -- the DIRICHLET boundary condition value acceleration
 *  * wW -- the boundary condition type
 *  * wWDm -- the maximum DIRICHLET boundary condition value
 *  * wWD -- the current DIRICHLET boundary conditon value
 *  * wWDa -- the DIRICHLET boundary condition value acceleration
 *  * wE -- the boundary condition type
 *  * wEDm -- the maximum DIRICHLET boundary condition value
 *  * wED -- the current DIRICHLET boundary conditon value
 *  * wEDa -- the DIRICHLET boundary condition value acceleration
 *  * wS -- the boundary condition type
 *  * wSDm -- the maximum DIRICHLET boundary condition value
 *  * wSD -- the current DIRICHLET boundary conditon value
 *  * wSDa -- the DIRICHLET boundary condition value acceleration
 *  * wN -- the boundary condition type
 *  * wNDm -- the maximum DIRICHLET boundary condition value
 *  * wND -- the current DIRICHLET boundary conditon value
 *  * wNDa -- the DIRICHLET boundary condition value acceleration
 *  * wB -- the boundary condition type
 *  * wBDm -- the maximum DIRICHLET boundary condition value
 *  * wBD -- the current DIRICHLET boundary conditon value
 *  * wBDa -- the DIRICHLET boundary condition value acceleration
 *  * wT -- the boundary condition type
 *  * wTDm -- the maximum DIRICHLET boundary condition value
 *  * wTD -- the current DIRICHLET boundary conditon value
 *  * wTDa -- the DIRICHLET boundary condition value acceleration
 *  * dsW -- the SCREEN boundary condition offset value
 *  * dsE -- the SCREEN boundary condition offset value
 *  * dsS -- the SCREEN boundary condition offset value
 *  * dsN -- the SCREEN boundary condition offset value
 *  * dsB -- the SCREEN boundary condition offset value
 *  * dsT -- the SCREEN boundary condition offset value
 ******
 */

/****v* bluebottle/bc
 * NAME
 *  bc
 * TYPE
 */
extern BC bc;
/*
 * PURPOSE
 *  Create an instance of the struct BC to carry boundary condition types.
 ******
 */

/****v* bluebottle/_bc
 * NAME
 *  _bc
 * TYPE
 */
extern BC *_bc;
/*
 * PURPOSE
 *  CUDA device analog for BC bc
 ******
 */

/****s* bluebottle/gradP_struct
 * NAME
 *  gradP_struct
 * TYPE
 */
typedef struct gradP_struct {
  real x;
  real xm;
  real xa;
  real y;
  real ym;
  real ya;
  real z;
  real zm;
  real za;
} gradP_struct;
/* 
 * PURPOSE
 *  Carry imposed pressure gradient values.
 ******
 */

/****v* bluebottle/gradP
 * NAME
 *  gradP
 * TYPE
 */
extern gradP_struct gradP;
/*
 * PURPOSE
 *  Create an instance of the struct gradP_struct to carry imposed pressure
 *  gradient values.
 ******
 */

/****s* bluebottle/g_struct
 * NAME
 *  g_struct
 * TYPE
 */
typedef struct g_struct {
  real x;
  real xm;
  real xa;
  real y;
  real ym;
  real ya;
  real z;
  real zm;
  real za;
} g_struct;
/*
 * PURPOSE
 *  Body force on flow.
 * MEMBERS
 *  * x -- x-direction
 *  * y -- y-direction
 *  * z -- z-direction
 ******
 */

/****v* bluebottle/g
 * NAME
 *  g
 * TYPE
 */
extern g_struct g;
/*
 * PURPOSE
 *  Create an instance of the struct g_struct to carry body forces.
 ******
 */

/****v* bluebottle/p
 * NAME
 *  p
 * TYPE
 */
extern real *p;
/*
 * PURPOSE
 *  Pressure field vector (grid type Gcc; x-component varies first, then
 *  y-component, then z-component).
 ******
 */

/****v* bluebottle/_p
 * NAME
 *  _p
 * TYPE
 */
extern real *_p;
/*
 * PURPOSE
 *  CUDA device analog for p. It is an array on each processor that contains the
 *  subdomain's field
 ******
 */

/****v* bluebottle/phi
 * NAME
 *  _phi
 * TYPE
 */
extern real *phi;
/*
 * PURPOSE
 *  Host array for phi. It contains the solution to the pressure
 *  poisson problem
 ******
 */

/****v* bluebottle/_phi
 * NAME
 *  _phi
 * TYPE
 */
extern real *_phi;
/*
 * PURPOSE
 *  CUDA device array for phi. It contains the solution to the pressure
 *  poisson problem
 ******
 */

/****v* bluebottle/_phinoghost
 * NAME
 *  _phinoghost
 * TYPE
 */
extern real *_phinoghost;
/*
 * PURPOSE
 *  CUDA device array for phinoghost. It contains the solution to the pressure
 *  poisson problem without the ghost cells.
 ******
 */

/****v* bluebottle/_invM
 * NAME
 *  _invM
 * TYPE
 */
extern real *_invM;
/*
 * PURPOSE
 *  CUDA device array for invM. It contains the diagonal jacobi preconditioner
 ******
 */

/****v* bluebottle/p0
 * NAME
 *  p0
 * TYPE
 */
extern real *p0;
/*
 * PURPOSE
 *  Host p stored from the previous timestep.
 ******
 */

/****v* bluebottle/_p0
 * NAME
 *  _p0
 * TYPE
 */
extern real *_p0;
/*
 * PURPOSE
 *  CUDA device analog for p stored from the previous timestep.
 ******
 */

/****v* bluebottle/u_star
  * NAME
  *  u_star
  * TYPE
  */
extern real *u_star;
/* 
  * PURPOSE
  * X-direction intermediate velocity. It is an array on each processor that
  * contains the subdomain's field.
  ******
  */

/****v* bluebottle/v_star
  * NAME
  *  v_star
  * TYPE
  */
extern real *v_star;
/* 
  * PURPOSE
  * Y-direction intermediate velocity. It is an array on each processor that
  * contains the subdomain's field.
  ******
  */

/****v* bluebottle/w_star
  * NAME
  *  w_star
  * TYPE
  */
extern real *w_star;
/* 
  * PURPOSE
  * X-direction intermediate velocity. It is an array on each processor that
  * contains the subdomain's field.
  ******
  */

/****v* bluebottle/_u_star
  * NAME
  *  _u_star
  * TYPE
  */
extern real *_u_star;
/* 
  * PURPOSE
  *  CUDA device array for the x-direction intermediate velocity
  ******
  */

/****v* bluebottle/_v_star
  * NAME
  *  _v_star
  * TYPE
  */
extern real *_v_star;
/* 
  * PURPOSE
  *  CUDA device array for the y-direction intermediate velocity
  ******
  */

/****v* bluebottle/_w_star
  * NAME
  *  _w_star
  * TYPE
  */
extern real *_w_star;
/* 
  * PURPOSE
  *  CUDA device array for the z-direction intermediate velocity
  ******
  */

/****v* bluebottle/u
 * NAME
 *  u
 * TYPE
 */
extern real *u;
/*
 * PURPOSE
 *  Velocity field vector u-component (grid type Gfx; x-component varies
 *  first, then y-component, then z-component).
 ******
 */

/****v* bluebottle/_u
 * NAME
 *  _u
 * TYPE
 */
extern real *_u;
/*
 * PURPOSE
 *  CUDA device analog for u.
 ******
 */

/****v* bluebottle/u0
 * NAME
 *  u0
 * TYPE
 */
extern real *u0;
/*
 * PURPOSE
 *  Host u stored from the previous timestep.
 ******
 */

/****v* bluebottle/_u0
 * NAME
 *  _u0
 * TYPE
 */
extern real *_u0;
/*
 * PURPOSE
 *  CUDA device analog for u stored from the previous timestep.
 ******
 */

/****v* bluebottle/v
 * NAME
 *  v
 * TYPE
 */
extern real *v;
/*
 * PURPOSE
 *  Velocity field vector v-component (grid type Gfy; x-component varies
 *  first, then y-component, then z-component).
 ******
 */

/****v* bluebottle/_v
 * NAME
 *  _v
 * TYPE
 */
extern real *_v;
/*
 * PURPOSE
 *  CUDA device analog for v.  
 ******
 */

/****v* bluebottle/v0
 * NAME
 *  v0
 * TYPE
 */
extern real *v0;
/*
 * PURPOSE
 *  Host v stored from the previous timestep.
 ******
 */

/****v* bluebottle/_v0
 * NAME
 *  _v0
 * TYPE
 */
extern real *_v0;
/*
 * PURPOSE
 *  CUDA device analog for v stored from the previous timestep.
 ******
 */

/****v* bluebottle/w
 * NAME
 *  w
 * TYPE
 */
extern real *w;
/*
 * PURPOSE
 *  Velocity field vector w-component (grid type Gfz; x-component varies
 *  first, then y-component, then z-component).
 ******
 */

/****v* bluebottle/_w
 * NAME
 *  _w
 * TYPE
 */
extern real *_w;
/*
 * PURPOSE
 *  CUDA device analog for w. 
 ******
 */

/****v* bluebottle/w0
 * NAME
 *  w0
 * TYPE
 */
extern real *w0;
/*
 * PURPOSE
 *  Host w stored from the previous timestep.
 ******
 */

/****v* bluebottle/_w0
 * NAME
 *  _w0
 * TYPE
 */
extern real *_w0;
/*
 * PURPOSE
 *  CUDA device analog for w stored from the previous timestep.
 ******
 */

/****v* bluebottle/f_x
 * NAME
 *  f_x
 * TYPE
 */
extern real *f_x;
/*
 * PURPOSE
 *  The body forcing in the x-direction.
 ******
 */

/****v* bluebottle/_f_x
 * NAME
 *  _f_x
 * TYPE
 */
extern real *_f_x;
/*
 * PURPOSE
 *  CUDA device analog for f_x.
 ******
 */

/****v* bluebottle/f_y
 * NAME
 *  f_y
 * TYPE
 */
extern real *f_y;
/*
 * PURPOSE
 *  The body forcing in the y-direction.
 ******
 */

/****v* bluebottle/_f_y
 * NAME
 *  _f_y
 * TYPE
 */
extern real *_f_y;
/*
 * PURPOSE
 *  CUDA device analog for f_y.
 ******
 */

/****v* bluebottle/f_z
 * NAME
 *  f_z
 * TYPE
 */
extern real *f_z;
/*
 * PURPOSE
 *  The body forcing in the z-direction.
 ******
 */

/****v* bluebottle/_f_z
 * NAME
 *  _f_z
 * TYPE
 */
extern real *_f_z;
/*
 * PURPOSE
 *  CUDA device analog for f_z.
 ******
 */

/****v* bluebottle/conv_u
 * NAME
 *  conv_u
 * TYPE
 */
extern real *conv_u;
/*
 * PURPOSE
 *  Host array to store the previous x-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/conv_v
 * NAME
 *  conv_v
 * TYPE
 */
extern real *conv_v;
/*
 * PURPOSE
 *  Host array to store the previous y-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/conv_w
 * NAME
 *  conv_w
 * TYPE
 */
extern real *conv_w;
/*
 * PURPOSE
 *  Host array to store the previous z-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_conv_u
 * NAME
 *  _conv_u
 * TYPE
 */
extern real *_conv_u;
/*
 * PURPOSE
 *  CUDA device array to store the previous x-component convection solution
 *  for use in the next Adams-Bashforth step. 
 ******
 */

/****v* bluebottle/_conv_v
 * NAME
 *  _conv_v
 * TYPE
 */
extern real *_conv_v;
/*
 * PURPOSE
 *  CUDA device array to store the previous y-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_conv_w
 * NAME
 *  _conv_w
 * TYPE
 */
extern real *_conv_w;
/*
 * PURPOSE
 *  CUDA device array to store the previous z-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/conv0_u
 * NAME
 *  conv0_u
 * TYPE
 */
extern real *conv0_u;
/*
 * PURPOSE
 *  Host array to store the previous x-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/conv0_v
 * NAME
 *  conv0_v
 * TYPE
 */
extern real *conv0_v;
/*
 * PURPOSE
 *  Host array to store the previous y-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/conv0_w
 * NAME
 *  conv0_w
 * TYPE
 */
extern real *conv0_w;
/*
 * PURPOSE
 *  Host array to store the previous z-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_conv0_u
 * NAME
 *  _conv0_u
 * TYPE
 */
extern real *_conv0_u;
/*
 * PURPOSE
 *  CUDA device array to store the previous x-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_conv0_v
 * NAME
 *  _conv0_v
 * TYPE
 */
extern real *_conv0_v;
/*
 * PURPOSE
 *  CUDA device array to store the previous y-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_conv0_w
 * NAME
 *  _conv0_w
 * TYPE
 */
extern real *_conv0_w;
/*
 * PURPOSE
 *  CUDA device array to store the previous z-component convection solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/diff_u
 * NAME
 *  diff_u
 * TYPE
 */
extern real *diff_u;
/*
 * PURPOSE
 *  Host array to store the x-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/diff_v
 * NAME
 *  diff_v
 * TYPE
 */
extern real *diff_v;
/*
 * PURPOSE
 *  Host array to store the y-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/diff_w
 * NAME
 *  diff_w
 * TYPE
 */
extern real *diff_w;
/*
 * PURPOSE
 *  Host array to store the z-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_diff_u
 * NAME
 *  _diff_u
 * TYPE
 */
extern real *_diff_u;
/*
 * PURPOSE
 *  CUDA device array to store the x-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_diff_v
 * NAME
 *  _diff_v
 * TYPE
 */
extern real *_diff_v;
/*
 * PURPOSE
 *  CUDA device array to store the y-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_diff_w
 * NAME
 *  _diff_w
 * TYPE
 */
extern real *_diff_w;
/*
 * PURPOSE
 *  CUDA device array to store the z-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/diff0_u
 * NAME
 *  diff0_u
 * TYPE
 */
extern real *diff0_u;
/*
 * PURPOSE
 *  Host array to store the previous x-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/diff0_v
 * NAME
 *  diff0_v
 * TYPE
 */
extern real *diff0_v;
/*
 * PURPOSE
 *  Host array to store the previous y-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/diff0_w
 * NAME
 *  diff0_w
 * TYPE
 */
extern real *diff0_w;
/*
 * PURPOSE
 *  Host array to store the previous z-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_diff0_u
 * NAME
 *  _diff0_u
 * TYPE
 */
extern real *_diff0_u;
/*
 * PURPOSE
 *  CUDA device array to store the previous x-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_diff0_v
 * NAME
 *  _diff0_v
 * TYPE
 */
extern real *_diff0_v;
/*
 * PURPOSE
 *  CUDA device array to store the previous y-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_diff0_w
 * NAME
 *  _diff0_w
 * TYPE
 */
extern real *_diff0_w;
/*
 * PURPOSE
 *  CUDA device array to store the previous z-component diffusion solution
 *  for use in the next Adams-Bashforth step.
 ******
 */

/****v* bluebottle/_send_Gcc_e
 * NAME
 *  _send_Gcc_e
 * TYPE
 */
extern real *_send_Gcc_e;
/*
 * PURPOSE
 *  CUDA contiguous package for eastern computational Gcc plane.
 ******
 */

/****v* bluebottle/_send_Gcc_w
 * NAME
 *  _send_Gcc_w
 * TYPE
 */
extern real *_send_Gcc_w;
/*
 * PURPOSE
 *  CUDA contiguous package for western computational Gcc plane.
 ******
 */

/****v* bluebottle/_send_Gcc_n
 * NAME
 *  _send_Gcc_n
 * TYPE
 */
extern real *_send_Gcc_n;
/*
 * PURPOSE
 *  CUDA contiguous package for northern computational Gcc plane.
 ******
 */

/****v* bluebottle/_send_Gcc_s
 * NAME
 *  _send_Gcc_s
 * TYPE
 */
extern real *_send_Gcc_s;
/*
 * PURPOSE
 *  CUDA contiguous package for southern computational Gcc plane.
 ******
 */

/****v* bluebottle/_send_Gcc_t
 * NAME
 *  _send_Gcc_t
 * TYPE
 */
extern real *_send_Gcc_t;
/*
 * PURPOSE
 *  CUDA contiguous package for top computational Gcc plane.
 ******
 */

/****v* bluebottle/_send_Gcc_b
 * NAME
 *  _send_Gcc_b
 * TYPE
 */
extern real *_send_Gcc_b;
/*
 * PURPOSE
 *  CUDA contiguous package for bottom computational Gcc plane.
 ******
 */

/****v* bluebottle/_recv_Gcc_e
 * NAME
 *  _recv_Gcc_e
 * TYPE
 */
extern real *_recv_Gcc_e;
/*
 * PURPOSE
 *  CUDA contiguous package for eastern ghost Gcc plane.
 ******
 */

/****v* bluebottle/_recv_Gcc_w
 * NAME
 *  _recv_Gcc_w
 * TYPE
 */
extern real *_recv_Gcc_w;
/*
 * PURPOSE
 *  CUDA contiguous package for western ghost Gcc plane.
 ******
 */

/****v* bluebottle/_recv_Gcc_n
 * NAME
 *  _recv_Gcc_n
 * TYPE
 */
extern real *_recv_Gcc_n;
/*
 * PURPOSE
 *  CUDA contiguous package for northern ghost Gcc plane.
 ******
 */

/****v* bluebottle/_recv_Gcc_s
 * NAME
 *  _recv_Gcc_s
 * TYPE
 */
extern real *_recv_Gcc_s;
/*
 * PURPOSE
 *  CUDA contiguous package for southern ghost Gcc plane.
 ******
 */

/****v* bluebottle/_recv_Gcc_t
 * NAME
 *  _recv_Gcc_t
 * TYPE
 */
extern real *_recv_Gcc_t;
/*
 * PURPOSE
 *  CUDA contiguous package for top ghost Gcc plane.
 ******
 */

/****v* bluebottle/_recv_Gcc_b
 * NAME
 *  _recv_Gcc_b
 * TYPE
 */
extern real *_recv_Gcc_b;
/*
 * PURPOSE
 *  CUDA contiguous package for bottom ghost Gcc plane.
 ******
 */

/****v* bluebottle/_send_Gfx_e
 * NAME
 *  _send_Gfx_e
 * TYPE
 */
extern real *_send_Gfx_e;
/*
 * PURPOSE
 *  CUDA contiguous package for eastern computational Gfx plane.
 ******
 */

/****v* bluebottle/_send_Gfx_w
 * NAME
 *  _send_Gfx_w
 * TYPE
 */
extern real *_send_Gfx_w;
/*
 * PURPOSE
 *  CUDA contiguous package for western computational Gfx plane.
 ******
 */

/****v* bluebottle/_send_Gfx_n
 * NAME
 *  _send_Gfx_n
 * TYPE
 */
extern real *_send_Gfx_n;
/*
 * PURPOSE
 *  CUDA contiguous package for northern computational Gfx plane.
 ******
 */

/****v* bluebottle/_send_Gfx_s
 * NAME
 *  _send_Gfx_s
 * TYPE
 */
extern real *_send_Gfx_s;
/*
 * PURPOSE
 *  CUDA contiguous package for southern computational Gfx plane.
 ******
 */

/****v* bluebottle/_send_Gfx_t
 * NAME
 *  _send_Gfx_t
 * TYPE
 */
extern real *_send_Gfx_t;
/*
 * PURPOSE
 *  CUDA contiguous package for top computational Gfx plane.
 ******
 */

/****v* bluebottle/_send_Gfx_b
 * NAME
 *  _send_Gfx_b
 * TYPE
 */
extern real *_send_Gfx_b;
/*
 * PURPOSE
 *  CUDA contiguous package for bottom computational Gfx plane.
 ******
 */

/****v* bluebottle/_recv_Gfx_e
 * NAME
 *  _recv_Gfx_e
 * TYPE
 */
extern real *_recv_Gfx_e;
/*
 * PURPOSE
 *  CUDA contiguous package for eastern ghost Gfx plane.
 ******
 */

/****v* bluebottle/_recv_Gfx_w
 * NAME
 *  _recv_Gfx_w
 * TYPE
 */
extern real *_recv_Gfx_w;
/*
 * PURPOSE
 *  CUDA contiguous package for western ghost Gfx plane.
 ******
 */

/****v* bluebottle/_recv_Gfx_n
 * NAME
 *  _recv_Gfx_n
 * TYPE
 */
extern real *_recv_Gfx_n;
/*
 * PURPOSE
 *  CUDA contiguous package for northern ghost Gfx plane.
 ******
 */

/****v* bluebottle/_recv_Gfx_s
 * NAME
 *  _recv_Gfx_s
 * TYPE
 */
extern real *_recv_Gfx_s;
/*
 * PURPOSE
 *  CUDA contiguous package for southern ghost Gfx plane.
 ******
 */

/****v* bluebottle/_recv_Gfx_t
 * NAME
 *  _recv_Gfx_t
 * TYPE
 */
extern real *_recv_Gfx_t;
/*
 * PURPOSE
 *  CUDA contiguous package for top ghost Gfx plane.
 ******
 */

/****v* bluebottle/_recv_Gfx_b
 * NAME
 *  _recv_Gfx_b
 * TYPE
 */
extern real *_recv_Gfx_b;
/*
 * PURPOSE
 *  CUDA contiguous package for bottom ghost Gfx plane.
 ******
 */

/****v* bluebottle/_send_Gfy_e
 * NAME
 *  _send_Gfy_e
 * TYPE
 */
extern real *_send_Gfy_e;
/*
 * PURPOSE
 *  CUDA contiguous package for eastern computational Gfy plane.
 ******
 */

/****v* bluebottle/_send_Gfy_w
 * NAME
 *  _send_Gfy_w
 * TYPE
 */
extern real *_send_Gfy_w;
/*
 * PURPOSE
 *  CUDA contiguous package for western computational Gfy plane.
 ******
 */

/****v* bluebottle/_send_Gfy_n
 * NAME
 *  _send_Gfy_n
 * TYPE
 */
extern real *_send_Gfy_n;
/*
 * PURPOSE
 *  CUDA contiguous package for northern computational Gfy plane.
 ******
 */

/****v* bluebottle/_send_Gfy_s
 * NAME
 *  _send_Gfy_s
 * TYPE
 */
extern real *_send_Gfy_s;
/*
 * PURPOSE
 *  CUDA contiguous package for southern computational Gfy plane.
 ******
 */

/****v* bluebottle/_send_Gfy_t
 * NAME
 *  _send_Gfy_t
 * TYPE
 */
extern real *_send_Gfy_t;
/*
 * PURPOSE
 *  CUDA contiguous package for top computational Gfy plane.
 ******
 */

/****v* bluebottle/_send_Gfy_b
 * NAME
 *  _send_Gfy_b
 * TYPE
 */
extern real *_send_Gfy_b;
/*
 * PURPOSE
 *  CUDA contiguous package for bottom computational Gfy plane.
 ******
 */

/****v* bluebottle/_recv_Gfy_e
 * NAME
 *  _recv_Gfy_e
 * TYPE
 */
extern real *_recv_Gfy_e;
/*
 * PURPOSE
 *  CUDA contiguous package for eastern ghost Gfy plane.
 ******
 */

/****v* bluebottle/_recv_Gfy_w
 * NAME
 *  _recv_Gfy_w
 * TYPE
 */
extern real *_recv_Gfy_w;
/*
 * PURPOSE
 *  CUDA contiguous package for western ghost Gfy plane.
 ******
 */

/****v* bluebottle/_recv_Gfy_n
 * NAME
 *  _recv_Gfy_n
 * TYPE
 */
extern real *_recv_Gfy_n;
/*
 * PURPOSE
 *  CUDA contiguous package for northern ghost Gfy plane.
 ******
 */

/****v* bluebottle/_recv_Gfy_s
 * NAME
 *  _recv_Gfy_s
 * TYPE
 */
extern real *_recv_Gfy_s;
/*
 * PURPOSE
 *  CUDA contiguous package for southern ghost Gfy plane.
 ******
 */

/****v* bluebottle/_recv_Gfy_t
 * NAME
 *  _recv_Gfy_t
 * TYPE
 */
extern real *_recv_Gfy_t;
/*
 * PURPOSE
 *  CUDA contiguous package for top ghost Gfy plane.
 ******
 */

/****v* bluebottle/_recv_Gfy_b
 * NAME
 *  _recv_Gfy_b
 * TYPE
 */
extern real *_recv_Gfy_b;
/*
 * PURPOSE
 *  CUDA contiguous package for bottom ghost Gfy plane.
 ******
 */

/****v* bluebottle/_send_Gfz_e
 * NAME
 *  _send_Gfz_e
 * TYPE
 */
extern real *_send_Gfz_e;
/*
 * PURPOSE
 *  CUDA contiguous package for eastern computational Gfz plane.
 ******
 */

/****v* bluebottle/_send_Gfz_w
 * NAME
 *  _send_Gfz_w
 * TYPE
 */
extern real *_send_Gfz_w;
/*
 * PURPOSE
 *  CUDA contiguous package for western computational Gfz plane.
 ******
 */

/****v* bluebottle/_send_Gfz_n
 * NAME
 *  _send_Gfz_n
 * TYPE
 */
extern real *_send_Gfz_n;
/*
 * PURPOSE
 *  CUDA contiguous package for northern computational Gfz plane.
 ******
 */

/****v* bluebottle/_send_Gfz_s
 * NAME
 *  _send_Gfz_s
 * TYPE
 */
extern real *_send_Gfz_s;
/*
 * PURPOSE
 *  CUDA contiguous package for southern computational Gfz plane.
 ******
 */

/****v* bluebottle/_send_Gfz_t
 * NAME
 *  _send_Gfz_t
 * TYPE
 */
extern real *_send_Gfz_t;
/*
 * PURPOSE
 *  CUDA contiguous package for top computational Gfz plane.
 ******
 */

/****v* bluebottle/_send_Gfz_b
 * NAME
 *  _send_Gfz_b
 * TYPE
 */
extern real *_send_Gfz_b;
/*
 * PURPOSE
 *  CUDA contiguous package for bottom computational Gfz plane.
 ******
 */

/****v* bluebottle/_recv_Gfz_e
 * NAME
 *  _recv_Gfz_e
 * TYPE
 */
extern real *_recv_Gfz_e;
/*
 * PURPOSE
 *  CUDA contiguous package for eastern ghost Gfz plane.
 ******
 */

/****v* bluebottle/_recv_Gfz_w
 * NAME
 *  _recv_Gfz_w
 * TYPE
 */
extern real *_recv_Gfz_w;
/*
 * PURPOSE
 *  CUDA contiguous package for western ghost Gfz plane.
 ******
 */

/****v* bluebottle/_recv_Gfz_n
 * NAME
 *  _recv_Gfz_n
 * TYPE
 */
extern real *_recv_Gfz_n;
/*
 * PURPOSE
 *  CUDA contiguous package for northern ghost Gfz plane.
 ******
 */

/****v* bluebottle/_recv_Gfz_s
 * NAME
 *  _recv_Gfz_s
 * TYPE
 */
extern real *_recv_Gfz_s;
/*
 * PURPOSE
 *  CUDA contiguous package for southern ghost Gfz plane.
 ******
 */

/****v* bluebottle/_recv_Gfz_t
 * NAME
 *  _recv_Gfz_t
 * TYPE
 */
extern real *_recv_Gfz_t;
/*
 * PURPOSE
 *  CUDA contiguous package for top ghost Gfz plane.
 ******
 */

/****v* bluebottle/_recv_Gfz_b
 * NAME
 *  _recv_Gfz_b
 * TYPE
 */
extern real *_recv_Gfz_b;
/*
 * PURPOSE
 *  CUDA contiguous package for bottom ghost Gfz plane.
 ******
 */

/****v* bluebottle/_rhs_p
 * NAME
 *  _rhs_p
 * TYPE
 */
extern real *_rhs_p;
/*
 * PURPOSE
 *  CUDA device array for storing the right-hand side of the pressure-Poisson 
 *  problem
 ******
 */

/****v* bluebottle/_r_q
 * NAME
 *  _r_q
 * TYPE
 */
extern real *_r_q;
/*
 * PURPOSE
 *  CUDA device array for the residual vector in the Poisson solver. Stores the
 *  residual at each subdomain's soln points (does not include exchange cells)
 ******
 */

/****v* bluebottle/_z_q
 * NAME
 *  _z_q
 * TYPE
 */
extern real *_z_q;
/*
 * PURPOSE
 *  CUDA device array for the residual vector in the Poisson solver. Stores the
 *  preconditioned residual at each subdomain's soln points (does not include 
 *  exchange cells)
 ******
 */

/****v* bluebottle/_rs_0
 * NAME
 *  _rs_0
 * TYPE
 */
//extern real *_rs_0;
/*
 * PURPOSE
 *  CUDA device array for the initial residual vector in the Poisson solver. 
 *  Stores the initial residual at each subdomain's soln points (does not 
 *  include exchange cells)
 ******
 */

/****v* bluebottle/_p_q
 * NAME
 *  _p_q
 * TYPE
 */
extern real *_p_q;
/*
 * PURPOSE
 *  CUDA device array for the descent ("pointing") direction in the CG solver on
 *  the inner computational domain. Stores the direction at each subdomain's
 *  soln points (does not include exchange cells)
 ******
 */

/****v* bluebottle/_pb_q
 * NAME
 *  _pb_q
 * TYPE
 */
extern real *_pb_q;
/*
 * PURPOSE
 *  CUDA device array for the descent ("pointing") direction in the CG solver on
 *  the ghost domain. Stores the direction at ALL grid points (including 
 *  exchange cells). This info is needed for the sparse matrix-vector product.
 ******
 */

/****v* bluebottle/_s_q
 * NAME
 *  _s_q
 * TYPE
 */
//extern real *_s_q;
/*
 * PURPOSE
 *  CUDA device array for the 2nd descent ("pointing") direction in the CG 
 *  solver on the inner computational domain. Stores the direction at each 
 *  subdomain's soln points (does not include exchange cells)
 ******
 */

/****v* bluebottle/_sb_q
 * NAME
 *  _sb_q
 * TYPE
 */
//extern real *_sb_q;
/*
 * PURPOSE
 *  CUDA device array for the 2nd descent ("pointing") direction in the CG 
 *  solver on the ghost domain. Stores the direction at ALL grid points
 *  (including exchange cells). This info is need for the SpMV product
 ******
 */

/****v* bluebottle/_Apb_q
 * NAME
 *  _Apb_q
 * TYPE
 */
extern real *_Apb_q;
/*
 * PURPOSE
 *  CUDA device array for the sparse matrix-vector product A*pb_q result
 ******
 */

/****v* bluebottle/_Asb_q
 * NAME
 *  _Asb_q
 * TYPE
 */
//extern real *_Asb_q;
/*
 * PURPOSE
 *  CUDA device array for the sparse matrix-vector product A*sb_q result
 ******
 */

/****v* bluebottle/cpumem
 * NAME
 *  cpumem
 * TYPE
 */
extern long int cpumem;
/*
 * PURPOSE
 *  Estimate of total cpu memory usage (per mpi process)
 ******
 */

/****v* bluebottle/gpumem
 * NAME
 *  gpumem
 * TYPE
 */
extern long int gpumem;
/*
 * PURPOSE
 *  Estimate of total gpu memory usage (per mpi process)
 ******
 */

/****v* bluebottle/dt
 * NAME
 *  dt
 * TYPE
 */
extern real dt;
/*
 * PURPOSE
 *  The current timestep size.  Upon initialization via input file, dt is the
 *  timestep size used for the first timestep.  It is subsequently updated
 *  dynamically for the remainder of the simulation.
 ******
 */

/****v* bluebottle/dt0
 * NAME
 *  dt0
 * TYPE
 */
extern real dt0;
/*
 * PURPOSE
 *  The previous timestep size.
 ******
 */

/****v* bluebottle/CFL
 * NAME
 *  CFL
 * TYPE
 */
extern real CFL;
/*
 * PURPOSE
 *  Define the CFL condition number for adaptive timestep calculation.
 ******
 */

/****v* bluebottle/duration
 * NAME
 *  duration
 * TYPE
 */
extern real duration;
/*
 * PURPOSE
 *  The duration of the simulation (i.e., stop time).
 ******
 */

/****v* bluebottle/ttime
 * NAME
 *  ttime
 * TYPE
 */
extern real ttime;
/*
 * PURPOSE
 *  The accumulated time since the simulation began.
 ******
 */

/****v* bluebottle/v_bc_tdelay
  * NAME
  * v_bc_tdelay
  * TYPE
  */
extern real v_bc_tdelay;
/*
  * PURPOSE
  *  The time when the dirichlet velocity boundary conditions should be applied
  ******
  */

/****v* bluebottle/p_bc_tdelay;
  * NAME
  * p_bc_tdelay;
  * TYPE
  */
extern real p_bc_tdelay;
/*
  * PURPOSE
  *  The time when the applied pressure should be applied
  ******
  */
 
/****v* bluebottle/g_bc_tdelay;
  * NAME
  * g_bc_tdelay;
  * TYPE
  */
extern real g_bc_tdelay;
/*
  * PURPOSE
  *  The time when the applied gravity should be applied
  ******
  */

/****v* bluebottle/stepnum
 * NAME
 *  stepnum
 * TYPE
 */
extern int stepnum;
/*
 * PURPOSE
 *  The current timestep number for the simulation.  The initial configuration
 *  is given by stepnum = 0.
 ******
 */

/****v* bluebottle/pid_int
 * NAME
 *  pid_int
 * TYPE
 */
extern real pid_int;
/*
 * PURPOSE
 *  Store the integral of the PID controller target.
 ******
 */

/****v* bluebottle/pid_back
 * NAME
 *  pid_back
 * TYPE
 */
extern real pid_back;
/*
 * PURPOSE
 *  Store the previous value of the PID controller target for derivative
 *  term calculation.
 ******
 */

/****v* bluebottle/Kp
 * NAME
 *  Kp
 * TYPE
 */
extern real Kp;
/*
 * PURPOSE
 *  PID controller proportional gain.
 ******
 */

/****v* bluebottle/Ki
 * NAME
 *  Ki
 * TYPE
 */
extern real Ki;
/*
 * PURPOSE
 *  PID controller integral gain.
 ******
 */

/****v* bluebottle/Kd
 * NAME
 *  Kd
 * TYPE
 */
extern real Kd;
/*
 * PURPOSE
 *  PID controller derivative gain.
 ******
 */

/****v* bluebottle/turbA
 * NAME
 *  turbA
 * TYPE
 */
extern real turbA;
/*
 * PURPOSE
 *  Turbulence precursor linear forcing magnitude (see Lundgren 2003, Rosales
 *  and Meneveau 2005, Carroll and Blanquart 2013)
 ******
 */

/****v* bluebottle/turbl
 * NAME
 *  turbl
 * TYPE
 */
extern real turbl;
/*
 * PURPOSE
 *  Turbulence precursor linear forcing integral scale (see Lundgren 2003,
 *  Rosales and Meneveau 2005, Carroll and Blanquart 2013)
 ******
 */

/****v* bluebottle/turb_k0
 * NAME
 *  turb_k0
 * TYPE
 */
extern real turb_k0;
/*
 * PURPOSE
 *  Turbulence precursor linear forcing mean kinetic energy (see Lundgren 2003,
 *  Rosales and Meneveau 2005, Carroll and Blanquart 2013)
 ******
 */



/* FUNCTIONS */

/****f* bluebottle/cuda_device_count()
 * NAME
 *  cuda_device_count()
 * USAGE
 */
int cuda_device_count(void);
/*
 * FUNCTION
 *  Return the cuda device count
 ******
 */

/****f* bluebottle/cuda_device_init()
 * NAME
 *  cuda_device_init()
 * USAGE
 */
void cuda_device_init(int device);
/*
 * FUNCTION
 *  Set the cuda device
 * ARGUMENTS
 *  * device -- Current process's device
 ******
 */

/****f* bluebottle/cuda_enable_peer()
 * NAME
 *  cuda_enable_peer()
 * USAGE
 */
void cuda_enable_peer(void);
/*
 * FUNCTION
 *  Enable peer-to-peer access inbetween gpus on a node
 ******
 */

/****f* bluebottle/cuda_block()
 * NAME
 *  cuda_block()
 * USAGE
 */
void cuda_block(void);
/*
 * FUNCTION
 *  Use cudaDeviceSynchronize() to sync all device activity before transfer.
 *  This is done so MPI calls aren't async with packing calls.
 ******
 */

/****f* bluebottle/cuda_dom_malloc_host()
 * NAME
 *  cuda_dom_malloc_host()
 * USAGE
 */
void cuda_dom_malloc_host(void);
/*
 * FUNCTION
 *  Allocate domain memory on host 
 ******
 */

/****f* bluebottle/cuda_dom_malloc_dev()
 * NAME
 *  cuda_dom_malloc_dev()
 * USAGE
 */
void cuda_update_bc();
void cuda_dom_malloc_dev(void);
/*
 * FUNCTION
 *  Allocate domain memory on device
 *  Allocate device memory reference pointers on host and device memory on
 *  device for the flow domain.
 *  Includes flow arrays and message packages
 ******
 */

/****f* bluebottle/cuda_dom_push()
 * NAME
 *  cuda_dom_push()
 * USAGE
 */
void cuda_dom_push(void);
/*
 * FUNCTION
 *  Copy p, u, v, w from host to device
 ******
 */

/****f* bluebottle/cuda_dom_pull()
 * NAME
 *  cuda_dom_pull()
 * USAGE
 */
void cuda_dom_pull(void);
/*
 * FUNCTION
 *  Copy p, u, v, w from device to host
 ******
 */

/****f* bluebottle/cuda_dom_pull_phase()
 * NAME
 *  cuda_dom_pull_phase()
 * USAGE
 */
void cuda_dom_pull_phase(void);
/*
 * FUNCTION
 *  Copy phase and phase_shell from device to host
 ******
 */

/****f* bluebottle/cuda_dom_pull_debug()
 * NAME
 *  cuda_dom_pull_debug()
 * USAGE
 */
void cuda_dom_pull_debug(void);
/*
 * FUNCTION
 *  Copy u*,v*,w*,phi,conv_{u,v,w},diff_{u,v,w},flag_{u,v,w} from device to
 *  host for debug output
 ******
 */

/****f* bluebottle/cuda_dom_pull_restart()
 * NAME
 *  cuda_dom_pull_restart()
 * USAGE
 */
void cuda_dom_pull_restart(void);
/*
 * FUNCTION
 *  Copy {u,v,w}0, diff0_{u,v,w}, conv0_{u,v,w} from device to host for restart
 *  output
 ******
 */

/****f* bluebottle/cuda_blocks_init()
 * NAME
 *  cuda_blocks_init()
 * USAGE
 */
void cuda_blocks_init(void);
/*
 * FUNCTION
 *  Initialize the cuda threads and block sizes
 ******
 */

/****f* bluebottle/cuda_blocks_write()
 * NAME
 *  cuda_blocks_write()
 * USAGE
 */
void cuda_blocks_write(void);
/*
 * FUNCTION
 *  Write block sizes to debug file.
 ******
 */

/****f* bluebottle/cuda_dom_BC()
 * NAME
 *  cuda_dom_BC()
 * USAGE
 */
void cuda_dom_BC(void);
/*
 * FUNCTION
 *  Enforce boundary conditions on velocity and pressure fields on domain
 *  boundaries.
 ******
 */

/****f* bluebottle/cuda_build_cages()
 * NAME
 *  cuda_build_cages()
 * USAGE
 */
void cuda_build_cages(void);
/*
 * FUNCTION
 *  Build the particle cages on the devices and generate phase array
 ******
 */

/****f* bluebottle/cuda_self_exchange_i()
 * NAME
 *  cuda_self_exchange_i()
 * USAGE
 */
void cuda_self_exchange_i(real *array);
/*
 * FUNCTION
 *  Directly move boundary data in cuda arrays on a device in i direction
 * ARGUMENTS
 *  * array -- array to be exchanged
 ******
 */

/****f* bluebottle/cuda_self_exchange_j()
 * NAME
 *  cuda_self_exchange_j()
 * USAGE
 */
void cuda_self_exchange_j(real *array);
/*
 * FUNCTION
 *  Directly move boundary data in cuda arrays on a device in j direction
 * ARGUMENTS
 *  * array -- array to be exchanged
 ******
 */

/****f* bluebottle/cuda_self_exchange_k()
 * NAME
 *  cuda_self_exchange_k()
 * USAGE
 */
void cuda_self_exchange_k(real *array);
/*
 * FUNCTION
 *  Directly move boundary data in cuda arrays on a device in k direction
 * ARGUMENTS
 *  * array -- array to be exchanged
 ******
 */

/****f* bluebottle/cuda_pack_planes_Gcc()
 * NAME
 *  cuda_pack_planes_Gcc()
 * USAGE
 */
void cuda_pack_planes_Gcc(real *array);
/*
 * FUNCTION
 *  Pack the discontigous Gcc bounding planes into a contiguous cuda array
 * ARGUMENTS
 *  * array -- array to be packed
 ******
 */

/****f* bluebottle/cuda_pack_planes_Gfx()
 * NAME
 *  cuda_pack_planes_Gfx()
 * USAGE
 */
void cuda_pack_planes_Gfx(real *array);
/*
 * FUNCTION
 *  Pack the discontigous Gfx bounding planes into a contiguous cuda array
 * ARGUMENTS
 *  * array -- array to be packed
 ******
 */

/****f* bluebottle/cuda_pack_planes_Gfy()
 * NAME
 *  cuda_pack_planes_Gfy()
 * USAGE
 */
void cuda_pack_planes_Gfy(real *array);
/*
 * FUNCTION
 *  Pack the discontigous Gfy bounding planes into a contiguous cuda array
 * ARGUMENTS
 *  * array -- array to be packed
 ******
 */

/****f* bluebottle/cuda_pack_planes_Gfz()
 * NAME
 *  cuda_pack_planes_Gfz()
 * USAGE
 */
void cuda_pack_planes_Gfz(real *array);
/*
 * FUNCTION
 *  Pack the discontigous Gfz bounding planes into a contiguous cuda array
 * ARGUMENTS
 *  * array -- array to be packed
 ******
 */

/****f* bluebottle/cuda_unpack_planes_Gcc()
 * NAME
 *  cuda_unpack_planes_Gcc()
 * USAGE
 */
void cuda_unpack_planes_Gcc(real *array);
/*
 * FUNCTION
 *  Unpack the contiguous planes Gcc into a discontiguous cuda array
 * ARGUMENTS
 *  * array -- array to be packed
 ******
 */

/****f* bluebottle/cuda_unpack_planes_Gfx()
 * NAME
 *  cuda_unpack_planes_Gfx()
 * USAGE
 */
void cuda_unpack_planes_Gfx(real *array);
/*
 * FUNCTION
 *  Unpack the contiguous planes Gfx into a discontiguous cuda array
 * ARGUMENTS
 *  * array -- array to be packed
 ******
 */

/****f* bluebottle/cuda_unpack_planes_Gfy()
 * NAME
 *  cuda_unpack_planes_Gfy()
 * USAGE
 */
void cuda_unpack_planes_Gfy(real *array);
/*
 * FUNCTION
 *  Unpack the contiguous planes Gfy into a discontiguous cuda array
 * ARGUMENTS
 *  * array -- array to be packed
 ******
 */

/****f* bluebottle/cuda_unpack_planes_Gfz()
 * NAME
 *  cuda_unpack_planes_Gfz()
 * USAGE
 */
void cuda_unpack_planes_Gfz(real *array);
/*
 * FUNCTION
 *  Unpack the contiguous planes Gfz into a discontiguous cuda array
 * ARGUMENTS
 *  * array -- array to be packed
 ******
 */

/****f* bluebottle/cuda_find_dt()
 * NAME
 *  cuda_find_dt()
 * USAGE
 */
void cuda_find_dt(void);
/*
 * FUNCTION
 *  Determine the timestep to use based on the current flow fields and
 *  the Courant-Friedrichs-Lewy (CFL) condition.
 ******
 */

/****f* bluebottle/cuda_compute_forcing()
 * NAME
 *  cuda_compute_forcing()
 * USAGE
 */
void cuda_compute_forcing(void);
/*
 * FUNCTION
 *  Set up the forcing array for this time step. 
 ******
 */

/****f* bluebottle/cuda_compute_turb_forcing()
 * NAME
 *  cuda_compute_turb_forcing()
 * USAGE
 */
void cuda_compute_turb_forcing(void);
/*
 * FUNCTION
 *  Set up the turbulent forcing for this time step. 
 ******
 */

/****f* bluebottle/cuda_U_star()
 * NAME
 *  cuda_U_star()
 * USAGE
 */
void cuda_U_star(void);
/*
 * FUNCTION
 *  Compute u_star, v_star, w_star to second-order time accuracy.
 ******
 */

/****f* bluebottle/cuda_dom_BC_star()
 * NAME
 *  cuda_dom_BC_star()
 * USAGE
 */
void cuda_dom_BC_star(void);
/*
 * FUNCTION
 *  Enforce boundary conditions in intermediate velocity fields.
 ******
 */

/****f* bluebottle/cuda_solvability()
 * NAME
 *  cuda_solvability()
 * USAGE
 */
void cuda_solvability(void);
/*
 * FUNCTION
 *  Enforce the solvability condition on the Poisson problem.
 ******
 */

/****f* bluebottle/cuda_project()
 * NAME
 *  cuda_project()
 * USAGE
 */
void cuda_project(void);
/*
 * FUNCTION
 *  Project the intermediate velocity U* onto a divergence-free space
 *  via the projected pressure.
 ******
 */

/****f* bluebottle/cuda_update_p()
 * NAME
 *  cuda_update_p()
 * USAGE
 */
void cuda_update_p(void);
/*
 * FUNCTION
 *  Update the pressure.
 ******
 */

/****f* bluebottle/cuda_dom_BC_p()
 * NAME
 *  cuda_dom_BC_p()
 * USAGE
 */
void cuda_dom_BC_p(real *array);
/*
 * FUNCTION
 *  Enforce zero normal gradient boundary conditions on projected pressure or
 *  phi (specified by array)
 * ARGUMENTS
 *  * array -- _p or _phi, field to apply neumann conditions to
 ******
 */ 

/****f* bluebottle/cuda_store_u()
 * NAME
 *  cuda_store_u()
 * USAGE
 */
void cuda_store_u(void);
/*
 * FUNCTION
 *  Store the previous u, v, w components for use in the next timestep.
 ******
 */

/****f* bluebottle/cuda_PP_init_jacobi_preconditioner()
 * NAME
 *  cuda_PP_init_jacobi_preconditioner()
 * USAGE
 */
void cuda_PP_init_jacobi_preconditioner(void);
/*
 * FUNCTION
 *  Initialize the Jacobi preconditioner
 ******
 */

/****f* bluebottle/cuda_PP_cg()
 * NAME
 *  cuda_PP_cg()
 * USAGE
 */
void cuda_PP_cg(void);
/*
 * FUNCTION
 *  Solve the pressure poisson equation for phi using the conjugate gradient
 *  method
 ******
 */

/****f* bluebottle/cuda_PP_cg_timed()
 * NAME
 *  cuda_PP_cg()
 * USAGE
 */
void cuda_PP_cg_timed(void);
/*
 * FUNCTION
 *  Solve the pressure poisson equation for phi using the conjugate gradient
 *  method. This has hooks to output profiling for individual portions of
 *  cg.
 ******
 */

/****f* bluebottle/cuda_PP_cg_noparts()
 * NAME
 *  cuda_PP_cg()
 * USAGE
 */
void cuda_PP_cg_noparts(void);
/*
 * FUNCTION
 *  Version of cuda_PP_cg with no particle-related methods
 ******
 */

/****f* bluebottle/cuda_PP_bicgstab()
 * NAME
 *  cuda_PP_bicgstab()
 * USAGE
 */
void cuda_PP_bicgstab(void);
/*
 * FUNCTION
 *  Solve the pressure poisson equation for phi using the biconjugate gradient
 *  method (stabilized)
 ******
 */

/****f* bluebottle/cuda_dom_free()
 * NAME
 *  cuda_dom_free()
 * USAGE
 */
void cuda_dom_free(void);
/*
 * FUNCTION
 *  Free device memory for the domain on device and device memory reference
 *  pointers on host.
 ******
 */

/****f* bluebottle/cuda_wall_shear_stress()
 * NAME
 *  cuda_wall_shear_stress()
 * USAGE
 */
void cuda_wall_shear_stress(void);
/*
 * FUNCTION
 *  Calculate the wall shear stress for turb channel flow
 *  and output to file
 ******
 */

#ifdef TEST
/****f* bluebottle/run_test()
 * NAME
 *  run_test()
 * USAGE
 */
void run_test(void);
/*
 * FUNCTION
 *  Run various tests
 ******
 */

/****f* bluebottle/cuda_U_star_test_init()
 * NAME
 *  cuda_U_star_test_init()
 * USAGE
 */
void cuda_U_star_test_init(void);
/*
 * FUNCTION
 *  Init variables for U star tests
 ******
 */

/****f* bluebottle/cuda_U_star_test_clean()
 * NAME
 *  cuda_U_star_test_clean()
 * USAGE
 */
void cuda_U_star_test_clean(void);
/*
 * FUNCTION
 *  Clean variables for U star tests
 ******
 */

/****f* bluebottle/cuda_U_star_test_exp()
 * NAME
 *  cuda_U_star_test_exp()
 * USAGE
 */
void cuda_U_star_test_exp(void);
/*
 * FUNCTION
 *  Test U_star calculation for an exponential initialization.
 *  Uses u = exp(x), v = exp(y), w = exp(z)
 ******
 */

/****f* bluebottle/cuda_U_star_test_sin()
 * NAME
 *  cuda_U_star_test_sin()
 * USAGE
 */
void cuda_U_star_test_sin(void);
/*
 * FUNCTION
 *  Test U_star calculation for an sine initialization.
 *  Uses u = sin(x), v = sin(y), w = sin(z)
 ******
 */

/****f* bluebottle/cuda_BC_test_periodic()
 * NAME
 *  cuda_BC_test_periodic()
 * USAGE
 */
void cuda_BC_test_periodic(void);
/*
 * FUNCTION
 *  Validation testbed for cuda_BC.  Domain must be -1 <= x <= 1,
 *  -1 <= y <= 1, -1 <= z <= 1.  In order to test cuda_BC, ensure that there
 *  are no pressure gradients applied to the system.
 ******
 */

/****f* bluebottle/cuda_BC_test_dirichlet()
 * NAME
 *  cuda_BC_test_dirichlet()
 * USAGE
 */
void cuda_BC_test_dirichlet(void);
/*
 * FUNCTION
 *  Validation testbed for cuda_BC.  Domain must be -1 <= x <= 1,
 *  -1 <= y <= 1, -1 <= z <= 1.  In order to test cuda_BC, ensure that there
 *  are no pressure gradients applied to the system.
 ******
 */

/****f* bluebottle/cuda_BC_test_neumann()
 * NAME
 *  cuda_BC_test_neumann()
 * USAGE
 */
void cuda_BC_test_neumann(void);
/*
 * FUNCTION
 *  Validation testbed for cuda_BC.  Domain must be -1 <= x <= 1,
 *  -1 <= y <= 1, -1 <= z <= 1.  In order to test cuda_BC, ensure that there
 *  are no pressure gradients applied to the system.
 ******
 */

/****f* bluebottle/cuda_quad_interp_test()
 * NAME
 *  cuda_quad_interp_test()
 * USAGE
 */
void cuda_quad_interp_test(void);
/*
 * FUNCTION
 *  Validation testbed for Lebedev interpolation.
 ******
 */

/****f* bluebottle/cuda_lamb_test()
 * NAME
 *  cuda_lamb_test()
 * USAGE
 */
void cuda_lamb_test(void);
/*
 * FUNCTION
 *  Validation testbed for Lamb's solution.
 ******
 */

#endif // TEST

/* Non-essential functions */
void cuda_check_errors(int line); // useful for debugging cuda errors

#endif // _BLUEBOTTLE_H
