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

/****h* Bluebottle/cuda_bluebottle
 * NAME
 *  cuda_bluebottle
 * FUNCTION
 *  Low-level gpu subdomain routines, including those associated with 
 *  communication
 ******
 */

#ifndef _CUDA_BLUEBOTTLE_H
#define _CUDA_BLUEBOTTLE_H

extern "C"
{
#include "bluebottle.h"
#include "bluebottle.cuh"
}

/* VARIABLES */

/* FUNCTIONS */

/****f* cuda_bluebottle_kernel/BC_p_W_N<<<>>>()
 * NAME
 *  BC_p_W_N<<<>>>()
 * USAGE
 */
__global__ void BC_p_W_N(real *p);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the west face pressure field.
 * ARGUMENTS
 *  * p -- the device pressure field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_p_E_N<<<>>>()
 * NAME
 *  BC_p_E_N<<<>>>()
 * USAGE
 */
__global__ void BC_p_E_N(real *p);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the east face pressure field.
 * ARGUMENTS
 *  * p -- the device pressure field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_p_S_N<<<>>>()
 * NAME
 *  BC_p_S_N<<<>>>()
 * USAGE
 */
__global__ void BC_p_S_N(real *p);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the south face pressure field.
 * ARGUMENTS
 *  * p -- the device pressure field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_p_N_N<<<>>>()
 * NAME
 *  BC_p_N_N<<<>>>()
 * USAGE
 */
__global__ void BC_p_N_N(real *p);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the north face pressure field.
 * ARGUMENTS
 *  * p -- the device pressure field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_p_B_N<<<>>>()
 * NAME
 *  BC_p_B_N<<<>>>()
 * USAGE
 */
__global__ void BC_p_B_N(real *p);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the bottom face pressure field.
 * ARGUMENTS
 *  * p -- the device pressure field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_p_T_N<<<>>>()
 * NAME
 *  BC_p_T_N<<<>>>()
 * USAGE
 */
__global__ void BC_p_T_N(real *p);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the top face pressure field.
 * ARGUMENTS
 *  * p -- the device pressure field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_u_W_N<<<>>>()
 * NAME
 *  BC_u_W_N<<<>>>()
 * USAGE
 */
__global__ void BC_u_W_N(real *u);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the west face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_E_N<<<>>>()
 * NAME
 *  BC_u_E_N<<<>>>()
 * USAGE
 */
__global__ void BC_u_E_N(real *u);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the east face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_S_N<<<>>>()
 * NAME
 *  BC_u_S_N<<<>>>()
 * USAGE
 */
__global__ void BC_u_S_N(real *u);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the south face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_N_N<<<>>>()
 * NAME
 *  BC_u_N_N<<<>>>()
 * USAGE
 */
__global__ void BC_u_N_N(real *u);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the north face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_B_N<<<>>>()
 * NAME
 *  BC_u_B_N<<<>>>()
 * USAGE
 */
__global__ void BC_u_B_N(real *u);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the bottom face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_T_N<<<>>>()
 * NAME
 *  BC_u_T_N<<<>>>()
 * USAGE
 */
__global__ void BC_u_T_N(real *u);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the top face u-velocity field.
 * ARGUMENTS
 *  * u -- the device pressure field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_u_W_D<<<>>>()
 * NAME
 *  BC_u_W_D<<<>>>()
 * USAGE
 */
__global__ void BC_u_W_D(real *u, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the west face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_E_D<<<>>>()
 * NAME
 *  BC_u_E_D<<<>>>()
 * USAGE
 */
__global__ void BC_u_E_D(real *u, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the east face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_S_D<<<>>>()
 * NAME
 *  BC_u_S_D<<<>>>()
 * USAGE
 */
__global__ void BC_u_S_D(real *u, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the south face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_N_D<<<>>>()
 * NAME
 *  BC_u_N_D<<<>>>()
 * USAGE
 */
__global__ void BC_u_N_D(real *u, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the north face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_B_D<<<>>>()
 * NAME
 *  BC_u_B_D<<<>>>()
 * USAGE
 */
__global__ void BC_u_B_D(real *u, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the bottom face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */
/****f* cuda_bluebottle_kernel/BC_u_T_D<<<>>>()
 * NAME
 *  BC_u_T_D<<<>>>()
 * USAGE
 */
__global__ void BC_u_T_D(real *u, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the top face u-velocity field.
 * ARGUMENTS
 *  * u -- the device u-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_W_N<<<>>>()
 * NAME
 *  BC_v_W_N<<<>>>()
 * USAGE
 */
__global__ void BC_v_W_N(real *v);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the west face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_E_N<<<>>>()
 * NAME
 *  BC_v_E_N<<<>>>()
 * USAGE
 */
__global__ void BC_v_E_N(real *v);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the east face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_S_N<<<>>>()
 * NAME
 *  BC_v_S_N<<<>>>()
 * USAGE
 */
__global__ void BC_v_S_N(real *v);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the south face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_N_N<<<>>>()
 * NAME
 *  BC_v_N_N<<<>>>()
 * USAGE
 */
__global__ void BC_v_N_N(real *v);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the north face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_B_N<<<>>>()
 * NAME
 *  BC_v_B_N<<<>>>()
 * USAGE
 */
__global__ void BC_v_B_N(real *v);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the bottom face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_T_N<<<>>>()
 * NAME
 *  BC_v_T_N<<<>>>()
 * USAGE
 */
__global__ void BC_v_T_N(real *v);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the top face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_W_D<<<>>>()
 * NAME
 *  BC_v_W_D<<<>>>()
 * USAGE
 */
__global__ void BC_v_W_D(real *v, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the west face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_E_D<<<>>>()
 * NAME
 *  BC_v_E_D<<<>>>()
 * USAGE
 */
__global__ void BC_v_E_D(real *v, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the east face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_S_D<<<>>>()
 * NAME
 *  BC_v_S_D<<<>>>()
 * USAGE
 */
__global__ void BC_v_S_D(real *v, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the south face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_N_D<<<>>>()
 * NAME
 *  BC_v_N_D<<<>>>()
 * USAGE
 */
__global__ void BC_v_N_D(real *v, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the north face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_B_D<<<>>>()
 * NAME
 *  BC_v_B_D<<<>>>()
 * USAGE
 */
__global__ void BC_v_B_D(real *v, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the bottom face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_v_T_D<<<>>>()
 * NAME
 *  BC_v_T_D<<<>>>()
 * USAGE
 */
__global__ void BC_v_T_D(real *v, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the top face v-velocity field.
 * ARGUMENTS
 *  * v -- the device v-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_W_N<<<>>>()
 * NAME
 *  BC_w_W_N<<<>>>()
 * USAGE
 */
__global__ void BC_w_W_N(real *w);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the west face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_E_N<<<>>>()
 * NAME
 *  BC_w_E_N<<<>>>()
 * USAGE
 */
__global__ void BC_w_E_N(real *w);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the east face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_S_N<<<>>>()
 * NAME
 *  BC_w_S_N<<<>>>()
 * USAGE
 */
__global__ void BC_w_S_N(real *w);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the south face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_N_N<<<>>>()
 * NAME
 *  BC_w_N_N<<<>>>()
 * USAGE
 */
__global__ void BC_w_N_N(real *w);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the north face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_B_N<<<>>>()
 * NAME
 *  BC_w_B_N<<<>>>()
 * USAGE
 */
__global__ void BC_w_B_N(real *w);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the bottom face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_T_N<<<>>>()
 * NAME
 *  BC_w_T_N<<<>>>()
 * USAGE
 */
__global__ void BC_w_T_N(real *w);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the top face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_W_D<<<>>>()
 * NAME
 *  BC_w_W_D<<<>>>()
 * USAGE
 */
__global__ void BC_w_W_D(real *w, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the west face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_E_D<<<>>>()
 * NAME
 *  BC_w_E_D<<<>>>()
 * USAGE
 */
__global__ void BC_w_E_D(real *w, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the east face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_S_D<<<>>>()
 * NAME
 *  BC_w_S_D<<<>>>()
 * USAGE
 */
__global__ void BC_w_S_D(real *w, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the south face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_N_D<<<>>>()
 * NAME
 *  BC_w_N_D<<<>>>()
 * USAGE
 */
__global__ void BC_w_N_D(real *w, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the north face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_B_D<<<>>>()
 * NAME
 *  BC_w_B_D<<<>>>()
 * USAGE
 */
__global__ void BC_w_B_D(real *w, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the bottom face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle_kernel/BC_w_T_D<<<>>>()
 * NAME
 *  BC_w_T_D<<<>>>()
 * USAGE
 */
__global__ void BC_w_T_D(real *w, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the top face w-velocity field.
 * ARGUMENTS
 *  * w -- the device w-velocity field subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */

/****f* cuda_bluebottle/self_exchange_Gcc_i()
 * NAME
 *  self_exchange_Gcc_i
 * TYPE
 */
__global__ void self_exchange_Gcc_i(real *array);
/*
 * FUNCTION
 *  Exchanges boundary data in i direction with self
 * ARGUMENTS
 *  * array -- Gcc domain data array
 ******
 */

/****f* cuda_bluebottle/self_exchange_Gcc_j()
 * NAME
 *  self_exchange_Gcc_j
 * TYPE
 */
__global__ void self_exchange_Gcc_j(real *array);
/*
 * FUNCTION
 *  Exchanges boundary data in j direction with self
 * ARGUMENTS
 *  * array -- Gcc domain data array
 ******
 */

/****f* cuda_bluebottle/self_exchange_Gcc_k()
 * NAME
 *  self_exchange_Gcc_k
 * TYPE
 */
__global__ void self_exchange_Gcc_k(real *array);
/*
 * FUNCTION
 *  Exchanges boundary data in k direction with self
 * ARGUMENTS
 *  * array -- Gcc domain data array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gcc_east()
 * NAME
 *  pack_planes_Gcc_east
 * TYPE
 */
__global__ void pack_planes_Gcc_east(real *contents, real *package);
/*
 * FUNCTION
 *  Packs an eastern Gcc computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gcc_west()
 * NAME
 *  pack_planes_Gcc_west
 * TYPE
 */
__global__ void pack_planes_Gcc_west(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a western Gcc computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gcc_north()
 * NAME
 *  pack_planes_Gcc_north
 * TYPE
 */
__global__ void pack_planes_Gcc_north(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a northern Gcc computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gcc_south()
 * NAME
 *  pack_planes_Gcc_south
 * TYPE
 */
__global__ void pack_planes_Gcc_south(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a southern Gcc computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gcc_top()
 * NAME
 *  pack_planes_Gcc_top
 * TYPE
 */
__global__ void pack_planes_Gcc_top(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a top Gcc computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gcc_bottom()
 * NAME
 *  pack_planes_Gcc_bottom
 * TYPE
 */
__global__ void pack_planes_Gcc_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a bottom Gcc computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfx_east()
 * NAME
 *  pack_planes_Gfx_east
 * TYPE
 */
__global__ void pack_planes_Gfx_east(real *contents, real *package);
/*
 * FUNCTION
 *  Packs an eastern Gfx computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfx_west()
 * NAME
 *  pack_planes_Gfx_west
 * TYPE
 */
__global__ void pack_planes_Gfx_west(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a western Gfx computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfx_north()
 * NAME
 *  pack_planes_Gfx_north
 * TYPE
 */
__global__ void pack_planes_Gfx_north(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a northern Gfx computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfx_south()
 * NAME
 *  pack_planes_Gfx_south
 * TYPE
 */
__global__ void pack_planes_Gfx_south(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a southern Gfx computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfx_top()
 * NAME
 *  pack_planes_Gfx_top
 * TYPE
 */
__global__ void pack_planes_Gfx_top(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a top Gfx computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfx_bottom()
 * NAME
 *  pack_planes_Gfx_bottom
 * TYPE
 */
__global__ void pack_planes_Gfx_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a bottom Gfx computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfy_east()
 * NAME
 *  pack_planes_Gfy_east
 * TYPE
 */
__global__ void pack_planes_Gfy_east(real *contents, real *package);
/*
 * FUNCTION
 *  Packs an eastern Gfy computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfy_west()
 * NAME
 *  pack_planes_Gfy_west
 * TYPE
 */
__global__ void pack_planes_Gfy_west(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a western Gfy computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfy_north()
 * NAME
 *  pack_planes_Gfy_north
 * TYPE
 */
__global__ void pack_planes_Gfy_north(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a northern Gfy computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfy_south()
 * NAME
 *  pack_planes_Gfy_south
 * TYPE
 */
__global__ void pack_planes_Gfy_south(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a southern Gfy computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfy_top()
 * NAME
 *  pack_planes_Gfy_top
 * TYPE
 */
__global__ void pack_planes_Gfy_top(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a top Gfy computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfy_bottom()
 * NAME
 *  pack_planes_Gfy_bottom
 * TYPE
 */
__global__ void pack_planes_Gfy_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a bottom Gfy computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfz_east()
 * NAME
 *  pack_planes_Gfz_east
 * TYPE
 */
__global__ void pack_planes_Gfz_east(real *contents, real *package);
/*
 * FUNCTION
 *  Packs an eastern Gfz computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfz_west()
 * NAME
 *  pack_planes_Gfz_west
 * TYPE
 */
__global__ void pack_planes_Gfz_west(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a western Gfz computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfz_north()
 * NAME
 *  pack_planes_Gfz_north
 * TYPE
 */
__global__ void pack_planes_Gfz_north(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a northern Gfz computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfz_south()
 * NAME
 *  pack_planes_Gfz_south
 * TYPE
 */
__global__ void pack_planes_Gfz_south(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a southern Gfz computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfz_top()
 * NAME
 *  pack_planes_Gfz_top
 * TYPE
 */
__global__ void pack_planes_Gfz_top(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a top Gfz computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/pack_planes_Gfz_bottom()
 * NAME
 *  pack_planes_Gfz_bottom
 * TYPE
 */
__global__ void pack_planes_Gfz_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Packs a bottom Gfz computational plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gcc_east()
 * NAME
 *  unpack_planes_Gcc_east
 * TYPE
 */
__global__ void unpack_planes_Gcc_east(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the east Gcc ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gcc_west()
 * NAME
 *  unpack_planes_Gcc_west
 * TYPE
 */
__global__ void unpack_planes_Gcc_west(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the west Gcc ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gcc_north()
 * NAME
 *  unpack_planes_Gcc_north
 * TYPE
 */
__global__ void unpack_planes_Gcc_north(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the north Gcc ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gcc_south()
 * NAME
 *  unpack_planes_Gcc_south
 * TYPE
 */
__global__ void unpack_planes_Gcc_south(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the south Gcc ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gcc_top()
 * NAME
 *  unpack_planes_Gcc_top
 * TYPE
 */
__global__ void unpack_planes_Gcc_top(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the top Gcc ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gcc_bottom()
 * NAME
 *  unpack_planes_Gcc_bottom
 * TYPE
 */
__global__ void unpack_planes_Gcc_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the bottom Gcc ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfx_east()
 * NAME
 *  unpack_planes_Gfx_east
 * TYPE
 */
__global__ void unpack_planes_Gfx_east(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the east Gfx ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfx_west()
 * NAME
 *  unpack_planes_Gfx_west
 * TYPE
 */
__global__ void unpack_planes_Gfx_west(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the west Gfx ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfx_north()
 * NAME
 *  unpack_planes_Gfx_north
 * TYPE
 */
__global__ void unpack_planes_Gfx_north(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the north Gfx ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfx_south()
 * NAME
 *  unpack_planes_Gfx_south
 * TYPE
 */
__global__ void unpack_planes_Gfx_south(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the south Gfx ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfx_top()
 * NAME
 *  unpack_planes_Gfx_top
 * TYPE
 */
__global__ void unpack_planes_Gfx_top(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the top Gfx ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfx_bottom()
 * NAME
 *  unpack_planes_Gfx_bottom
 * TYPE
 */
__global__ void unpack_planes_Gfx_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the bottom Gfx ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfy_east()
 * NAME
 *  unpack_planes_Gfy_east
 * TYPE
 */
__global__ void unpack_planes_Gfy_east(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the east Gfy ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfy_west()
 * NAME
 *  unpack_planes_Gfy_west
 * TYPE
 */
__global__ void unpack_planes_Gfy_west(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the west Gfy ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfy_north()
 * NAME
 *  unpack_planes_Gfy_north
 * TYPE
 */
__global__ void unpack_planes_Gfy_north(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the north Gfy ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfy_south()
 * NAME
 *  unpack_planes_Gfy_south
 * TYPE
 */
__global__ void unpack_planes_Gfy_south(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the south Gfy ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfy_top()
 * NAME
 *  unpack_planes_Gfy_top
 * TYPE
 */
__global__ void unpack_planes_Gfy_top(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the top Gfy ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfy_bottom()
 * NAME
 *  unpack_planes_Gfy_bottom
 * TYPE
 */
__global__ void unpack_planes_Gfy_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the bottom Gfy ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_east()
 * NAME
 *  unpack_planes_Gfz_east
 * TYPE
 */
__global__ void unpack_planes_Gfz_east(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the east Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_west()
 * NAME
 *  unpack_planes_Gfz_west
 * TYPE
 */
__global__ void unpack_planes_Gfz_west(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the west Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_north()
 * NAME
 *  unpack_planes_Gfz_north
 * TYPE
 */
__global__ void unpack_planes_Gfz_north(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the north Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_south()
 * NAME
 *  unpack_planes_Gfz_south
 * TYPE
 */
__global__ void unpack_planes_Gfz_south(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the south Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_top()
 * NAME
 *  unpack_planes_Gfz_top
 * TYPE
 */
__global__ void unpack_planes_Gfz_top(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the top Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_bottom()
 * NAME
 *  unpack_planes_Gfz_bottom
 * TYPE
 */
__global__ void unpack_planes_Gfz_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the bottom Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_east()
 * NAME
 *  unpack_planes_Gfz_east
 * TYPE
 */
__global__ void unpack_planes_Gfz_east(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the east Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_west()
 * NAME
 *  unpack_planes_Gfz_west
 * TYPE
 */
__global__ void unpack_planes_Gfz_west(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the west Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_north()
 * NAME
 *  unpack_planes_Gfz_north
 * TYPE
 */
__global__ void unpack_planes_Gfz_north(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the north Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_south()
 * NAME
 *  unpack_planes_Gfz_south
 * TYPE
 */
__global__ void unpack_planes_Gfz_south(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the south Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_top()
 * NAME
 *  unpack_planes_Gfz_top
 * TYPE
 */
__global__ void unpack_planes_Gfz_top(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the top Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle/unpack_planes_Gfz_bottom()
 * NAME
 *  unpack_planes_Gfz_bottom
 * TYPE
 */
__global__ void unpack_planes_Gfz_bottom(real *contents, real *package);
/*
 * FUNCTION
 *  Unpack the bottom Gfz ghost plane
 * ARGUMENTS
 *  * contents -- discontinous domain data array
 *  * package -- continuous package array
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_reset_x<<<>>>()
 * NAME
 *  forcing_reset_x<<<>>>()
 * USAGE
 */
__global__ void forcing_reset_x(real *fx);
/*
 * FUNCTION 
 *  Reset the x-direction forcing array to zero.
 * ARGUMENTS
 *  * fx -- the forcing array to be reset
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_reset_y<<<>>>()
 * NAME
 *  forcing_reset_y<<<>>>()
 * USAGE
 */
__global__ void forcing_reset_y(real *fy);
/*
 * FUNCTION 
 *  Reset the y-direction forcing array to zero.
 * ARGUMENTS
 *  * fy -- the forcing array to be reset
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_reset_z<<<>>>()
 * NAME
 *  forcing_reset_z<<<>>>()
 * USAGE
 */
__global__ void forcing_reset_z(real *fz);
/*
 * FUNCTION 
 *  Reset the z-direction forcing array to zero.
 * ARGUMENTS
 *  * fz -- the forcing array to be reset
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_add_c_const<<<>>>()
 * NAME
 *  forcing_add_c_const<<<>>>()
 * USAGE
 */
__global__ void forcing_add_c_const(real val, real *cc);
/*
 * FUNCTION 
 *  Add a constant.
 * ARGUMENTS
 *  * val -- the value of the force to be added to the array
 *  * cc -- the array
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_add_x_const<<<>>>()
 * NAME
 *  forcing_add_x_const<<<>>>()
 * USAGE
 */
__global__ void forcing_add_x_const(real val, real *fx);
/*
 * FUNCTION 
 *  Add a constant force to the forcing array.
 * ARGUMENTS
 *  * val -- the value of the force to be added to the array
 *  * fx -- the forcing array to be reset
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_add_y_const<<<>>>()
 * NAME
 *  forcing_add_y_const<<<>>>()
 * USAGE
 */
__global__ void forcing_add_y_const(real val, real *fy);
/*
 * FUNCTION 
 *  Add a constant force to the forcing array.
 * ARGUMENTS
 *  * val -- the value of the force to be added to the array
 *  * fy -- the forcing array to be reset
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_add_z_const<<<>>>()
 * NAME
 *  forcing_add_z_const<<<>>>()
 * USAGE
 */
__global__ void forcing_add_z_const(real val, real *fz);
/*
 * FUNCTION 
 *  Add a constant force to the forcing array.
 * ARGUMENTS
 *  * val -- the value of the force to be added to the array
 *  * fz -- the forcing array to be reset
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_add_x_field<<<>>>()
 * NAME
 *  forcing_add_x_field<<<>>>()
 * USAGE
 */
__global__ void forcing_add_x_field(real scale, real *val, real *fx);
/*
 * FUNCTION 
 *  Add a field force to the forcing array.
 * ARGUMENTS
 *  * scale -- a constant scaling
 *  * val -- the value of the force to be added to the array
 *  * fx -- the forcing array to be reset
 *  * dom -- the current subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_add_y_field<<<>>>()
 * NAME
 *  forcing_add_y_field<<<>>>()
 * USAGE
 */
__global__ void forcing_add_y_field(real scale, real *val, real *fy);
/*
 * FUNCTION 
 *  Add a field force to the forcing array.
 * ARGUMENTS
 *  * scale -- a constant scaling
 *  * val -- the value of the force to be added to the array
 *  * fy -- the forcing array to be reset
 *  * dom -- the current subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/forcing_add_z_field<<<>>>()
 * NAME
 *  forcing_add_z_field<<<>>>()
 * USAGE
 */
__global__ void forcing_add_z_field(real scale, real *val, real *fz);
/*
 * FUNCTION 
 *  Add a field force to the forcing array.
 * ARGUMENTS
 *  * scale -- a constant scaling
 *  * val -- the value of the force to be added to the array
 *  * fz -- the forcing array to be reset
 *  * dom -- the current subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/pull_wdot<<<>>>()
 * NAME
 *  pull_wdot<<<>>>()
 * USAGE
 */
__global__ void pull_wdot(real *wdot, part_struct *parts, int *bin_start,
  int *bin_count, int *part_ind);
/*
 * FUNCTION 
 *  Pull wdot to an array for each bin
 * ARGUMENTS
 *  * wdot -- array of wdot sum for parts in each bin
 *  * parts -- particle structure
 *  * bin_start -- start index of each bin
 *  * bin_count -- nparts per bind
 *  * part_ind -- mapping from bins -> parts
 ******
 */

/****f* cuda_bluebottle_kernel/calc_u_star<<<>>>()
 * NAME
 *  calc_u_star<<<>>>()
 * USAGE
 */
__global__ void calc_u_star(real rho_f, real nu,
  real *u0, real *v0, real *w0, real *p, real *f,
  real *diff0, real *conv0, real *diff, real *conv, real *u_star,
  real dt0, real dt, int *phase);
/*
 * FUNCTION
 *  Compute the intermediate velocity field u_star (2nd-order in time).
 * ARGUMENTS
 *  * rho_f -- fluid density
 *  * nu -- fluid kinematic viscosity
 *  * u0 -- device subdomain u-component velocity field from previous timestep
 *  * v0 -- device subdomain v-component velocity field from previous timestep
 *  * w0 -- device subdomain w-component velocity field from previous timestep
 *  * p0 -- device subdomain pressure field from previous timestep
 *  * f -- the forcing array
 *  * diff0 -- device subdomain previous diffusion term
 *  * conv0 -- device subdomain previous convection term
 *  * u_star -- the intermediate velocity field
 *  * dt0 -- the previous timestep
 *  * dt -- the current timestep
 *  * phase -- phase indicator function
 ******
 */

/****f* cuda_bluebottle_kernel/calc_v_star<<<>>>()
 * NAME
 *  calc_v_star<<<>>>()
 * USAGE
 */
__global__ void calc_v_star(real rho_f, real nu,
  real *u0, real *v0, real *w0, real *p, real *f,
  real *diff0, real *conv0, real *diff, real *conv, real *v_star,
  real dt0, real dt, int *phase);
/*
 * FUNCTION
 *  Compute the intermediate velocity field v_star (2nd-order in time).
 * ARGUMENTS
 *  * rho_f -- fluid density
 *  * nu -- fluid kinematic viscosity
 *  * u0 -- device subdomain u-component velocity field from previous timestep
 *  * v0 -- device subdomain v-component velocity field from previous timestep
 *  * w0 -- device subdomain w-component velocity field from previous timestep
 *  * p0 -- device subdomain pressure field from previous timestep
 *  * f -- the forcing array
 *  * diff0 -- device subdomain previous diffusion term
 *  * conv0 -- device subdomain previous convection term
 *  * v_star -- the intermediate velocity field
 *  * dt0 -- the previous timestep
 *  * dt -- the current timestep
 *  * phase -- phase indicator function
 ******
 ******
 */

/****f* cuda_bluebottle_kernel/calc_w_star<<<>>>()
 * NAME
 *  calc_w_star<<<>>>()
 * USAGE
 */
__global__ void calc_w_star(real rho_f, real nu,
  real *u0, real *v0, real *w0, real *p, real *f,
  real *diff0, real *conv0, real *diff, real *conv, real *w_star,
  real dt0, real dt, int *phase);
/*
 * FUNCTION
 *  Compute the intermediate velocity field w_star (2nd-order in time).
 * ARGUMENTS
 *  * rho_f -- fluid density
 *  * nu -- fluid kinematic viscosity
 *  * u0 -- device subdomain u-component velocity field from previous timestep
 *  * v0 -- device subdomain v-component velocity field from previous timestep
 *  * w0 -- device subdomain w-component velocity field from previous timestep
 *  * p0 -- device subdomain pressure field from previous timestep
 *  * f -- the forcing array
 *  * diff0 -- device subdomain previous diffusion term
 *  * conv0 -- device subdomain previous convection term
 *  * w_star -- the intermediate velocity field
 *  * dt0 -- the previous timestep
 *  * dt -- the current timestep
 *  * phase -- phase indicator function
 ******
 ******
 */

/****f* cuda_bluebottle_kernel/surf_int_xs<<<>>>()
 * NAME
 *  surf_int_xs<<<>>>()
 * USAGE
 */
__global__ void surf_int_xs(real *u_star, real *u_star_tmp);
/*
 * FUNCTION
 *  Copy the West face of u_star to a temporary vector to be
 *  summed in order to calculate the surface integral u*.n on these faces.
 * ARGUMENTS
 *  * u_star -- subdomain intermediate velocity in the x-direction on the GPU
 *  * u_star_tmp -- the location to which to copy part of u_star
 ******
 */

/****f* cuda_bluebottle_kernel/surf_int_xe<<<>>>()
 * NAME
 *  surf_int_xe<<<>>>()
 * USAGE
 */
__global__ void surf_int_xe(real *u_star, real *u_star_tmp);
/*
 * FUNCTION
 *  Copy the East face of u_star to a temporary vector to be
 *  summed in order to calculate the surface integral u*.n on these faces.
 * ARGUMENTS
 *  * u_star -- subdomain intermediate velocity in the x-direction on the GPU
 *  * u_star_tmp -- the location to which to copy part of u_star
 ******
 */

/****f* cuda_bluebottle_kernel/surf_int_ys<<<>>>()
 * NAME
 *  surf_int_ys<<<>>>()
 * USAGE
 */
__global__ void surf_int_ys(real *v_star, real *v_star_tmp);
/*
 * FUNCTION
 *  Copy the South face of v_star to a temporary vector to be
 *  summed in order to calculate the surface integral u*.n on these faces.
 * ARGUMENTS
 *  * v_star -- subdomain intermediate velocity in the y-direction on the GPU
 *  * v_star_tmp -- the location to which to copy part of v_star
 ******
 */

/****f* cuda_bluebottle_kernel/surf_int_ye<<<>>>()
 * NAME
 *  surf_int_ye<<<>>>()
 * USAGE
 */
__global__ void surf_int_ye(real *v_star, real *v_star_tmp);
/*
 * FUNCTION
 *  Copy the North face of v_star to a temporary vector to be
 *  summed in order to calculate the surface integral u*.n on these faces.
 * ARGUMENTS
 *  * v_star -- subdomain intermediate velocity in the y-direction on the GPU
 *  * v_star_tmp -- the location to which to copy part of v_star
 ******
 */

/****f* cuda_bluebottle_kernel/surf_int_zs<<<>>>()
 * NAME
 *  surf_int_zs<<<>>>()
 * USAGE
 */
__global__ void surf_int_zs(real *w_star, real *w_star_tmp);
/*
 * FUNCTION
 *  Copy the Bottom face of w_star to a temporary vector to be
 *  summed in order to calculate the surface integral u*.n on these faces.
 * ARGUMENTS
 *  * w_star -- subdomain intermediate velocity in the w-direction on the GPU
 *  * w_star_tmp -- the location to which to copy part of w_star
 ******
 */

/****f* cuda_bluebottle_kernel/surf_int_ze<<<>>>()
 * NAME
 *  surf_int_ze<<<>>>()
 * USAGE
 */
__global__ void surf_int_ze(real *w_star, real *w_star_tmp);
/*
 * FUNCTION
 *  Copy the Top face of w_star to a temporary vector to be
 *  summed in order to calculate the surface integral u*.n on these faces.
 * ARGUMENTS
 *  * w_star -- subdomain intermediate velocity in the w-direction on the GPU
 *  * w_star_tmp -- the location to which to copy part of w_star
 ******
 */

/****f* cuda_bluebottle_kernel/plane_eps_x_W<<<>>>()
 * NAME
 *  plane_eps_x_W<<<>>>()
 * USAGE
 */
__global__ void plane_eps_x_W(real *u_star, real eps_x);
/*
 * FUNCTION
 *  Subtract eps from each node of the WEST face to fudge the outflot plane
 *  just enough for the solvablility condition to hold to machine accuracy.
 * ARGUMENTS
 *  * u_star -- the subdomain u_star velocity from which to subtract eps
 *  * eps_x -- the epsilon value to subtract from each node of the outflow plane
 ******
 */

/****f* cuda_bluebottle_kernel/plane_eps_x_E<<<>>>()
 * NAME
 *  plane_eps_x_E<<<>>>()
 * USAGE
 */
__global__ void plane_eps_x_E(real *u_star, real eps_x);
/*
 * FUNCTION
 *  Subtract eps from each node of the EAST face to fudge the outflot plane
 *  just enough for the solvablility condition to hold to machine accuracy.
 * ARGUMENTS
 *  * u_star -- the subdomain u_star velocity from which to subtract eps
 *  * eps_x -- the epsilon value to subtract from each node of the outflow plane
 ******
 */

/****f* cuda_bluebottle_kernel/plane_eps_y_S<<<>>>()
 * NAME
 *  plane_eps_y_S<<<>>>()
 * USAGE
 */
__global__ void plane_eps_y_S(real *v_star, real eps_y);
/*
 * FUNCTION
 *  Subtract eps from each node of the SOUTH face to fudge the outflot plane
 *  just enough for the solvablility condition to hold to machine accuracy.
 * ARGUMENTS
 *  * v_star -- the subdomain v_star velocity from which to subtract eps
 *  * eps_y -- the epsilon value to subtract from each node of the outflow plane
 ******
 */

/****f* cuda_bluebottle_kernel/plane_eps_y_N<<<>>>()
 * NAME
 *  plane_eps_y_N<<<>>>()
 * USAGE
 */
__global__ void plane_eps_y_N(real *v_star, real eps_y);
/*
 * FUNCTION
 *  Subtract eps from each node of the North face to fudge the outflot plane
 *  just enough for the solvablility condition to hold to machine accuracy.
 * ARGUMENTS
 *  * v_star -- the subdomain v_star velocity from which to subtract eps
 *  * eps_y -- the epsilon value to subtract from each node of the outflow plane
 ******
 */

/****f* cuda_bluebottle_kernel/plane_eps_z_B<<<>>>()
 * NAME
 *  plane_eps_z_B<<<>>>()
 * USAGE
 */
__global__ void plane_eps_z_B(real *w_star, real eps_z);
/*
 * FUNCTION
 *  Subtract eps from each node of the BOTTOM face to fudge the outflot plane
 *  just enough for the solvablility condition to hold to machine accuracy.
 * ARGUMENTS
 *  * w_star -- the subdomain w_star velocity from which to subtract eps
 *  * eps_z -- the epsilon value to subtract from each node of the outflow plane
 ******
 */

/****f* cuda_bluebottle_kernel/plane_eps_z_T<<<>>>()
 * NAME
 *  plane_eps_z_T<<<>>>()
 * USAGE
 */
__global__ void plane_eps_z_T(real *w_star, real eps_z);
/*
 * FUNCTION
 *  Subtract eps from each node of the TOP face to fudge the outflot plane
 *  just enough for the solvablility condition to hold to machine accuracy.
 * ARGUMENTS
 *  * w_star -- the subdomain w_star velocity from which to subtract eps
 *  * eps_z -- the epsilon value to subtract from each node of the outflow plane
 ******
 */

/****f* cuda_bluebottle_kernel/project_u<<<>>>()
 * NAME
 *  project_u<<<>>>()
 * USAGE
 */
__global__ void project_u(real *u_star, real *p, real rho_f, real dt,
  real *u, real ddx, int *flag_u);
/*
 * FUNCTION
 *  Project the intermediate velocity u_star onto a divergence-free space via
 *  p.
 * ARGUMENTS
 *  * u_star -- the intermediate velocity field
 *  * p -- the projected pressure
 *  * rho_f -- the fluid density
 *  * dt -- the timestep
 *  * u -- the solution velocity field
 *  * ddx -- 1 / dx
 *  * flag_u -- the x-direction boundary flag
 ******
 */

/****f* cuda_bluebottle_kernel/project_v<<<>>>()
 * NAME
 *  project_v<<<>>>()
 * USAGE
 */
__global__ void project_v(real *v_star, real *p, real rho_f, real dt,
  real *v, real ddy, int *flag_v);
/*
 * FUNCTION
 *  Project the intermediate velocity v_star onto a divergence-free space via
 *  p.
 * ARGUMENTS
 *  * v_star -- the intermediate velocity field
 *  * p -- the projected pressure
 *  * rho_f -- the fluid density
 *  * dt -- the timestep
 *  * v -- the solution velocity field
 *  * ddy -- 1 / dy
 *  * flag_v -- the y-direction boundary flag
 ******
 */

/****f* cuda_bluebottle_kernel/project_w<<<>>>()
 * NAME
 *  project_w<<<>>>()
 * USAGE
 */
__global__ void project_w(real *w_star, real *p, real rho_f, real dt,
  real *w, real ddz, int *flag_w);
/*
 * FUNCTION
 *  Project the intermediate velocity w_star onto a divergence-free space via
 *  p.
 * ARGUMENTS
 *  * w_star -- the intermediate velocity field
 *  * p -- the projected pressure
 *  * rho_f -- the fluid density
 *  * dt -- the timestep
 *  * w -- the solution velocity field
 *  * ddz -- 1 / dz
 *  * flag_w -- the z-direction boundary flag
 ******
 */

/****f* cuda_bluebottle_kernel/update_p_laplacian<<<>>>()
 * NAME
 *  update_p_laplacian<<<>>>()
 * USAGE
 */
__global__ void update_p_laplacian(real *Lp, real *phi);
/*
 * FUNCTION
 *  Update the pressure according to Brown, Cortez, and Minion (2000), eq. 74.
 * ARGUMENTS
 *  * Lp -- Laplacian of phi
 *  * phi -- intermediate pressure given by solution of pressure-Poisson problem
 ******
 */

/****f* cuda_bluebottle_kernel/update_p<<<>>>()
 * NAME
 *  update_p<<<>>>()
 * USAGE
 */
__global__ void update_p(real *Lp, real *p0, real *p, real *phi,
  real nu, real dt, int *phase);
/*
 * FUNCTION
 *  Update the pressure according to Brown, Cortez, and Minion (2000), eq. 74.
 * ARGUMENTS
 *  * Lp -- Laplacian of p
 *  * p0 -- previous pressure
 *  * p -- next pressure
 *  * phi -- intermediate pressure given by solution of pressure-Poisson problem
 *  * nu -- kinematic viscosity
 *  * dt -- current timestep size
 *  * phase -- phase indicator function
 ******
 */

/****f* cuda_bluebottle_kernel/copy_p_noghost<<<>>>()
 * NAME
 *  copy_p_noghost<<<>>>()
 * USAGE
 */
__global__ void copy_p_p_noghost(real *p_noghost, real *p_ghost);
/*
 * FUNCTION
 *  Copy the pressure field containing ghost cells to a new array without
 *  ghost cells.
 * ARGUMENTS
 *  p_noghost -- the destination data structure without ghost cells
 *  p_ghost -- the source data structure with ghost cells
 ******
 */

/****f* cuda_bluebottle_kernel/copy_p_noghost<<<>>>()
 * NAME
 *  copy_p_noghost<<<>>>()
 * USAGE
 */
__global__ void copy_p_noghost_p(real *p_noghost, real *p_ghost);
/*
 * FUNCTION
 *  Copy the pressure field containing ghost cells to a new array without
 *  ghost cells.
 * ARGUMENTS
 *  p_noghost -- the destination data structure without ghost cells
 *  p_ghost -- the source data structure with ghost cells
 ******
 */

/****f* cuda_bluebottle_kernel/copy_u_square_noghost<<<>>>()
 * NAME
 *  copy_u_square_noghost<<<>>>()
 * USAGE
 */
__global__ void copy_u_square_noghost(real *utmp, real *u);
/*
 * FUNCTION
 *  Copy computational cells to a tmp array, squaring along the way
 * ARGUMENTS
 *  utmp -- the destination data structure without ghost cells
 *  u -- the source data structure with ghost cells
 ******
 */

/****f* cuda_bluebottle_kernel/copy_v_square_noghost<<<>>>()
 * NAME
 *  copy_v_square_noghost<<<>>>()
 * USAGE
 */
__global__ void copy_v_square_noghost(real *vtmp, real *v);
/*
 * FUNCTION
 *  Copy computational cells to a tmp array, squaring along the way
 * ARGUMENTS
 *  vtmp -- the destination data structure without ghost cells
 *  v -- the source data structure with ghost cells
 ******
 */

/****f* cuda_bluebottle_kernel/copy_w_square_noghost<<<>>>()
 * NAME
 *  copy_w_square_noghost<<<>>>()
 * USAGE
 */
__global__ void copy_w_square_noghost(real *wtmp, real *w);
/*
 * FUNCTION
 *  Copy computational cells to a tmp array, squaring along the way
 * ARGUMENTS
 *  wtmp -- the destination data structure without ghost cells
 *  w -- the source data structure with ghost cells
 ******
 */

/****f* cuda_bluebottle_kernel/copy_u_noghost<<<>>>()
 * NAME
 *  copy_u_noghost<<<>>>()
 * USAGE
 */
__global__ void copy_u_noghost(real *utmp, real *u);
/*
 * FUNCTION
 *  Copy computational cells to a tmp array
 * ARGUMENTS
 *  utmp -- the destination data structure without ghost cells
 *  u -- the source data structure with ghost cells
 ******
 */

/****f* cuda_bluebottle_kernel/copy_v_noghost<<<>>>()
 * NAME
 *  copy_v_noghost<<<>>>()
 * USAGE
 */
__global__ void copy_v_noghost(real *vtmp, real *v);
/*
 * FUNCTION
 *  Copy computational cells to a tmp array
 * ARGUMENTS
 *  vtmp -- the destination data structure without ghost cells
 *  v -- the source data structure with ghost cells
 ******
 */

/****f* cuda_bluebottle_kernel/copy_w_noghost<<<>>>()
 * NAME
 *  copy_w_noghost<<<>>>()
 * USAGE
 */
__global__ void copy_w_noghost(real *wtmp, real *w);
/*
 * FUNCTION
 *  Copy computational cells to a tmp array
 * ARGUMENTS
 *  wtmp -- the destination data structure without ghost cells
 *  w -- the source data structure with ghost cells
 ******
 */

/****f* cuda_bluebottle_kernel/calc_dissipation<<<>>>()
 * NAME
 *  calc_dissipation<<<>>>()
 * USAGE
 */
__global__ void calc_dissipation(real *u, real *v, real *w, real *eps);
/*
 * FUNCTION
 *  Calculate dudy at a given y location
 * ARGUMENTS
 *  u -- u velocity
 *  v -- v velocity
 *  w -- w velocity
 *  eps -- dissipation field
 ******
 */

/****f* cuda_bluebottle_kernel/calc_dudy<<<>>>()
 * NAME
 *  calc_dudy<<<>>>()
 * USAGE
 */
__global__ void calc_dudy(real *u, real *dudy, int j);
/*
 * FUNCTION
 *  Calculate dudy at a given y location
 * ARGUMENTS
 *  u -- u velocity
 *  dudy -- result
 *  j -- y location
 ******
 */

__global__ void update_vel_BC(BC *bc, real v_bc_tdelay, real ttime);

#endif
