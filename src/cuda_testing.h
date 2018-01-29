/*******************************************************************************
 ********************************** BLUEBOTTLE *********************************
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

/****h* bluebottle/cuda_testing_kernel
 * NAME
 *  cuda_testing_kernel
 * FUNCTION
 *  bluebottle testing CUDA kernel functions.
 ******
 */

#ifndef _CUDA_TESTING_H
#define _CUDA_TESTING_H

#ifdef TEST

extern "C"
{
#include "bluebottle.h"
#include "bluebottle.cuh"
}

/****d* bluebottle/us_a
 * NAME
 *  us_a
 * TYPE
 */
extern real *us_a;
/*
 * PURPOSE
 *  Expected/actual solution for ustar
 ******
 */

/****d* bluebottle/vs_a
 * NAME
 *  vs_a
 * TYPE
 */
extern real *vs_a;
/*
 * PURPOSE
 *  Expected/actual solution for vstar
 ******
 */

/****d* bluebottle/ws_a
 * NAME
 *  ws_a
 * TYPE
 */
extern real *ws_a;
/*
 * PURPOSE
 *  Expected/actual solution for wstar
 ******
 */

/****d* bluebottle/uc_a
 * NAME
 *  uc_a
 * TYPE
 */
extern real *uc_a;
/*
 * PURPOSE
 *  Expected/actual solution for conv_u
 ******
 */

/****d* bluebottle/vc_a
 * NAME
 *  vc_a
 * TYPE
 */
extern real *vc_a;
/*
 * PURPOSE
 *  Expected/actual solution for conv_v
 ******
 */

/****d* bluebottle/wc_a
 * NAME
 *  wc_a
 * TYPE
 */
extern real *wc_a;
/*
 * PURPOSE
 *  Expected/actual solution for conv_w
 ******
 */

/****d* bluebottle/ud_a
 * NAME
 *  ud_a
 * TYPE
 */
extern real *ud_a;
/*
 * PURPOSE
 *  Expected/actual solution for diff_u
 ******
 */

/****d* bluebottle/vd_a
 * NAME
 *  vd_a
 * TYPE
 */
extern real *vd_a;
/*
 * PURPOSE
 *  Expected/actual solution for diff_v
 ******
 */

/****d* bluebottle/wd_a
 * NAME
 *  wd_a
 * TYPE
 */
extern real *wd_a;
/*
 * PURPOSE
 *  Expected/actual solution for diff_w
 ******
 */

/****d* bluebottle/p_i
 * NAME
 *  p_i
 * TYPE
 */
extern real *p_i;
/*
 * PURPOSE
 *  Expected/actual solution for p BC
 ******
 */

/****d* bluebottle/u_i
 * NAME
 *  u_i
 * TYPE
 */
extern real *u_i;
/*
 * PURPOSE
 *  Expected/actual solution for u BC
 ******
 */

/****d* bluebottle/v_i
 * NAME
 *  v_i
 * TYPE
 */
extern real *v_i;
/*
 * PURPOSE
 *  Expected/actual solution for v BC
 ******
 */

/****d* bluebottle/w_i
 * NAME
 *  w_i
 * TYPE
 */
extern real *w_i;
/*
 * PURPOSE
 *  Expected/actual solution for w BC
 ******
 */

/****d* bluebottle/u_star_err_min
 * NAME
 *  u_star_err_min
 * TYPE
 */
extern real u_star_err_min;
/*
 * PURPOSE
 *  Minimum error in u_star
 ******
 */

/****d* bluebottle/v_star_err_min
 * NAME
 *  v_star_err_min
 * TYPE
 */
extern real v_star_err_min;
/*
 * PURPOSE
 *  Minimum error in v_star
 ******
 */

/****d* bluebottle/w_star_err_min
 * NAME
 *  w_star_err_min
 * TYPE
 */
extern real w_star_err_min;
/*
 * PURPOSE
 *  Minimum error in w_star
 ******
 */

/****d* bluebottle/u_star_err_max
 * NAME
 *  u_star_err_max
 * TYPE
 */
extern real u_star_err_max;
/*
 * PURPOSE
 *  Maximum error in u_star
 ******
 */

/****d* bluebottle/v_star_err_max
 * NAME
 *  v_star_err_max
 * TYPE
 */
extern real v_star_err_max;
/*
 * PURPOSE
 *  Maximum error in v_star
 ******
 */

/****d* bluebottle/w_star_err_max
 * NAME
 *  w_star_err_max
 * TYPE
 */
extern real w_star_err_max;
/*
 * PURPOSE
 *  Maximum error in w_star
 ******
 */

/****d* bluebottle/conv_u_err_min
 * NAME
 *  conv_u_err_min
 * TYPE
 */
extern real conv_u_err_min;
/*
 * PURPOSE
 *  Minimum error in conv_u
 ******
 */

/****d* bluebottle/conv_v_err_min
 * NAME
 *  conv_v_err_min
 * TYPE
 */
extern real conv_v_err_min;
/*
 * PURPOSE
 *  Minimum error in conv_v
 ******
 */

/****d* bluebottle/conv_w_err_min
 * NAME
 *  conv_w_err_min
 * TYPE
 */
extern real conv_w_err_min;
/*
 * PURPOSE
 *  Minimum error in conv_w
 ******
 */

/****d* bluebottle/conv_u_err_max
 * NAME
 *  conv_u_err_max
 * TYPE
 */
extern real conv_u_err_max;
/*
 * PURPOSE
 *  Maximum error in conv_u
 ******
 */

/****d* bluebottle/conv_v_err_max
 * NAME
 *  conv_v_err_max
 * TYPE
 */
extern real conv_v_err_max;
/*
 * PURPOSE
 *  Maximum error in conv_v
 ******
 */

/****d* bluebottle/conv_w_err_max
 * NAME
 *  conv_w_err_max
 * TYPE
 */
extern real conv_w_err_max;
/*
 * PURPOSE
 *  Maximum error in conv_w
 ******
 */

/****d* bluebottle/diff_u_err_min
 * NAME
 *  diff_u_err_min
 * TYPE
 */
extern real diff_u_err_min;
/*
 * PURPOSE
 *  Minimum error in diff_u
 ******
 */

/****d* bluebottle/diff_v_err_min
 * NAME
 *  diff_v_err_min
 * TYPE
 */
extern real diff_v_err_min;
/*
 * PURPOSE
 *  Minimum error in diff_v
 ******
 */

/****d* bluebottle/diff_w_err_min
 * NAME
 *  diff_w_err_min
 * TYPE
 */
extern real diff_w_err_min;
/*
 * PURPOSE
 *  Minimum error in diff_w
 ******
 */

/****d* bluebottle/diff_u_err_max
 * NAME
 *  diff_u_err_max
 * TYPE
 */
extern real diff_u_err_max;
/*
 * PURPOSE
 *  Maximum error in diff_u
 ******
 */

/****d* bluebottle/diff_v_err_max
 * NAME
 *  diff_v_err_max
 * TYPE
 */
extern real diff_v_err_max;
/*
 * PURPOSE
 *  Maximum error in diff_v
 ******
 */

/****d* bluebottle/diff_w_err_max
 * NAME
 *  diff_w_err_max
 * TYPE
 */
extern real diff_w_err_max;
/*
 * PURPOSE
 *  Maximum error in diff_w
 ******
 */

/****d* bluebottle/p_err_max
 * NAME
 *  p_err_max
 * TYPE
 */
extern real p_err_max;
/*
 * PURPOSE
 *  maximum error in p
 ******
 */

/****d* bluebottle/u_err_max
 * NAME
 *  u_err_max
 * TYPE
 */
extern real u_err_max;
/*
 * PURPOSE
 *  maximum error in u
 ******
 */

/****d* bluebottle/v_err_max
 * NAME
 *  v_err_max
 * TYPE
 */
extern real v_err_max;
/*
 * PURPOSE
 *  maximum error in v
 ******
 */

/****d* bluebottle/w_err_max
 * NAME
 *  w_err_max
 * TYPE
 */
extern real w_err_max;
/*
 * PURPOSE
 *  maximum error in w
 ******
 */

/****d* bluebottle/p_err_min
 * NAME
 *  p_err_min
 * TYPE
 */
extern real p_err_min;
/*
 * PURPOSE
 *  Minimum error in p
 ******
 */

/****d* bluebottle/u_err_min
 * NAME
 *  u_err_min
 * TYPE
 */
extern real u_err_min;
/*
 * PURPOSE
 *  Minimum error in u
 ******
 */

/****d* bluebottle/v_err_min
 * NAME
 *  v_err_min
 * TYPE
 */
extern real v_err_min;
/*
 * PURPOSE
 *  Minimum error in v
 ******
 */

/****d* bluebottle/w_err_min
 * NAME
 *  w_err_min
 * TYPE
 */
extern real w_err_min;
/*
 * PURPOSE
 *  Minimum error in w
 ******
 */

/* FUNCTIONS */

/****f* cuda_testing/is_corner()
  * NAME
  *  is_corner
  * TYPE
  */
extern "C"
int is_corner(int i, int j, int k, grid_info grid);
/*
 * FUNCTION
 *  Determines if the given local i,j,k location is a corner in the given local
 *  grid
 ******
 */
 


#endif // TEST

#endif // _CUDA_TESTING_H
