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
#ifndef _BLUEBOTTLE_CUH_
#define _BLUEBOTTLE_CUH_

extern "C"
{
#include "bluebottle.h"
}

/****s* bluebottle/_dom
 * NAME
 *  _dom
 * TYPE
 */
extern __constant__ dom_struct _dom;
/*
 * PURPOSE
 *  Device analog of dom containing domain information for current MPI process.
 *  Bound to device constant memory for rapid access.
 */

/****s* bluebottle/_bins
 * NAME
 *  _bins
 * TYPE
 */
extern __constant__ bin_struct _bins;
/*
 * PURPOSE
 *  Device analog of _binscontaining bin information for current MPI process.
 *  Bound to device constant memory for rapid access.
 */

/****s* bluebottle/cuda_blocks_info
 * NAME
 *  cuda_blocks_info
 * TYPE
 */
typedef struct cuda_blocks_info {
  dim3 dim_in;
  dim3 dim_jn;
  dim3 dim_kn;
  dim3 num_in;
  dim3 num_jn;
  dim3 num_kn;
  dim3 dim_in_s;
  dim3 dim_jn_s;
  dim3 dim_kn_s;
  dim3 num_in_s;
  dim3 num_jn_s;
  dim3 num_kn_s;
  dim3 dim_inb;
  dim3 dim_jnb;
  dim3 dim_knb;
  dim3 num_inb;
  dim3 num_jnb;
  dim3 num_knb;
  dim3 dim_inb_s;
  dim3 dim_jnb_s;
  dim3 dim_knb_s;
  dim3 num_inb_s;
  dim3 num_jnb_s;
  dim3 num_knb_s;
  dim3 num_s3;
  dim3 dim_s3;
  dim3 num_s3b;
  dim3 dim_s3b;
} cuda_blocks_info;
/*
 * PURPOSE
 *  Carry information related to the different cuda thread block sizes on each
 *  grid
 * MEMBERS
 *  * dim_in -- Dimensions of blocks for EAST/WEST computational grid
 *  * dim_jn -- Dimensions of blocks for NORTH/SOUTH computational grid
 *  * dim_kn -- Dimensions of blocks for TOP/BOTTOM computational grid
 *  * num_in -- Number of threads for EAST/WEST computational grid
 *  * num_jn -- Number of threads for NORTH/SOUTH computational grid
 *  * num_kn -- Number of threads for TOP/BOTTOM computational grid
 *  * dim_in_s -- Block dimension for E/W shared memory comp. grids
 *  * dim_jn_s -- Block dimension for N/S shared memory comp. grids
 *  * dim_kn_s -- Block dimension for T/B shared memory comp. grids
 *  * num_in_s -- Thread numbers for E/W shared memory comp. grids
 *  * num_jn_s -- Thread numbers for N/S shared memory comp. grids
 *  * num_kn_s -- Thread numbers for T/B shared memory comp. grids
 *  * dim_inb -- Dimensions of blocks for EAST/WEST ghost grid
 *  * dim_jnb -- Dimensions of blocks for NORTH/SOUTH ghost grid
 *  * dim_knb -- Dimensions of blocks for TOP/BOTTOM ghost grid
 *  * num_inb -- Number of threads for EAST/WEST ghost grid
 *  * num_jnb -- Number of threads for NORTH/SOUTH ghost grid
 *  * num_knb -- Number of threads for TOP/BOTTOM ghost grid
 *  * dim_inb_s -- Block dimension for E/W shared memory ghost grids
 *  * dim_jnb_s -- Block dimension for N/S shared memory ghost grids
 *  * dim_knb_s -- Block dimension for T/B shared memory ghost grids
 *  * num_inb_s -- Thread numbers for E/W shared memory ghost grids
 *  * num_jnb_s -- Thread numbers for N/S shared memory ghost grids
 *  * num_knb_s -- Thread numbers for T/B shared memory ghost grids
 *  * num_s3 -- Number of threads for inner computational cells
 *  * dim_s3 -- Dimension of blocks for inner computational cells
 *  * num_s3b -- Number of threads for computational cells (incl ghost)
 *  * dim_s3b -- Dimension of blocks for computational cells (incl ghost)
 */

/****s* bluebottle/cuda_blocks_struct
 * NAME
 *  cuda_blocks_struct
 * TYPE
 */
typedef struct cuda_blocks_struct {
  cuda_blocks_info Gcc;
  cuda_blocks_info Gfx;
  cuda_blocks_info Gfy;
  cuda_blocks_info Gfz;
} cuda_blocks_struct;
/*
 * PURPOSE
 *  Carry information related to different cuda thread block sizes
 * MEMBERS
 *  * Gcc -- cell-centered grid information
 *  * Gfx -- face-centered grid information (x)
 *  * Gfy -- face-centered grid information (y)
 *  * Gfz -- face-centered grid information (z)
 */

/****v* bluebottle/blocks
 * NAME
 *  blocks
 * TYPE
 */
extern cuda_blocks_struct blocks;
/*
 * PURPOSE
 *  Contains cuda thread size information information
 */

/****v* bluebottle/_A1
 * NAME
 *  _A1
 * TYPE
 */
extern __constant__ real _A1;
/*
 * PURPOSE
 *  Lebedev quadrature constant weight _A1
 */

/****v* bluebottle/_A2
 * NAME
 *  _A2
 * TYPE
 */
extern __constant__ real _A2;
/*
 * PURPOSE
 *  Lebedev quadrature constant weight _A2
 */

/****v* bluebottle/_A3
 * NAME
 *  _A3
 * TYPE
 */
extern __constant__ real _A3;
/*
 * PURPOSE
 *  Lebedev quadrature constant weight _A3
 */

/****v* bluebottle/_B
 * NAME
 *  _B
 * TYPE
 */
extern __constant__ real _B;
/*
 * PURPOSE
 *  Lebedev quadrature constant weight _B
 */

/****v* bluebottle/_nn
 * NAME
 *  _nn
 * TYPE
 */
extern __constant__ int _nn[NCOEFFS];
/*
 * PURPOSE
 *  List of coefficients nn
 */

/****v* bluebottle/_mm
 * NAME
 *  _mm
 * TYPE
 */
extern __constant__ int _mm[NCOEFFS];
/*
 * PURPOSE
 *  List of coefficients mm
 */

/****v* bluebottle/_node_t
 * NAME
 *  _node_t
 * TYPE
 */
extern __constant__ real _node_t[NNODES];
/*
 * PURPOSE
 *  Theta location of nodes
 */

/****v* bluebottle/_node_p
 * NAME
 *  _node_p
 * TYPE
 */
extern __constant__ real _node_p[NNODES];
/*
 * PURPOSE
 *  phi location of nodes
 */

#endif
