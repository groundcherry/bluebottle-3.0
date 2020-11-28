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

/****h* Bluebottle/physalis
 * NAME
 *  cuda_physalis
 * FUNCTION
 *  Physalis-related operations
 */
#ifndef _PHYSALIS_H
#define _PHYSALIS_H

#include "bluebottle.h"

/****d* physalis/DIV_ST
 * NAME
 *  DIV_ST
 * USAGE
 */
#define DIV_ST -1e-10
/* FUNCTION
 *  Define a value to use for fudging the value of theta when a division by
 *  sin(theta) occurs
 ******
 */

/****d* physalis/NCOEFFS
 * NAME
 *  NCOEFFS
 * USAGE
 */
#define NCOEFFS 15
/* FUNCTION
 *  Based on a truncation order 2 <= L <= 4, there are a maximum of
 *  M = 0.5 * (L + 1)(L + 2) of each coefficient
 ******
 */

/****d* physalis/WEST_WALL_D
 * NAME
 *  WEST_WALL_D
 * USAGE
 */
#define WEST_WALL_D -100
/* FUNCTION
 *  Define a value to use for identifying that the west wall(dirichlet) is
 *  interfering with a quadrature node
 ******
 */

/****d* physalis/WEST_WALL_N
 * NAME
 *  WEST_WALL_N
 * USAGE
 */
#define WEST_WALL_N -101
/* FUNCTION
 *  Define a value to use for identifying that the west wall(neumann) is
 *  interfering with a quadrature node
 ******
 */

/****d* physalis/EAST_WALL_D
 * NAME
 *  EAST_WALL_D
 * USAGE
 */
#define EAST_WALL_D -110
/* FUNCTION
 *  Define a value to use for identifying that the east wall(dirichlet) is
 *  interfering with a quadrature node
 ******
 */

/****d* physalis/EAST_WALL_N
 * NAME
 *  EAST_WALL_N
 * USAGE
 */
#define EAST_WALL_N -111
/* FUNCTION
 *  Define a value to use for identifying that the east wall(neumann) is
 *  interfering with a quadrature node
 ******
 */

/****d* physalis/SOUTH_WALL_D
 * NAME
 *  SOUTH_WALL_D
 * USAGE
 */
#define SOUTH_WALL_D -120
/* FUNCTION
 *  Define a value to use for identifying that the SOUTH wall(dirichlet)
 *  is interfering with a quadrature node
 ******
 */

/****d* physalis/SOUTH_WALL_N
 * NAME
 *  SOUTH_WALL_N
 * USAGE
 */
#define SOUTH_WALL_N -121
/* FUNCTION
 *  Define a value to use for identifying that the SOUTH wall(neumann)
 *  is interfering with a quadrature node
 ******
 */

/****d* physalis/NORTH_WALL_D
 * NAME
 *  NORTH_WALL_D
 * USAGE
 */
#define NORTH_WALL_D -130
/* FUNCTION
 *  Define a value to use for identifying that the NORTH wall(dirichlet)
 *  is interfering with a quadrature node
 ******
 */

/****d* physalis/NORTH_WALL_N
 * NAME
 *  NORTH_WALL_N
 * USAGE
 */
#define NORTH_WALL_N -131
/* FUNCTION
 *  Define a value to use for identifying that the NORTH wall(neumann)
 *  is interfering with a quadrature node
 ******
 */

/****d* physalis/BOTTOM_WALL_D
 * NAME
 *  BOTTOM_WALL_D
 * USAGE
 */
#define BOTTOM_WALL_D -140
/* FUNCTION
 *  Define a value to use for identifying that the BOTTOM wall(dirichlet)
 *  is interfering with a quadrature node
 ******
 */

/****d* physalis/BOTTOM_WALL_N
 * NAME
 *  BOTTOM_WALL_N
 * USAGE
 */
#define BOTTOM_WALL_N -141
/* FUNCTION
 *  Define a value to use for identifying that the BOTTOM wall(neumann)
 *  is interfering with a quadrature node
 ******
 */

/****d* physalis/TOP_WALL_D
 * NAME
 *  TOP_WALL_D
 * USAGE
 */
#define TOP_WALL_D -150
/* FUNCTION
 *  Define a value to use for identifying that the TOP wall(dirichlet)
 *  is interfering with a quadrature node
 ******
 */

/****d* physalis/TOP_WALL_N
 * NAME
 *  TOP_WALL_N
 * USAGE
 */
#define TOP_WALL_N -151
/* FUNCTION
 *  Define a value to use for identifying that the TOP wall(neumann)
 *  is interfering with a quadrature node
 ******
 */

/** Variables **/
/****v* cuda_physalis/_sum_send_e
 * NAME
 *  _sum_send_e
 * USAGE
 */
extern real *_sum_send_e;
/* FUNCTION
 * Contigous array of partial sums that need to be communicated to the east
 ******
 */

/****v* cuda_physalis/_sum_send_w
 * NAME
 *  _sum_send_w
 * USAGE
 */
extern real *_sum_send_w;
/* FUNCTION
 * Contigous array of partial sums that need to be communicated to the west
 ******
 */

/****v* cuda_physalis/_sum_send_n
 * NAME
 *  _sum_send_n
 * USAGE
 */
extern real *_sum_send_n;
/* FUNCTION
 * Contigous array of partial sums that need to be communicated to the north
 ******
 */

/****v* cuda_physalis/_sum_send_s
 * NAME
 *  _sum_send_s
 * USAGE
 */
extern real *_sum_send_s;
/* FUNCTION
 * Contigous array of partial sums that need to be communicated to the south
 ******
 */

/****v* cuda_physalis/_sum_send_t
 * NAME
 *  _sum_send_t
 * USAGE
 */
extern real *_sum_send_t;
/* FUNCTION
 * Contigous array of partial sums that need to be communicated to the top
 ******
 */

/****v* cuda_physalis/_sum_send_b
 * NAME
 *  _sum_send_b
 * USAGE
 */
extern real *_sum_send_b;
/* FUNCTION
 * Contigous array of partial sums that need to be communicated to the bottom
 ******
 */

/****v* cuda_physalis/_sum_recv_e
 * NAME
 *  _sum_recv_e
 * USAGE
 */
extern real *_sum_recv_e;
/* FUNCTION
 * Contigous array of received partial sums from the east that need to be
 *  accumulated
 ******
 */

/****v* cuda_physalis/_sum_recv_w
 * NAME
 *  _sum_recv_w
 * USAGE
 */
extern real *_sum_recv_w;
/* FUNCTION
 * Contigous array of received partial sums from the west that need to be
 *  communicated
 ******
 */

/****v* cuda_physalis/_sum_recv_n
 * NAME
 *  _sum_recv_n
 * USAGE
 */
extern real *_sum_recv_n;
/* FUNCTION
 * Contigous array of received partial sums from the north that need to be
 *  accumulated
 ******
 */

/****v* cuda_physalis/_sum_recv_s
 * NAME
 *  _sum_recv_s
 * USAGE
 */
extern real *_sum_recv_s;
/* FUNCTION
 * Contigous array of received partial sums from the south that need to be
 *  accumulated
 ******
 */

/****v* cuda_physalis/_sum_recv_t
 * NAME
 *  _sum_recv_t
 * USAGE
 */
extern real *_sum_recv_t;
/* FUNCTION
 * Contigous array of received partial sums from the top that need to be
 *  accumulated
 ******
 */

/****v* cuda_physalis/_sum_recv_b
 * NAME
 *  _sum_recv_b
 * USAGE
 */
extern real *_sum_recv_b;
/* FUNCTION
 * Contigous array of received partial sums from the bottom that need to be
 *  accumulated
 ******
 */

/*** Functions */

/****f* physalis/cuda_init_physalis()
 * NAME
 *  cuda_init_physalis()
 * USAGE
 */
void cuda_init_physalis(void);
/* FUNCTION
 *  Initialize the coefficient table and lebedev quadrature in constant memory
 ******
 */

/****f* physalis/cuda_lamb()
 * NAME
 *  cuda_lamb()
 * USAGE
 */
void cuda_lamb(void);
/* FUNCTION
 *  Compute the Lamb's coefficients
 ******
 */

/****f* physalis/cuda_lamb_err()
 * NAME
 *  cuda_lamb_err()
 * USAGE
 */
void cuda_lamb_err(real *error, int *num);
/*
 * FUNCTION
 *  Compute the error between the current and previous sets of Lamb's
 *  coefficients.  It calculates the error then copies the current set
 *  of coefficients to the storage array to be saved for the next iteration
 *  error calculation.
 ******
 */

/****f* physalis/cuda_partial_sum_i()
 * NAME
 *  cuda_partial_sum_i()
 * USAGE
 */
void cuda_partial_sum_i(void);
/* FUNCTION
 *  Communicate partial sums in the i direction
 ******
 */

/****f* physalis/cuda_partial_sum_j()
 * NAME
 *  cuda_partial_sum_j()
 * USAGE
 */
void cuda_partial_sum_j(void);
/* FUNCTION
 *  Communicate partial sums in the j direction
 ******
 */

/****f* physalis/cuda_partial_sum_k()
 * NAME
 *  cuda_partial_sum_k()
 * USAGE
 */
void cuda_partial_sum_k(void);
/* FUNCTION
 *  Communicate partial sums in the k direction
 ******
 */

#endif // _PHYSALIS_H
