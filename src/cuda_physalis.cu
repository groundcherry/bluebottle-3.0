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
#include <cuda.h>
#include <thrust/sort.h>

#include "cuda_physalis.h"
#include "cuda_particle.h"

#include <helper_cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

__constant__ real _A1;
__constant__ real _A2;
__constant__ real _A3;
__constant__ real _B;
__constant__ int _nn[NCOEFFS];
__constant__ int _mm[NCOEFFS];
__constant__ real _node_t[NNODES];
__constant__ real _node_p[NNODES];
real *_int_Yp_re;
real *_int_Yp_im;
real *_int_rDYu_re;
real *_int_rDYu_im;
real *_int_xXDYu_re;
real *_int_xXDYu_im;
real *_sum_send_e;
real *_sum_send_w;
real *_sum_send_n;
real *_sum_send_s;
real *_sum_send_t;
real *_sum_send_b;
real *_sum_recv_e;
real *_sum_recv_w;
real *_sum_recv_n;
real *_sum_recv_s;
real *_sum_recv_t;
real *_sum_recv_b;

extern "C"
void cuda_init_physalis(void)
{
  if (NPARTS > 0) {
    /* set up coefficient table */
    int nn[NCOEFFS] = {0,
                       1, 1,
                       2, 2, 2,
                       3, 3, 3, 3,
                       4, 4, 4, 4, 4};
    int mm[NCOEFFS] = {0,
                       0, 1,
                       0, 1, 2,
                       0, 1, 2, 3,
                       0, 1, 2, 3, 4};

    /* set up quadrature nodes for 7th-order Lebedev quadrature */
    // NOTE: Higher order quadratures exist as comments in bluebottle, in
    // cuda_quadrature.cu:cuda_Lamb()
    real PI14 = 0.25 * PI;
    real PI12 = 0.5 * PI;
    real PI34 = 0.75 * PI;
    real PI54 = 1.25 * PI;
    real PI32 = 1.5 * PI;
    real PI74 = 1.75 * PI;
    real alph1 = 0.955316618124509;
    real alph2 = 2.186276035465284;

    /* weights */
    real A1 = 0.598398600683775;
    real A2 = 0.478718880547015;
    real A3 = 0.403919055461543;
    real B = 0.;

    /* nodes */
    // Find a more elegant way of fixing the divide by sin(0)
    real a1_t[6] = {PI12, PI12, PI12, PI12, 0.+DIV_ST, PI-DIV_ST};
    real a1_p[6] = {0., PI12, PI, PI32, 0., 0.};
    real a2_t[12] = {PI12, PI12, PI12, PI12,
                     PI14, PI14, PI14, PI14,
                     PI34, PI34, PI34, PI34};
    real a2_p[12] = {PI14, PI34, PI54, PI74,
                     0., PI12, PI, PI32,
                     0., PI12, PI, PI32};
    real a3_t[8] = {alph1, alph1, alph1, alph1,
                    alph2, alph2, alph2, alph2};
    real a3_p[8] = {PI14, PI34, PI54, PI74,
                    PI14, PI34, PI54, PI74};

    /* put all quadrature nodes together for interpolation */
    real node_t[NNODES];
    real node_p[NNODES];
    for (int i = 0; i < 6; i++) {
      node_t[i] = a1_t[i];
      node_p[i] = a1_p[i];
    }
    for (int i = 0; i < 12; i++) {
      node_t[6+i] = a2_t[i];
      node_p[6+i] = a2_p[i];
    }
    for (int i = 0; i < 8; i++) {
      node_t[18+i] = a3_t[i];
      node_p[18+i] = a3_p[i];
    }

    /* Bind to cuda device constant memory */
    checkCudaErrors(cudaMemcpyToSymbol(_nn, &nn, NCOEFFS * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(_mm, &mm, NCOEFFS * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(_A1, &A1, sizeof(real)));
    checkCudaErrors(cudaMemcpyToSymbol(_A2, &A2, sizeof(real)));
    checkCudaErrors(cudaMemcpyToSymbol(_A3, &A3, sizeof(real)));
    checkCudaErrors(cudaMemcpyToSymbol(_B, &B, sizeof(real)));
    checkCudaErrors(cudaMemcpyToSymbol(_node_t, &node_t, NNODES * sizeof(real)));
    checkCudaErrors(cudaMemcpyToSymbol(_node_p, &node_p, NNODES * sizeof(real)));
  }
}

extern "C"
void cuda_lamb(void)
{
  /* CUDA exec config */
  dim3 num_parts(nparts); // nparts blocks with nnodes threads each
  dim3 dim_nodes(NNODES);
  dim3 num_partcoeff(nparts, ncoeffs_max);
  dim3 dim_coeff(ncoeffs_max);

  //printf("N%d >> Determining Lamb's coefficients (nparts = %d)\n", rank, nparts);
  if (nparts > 0) {
    /* Temp storage for field variables at quadrature nodes */
    real *_pp;    // pressure
    real *_ur;    // radial velocity
    real *_ut;    // theta velocity
    real *_up;    // phi velocity

    checkCudaErrors(cudaMalloc(&_pp, NNODES * nparts * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_ur, NNODES * nparts * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_ut, NNODES * nparts * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_up, NNODES * nparts * sizeof(real)));

    /* Interpolate field varaibles to quadrature nodes */
    check_nodes<<<num_parts, dim_nodes>>>(nparts, _parts, _bc, _DOM);
    interpolate_nodes<<<num_parts, dim_nodes>>>(_p, _u, _v, _w, rho_f, nu,
      gradP, _parts, _pp, _ur, _ut, _up, _bc);
    
    /* Create scalar product storage using max particle coefficient size */
    int sp_size = nparts * NNODES * ncoeffs_max;
    checkCudaErrors(cudaMalloc(&_int_Yp_re, sp_size * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_int_Yp_im, sp_size * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_int_rDYu_re, sp_size * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_int_rDYu_im, sp_size * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_int_xXDYu_re, sp_size * sizeof(real)));
    checkCudaErrors(cudaMalloc(&_int_xXDYu_im, sp_size * sizeof(real)));

    /* Perform partial sums of lebedev quadrature */
    lebedev_quadrature<<<num_partcoeff, dim_nodes>>>(_parts, ncoeffs_max,
      _pp, _ur, _ut, _up,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);

    checkCudaErrors(cudaFree(_pp));
    checkCudaErrors(cudaFree(_ur));
    checkCudaErrors(cudaFree(_ut));
    checkCudaErrors(cudaFree(_up));
  }

  /* Accumulate partial sums (all procs need to be involved) */
  cuda_partial_sum_i();  // 2a) Calculate partial sums over x face
  cuda_partial_sum_j();  // 2b) Calculate partial sums over y face
  cuda_partial_sum_k();  // 2c) Calculate partial sums over z face

  if (nparts > 0) {
    /* Compute lambs coefficients from partial sums */
    compute_lambs_coeffs<<<num_parts, dim_coeff>>>(_parts, lamb_relax, mu, nu,
      ncoeffs_max, nparts,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);

    /* Calculate hydrodynamic forces */
    // Thread over nparts
    int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                  + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
    int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

    dim3 dim_nparts(t_nparts);
    dim3 num_nparts(b_nparts);

    calc_forces<<<num_nparts, dim_nparts>>>(_parts, nparts, gradP.x, gradP.y,
      gradP.z, rho_f, mu, nu);

    /* Free */
    checkCudaErrors(cudaFree(_int_Yp_re));
    checkCudaErrors(cudaFree(_int_Yp_im));
    checkCudaErrors(cudaFree(_int_rDYu_re));
    checkCudaErrors(cudaFree(_int_rDYu_im));
    checkCudaErrors(cudaFree(_int_xXDYu_re));
    checkCudaErrors(cudaFree(_int_xXDYu_im));
  }
}

extern "C"
void cuda_partial_sum_i(void)
{
  //printf("N%d >> Communicating partial sums in i (nparts %d)\n", rank, nparts);
  /* Outline of communication of partial sums for Lebedev integration
   * 1) Finish local Lebedev integration in lebedev_quad<<<>>>. For a given
   *    scalar product, the partial sum for the jth coefficient of the nth
   *    particle is stored in: _int_someint[0 + NNODES*j + nparts*NNODES*n]
   * 2) All particles at the outermost two bin planes need their sums
   *    accumulated (e.g., (j,k) planes at _bins.Gcc.{_isb->_is,_ie->_ieb})
   * 3) Bin the particles using i indexing (find _bin_{start,end,count})
   * 4) Reduce _bin_count at _isb:_is, _ie:_ieb to find nparts_send_{e,w}
   * 5) Communicate nparts_send_{e,w} with adjacent subdomains to find
   *    nparts_recv_{w,e}
   * 6) Excl. prefix scan _bin_count over the _isb:_is, _ie:_ieb planes to find
   *    destination index for particle data packed into sending aray
   * 7) Allocate send array, int_send_{e,w} * 6 * sizeof(real). 6 comes from
   *    the number of integrals
   * 8) Allocate recv array, int_recv_{e,w} * 6 * sizeof(real).
   * 9) Communicate int_send_{e,w} to int_recv_{e,w}
   * 10)  Excl. prefix scan _bin_count over _isb:_is, _ie:_ieb planes to find unpacking
   *      incides - this already exists from earlier
   * 11)  Unpack and accumulate
   * 12)  Repeat for j, k
   */

  /* Initialize execution config */
  // Thread over east/west faces
  int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);
  int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);

  int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);
  int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);

  dim3 bin_num_inb(by, bz);
  dim3 bin_dim_inb(ty, tz);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b, s2b; // custom strides
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_e;
  int *_offset_w;
  checkCudaErrors(cudaMalloc(&_offset_e, 2 * bins.Gcc.s2b_i * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_w, 2 * bins.Gcc.s2b_i * sizeof(int)));
  thrust::device_ptr<int> t_offset_e(_offset_e);
  thrust::device_ptr<int> t_offset_w(_offset_w);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_i<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_i<<<bin_num_inb, bin_dim_inb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send and packing offsets */
    s1b = bins.Gcc.jnb;
    s2b = s1b * bins.Gcc.knb;

    // East: _ie and _ieb planes
    if (dom[rank].e != MPI_PROC_NULL) {
      // _bin_count is indexed with i varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _ie plane
      offset = GFX_LOC(bins.Gcc._ie, 0, 0, s1b, s2b);
      nparts_send[EAST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + 2 * bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[EAST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_i, t_offset_e);
      } else {
        cudaMemset(_offset_e, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
      }

    } else { // no parts to send
      nparts_send[EAST] = 0;
      cudaMemset(_offset_e, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
    }

    // West: _isb and _is planes
    if (dom[rank].w != MPI_PROC_NULL) {
      offset = GFX_LOC(bins.Gcc._isb, 0, 0, s1b, s2b);
      nparts_send[WEST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + 2 * bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());
      if (nparts_send[WEST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_i, t_offset_w);
      } else {
        cudaMemset(_offset_w, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
      }

    } else {
      nparts_send[WEST] = 0;
      cudaMemset(_offset_w, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
    }
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[EAST] = 0;
    nparts_send[WEST] = 0;
    cudaMemset(_offset_e, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
    cudaMemset(_offset_w, 0., 2 * bins.Gcc.s2b_i * sizeof(int));
  }

  // Sending and receiving is the same since the outer two bin planes are shared
  nparts_recv[EAST] = nparts_send[EAST];
  nparts_recv[WEST] = nparts_send[WEST];

  /* Send number of parts to east/west */
  //    origin                target
  // nparts_send[WEST] -> nparts_recv[EAST]
  // nparts_recv[WEST] <- nparts_send[EAST]
  //nparts_recv[WEST] = 0; // init
  //nparts_recv[EAST] = 0;
  //mpi_send_nparts_i();

  /* Allocate memory for send and recv partial sums */
  int npsums = NSP * ncoeffs_max;  // 6 scalar products * ncoeffs
  // Indexing is, for example:
  //  _sum_send_e[coeff + ncoeffs_max*sp + ncoeffs_max*nsp*part_id]
  // where
  //  part_id = [0, nparts) and sp = [0, 6)
  //    0:  Yp_re     1:  Yp_im
  //    2:  rDYu_re   3:  rDYu_im
  //    4:  xXDYu_re  5:  xXDYu_im

  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_e = nparts_send[EAST]*(nparts_send[EAST] > 0) + (nparts_send[EAST] == 0);
  int send_alloc_w = nparts_send[WEST]*(nparts_send[WEST] > 0) + (nparts_send[WEST] == 0);
  int recv_alloc_e = nparts_recv[EAST]*(nparts_recv[EAST] > 0) + (nparts_recv[EAST] == 0);
  int recv_alloc_w = nparts_recv[WEST]*(nparts_recv[WEST] > 0) + (nparts_recv[WEST] == 0);

  checkCudaErrors(cudaMalloc(&_sum_send_e, send_alloc_e*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_send_w, send_alloc_w*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_e, recv_alloc_e*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_w, recv_alloc_w*npsums*sizeof(real)));

  /* Pack partial sums */
  if (nparts_send[EAST] > 0) {
    pack_sums_e<<<bin_num_inb, bin_dim_inb>>>(_sum_send_e, _offset_e,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  } else {
    //cudaMemset(_sum_send_e, 0., send_alloc_e * npsums * sizeof(real));
  }

  if (nparts_send[WEST] > 0) {
    pack_sums_w<<<bin_num_inb, bin_dim_inb>>>(_sum_send_w, _offset_w,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  } else {
    //cudaMemset(_sum_send_w, 0., send_alloc_w * npsums * sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate partial sums with MPI */
  mpi_send_psums_i();

  // Offsets are the same since they're over both ghost bins and edge bins
  /* Unpack and complete partial sums */
  if (nparts_recv[EAST] > 0) {
    unpack_sums_e<<<bin_num_inb, bin_dim_inb>>>(_sum_recv_e, _offset_e,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  }
  if (nparts_recv[WEST] > 0) {
    unpack_sums_w<<<bin_num_inb, bin_dim_inb>>>(_sum_recv_w, _offset_w,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Free */
  cudaFree(_sum_send_e);
  cudaFree(_sum_send_w);
  cudaFree(_sum_recv_e);
  cudaFree(_sum_recv_w);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_e);
  cudaFree(_offset_w);
}

extern "C"
void cuda_partial_sum_j(void)
{
  //printf("N%d >> Communicating partial sums in j\n", rank);
  /* Initialize execution config */
  // Thread over north/south faces
  int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);
  int tx = bins.Gcc.inb * (bins.Gcc.inb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.inb >= MAX_THREADS_DIM);

  int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);
  int bx = (int) ceil((real) bins.Gcc.inb / (real) tx);

  dim3 bin_num_jnb(bz, bx);
  dim3 bin_dim_jnb(tz, tx);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b, s2b; // custom strides
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_n;
  int *_offset_s;
  checkCudaErrors(cudaMalloc(&_offset_n, 2 * bins.Gcc.s2b_j * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_s, 2 * bins.Gcc.s2b_j * sizeof(int)));
  thrust::device_ptr<int> t_offset_n(_offset_n);
  thrust::device_ptr<int> t_offset_s(_offset_s);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_j<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_j<<<bin_num_jnb, bin_dim_jnb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send and packing offsets */
    s1b = bins.Gcc.knb;
    s2b = s1b * bins.Gcc.inb;

    // North: _je and _jeb planes
    if (dom[rank].n != MPI_PROC_NULL) {
      // _bin_count is indexed with i varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _je plane
      offset = GFY_LOC(0, bins.Gcc._je, 0, s1b, s2b);
      nparts_send[NORTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + 2 * bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[NORTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_j, t_offset_n);
      } else {
        cudaMemset(_offset_n, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
      }

    } else { // no parts to send
      nparts_send[NORTH] = 0;
      cudaMemset(_offset_n, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
    }

    // South: _jsb and _js planes
    if (dom[rank].s != MPI_PROC_NULL) {
      offset = GFY_LOC(0, bins.Gcc._jsb, 0, s1b, s2b);
      nparts_send[SOUTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + 2 * bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());
      if (nparts_send[SOUTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_j, t_offset_s);
      } else {
        cudaMemset(_offset_s, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
      }

    } else {
      nparts_send[SOUTH] = 0;
      cudaMemset(_offset_s, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
    }
  } else { // nparts == 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[NORTH] = 0;
    nparts_send[SOUTH] = 0;
    cudaMemset(_offset_n, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
    cudaMemset(_offset_s, 0., 2 * bins.Gcc.s2b_j * sizeof(int));
  }

  // Sending and receiving is the same since the outer two bin planes are shared
  nparts_recv[NORTH] = nparts_send[NORTH];
  nparts_recv[SOUTH] = nparts_send[SOUTH];

  /* Send number of parts to north/south */
  //    origin                target
  // nparts_send[SOUTH] -> nparts_recv[NORTH]
  // nparts_recv[SOUTH] <- nparts_send[NORTH]
  //nparts_recv[SOUTH] = 0; // init
  //nparts_recv[NORTH] = 0;
  //mpi_send_nparts_j();

  /* Allocate memory for send and recv partial sums */
  int npsums = NSP * ncoeffs_max;  // 6 scalar products * ncoeffs
  // Indexing is, for example:
  //  _sum_send_n[coeff + ncoeffs_max*sp + ncoeffs_max*nsp*part_id]
  // where
  //  part_id = [0, nparts) and sp = [0, 6)
  //    0:  Yp_re     1:  Yp_im
  //    2:  rDYu_re   3:  rDYu_im
  //    4:  xXDYu_re  5:  xXDYu_im

  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_n = nparts_send[NORTH]*(nparts_send[NORTH] > 0) + (nparts_send[NORTH] == 0);
  int send_alloc_s = nparts_send[SOUTH]*(nparts_send[SOUTH] > 0) + (nparts_send[SOUTH] == 0);
  int recv_alloc_n = nparts_recv[NORTH]*(nparts_recv[NORTH] > 0) + (nparts_recv[NORTH] == 0);
  int recv_alloc_s = nparts_recv[SOUTH]*(nparts_recv[SOUTH] > 0) + (nparts_recv[SOUTH] == 0);

  checkCudaErrors(cudaMalloc(&_sum_send_n, send_alloc_n*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_send_s, send_alloc_s*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_n, recv_alloc_n*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_s, recv_alloc_s*npsums*sizeof(real)));

  /* Pack partial sums */
  if (nparts_send[NORTH] > 0) {
    pack_sums_n<<<bin_num_jnb, bin_dim_jnb>>>(_sum_send_n, _offset_n,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  } else {
    //cudaMemset(_sum_send_n, 0., send_alloc_n * npsums * sizeof(real));
  }

  if (nparts_send[SOUTH] > 0) {
    pack_sums_s<<<bin_num_jnb, bin_dim_jnb>>>(_sum_send_s, _offset_s,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  } else {
    //cudaMemset(_sum_send_s, 0., send_alloc_s * npsums * sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate partial sums with MPI */
  mpi_send_psums_j();

  // Offsets are the same since they're over both ghost bins and edge bins
  /* Unpack and complete partial sums */
  if (nparts_recv[NORTH] > 0) {
    unpack_sums_n<<<bin_num_jnb, bin_dim_jnb>>>(_sum_recv_n, _offset_n,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  }
  if (nparts_recv[SOUTH] > 0) {
    unpack_sums_s<<<bin_num_jnb, bin_dim_jnb>>>(_sum_recv_s, _offset_s,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Free */
  cudaFree(_sum_send_n);
  cudaFree(_sum_send_s);
  cudaFree(_sum_recv_n);
  cudaFree(_sum_recv_s);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_n);
  cudaFree(_offset_s);
}

extern "C"
void cuda_partial_sum_k(void)
{
  //printf("N%d >> Communicating partial sums in k\n", rank);
  /* Initialize execution config */
  // Thread over top/bottom faces
  int tx = bins.Gcc.inb * (bins.Gcc.inb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.inb >= MAX_THREADS_DIM);
  int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
       + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);

  int bx = (int) ceil((real) bins.Gcc.inb / (real) tx);
  int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);

  dim3 bin_num_knb(bx, by);
  dim3 bin_dim_knb(tx, ty);

  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Declare things we might need */
  int s1b, s2b; // custom strides
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_t;
  int *_offset_b;
  checkCudaErrors(cudaMalloc(&_offset_t, 2 * bins.Gcc.s2b_k * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_b, 2 * bins.Gcc.s2b_k * sizeof(int)));
  thrust::device_ptr<int> t_offset_t(_offset_t);
  thrust::device_ptr<int> t_offset_b(_offset_b);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send and packing offsets */
    s1b = bins.Gcc.inb;
    s2b = s1b * bins.Gcc.jnb;

    // North: _ke and _keb planes
    if (dom[rank].t != MPI_PROC_NULL) {
      // _bin_count is indexed with k varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _ke plane
      offset = GFZ_LOC(0, 0, bins.Gcc._ke, s1b, s2b);
      nparts_send[TOP] = thrust::reduce(t_bin_count + offset,
                                        t_bin_count + offset + 2 * bins.Gcc.s2b_k,
                                        0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[TOP] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_k, t_offset_t);
      } else {
        cudaMemset(_offset_t, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
      }

    } else { // no parts to send
      nparts_send[TOP] = 0;
      cudaMemset(_offset_t, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
    }

    // South: _ksb and _ks planes
    if (dom[rank].b != MPI_PROC_NULL) {
      offset = GFZ_LOC(0, 0, bins.Gcc._ksb, s1b, s2b);
      nparts_send[BOTTOM] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + 2 * bins.Gcc.s2b_k,
                                           0., thrust::plus<int>());
      if (nparts_send[BOTTOM] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + 2 * bins.Gcc.s2b_k, t_offset_b);
      } else {
        cudaMemset(_offset_b, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
      }

    } else {
      nparts_send[BOTTOM] = 0;
      cudaMemset(_offset_b, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
    }
  } else { // nparts = 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[TOP] = 0;
    nparts_send[BOTTOM] = 0;
    cudaMemset(_offset_t, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
    cudaMemset(_offset_b, 0., 2 * bins.Gcc.s2b_k * sizeof(int));
  }

  // Sending and receiving is the same since the outer two bin planes are shared
  nparts_recv[TOP] = nparts_send[TOP];
  nparts_recv[BOTTOM] = nparts_send[BOTTOM];

  /* Send number of parts to top/bottom */
  //    origin                target
  // nparts_send[BOTTOM] -> nparts_recv[TOP]
  // nparts_recv[BOTTOM] <- nparts_send[TOP]
  //nparts_recv[BOTTOM] = 0; // init
  //nparts_recv[TOP] = 0;
  //mpi_send_nparts_k();

  /* Allocate memory for send and recv partial sums */
  int npsums = NSP * ncoeffs_max;  // 6 scalar products * ncoeffs
  // Indexing is, for example:
  //  _sum_send_t[coeff + ncoeffs_max*sp + ncoeffs_max*nsp*part_id]
  // where
  //  part_id = [0, nparts) and sp = [0, 6)
  //    0:  Yp_re     1:  Yp_im
  //    2:  rDYu_re   3:  rDYu_im
  //    4:  xXDYu_re  5:  xXDYu_im

  int send_alloc_t = nparts_send[TOP]*(nparts_send[TOP] > 0) + (nparts_send[TOP] == 0);
  int send_alloc_b = nparts_send[BOTTOM]*(nparts_send[BOTTOM] > 0) + (nparts_send[BOTTOM] == 0);
  int recv_alloc_t = nparts_recv[TOP]*(nparts_recv[TOP] > 0) + (nparts_recv[TOP] == 0);
  int recv_alloc_b = nparts_recv[BOTTOM]*(nparts_recv[BOTTOM] > 0) + (nparts_recv[BOTTOM] == 0);

  checkCudaErrors(cudaMalloc(&_sum_send_t, send_alloc_t*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_send_b, send_alloc_b*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_t, recv_alloc_t*npsums*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_sum_recv_b, recv_alloc_b*npsums*sizeof(real)));

  /* Pack partial sums */
  if (nparts_send[TOP] > 0) {
    pack_sums_t<<<bin_num_knb, bin_dim_knb>>>(_sum_send_t, _offset_t,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  } else {
    //cudaMemset(_sum_send_t, 0., send_alloc_t * npsums * sizeof(real));
  }

  if (nparts_send[BOTTOM] > 0) {
    pack_sums_b<<<bin_num_knb, bin_dim_knb>>>(_sum_send_b, _offset_b,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  } else {
    //cudaMemset(_sum_send_b, 0., send_alloc_b * npsums * sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate partial sums with MPI */
  mpi_send_psums_k();

  // Offsets are the same since they're over both ghost bins and edge bins
  /* Unpack and complete partial sums */
  if (nparts_recv[TOP] > 0) {
    unpack_sums_t<<<bin_num_knb, bin_dim_knb>>>(_sum_recv_t, _offset_t,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  }
  if (nparts_recv[BOTTOM] > 0) {
    unpack_sums_b<<<bin_num_knb, bin_dim_knb>>>(_sum_recv_b, _offset_b,
      _bin_start, _bin_count, _part_ind, ncoeffs_max,
      _int_Yp_re, _int_Yp_im,
      _int_rDYu_re, _int_rDYu_im,
      _int_xXDYu_re, _int_xXDYu_im);
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Free */
  cudaFree(_sum_send_t);
  cudaFree(_sum_send_b);
  cudaFree(_sum_recv_t);
  cudaFree(_sum_recv_b);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_t);
  cudaFree(_offset_b);
}

extern "C"
real cuda_lamb_err(void)
{
  //printf("N%d >> Determining Lamb's error\n", rank);
  real error = DBL_MIN;
  if (nparts > 0) {
    // create a place to store sorted coefficients and errors
    real *_part_errors;
    cudaMalloc((void**) &_part_errors, nparts*sizeof(real));
    
    // sort the coefficients and calculate errors along the way
    dim3 numBlocks(nparts);
    dim3 dimBlocks(ncoeffs_max);

    compute_error<<<numBlocks, dimBlocks>>>(lamb_cut, ncoeffs_max, nparts,
     _parts, _part_errors);

    // find maximum error of all particles
    thrust::device_ptr<real> t_part_errors(_part_errors);
    error = thrust::reduce(t_part_errors,
                           t_part_errors + nparts,
                           0., thrust::maximum<real>());

    // clean up
    cudaFree(_part_errors);

    // store copy of coefficients for future calculation
    store_coeffs<<<numBlocks, dimBlocks>>>(_parts, nparts, ncoeffs_max);
  }

  // MPI reduce to find max error
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, mpi_real, MPI_MAX, MPI_COMM_WORLD);
  return error;
}
