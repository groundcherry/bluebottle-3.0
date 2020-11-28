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

#include "cuda_particle.h"

#include <helper_cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

__constant__ bin_struct _bins;
real *_force_send_e;
real *_force_send_w;
real *_force_send_n;
real *_force_send_s;
real *_force_send_t;
real *_force_send_b;
real *_force_recv_e;
real *_force_recv_w;
real *_force_recv_n;
real *_force_recv_s;
real *_force_recv_t;
real *_force_recv_b;

extern "C"
void cuda_part_malloc_host(void)
{
  // Flags in cuda_bluebottle.cu:cuda_dom_malloc_host since they are needed even
  // without particles
  checkCudaErrors(cudaMallocHost(&phase, dom[rank].Gcc.s3b * sizeof(int)));
    cpumem += dom[rank].Gcc.s3b * sizeof(int);
  checkCudaErrors(cudaMallocHost(&phase_shell, dom[rank].Gcc.s3b * sizeof(int)));
    cpumem += dom[rank].Gcc.s3b * sizeof(int);
}

extern "C"
void cuda_part_malloc_dev(void)
{
  //printf("N%d >> Allocating device particle memory...\n", rank);

  // Flags in cuda_bluebottle.cu:cuda_dom_malloc_dev since they are needed even
  // without particles

  // Phase
  checkCudaErrors(cudaMalloc(&_phase, dom[rank].Gcc.s3b * sizeof(int)));
    gpumem += dom[rank].Gcc.s3b * sizeof(int);
  checkCudaErrors(cudaMalloc(&_phase_shell, dom[rank].Gcc.s3b * sizeof(int)));
    gpumem += dom[rank].Gcc.s3b * sizeof(int);

  // Allocate device variables
  if (NPARTS > 0) {
    checkCudaErrors(cudaMalloc(&_parts, nparts * sizeof(part_struct)));
      cpumem += nparts * sizeof(part_struct);
    checkCudaErrors(cudaMemcpyToSymbol(_bins, &bins, sizeof(bin_struct)));
    checkCudaErrors(cudaMalloc(&_bin_start, bins.Gcc.s3b * sizeof(int)));
      gpumem += bins.Gcc.s3b * sizeof(int);
    checkCudaErrors(cudaMalloc(&_bin_end, bins.Gcc.s3b * sizeof(int)));
      gpumem += bins.Gcc.s3b * sizeof(int);
    checkCudaErrors(cudaMalloc(&_bin_count, bins.Gcc.s3b * sizeof(int)));
      gpumem += bins.Gcc.s3b * sizeof(int);
  }

  /* These arrays are allocated/free'd in their functions, but listed here for
   * reference
   * _part_ind
   * _part_bin
   * _send_parts_{e,w}
   * _recv_parts_{e,w}
   */

  /* For pointers to pointers, if we need to go back... */
  // https://stackoverflow.com/questions/26111794/how-to-use-pointer-to-pointer-
  //  in-cuda
  // https://stackoverflow.com/questions/15113960/cuda-allocating-array-of-
  //  pointers-to-images-and-the-images
  // https://stackoverflow.com/questions/23609770/cuda-double-pointer-memory-copy
  // -->https://stackoverflow.com/questions/27931630/copying-array-of-pointers-
  // into-device-memory-and-back-cuda
}

extern "C"
void cuda_part_push(void)
{
  if (NPARTS > 0) {
    checkCudaErrors(cudaMemcpy(_parts, parts, nparts * sizeof(part_struct),
      cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaMemcpy(_phase, phase, dom[rank].Gcc.s3b * sizeof(int),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_phase_shell, phase_shell, dom[rank].Gcc.s3b * sizeof(int),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_flag_u, flag_u, dom[rank].Gfx.s3b * sizeof(int),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_flag_v, flag_v, dom[rank].Gfy.s3b * sizeof(int),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_flag_w, flag_w, dom[rank].Gfz.s3b * sizeof(int),
    cudaMemcpyHostToDevice));

}

extern "C"
void cuda_part_pull(void)
{
  /* Declare temporary part structure and nparts_subdom */
  part_struct *_tmp_parts;
  nparts_subdom = 0;

  /* Re-allocate memory */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  if (nparts > 0) {
    // Thread over nparts
    int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                  + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
    int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

    dim3 dim_nparts(t_nparts);
    dim3 num_nparts(b_nparts);

    // thread over top/bottom faces 
    int tx = bins.Gcc.inb * (bins.Gcc.inb < MAX_THREADS_DIM)
         + MAX_THREADS_DIM * (bins.Gcc.inb >= MAX_THREADS_DIM);
    int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
         + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);
    int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
         + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);

    int bx = (int) ceil((real) bins.Gcc.inb / (real) tx);
    int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);
    int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);

    dim3 bin_num_inb(by, bz);
    dim3 bin_dim_inb(ty, tz);
    dim3 bin_num_jnb(bz, bx);
    dim3 bin_dim_jnb(tz, tx);
    dim3 bin_num_knb(bx, by);
    dim3 bin_dim_knb(tx, ty);

    /* Find each particle's bin */
    bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Set ghost bin count to zero (GFZ indexed) */
    zero_ghost_bins_i<<<bin_num_inb, bin_dim_inb>>>(_bin_count);
    zero_ghost_bins_j<<<bin_num_jnb, bin_dim_jnb>>>(_bin_count);
    zero_ghost_bins_k<<<bin_num_knb, bin_dim_knb>>>(_bin_count);

    /* Allocate memory to find bin offset target indices in tmp part_struct */
    int *_bin_offset;
    checkCudaErrors(cudaMalloc(&_bin_offset, bins.Gcc.s3b * sizeof(int)));

    /* Prefix scan _bin_count to find target indices in tmp part_struct */
    thrust::device_ptr<int> t_bin_count(_bin_count);
    thrust::device_ptr<int> t_bin_offset(_bin_offset);
    thrust::exclusive_scan(t_bin_count, t_bin_count + bins.Gcc.s3b, t_bin_offset);

    /* Reduce bin_count to find nparts in subdomain (ghost bins are zero'd) */
    nparts_subdom = thrust::reduce(t_bin_count, t_bin_count + bins.Gcc.s3b,
                                        0., thrust::plus<int>());

    /* Allocate new device part struct (no ghost particles) */
    checkCudaErrors(cudaMalloc(&_tmp_parts, nparts_subdom * sizeof(part_struct)));

    /* Copy subdom parts to tmp part_struct (only in subdom, so [in, jn]) */
    // thread over inner bins (no ghost bins)
    tx = bins.Gcc.in * (bins.Gcc.in < MAX_THREADS_DIM)
     + MAX_THREADS_DIM * (bins.Gcc.in >= MAX_THREADS_DIM);
    ty = bins.Gcc.jn * (bins.Gcc.jn < MAX_THREADS_DIM)
     + MAX_THREADS_DIM * (bins.Gcc.jn >= MAX_THREADS_DIM);
    bx = (int) ceil((real) bins.Gcc.in / (real) tx);
    by = (int) ceil((real) bins.Gcc.jn / (real) ty);
    dim3 bin_num_kn(bx, by);
    dim3 bin_dim_kn(tx, ty);

    copy_subdom_parts<<<bin_num_kn, bin_dim_kn>>>(_tmp_parts, _parts,
      _bin_start, _bin_count, _part_ind, _bin_offset);

    cudaFree(_bin_offset);

  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_subdom = 0;
    checkCudaErrors(cudaMalloc(&_tmp_parts, nparts_subdom * sizeof(part_struct)));
  }

  /* Allocate new host parts with nparts in subdom */
  free(parts);
  parts = (part_struct*) malloc(nparts_subdom * sizeof(part_struct));

  // Pull from device
  checkCudaErrors(cudaMemcpy(parts, _tmp_parts, nparts_subdom * sizeof(part_struct),
    cudaMemcpyDeviceToHost));

  // Free
  cudaFree(_tmp_parts);
  cudaFree(_part_ind);
  cudaFree(_part_bin);

  // Double check the number of particles is correct
  int reduce_parts = 0;
  MPI_Allreduce(&nparts_subdom, &reduce_parts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (reduce_parts != NPARTS) {
    printf("N%d >> Something went wrong. NPARTS = %d, but %d exist\n",
      rank, NPARTS, reduce_parts);
    printf("N%d >> Has %d parts\n", rank, nparts_subdom);
    exit(EXIT_FAILURE);
  }
}

extern "C"
void cuda_part_pull_debug(void)
{
  // Pull ALL particles, including ghosts

  // Allocate new host part_struct with new nparts
  free(parts);
  parts = (part_struct*) malloc(nparts * sizeof(part_struct));

  // Pull all particles from device
  checkCudaErrors(cudaMemcpy(parts, _parts, nparts * sizeof(part_struct),
    cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_part_free(void)
{
  //printf("N%d >> Freeing device particle memory...\n", rank);
  // Flags in cuda_dom_free
  checkCudaErrors(cudaFreeHost(phase));
  checkCudaErrors(cudaFreeHost(phase_shell));

  checkCudaErrors(cudaFree(_phase));
  checkCudaErrors(cudaFree(_phase_shell));
  if (NPARTS > 0) {
    checkCudaErrors(cudaFree(_parts));
    checkCudaErrors(cudaFree(_bin_start));
    checkCudaErrors(cudaFree(_bin_end));
    checkCudaErrors(cudaFree(_bin_count));
  }
}

extern "C"
void cuda_transfer_parts_i(void)
{
  //printf("N%d >> Transfering parts in i, nparts = %d\n", rank, nparts);
  /* Transfer particles east and west
   *  * Bin the particles, indexing with `i` varying slowest
   *  * Sort particles by their bin
   *  * Find start and end of each bin's particles
   *  * Find number of particles in each bin
   *  * Find number of particles in _is & _ie planes. These need to be sent W/E
   *  * Communicate these number east and west. Each process now knows how many
   *    to send and recv
   *  * Allocate memory for particle send and recv
   *  * Copy particles into sending arrays. Each bin can find the offset target
   *    index for its particles by performing a prefix scan.
   *  * Communicate particles east and west, send -> recv
   *  * Recv'd parts exist in the ghost bins and replace whatever existed there
   *    at the last time step. Sum the particles in _isb & _ieb and subtract
   *    from nparts. This, plus the number of particle recv'd from E/W, is the
   *    number of new particles
   *  * Allocate temp part structure to hold all new particles.
   *  * Reduce bin_count from _is->_ie to find nparts that we will keep
   *  * Prefix scan from _ie -> _ie to find offset index for particle copy to
   *    temp struct
   *  * Backfill recv'd particles to the end of the temp array
   *  * Repeat process for j, k to take care of edge, corner. Indexing will be
   *    different to take advantage of memory coalescence and the prefix scan
   *    two steps back
   */

  /* NOTE
   *  cuda-memcheck occasionally produces the error "bulk_kernel_by_value: an
   *  illegal memory address was encountered" error on a (thrust) call to
   *  cudaDeviceSynchronize. This doesn't seem to be reliably reproducible
   *  (occurs on any of the several thrust calls in this function). This does
   *  not seem to affect the results in any way, but should be further
   *  investigated. See bug id 008.
   */

  /* Init execution config -- thread over east/west faces */
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

  /* Allocate memory */
  // These are realloc'd every time
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_e;
  int *_offset_w;
  checkCudaErrors(cudaMalloc(&_offset_e, bins.Gcc.s2b_i * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_w, bins.Gcc.s2b_i * sizeof(int)));
  thrust::device_ptr<int> t_offset_e(_offset_e);
  thrust::device_ptr<int> t_offset_w(_offset_w);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
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

    /* Find number of particles to send, and packing offsets */
    s1b = bins.Gcc.jnb;
    s2b = s1b * bins.Gcc.knb;

    // East
    offset = GFX_LOC(bins.Gcc._ie, 0, 0, s1b, s2b);
    if (dom[rank].e != MPI_PROC_NULL) {
      // _bin_count is indexed with i varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _ie plane
      nparts_send[EAST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[EAST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_i, t_offset_e);
      } else {
        cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
      }

    } else { // no parts to send
      nparts_send[EAST] = 0;
      cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
    }

    // West
    offset = GFX_LOC(bins.Gcc._is, 0, 0, s1b, s2b);
    if (dom[rank].w != MPI_PROC_NULL) {
      nparts_send[WEST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());
      if (nparts_send[WEST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_i, t_offset_w);
      } else {
        cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
      }

    } else {
      nparts_send[WEST] = 0;
      cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
    }

  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[EAST] = 0;
    nparts_send[WEST] = 0;
    cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
    cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
  }

  /* Send number of parts to east/west */
  //    origin                target
  // nparts_send[WEST] -> nparts_recv[EAST]
  // nparts_recv[WEST] <- nparts_send[EAST]
  nparts_recv[WEST] = 0; // init
  nparts_recv[EAST] = 0;
  mpi_send_nparts_i();

  /* Allocate memory for send and receiving particles */
  // NOTE: If no particles need to be sent/received in a given direction, this
  //  allocates a memory location with size zero which returns a null device
  //  pointer. If this is passed to MPI_Win_create(base, ...) as the base in
  //  CUDA 9.0, it causes MPI to hang. This was not an issue in CUDA 7.5
  //
  // The fix involves fooling MPI by allocating a very small amount of dummy
  // information if no particles are to be sent. This gives the location a valid
  // memory pointer, than than a null pointer. The MPI communication still knows
  // that the allocated window size and info to be sent is zero, and nothing is
  // unpacked because that is wrapped in an if-statement already. This doesn't 
  // affect most cases where particles are communicated every direction at every
  // time; this will only affect extremely dilute cases.

  int send_alloc_e = nparts_send[EAST]*(nparts_send[EAST] > 0) + (nparts_send[EAST] == 0);
  int send_alloc_w = nparts_send[WEST]*(nparts_send[WEST] > 0) + (nparts_send[WEST] == 0);
  int recv_alloc_e = nparts_recv[EAST]*(nparts_recv[EAST] > 0) + (nparts_recv[EAST] == 0);
  int recv_alloc_w = nparts_recv[WEST]*(nparts_recv[WEST] > 0) + (nparts_recv[WEST] == 0);

  checkCudaErrors(cudaMalloc(&_send_parts_e, send_alloc_e * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_parts_w, send_alloc_w * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_e, recv_alloc_e * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_w, recv_alloc_w * sizeof(part_struct)));

  /* Pack particles into _send_parts */
  if (nparts_send[EAST] > 0) {
    pack_parts_e<<<bin_num_inb, bin_dim_inb>>>(_send_parts_e, _parts, _offset_e,
      _bin_start, _bin_count, _part_ind);
  } else {  // fill dummy data
    //cudaMemset(_send_parts_e, 0., send_alloc_e * sizeof(part_struct));
  }

  if (nparts_send[WEST] > 0) {
    pack_parts_w<<<bin_num_inb, bin_dim_inb>>>(_send_parts_w, _parts, _offset_w,
      _bin_start, _bin_count, _part_ind);
  } else {  // fill dummy data
    //cudaMemset(_send_parts_w, 0., send_alloc_w * sizeof(part_struct));
  }
  cudaDeviceSynchronize(); // To ensure packing is complete before sending

  /* Communicate particles with MPI */
  mpi_send_parts_i();

  /* Find number of particles currently in the EAST/WEST ghost bins */
  int nparts_ghost[6];

  if (nparts > 0) {
    // East
    offset = GFX_LOC(bins.Gcc._ieb, 0, 0, s1b, s2b);
    if (dom[rank].e != MPI_PROC_NULL) {
      nparts_ghost[EAST] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_i,
                                          0., thrust::plus<int>());
    } else {
      nparts_ghost[EAST] = 0;
    }

    // West
    offset = GFX_LOC(bins.Gcc._isb, 0, 0, s1b, s2b);
    if (dom[rank].w != MPI_PROC_NULL) {
      nparts_ghost[WEST] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_i,
                                          0., thrust::plus<int>());
    } else {
      nparts_ghost[WEST] = 0;
    }
  } else { // no parts
    nparts_ghost[EAST] = 0;
    nparts_ghost[WEST] = 0;
  }

  /* Calculate new number of particles */
  int nparts_old = nparts;
  nparts += nparts_recv[EAST] + nparts_recv[WEST] 
          - nparts_ghost[EAST] - nparts_ghost[WEST];

  /* allocate temporary part struct */
  part_struct *_tmp_parts;
  checkCudaErrors(cudaMalloc(&_tmp_parts, nparts * sizeof(part_struct)));

  if (nparts_old > 0) {
    /* parallel prefix scan of [_is, _ie] of _bin_count */
    int *_offset_all;
    checkCudaErrors(cudaMalloc(&_offset_all, bins.Gcc.s3b * sizeof(int)));
    thrust::device_ptr<int> t_offset_all(_offset_all);

    // Scan over bin_count[_is->_ie, j, k]
    int size = bins.Gcc.s3b - 2*bins.Gcc.s2b_i;
    
    thrust::exclusive_scan(t_bin_count + bins.Gcc.s2b_i,
                           t_bin_count + bins.Gcc.s2b_i + size,
                           t_offset_all + bins.Gcc.s2b_i);

    /* copy bins of particles to tmp_parts */
    copy_central_bin_parts_i<<<bin_num_inb, bin_dim_inb>>>(_tmp_parts, _parts,
      _bin_start, _bin_count, _part_ind, _offset_all);

    cudaFree(_offset_all);

  } else { // no (old) parts
    // Do not need to copy or prefix scan
  }

  /* Copy ghost particles received from WEST */
  if (nparts_recv[WEST] > 0) {
    t_nparts = nparts_recv[WEST] * (nparts_recv[WEST] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[WEST] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[WEST] / (real) t_nparts);

    dim3 dim_nparts_w(t_nparts);
    dim3 num_nparts_w(b_nparts);

    offset = nparts_old - nparts_ghost[WEST] - nparts_ghost[EAST];
    copy_ghost_bin_parts<<<num_nparts_w, dim_nparts_w>>>(_tmp_parts, _recv_parts_w,
      nparts_recv[WEST], offset, WEST, _DOM);
  } else { // nparts_recv[WEST] <= 0
    // Do nothing
  }

  /* Copy ghost particles received from EAST */
  if (nparts_recv[EAST] > 0) {
    t_nparts = nparts_recv[EAST] * (nparts_recv[EAST] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[EAST] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[EAST] / (real) t_nparts);

    dim3 dim_nparts_e(t_nparts);
    dim3 num_nparts_e(b_nparts);

    offset = nparts_old - nparts_ghost[WEST] - nparts_ghost[EAST] 
            + nparts_recv[WEST];
    copy_ghost_bin_parts<<<num_nparts_e, dim_nparts_e>>>(_tmp_parts, _recv_parts_e,
      nparts_recv[EAST], offset, EAST, _DOM);
  } else { // npats_recv[EAST] <= 0
    // Do nothing
  }

  /* Swap pointers to _parts and _tmp_parts */
  part_struct *tmp = _parts;
  _parts = _tmp_parts;
  _tmp_parts = tmp;

//  /* Correct ghost particle position for periodic boundaries */
//  int nparts_added = nparts_recv[EAST] + nparts_recv[WEST];
//  if (nparts_added > 0) {
//    t_nparts = nparts_added * (nparts_added < MAX_THREADS_1D)
//              + MAX_THREADS_1D * (nparts_added >= MAX_THREADS_1D);
//    b_nparts = (int) ceil((real) nparts_added / (real) t_nparts);
//
//    dim3 dim_nparts_a(t_nparts);
//    dim3 num_nparts_a(b_nparts);
//
//    offset = nparts_old - nparts_ghost[WEST] - nparts_ghost[EAST];
//    correct_periodic_boundaries_i<<<num_nparts_a, dim_nparts_a>>>(_parts, 
//      offset, nparts_added, _bc, _DOM);
//  }

  // Free memory
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_e);
  cudaFree(_offset_w);
  cudaFree(_send_parts_e);
  cudaFree(_send_parts_w);
  cudaFree(_recv_parts_e);
  cudaFree(_recv_parts_w);
  cudaFree(_tmp_parts);
}

extern "C"
void cuda_transfer_parts_j(void)
{
  // Steps are the same as in cuda_transfer_part_i, except we index with 'j'
  // varying the slowest

  /* Init execution config */

  // thread over north/south faces 
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

  /* Allocate memory */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_n;
  int *_offset_s;
  checkCudaErrors(cudaMalloc(&_offset_n, bins.Gcc.s2b_j * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_s, bins.Gcc.s2b_j * sizeof(int)));
  thrust::device_ptr<int> t_offset_n(_offset_n);
  thrust::device_ptr<int> t_offset_s(_offset_s);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
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

    /* Find number of particles to send, and packing offsets */
    s1b = bins.Gcc.knb;
    s2b = s1b * bins.Gcc.inb;
  
    // North
    offset = GFY_LOC(0, bins.Gcc._je, 0, s1b, s2b);
    if (dom[rank].n != MPI_PROC_NULL) {
      // _bin_count is indexed with j varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _je plane
      nparts_send[NORTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[NORTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_j, t_offset_n);
      } else {
        cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
      }
      
    } else {
      nparts_send[NORTH] = 0;
      cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
    }
  
    // South
    offset = GFY_LOC(0, bins.Gcc._js, 0, s1b, s2b);
    if (dom[rank].s != MPI_PROC_NULL) {
      nparts_send[SOUTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());

      if (nparts_send[SOUTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_j, t_offset_s);
      } else {
        cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
      }

    } else {
      nparts_send[SOUTH] = 0;
      cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
    }
  
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[NORTH] = 0;
    nparts_send[SOUTH] = 0;
    cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
    cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
  }

  /* Send number of parts to north/south */
  nparts_recv[SOUTH] = 0; // init
  nparts_recv[NORTH] = 0;
  mpi_send_nparts_j();

  /* Allocate memory for send and receiving particles */
  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_n = nparts_send[NORTH]*(nparts_send[NORTH] > 0) + (nparts_send[NORTH] == 0);
  int send_alloc_s = nparts_send[SOUTH]*(nparts_send[SOUTH] > 0) + (nparts_send[SOUTH] == 0);
  int recv_alloc_n = nparts_recv[NORTH]*(nparts_recv[NORTH] > 0) + (nparts_recv[NORTH] == 0);
  int recv_alloc_s = nparts_recv[SOUTH]*(nparts_recv[SOUTH] > 0) + (nparts_recv[SOUTH] == 0);

  checkCudaErrors(cudaMalloc(&_send_parts_n, send_alloc_n * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_parts_s, send_alloc_s * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_n, recv_alloc_n * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_s, recv_alloc_s * sizeof(part_struct)));

  /* Pack particles into _send_parts */
  if (nparts_send[NORTH] > 0)  {
    pack_parts_n<<<bin_num_jnb, bin_dim_jnb>>>(_send_parts_n, _parts, _offset_n,
      _bin_start, _bin_count, _part_ind);
  } else { // fill dummy data
    //cudaMemset(_send_parts_n, 0., send_alloc_n * sizeof(part_struct));
  }

  if (nparts_send[SOUTH] > 0)  {
    pack_parts_s<<<bin_num_jnb, bin_dim_jnb>>>(_send_parts_s, _parts, _offset_s,
      _bin_start, _bin_count, _part_ind);
  } else { // fill dummy data
    //cudaMemset(_send_parts_s, 0., send_alloc_s * sizeof(part_struct));
  }
  cudaDeviceSynchronize(); // To ensure packing is complete before sending

  /* Communicate particles with MPI */
  mpi_send_parts_j();

  /* Find number of particles currently in the NORTH/SOUTH ghost bins */
  int nparts_ghost[6];

  if (nparts > 0) {
    // North
    offset = GFY_LOC(0, bins.Gcc._jeb, 0, s1b, s2b);
    if (dom[rank].n != MPI_PROC_NULL) {
      nparts_ghost[NORTH] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + bins.Gcc.s2b_j,
                                           0., thrust::plus<int>());
    } else {
      nparts_ghost[NORTH] = 0;
    }

    // South
    offset = GFY_LOC(0, bins.Gcc._jsb, 0, s1b, s2b);
    if (dom[rank].s != MPI_PROC_NULL) {
      nparts_ghost[SOUTH] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + bins.Gcc.s2b_j,
                                           0., thrust::plus<int>());
    } else {
      nparts_ghost[SOUTH] = 0;
    }
  } else { // no parts
    nparts_ghost[NORTH] = 0;
    nparts_ghost[SOUTH] = 0;
  }

  /* Calculate new number of particles */
  int nparts_old = nparts;
  nparts += nparts_recv[NORTH] + nparts_recv[SOUTH] 
          - nparts_ghost[NORTH] - nparts_ghost[SOUTH];

  /* allocate temporary part struct */
  part_struct *_tmp_parts;
  checkCudaErrors(cudaMalloc(&_tmp_parts, nparts * sizeof(part_struct)));

  if (nparts_old > 0) {
    /* parallel prefix scan of ALL of _bin_count */
    int *_offset_all;
    checkCudaErrors(cudaMalloc(&_offset_all, bins.Gcc.s3b * sizeof(int)));

    // Scan over bin_count[i, _js->_je, k]
    int size = bins.Gcc.s3b - 2*bins.Gcc.s2b_j;
    thrust::device_ptr<int> t_offset_all(_offset_all);
    thrust::exclusive_scan(t_bin_count + bins.Gcc.s2b_j,
                           t_bin_count + bins.Gcc.s2b_j + size,
                           t_offset_all + bins.Gcc.s2b_j);


    /* copy bins of particles to tmp_parts */
    copy_central_bin_parts_j<<<bin_num_jnb, bin_dim_jnb>>>(_tmp_parts, _parts,
      _bin_start, _bin_count, _part_ind, _offset_all);

    cudaFree(_offset_all);

  } else { // no (old) parts
    // Do nothing
  }

  /* Copy ghost particles recieved from SOUTH */
  if (nparts_recv[SOUTH] > 0) {
    t_nparts = nparts_recv[SOUTH] * (nparts_recv[SOUTH] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[SOUTH] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[SOUTH] / (real) t_nparts);

    dim3 dim_nparts_s(t_nparts);
    dim3 num_nparts_s(b_nparts);

    offset = nparts_old - nparts_ghost[SOUTH] - nparts_ghost[NORTH];
    copy_ghost_bin_parts<<<num_nparts_s, dim_nparts_s>>>(_tmp_parts, _recv_parts_s,
      nparts_recv[SOUTH], offset, SOUTH, _DOM);
  } else { // nparts_recv[SOUTH] <= 0
    // Do nothing
  }

  /* Copy ghost particles received from NORTH */
  if (nparts_recv[NORTH] > 0) {
    t_nparts = nparts_recv[NORTH] * (nparts_recv[NORTH] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[NORTH] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[NORTH] / (real) t_nparts);

    dim3 dim_nparts_n(t_nparts);
    dim3 num_nparts_n(b_nparts);

    offset = nparts_old - nparts_ghost[SOUTH] - nparts_ghost[NORTH]
            + nparts_recv[SOUTH];
    copy_ghost_bin_parts<<<num_nparts_n, dim_nparts_n>>>(_tmp_parts, _recv_parts_n,
      nparts_recv[NORTH], offset, NORTH, _DOM);
  } else { // nparts_recv[NORTH] <= 0
    // Do nothing
  }

  /* Swap pointers to _parts and _tmp_parts */
  part_struct *tmp = _parts;
  _parts = _tmp_parts;
  _tmp_parts = tmp;

//  /* Correct ghost particle position for periodic boundaries */
//  int nparts_added = nparts_recv[NORTH] + nparts_recv[SOUTH];
//  if (nparts_added > 0) {
//    t_nparts = nparts_added * (nparts_added < MAX_THREADS_1D)
//              + MAX_THREADS_1D * (nparts_added >= MAX_THREADS_1D);
//    b_nparts = (int) ceil((real) nparts_added / (real) t_nparts);
//
//    dim3 dim_nparts_a(t_nparts);
//    dim3 num_nparts_a(b_nparts);
//
//    offset = nparts_old - nparts_ghost[SOUTH] - nparts_ghost[NORTH];
//    correct_periodic_boundaries_j<<<num_nparts_a, dim_nparts_a>>>(_parts, 
//      offset, nparts_added, _bc, _DOM);
//  }

  // Free memory
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_n);
  cudaFree(_offset_s);
  cudaFree(_send_parts_n);
  cudaFree(_send_parts_s);
  cudaFree(_recv_parts_n);
  cudaFree(_recv_parts_s);
  cudaFree(_tmp_parts);
}

extern "C"
void cuda_transfer_parts_k(void)
{
  // Steps are the same as in cuda_transfer_part_i, except we index with 'k'
  // varying the slowest

  /* Init execution config */

  // thread over top/bottom faces 
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
  int s1b = bins.Gcc.inb;
  int s2b = s1b * bins.Gcc.jnb;
  int offset;

  /* Allocate memory */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_t;
  int *_offset_b;
  checkCudaErrors(cudaMalloc(&_offset_t, bins.Gcc.s2b_k * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_b, bins.Gcc.s2b_k * sizeof(int)));
  thrust::device_ptr<int> t_offset_t(_offset_t);
  thrust::device_ptr<int> t_offset_b(_offset_b);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
  if (nparts > 0) {
    /* Find each particle's bin */
    bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    if (nparts > 1) {
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);
    }
    //_part_bin = thrust::raw_pointer_cast(t_part_bin);
    //_part_ind = thrust::raw_pointer_cast(t_part_ind);

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    count_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_bin_start, _bin_end,
      _bin_count);

    /* Find number of particles to send, and packing offsets */
    // Top
    offset = GFZ_LOC(0, 0, bins.Gcc._ke, s1b, s2b);
    if (dom[rank].t != MPI_PROC_NULL) {
      // _bin_count is indexed with k varying slowest -- can do a reduction
      // directly from _bin_count, given the offset of the start of the _ke plane
      nparts_send[TOP] = thrust::reduce(t_bin_count + offset,
                                        t_bin_count + offset + bins.Gcc.s2b_k,
                                          0., thrust::plus<int>());
    
      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[TOP] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_k, t_offset_t);
      } else {
        cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
      }

    } else {
      nparts_send[TOP] = 0;
      cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
    }

    // Bottom
    offset = GFZ_LOC(0, 0, bins.Gcc._ks, s1b, s2b);
    if (dom[rank].b != MPI_PROC_NULL) {
      nparts_send[BOTTOM] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + bins.Gcc.s2b_k,
                                           0., thrust::plus<int>());

      if (nparts_send[BOTTOM] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_k, t_offset_b);
      } else {
        cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
      }

    } else {
      nparts_send[BOTTOM] = 0;
      cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
    }
    
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[TOP] = 0;
    nparts_send[BOTTOM] = 0;
    cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
    cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
  }

  /* Send number of parts to top/bottom */
  nparts_recv[TOP] = 0; // init
  nparts_recv[BOTTOM] = 0;
  mpi_send_nparts_k();

  /* Allocate memory for send and receiving particles */
  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_t = nparts_send[TOP]*(nparts_send[TOP] > 0) + (nparts_send[TOP] == 0);
  int send_alloc_b = nparts_send[BOTTOM]*(nparts_send[BOTTOM] > 0) + (nparts_send[BOTTOM] == 0);
  int recv_alloc_t = nparts_recv[TOP]*(nparts_recv[TOP] > 0) + (nparts_recv[TOP] == 0);
  int recv_alloc_b = nparts_recv[BOTTOM]*(nparts_recv[BOTTOM] > 0) + (nparts_recv[BOTTOM] == 0);

  checkCudaErrors(cudaMalloc(&_send_parts_t, send_alloc_t * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_send_parts_b, send_alloc_b * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_t, recv_alloc_t * sizeof(part_struct)));
  checkCudaErrors(cudaMalloc(&_recv_parts_b, recv_alloc_b * sizeof(part_struct)));

  /* Pack particles into _send_parts */
  if (nparts_send[TOP] > 0) {
    pack_parts_t<<<bin_num_knb, bin_dim_knb>>>(_send_parts_t, _parts, _offset_t,
      _bin_start, _bin_count, _part_ind);
  } else {  // fill dummy data
    //cudaMemset(_send_parts_t, 0., send_alloc_t * sizeof(part_struct));
  }

  if (nparts_send[BOTTOM] > 0) {
    pack_parts_b<<<bin_num_knb, bin_dim_knb>>>(_send_parts_b, _parts, _offset_b,
      _bin_start, _bin_count, _part_ind);
  } else {  // fill dummy data
    //cudaMemset(_send_parts_b, 0., send_alloc_b * sizeof(part_struct));
  }
  cudaDeviceSynchronize(); // To ensure packing is complete before sending

  /* Communicate particles with MPI */
  mpi_send_parts_k();

  /* Find number of particles currently in the TOP/BOTTOM ghost bins */
  int nparts_ghost[6];
  
  if (nparts > 0) {
    // TOP
    offset = GFZ_LOC(0, 0, bins.Gcc._keb, s1b, s2b);
    if (dom[rank].t != MPI_PROC_NULL) {
      nparts_ghost[TOP] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_k,
                                         0., thrust::plus<int>());
    } else {
      nparts_ghost[TOP] = 0;
    }

    // BOTTOM
    offset = GFZ_LOC(0, 0, bins.Gcc._ksb, s1b, s2b);
    if (dom[rank].b != MPI_PROC_NULL) {
      nparts_ghost[BOTTOM] = thrust::reduce(t_bin_count + offset,
                                            t_bin_count + offset + bins.Gcc.s2b_k,
                                            0., thrust::plus<int>());
    } else {
      nparts_ghost[BOTTOM] = 0;
    }
  } else { // no parts
    nparts_ghost[TOP] = 0;
    nparts_ghost[BOTTOM] = 0;
  }

  /* Calculate new number of particles */
  int nparts_old = nparts;
  nparts += nparts_recv[TOP] + nparts_recv[BOTTOM] 
          - nparts_ghost[TOP] - nparts_ghost[BOTTOM];

  /* allocate temporary part struct */
  part_struct *_tmp_parts;
  checkCudaErrors(cudaMalloc(&_tmp_parts, nparts * sizeof(part_struct)));

  if (nparts_old > 0) {
    /* parallel prefix scan of ALL of _bin_count */
    int *_offset_all;
    checkCudaErrors(cudaMalloc(&_offset_all, bins.Gcc.s3b * sizeof(int)));

    // Scan over bin_count[i, m, _ks->_ke]
    int size = bins.Gcc.s3b - 2*bins.Gcc.s2b_k;
    thrust::device_ptr<int> t_offset_all(_offset_all);
    thrust::exclusive_scan(t_bin_count + bins.Gcc.s2b_k,
                           t_bin_count + bins.Gcc.s2b_k + size,
                           t_offset_all + bins.Gcc.s2b_k);


    /* copy bins of particles to tmp_parts */
    copy_central_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_tmp_parts, _parts,
      _bin_start, _bin_count, _part_ind, _offset_all);

    cudaFree(_offset_all);

  } else { // no (old) parts
    // Do nothing
  }

  /* Copy ghost particles recieved from BOTTOM */
  if (nparts_recv[BOTTOM] > 0) {
    t_nparts = nparts_recv[BOTTOM] * (nparts_recv[BOTTOM] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[BOTTOM] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[BOTTOM] / (real) t_nparts);

    dim3 dim_nparts_b(t_nparts);
    dim3 num_nparts_b(b_nparts);

    offset = nparts_old - nparts_ghost[BOTTOM] - nparts_ghost[TOP];
    copy_ghost_bin_parts<<<num_nparts_b, dim_nparts_b>>>(_tmp_parts, _recv_parts_b,
      nparts_recv[BOTTOM], offset, BOTTOM, _DOM);
  } else { // nparts_recv[BOTTOM] <= 0
    // Do nothing
  }

  /* Copy ghost particles received from TOP */
  if (nparts_recv[TOP] > 0) {
    t_nparts = nparts_recv[TOP] * (nparts_recv[TOP] < MAX_THREADS_1D)
              + MAX_THREADS_1D * (nparts_recv[TOP] >= MAX_THREADS_1D);
    b_nparts = (int) ceil((real) nparts_recv[TOP] / (real) t_nparts);

    dim3 dim_nparts_t(t_nparts);
    dim3 num_nparts_t(b_nparts);

    offset = nparts_old - nparts_ghost[BOTTOM] - nparts_ghost[TOP]
            + nparts_recv[BOTTOM];
    copy_ghost_bin_parts<<<num_nparts_t, dim_nparts_t>>>(_tmp_parts, _recv_parts_t,
      nparts_recv[TOP], offset, TOP, _DOM);
  } else { // nparts_recv[TOP] <= 0
    // Do nothing
  }

  /* Swap pointers to _parts and _tmp_parts */
  part_struct *tmp = _parts;
  _parts = _tmp_parts;
  _tmp_parts = tmp;

//  /* Correct ghost particle position for periodic boundaries */
//  int nparts_added = nparts_recv[TOP] + nparts_recv[BOTTOM];
//  if (nparts_added > 0) {
//    t_nparts = nparts_added * (nparts_added < MAX_THREADS_1D)
//              + MAX_THREADS_1D * (nparts_added >= MAX_THREADS_1D);
//    b_nparts = (int) ceil((real) nparts_added / (real) t_nparts);
//
//    dim3 dim_nparts_a(t_nparts);
//    dim3 num_nparts_a(b_nparts);
//
//    offset = nparts_old - nparts_ghost[BOTTOM] - nparts_ghost[TOP];
//    correct_periodic_boundaries_k<<<num_nparts_a, dim_nparts_a>>>(_parts, 
//      offset, nparts_added, _bc, _DOM);
//   
//  }

  // Free memory
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_t);
  cudaFree(_offset_b);
  cudaFree(_send_parts_t);
  cudaFree(_send_parts_b);
  cudaFree(_recv_parts_t);
  cudaFree(_recv_parts_b);
  cudaFree(_tmp_parts);
}

extern "C"
void cuda_move_parts()
{
  //printf("N%d >> Moving parts (nparts %d)\n", rank, nparts);
  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  if (nparts > 0) {
    real eps = 0.01;  // compact support parameter

    if (nparts == 1) {
      collision_init<<<num_nparts, dim_nparts>>>(_parts, nparts);
      spring_parts<<<num_nparts, dim_nparts>>>(_parts, nparts, _DOM);
      collision_walls<<<num_nparts, dim_nparts>>>(_parts, nparts, _bc, eps, mu,
        rho_f, nu, interaction_length_ratio, dt, _DOM);

      /* Communicate forces to prevent MPI hang */
      cuda_update_part_forces_i();
      cuda_update_part_forces_j();
      cuda_update_part_forces_k();

      move_parts_a<<<num_nparts, dim_nparts>>>(_parts, nparts, dt, g, gradP,
       rho_f);

      collision_init<<<num_nparts, dim_nparts>>>(_parts, nparts);
      spring_parts<<<num_nparts, dim_nparts>>>(_parts, nparts, _DOM);
      collision_walls<<<num_nparts, dim_nparts>>>(_parts, nparts, _bc, eps, mu,
        rho_f, nu, interaction_length_ratio, dt, _DOM);

    } else if (nparts > 1) {
      /* Initialize forces to zero */
      collision_init<<<num_nparts, dim_nparts>>>(_parts, nparts);

      /* Allocate memory */
      checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
      checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
      thrust::device_ptr<int> t_part_ind(_part_ind);
      thrust::device_ptr<int> t_part_bin(_part_bin);
      
      /* Reset memory */
      checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
      checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
      checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
      thrust::device_ptr<int> t_bin_count(_bin_count);

      /* Bin particles */
      bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
        _DOM);

      /* Sort _part_ind by _part_bin (sort key by value) */
      thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);

      /* Find start and ending index of each bin */
      int smem_size = (nparts + 1) * sizeof(int);
      find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
        _part_bin, nparts);

      /* Find number of particles in each bin */
      //count_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_bin_start, _bin_end,
      //  _bin_count);

      /* Deal with particle-particle collisions */
      collision_parts<<<num_nparts, dim_nparts>>>(_parts, nparts,
       eps, mu, rho_f, nu, _bc, _bin_start, _bin_end, _part_bin,
       _part_ind, interaction_length_ratio, dt);

      /* Calculate spring forces on particles */
      spring_parts<<<num_nparts, dim_nparts>>>(_parts, nparts, _DOM);

      /* Calculate wall collision forces */
      collision_walls<<<num_nparts, dim_nparts>>>(_parts, nparts, _bc, eps, mu,
        rho_f, nu, interaction_length_ratio, dt, _DOM);

      /* Free _part_bin, _part_ind (re-malloc'd in comm functions) */
      cudaFree(_part_ind);
      cudaFree(_part_bin);

      /* Communicate forces */
      cuda_update_part_forces_i();
      cuda_update_part_forces_j();
      cuda_update_part_forces_k();

      /*** Update velocities and accelerations ***/
      move_parts_a<<<num_nparts, dim_nparts>>>(_parts, nparts, dt, g, gradP,
       rho_f);

      /* Re-alloc memory */
      checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
      checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
      thrust::device_ptr<int> t_part_ind2(_part_ind);
      thrust::device_ptr<int> t_part_bin2(_part_bin);

      /* Reset memory */
      checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
      checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
      //checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
      //thrust::device_ptr<int> t_bin_count(_bin_count);

      /* Bin particles */
      bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
        _DOM);

      /* Sort _part_ind by _part_bin (sort key by value) */
      thrust::sort_by_key(t_part_bin2, t_part_bin2 + nparts, t_part_ind2);

      /* Find start and ending index of each bin */
      find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
        _part_bin, nparts);

      /* Initialize forces to zero */
      collision_init<<<num_nparts, dim_nparts>>>(_parts, nparts);

      /* Deal with particle-particle collisions */
      collision_parts<<<num_nparts, dim_nparts>>>(_parts, nparts,
       eps, mu, rho_f, nu, _bc, _bin_start, _bin_end, _part_bin,
       _part_ind, interaction_length_ratio, dt);

      /* Calculate spring forces on particles */
      spring_parts<<<num_nparts, dim_nparts>>>(_parts, nparts, _DOM);

      /* Calculate wall collision forces */
      collision_walls<<<num_nparts, dim_nparts>>>(_parts, nparts, _bc, eps, mu,
        rho_f, nu, interaction_length_ratio, dt, _DOM);
    
      /* Free memory */
      cudaFree(_part_ind);
      cudaFree(_part_bin);
    } // end if (nparts > 1)

    /* Communicate forces */
    cuda_update_part_forces_i();
    cuda_update_part_forces_j();
    cuda_update_part_forces_k();

    /* Move particles */
    move_parts_b<<<num_nparts, dim_nparts>>>(_parts, nparts, dt, g, gradP,
     rho_f);

  } // end if (nparts > 0)
}

extern "C"
void cuda_move_parts_sub()
{
  //printf("N%d >> Moving parts (sub-Lamb's iteration) (nparts %d)\n", rank, nparts);
  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  real eps = 0.01;  // compact support parameter

  if (nparts == 0) {
    /* Communicate forces to prevent MPI hang */
    cuda_update_part_forces_i();
    cuda_update_part_forces_j();
    cuda_update_part_forces_k();

  } else if (nparts == 1) {
    collision_init<<<num_nparts, dim_nparts>>>(_parts, nparts);
    spring_parts<<<num_nparts, dim_nparts>>>(_parts, nparts, _DOM);
    collision_walls<<<num_nparts, dim_nparts>>>(_parts, nparts, _bc, eps, mu,
      rho_f, nu, interaction_length_ratio, dt, _DOM);

    /* Communicate forces to prevent MPI hang */
    cuda_update_part_forces_i();
    cuda_update_part_forces_j();
    cuda_update_part_forces_k();

    move_parts_a<<<num_nparts, dim_nparts>>>(_parts, nparts, dt, g, gradP,
     rho_f);

    collision_init<<<num_nparts, dim_nparts>>>(_parts, nparts);
    spring_parts<<<num_nparts, dim_nparts>>>(_parts, nparts, _DOM);
    collision_walls<<<num_nparts, dim_nparts>>>(_parts, nparts, _bc, eps, mu,
      rho_f, nu, interaction_length_ratio, dt, _DOM);

  } else if (nparts > 1) {
    /* Initialize forces to zero */
    collision_init<<<num_nparts, dim_nparts>>>(_parts, nparts);

    /* Allocate memory */
    checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
    checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
    thrust::device_ptr<int> t_part_ind(_part_ind);
    thrust::device_ptr<int> t_part_bin(_part_bin);
    
    /* Reset memory */
    checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
    checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
    checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
    thrust::device_ptr<int> t_bin_count(_bin_count);

    /* Bin particles */
    bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    thrust::sort_by_key(t_part_bin, t_part_bin + nparts, t_part_ind);

    /* Find start and ending index of each bin */
    int smem_size = (nparts + 1) * sizeof(int);
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Find number of particles in each bin */
    //count_bin_parts_k<<<bin_num_knb, bin_dim_knb>>>(_bin_start, _bin_end,
    //  _bin_count);

    /* Deal with particle-particle collisions */
    collision_parts<<<num_nparts, dim_nparts>>>(_parts, nparts,
     eps, mu, rho_f, nu, _bc, _bin_start, _bin_end, _part_bin,
     _part_ind, interaction_length_ratio, dt);

    /* Calculate spring forces on particles */
    spring_parts<<<num_nparts, dim_nparts>>>(_parts, nparts, _DOM);

    /* Calculate wall collision forces */
    collision_walls<<<num_nparts, dim_nparts>>>(_parts, nparts, _bc, eps, mu,
      rho_f, nu, interaction_length_ratio, dt, _DOM);

    /* Free _part_bin, _part_ind (re-malloc'd in comm functions) */
    checkCudaErrors(cudaFree(_part_ind));
    checkCudaErrors(cudaFree(_part_bin));

    /* Communicate forces */
    cuda_update_part_forces_i();
    cuda_update_part_forces_j();
    cuda_update_part_forces_k();  // uses bin_fill_k

    /*** Update velocities and accelerations ***/
    move_parts_a<<<num_nparts, dim_nparts>>>(_parts, nparts, dt, g, gradP,
     rho_f);

    /* Re-alloc memory */
    checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
    checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
    thrust::device_ptr<int> t_part_ind2(_part_ind);
    thrust::device_ptr<int> t_part_bin2(_part_bin);

    /* Reset memory */
    checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
    checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
    //checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
    //thrust::device_ptr<int> t_bin_count(_bin_count);

    /* Bin particles */
    bin_fill_k<<<num_nparts, dim_nparts>>>(_part_ind, _part_bin, _parts, nparts,
      _DOM);

    /* Sort _part_ind by _part_bin (sort key by value) */
    thrust::sort_by_key(t_part_bin2, t_part_bin2 + nparts, t_part_ind2);

    /* Find start and ending index of each bin */
    find_bin_start_end<<<b_nparts, t_nparts, smem_size>>>(_bin_start, _bin_end,
      _part_bin, nparts);

    /* Initialize forces to zero */
    collision_init<<<num_nparts, dim_nparts>>>(_parts, nparts);

    /* Deal with particle-particle collisions */
    collision_parts<<<num_nparts, dim_nparts>>>(_parts, nparts,
     eps, mu, rho_f, nu, _bc, _bin_start, _bin_end, _part_bin,
     _part_ind, interaction_length_ratio, dt);

    /* Calculate spring forces on particles */
    spring_parts<<<num_nparts, dim_nparts>>>(_parts, nparts, _DOM);

    /* Calculate wall collision forces */
    collision_walls<<<num_nparts, dim_nparts>>>(_parts, nparts, _bc, eps, mu,
      rho_f, nu, interaction_length_ratio, dt, _DOM);
  
    /* Free memory */
    checkCudaErrors(cudaFree(_part_ind));
    checkCudaErrors(cudaFree(_part_bin));

  } // end if (nparts > 1)
}


extern "C"
void cuda_update_part_velocity()
{
  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  /* Communicate forces to prevent MPI hang */
  cuda_update_part_forces_i();
  cuda_update_part_forces_j();
  cuda_update_part_forces_k();

  if (nparts > 0) {
    move_parts_a<<<num_nparts, dim_nparts>>>(_parts, nparts, dt, g, gradP,
     rho_f);
  }
}

extern "C"
void cuda_update_part_position()
{
  // Thread over nparts
  int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
  int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

  dim3 dim_nparts(t_nparts);
  dim3 num_nparts(b_nparts);

  if (nparts > 0) {
    move_parts_b<<<num_nparts, dim_nparts>>>(_parts, nparts, dt, g, gradP,
     rho_f);
  }
}

extern "C"
void cuda_build_cages(void)
{
  /* Reset flag_{u,v,w} to fluid */
  reset_flag_u<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(_flag_u);
  reset_flag_v<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(_flag_v);
  reset_flag_w<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(_flag_w);

  /* Reset phase, phase_shell to fluid */
  if (NPARTS > 0) {
    reset_phases<<<blocks.Gcc.num_knb, blocks.Gcc.dim_knb>>>(_phase, _phase_shell);

    /* Init exec configuration */
    int tx = 0.5*MAX_THREADS_DIM;
    int ty = 0.5*MAX_THREADS_DIM;
    int tz = 0.5*MAX_THREADS_DIM;

    real itx = 1./tx;
    real ity = 1./ty;
    real itz = 1./tz;
    
    int cage_dim[3];
    int *_cage_dim;
    checkCudaErrors(cudaMalloc(&_cage_dim, 3 * sizeof(int)));

    /* build phase */
    for (int n = 0; n < nparts; n++) {
      // Set up cage extents
      // _parts is different than parts, so we need to do this device-side
      // and copy back to get exec config
      cage_setup<<<1,1>>>(_parts, n, _cage_dim);

      cudaMemcpy(cage_dim, _cage_dim, 3 * sizeof(int), cudaMemcpyDeviceToHost);


      int bx = (int) ceil((real) cage_dim[0] * itx);
      int by = (int) ceil((real) cage_dim[1] * ity);
      int bz = (int) ceil((real) cage_dim[2] * itz);

      dim3 dimb_3(tx, ty, tz);
      dim3 numb_3(bx, by, bz);

      if (bx > 0 && by > 0 && bz > 0) {
        build_phase<<<numb_3, dimb_3>>>(_parts, n, _cage_dim, _phase,
          _phase_shell, _DOM, _bc);
      }
    }

    /* build phase_shell (needs phase to exist) */
    for (int n = 0; n < nparts; n++) {
      // Set up cage extents
      // _parts is different than parts, so we need to do this device-side
      // and copy back to get exec config
      cage_setup<<<1,1>>>(_parts, n, _cage_dim);

      cudaMemcpy(cage_dim, _cage_dim, 3 * sizeof(int), cudaMemcpyDeviceToHost);


      int bx = (int) ceil((real) cage_dim[0] * itx);
      int by = (int) ceil((real) cage_dim[1] * ity);
      int bz = (int) ceil((real) cage_dim[2] * itz);

      dim3 dimb_3(tx, ty, tz);
      dim3 numb_3(bx, by, bz);

      if (bx > 0 && by > 0 && bz > 0) {
        build_phase_shell<<<numb_3, dimb_3>>>(_parts, n, _cage_dim, _phase,
          _phase_shell, _DOM, _bc);
      }
    }


    cudaFree(_cage_dim);

    //phase_shell_x<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(_parts, _phase, _phase_shell);
    //phase_shell_y<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(_parts, _phase, _phase_shell);
    //phase_shell_z<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_parts, _phase, _phase_shell);

    /* Build flags from phase, phase_shell */
    // Need phase shell at ghost cells, but not flag
    cage_flag_u<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(_flag_u, _phase, _phase_shell);
    cage_flag_v<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(_flag_v, _phase, _phase_shell);
    cage_flag_w<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(_flag_w, _phase, _phase_shell);
  }

  /* Flag external boundaries 
   *  * Only for non-periodic conditions 
   *  * Only if subdomain is on domain boundary
   */

  // i direction
  if (bc.pW != PERIODIC && bc.pE != PERIODIC) {
    if (dom[rank].I == DOM.Is) {
      flag_external_u<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(_flag_u, 
        dom[rank].Gfx._is);
    }
    if (dom[rank].I == DOM.Ie) {
      flag_external_u<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(_flag_u, 
        dom[rank].Gfx._ie);
    }
  }

  // j direction
  if (bc.pS != PERIODIC && bc.pN != PERIODIC)  {
    if (dom[rank].J == DOM.Js) {
      flag_external_v<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(_flag_v,
        dom[rank].Gfy._js);
    }
    if (dom[rank].J == DOM.Je) {
      flag_external_v<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(_flag_v,
        dom[rank].Gfy._je);
    }
  }

  // k direction
  if (bc.pB != PERIODIC && bc.pT != PERIODIC) {
    if (dom[rank].K == DOM.Ks) {
      flag_external_w<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(_flag_w,
        dom[rank].Gfz._ks);
    }
    if (dom[rank].K == DOM.Ke) {
      flag_external_w<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(_flag_w,
        dom[rank].Gfz._ke);
    }
  }

  /* Fill in flag_{u,v,w} ghost cells for periodic boundary conditions -- only
      necessary with particles bc of cage */
    // Do this exactly like we do ghost cell exchanges -- since dom[rank].e will
    // be MPI_PROC_NULL if need be, we don't need to worry about exchanging over
    // periodic boundaries
}

extern "C"
void cuda_part_BC(void)
{
  //printf("N%d >> Applying particle boundary conditions to u...\n", rank);
  // u
  part_BC_u<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(_u, _phase, _flag_u,
    _parts, nu, nparts);
  // v
  part_BC_v<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(_v, _phase, _flag_v,
    _parts, nu, nparts);
  // w
  part_BC_w<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(_w, _phase, _flag_w,
    _parts, nu, nparts);
}

extern "C"
void cuda_part_BC_star(void)
{
  // u
  part_BC_u<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(_u_star, _phase,
    _flag_u, _parts, nu, nparts);

  // v
  part_BC_v<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(_v_star, _phase,
    _flag_v, _parts, nu, nparts);

  // w
  part_BC_w<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(_w_star, _phase,
    _flag_w, _parts, nu, nparts);
}

extern "C"
void cuda_part_BC_p(void)
{
  part_BC_p<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p0, _rhs_p, _phase,
    _phase_shell, _parts, mu, nu, dt, dt0, gradP, rho_f, nparts, s_beta, s_ref, g);
}

extern "C"
void cuda_part_p_fill(void)
{
    part_BC_p_fill<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p, _phase, _parts,
      mu, nu, rho_f, gradP, nparts, s_beta, s_ref, g);
}

extern "C"
void cuda_parts_internal(void)
{
  if (nparts > 0) {
    internal_u<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u, _parts, _flag_u,
      _phase, nparts);
    internal_v<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v, _parts, _flag_v,
      _phase, nparts);
    internal_w<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w, _parts, _flag_w,
      _phase, nparts);
  }
}

extern "C"
void cuda_update_part_forces_i(void)
{
  /* Outline of communication
   * The following need to be communicated before move_parts_{a,b}
   *  * kFx, kFy, kFz -- subdom + ghost, but same 
   *  * iFx, iFy, iFz -- subdom
   *  * iLx, iLy, iLz -- subdom
   *  * iSt, St       -- subdom
   * This communication is similar to the communication of partial sums during
   *  the Lebedev quadrature (see cuda_physalis.cu:cuda_partial_sum_i)
   * 1) All particles in the outer computational bin plane need to be sent,
   *    for example the (j,k) planes at _bins.Gcc.{_is, _ie}.
   * 2) Bin the particles using i indexing to find _bin_{start,end,count}
   * 3) Reduce _bin_count at _is, _ie to find nparts_send_{e,w}
   * 4) Communicate nparts_send_{e,w} with appropriate subdom to find
   *    nparts_recv_{e,w}
   * 5) Excl. prefix scan bin_count over _is, _ie to find destination index for
   *    packed particle data
   * 6) Allocate send and recv array
   * 7) Pack send array using destination offsetes
   * 8) Communicate send->recv
   * 9) Excl. prefix over _isb, _ieb to find unpacking indices
   * 10) Unpack
   * 11) Repeat for j, k
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
  int s1b = bins.Gcc.jnb; // custom strides
  int s2b = s1b * bins.Gcc.knb;
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_e;
  int *_offset_w;
  checkCudaErrors(cudaMalloc(&_offset_e, bins.Gcc.s2b_i * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_w, bins.Gcc.s2b_i * sizeof(int)));
  thrust::device_ptr<int> t_offset_e(_offset_e);
  thrust::device_ptr<int> t_offset_w(_offset_w);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
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

    // East: _ie, _ieb
    if (dom[rank].e != MPI_PROC_NULL) {
      // _bin_count is indexed with i varying slowest
      // Do reduction over bin_count, given correct starting offset of _ie plane
      offset = GFX_LOC(bins.Gcc._ie, 0, 0, s1b, s2b);
      nparts_send[EAST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_send[EAST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_i, t_offset_e);
      } else {
        cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
      }

      /* Also determine number of parts to recv */
      offset = GFX_LOC(bins.Gcc._ieb, 0, 0, s1b, s2b);
      nparts_recv[EAST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());

    } else { // no parts to send or recv
      nparts_send[EAST] = 0;
      nparts_recv[EAST] = 0;
      cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
    }

    // West: _is, _isb
    if (dom[rank].w != MPI_PROC_NULL) {
      // nparts_send
      offset = GFX_LOC(bins.Gcc._is, 0, 0, s1b, s2b);
      nparts_send[WEST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());

      // send offsets
      if (nparts_send[WEST] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_i, t_offset_w);
      } else {
        cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
      }

      // nparts_recv
      offset = GFX_LOC(bins.Gcc._isb, 0, 0, s1b, s2b);
      nparts_recv[WEST] = thrust::reduce(t_bin_count + offset,
                                         t_bin_count + offset + bins.Gcc.s2b_i,
                                         0., thrust::plus<int>());
    } else {
      nparts_send[WEST] = 0;
      nparts_recv[WEST] = 0;
      cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
    }
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[EAST] = 0;
    nparts_send[WEST] = 0;
    nparts_recv[EAST] = 0;
    nparts_recv[WEST] = 0;
    cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
    cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
  }

  /* Send number of parts to east/west */
  //    origin                target
  // nparts_send[WEST] -> nparts_recv[EAST]
  // nparts_recv[WEST] <- nparts_send[EAST]
  //nparts_recv[WEST] = 0; // init
  //nparts_recv[EAST] = 0;
  //mpi_send_nparts_i();

  /* Allocate memory for send and recv forces */
  int n_send = 9;
  // * kFx, kFy, kFz
  // * iFx, iFy, iFz
  // * iLx, iLy, iLz
  // Indexing is, for example:
  //  _force_send_e[force + 9*part_id]
  // where
  //  part_id = [0, nparts) and force = [0, 9)
  //    0: kFx  1: kFy  2: kFz
  //    3: iFx  4: iFy  5: iFz
  //    6: iLx  7: iLy  8: iLz

  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_e = nparts_send[EAST]*(nparts_send[EAST] > 0) + (nparts_send[EAST] == 0);
  int send_alloc_w = nparts_send[WEST]*(nparts_send[WEST] > 0) + (nparts_send[WEST] == 0);
  int recv_alloc_e = nparts_recv[EAST]*(nparts_recv[EAST] > 0) + (nparts_recv[EAST] == 0);
  int recv_alloc_w = nparts_recv[WEST]*(nparts_recv[WEST] > 0) + (nparts_recv[WEST] == 0);

  checkCudaErrors(cudaMalloc(&_force_send_e, send_alloc_e*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_send_w, send_alloc_w*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_recv_e, recv_alloc_e*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_recv_w, recv_alloc_w*n_send*sizeof(real)));

  /* Pack partial forces */
  if (nparts_send[EAST] > 0) {
    pack_forces_e<<<bin_num_inb, bin_dim_inb>>>(_force_send_e, _offset_e,
      _bin_start, _bin_count, _part_ind, _parts);
  } else {  // fill dummy data
    //cudaMemset(_force_send_e, 0., send_alloc_e * n_send * sizeof(real));
  }

  if (nparts_send[WEST] > 0) {
    pack_forces_w<<<bin_num_inb, bin_dim_inb>>>(_force_send_w, _offset_w,
      _bin_start, _bin_count, _part_ind, _parts);
  } else {  // fill dummy data
    //cudaMemset(_force_send_w, 0., send_alloc_w * n_send * sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate forces with MPI */
  mpi_send_forces_i();

  /* Find offsets in ghost bins */
  if (nparts > 0) {
    // East: _ieb
    if (dom[rank].e != MPI_PROC_NULL) {

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_recv[EAST] > 0) {
        offset = GFX_LOC(bins.Gcc._ieb, 0, 0, s1b, s2b);
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_i, t_offset_e);
      } else {
        cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
      }

    } else {
      cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
    }

    // West: _isb plane
    if (dom[rank].w != MPI_PROC_NULL) {

      if (nparts_recv[WEST] > 0) {
        offset = GFX_LOC(bins.Gcc._isb, 0, 0, s1b, s2b);
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_i, t_offset_w);
      } else {
        cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
      }

    } else {
      cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
    }

    /* Unpack and complete forces */
    if (nparts_recv[EAST] > 0) {
      unpack_forces_e<<<bin_num_inb, bin_dim_inb>>>(_force_recv_e, _offset_e,
        _bin_start, _bin_count, _part_ind, _parts);
    }
    if (nparts_recv[WEST] > 0) {
      unpack_forces_w<<<bin_num_inb, bin_dim_inb>>>(_force_recv_w, _offset_w,
        _bin_start, _bin_count, _part_ind, _parts);
    }
    cudaDeviceSynchronize();  // ensure packing is complete

  } else { // nparts <= 0
    cudaMemset(_offset_e, 0., bins.Gcc.s2b_i * sizeof(int));
    cudaMemset(_offset_w, 0., bins.Gcc.s2b_i * sizeof(int));
  }

  /* Free */
  cudaFree(_force_send_e);
  cudaFree(_force_send_w);
  cudaFree(_force_recv_e);
  cudaFree(_force_recv_w);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_e);
  cudaFree(_offset_w);
}

extern "C"
void cuda_update_part_forces_j(void)
{
  //printf("N%d >> Updating particle forces in j... (nparts %d)\n", rank, nparts);
  /* Communication follows same pattern as cuda_update_part_forces_i */

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
  int s1b = bins.Gcc.knb;
  int s2b = s1b * bins.Gcc.inb;
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_n;
  int *_offset_s;
  checkCudaErrors(cudaMalloc(&_offset_n, bins.Gcc.s2b_j * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_s, bins.Gcc.s2b_j * sizeof(int)));
  thrust::device_ptr<int> t_offset_n(_offset_n);
  thrust::device_ptr<int> t_offset_s(_offset_s);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
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

    // north: _je 
    if (dom[rank].n != MPI_PROC_NULL) {
      // _bin_count is indexed with j varying slowest
      // nparts_send
      offset = GFY_LOC(0, bins.Gcc._je, 0, s1b, s2b);
      nparts_send[NORTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());

      // send offsets
      if (nparts_send[NORTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_j, t_offset_n);
      } else {
        cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
      }

      // nparts_recv
      offset = GFY_LOC(0, bins.Gcc._jeb, 0, s1b, s2b);
      nparts_recv[NORTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());

    } else { // no parts to send
      nparts_send[NORTH] = 0;
      nparts_recv[NORTH] = 0;
      cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
    }

    // SOUTH: _js planes
    if (dom[rank].s != MPI_PROC_NULL) {
      // nparts_send
      offset = GFY_LOC(0, bins.Gcc._js, 0, s1b, s2b);
      nparts_send[SOUTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());
      // sending offsets
      if (nparts_send[SOUTH] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_j, t_offset_s);
      } else {
        cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
      }

      // nparts_recv
      offset = GFY_LOC(0, bins.Gcc._jsb, 0, s1b, s2b);
      nparts_recv[SOUTH] = thrust::reduce(t_bin_count + offset,
                                          t_bin_count + offset + bins.Gcc.s2b_j,
                                          0., thrust::plus<int>());
    } else {
      nparts_send[SOUTH] = 0;
      nparts_recv[SOUTH] = 0;
      cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
    }
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[NORTH] = 0;
    nparts_send[SOUTH] = 0;
    nparts_recv[NORTH] = 0;
    nparts_recv[SOUTH] = 0;
    cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
    cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
  }

  /* Send number of parts to NORTH/SOUTH */
  //    origin                target
  // nparts_send[SOUTH] -> nparts_recv[NORTH]
  // nparts_recv[SOUTH] <- nparts_send[NORTH]
  //nparts_recv[SOUTH] = 0; // init
  //nparts_recv[NORTH] = 0;
  //mpi_send_nparts_j();

  /* Allocate memory for send and recv forces */
  int n_send = 9;
  // * kFx, kFy, kFz
  // * iFx, iFy, iFz
  // * iLx, iLy, iLz
  // Indexing is, for example:
  //  _force_send_n[force + 9*part_id]
  // where
  //  part_id = [0, nparts) and force = [0, 9)
  //    0: kFx  1: kFy  2: kFz
  //    3: iFx  4: iFy  5: iFz
  //    6: iLx  7: iLy  8: iLz

  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_n = nparts_send[NORTH]*(nparts_send[NORTH] > 0) + (nparts_send[NORTH] == 0);
  int send_alloc_s = nparts_send[SOUTH]*(nparts_send[SOUTH] > 0) + (nparts_send[SOUTH] == 0);
  int recv_alloc_n = nparts_recv[NORTH]*(nparts_recv[NORTH] > 0) + (nparts_recv[NORTH] == 0);
  int recv_alloc_s = nparts_recv[SOUTH]*(nparts_recv[SOUTH] > 0) + (nparts_recv[SOUTH] == 0);

  checkCudaErrors(cudaMalloc(&_force_send_n, send_alloc_n*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_send_s, send_alloc_s*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_recv_n, recv_alloc_n*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_recv_s, recv_alloc_s*n_send*sizeof(real)));

  /* Pack partial forces */
  if (nparts_send[NORTH] > 0) {
    pack_forces_n<<<bin_num_jnb, bin_dim_jnb>>>(_force_send_n, _offset_n,
      _bin_start, _bin_count, _part_ind, _parts);
  } else {
    cudaMemset(_force_send_n, 0., send_alloc_n*n_send*sizeof(real));
  }

  if (nparts_send[SOUTH] > 0) {
    pack_forces_s<<<bin_num_jnb, bin_dim_jnb>>>(_force_send_s, _offset_s,
      _bin_start, _bin_count, _part_ind, _parts);
  } else {
    cudaMemset(_force_send_s, 0., send_alloc_s*n_send*sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate forces with MPI */
  mpi_send_forces_j();

  /* Find offsets in ghost bins */
  if (nparts > 0) {

    // NORTH: _jeb
    if (dom[rank].n != MPI_PROC_NULL) {

      /* Determine packing offsets with an excl prefix scan */
      if (nparts_recv[NORTH] > 0) {
        offset = GFY_LOC(0, bins.Gcc._jeb, 0, s1b, s2b);
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_j, t_offset_n);
      } else {
        cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
      }

    } else { // no parts to send
      cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
    }

    // SOUTH: _jsb plane
    if (dom[rank].s != MPI_PROC_NULL) {

      if (nparts_recv[SOUTH] > 0) {
        offset = GFY_LOC(0, bins.Gcc._jsb, 0, s1b, s2b);
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_j, t_offset_s);
      } else {
        cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
      }

    } else {
      cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
    }

    /* Unpack and complete forces */
    if (nparts_recv[NORTH] > 0) {
      unpack_forces_n<<<bin_num_jnb, bin_dim_jnb>>>(_force_recv_n, _offset_n,
        _bin_start, _bin_count, _part_ind, _parts);
    }
    if (nparts_recv[SOUTH] > 0) {
      unpack_forces_s<<<bin_num_jnb, bin_dim_jnb>>>(_force_recv_s, _offset_s,
        _bin_start, _bin_count, _part_ind, _parts);
    }
    cudaDeviceSynchronize();  // ensure packing is complete

  } else { // nparts <= 0
    cudaMemset(_offset_n, 0., bins.Gcc.s2b_j * sizeof(int));
    cudaMemset(_offset_s, 0., bins.Gcc.s2b_j * sizeof(int));
  }

  /* Free */
  cudaFree(_force_send_n);
  cudaFree(_force_send_s);
  cudaFree(_force_recv_n);
  cudaFree(_force_recv_s);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_n);
  cudaFree(_offset_s);
}

extern "C"
void cuda_update_part_forces_k(void)
{
  //printf("N%d >> Updating particle forces in k... (nparts %d)\n", rank, nparts);
  /* Communication follows same pattern as cuda_update_part_forces_i */

  /* Initialize execution config */
  // thread over top/bottom faces 
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
  int s1b = bins.Gcc.inb;
  int s2b = s1b * bins.Gcc.jnb;
  int offset;

  /* Allocate */
  checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
  thrust::device_ptr<int> t_part_ind(_part_ind);
  thrust::device_ptr<int> t_part_bin(_part_bin);

  int *_offset_t;
  int *_offset_b;
  checkCudaErrors(cudaMalloc(&_offset_t, bins.Gcc.s2b_k * sizeof(int)));
  checkCudaErrors(cudaMalloc(&_offset_b, bins.Gcc.s2b_k * sizeof(int)));
  thrust::device_ptr<int> t_offset_t(_offset_t);
  thrust::device_ptr<int> t_offset_b(_offset_b);

  checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
  checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
  thrust::device_ptr<int> t_bin_count(_bin_count);

  // If we have parts...
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

    // TOP: _ke 
    if (dom[rank].t != MPI_PROC_NULL) {
      // _bin_count is indexed with k varying slowest
      // nparts_send
      offset = GFZ_LOC(0, 0, bins.Gcc._ke, s1b, s2b);
      nparts_send[TOP] = thrust::reduce(t_bin_count + offset,
                                        t_bin_count + offset + bins.Gcc.s2b_k,
                                        0., thrust::plus<int>());

      // sending offsets
      if (nparts_send[TOP] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_k, t_offset_t);
      } else {
        cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
      }

      // nparts_recv
      offset = GFZ_LOC(0, 0, bins.Gcc._keb, s1b, s2b);
      nparts_recv[TOP] = thrust::reduce(t_bin_count + offset,
                                        t_bin_count + offset + bins.Gcc.s2b_k,
                                        0., thrust::plus<int>());

    } else { // no parts to send
      nparts_send[TOP] = 0;
      nparts_recv[TOP] = 0;
      cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
    }

    // BOTTOM: _ks planes
    if (dom[rank].b != MPI_PROC_NULL) {
      // nparts_send
      offset = GFZ_LOC(0, 0, bins.Gcc._ks, s1b, s2b);
      nparts_send[BOTTOM] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + bins.Gcc.s2b_k,
                                           0., thrust::plus<int>());
      // sending offsets
      if (nparts_send[BOTTOM] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_k, t_offset_b);
      } else {
        cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
      }

      // nparts_recv
      offset = GFZ_LOC(0, 0, bins.Gcc._ksb, s1b, s2b);
      nparts_recv[BOTTOM] = thrust::reduce(t_bin_count + offset,
                                           t_bin_count + offset + bins.Gcc.s2b_k,
                                           0., thrust::plus<int>());
    } else {
      nparts_send[BOTTOM] = 0;
      nparts_recv[BOTTOM] = 0;
      cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
    }
  } else { // nparts <= 0
    checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
    checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
    nparts_send[TOP] = 0;
    nparts_send[BOTTOM] = 0;
    nparts_recv[TOP] = 0;
    nparts_recv[BOTTOM] = 0;
    cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
    cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
  }

  /* Send number of parts to TOP/BOTTOM */
  //    origin                target
  // nparts_send[BOTTOM] -> nparts_recv[TOP]
  // nparts_recv[BOTTOM] <- nparts_send[TOP]
  //nparts_recv[BOTTOM] = 0; // init
  //nparts_recv[TOP] = 0;
  //mpi_send_nparts_k();

  /* Allocate memory for send and recv forces */
  int n_send = 9;
  // * kFx, kFy, kFz
  // * iFx, iFy, iFz
  // * iLx, iLy, iLz
  // Indexing is, for example:
  //  _force_send_t[force + 9*part_id]
  // where
  //  part_id = [0, nparts) and force = [0, 9)
  //    0: kFx  1: kFy  2: kFz
  //    3: iFx  4: iFy  5: iFz
  //    6: iLx  7: iLy  8: iLz

  // See accompanying note at the same location in cuda_transfer_parts_i
  int send_alloc_t = nparts_send[TOP]*(nparts_send[TOP] > 0) + (nparts_send[TOP] == 0);
  int send_alloc_b = nparts_send[BOTTOM]*(nparts_send[BOTTOM] > 0) + (nparts_send[BOTTOM] == 0);
  int recv_alloc_t = nparts_recv[TOP]*(nparts_recv[TOP] > 0) + (nparts_recv[TOP] == 0);
  int recv_alloc_b = nparts_recv[BOTTOM]*(nparts_recv[BOTTOM] > 0) + (nparts_recv[BOTTOM] == 0);

  checkCudaErrors(cudaMalloc(&_force_send_t, send_alloc_t*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_send_b, send_alloc_b*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_recv_t, recv_alloc_t*n_send*sizeof(real)));
  checkCudaErrors(cudaMalloc(&_force_recv_b, recv_alloc_b*n_send*sizeof(real)));

  /* Pack partial forces */
  if (nparts_send[TOP] > 0) {
    pack_forces_t<<<bin_num_knb, bin_dim_knb>>>(_force_send_t, _offset_t,
      _bin_start, _bin_count, _part_ind, _parts);
  } else { // fill dummy data
    //cudaMemset(_force_send_t, 0., send_alloc_t * n_send * sizeof(real));
  }


  if (nparts_send[BOTTOM] > 0) {
    pack_forces_b<<<bin_num_knb, bin_dim_knb>>>(_force_send_b, _offset_b,
      _bin_start, _bin_count, _part_ind, _parts);
  } else { // fill dummy data
    //cudaMemset(_force_send_b, 0., send_alloc_b * n_send * sizeof(real));
  }
  cudaDeviceSynchronize();  // ensure packing is complete

  /* Communicate forces with MPI */
  mpi_send_forces_k();

  if (nparts > 0) {
    /* Find offsets in ghost bins */
    // TOP: _keb
    if (dom[rank].t != MPI_PROC_NULL) {

      offset = GFZ_LOC(0, 0, bins.Gcc._keb, s1b, s2b);
      /* Determine packing offsets with an excl prefix scan */
      if (nparts_recv[TOP] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_k, t_offset_t);
      } else {
        cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
      }

    } else { // no parts to send
      cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
    }

    // BOTTOM: _ksb plane
    if (dom[rank].b != MPI_PROC_NULL) {
      offset = GFZ_LOC(0, 0, bins.Gcc._ksb, s1b, s2b);
      if (nparts_recv[BOTTOM] > 0) {
        thrust::exclusive_scan(t_bin_count + offset,
                               t_bin_count + offset + bins.Gcc.s2b_k, t_offset_b);
      } else {
        cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
      }

    } else {
      cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
    }

    /* Unpack and complete forces */
    if (nparts_recv[TOP] > 0) {
      unpack_forces_t<<<bin_num_knb, bin_dim_knb>>>(_force_recv_t, _offset_t,
        _bin_start, _bin_count, _part_ind, _parts);
    }
    if (nparts_recv[BOTTOM] > 0) {
      unpack_forces_b<<<bin_num_knb, bin_dim_knb>>>(_force_recv_b, _offset_b,
        _bin_start, _bin_count, _part_ind, _parts);
    }
    cudaDeviceSynchronize();  // ensure packing is complete
  } else {
    cudaMemset(_offset_t, 0., bins.Gcc.s2b_k * sizeof(int));
    cudaMemset(_offset_b, 0., bins.Gcc.s2b_k * sizeof(int));
  }

  /* Free */
  cudaFree(_force_send_t);
  cudaFree(_force_send_b);
  cudaFree(_force_recv_t);
  cudaFree(_force_recv_b);
  cudaFree(_part_ind);
  cudaFree(_part_bin);
  cudaFree(_offset_t);
  cudaFree(_offset_b);
}
