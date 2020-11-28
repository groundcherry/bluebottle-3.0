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
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h> // FOR DEBUG!!
#include <helper_cuda.h>

#include "cuda_bluebottle.h"
#include "cuda_particle.h"

__constant__ dom_struct _dom;
cuda_blocks_struct blocks;

extern "C"
void cuda_check_errors(int line)
{
  printf("N%d >> Checking errors on line %d\n", rank, line);
  checkCudaErrors(cudaDeviceSynchronize());
}

extern "C"
int cuda_device_count(void)
{
  // Get number of cuda devices
  int dev_count = 0;
  checkCudaErrors(cudaGetDeviceCount(&dev_count));

  return dev_count;
}

extern "C"
void cuda_device_init(int device)
{
   checkCudaErrors(cudaSetDevice(device));
}

extern "C"
void cuda_enable_peer(void)
{
  int target_peer = (rank + 1) % nprocs;
  printf("Enabling peer access from %d to %d\n", rank, target_peer);
  checkCudaErrors(cudaDeviceEnablePeerAccess(target_peer, 0));
}

extern "C"
void cuda_block(void)
{
  cudaDeviceSynchronize();
}


extern "C"
void cuda_dom_malloc_host(void)
{
  //printf("N%d >> Allocating pinned host memory... \n", rank);
  // Allocate (pinned) device memory on host
  checkCudaErrors(cudaMallocHost(&p, dom[rank].Gcc.s3b * sizeof(real))); 
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&p0, dom[rank].Gcc.s3b * sizeof(real)));
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&phi, dom[rank].Gcc.s3b * sizeof(real))); 
    cpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&u, dom[rank].Gfx.s3b * sizeof(real)));
    cpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&v, dom[rank].Gfy.s3b * sizeof(real)));
    cpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&w, dom[rank].Gfz.s3b * sizeof(real)));
    cpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&u0, dom[rank].Gfx.s3b * sizeof(real)));
    cpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&v0, dom[rank].Gfy.s3b * sizeof(real)));
    cpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&w0, dom[rank].Gfz.s3b * sizeof(real)));
    cpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&conv_u, dom[rank].Gfx.s3b * sizeof(real)));
    cpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&conv_v, dom[rank].Gfy.s3b * sizeof(real)));
    cpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&conv_w, dom[rank].Gfz.s3b * sizeof(real)));
    cpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&conv0_u, dom[rank].Gfx.s3b * sizeof(real)));
    cpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&conv0_v, dom[rank].Gfy.s3b * sizeof(real)));
    cpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&conv0_w, dom[rank].Gfz.s3b * sizeof(real)));
    cpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&diff_u, dom[rank].Gfx.s3b * sizeof(real)));
    cpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&diff_v, dom[rank].Gfy.s3b * sizeof(real)));
    cpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&diff_w, dom[rank].Gfz.s3b * sizeof(real)));
    cpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&diff0_u, dom[rank].Gfx.s3b * sizeof(real)));
    cpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&diff0_v, dom[rank].Gfy.s3b * sizeof(real)));
    cpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&diff0_w, dom[rank].Gfz.s3b * sizeof(real)));
    cpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&f_x, dom[rank].Gfx.s3b * sizeof(real)));
    cpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&f_y, dom[rank].Gfy.s3b * sizeof(real)));
    cpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&f_z, dom[rank].Gfz.s3b * sizeof(real)));
    cpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&u_star, dom[rank].Gfx.s3b * sizeof(real)));
    cpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&v_star, dom[rank].Gfy.s3b * sizeof(real)));
    cpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMallocHost(&w_star, dom[rank].Gfz.s3b * sizeof(real)));
    cpumem += dom[rank].Gfz.s3b * sizeof(real);

  checkCudaErrors(cudaMallocHost(&flag_u, dom[rank].Gfx.s3b * sizeof(int)));
    cpumem += dom[rank].Gfx.s3b * sizeof(int);
  checkCudaErrors(cudaMallocHost(&flag_v, dom[rank].Gfy.s3b * sizeof(int)));
    cpumem += dom[rank].Gfy.s3b * sizeof(int);
  checkCudaErrors(cudaMallocHost(&flag_w, dom[rank].Gfz.s3b * sizeof(int)));
    cpumem += dom[rank].Gfz.s3b * sizeof(int);
}

extern "C"
void cuda_dom_malloc_dev(void)
{
  // Allocate device memory on device
  // Don't need to free device constant memory
  checkCudaErrors(cudaMemcpyToSymbol(_dom, &dom[rank], sizeof(dom_struct)));

  checkCudaErrors(cudaMalloc((void**) &_DOM, sizeof(dom_struct)));
    gpumem += sizeof(dom_struct);
  checkCudaErrors(cudaMemcpy(_DOM, &DOM, sizeof(dom_struct), 
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void**) &_bc, sizeof(BC)));
    gpumem += sizeof(BC);
  checkCudaErrors(cudaMemcpy(_bc, &bc, sizeof(BC), 
    cudaMemcpyHostToDevice));


  /* Flow solver variables */
  checkCudaErrors(cudaMalloc(&_phi, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_phinoghost, dom[rank].Gcc.s3 * sizeof(real)));
    gpumem += dom[rank].Gcc.s3 * sizeof(real);
  checkCudaErrors(cudaMalloc(&_invM, dom[rank].Gcc.s3 * sizeof(real)));
    gpumem += dom[rank].Gcc.s3 * sizeof(real);
  checkCudaErrors(cudaMalloc(&_p, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_p0, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_u, dom[rank].Gfx.s3b * sizeof(real)));
    gpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_v, dom[rank].Gfy.s3b * sizeof(real)));
    gpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_w, dom[rank].Gfz.s3b * sizeof(real)));
    gpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_u0, dom[rank].Gfx.s3b * sizeof(real)));
    gpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_v0, dom[rank].Gfy.s3b * sizeof(real)));
    gpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_w0, dom[rank].Gfz.s3b * sizeof(real)));
    gpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_conv_u, dom[rank].Gfx.s3b * sizeof(real)));
    gpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_conv_v, dom[rank].Gfy.s3b * sizeof(real)));
    gpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_conv_w, dom[rank].Gfz.s3b * sizeof(real)));
    gpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_conv0_u, dom[rank].Gfx.s3b * sizeof(real)));
    gpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_conv0_v, dom[rank].Gfy.s3b * sizeof(real)));
    gpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_conv0_w, dom[rank].Gfz.s3b * sizeof(real)));
    gpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_diff_u, dom[rank].Gfx.s3b * sizeof(real)));
    gpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_diff_v, dom[rank].Gfy.s3b * sizeof(real)));
    gpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_diff_w, dom[rank].Gfz.s3b * sizeof(real)));
    gpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_diff0_u, dom[rank].Gfx.s3b * sizeof(real)));
    gpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_diff0_v, dom[rank].Gfy.s3b * sizeof(real)));
    gpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_diff0_w, dom[rank].Gfz.s3b * sizeof(real)));
    gpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_f_x, dom[rank].Gfx.s3b * sizeof(real)));
    gpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_f_y, dom[rank].Gfy.s3b * sizeof(real)));
    gpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_f_z, dom[rank].Gfz.s3b * sizeof(real)));
    gpumem += dom[rank].Gfz.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_u_star, dom[rank].Gfx.s3b * sizeof(real)));
    gpumem += dom[rank].Gfx.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_v_star, dom[rank].Gfy.s3b * sizeof(real)));
    gpumem += dom[rank].Gfy.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_w_star, dom[rank].Gfz.s3b * sizeof(real)));
    gpumem += dom[rank].Gfz.s3b * sizeof(real);

  // Flags
  checkCudaErrors(cudaMalloc(&_flag_u, dom[rank].Gfx.s3b * sizeof(int)));
    gpumem += dom[rank].Gfx.s3b * sizeof(int);
  checkCudaErrors(cudaMalloc(&_flag_v, dom[rank].Gfy.s3b * sizeof(int)));
    gpumem += dom[rank].Gfy.s3b * sizeof(int);
  checkCudaErrors(cudaMalloc(&_flag_w, dom[rank].Gfz.s3b * sizeof(int)));
    gpumem += dom[rank].Gfz.s3b * sizeof(int);

  /* Poisson Equation Variables */
  checkCudaErrors(cudaMalloc(&_r_q, dom[rank].Gcc.s3 * sizeof(real)));
    gpumem += dom[rank].Gcc.s3 * sizeof(real);
  checkCudaErrors(cudaMalloc(&_z_q, dom[rank].Gcc.s3 * sizeof(real)));
    gpumem += dom[rank].Gcc.s3 * sizeof(real);
  //checkCudaErrors(cudaMalloc(&_rs_0, dom[rank].Gcc.s3 * sizeof(real)));
  //  gpumem += dom[rank].Gcc.s3 * sizeof(real);
  checkCudaErrors(cudaMalloc(&_p_q, dom[rank].Gcc.s3 * sizeof(real)));
    gpumem += dom[rank].Gcc.s3 * sizeof(real);
  //checkCudaErrors(cudaMalloc(&_s_q, dom[rank].Gcc.s3 * sizeof(real)));
  //  gpumem += dom[rank].Gcc.s3 * sizeof(real);
  checkCudaErrors(cudaMalloc(&_Apb_q, dom[rank].Gcc.s3 * sizeof(real)));
    gpumem += dom[rank].Gcc.s3 * sizeof(real);
  //checkCudaErrors(cudaMalloc(&_Asb_q, dom[rank].Gcc.s3 * sizeof(real)));
  //  gpumem += dom[rank].Gcc.s3 * sizeof(real);

  // These are s3b because the SpMv requires more info
  checkCudaErrors(cudaMalloc(&_rhs_p, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  checkCudaErrors(cudaMalloc(&_pb_q, dom[rank].Gcc.s3b * sizeof(real)));
    gpumem += dom[rank].Gcc.s3b * sizeof(real);
  //checkCudaErrors(cudaMalloc(&_sb_q, dom[rank].Gcc.s3b * sizeof(real)));
  //  gpumem += dom[rank].Gcc.s3b * sizeof(real);

  /* Subdomain communication variables */
  // Outer computational planes 
  checkCudaErrors(cudaMalloc(&_send_Gcc_e, dom[rank].Gcc.s2_i * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gcc_w, dom[rank].Gcc.s2_i * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gcc_n, dom[rank].Gcc.s2_j * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gcc_s, dom[rank].Gcc.s2_j * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gcc_t, dom[rank].Gcc.s2_k * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_k * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gcc_b, dom[rank].Gcc.s2_k * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_k * sizeof(real);

  checkCudaErrors(cudaMalloc(&_send_Gfx_e, dom[rank].Gfx.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfx_w, dom[rank].Gfx.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfx_n, dom[rank].Gfx.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfx_s, dom[rank].Gfx.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfx_t, dom[rank].Gfx.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_k * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfx_b, dom[rank].Gfx.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_k * sizeof(real);

  checkCudaErrors(cudaMalloc(&_send_Gfy_e, dom[rank].Gfy.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfy_w, dom[rank].Gfy.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfy_n, dom[rank].Gfy.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfy_s, dom[rank].Gfy.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfy_t, dom[rank].Gfy.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_k * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfy_b, dom[rank].Gfy.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_k * sizeof(real);

  checkCudaErrors(cudaMalloc(&_send_Gfz_e, dom[rank].Gfz.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfz_w, dom[rank].Gfz.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfz_n, dom[rank].Gfz.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfz_s, dom[rank].Gfz.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfz_t, dom[rank].Gfz.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_k * sizeof(real);
  checkCudaErrors(cudaMalloc(&_send_Gfz_b, dom[rank].Gfz.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_k * sizeof(real);

  // Ghost cell planes
  checkCudaErrors(cudaMalloc(&_recv_Gcc_e, dom[rank].Gcc.s2_i * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gcc_w, dom[rank].Gcc.s2_i * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gcc_n, dom[rank].Gcc.s2_j * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gcc_s, dom[rank].Gcc.s2_j * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gcc_t, dom[rank].Gcc.s2_k * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_k * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gcc_b, dom[rank].Gcc.s2_k * sizeof(real)));
    gpumem += dom[rank].Gcc.s2_k * sizeof(real);

  checkCudaErrors(cudaMalloc(&_recv_Gfx_e, dom[rank].Gfx.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfx_w, dom[rank].Gfx.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfx_n, dom[rank].Gfx.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfx_s, dom[rank].Gfx.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfx_t, dom[rank].Gfx.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_k * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfx_b, dom[rank].Gfx.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfx.s2_k * sizeof(real);

  checkCudaErrors(cudaMalloc(&_recv_Gfy_e, dom[rank].Gfy.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfy_w, dom[rank].Gfy.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfy_n, dom[rank].Gfy.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfy_s, dom[rank].Gfy.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfy_t, dom[rank].Gfy.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_k * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfy_b, dom[rank].Gfy.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfy.s2_k * sizeof(real);

  checkCudaErrors(cudaMalloc(&_recv_Gfz_e, dom[rank].Gfz.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfz_w, dom[rank].Gfz.s2_i * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_i * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfz_n, dom[rank].Gfz.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfz_s, dom[rank].Gfz.s2_j * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_j * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfz_t, dom[rank].Gfz.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_k * sizeof(real);
  checkCudaErrors(cudaMalloc(&_recv_Gfz_b, dom[rank].Gfz.s2_k * sizeof(real)));
    gpumem += dom[rank].Gfz.s2_k * sizeof(real);

  // Init things that we will need
  checkCudaErrors(cudaMemset(_u, 0., dom[rank].Gfx.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_v, 0., dom[rank].Gfy.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_w, 0., dom[rank].Gfz.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_p, 0., dom[rank].Gcc.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_u0, 0., dom[rank].Gfx.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_v0, 0., dom[rank].Gfy.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_w0, 0., dom[rank].Gfz.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_p0, 0., dom[rank].Gcc.s3b * sizeof(real)));

  checkCudaErrors(cudaMemset(_phi, 0., dom[rank].Gcc.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_rhs_p, 0., dom[rank].Gcc.s3b * sizeof(real)));
  checkCudaErrors(cudaMemset(_p_q, 0., dom[rank].Gcc.s3 * sizeof(real)));
  checkCudaErrors(cudaMemset(_pb_q, 0., dom[rank].Gcc.s3b * sizeof(real)));
  //checkCudaErrors(cudaMemset(_s_q, 0., dom[rank].Gcc.s3 * sizeof(real)));
  //checkCudaErrors(cudaMemset(_sb_q, 0., dom[rank].Gcc.s3b * sizeof(real)));

  checkCudaErrors(cudaMemset(_send_Gcc_e, 0., dom[rank].Gcc.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gcc_w, 0., dom[rank].Gcc.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gcc_n, 0., dom[rank].Gcc.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gcc_s, 0., dom[rank].Gcc.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gcc_t, 0., dom[rank].Gcc.s2_k * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gcc_b, 0., dom[rank].Gcc.s2_k * sizeof(real)));

  checkCudaErrors(cudaMemset(_send_Gfx_e, 0., dom[rank].Gfx.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfx_w, 0., dom[rank].Gfx.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfx_n, 0., dom[rank].Gfx.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfx_s, 0., dom[rank].Gfx.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfx_t, 0., dom[rank].Gfx.s2_k * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfx_b, 0., dom[rank].Gfx.s2_k * sizeof(real)));

  checkCudaErrors(cudaMemset(_send_Gfy_e, 0., dom[rank].Gfy.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfy_w, 0., dom[rank].Gfy.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfy_n, 0., dom[rank].Gfy.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfy_s, 0., dom[rank].Gfy.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfy_t, 0., dom[rank].Gfy.s2_k * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfy_b, 0., dom[rank].Gfy.s2_k * sizeof(real)));

  checkCudaErrors(cudaMemset(_send_Gfz_e, 0., dom[rank].Gfz.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfz_w, 0., dom[rank].Gfz.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfz_n, 0., dom[rank].Gfz.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfz_s, 0., dom[rank].Gfz.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfz_t, 0., dom[rank].Gfz.s2_k * sizeof(real)));
  checkCudaErrors(cudaMemset(_send_Gfz_b, 0., dom[rank].Gfz.s2_k * sizeof(real)));

  checkCudaErrors(cudaMemset(_recv_Gcc_e, 0., dom[rank].Gcc.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gcc_w, 0., dom[rank].Gcc.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gcc_n, 0., dom[rank].Gcc.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gcc_s, 0., dom[rank].Gcc.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gcc_t, 0., dom[rank].Gcc.s2_k * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gcc_b, 0., dom[rank].Gcc.s2_k * sizeof(real)));

  checkCudaErrors(cudaMemset(_recv_Gfx_e, 0., dom[rank].Gfx.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfx_w, 0., dom[rank].Gfx.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfx_n, 0., dom[rank].Gfx.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfx_s, 0., dom[rank].Gfx.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfx_t, 0., dom[rank].Gfx.s2_k * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfx_b, 0., dom[rank].Gfx.s2_k * sizeof(real)));

  checkCudaErrors(cudaMemset(_recv_Gfy_e, 0., dom[rank].Gfy.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfy_w, 0., dom[rank].Gfy.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfy_n, 0., dom[rank].Gfy.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfy_s, 0., dom[rank].Gfy.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfy_t, 0., dom[rank].Gfy.s2_k * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfy_b, 0., dom[rank].Gfy.s2_k * sizeof(real)));

  checkCudaErrors(cudaMemset(_recv_Gfz_e, 0., dom[rank].Gfz.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfz_w, 0., dom[rank].Gfz.s2_i * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfz_n, 0., dom[rank].Gfz.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfz_s, 0., dom[rank].Gfz.s2_j * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfz_t, 0., dom[rank].Gfz.s2_k * sizeof(real)));
  checkCudaErrors(cudaMemset(_recv_Gfz_b, 0., dom[rank].Gfz.s2_k * sizeof(real)));
}

extern "C"
void cuda_update_bc(void)
{
    //printf("\nupdate bc\n");
    update_vel_BC<<<1, 1>>>(_bc, v_bc_tdelay, ttime);

}

extern "C"
void cuda_dom_push(void)
{
  // Push initialized domain data from host to device
  checkCudaErrors(cudaMemcpy(_p, p, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_u, u, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_v, v, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_w, w, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_p0, p0, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_u0, u0, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_v0, v0, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_w0, w0, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_phi, phi, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_u_star, u_star, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_v_star, v_star, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_w_star, w_star, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_conv_u, conv_u, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_conv_v, conv_v, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_conv_w, conv_w, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_conv0_u, conv0_u, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_conv0_v, conv0_v, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_conv0_w, conv0_w, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_diff_u, diff_u, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_diff_v, diff_v, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_diff_w, diff_w, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_diff0_u, diff0_u, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_diff0_v, diff0_v, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_diff0_w, diff0_w, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(_f_x, f_x, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_f_y, f_y, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(_f_z, f_z, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyHostToDevice));
}

extern "C"
void cuda_blocks_init()
{
  //printf("N%d >> Creating cuda thread dimensions and size\n", rank);
  gpumem += sizeof(cuda_blocks_struct);

  int threads_x = 0;
  int threads_y = 0;
  int threads_z = 0;
  int blocks_x = 0;
  int blocks_y = 0;
  int blocks_z = 0;

  /* Computational Grid - Gcc */
  threads_x = dom[rank].Gcc.in * (dom[rank].Gcc.in  < MAX_THREADS_DIM)
            + MAX_THREADS_DIM *   (dom[rank].Gcc.in >= MAX_THREADS_DIM);
  threads_y = dom[rank].Gcc.jn * (dom[rank].Gcc.jn  < MAX_THREADS_DIM)
            + MAX_THREADS_DIM *   (dom[rank].Gcc.jn >= MAX_THREADS_DIM);
  threads_z = dom[rank].Gcc.kn * (dom[rank].Gcc.kn  < MAX_THREADS_DIM)
            + MAX_THREADS_DIM *   (dom[rank].Gcc.kn >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gcc.in / (real) threads_x);
  blocks_y = (int) ceil((real) dom[rank].Gcc.jn / (real) threads_y);
  blocks_z = (int) ceil((real) dom[rank].Gcc.kn / (real) threads_z);

  // Create tmp variables
  dim3 Gcc_dim_in(threads_y, threads_z);
  dim3 Gcc_dim_jn(threads_z, threads_x);
  dim3 Gcc_dim_kn(threads_x, threads_y);
  dim3 Gcc_num_in(blocks_y, blocks_z);
  dim3 Gcc_num_jn(blocks_z, blocks_x);
  dim3 Gcc_num_kn(blocks_x, blocks_y);
  dim3 Gcc_dim_s3(threads_x, threads_y, threads_z);
  dim3 Gcc_num_s3(blocks_x, blocks_y, blocks_z);

  // Copy (by value) to structs
  blocks.Gcc.dim_in = Gcc_dim_in;
  blocks.Gcc.dim_jn = Gcc_dim_jn;
  blocks.Gcc.dim_kn = Gcc_dim_kn;
  blocks.Gcc.num_in = Gcc_num_in;
  blocks.Gcc.num_jn = Gcc_num_jn;
  blocks.Gcc.num_kn = Gcc_num_kn;
  blocks.Gcc.dim_s3 = Gcc_dim_s3;
  blocks.Gcc.num_s3 = Gcc_num_s3;

  /* Computational Shared Grid, GCC */
  threads_x = (dom[rank].Gcc.in+2) *((dom[rank].Gcc.in+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gcc.in+2) >= MAX_THREADS_DIM);
  threads_y = (dom[rank].Gcc.jn+2) *((dom[rank].Gcc.jn+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gcc.jn+2) >= MAX_THREADS_DIM);
  threads_z = (dom[rank].Gcc.kn+2) *((dom[rank].Gcc.kn+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gcc.kn+2) >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gcc.in / (real) (threads_x - 2));
  blocks_y = (int) ceil((real) dom[rank].Gcc.jn / (real) (threads_y - 2));
  blocks_z = (int) ceil((real) dom[rank].Gcc.kn / (real) (threads_z - 2));

  // Create tmp variables
  dim3 Gcc_dim_in_s(threads_y, threads_z);
  dim3 Gcc_dim_jn_s(threads_z, threads_x);
  dim3 Gcc_dim_kn_s(threads_x, threads_y);
  dim3 Gcc_num_in_s(blocks_y, blocks_z);
  dim3 Gcc_num_jn_s(blocks_z, blocks_x);
  dim3 Gcc_num_kn_s(blocks_x, blocks_y);

  // Copy (by value) to structs
  blocks.Gcc.dim_in_s = Gcc_dim_in_s;
  blocks.Gcc.dim_jn_s = Gcc_dim_jn_s;
  blocks.Gcc.dim_kn_s = Gcc_dim_kn_s;
  blocks.Gcc.num_in_s = Gcc_num_in_s;
  blocks.Gcc.num_jn_s = Gcc_num_jn_s;
  blocks.Gcc.num_kn_s = Gcc_num_kn_s;

  /* Computational Grid - Gfx */
  threads_x = dom[rank].Gfx.in * (dom[rank].Gfx.in < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfx.in >= MAX_THREADS_DIM);
  threads_y = dom[rank].Gfx.jn * (dom[rank].Gfx.jn < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfx.jn >= MAX_THREADS_DIM);
  threads_z = dom[rank].Gfx.kn * (dom[rank].Gfx.kn < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfx.kn >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfx.in / (real) threads_x);
  blocks_y = (int) ceil((real) dom[rank].Gfx.jn / (real) threads_y);
  blocks_z = (int) ceil((real) dom[rank].Gfx.kn / (real) threads_z);

  // Create tmp variables
  dim3 Gfx_dim_in(threads_y, threads_z);
  dim3 Gfx_dim_jn(threads_z, threads_x);
  dim3 Gfx_dim_kn(threads_x, threads_y);
  dim3 Gfx_num_in(blocks_y, blocks_z);
  dim3 Gfx_num_jn(blocks_z, blocks_x);
  dim3 Gfx_num_kn(blocks_x, blocks_y);
  dim3 Gfx_dim_s3(threads_x, threads_y, threads_z);
  dim3 Gfx_num_s3(blocks_x, blocks_y, blocks_z);

  // Copy (by value) to structs
  blocks.Gfx.dim_in = Gfx_dim_in;
  blocks.Gfx.dim_jn = Gfx_dim_jn;
  blocks.Gfx.dim_kn = Gfx_dim_kn;
  blocks.Gfx.num_in = Gfx_num_in;
  blocks.Gfx.num_jn = Gfx_num_jn;
  blocks.Gfx.num_kn = Gfx_num_kn;
  blocks.Gfx.dim_s3 = Gfx_dim_s3;
  blocks.Gfx.num_s3 = Gfx_num_s3;

  /* Computational Shared Grid - Gfx */
  threads_x = (dom[rank].Gfx.in+2) *((dom[rank].Gfx.in+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfx.in+2) >= MAX_THREADS_DIM);
  threads_y = (dom[rank].Gfx.jn+2) *((dom[rank].Gfx.jn+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfx.jn+2) >= MAX_THREADS_DIM);
  threads_z = (dom[rank].Gfx.kn+2) *((dom[rank].Gfx.kn+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfx.kn+2) >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfx.in / (real) (threads_x - 2));
  blocks_y = (int) ceil((real) dom[rank].Gfx.jn / (real) (threads_y - 2));
  blocks_z = (int) ceil((real) dom[rank].Gfx.kn / (real) (threads_z - 2));

  // Create tmp variables
  dim3 Gfx_dim_in_s(threads_y, threads_z);
  dim3 Gfx_dim_jn_s(threads_z, threads_x);
  dim3 Gfx_dim_kn_s(threads_x, threads_y);
  dim3 Gfx_num_in_s(blocks_y, blocks_z);
  dim3 Gfx_num_jn_s(blocks_z, blocks_x);
  dim3 Gfx_num_kn_s(blocks_x, blocks_y);

  // Copy (by value) to structs
  blocks.Gfx.dim_in_s = Gfx_dim_in_s;
  blocks.Gfx.dim_jn_s = Gfx_dim_jn_s;
  blocks.Gfx.dim_kn_s = Gfx_dim_kn_s;
  blocks.Gfx.num_in_s = Gfx_num_in_s;
  blocks.Gfx.num_jn_s = Gfx_num_jn_s;
  blocks.Gfx.num_kn_s = Gfx_num_kn_s;

  /* Computational Grid - Gfy */
  threads_x = dom[rank].Gfy.in * (dom[rank].Gfy.in < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfy.in >= MAX_THREADS_DIM);
  threads_y = dom[rank].Gfy.jn * (dom[rank].Gfy.jn < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfy.jn >= MAX_THREADS_DIM);
  threads_z = dom[rank].Gfy.kn * (dom[rank].Gfy.kn < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfy.kn >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfy.in / (real) threads_x);
  blocks_y = (int) ceil((real) dom[rank].Gfy.jn / (real) threads_y);
  blocks_z = (int) ceil((real) dom[rank].Gfy.kn / (real) threads_z);

  // Create tmp variables
  dim3 Gfy_dim_in(threads_y, threads_z);
  dim3 Gfy_dim_jn(threads_z, threads_x);
  dim3 Gfy_dim_kn(threads_x, threads_y);
  dim3 Gfy_num_in(blocks_y, blocks_z);
  dim3 Gfy_num_jn(blocks_z, blocks_x);
  dim3 Gfy_num_kn(blocks_x, blocks_y);
  dim3 Gfy_dim_s3(threads_x, threads_y, threads_z);
  dim3 Gfy_num_s3(blocks_x, blocks_y, blocks_z);

  // Copy (by value) to structs
  blocks.Gfy.dim_in = Gfy_dim_in;
  blocks.Gfy.dim_jn = Gfy_dim_jn;
  blocks.Gfy.dim_kn = Gfy_dim_kn;
  blocks.Gfy.num_in = Gfy_num_in;
  blocks.Gfy.num_jn = Gfy_num_jn;
  blocks.Gfy.num_kn = Gfy_num_kn;
  blocks.Gfy.dim_s3 = Gfy_dim_s3;
  blocks.Gfy.num_s3 = Gfy_num_s3;

  /* Computational Shared Grid - Gfy */
  threads_x = (dom[rank].Gfy.in+2) *((dom[rank].Gfy.in+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfy.in+2) >= MAX_THREADS_DIM);
  threads_y = (dom[rank].Gfy.jn+2) *((dom[rank].Gfy.jn+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfy.jn+2) >= MAX_THREADS_DIM);
  threads_z = (dom[rank].Gfy.kn+2) *((dom[rank].Gfy.kn+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfy.kn+2) >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfy.in / (real) (threads_x - 2));
  blocks_y = (int) ceil((real) dom[rank].Gfy.jn / (real) (threads_y - 2));
  blocks_z = (int) ceil((real) dom[rank].Gfy.kn / (real) (threads_z - 2));

  // Create tmp variables
  dim3 Gfy_dim_in_s(threads_y, threads_z);
  dim3 Gfy_dim_jn_s(threads_z, threads_x);
  dim3 Gfy_dim_kn_s(threads_x, threads_y);
  dim3 Gfy_num_in_s(blocks_y, blocks_z);
  dim3 Gfy_num_jn_s(blocks_z, blocks_x);
  dim3 Gfy_num_kn_s(blocks_x, blocks_y);

  // Copy (by value) to structs
  blocks.Gfy.dim_in_s = Gfy_dim_in_s;
  blocks.Gfy.dim_jn_s = Gfy_dim_jn_s;
  blocks.Gfy.dim_kn_s = Gfy_dim_kn_s;
  blocks.Gfy.num_in_s = Gfy_num_in_s;
  blocks.Gfy.num_jn_s = Gfy_num_jn_s;
  blocks.Gfy.num_kn_s = Gfy_num_kn_s;

  /* Computational Grid - Gfz */
  threads_x = dom[rank].Gfz.in * (dom[rank].Gfz.in < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfz.in >= MAX_THREADS_DIM);
  threads_y = dom[rank].Gfz.jn * (dom[rank].Gfz.jn < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfz.jn >= MAX_THREADS_DIM);
  threads_z = dom[rank].Gfz.kn * (dom[rank].Gfz.kn < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfz.kn >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfz.in / (real) threads_x);
  blocks_y = (int) ceil((real) dom[rank].Gfz.jn / (real) threads_y);
  blocks_z = (int) ceil((real) dom[rank].Gfz.kn / (real) threads_z);

  // Create tmp variables
  dim3 Gfz_dim_in(threads_y, threads_z);
  dim3 Gfz_dim_jn(threads_z, threads_x);
  dim3 Gfz_dim_kn(threads_x, threads_y);
  dim3 Gfz_num_in(blocks_y, blocks_z);
  dim3 Gfz_num_jn(blocks_z, blocks_x);
  dim3 Gfz_num_kn(blocks_x, blocks_y);
  dim3 Gfz_dim_s3(threads_x, threads_y, threads_z);
  dim3 Gfz_num_s3(blocks_x, blocks_y, blocks_z);

  // Copy (by value) to structs
  blocks.Gfz.dim_in = Gfz_dim_in;
  blocks.Gfz.dim_jn = Gfz_dim_jn;
  blocks.Gfz.dim_kn = Gfz_dim_kn;
  blocks.Gfz.num_in = Gfz_num_in;
  blocks.Gfz.num_jn = Gfz_num_jn;
  blocks.Gfz.num_kn = Gfz_num_kn;
  blocks.Gfz.dim_s3 = Gfz_dim_s3;
  blocks.Gfz.num_s3 = Gfz_num_s3;

  /* Computational Shared Grid - Gfz */
  threads_x = (dom[rank].Gfz.in+2) *((dom[rank].Gfz.in+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfz.in+2) >= MAX_THREADS_DIM);
  threads_y = (dom[rank].Gfz.jn+2) *((dom[rank].Gfz.jn+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfz.jn+2) >= MAX_THREADS_DIM);
  threads_z = (dom[rank].Gfz.kn+2) *((dom[rank].Gfz.kn+2) < MAX_THREADS_DIM)
                  + MAX_THREADS_DIM *((dom[rank].Gfz.kn+2) >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfz.in / (real) (threads_x - 2));
  blocks_y = (int) ceil((real) dom[rank].Gfz.jn / (real) (threads_y - 2));
  blocks_z = (int) ceil((real) dom[rank].Gfz.kn / (real) (threads_z - 2));

  // Create tmp variables
  dim3 Gfz_dim_in_s(threads_y, threads_z);
  dim3 Gfz_dim_jn_s(threads_z, threads_x);
  dim3 Gfz_dim_kn_s(threads_x, threads_y);
  dim3 Gfz_num_in_s(blocks_y, blocks_z);
  dim3 Gfz_num_jn_s(blocks_z, blocks_x);
  dim3 Gfz_num_kn_s(blocks_x, blocks_y);

  // Copy (by value) to structs
  blocks.Gfz.dim_in_s = Gfz_dim_in_s;
  blocks.Gfz.dim_jn_s = Gfz_dim_jn_s;
  blocks.Gfz.dim_kn_s = Gfz_dim_kn_s;
  blocks.Gfz.num_in_s = Gfz_num_in_s;
  blocks.Gfz.num_jn_s = Gfz_num_jn_s;
  blocks.Gfz.num_kn_s = Gfz_num_kn_s;

  /* Ghost grid - Gcc */
  threads_x = dom[rank].Gcc.inb * (dom[rank].Gcc.inb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gcc.inb >= MAX_THREADS_DIM);
  threads_y = dom[rank].Gcc.jnb * (dom[rank].Gcc.jnb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gcc.jnb >= MAX_THREADS_DIM);
  threads_z = dom[rank].Gcc.knb * (dom[rank].Gcc.knb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gcc.knb >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gcc.inb / (real) threads_x);
  blocks_y = (int) ceil((real) dom[rank].Gcc.jnb / (real) threads_y);
  blocks_z = (int) ceil((real) dom[rank].Gcc.knb / (real) threads_z);

  // Create tmp variables
  dim3 Gcc_dim_inb(threads_y, threads_z);
  dim3 Gcc_dim_jnb(threads_z, threads_x);
  dim3 Gcc_dim_knb(threads_x, threads_y);
  dim3 Gcc_num_inb(blocks_y, blocks_z);
  dim3 Gcc_num_jnb(blocks_z, blocks_x);
  dim3 Gcc_num_knb(blocks_x, blocks_y);
  dim3 Gcc_dim_s3b(threads_x, threads_y, threads_z);
  dim3 Gcc_num_s3b(blocks_x, blocks_y, blocks_z);

  // Copy (by value) to structs
  blocks.Gcc.dim_inb = Gcc_dim_inb;
  blocks.Gcc.dim_jnb = Gcc_dim_jnb;
  blocks.Gcc.dim_knb = Gcc_dim_knb;
  blocks.Gcc.num_inb = Gcc_num_inb;
  blocks.Gcc.num_jnb = Gcc_num_jnb;
  blocks.Gcc.num_knb = Gcc_num_knb;
  blocks.Gcc.dim_s3b = Gcc_dim_s3b;
  blocks.Gcc.num_s3b = Gcc_num_s3b;

  /* Ghost Shared Grid - Gcc */
  threads_x = (dom[rank].Gcc.inb+2) *((dom[rank].Gcc.inb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gcc.inb+2) >= MAX_THREADS_DIM);
  threads_y = (dom[rank].Gcc.jnb+2) *((dom[rank].Gcc.jnb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gcc.jnb+2) >= MAX_THREADS_DIM);
  threads_z = (dom[rank].Gcc.knb+2) *((dom[rank].Gcc.knb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gcc.knb+2) >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gcc.inb / (real) (threads_x - 2));
  blocks_y = (int) ceil((real) dom[rank].Gcc.jnb / (real) (threads_y - 2));
  blocks_z = (int) ceil((real) dom[rank].Gcc.knb / (real) (threads_z - 2));

  // Create tmp variables
  dim3 Gcc_dim_inb_s(threads_y, threads_z);
  dim3 Gcc_dim_jnb_s(threads_z, threads_x);
  dim3 Gcc_dim_knb_s(threads_x, threads_y);
  dim3 Gcc_num_inb_s(blocks_y, blocks_z);
  dim3 Gcc_num_jnb_s(blocks_z, blocks_x);
  dim3 Gcc_num_knb_s(blocks_x, blocks_y);

  // Copy (by value) to structs
  blocks.Gcc.dim_inb_s = Gcc_dim_inb_s;
  blocks.Gcc.dim_jnb_s = Gcc_dim_jnb_s;
  blocks.Gcc.dim_knb_s = Gcc_dim_knb_s;
  blocks.Gcc.num_inb_s = Gcc_num_inb_s;
  blocks.Gcc.num_jnb_s = Gcc_num_jnb_s;
  blocks.Gcc.num_knb_s = Gcc_num_knb_s;

  /* Ghost grid - Gfx */
  threads_x = dom[rank].Gfx.inb * (dom[rank].Gfx.inb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfx.inb >= MAX_THREADS_DIM);
  threads_y = dom[rank].Gfx.jnb * (dom[rank].Gfx.jnb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfx.jnb >= MAX_THREADS_DIM);
  threads_z = dom[rank].Gfx.knb * (dom[rank].Gfx.knb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfx.knb >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfx.inb / (real) threads_x);
  blocks_y = (int) ceil((real) dom[rank].Gfx.jnb / (real) threads_y);
  blocks_z = (int) ceil((real) dom[rank].Gfx.knb / (real) threads_z);

  // Create tmp variables
  dim3 Gfx_dim_inb(threads_y, threads_z);
  dim3 Gfx_dim_jnb(threads_z, threads_x);
  dim3 Gfx_dim_knb(threads_x, threads_y);
  dim3 Gfx_num_inb(blocks_y, blocks_z);
  dim3 Gfx_num_jnb(blocks_z, blocks_x);
  dim3 Gfx_num_knb(blocks_x, blocks_y);
  dim3 Gfx_dim_s3b(threads_x, threads_y, threads_z);
  dim3 Gfx_num_s3b(blocks_x, blocks_y, blocks_z);

  // Copy (by value) to structs
  blocks.Gfx.dim_inb = Gfx_dim_inb;
  blocks.Gfx.dim_jnb = Gfx_dim_jnb;
  blocks.Gfx.dim_knb = Gfx_dim_knb;
  blocks.Gfx.num_inb = Gfx_num_inb;
  blocks.Gfx.num_jnb = Gfx_num_jnb;
  blocks.Gfx.num_knb = Gfx_num_knb;
  blocks.Gfx.dim_s3b = Gfx_dim_s3b;
  blocks.Gfx.num_s3b = Gfx_num_s3b;

  /* Ghost Shared Grid - Gfx */
  threads_x = (dom[rank].Gfx.inb+2) *((dom[rank].Gfx.inb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfx.inb+2) >= MAX_THREADS_DIM);
  threads_y = (dom[rank].Gfx.jnb+2) *((dom[rank].Gfx.jnb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfx.jnb+2) >= MAX_THREADS_DIM);
  threads_z = (dom[rank].Gfx.knb+2) *((dom[rank].Gfx.knb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfx.knb+2) >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfx.inb / (real) (threads_x - 2));
  blocks_y = (int) ceil((real) dom[rank].Gfx.jnb / (real) (threads_y - 2));
  blocks_z = (int) ceil((real) dom[rank].Gfx.knb / (real) (threads_z - 2));

  // Create tmp variables
  dim3 Gfx_dim_inb_s(threads_y, threads_z);
  dim3 Gfx_dim_jnb_s(threads_z, threads_x);
  dim3 Gfx_dim_knb_s(threads_x, threads_y);
  dim3 Gfx_num_inb_s(blocks_y, blocks_z);
  dim3 Gfx_num_jnb_s(blocks_z, blocks_x);
  dim3 Gfx_num_knb_s(blocks_x, blocks_y);

  // Copy (by value) to structs
  blocks.Gfx.dim_inb_s = Gfx_dim_inb_s;
  blocks.Gfx.dim_jnb_s = Gfx_dim_jnb_s;
  blocks.Gfx.dim_knb_s = Gfx_dim_knb_s;
  blocks.Gfx.num_inb_s = Gfx_num_inb_s;
  blocks.Gfx.num_jnb_s = Gfx_num_jnb_s;
  blocks.Gfx.num_knb_s = Gfx_num_knb_s;

  /* Ghost grid - Gfy */
  threads_x = dom[rank].Gfy.inb * (dom[rank].Gfy.inb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfy.inb >= MAX_THREADS_DIM);
  threads_y = dom[rank].Gfy.jnb * (dom[rank].Gfy.jnb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfy.jnb >= MAX_THREADS_DIM);
  threads_z = dom[rank].Gfy.knb * (dom[rank].Gfy.knb < MAX_THREADS_DIM)
            + MAX_THREADS_DIM * (dom[rank].Gfy.knb >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfy.inb / (real) threads_x);
  blocks_y = (int) ceil((real) dom[rank].Gfy.jnb / (real) threads_y);
  blocks_z = (int) ceil((real) dom[rank].Gfy.knb / (real) threads_z);

  // Create tmp variables
  dim3 Gfy_dim_inb(threads_y, threads_z);
  dim3 Gfy_dim_jnb(threads_z, threads_x);
  dim3 Gfy_dim_knb(threads_x, threads_y);
  dim3 Gfy_num_inb(blocks_y, blocks_z);
  dim3 Gfy_num_jnb(blocks_z, blocks_x);
  dim3 Gfy_num_knb(blocks_x, blocks_y);
  dim3 Gfy_dim_s3b(threads_x, threads_y, threads_z);
  dim3 Gfy_num_s3b(blocks_x, blocks_y, blocks_z);

  // Copy (by value) to structs
  blocks.Gfy.dim_inb = Gfy_dim_inb;
  blocks.Gfy.dim_jnb = Gfy_dim_jnb;
  blocks.Gfy.dim_knb = Gfy_dim_knb;
  blocks.Gfy.num_inb = Gfy_num_inb;
  blocks.Gfy.num_jnb = Gfy_num_jnb;
  blocks.Gfy.num_knb = Gfy_num_knb;
  blocks.Gfy.dim_s3b = Gfy_dim_s3b;
  blocks.Gfy.num_s3b = Gfy_num_s3b;

  /* Ghost Shared Grid - Gfy */
  threads_x = (dom[rank].Gfy.inb+2) *((dom[rank].Gfy.inb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfy.inb+2) >= MAX_THREADS_DIM);
  threads_y = (dom[rank].Gfy.jnb+2) *((dom[rank].Gfy.jnb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfy.jnb+2) >= MAX_THREADS_DIM);
  threads_z = (dom[rank].Gfy.knb+2) *((dom[rank].Gfy.knb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfy.knb+2) >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfy.inb / (real) (threads_x - 2));
  blocks_y = (int) ceil((real) dom[rank].Gfy.jnb / (real) (threads_y - 2));
  blocks_z = (int) ceil((real) dom[rank].Gfy.knb / (real) (threads_z - 2));

  // Create tmp variables
  dim3 Gfy_dim_inb_s(threads_y, threads_z);
  dim3 Gfy_dim_jnb_s(threads_z, threads_x);
  dim3 Gfy_dim_knb_s(threads_x, threads_y);
  dim3 Gfy_num_inb_s(blocks_y, blocks_z);
  dim3 Gfy_num_jnb_s(blocks_z, blocks_x);
  dim3 Gfy_num_knb_s(blocks_x, blocks_y);

  // Copy (by value) to structs
  blocks.Gfy.dim_inb_s = Gfy_dim_inb_s;
  blocks.Gfy.dim_jnb_s = Gfy_dim_jnb_s;
  blocks.Gfy.dim_knb_s = Gfy_dim_knb_s;
  blocks.Gfy.num_inb_s = Gfy_num_inb_s;
  blocks.Gfy.num_jnb_s = Gfy_num_jnb_s;
  blocks.Gfy.num_knb_s = Gfy_num_knb_s;

  /* Ghost grid - Gfz */
  threads_x = dom[rank].Gfz.inb * (dom[rank].Gfz.inb < MAX_THREADS_DIM)
               + MAX_THREADS_DIM * (dom[rank].Gfz.inb >= MAX_THREADS_DIM);
  threads_y = dom[rank].Gfz.jnb * (dom[rank].Gfz.jnb < MAX_THREADS_DIM)
               + MAX_THREADS_DIM * (dom[rank].Gfz.jnb >= MAX_THREADS_DIM);
  threads_z = dom[rank].Gfz.knb * (dom[rank].Gfz.knb < MAX_THREADS_DIM)
               + MAX_THREADS_DIM * (dom[rank].Gfz.knb >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfz.inb / (real) threads_x);
  blocks_y = (int) ceil((real) dom[rank].Gfz.jnb / (real) threads_y);
  blocks_z = (int) ceil((real) dom[rank].Gfz.knb / (real) threads_z);

  // Create tmp variables
  dim3 Gfz_dim_inb(threads_y, threads_z);
  dim3 Gfz_dim_jnb(threads_z, threads_x);
  dim3 Gfz_dim_knb(threads_x, threads_y);
  dim3 Gfz_num_inb(blocks_y, blocks_z);
  dim3 Gfz_num_jnb(blocks_z, blocks_x);
  dim3 Gfz_num_knb(blocks_x, blocks_y);
  dim3 Gfz_dim_s3b(threads_x, threads_y, threads_z);
  dim3 Gfz_num_s3b(blocks_x, blocks_y, blocks_z);

  // Copy (by value) to structs
  blocks.Gfz.dim_inb = Gfz_dim_inb;
  blocks.Gfz.dim_jnb = Gfz_dim_jnb;
  blocks.Gfz.dim_knb = Gfz_dim_knb;
  blocks.Gfz.num_inb = Gfz_num_inb;
  blocks.Gfz.num_jnb = Gfz_num_jnb;
  blocks.Gfz.num_knb = Gfz_num_knb;
  blocks.Gfz.dim_s3b = Gfz_dim_s3b;
  blocks.Gfz.num_s3b = Gfz_num_s3b;

  /* Ghost Shared Grid - Gfz */
  threads_x = (dom[rank].Gfz.inb+2) *((dom[rank].Gfz.inb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfz.inb+2) >= MAX_THREADS_DIM);
  threads_y = (dom[rank].Gfz.jnb+2) *((dom[rank].Gfz.jnb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfz.jnb+2) >= MAX_THREADS_DIM);
  threads_z = (dom[rank].Gfz.knb+2) *((dom[rank].Gfz.knb+2) < MAX_THREADS_DIM)
                 + MAX_THREADS_DIM *((dom[rank].Gfz.knb+2) >= MAX_THREADS_DIM);

  blocks_x = (int) ceil((real) dom[rank].Gfz.inb / (real) (threads_x - 2));
  blocks_y = (int) ceil((real) dom[rank].Gfz.jnb / (real) (threads_y - 2));
  blocks_z = (int) ceil((real) dom[rank].Gfz.knb / (real) (threads_z - 2));

  // Create tmp variables
  dim3 Gfz_dim_inb_s(threads_y, threads_z);
  dim3 Gfz_dim_jnb_s(threads_z, threads_x);
  dim3 Gfz_dim_knb_s(threads_x, threads_y);
  dim3 Gfz_num_inb_s(blocks_y, blocks_z);
  dim3 Gfz_num_jnb_s(blocks_z, blocks_x);
  dim3 Gfz_num_knb_s(blocks_x, blocks_y);

  // Copy (by value) to structs
  blocks.Gfz.dim_inb_s = Gfz_dim_inb_s;
  blocks.Gfz.dim_jnb_s = Gfz_dim_jnb_s;
  blocks.Gfz.dim_knb_s = Gfz_dim_knb_s;
  blocks.Gfz.num_inb_s = Gfz_num_inb_s;
  blocks.Gfz.num_jnb_s = Gfz_num_jnb_s;
  blocks.Gfz.num_knb_s = Gfz_num_knb_s;

  #ifdef DDEBUG
    cuda_blocks_write();
  #endif
}

extern "C"
void cuda_blocks_write(void)
{
  char fname[CHAR_BUF_SIZE];
  sprintf(fname, "%s/rank-%d-map.debug", ROOT_DIR, rank);
  FILE *outfile = fopen(fname, "a");
  if (outfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  fprintf(outfile, "\n\n");
  fprintf(outfile, "Blocks:\n");

  /* Computational Grid - Gcc */
  fprintf(outfile, "  blocks.Gcc:\n");
  fprintf(outfile, "    num_in = (%d, %d, %d), dim_in = (%d, %d, %d)\n",
    blocks.Gcc.num_in.x, blocks.Gcc.num_in.y, blocks.Gcc.num_in.z,
    blocks.Gcc.dim_in.x, blocks.Gcc.dim_in.y, blocks.Gcc.dim_in.z);
  fprintf(outfile, "    num_jn = (%d, %d, %d), dim_jn = (%d, %d, %d)\n",
    blocks.Gcc.num_jn.x, blocks.Gcc.num_jn.y, blocks.Gcc.num_jn.z,
    blocks.Gcc.dim_jn.x, blocks.Gcc.dim_jn.y, blocks.Gcc.dim_jn.z);
  fprintf(outfile, "    num_kn = (%d, %d, %d), dim_kn = (%d, %d, %d)\n",
    blocks.Gcc.num_kn.x, blocks.Gcc.num_kn.y, blocks.Gcc.num_kn.z,
    blocks.Gcc.dim_kn.x, blocks.Gcc.dim_kn.y, blocks.Gcc.dim_kn.z);
  fprintf(outfile, "    num_s3 = (%d, %d, %d), dim_s3 = (%d, %d, %d)\n",
    blocks.Gcc.num_s3.x, blocks.Gcc.num_s3.y, blocks.Gcc.num_s3.z,
    blocks.Gcc.dim_s3.x, blocks.Gcc.dim_s3.y, blocks.Gcc.dim_s3.z);
  /* Computational Shared Grid, Gcc */
  fprintf(outfile, "    num_in_s = (%d, %d, %d), dim_in_s = (%d, %d, %d)\n",
    blocks.Gcc.num_in_s.x, blocks.Gcc.num_in_s.y, blocks.Gcc.num_in_s.z,
    blocks.Gcc.dim_in_s.x, blocks.Gcc.dim_in_s.y, blocks.Gcc.dim_in_s.z);
  fprintf(outfile, "    num_jn_s = (%d, %d, %d), dim_jn_s = (%d, %d, %d)\n",
    blocks.Gcc.num_jn_s.x, blocks.Gcc.num_jn_s.y, blocks.Gcc.num_jn_s.z,
    blocks.Gcc.dim_jn_s.x, blocks.Gcc.dim_jn_s.y, blocks.Gcc.dim_jn_s.z);
  fprintf(outfile, "    num_kn_s = (%d, %d, %d), dim_kn_s = (%d, %d, %d)\n",
    blocks.Gcc.num_kn_s.x, blocks.Gcc.num_kn_s.y, blocks.Gcc.num_kn_s.z,
    blocks.Gcc.dim_kn_s.x, blocks.Gcc.dim_kn_s.y, blocks.Gcc.dim_kn_s.z);
  /* Ghost grid - Gcc */
  fprintf(outfile, "    num_inb = (%d, %d, %d), dim_inb = (%d, %d, %d)\n",
    blocks.Gcc.num_inb.x, blocks.Gcc.num_inb.y, blocks.Gcc.num_inb.z,
    blocks.Gcc.dim_inb.x, blocks.Gcc.dim_inb.y, blocks.Gcc.dim_inb.z);
  fprintf(outfile, "    num_jnb = (%d, %d, %d), dim_jnb = (%d, %d, %d)\n",
    blocks.Gcc.num_jnb.x, blocks.Gcc.num_jnb.y, blocks.Gcc.num_jnb.z,
    blocks.Gcc.dim_jnb.x, blocks.Gcc.dim_jnb.y, blocks.Gcc.dim_jnb.z);
  fprintf(outfile, "    num_knb = (%d, %d, %d), dim_knb = (%d, %d, %d)\n",
    blocks.Gcc.num_knb.x, blocks.Gcc.num_knb.y, blocks.Gcc.num_knb.z,
    blocks.Gcc.dim_knb.x, blocks.Gcc.dim_knb.y, blocks.Gcc.dim_knb.z);
  fprintf(outfile, "    num_s3b = (%d, %d, %d), dim_s3b = (%d, %d, %d)\n",
    blocks.Gcc.num_s3b.x, blocks.Gcc.num_s3b.y, blocks.Gcc.num_s3b.z,
    blocks.Gcc.dim_s3b.x, blocks.Gcc.dim_s3b.y, blocks.Gcc.dim_s3b.z);
  /* Ghost Shared Grid - Gcc */
  fprintf(outfile, "    num_inb_s = (%d, %d, %d), dim_inb_s = (%d, %d, %d)\n",
    blocks.Gcc.num_inb_s.x, blocks.Gcc.num_inb_s.y, blocks.Gcc.num_inb_s.z,
    blocks.Gcc.dim_inb_s.x, blocks.Gcc.dim_inb_s.y, blocks.Gcc.dim_inb_s.z);
  fprintf(outfile, "    num_jnb_s = (%d, %d, %d), dim_jnb_s = (%d, %d, %d)\n",
    blocks.Gcc.num_jnb_s.x, blocks.Gcc.num_jnb_s.y, blocks.Gcc.num_jnb_s.z,
    blocks.Gcc.dim_jnb_s.x, blocks.Gcc.dim_jnb_s.y, blocks.Gcc.dim_jnb_s.z);
  fprintf(outfile, "    num_knb_s = (%d, %d, %d), dim_knb_s = (%d, %d, %d)\n",
    blocks.Gcc.num_knb_s.x, blocks.Gcc.num_knb_s.y, blocks.Gcc.num_knb_s.z,
    blocks.Gcc.dim_knb_s.x, blocks.Gcc.dim_knb_s.y, blocks.Gcc.dim_knb_s.z);

  /* Computational Grid - Gfx */
  fprintf(outfile, "  blocks.Gfx:\n");
  fprintf(outfile, "    num_in = (%d, %d, %d), dim_in = (%d, %d, %d)\n",
    blocks.Gfx.num_in.x, blocks.Gfx.num_in.y, blocks.Gfx.num_in.z,
    blocks.Gfx.dim_in.x, blocks.Gfx.dim_in.y, blocks.Gfx.dim_in.z);
  fprintf(outfile, "    num_jn = (%d, %d, %d), dim_jn = (%d, %d, %d)\n",
    blocks.Gfx.num_jn.x, blocks.Gfx.num_jn.y, blocks.Gfx.num_jn.z,
    blocks.Gfx.dim_jn.x, blocks.Gfx.dim_jn.y, blocks.Gfx.dim_jn.z);
  fprintf(outfile, "    num_kn = (%d, %d, %d), dim_kn = (%d, %d, %d)\n",
    blocks.Gfx.num_kn.x, blocks.Gfx.num_kn.y, blocks.Gfx.num_kn.z,
    blocks.Gfx.dim_kn.x, blocks.Gfx.dim_kn.y, blocks.Gfx.dim_kn.z);
  fprintf(outfile, "    num_s3 = (%d, %d, %d), dim_s3 = (%d, %d, %d)\n",
    blocks.Gfx.num_s3.x, blocks.Gfx.num_s3.y, blocks.Gfx.num_s3.z,
    blocks.Gfx.dim_s3.x, blocks.Gfx.dim_s3.y, blocks.Gfx.dim_s3.z);
  /* Computational Shared Grid - Gfx */
  fprintf(outfile, "    num_in_s = (%d, %d, %d), dim_in_s = (%d, %d, %d)\n",
    blocks.Gfx.num_in_s.x, blocks.Gfx.num_in_s.y, blocks.Gfx.num_in_s.z,
    blocks.Gfx.dim_in_s.x, blocks.Gfx.dim_in_s.y, blocks.Gfx.dim_in_s.z);
  fprintf(outfile, "    num_jn_s = (%d, %d, %d), dim_jn_s = (%d, %d, %d)\n",
    blocks.Gfx.num_jn_s.x, blocks.Gfx.num_jn_s.y, blocks.Gfx.num_jn_s.z,
    blocks.Gfx.dim_jn_s.x, blocks.Gfx.dim_jn_s.y, blocks.Gfx.dim_jn_s.z);
  fprintf(outfile, "    num_kn_s = (%d, %d, %d), dim_kn_s = (%d, %d, %d)\n",
    blocks.Gfx.num_kn_s.x, blocks.Gfx.num_kn_s.y, blocks.Gfx.num_kn_s.z,
    blocks.Gfx.dim_kn_s.x, blocks.Gfx.dim_kn_s.y, blocks.Gfx.dim_kn_s.z);
  /* Ghost grid - Gfx */
  fprintf(outfile, "    num_inb = (%d, %d, %d), dim_inb = (%d, %d, %d)\n",
    blocks.Gfx.num_inb.x, blocks.Gfx.num_inb.y, blocks.Gfx.num_inb.z,
    blocks.Gfx.dim_inb.x, blocks.Gfx.dim_inb.y, blocks.Gfx.dim_inb.z);
  fprintf(outfile, "    num_jnb = (%d, %d, %d), dim_jnb = (%d, %d, %d)\n",
    blocks.Gfx.num_jnb.x, blocks.Gfx.num_jnb.y, blocks.Gfx.num_jnb.z,
    blocks.Gfx.dim_jnb.x, blocks.Gfx.dim_jnb.y, blocks.Gfx.dim_jnb.z);
  fprintf(outfile, "    num_knb = (%d, %d, %d), dim_knb = (%d, %d, %d)\n",
    blocks.Gfx.num_knb.x, blocks.Gfx.num_knb.y, blocks.Gfx.num_knb.z,
    blocks.Gfx.dim_knb.x, blocks.Gfx.dim_knb.y, blocks.Gfx.dim_knb.z);
  fprintf(outfile, "    num_s3b = (%d, %d, %d), dim_s3b = (%d, %d, %d)\n",
    blocks.Gfx.num_s3b.x, blocks.Gfx.num_s3b.y, blocks.Gfx.num_s3b.z,
    blocks.Gfx.dim_s3b.x, blocks.Gfx.dim_s3b.y, blocks.Gfx.dim_s3b.z);
  /* Ghost Shared Grid - Gfx */
  fprintf(outfile, "    num_inb_s = (%d, %d, %d), dim_inb_s = (%d, %d, %d)\n",
    blocks.Gfx.num_inb_s.x, blocks.Gfx.num_inb_s.y, blocks.Gfx.num_inb_s.z,
    blocks.Gfx.dim_inb_s.x, blocks.Gfx.dim_inb_s.y, blocks.Gfx.dim_inb_s.z);
  fprintf(outfile, "    num_jnb_s = (%d, %d, %d), dim_jnb_s = (%d, %d, %d)\n",
    blocks.Gfx.num_jnb_s.x, blocks.Gfx.num_jnb_s.y, blocks.Gfx.num_jnb_s.z,
    blocks.Gfx.dim_jnb_s.x, blocks.Gfx.dim_jnb_s.y, blocks.Gfx.dim_jnb_s.z);
  fprintf(outfile, "    num_knb_s = (%d, %d, %d), dim_knb_s = (%d, %d, %d)\n",
    blocks.Gfx.num_knb_s.x, blocks.Gfx.num_knb_s.y, blocks.Gfx.num_knb_s.z,
    blocks.Gfx.dim_knb_s.x, blocks.Gfx.dim_knb_s.y, blocks.Gfx.dim_knb_s.z);

  /* Computational Grid - Gfy */
  fprintf(outfile, "  blocks.Gfy:\n");
  fprintf(outfile, "    num_in = (%d, %d, %d), dim_in = (%d, %d, %d)\n",
    blocks.Gfy.num_in.x, blocks.Gfy.num_in.y, blocks.Gfy.num_in.z,
    blocks.Gfy.dim_in.x, blocks.Gfy.dim_in.y, blocks.Gfy.dim_in.z);
  fprintf(outfile, "    num_jn = (%d, %d, %d), dim_jn = (%d, %d, %d)\n",
    blocks.Gfy.num_jn.x, blocks.Gfy.num_jn.y, blocks.Gfy.num_jn.z,
    blocks.Gfy.dim_jn.x, blocks.Gfy.dim_jn.y, blocks.Gfy.dim_jn.z);
  fprintf(outfile, "    num_kn = (%d, %d, %d), dim_kn = (%d, %d, %d)\n",
    blocks.Gfy.num_kn.x, blocks.Gfy.num_kn.y, blocks.Gfy.num_kn.z,
    blocks.Gfy.dim_kn.x, blocks.Gfy.dim_kn.y, blocks.Gfy.dim_kn.z);
  fprintf(outfile, "    num_s3 = (%d, %d, %d), dim_s3 = (%d, %d, %d)\n",
    blocks.Gfy.num_s3.x, blocks.Gfy.num_s3.y, blocks.Gfy.num_s3.z,
    blocks.Gfy.dim_s3.x, blocks.Gfy.dim_s3.y, blocks.Gfy.dim_s3.z);
  /* Computational Shared Grid - Gfy */
  fprintf(outfile, "    num_in_s = (%d, %d, %d), dim_in_s = (%d, %d, %d)\n",
    blocks.Gfy.num_in_s.x, blocks.Gfy.num_in_s.y, blocks.Gfy.num_in_s.z,
    blocks.Gfy.dim_in_s.x, blocks.Gfy.dim_in_s.y, blocks.Gfy.dim_in_s.z);
  fprintf(outfile, "    num_jn_s = (%d, %d, %d), dim_jn_s = (%d, %d, %d)\n",
    blocks.Gfy.num_jn_s.x, blocks.Gfy.num_jn_s.y, blocks.Gfy.num_jn_s.z,
    blocks.Gfy.dim_jn_s.x, blocks.Gfy.dim_jn_s.y, blocks.Gfy.dim_jn_s.z);
  fprintf(outfile, "    num_kn_s = (%d, %d, %d), dim_kn_s = (%d, %d, %d)\n",
    blocks.Gfy.num_kn_s.x, blocks.Gfy.num_kn_s.y, blocks.Gfy.num_kn_s.z,
    blocks.Gfy.dim_kn_s.x, blocks.Gfy.dim_kn_s.y, blocks.Gfy.dim_kn_s.z);
  /* Ghost grid - Gfy */
  fprintf(outfile, "    num_inb = (%d, %d, %d), dim_inb = (%d, %d, %d)\n",
    blocks.Gfy.num_inb.x, blocks.Gfy.num_inb.y, blocks.Gfy.num_inb.z,
    blocks.Gfy.dim_inb.x, blocks.Gfy.dim_inb.y, blocks.Gfy.dim_inb.z);
  fprintf(outfile, "    num_jnb = (%d, %d, %d), dim_jnb = (%d, %d, %d)\n",
    blocks.Gfy.num_jnb.x, blocks.Gfy.num_jnb.y, blocks.Gfy.num_jnb.z,
    blocks.Gfy.dim_jnb.x, blocks.Gfy.dim_jnb.y, blocks.Gfy.dim_jnb.z);
  fprintf(outfile, "    num_knb = (%d, %d, %d), dim_knb = (%d, %d, %d)\n",
    blocks.Gfy.num_knb.x, blocks.Gfy.num_knb.y, blocks.Gfy.num_knb.z,
    blocks.Gfy.dim_knb.x, blocks.Gfy.dim_knb.y, blocks.Gfy.dim_knb.z);
  fprintf(outfile, "    num_s3b = (%d, %d, %d), dim_s3b = (%d, %d, %d)\n",
    blocks.Gfy.num_s3b.x, blocks.Gfy.num_s3b.y, blocks.Gfy.num_s3b.z,
    blocks.Gfy.dim_s3b.x, blocks.Gfy.dim_s3b.y, blocks.Gfy.dim_s3b.z);
  /* Ghost Shared Grid - Gfy */
  fprintf(outfile, "    num_inb_s = (%d, %d, %d), dim_inb_s = (%d, %d, %d)\n",
    blocks.Gfy.num_inb_s.x, blocks.Gfy.num_inb_s.y, blocks.Gfy.num_inb_s.z,
    blocks.Gfy.dim_inb_s.x, blocks.Gfy.dim_inb_s.y, blocks.Gfy.dim_inb_s.z);
  fprintf(outfile, "    num_jnb_s = (%d, %d, %d), dim_jnb_s = (%d, %d, %d)\n",
    blocks.Gfy.num_jnb_s.x, blocks.Gfy.num_jnb_s.y, blocks.Gfy.num_jnb_s.z,
    blocks.Gfy.dim_jnb_s.x, blocks.Gfy.dim_jnb_s.y, blocks.Gfy.dim_jnb_s.z);
  fprintf(outfile, "    num_knb_s = (%d, %d, %d), dim_knb_s = (%d, %d, %d)\n",
    blocks.Gfy.num_knb_s.x, blocks.Gfy.num_knb_s.y, blocks.Gfy.num_knb_s.z,
    blocks.Gfy.dim_knb_s.x, blocks.Gfy.dim_knb_s.y, blocks.Gfy.dim_knb_s.z);

  /* Computational Grid - Gfz */
  fprintf(outfile, "  blocks.Gfz:\n");
  fprintf(outfile, "    num_in = (%d, %d, %d), dim_in = (%d, %d, %d)\n",
    blocks.Gfz.num_in.x, blocks.Gfz.num_in.y, blocks.Gfz.num_in.z,
    blocks.Gfz.dim_in.x, blocks.Gfz.dim_in.y, blocks.Gfz.dim_in.z);
  fprintf(outfile, "    num_jn = (%d, %d, %d), dim_jn = (%d, %d, %d)\n",
    blocks.Gfz.num_jn.x, blocks.Gfz.num_jn.y, blocks.Gfz.num_jn.z,
    blocks.Gfz.dim_jn.x, blocks.Gfz.dim_jn.y, blocks.Gfz.dim_jn.z);
  fprintf(outfile, "    num_kn = (%d, %d, %d), dim_kn = (%d, %d, %d)\n",
    blocks.Gfz.num_kn.x, blocks.Gfz.num_kn.y, blocks.Gfz.num_kn.z,
    blocks.Gfz.dim_kn.x, blocks.Gfz.dim_kn.y, blocks.Gfz.dim_kn.z);
  fprintf(outfile, "    num_s3 = (%d, %d, %d), dim_s3 = (%d, %d, %d)\n",
    blocks.Gfz.num_s3.x, blocks.Gfz.num_s3.y, blocks.Gfz.num_s3.z,
    blocks.Gfz.dim_s3.x, blocks.Gfz.dim_s3.y, blocks.Gfz.dim_s3.z);
  /* Computational Shared Grid - Gfz */
  fprintf(outfile, "    num_in_s = (%d, %d, %d), dim_in_s = (%d, %d, %d)\n",
    blocks.Gfz.num_in_s.x, blocks.Gfz.num_in_s.y, blocks.Gfz.num_in_s.z,
    blocks.Gfz.dim_in_s.x, blocks.Gfz.dim_in_s.y, blocks.Gfz.dim_in_s.z);
  fprintf(outfile, "    num_jn_s = (%d, %d, %d), dim_jn_s = (%d, %d, %d)\n",
    blocks.Gfz.num_jn_s.x, blocks.Gfz.num_jn_s.y, blocks.Gfz.num_jn_s.z,
    blocks.Gfz.dim_jn_s.x, blocks.Gfz.dim_jn_s.y, blocks.Gfz.dim_jn_s.z);
  fprintf(outfile, "    num_kn_s = (%d, %d, %d), dim_kn_s = (%d, %d, %d)\n",
    blocks.Gfz.num_kn_s.x, blocks.Gfz.num_kn_s.y, blocks.Gfz.num_kn_s.z,
    blocks.Gfz.dim_kn_s.x, blocks.Gfz.dim_kn_s.y, blocks.Gfz.dim_kn_s.z);
  /* Ghost grid - Gfz */
  fprintf(outfile, "    num_inb = (%d, %d, %d), dim_inb = (%d, %d, %d)\n",
    blocks.Gfz.num_inb.x, blocks.Gfz.num_inb.y, blocks.Gfz.num_inb.z,
    blocks.Gfz.dim_inb.x, blocks.Gfz.dim_inb.y, blocks.Gfz.dim_inb.z);
  fprintf(outfile, "    num_jnb = (%d, %d, %d), dim_jnb = (%d, %d, %d)\n",
    blocks.Gfz.num_jnb.x, blocks.Gfz.num_jnb.y, blocks.Gfz.num_jnb.z,
    blocks.Gfz.dim_jnb.x, blocks.Gfz.dim_jnb.y, blocks.Gfz.dim_jnb.z);
  fprintf(outfile, "    num_knb = (%d, %d, %d), dim_knb = (%d, %d, %d)\n",
    blocks.Gfz.num_knb.x, blocks.Gfz.num_knb.y, blocks.Gfz.num_knb.z,
    blocks.Gfz.dim_knb.x, blocks.Gfz.dim_knb.y, blocks.Gfz.dim_knb.z);
  fprintf(outfile, "    num_s3b = (%d, %d, %d), dim_s3b = (%d, %d, %d)\n",
    blocks.Gfz.num_s3b.x, blocks.Gfz.num_s3b.y, blocks.Gfz.num_s3b.z,
    blocks.Gfz.dim_s3b.x, blocks.Gfz.dim_s3b.y, blocks.Gfz.dim_s3b.z);
  /* Ghost Shared Grid - Gfz */
  fprintf(outfile, "    num_inb_s = (%d, %d, %d), dim_inb_s = (%d, %d, %d)\n",
    blocks.Gfz.num_inb_s.x, blocks.Gfz.num_inb_s.y, blocks.Gfz.num_inb_s.z,
    blocks.Gfz.dim_inb_s.x, blocks.Gfz.dim_inb_s.y, blocks.Gfz.dim_inb_s.z);
  fprintf(outfile, "    num_jnb_s = (%d, %d, %d), dim_jnb_s = (%d, %d, %d)\n",
    blocks.Gfz.num_jnb_s.x, blocks.Gfz.num_jnb_s.y, blocks.Gfz.num_jnb_s.z,
    blocks.Gfz.dim_jnb_s.x, blocks.Gfz.dim_jnb_s.y, blocks.Gfz.dim_jnb_s.z);
  fprintf(outfile, "    num_knb_s = (%d, %d, %d), dim_knb_s = (%d, %d, %d)\n",
    blocks.Gfz.num_knb_s.x, blocks.Gfz.num_knb_s.y, blocks.Gfz.num_knb_s.z,
    blocks.Gfz.dim_knb_s.x, blocks.Gfz.dim_knb_s.y, blocks.Gfz.dim_knb_s.z);

  fclose(outfile);
}

extern "C"
void cuda_dom_BC(void)
{
  //printf("N%d >> Applying boundary conditions to u_star.\n", rank);
  // Check whether each subdom boundary is an external boundary, then 
  // apply the correct boundary conditions to all fields on that face

  // Only apply boundary conditions on the inner [*n x *n] plane, not the
  //  [*nb x *nb] -- this ensures we don't set the points that don't contain
  //  any solution, and we also don't set points twice

  /* WEST */
  if (dom[rank].w == MPI_PROC_NULL) {
    switch (bc.pW) {
      case NEUMANN:
        BC_p_W_N<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(_p);
        break;
    }

    switch (bc.uW) {
      case DIRICHLET:
        BC_u_W_D<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u, bc.uWD);
        break;
      case NEUMANN:
        BC_u_W_N<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u);
        break;
    }
    
    switch (bc.vW) {
      case DIRICHLET:
        BC_v_W_D<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(_v, bc.vWD);
        break;
      case NEUMANN:
        BC_v_W_N<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(_v);
        break;
    }

    switch (bc.wW) {
      case DIRICHLET:
        BC_w_W_D<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(_w, bc.wWD);
        break;
      case NEUMANN:
        BC_w_W_N<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(_w);
        break;
    }
  }

  /* EAST */
  if (dom[rank].e == MPI_PROC_NULL) {
    switch (bc.pE) {
      case NEUMANN:
        BC_p_E_N<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(_p);
    }

    switch (bc.uE) {
      case DIRICHLET:
        BC_u_E_D<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u, bc.uED);
        break;
      case NEUMANN:
        BC_u_E_N<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u);
        break;
    }
    
    switch (bc.vE) {
      case DIRICHLET:
        BC_v_E_D<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(_v, bc.vED);
        break;
      case NEUMANN:
        BC_v_E_N<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(_v);
        break;
    }

    switch (bc.wE) {
      case DIRICHLET:
        BC_w_E_D<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(_w, bc.wED);
        break;
      case NEUMANN:
        BC_w_E_N<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(_w);
        break;
    }
  }

  /* SOUTH */
  if (dom[rank].s == MPI_PROC_NULL) {
    switch (bc.pS) {
      case NEUMANN:
        BC_p_S_N<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(_p);
    }

    switch (bc.uS) {
      case DIRICHLET:
        BC_u_S_D<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u, bc.uSD);
        break;
      case NEUMANN:
        BC_u_S_N<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u);
        break;
    }
    
    switch (bc.vS) {
      case DIRICHLET:
        BC_v_S_D<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v, bc.vSD);
        break;
      case NEUMANN:
        BC_v_S_N<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v);
        break;
    }

    switch (bc.wS) {
      case DIRICHLET:
        BC_w_S_D<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(_w, bc.wSD);
        break;
      case NEUMANN:
        BC_w_S_N<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(_w);
        break;
    }
  }

  /* NORTH */
  if (dom[rank].n == MPI_PROC_NULL) {
    switch (bc.pN) {
      case NEUMANN:
        BC_p_N_N<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(_p);
    }

    switch (bc.uN) {
      case DIRICHLET:
        BC_u_N_D<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u, bc.uND);
        break;
      case NEUMANN:
        BC_u_N_N<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u);
        break;
    }
    
    switch (bc.vN) {
      case DIRICHLET:
        BC_v_N_D<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v, bc.vND);
        break;
      case NEUMANN:
        BC_v_N_N<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v);
        break;
    }

    switch (bc.wN) {
      case DIRICHLET:
        BC_w_N_D<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(_w, bc.wND);
        break;
      case NEUMANN:
        BC_w_N_N<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(_w);
        break;
    }
  }

  /* BOTTOM */
  if (dom[rank].b == MPI_PROC_NULL) {
    switch (bc.pB) {
      case NEUMANN:
        BC_p_B_N<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p);
    }

    switch (bc.uB) {
      case DIRICHLET:
        BC_u_B_D<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(_u, bc.uBD);
        break;
      case NEUMANN:
        BC_u_B_N<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(_u);
        break;
    }
    
    switch (bc.vB) {
      case DIRICHLET:
        BC_v_B_D<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(_v, bc.vBD);
        break;
      case NEUMANN:
        BC_v_B_N<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(_v);
        break;
    }

    switch (bc.wB) {
      case DIRICHLET:
        BC_w_B_D<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w, bc.wBD);
        break;
      case NEUMANN:
        BC_w_B_N<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w);
        break;
    }
  }

  /* TOP */
  if (dom[rank].t == MPI_PROC_NULL) {
    switch (bc.pT) {
      case NEUMANN:
        BC_p_T_N<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p);
    }

    switch (bc.uT) {
      case DIRICHLET:
        BC_u_T_D<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(_u, bc.uTD);
        break;
      case NEUMANN:
        BC_u_T_N<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(_u);
        break;
    }
    
    switch (bc.vT) {
      case DIRICHLET:
        BC_v_T_D<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(_v, bc.vTD);
        break;
      case NEUMANN:
        BC_v_T_N<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(_v);
        break;
    }

    switch (bc.wT) {
      case DIRICHLET:
        BC_w_T_D<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w, bc.wTD);
        break;
      case NEUMANN:
        BC_w_T_N<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w);
        break;
    }
  }
}

extern "C"
void cuda_dom_pull(void)
{
  // Pull domain data from device to host
  checkCudaErrors(cudaMemcpy(p, _p, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(u, _u, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));   
  checkCudaErrors(cudaMemcpy(v, _v, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));   
  checkCudaErrors(cudaMemcpy(w, _w, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_dom_pull_phase(void)
{
  checkCudaErrors(cudaMemcpy(phase, _phase, dom[rank].Gcc.s3b * sizeof(int),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(phase_shell, _phase_shell, dom[rank].Gcc.s3b * sizeof(int),
    cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_dom_pull_debug(void)
{
  //printf("N%d >> Pulling dom device->host (debug)\n", rank);

  checkCudaErrors(cudaMemcpy(phi, _phi, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpy(u_star, _u_star, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));        
  checkCudaErrors(cudaMemcpy(v_star, _v_star, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));        
  checkCudaErrors(cudaMemcpy(w_star, _w_star, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpy(conv_u, _conv_u, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));   
  checkCudaErrors(cudaMemcpy(conv_v, _conv_v, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));   
  checkCudaErrors(cudaMemcpy(conv_w, _conv_w, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpy(diff_u, _diff_u, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(diff_v, _diff_v, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(diff_w, _diff_w, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpy(flag_u, _flag_u, dom[rank].Gfx.s3b * sizeof(int),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(flag_v, _flag_v, dom[rank].Gfy.s3b * sizeof(int),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(flag_w, _flag_w, dom[rank].Gfz.s3b * sizeof(int),
    cudaMemcpyDeviceToHost));

  //if (NPARTS > 0) { // Already pulled in cuda_dom_pull()
  //  checkCudaErrors(cudaMemcpy(phase, _phase, dom[rank].Gcc.s3b * sizeof(int),
  //    cudaMemcpyDeviceToHost));
  //  checkCudaErrors(cudaMemcpy(phase_shell, _phase_shell, dom[rank].Gcc.s3b * sizeof(int),
  //    cudaMemcpyDeviceToHost));
  //}
}

extern "C"
void cuda_dom_pull_restart(void) {
  checkCudaErrors(cudaMemcpy(p0, _p0, dom[rank].Gcc.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(u0, _u0, dom[rank].Gfx.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(v0, _v0, dom[rank].Gfy.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(w0, _w0, dom[rank].Gfz.s3b * sizeof(real),
    cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpy(conv0_u, _conv0_u, dom[rank].Gfx.s3b *sizeof(real),
    cudaMemcpyDeviceToHost));   
  checkCudaErrors(cudaMemcpy(conv0_v, _conv0_v, dom[rank].Gfy.s3b *sizeof(real),
    cudaMemcpyDeviceToHost));   
  checkCudaErrors(cudaMemcpy(conv0_w, _conv0_w, dom[rank].Gfz.s3b *sizeof(real),
    cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpy(diff0_u, _diff0_u, dom[rank].Gfx.s3b *sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(diff0_v, _diff0_v, dom[rank].Gfy.s3b *sizeof(real),
    cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(diff0_w, _diff0_w, dom[rank].Gfz.s3b *sizeof(real),
    cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_self_exchange_i(real *array)
{
  self_exchange_Gcc_i<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array);
}

extern "C"
void cuda_self_exchange_j(real *array)
{
  self_exchange_Gcc_j<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array);
}

extern "C"
void cuda_self_exchange_k(real *array)
{
  self_exchange_Gcc_k<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array);
}

extern "C" 
void cuda_pack_planes_Gcc(real *array)
{
  if (dom[rank].e != MPI_PROC_NULL) 
    pack_planes_Gcc_east<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array,
      _send_Gcc_e);
  if (dom[rank].w != MPI_PROC_NULL)
    pack_planes_Gcc_west<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array, 
      _send_Gcc_w);

  if (dom[rank].n != MPI_PROC_NULL)
    pack_planes_Gcc_north<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array,
      _send_Gcc_n);
  if (dom[rank].s != MPI_PROC_NULL)
    pack_planes_Gcc_south<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array, 
      _send_Gcc_s);

  if (dom[rank].t != MPI_PROC_NULL)
    pack_planes_Gcc_top<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array,
      _send_Gcc_t);
  if (dom[rank].b != MPI_PROC_NULL)
    pack_planes_Gcc_bottom<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array, 
      _send_Gcc_b);
}

extern "C" 
void cuda_pack_planes_Gfx(real *array)
{
  if (dom[rank].e != MPI_PROC_NULL) 
    pack_planes_Gfx_east<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(array,
      _send_Gfx_e);
  if (dom[rank].w != MPI_PROC_NULL)
    pack_planes_Gfx_west<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(array, 
      _send_Gfx_w);

  if (dom[rank].n != MPI_PROC_NULL)
    pack_planes_Gfx_north<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(array,
      _send_Gfx_n);
  if (dom[rank].s != MPI_PROC_NULL)
    pack_planes_Gfx_south<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(array, 
      _send_Gfx_s);

  if (dom[rank].t != MPI_PROC_NULL)
    pack_planes_Gfx_top<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(array,
      _send_Gfx_t);
  if (dom[rank].b != MPI_PROC_NULL)
    pack_planes_Gfx_bottom<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(array, 
      _send_Gfx_b);
}

extern "C" 
void cuda_pack_planes_Gfy(real *array)
{
  if (dom[rank].e != MPI_PROC_NULL) 
    
    pack_planes_Gfy_east<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(array,
      _send_Gfy_e);
  if (dom[rank].w != MPI_PROC_NULL)
    pack_planes_Gfy_west<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(array, 
      _send_Gfy_w);

  if (dom[rank].n != MPI_PROC_NULL)
    pack_planes_Gfy_north<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(array,
      _send_Gfy_n);
  if (dom[rank].s != MPI_PROC_NULL)
    pack_planes_Gfy_south<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(array, 
      _send_Gfy_s);

  if (dom[rank].t != MPI_PROC_NULL)
    pack_planes_Gfy_top<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(array,
      _send_Gfy_t);
  if (dom[rank].b != MPI_PROC_NULL)
    pack_planes_Gfy_bottom<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(array, 
      _send_Gfy_b);
}

extern "C" 
void cuda_pack_planes_Gfz(real *array)
{
  if (dom[rank].e != MPI_PROC_NULL) 
    pack_planes_Gfz_east<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(array,
      _send_Gfz_e);
  if (dom[rank].w != MPI_PROC_NULL)
    pack_planes_Gfz_west<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(array, 
      _send_Gfz_w);

  if (dom[rank].n != MPI_PROC_NULL)
    pack_planes_Gfz_north<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(array,
      _send_Gfz_n);
  if (dom[rank].s != MPI_PROC_NULL)
    pack_planes_Gfz_south<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(array, 
      _send_Gfz_s);

  if (dom[rank].t != MPI_PROC_NULL)
    pack_planes_Gfz_top<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(array,
      _send_Gfz_t);
  if (dom[rank].b != MPI_PROC_NULL)
    pack_planes_Gfz_bottom<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(array, 
      _send_Gfz_b);
}

extern "C"
void cuda_unpack_planes_Gcc(real *array)
{
  if (dom[rank].e != MPI_PROC_NULL)
    unpack_planes_Gcc_east<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array, 
      _recv_Gcc_e);
  if (dom[rank].w != MPI_PROC_NULL)
    unpack_planes_Gcc_west<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array, 
      _recv_Gcc_w);

  if (dom[rank].n != MPI_PROC_NULL)
    unpack_planes_Gcc_north<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array, 
      _recv_Gcc_n);
  if (dom[rank].s != MPI_PROC_NULL)
    unpack_planes_Gcc_south<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array, 
      _recv_Gcc_s);

  if (dom[rank].t != MPI_PROC_NULL)
    unpack_planes_Gcc_top<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array, 
      _recv_Gcc_t);
  if (dom[rank].b != MPI_PROC_NULL)
    unpack_planes_Gcc_bottom<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array, 
      _recv_Gcc_b);
}

extern "C"
void cuda_unpack_planes_Gfx(real *array)
{
  if (dom[rank].e != MPI_PROC_NULL)
    unpack_planes_Gfx_east<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(array, 
      _recv_Gfx_e);
  if (dom[rank].w != MPI_PROC_NULL)
    unpack_planes_Gfx_west<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(array, 
      _recv_Gfx_w);

  if (dom[rank].n != MPI_PROC_NULL)
    unpack_planes_Gfx_north<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(array, 
      _recv_Gfx_n);
  if (dom[rank].s != MPI_PROC_NULL)
    unpack_planes_Gfx_south<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(array, 
      _recv_Gfx_s);

  if (dom[rank].t != MPI_PROC_NULL)
    unpack_planes_Gfx_top<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(array, 
      _recv_Gfx_t);
  if (dom[rank].b != MPI_PROC_NULL)
    unpack_planes_Gfx_bottom<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(array, 
      _recv_Gfx_b);
}

extern "C"
void cuda_unpack_planes_Gfy(real *array)
{
  if (dom[rank].e != MPI_PROC_NULL)
    unpack_planes_Gfy_east<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(array, 
      _recv_Gfy_e);
  if (dom[rank].w != MPI_PROC_NULL)
    unpack_planes_Gfy_west<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(array, 
      _recv_Gfy_w);

  if (dom[rank].n != MPI_PROC_NULL)
    unpack_planes_Gfy_north<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(array, 
      _recv_Gfy_n);
  if (dom[rank].s != MPI_PROC_NULL)
    unpack_planes_Gfy_south<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(array, 
      _recv_Gfy_s);

  if (dom[rank].t != MPI_PROC_NULL)
    unpack_planes_Gfy_top<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(array, 
      _recv_Gfy_t);
  if (dom[rank].b != MPI_PROC_NULL)
    unpack_planes_Gfy_bottom<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(array, 
      _recv_Gfy_b);
}

extern "C"
void cuda_unpack_planes_Gfz(real *array)
{
  if (dom[rank].e != MPI_PROC_NULL)
    unpack_planes_Gfz_east<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(array, 
      _recv_Gfz_e);
  if (dom[rank].w != MPI_PROC_NULL)
    unpack_planes_Gfz_west<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(array, 
      _recv_Gfz_w);

  if (dom[rank].n != MPI_PROC_NULL)
    unpack_planes_Gfz_north<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(array, 
      _recv_Gfz_n);
  if (dom[rank].s != MPI_PROC_NULL)
    unpack_planes_Gfz_south<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(array, 
      _recv_Gfz_s);

  if (dom[rank].t != MPI_PROC_NULL)
    unpack_planes_Gfz_top<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(array, 
      _recv_Gfz_t);
  if (dom[rank].b != MPI_PROC_NULL)
    unpack_planes_Gfz_bottom<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(array, 
      _recv_Gfz_b);
}

extern "C"
void cuda_find_dt(void)
{
  // Only want max values over the computational domain, not ghost domain
  // Copy to new array and square the value. Then find max of that result
  // and take sqrt
  real *utmp;
  real *vtmp;
  real *wtmp;
  cudaMalloc((void**) &utmp, sizeof(real)*dom[rank].Gfx.s3);
  cudaMalloc((void**) &vtmp, sizeof(real)*dom[rank].Gfy.s3);
  cudaMalloc((void**) &wtmp, sizeof(real)*dom[rank].Gfz.s3);

  copy_u_square_noghost<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u, utmp);
  copy_v_square_noghost<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v, vtmp);
  copy_w_square_noghost<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w, wtmp);

  // device pointers to utmp, vtmp, wtmp
  thrust::device_ptr<real> t_umax(utmp);
  thrust::device_ptr<real> t_vmax(vtmp);
  thrust::device_ptr<real> t_wmax(wtmp);

  real u_max = thrust::reduce(t_umax, t_umax + dom[rank].Gfx.s3, 0., 
                              thrust::maximum<real>());
  real v_max = thrust::reduce(t_vmax, t_vmax + dom[rank].Gfy.s3, 0., 
                              thrust::maximum<real>());
  real w_max = thrust::reduce(t_wmax, t_wmax + dom[rank].Gfz.s3, 0., 
                              thrust::maximum<real>());

  u_max = sqrt(u_max);
  v_max = sqrt(v_max);
  w_max = sqrt(w_max);

  cudaFree(utmp);
  cudaFree(vtmp);
  cudaFree(wtmp);

  // find dt on each subdomain
  if(SCALAR >= 1 && s_D > nu) {
    dt = u_max/dom[rank].dx + 2.*s_D/(dom[rank].dx * dom[rank].dx);
    dt += v_max/dom[rank].dy + 2.*s_D/(dom[rank].dy * dom[rank].dy);
    dt += w_max/dom[rank].dz + 2.*s_D/(dom[rank].dz * dom[rank].dz);
  } else {
    dt = u_max/dom[rank].dx + 2.*nu/(dom[rank].dx * dom[rank].dx);
    dt += v_max/dom[rank].dy + 2.*nu/(dom[rank].dy * dom[rank].dy);
    dt += w_max/dom[rank].dz + 2.*nu/(dom[rank].dz * dom[rank].dz);
  }
  dt = CFL/dt;

  // MPI reduce to find minimum timestep over all ranks
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, mpi_real, MPI_MIN, MPI_COMM_WORLD);


  /* An alternative method is to find max(u,v,w) over all domains, and then 
   *  calculate dt. This will be <= the dt as it is currently calculated.
   */
}

extern "C"
void cuda_compute_forcing(void)
{
  // reset forcing arrays
  forcing_reset_x<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(_f_x);
  forcing_reset_y<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(_f_y);
  forcing_reset_z<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(_f_z);

  // linearly accelerate pressure gradient from zero
  real delta = ttime - p_bc_tdelay;
  if (delta >= 0) {
    if (gradP.xa == 0) {
      gradP.x = gradP.xm;
    } else if (fabs(delta*gradP.xa) > fabs(gradP.xm)) {
      gradP.x = gradP.xm;
    } else {
      gradP.x = delta*gradP.xa;
    }

    if (gradP.ya == 0) {
      gradP.y = gradP.ym;
    } else if (fabs(delta*gradP.ya) > fabs(gradP.ym)) {
      gradP.y = gradP.ym;
    } else {
      gradP.y = delta*gradP.ya;
    }

    // Turn off if PID controller is on
    if (!(Kp > 0 || Ki > 0 || Kd > 0)) {
      if (gradP.za == 0) {
        gradP.z = gradP.zm;
      } else if (fabs(delta*gradP.za) > fabs(gradP.zm)) {
        gradP.z = gradP.zm;
      } else {
        gradP.z = delta*gradP.za;
      }
    }
  }
  gradP.z = gradP.z * cos(osci_f*ttime);

  // linearly accelerate gravitational acceleration from zero
  delta = ttime - g_bc_tdelay;
  if (delta >= 0) {
    if (g.xa == 0) {
      g.x = g.xm;
    } else if (fabs(delta*g.xa) > fabs(g.xm)) {
      g.x = g.xm;
    } else {
      g.x = delta*g.xa;
    }

    if (g.ya == 0) {
      g.y = g.ym;
    } else if (fabs(delta*g.ya) > fabs(g.ym)) {
      g.y = g.ym;
    } else {
      g.y = delta*g.ya;
    }

    if (g.za == 0) {
      g.z = g.zm;
    } else if (fabs(delta*g.za) > fabs(g.zm)) {
      g.z = g.zm;
    } else {
      g.z = delta*g.za;
    }
  }

  delta = ttime - p_bc_tdelay;
  // PID controller  
  if (delta >= 0) {
    if(Kp > 0 || Ki > 0 || Kd > 0) {

      /* Init execution config */
      // Ghost cells
      int ty = bins.Gcc.jnb * (bins.Gcc.jnb < MAX_THREADS_DIM)
           + MAX_THREADS_DIM * (bins.Gcc.jnb >= MAX_THREADS_DIM);
      int tz = bins.Gcc.knb * (bins.Gcc.knb < MAX_THREADS_DIM)
           + MAX_THREADS_DIM * (bins.Gcc.knb >= MAX_THREADS_DIM);

      int by = (int) ceil((real) bins.Gcc.jnb / (real) ty);
      int bz = (int) ceil((real) bins.Gcc.knb / (real) tz);

      dim3 bin_num_inb(by, bz);
      dim3 bin_dim_inb(ty, tz);

      // No ghost
      ty = bins.Gcc.jn * (bins.Gcc.jn < MAX_THREADS_DIM)
           + MAX_THREADS_DIM * (bins.Gcc.jn >= MAX_THREADS_DIM);
      tz = bins.Gcc.kn * (bins.Gcc.kn < MAX_THREADS_DIM)
           + MAX_THREADS_DIM * (bins.Gcc.kn >= MAX_THREADS_DIM);

      by = (int) ceil((real) bins.Gcc.jn / (real) ty);
      bz = (int) ceil((real) bins.Gcc.kn / (real) tz);

      dim3 bin_num_in(by, bz);
      dim3 bin_dim_in(ty, tz);

      // Thread over nparts
      int t_nparts = nparts * (nparts < MAX_THREADS_1D)
                    + MAX_THREADS_1D * (nparts >= MAX_THREADS_1D);
      int b_nparts = (int) ceil((real) nparts / (real) t_nparts);

      dim3 dim_nparts(t_nparts);
      dim3 num_nparts(b_nparts);

      /* Allocate memory */
      checkCudaErrors(cudaMalloc(&_part_ind, nparts * sizeof(int)));
      checkCudaErrors(cudaMalloc(&_part_bin, nparts * sizeof(int)));
      thrust::device_ptr<int> t_part_ind(_part_ind);
      thrust::device_ptr<int> t_part_bin(_part_bin);

      checkCudaErrors(cudaMemset(_bin_start, -1, bins.Gcc.s3b * sizeof(int)));
      checkCudaErrors(cudaMemset(_bin_end, -1, bins.Gcc.s3b * sizeof(int)));
      checkCudaErrors(cudaMemset(_bin_count, 0, bins.Gcc.s3b * sizeof(int)));
      thrust::device_ptr<int> t_bin_count(_bin_count);

      real *_wdot;
      checkCudaErrors(cudaMalloc(&_wdot, bins.Gcc.s3 * sizeof(real)));
      thrust::device_ptr<real> t_wdot(_wdot);

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

        /* Pull wdot to an array for each bin */
        pull_wdot<<<bin_num_in, bin_dim_in>>>(_wdot, _parts, _bin_start,
          _bin_count, _part_ind);

      } else { // nparts <= 0
        checkCudaErrors(cudaMemset(_part_ind, -1, nparts * sizeof(int)));
        checkCudaErrors(cudaMemset(_part_bin, -1, nparts * sizeof(int)));
        checkCudaErrors(cudaMemset(_wdot, 0., nparts * sizeof(real)));
      }

      real acc_z = thrust::reduce(t_wdot, t_wdot + bins.Gcc.s3, 0., thrust::plus<real>());
      MPI_Allreduce(MPI_IN_PLACE, &acc_z, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
      acc_z /= (real) NPARTS;

      pid_int = pid_int + acc_z*dt;
      gradP.z = gradP.z
        + (Kp*acc_z + Ki*pid_int/ttime + Kd*(acc_z-pid_back))*rho_avg;
      pid_back = acc_z;

      checkCudaErrors(cudaFree(_wdot));
      checkCudaErrors(cudaFree(_part_ind));
      checkCudaErrors(cudaFree(_part_bin));
    }
  }

  // forcing
  forcing_add_x_const<<<blocks.Gfx.num_inb,blocks.Gfx.dim_inb>>>(-gradP.x/rho_f,
    _f_x);
  forcing_add_y_const<<<blocks.Gfy.num_jnb,blocks.Gfy.dim_jnb>>>(-gradP.y/rho_f,
    _f_y);
  forcing_add_z_const<<<blocks.Gfz.num_knb,blocks.Gfz.dim_knb>>>(-gradP.z/rho_f,
    _f_z);
}

extern "C"
void cuda_compute_turb_forcing(void)
{
  if (init_cond == TURBULENT) {
    /* Calculate current kinetic energy */
    real *utmp;
    real *vtmp;
    real *wtmp;
    cudaMalloc((void**) &utmp, sizeof(real)*dom[rank].Gfx.s3);
    cudaMalloc((void**) &vtmp, sizeof(real)*dom[rank].Gfy.s3);
    cudaMalloc((void**) &wtmp, sizeof(real)*dom[rank].Gfz.s3);

    // Square entries
    copy_u_square_noghost<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u, utmp);
    copy_v_square_noghost<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v, vtmp);
    copy_w_square_noghost<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w, wtmp);

    // device pointers to utmp, vtmp, wtmp
    thrust::device_ptr<real> t_utmp(utmp);
    thrust::device_ptr<real> t_vtmp(vtmp);
    thrust::device_ptr<real> t_wtmp(wtmp);

    // Sum fields
    // -- Sum should not double count staggered velocities at subdomain and
    //     periodic interfaces. For now, just don't loop over those points.
    // -- This assumes then that we're using periodic boundary conditions
    real su2 = thrust::reduce(t_utmp, t_utmp + dom[rank].Gfx.s3 - dom[rank].Gfx.s2_i,
                                0., thrust::plus<real>());
    real sv2 = thrust::reduce(t_vtmp, t_vtmp + dom[rank].Gfy.s3 - dom[rank].Gfy.s2_j,
                                0., thrust::plus<real>());
    real sw2 = thrust::reduce(t_wtmp, t_wtmp + dom[rank].Gfz.s3 - dom[rank].Gfz.s2_k,
                                0., thrust::plus<real>());

    // Sum results
    real k = 0.5 * (su2 + sv2 + sw2);

    // Find total energy
    MPI_Allreduce(MPI_IN_PLACE, &k, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

    // Average
    k /= DOM.Gcc.s3;

    /* Find mean u,v,w velocity */
    // Copy to array with no ghost cells
    copy_u_noghost<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u, utmp);
    copy_v_noghost<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v, vtmp);
    copy_w_noghost<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w, wtmp);

    // Sum
    // Sum should not double count staggered vels at bouundaries
    real umean = thrust::reduce(t_utmp, t_utmp + dom[rank].Gfx.s3 - dom[rank].Gfx.s2_i,
                                0., thrust::plus<real>());
    real vmean = thrust::reduce(t_vtmp, t_vtmp + dom[rank].Gfy.s3 - dom[rank].Gfy.s2_j,
                                0., thrust::plus<real>());
    real wmean = thrust::reduce(t_wtmp, t_wtmp + dom[rank].Gfz.s3 - dom[rank].Gfz.s2_k,
                                0., thrust::plus<real>());
    
    // Reduce over all ranks
    MPI_Allreduce(MPI_IN_PLACE, &umean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vmean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &wmean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

    // Average
    umean /= DOM.Gfx.s3;
    vmean /= DOM.Gfy.s3;
    wmean /= DOM.Gfz.s3;

    // Calculate forcing
    real turb_force = turbA * turb_k0 / k;

    // Add forcing to velocity field
    forcing_add_x_field<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(turb_force,
      _u, _f_x);
    forcing_add_y_field<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(turb_force,
      _v, _f_y);
    forcing_add_z_field<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(turb_force,
      _w, _f_z);

    // Subtract mean to get perturbation
    forcing_add_x_const<<<blocks.Gfx.num_inb, blocks.Gfx.dim_inb>>>(-turb_force*umean,
      _f_x);
    forcing_add_y_const<<<blocks.Gfy.num_jnb, blocks.Gfy.dim_jnb>>>(-turb_force*vmean,
      _f_y);
    forcing_add_z_const<<<blocks.Gfz.num_knb, blocks.Gfz.dim_knb>>>(-turb_force*wmean,
      _f_z);

    // Free
    cudaFree(utmp);
    cudaFree(vtmp);
    cudaFree(wtmp);

    /* Dissipation Rate */
    real *_eps;
    cudaMalloc((void**) &_eps, sizeof(real) * dom[rank].Gcc.s3);
    calc_dissipation<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(_u, _v, _w, _eps);

    thrust::device_ptr<real> t_eps(_eps);
    real eps = thrust::reduce(t_eps, t_eps + dom[rank].Gcc.s3,
                                0., thrust::plus<real>());
    MPI_Allreduce(MPI_IN_PLACE, &eps, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
    eps *= nu / DOM.Gcc.s3;

    cudaFree(_eps);

    // Record this
    char rname[FILE_NAME_SIZE] = "turb.rec";
    recorder_turb(rname, k, eps);
  }
}

extern "C"
void cuda_U_star(void)
{
  calc_u_star<<<blocks.Gfx.num_inb_s, blocks.Gfx.dim_inb_s>>>(rho_f, nu, _u0,
    _v0, _w0, _p0, _f_x, _diff0_u, _conv0_u, _diff_u, _conv_u, _u_star, dt0,
    dt, _phase);

  calc_v_star<<<blocks.Gfy.num_jnb_s, blocks.Gfy.dim_jnb_s>>>(rho_f, nu,  _u0,
    _v0, _w0, _p0, _f_y, _diff0_v, _conv0_v, _diff_v, _conv_v, _v_star, dt0,
    dt, _phase);

  calc_w_star<<<blocks.Gfy.num_knb_s, blocks.Gfz.dim_knb_s>>>(rho_f, nu,  _u0,
    _v0, _w0, _p0, _f_z, _diff0_w, _conv0_w, _diff_w, _conv_w, _w_star, dt0,
    dt, _phase);
}

extern "C"
void cuda_dom_BC_star(void)
{
  // west
  if (dom[rank].w == MPI_PROC_NULL) {
    // u
    switch (bc.uW) {
      case DIRICHLET:
        BC_u_W_D<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, bc.uWD);
        break;
      case NEUMANN:
        BC_u_W_N<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star);
        break;
    }
    
    // v
    switch (bc.vW) {
      case DIRICHLET:
        BC_v_W_D<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(_v_star, bc.vWD);
        break;
      case NEUMANN:
        BC_v_W_N<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(_v_star);
        break;
    }

    // w
    switch (bc.wW) {
      case DIRICHLET:
        BC_w_W_D<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(_w_star, bc.wWD);
        break;
      case NEUMANN:
        BC_w_W_N<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(_w_star);
        break;
    }
  }

  // east
  if (dom[rank].e == MPI_PROC_NULL) {
    // u
    switch (bc.uE) {
      case DIRICHLET:
        BC_u_E_D<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, bc.uED);
        break;
      case NEUMANN:
        BC_u_E_N<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star);
        break;
    }
    
    // v
    switch (bc.vE) {
      case DIRICHLET:
        BC_v_E_D<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(_v_star, bc.vED);
        break;
      case NEUMANN:
        BC_v_E_N<<<blocks.Gfy.num_in, blocks.Gfy.dim_in>>>(_v_star);
        break;
    }

    // w
    switch (bc.wE) {
      case DIRICHLET:
        BC_w_E_D<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(_w_star, bc.wED);
        break;
      case NEUMANN:
        BC_w_E_N<<<blocks.Gfz.num_in, blocks.Gfz.dim_in>>>(_w_star);
        break;
    }
  }

  // south
  if (dom[rank].s == MPI_PROC_NULL) {
    // u
    switch (bc.uS) {
      case DIRICHLET:
        BC_u_S_D<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u_star, bc.uSD);
        break;
      case NEUMANN:
        BC_u_S_N<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u_star);
        break;
    }
    
    // v
    switch (bc.vS) {
      case DIRICHLET:
        BC_v_S_D<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, bc.vSD);
        break;
      case NEUMANN:
        BC_v_S_N<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star);
        break;
    }

    // w
    switch (bc.wS) {
      case DIRICHLET:
        BC_w_S_D<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(_w_star, bc.wSD);
        break;
      case NEUMANN:
        BC_w_S_N<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(_w_star);
        break;
    }
  }

  // north
  if (dom[rank].n == MPI_PROC_NULL) {
    // u
    switch (bc.uN) {
      case DIRICHLET:
        BC_u_N_D<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u_star, bc.uND);
        break;
      case NEUMANN:
        BC_u_N_N<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u_star);
        break;
    }
    
    // v
    switch (bc.vN) {
      case DIRICHLET:
        BC_v_N_D<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, bc.vND);
        break;
      case NEUMANN:
        BC_v_N_N<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star);
        break;
    }

    // w
    switch (bc.wN) {
      case DIRICHLET:
        BC_w_N_D<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(_w_star, bc.wND);
        break;
      case NEUMANN:
        BC_w_N_N<<<blocks.Gfz.num_jn, blocks.Gfz.dim_jn>>>(_w_star);
        break;
    }
  }

  // bottom
  if (dom[rank].b == MPI_PROC_NULL) {
    // u
    switch (bc.uB) {
      case DIRICHLET:
        BC_u_B_D<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(_u_star, bc.uBD);
        break;
      case NEUMANN:
        BC_u_B_N<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(_u_star);
        break;
    }
    
    // v
    switch (bc.vB) {
      case DIRICHLET:
        BC_v_B_D<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(_v_star, bc.vBD);
        break;
      case NEUMANN:
        BC_v_B_N<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(_v_star);
        break;
    }

    // w
    switch (bc.wB) {
      case DIRICHLET:
        BC_w_B_D<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, bc.wBD);
        break;
      case NEUMANN:
        BC_w_B_N<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star);
        break;
    }
  }

  // top
  if (dom[rank].t == MPI_PROC_NULL) {
    // u
    switch (bc.uT) {
      case DIRICHLET:
        BC_u_T_D<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(_u_star, bc.uTD);
        break;
      case NEUMANN:
        BC_u_T_N<<<blocks.Gfx.num_kn, blocks.Gfx.dim_kn>>>(_u_star);
        break;
    }
    
    // v
    switch (bc.vT) {
      case DIRICHLET:
        BC_v_T_D<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(_v_star, bc.vTD);
        break;
      case NEUMANN:
        BC_v_T_N<<<blocks.Gfy.num_kn, blocks.Gfy.dim_kn>>>(_v_star);
        break;
    }

    // w
    switch (bc.wT) {
      case DIRICHLET:
        BC_w_T_D<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, bc.wTD);
        break;
      case NEUMANN:
        BC_w_T_N<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star);
        break;
    }
  }
}

extern "C"
void cuda_solvability(void)
{
  //printf("N%d >> Enforcing solvability...\n", rank);
  /* Calculate difference from zero on each domain, then MPI_Allreduce that
   *  value. It would be better to define an MPI_COMM for the edge cells,
   *  but that's an optimization.
   */

  /* Differences from zero on each plane */
  real eps_xs = 0.;
  real eps_xe = 0.;                
  real eps_ys = 0.;                
  real eps_ye = 0.;                
  real eps_zs = 0.;                
  real eps_ze = 0.;                
  real eps[3];      // [x, y, z]

  // local reduction, then global reduction
  if (dom[rank].I == DOM.Is) {
    /* Temporary storage for reduction */
    real *u_star_tmp;
    cudaMalloc((void**) &u_star_tmp, dom[rank].Gfx.s2_i * sizeof(real));

    /* Calculate x-face integral (is) */
    surf_int_xs<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, u_star_tmp);

    /* Reduction */
    thrust::device_ptr<real> t_us_tmp(u_star_tmp);
    eps_xs = thrust::reduce(t_us_tmp, t_us_tmp + dom[rank].Gfx.s2_i, 0.,
                thrust::plus<real>());
    eps_xs *= dom[rank].dy * dom[rank].dz;                

    /* clean up */
    cudaFree(u_star_tmp);
  }
  if (dom[rank].I == DOM.Ie) {
    real *u_star_tmp;
    cudaMalloc((void**) &u_star_tmp, dom[rank].Gfx.s2_i * sizeof(real));

    surf_int_xe<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, u_star_tmp);

    thrust::device_ptr<real> t_us_tmp(u_star_tmp);
    eps_xe = thrust::reduce(t_us_tmp, t_us_tmp + dom[rank].Gfx.s2_i, 0.,
                thrust::plus<real>());
    eps_xe *= dom[rank].dy * dom[rank].dz;                

    cudaFree(u_star_tmp);
  }
  if (dom[rank].J == DOM.Js) {
    real *v_star_tmp;
    cudaMalloc((void**) &v_star_tmp, dom[rank].Gfy.s2_j * sizeof(real));
     
    surf_int_ys<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, v_star_tmp);

    thrust::device_ptr<real> t_vs_tmp(v_star_tmp);
    eps_ys = thrust::reduce(t_vs_tmp, t_vs_tmp + dom[rank].Gfy.s2_j, 0.,
                thrust::plus<real>());
    eps_ys *= dom[rank].dz * dom[rank].dx;                

    cudaFree(v_star_tmp);
  }
  if (dom[rank].J == DOM.Je) {
    real *v_star_tmp;
    cudaMalloc((void**) &v_star_tmp, dom[rank].Gfy.s2_j * sizeof(real));
     
    surf_int_ye<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, v_star_tmp);

    thrust::device_ptr<real> t_vs_tmp(v_star_tmp);
    eps_ye = thrust::reduce(t_vs_tmp, t_vs_tmp + dom[rank].Gfy.s2_j, 0.,
                thrust::plus<real>());
    eps_ye *= dom[rank].dz * dom[rank].dx;                

    cudaFree(v_star_tmp);
  }
  if (dom[rank].K == DOM.Ks) {
    real *w_star_tmp;
    cudaMalloc((void**) &w_star_tmp, dom[rank].Gfz.s2_k * sizeof(real));
     
    surf_int_zs<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, w_star_tmp);

    thrust::device_ptr<real> t_ws_tmp(w_star_tmp);
    eps_zs = thrust::reduce(t_ws_tmp, t_ws_tmp + dom[rank].Gfz.s2_k, 0.,
                thrust::plus<real>());
    eps_zs *= dom[rank].dx * dom[rank].dy;                

    cudaFree(w_star_tmp);
  }
  if (dom[rank].K == DOM.Ke) {
    real *w_star_tmp;
    cudaMalloc((void**) &w_star_tmp, dom[rank].Gfz.s2_k * sizeof(real));
     
    surf_int_ze<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, w_star_tmp);

    thrust::device_ptr<real> t_ws_tmp(w_star_tmp);
    eps_ze = thrust::reduce(t_ws_tmp, t_ws_tmp + dom[rank].Gfz.s2_k, 0.,
                thrust::plus<real>());
    eps_ze *= dom[rank].dx * dom[rank].dy;                

    cudaFree(w_star_tmp);
  }

  /* Find difference in each direction */
  eps[0] = eps_xe - eps_xs;
  eps[1] = eps_ye - eps_ys;
  eps[2] = eps_ze - eps_zs;

  /* MPI_Allreduce */
  MPI_Allreduce(MPI_IN_PLACE, &eps, 3, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  /* subtract eps from outflow plane */
  real sum;
  switch (out_plane) {
    case WEST:
      if (dom[rank].I == DOM.Is) {
        sum = (eps[0] + eps[1] + eps[2])/(DOM.yl * DOM.zl);
        plane_eps_x_W<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, sum);
      }

      break;
    case EAST:
      if (dom[rank].I == DOM.Ie) {
        sum = (eps[0] + eps[1] + eps[2])/(DOM.yl * DOM.zl);
        plane_eps_x_E<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, sum);
      }

      break;

    case SOUTH:
      if (dom[rank].J == DOM.Js) {
        sum = (eps[0] + eps[1] + eps[2])/(DOM.zl * DOM.xl);
        plane_eps_y_S<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, sum);
      }

      break;
    case NORTH:
      if (dom[rank].J == DOM.Je) {
        sum = (eps[0] + eps[1] + eps[2])/(DOM.zl * DOM.xl);
        plane_eps_y_N<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, sum);
      }

      break;

    case BOTTOM:
      if (dom[rank].K == DOM.Ks) {
        sum = (eps[0] + eps[1] + eps[2])/(DOM.xl * DOM.yl);
        plane_eps_z_B<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, sum);
      }

      break;
    case TOP:
      if (dom[rank].K == DOM.Ke) {
        sum = (eps[0] + eps[1] + eps[2])/(DOM.xl * DOM.yl);
        plane_eps_z_T<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, sum);
      }

      break;
    case HOMOGENEOUS:
      // spread over entire domain
      real sum_x = 0.5*eps[0]/(DOM.yl * DOM.zl);
      real sum_y = 0.5*eps[1]/(DOM.zl * DOM.xl);
      real sum_z = 0.5*eps[2]/(DOM.xl * DOM.yl);

      if (dom[rank].I == DOM.Is)
        plane_eps_x_W<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, sum_x);
      if (dom[rank].I == DOM.Ie)
        plane_eps_x_E<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, sum_x);

      if (dom[rank].J == DOM.Js)
        plane_eps_y_S<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, sum_y);
      if (dom[rank].J == DOM.Je)
        plane_eps_y_N<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, sum_y);

      if (dom[rank].K == DOM.Ks)
        plane_eps_z_B<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, sum_z);
      if (dom[rank].K == DOM.Ke)
        plane_eps_z_T<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, sum_z);

      break;
  }
}

extern "C"
void cuda_project(void)
{
  project_u<<<blocks.Gfx.num_in, blocks.Gfx.dim_in>>>(_u_star, _phi, rho_f, dt,
    _u, 1. / dom[rank].dx, _flag_u);
  project_v<<<blocks.Gfy.num_jn, blocks.Gfy.dim_jn>>>(_v_star, _phi, rho_f, dt,
    _v, 1. / dom[rank].dy, _flag_v);
  project_w<<<blocks.Gfz.num_kn, blocks.Gfz.dim_kn>>>(_w_star, _phi, rho_f, dt,
    _w, 1. / dom[rank].dz, _flag_w);
}

extern "C"
void cuda_update_p()
{
  /* Calculate laplacian of phi and update */
  real *_Lp;
  cudaMalloc((void**) &_Lp, sizeof(real)*dom[rank].Gcc.s3b);

  update_p_laplacian<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_Lp, _phi);
  update_p<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_Lp, _p0, _p, _phi, nu,
    dt, _phase);

  cudaFree(_Lp);


  /* set mean pressure to zero */
  real *_p_mean;
  cudaMalloc((void**) &_p_mean, sizeof(real)*dom[rank].Gcc.s3);

  copy_p_p_noghost<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(_p_mean, _p);

  thrust::device_ptr<real> t_p_mean(_p_mean);
  real pmean = thrust::reduce(t_p_mean, t_p_mean + dom[rank].Gcc.s3, 0.,
    thrust::plus<real>());
  MPI_Allreduce(MPI_IN_PLACE, &pmean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  pmean /= (real) DOM.Gcc.s3;
  // numerical reproducibility? + associativity of floating point addition

  cudaFree(_p_mean);

  forcing_add_c_const<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(-pmean, _p);
}

extern "C"
void cuda_dom_BC_p(real *array)
{
  // Can do this with an if/else, not switch/case
  // if (bc.pW == NEUMANN)...
  // Could also do this with dom[rank].I == DOM.Is
  if (dom[rank].w == MPI_PROC_NULL) { // WEST
    switch (bc.pW) {
      case NEUMANN:
        BC_p_W_N<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array);
        break;
    }
  }
  if (dom[rank].e == MPI_PROC_NULL) { // EAST
    switch (bc.pE) {
      case NEUMANN:
        BC_p_E_N<<<blocks.Gcc.num_in, blocks.Gcc.dim_in>>>(array);
        break;
    }
  }

  if (dom[rank].s == MPI_PROC_NULL) { // SOUTH
    switch (bc.pS) {
      case NEUMANN:
        BC_p_S_N<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array);
        break;
    }
  }

  if (dom[rank].n == MPI_PROC_NULL) { // NORTH
    switch (bc.pN) {
      case NEUMANN:
        BC_p_N_N<<<blocks.Gcc.num_jn, blocks.Gcc.dim_jn>>>(array);
        break;
    }
  }

  if (dom[rank].b == MPI_PROC_NULL) { // BOTTOM
    switch (bc.pB) {
      case NEUMANN:
        BC_p_B_N<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array);
        break;
    }
  }

  if (dom[rank].t == MPI_PROC_NULL) { // TOP
    switch (bc.pT) {
      case NEUMANN:
        BC_p_T_N<<<blocks.Gcc.num_kn, blocks.Gcc.dim_kn>>>(array);
        break;
    }
  }
}

extern "C"
void cuda_store_u(void)
{
  cudaMemcpy(_conv0_u, _conv_u, dom[rank].Gfx.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(_conv0_v, _conv_v, dom[rank].Gfy.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(_conv0_w, _conv_w, dom[rank].Gfz.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);

  cudaMemcpy(_diff0_u, _diff_u, dom[rank].Gfx.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(_diff0_v, _diff_v, dom[rank].Gfy.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(_diff0_w, _diff_w, dom[rank].Gfz.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);

  cudaMemcpy(_p0, _p, dom[rank].Gcc.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(_u0, _u, dom[rank].Gfx.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(_v0, _v, dom[rank].Gfy.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(_w0, _w, dom[rank].Gfz.s3b*sizeof(real), 
    cudaMemcpyDeviceToDevice);
}

extern "C"
void cuda_dom_free(void)
{
  // Free cuda memory on host
  checkCudaErrors(cudaFreeHost(p));
  checkCudaErrors(cudaFreeHost(p0));
  checkCudaErrors(cudaFreeHost(u));
  checkCudaErrors(cudaFreeHost(v));
  checkCudaErrors(cudaFreeHost(w));
  checkCudaErrors(cudaFreeHost(u0));
  checkCudaErrors(cudaFreeHost(v0));
  checkCudaErrors(cudaFreeHost(w0));
  checkCudaErrors(cudaFreeHost(conv_u));
  checkCudaErrors(cudaFreeHost(conv_v));
  checkCudaErrors(cudaFreeHost(conv_w));
  checkCudaErrors(cudaFreeHost(conv0_u));
  checkCudaErrors(cudaFreeHost(conv0_v));
  checkCudaErrors(cudaFreeHost(conv0_w));
  checkCudaErrors(cudaFreeHost(diff_u));
  checkCudaErrors(cudaFreeHost(diff_v));
  checkCudaErrors(cudaFreeHost(diff_w));
  checkCudaErrors(cudaFreeHost(diff0_u));
  checkCudaErrors(cudaFreeHost(diff0_v));
  checkCudaErrors(cudaFreeHost(diff0_w));
  checkCudaErrors(cudaFreeHost(f_x));
  checkCudaErrors(cudaFreeHost(f_y));
  checkCudaErrors(cudaFreeHost(f_z));
  checkCudaErrors(cudaFreeHost(u_star));
  checkCudaErrors(cudaFreeHost(v_star));
  checkCudaErrors(cudaFreeHost(w_star));
  checkCudaErrors(cudaFreeHost(flag_u));
  checkCudaErrors(cudaFreeHost(flag_v));
  checkCudaErrors(cudaFreeHost(flag_w));

  checkCudaErrors(cudaFreeHost(phi));

  // Free cuda memory on device
  checkCudaErrors(cudaFree(_DOM));
  checkCudaErrors(cudaFree(_bc));

  checkCudaErrors(cudaFree(_p));
  checkCudaErrors(cudaFree(_p0));
  checkCudaErrors(cudaFree(_phi));
  checkCudaErrors(cudaFree(_phinoghost));
  checkCudaErrors(cudaFree(_invM));
  checkCudaErrors(cudaFree(_u));
  checkCudaErrors(cudaFree(_v));
  checkCudaErrors(cudaFree(_w));
  checkCudaErrors(cudaFree(_u0));
  checkCudaErrors(cudaFree(_v0));
  checkCudaErrors(cudaFree(_w0));
  checkCudaErrors(cudaFree(_conv_u));
  checkCudaErrors(cudaFree(_conv_v));
  checkCudaErrors(cudaFree(_conv_w));
  checkCudaErrors(cudaFree(_conv0_u));
  checkCudaErrors(cudaFree(_conv0_v));
  checkCudaErrors(cudaFree(_conv0_w));
  checkCudaErrors(cudaFree(_diff_u));
  checkCudaErrors(cudaFree(_diff_v));
  checkCudaErrors(cudaFree(_diff_w));
  checkCudaErrors(cudaFree(_diff0_u));
  checkCudaErrors(cudaFree(_diff0_v));
  checkCudaErrors(cudaFree(_diff0_w));
  checkCudaErrors(cudaFree(_f_x));
  checkCudaErrors(cudaFree(_f_y));
  checkCudaErrors(cudaFree(_f_z));
  checkCudaErrors(cudaFree(_u_star));
  checkCudaErrors(cudaFree(_v_star));
  checkCudaErrors(cudaFree(_w_star));
  checkCudaErrors(cudaFree(_flag_u));
  checkCudaErrors(cudaFree(_flag_v));
  checkCudaErrors(cudaFree(_flag_w));

  checkCudaErrors(cudaFree(_rhs_p));
  checkCudaErrors(cudaFree(_r_q));
  checkCudaErrors(cudaFree(_z_q));
  //checkCudaErrors(cudaFree(_rs_0));
  checkCudaErrors(cudaFree(_p_q));
  checkCudaErrors(cudaFree(_pb_q));
  //checkCudaErrors(cudaFree(_s_q));
  //checkCudaErrors(cudaFree(_sb_q));
  checkCudaErrors(cudaFree(_Apb_q));
  //checkCudaErrors(cudaFree(_Asb_q));

  checkCudaErrors(cudaFree(_send_Gcc_e));
  checkCudaErrors(cudaFree(_send_Gcc_w));
  checkCudaErrors(cudaFree(_send_Gcc_n));
  checkCudaErrors(cudaFree(_send_Gcc_s));
  checkCudaErrors(cudaFree(_send_Gcc_t));
  checkCudaErrors(cudaFree(_send_Gcc_b));

  checkCudaErrors(cudaFree(_send_Gfx_e));
  checkCudaErrors(cudaFree(_send_Gfx_w));
  checkCudaErrors(cudaFree(_send_Gfx_n));
  checkCudaErrors(cudaFree(_send_Gfx_s));
  checkCudaErrors(cudaFree(_send_Gfx_t));
  checkCudaErrors(cudaFree(_send_Gfx_b));

  checkCudaErrors(cudaFree(_send_Gfy_e));
  checkCudaErrors(cudaFree(_send_Gfy_w));
  checkCudaErrors(cudaFree(_send_Gfy_n));
  checkCudaErrors(cudaFree(_send_Gfy_s));
  checkCudaErrors(cudaFree(_send_Gfy_t));
  checkCudaErrors(cudaFree(_send_Gfy_b));

  checkCudaErrors(cudaFree(_send_Gfz_e));
  checkCudaErrors(cudaFree(_send_Gfz_w));
  checkCudaErrors(cudaFree(_send_Gfz_n));
  checkCudaErrors(cudaFree(_send_Gfz_s));
  checkCudaErrors(cudaFree(_send_Gfz_t));
  checkCudaErrors(cudaFree(_send_Gfz_b));

  checkCudaErrors(cudaFree(_recv_Gcc_e));
  checkCudaErrors(cudaFree(_recv_Gcc_w));
  checkCudaErrors(cudaFree(_recv_Gcc_n));
  checkCudaErrors(cudaFree(_recv_Gcc_s));
  checkCudaErrors(cudaFree(_recv_Gcc_t));
  checkCudaErrors(cudaFree(_recv_Gcc_b));

  checkCudaErrors(cudaFree(_recv_Gfx_e));
  checkCudaErrors(cudaFree(_recv_Gfx_w));
  checkCudaErrors(cudaFree(_recv_Gfx_n));
  checkCudaErrors(cudaFree(_recv_Gfx_s));
  checkCudaErrors(cudaFree(_recv_Gfx_t));
  checkCudaErrors(cudaFree(_recv_Gfx_b));

  checkCudaErrors(cudaFree(_recv_Gfy_e));
  checkCudaErrors(cudaFree(_recv_Gfy_w));
  checkCudaErrors(cudaFree(_recv_Gfy_n));
  checkCudaErrors(cudaFree(_recv_Gfy_s));
  checkCudaErrors(cudaFree(_recv_Gfy_t));
  checkCudaErrors(cudaFree(_recv_Gfy_b));

  checkCudaErrors(cudaFree(_recv_Gfz_e));
  checkCudaErrors(cudaFree(_recv_Gfz_w));
  checkCudaErrors(cudaFree(_recv_Gfz_n));
  checkCudaErrors(cudaFree(_recv_Gfz_s));
  checkCudaErrors(cudaFree(_recv_Gfz_t));
  checkCudaErrors(cudaFree(_recv_Gfz_b));

  // Reset devices
  checkCudaErrors(cudaDeviceReset());
}

// Miscellaneous functions
extern "C"
void cuda_wall_shear_stress()
{
  real *_dudy;
  cudaMalloc(&_dudy, dom[rank].Gfx.s2_j * sizeof(real));
  cudaMemset(_dudy, 0., dom[rank].Gfx.s2_j);
  thrust::device_ptr<real> t_dudy(_dudy);

  real dudy_s = 0.;
  real dudy_n = 0.;

  // On south face
  if (dom[rank].J == DOM.Js) {
    calc_dudy<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u, _dudy,
      dom[rank].Gfx._jsb);

    dudy_s = thrust::reduce(t_dudy, t_dudy + dom[rank].Gfx.s2_j, 0.,
      thrust::plus<real>());
  } else {
    dudy_s = 0.;
  }
  MPI_Allreduce(MPI_IN_PLACE, &dudy_s, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  dudy_s /= DOM.Gfx.s2_j;

  // On north face
  if (dom[rank].J == DOM.Je) {
    calc_dudy<<<blocks.Gfx.num_jn, blocks.Gfx.dim_jn>>>(_u, _dudy,
      dom[rank].Gfx._je);

    dudy_n = thrust::reduce(t_dudy, t_dudy + dom[rank].Gfx.s2_j, 0.,
      thrust::plus<real>());
  } else {
    dudy_n = 0.;
  }

  MPI_Allreduce(MPI_IN_PLACE, &dudy_n, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  dudy_n /= DOM.Gfx.s2_j;

  // Open file for writing
  if (rank == 0) {
    char fname[FILE_NAME_SIZE];
    sprintf(fname, "%s/%s/wss.dat", ROOT_DIR, OUTPUT_DIR);
    FILE *file;

    if (stepnum == 1) {
      file = fopen(fname, "w");
      if (file == NULL) {
        fprintf(stderr, "Could not open file %s\n", fname);
        exit(EXIT_FAILURE);
      }
      fprintf(file, "%-9s", "stepnum");
      fprintf(file, "%-11s", "ttime");
      fprintf(file, "%-11s", "wss-s");
      fprintf(file, "%-11s", "wss-n");
    } else {
      file = fopen(fname, "a");
      if (file == NULL) {
        fprintf(stderr, "Could not open file %s\n", fname);
        exit(EXIT_FAILURE);
      }
    }

    fprintf(file, "\n");
    fprintf(file, "%-9d", stepnum);
    fprintf(file, "%-11.3e", ttime);
    fprintf(file, "%-11.3e", rho_f*nu*dudy_s);
    fprintf(file, "%-11.3e", rho_f*nu*dudy_n);

    fclose(file);
  }

  // Free
  cudaFree(_dudy);
}
