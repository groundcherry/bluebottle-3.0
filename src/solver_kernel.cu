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

#include "cuda_solver.h"

__global__ void PP_jacobi_init(int *flag_u, int *flag_v, int *flag_w, 
  real *invM)
{
  // we can do this in shared memory... not terribly urgent since it happens 1x

  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  // Location indexes
  int c;                          // current thread index in comp domain
  #ifdef JACOBI
    int TI, TJ, TK;               // s3b indices
    int Efx, Wfx;                 // Gfx-centered global indices (flag_u)
    int Nfy, Sfy;                 // Gfy-centered global indices (flag_v)
    int Tfz, Bfz;                 // Gfz-centered global indices (flag_w)

    // Prefactors
    real idx2= 1./(_dom.dx * _dom.dx);
    real idy2= 1./(_dom.dy * _dom.dy);
    real idz2= 1./(_dom.dz * _dom.dz);

    real M;
  #endif


  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn)) {
    for (int tk = 0; tk < _dom.Gcc.kn; tk++) {
      /* s3 indices */
      c = GCC_LOC(ti, tj, tk, _dom.Gcc.s1, _dom.Gcc.s2);
      
      #ifdef JACOBI
        /* s3b indices */
        TI = ti + DOM_BUF;
        TJ = tj + DOM_BUF;
        TK = tk + DOM_BUF;

        /* s3b neighbor indices */
        Wfx = GFX_LOC(TI, TJ, TK, _dom.Gfx.s1b, _dom.Gfx.s2b);
        Efx = GFX_LOC(TI + 1, TJ, TK, _dom.Gfx.s1b, _dom.Gfx.s2b);
        Sfy = GFY_LOC(TI, TJ, TK, _dom.Gfy.s1b, _dom.Gfy.s2b);
        Nfy = GFY_LOC(TI, TJ + 1, TK, _dom.Gfy.s1b, _dom.Gfy.s2b);
        Bfz = GFZ_LOC(TI, TJ, TK, _dom.Gfz.s1b, _dom.Gfz.s2b);
        Tfz = GFZ_LOC(TI, TJ, TK + 1, _dom.Gfz.s1b, _dom.Gfz.s2b);

        // M cannot be zero here, since flag == 0 only on an external boundary.
        M = -idx2*(flag_u[Efx]*flag_u[Efx] + flag_u[Wfx]*flag_u[Wfx])
            -idy2*(flag_v[Nfy]*flag_v[Nfy] + flag_v[Sfy]*flag_v[Sfy])
            -idz2*(flag_w[Tfz]*flag_w[Tfz] + flag_w[Bfz]*flag_w[Bfz]);

        // If solving Ax = b
        // invM[c] = 1./M;

        // If solving -Ax = -b
        invM[c] = -1./M;

      #elif NOPRECOND
        invM[c] = 1.;
      #endif // JACOBI
    }
  }
}

__global__ void PP_rhs(real rho_f, real *u_star, real *v_star, real *w_star,
  real *rhs, real dt)
{
  /* Fill in RHS only on solution points in current subdom
   *  MPI exchange will take care of the boundary cells
   */

  /* Prefactors and constants */
  real idx = 1. / _dom.dx;
  real idy = 1. / _dom.dy;
  real idz = 1. / _dom.dz;
  real rho_idt = rho_f / dt;

  /* Create shared memory */
  __shared__ real s_w0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w back
  __shared__ real s_w1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w center
  __shared__ real s_u[MAX_THREADS_DIM * MAX_THREADS_DIM];       // u
  __shared__ real s_v[MAX_THREADS_DIM * MAX_THREADS_DIM];       // v
  __shared__ real s_rhs[MAX_THREADS_DIM * MAX_THREADS_DIM];     // solution

  /* s3b subdomain indices */
  // num_kn, dim_kn
  int TI = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
  int TJ = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;
  int TK;                                 // calculate in loop

  /* Shared memory indices */
  int si = threadIdx.x;         // x-position in 2d grid
  int sj = threadIdx.y;         // y-position in 2d grid
  int sc = si + sj*blockDim.x;  // index in 1-d strided grid
  int se = sc + 1;
  int sn = sc + blockDim.x;

  /* loop over z-planes and load shared memory */
  for (TK = _dom.Gcc._ks; TK <= _dom.Gcc._ke; TK++) {
    // u: [_is, _ie] because only calculating rhs there. Gfx/Gcc takes care of
    //  staggered grid, and ghost values filled in after
    if ((TI >= _dom.Gfx._is && TI <= _dom.Gfx._ie) &&
        (TJ >= _dom.Gfx._js && TJ <= _dom.Gfx._je)) {
      s_u[sc] = u_star[GFX_LOC(TI, TJ, TK, _dom.Gfx.s1b, _dom.Gfx.s2b)];
    }

    // v
    if ((TI >= _dom.Gfy._is && TI <= _dom.Gfy._ie) &&
        (TJ >= _dom.Gfy._js && TJ <= _dom.Gfy._je)) {
      s_v[sc] = v_star[GFY_LOC(TI, TJ, TK, _dom.Gfy.s1b, _dom.Gfy.s2b)];
    }

    // w
    if ((TI >= _dom.Gfz._is && TI <= _dom.Gfz._ie) &&
        (TJ >= _dom.Gfz._js && TJ <= _dom.Gfz._je)) {
      s_w0[sc] = w_star[GFZ_LOC(TI, TJ, TK, _dom.Gfz.s1b, _dom.Gfz.s2b)];
      s_w1[sc] = w_star[GFZ_LOC(TI, TJ, TK + 1, _dom.Gfz.s1b, _dom.Gfz.s2b)];
    }

    s_rhs[sc] = 0.;

    __syncthreads();

    /* Compute RHS */
    // if off the shared memory boundary block
    if ((si > 0) && (si < (blockDim.x-1)) &&
        (sj > 0) && (sj < (blockDim.y-1))) {
      s_rhs[sc] =  (s_u[se] - s_u[sc]) * idx;
      s_rhs[sc] += (s_v[sn] - s_v[sc]) * idy;
      s_rhs[sc] += (s_w1[sc] - s_w0[sc]) * idz;


      s_rhs[sc] *= rho_idt;
    }

    __syncthreads();

    /* Copy back to global */
    if ((TI >= _dom.Gcc._is && TI <= _dom.Gcc._ie) &&
        (TJ >= _dom.Gcc._js && TJ <= _dom.Gcc._je) &&
        (si > 0) && (si < blockDim.x-1) &&
        (sj > 0) && (sj < blockDim.y-1)) {
      int cc = GCC_LOC(TI, TJ, TK, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // If solving Ax = b
      //rhs[cc] = s_rhs[sc];

      // If solving -Ax = -b
      rhs[cc] = -s_rhs[sc];
    }
  }
}

__global__ void zero_rhs_ghost_i(real *rhs_p)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  if (tj < _dom.Gcc.jnb && tk < _dom.Gcc.knb) {
    rhs_p[GCC_LOC(_dom.Gcc._isb, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b)] = 0.;
    rhs_p[GCC_LOC(_dom.Gcc._ieb, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b)] = 0.;
  }
}

__global__ void zero_rhs_ghost_j(real *rhs_p)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  if (tk < _dom.Gcc.knb && ti < _dom.Gcc.inb) {
    rhs_p[GCC_LOC(ti, _dom.Gcc._jsb, tk, _dom.Gcc.s1b, _dom.Gcc.s2b)] = 0.;
    rhs_p[GCC_LOC(ti, _dom.Gcc._jeb, tk, _dom.Gcc.s1b, _dom.Gcc.s2b)] = 0.;
  }
}

__global__ void zero_rhs_ghost_k(real *rhs_p)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  if (ti < _dom.Gcc.inb && tj < _dom.Gcc.jnb) {
    rhs_p[GCC_LOC(ti, tj, _dom.Gcc._ksb, _dom.Gcc.s1b, _dom.Gcc.s2b)] = 0.;
    rhs_p[GCC_LOC(ti, tj, _dom.Gcc._keb, _dom.Gcc.s1b, _dom.Gcc.s2b)] = 0.;
  }
}


__global__ void coeffs_refine(real *rhs, int *phase, int *flag_u, int *flag_v,
  int *flag_w)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tj = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;

  int CC;
  int is_fluid; // phase of CC

  // stencil locations
  int CE, CW;
  int CN, CS;
  int CT, CB;

  // Prefactors
  real idx2 = 1./(_dom.dx * _dom.dx);
  real idy2 = 1./(_dom.dy * _dom.dy);
  real idz2 = 1./(_dom.dz * _dom.dz);

  if (ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for (int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      // Set stencil locations
      CC = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CE = GCC_LOC(ti + 1, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CW = GCC_LOC(ti - 1, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CN = GCC_LOC(ti, tj + 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CS = GCC_LOC(ti, tj - 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CT = GCC_LOC(ti, tj, k + 1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CB = GCC_LOC(ti, tj, k - 1, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Check if stencil locations are in a particle. If so, and CC is fluid,
      //  we need to adjust rhs with the term associated with the stencil point
      is_fluid = (phase[CC] == -1);

      // If solving -Ax = -b
      rhs[CC] += is_fluid * (phase[CE] > -1) * idx2 * (-rhs[CE]);
      rhs[CC] += is_fluid * (phase[CW] > -1) * idx2 * (-rhs[CW]);
      rhs[CC] += is_fluid * (phase[CN] > -1) * idy2 * (-rhs[CN]);
      rhs[CC] += is_fluid * (phase[CS] > -1) * idy2 * (-rhs[CS]);
      rhs[CC] += is_fluid * (phase[CT] > -1) * idz2 * (-rhs[CT]);
      rhs[CC] += is_fluid * (phase[CB] > -1) * idz2 * (-rhs[CB]);
    }
  }
}

__global__ void PP_cg_init(real *r_q, real *p_q, real *pb_q,
  real *phi, real *rhs, real *z_q, real *invM)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  // Loop over z-planes
  if ((ti < _dom.Gcc.in) & (tj < _dom.Gcc.jn)) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      int c = GCC_LOC(ti, tj , k, _dom.Gcc.s1, _dom.Gcc.s2);
      int C = GCC_LOC(ti + DOM_BUF, tj + DOM_BUF, k + DOM_BUF,
                      _dom.Gcc.s1b, _dom.Gcc.s2b);

      real tmp = rhs[C];
      real tmp2 = tmp*invM[c];

      r_q[c] = tmp;       // residual
      z_q[c] = tmp2;      // aux vector for precond
      p_q[c] = tmp2;      // search direction (local)

      pb_q[C] = tmp2;     // search direction (global)
      phi[C] = 0.;        // init solution (global)
    }
  }
}

__global__ void PP_init_search_ghost(real *pb_q, real *rhs, 
  dom_struct *DOM, BC *bc)
{
  // num_knb, dim_knb
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  // Edge flags -- 1 iff global dirichlet or neumann edge, 0 else
  //int flag_e, flag_w, flag_n, flag_s, flag_t, flag_b;
  //int prod;

  // loop over z-planes
  if ((ti < _dom.Gcc.inb) & (tj < _dom.Gcc.jnb)) {
    for (int k = _dom.Gcc._ksb; k <= _dom.Gcc._keb; k++) {
      int C = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Set _pb_q = 0 if it is a !PERIODIC external ghost cell boundary
      // Determine if C is a global domain boundary
     // flag_w=(bc->pW != PERIODIC)*((_dom.I == DOM->Is) && (ti == _dom.Gcc.isb));
     // flag_e=(bc->pE != PERIODIC)*((_dom.I == DOM->Ie) && (ti == _dom.Gcc.ieb));
     // flag_s=(bc->pS != PERIODIC)*((_dom.J == DOM->Js) && (tj == _dom.Gcc.jsb));
     // flag_n=(bc->pN != PERIODIC)*((_dom.J == DOM->Je) && (tj == _dom.Gcc.jeb));
     // flag_b=(bc->pB != PERIODIC)*((_dom.K == DOM->Ks) && (k == _dom.Gcc.ksb));
     // flag_t=(bc->pT != PERIODIC)*((_dom.K == DOM->Ke) && (k == _dom.Gcc.keb));

     // prod = (1 - flag_w)*(1 - flag_e)*
     //        (1 - flag_s)*(1 - flag_n)*
     //        (1 - flag_b)*(1 - flag_t);

      pb_q[C] = rhs[C];//*prod;
    }
  }
}

__global__ void PP_spmv(int *flag_u, int *flag_v, int *flag_w,
  real *pb_q, real *Apb_q)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  // Location indexes
  int c;                          // current thread index in comp domain
  int TI, TJ, TK;                 // s3b indices
  int CC, CE, CW, CN, CS, CT, CB; // cell-centered global indices
  int Efx, Wfx;                   // Gfx-centered global indices (flag_u)
  int Nfy, Sfy;                   // Gfy-centered global indices (flag_v)
  int Tfz, Bfz;                   // Gfz-centered global indices (flag_w)

  // Prefactors
  real idx2 = 1./(_dom.dx * _dom.dx);
  real idy2 = 1./(_dom.dy * _dom.dy);
  real idz2 = 1./(_dom.dz * _dom.dz);

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn)) {
    for (int tk = 0; tk < _dom.Gcc.kn; tk++) {
      /* s3 indices */
      c = GCC_LOC(ti, tj, tk, _dom.Gcc.s1, _dom.Gcc.s2);
      
      /* s3b indices */
      TI = ti + DOM_BUF;
      TJ = tj + DOM_BUF;
      TK = tk + DOM_BUF;
      CC = GCC_LOC(TI, TJ, TK, _dom.Gcc.s1b, _dom.Gcc.s2b);

      /* s3b neighbor indices */
      CE = GCC_LOC(TI + 1, TJ, TK, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CW = GCC_LOC(TI - 1, TJ, TK, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CN = GCC_LOC(TI, TJ + 1, TK, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CS = GCC_LOC(TI, TJ - 1, TK, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CT = GCC_LOC(TI, TJ, TK + 1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CB = GCC_LOC(TI, TJ, TK - 1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CC = GCC_LOC(TI, TJ, TK, _dom.Gcc.s1b, _dom.Gcc.s2b);
      Wfx = GFX_LOC(TI, TJ, TK, _dom.Gfx.s1b, _dom.Gfx.s2b);
      Efx = GFX_LOC(TI + 1, TJ, TK, _dom.Gfx.s1b, _dom.Gfx.s2b);
      Sfy = GFY_LOC(TI, TJ, TK, _dom.Gfy.s1b, _dom.Gfy.s2b);
      Nfy = GFY_LOC(TI, TJ + 1, TK, _dom.Gfy.s1b, _dom.Gfy.s2b);
      Bfz = GFZ_LOC(TI, TJ, TK, _dom.Gfz.s1b, _dom.Gfz.s2b);
      Tfz = GFZ_LOC(TI, TJ, TK + 1, _dom.Gfz.s1b, _dom.Gfz.s2b);

      /* Matrix-free multiplication */
      // If solving Ax = b
      //Apb_q[c] = idx2 * (flag_u[Efx]*flag_u[Efx]*(pb_q[CE] - pb_q[CC])
      //                 - flag_u[Wfx]*flag_u[Wfx]*(pb_q[CC] - pb_q[CW])) + 
      //           idy2 * (flag_v[Nfy]*flag_v[Nfy]*(pb_q[CN] - pb_q[CC])
      //                 - flag_v[Sfy]*flag_v[Sfy]*(pb_q[CC] - pb_q[CS])) +
      //           idz2 * (flag_w[Tfz]*flag_w[Tfz]*(pb_q[CT] - pb_q[CC])
      //                 - flag_w[Bfz]*flag_w[Bfz]*(pb_q[CC] - pb_q[CB]));

      // If solving -Ax = -b
      Apb_q[c] = -idx2 * (flag_u[Efx]*flag_u[Efx]*(pb_q[CE] - pb_q[CC])
                        - flag_u[Wfx]*flag_u[Wfx]*(pb_q[CC] - pb_q[CW]))
                 -idy2 * (flag_v[Nfy]*flag_v[Nfy]*(pb_q[CN] - pb_q[CC])
                        - flag_v[Sfy]*flag_v[Sfy]*(pb_q[CC] - pb_q[CS]))
                 -idz2 * (flag_w[Tfz]*flag_w[Tfz]*(pb_q[CT] - pb_q[CC])
                        - flag_w[Bfz]*flag_w[Bfz]*(pb_q[CC] - pb_q[CB]));
    }
  }
}

__global__ void PP_spmv_shared(int *flag_u, int *flag_v, int *flag_w,
  real *pb_q, real *Apb_q)
{
  // num_kn_s, dim_kn_s

  /* Prefactors */
  real idx2 = 1./(_dom.dx * _dom.dx);
  real idy2 = 1./(_dom.dy * _dom.dy);
  real idz2 = 1./(_dom.dz * _dom.dz);

  /* Create shared memory */
  // pb_q and flag_w need 3 and 2 planes, respectively
  __shared__ real s_pb_q_t[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ real s_pb_q[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ real s_pb_q_b[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_u[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_v[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_w_t[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_w_b[MAX_THREADS_DIM * MAX_THREADS_DIM];

  //__shared__ real s_pb_q_top[MAX_THREADS_DIM * MAX_THREADS_DIM];
  //__shared__ real s_pb_q_mid[MAX_THREADS_DIM * MAX_THREADS_DIM];
  //__shared__ real s_pb_q_bot[MAX_THREADS_DIM * MAX_THREADS_DIM];
  //real *ptmp;
  //real *pt = s_pb_q_top;
  //real *pm = s_pb_q_mid;
  //real *pb = s_pb_q_bot;

  /* Shared memory indices */
  int si = threadIdx.x;         // x-position in 2D grid
  int sj = threadIdx.y;         // y-position in 2D grid
  int sc = si + sj*blockDim.x;  // indices in 1D array
  int sw = sc - 1;
  int se = sc + 1;
  int ss = sc - blockDim.x;
  int sn = sc + blockDim.x;

  /* s3b subdomain indices */
  // -2*BlockIdx.{x,y} corrects for shared memory overlap 
  int TI = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
  int TJ = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;
  int TK;                                 // calculate in loop

  /* s3 subdomain location */
  int cc;

//  /* Load initial shared memory */
//  if ((TI >= _dom.Gcc._isb && TI <= _dom.Gcc._ieb) &&
//      (TJ >= _dom.Gcc._jsb && TJ <= _dom.Gcc._jeb)) {
////  with GCC_LOC!!
//    s_pb_q_t[sc] = pb_q[TI + TJ*_dom.Gcc.s1b + _dom.Gcc._ks*_dom.Gcc.s2b];
//    s_pb_q[sc] = pb_q[TI + TJ*_dom.Gcc.s1b + (_dom.Gcc._ks - 1)*_dom.Gcc.s2b];
//    //*(pt + sc) = pb_q[TI + TJ*_dom.Gcc.s1b + _dom.Gcc._ks*_dom.Gcc.s2b];
//    //*(pm + sc) = pb_q[TI + TJ*_dom.Gcc.s1b + (_dom.Gcc._ks - 1)*_dom.Gcc.s2b];
//  }
//  if ((TI >= _dom.Gfz._isb && TI <= _dom.Gfz._ieb) &&
//      (TJ >= _dom.Gfz._jsb && TJ <= _dom.Gfz._jeb)) {
//    s_flag_w_t[sc] = flag_w[TI + TJ*_dom.Gfz.s1b + _dom.Gcc._ks*_dom.Gfz.s2b];
//  }

  /* Ensure initial load is done */
  __syncthreads();

  /* Loop over z planes and load shared memory */
  // WHAT IF WE LOOP OVER I??? think lines, not planes
  // ... or over both? need to change planes that are grabbed?
  for (TK = _dom.Gcc._ks; TK <= _dom.Gcc._ke; TK++) {
    // pb_q -- top, middle, and bottom planes
    //ptmp = pt;
    //pt = pb;
    //pb = pm;
    //pm = ptmp;
    if ((TI >= _dom.Gcc._isb && TI <= _dom.Gcc._ieb) &&
        (TJ >= _dom.Gcc._jsb && TJ <= _dom.Gcc._jeb)) {
        // Reload bottom, middle from shared, not global
        // 1. initial load of top <- ks
        //                    mid <- ks-1
        // 2. thereafter, bot <- mid
        //                mid <- top
        //                top <- load
       //s_pb_q_b[sc] = s_pb_q[sc];  // can we do a pointer swap?
       //s_pb_q[sc] = s_pb_q_t[sc];
       s_pb_q_t[sc] = pb_q[GCC_LOC(TI, TJ, (TK + 1), _dom.Gcc.s1b, _dom.Gcc.s2b)];
       s_pb_q[sc] = pb_q[GCC_LOC(TI, TJ, TK, _dom.Gcc.s1b, _dom.Gcc.s2b)];
       s_pb_q_b[sc] = pb_q[GCC_LOC(TI, TJ, (TK - 1), _dom.Gcc.s1b, _dom.Gcc.s2b)];
       //*(pt + sc) = pb_q[TI + TJ*_dom.Gcc.s1b + (TK + 1)*_dom.Gcc.s2b];
     }

    // flag_u
    if ((TI >= _dom.Gfx._isb && TI <= _dom.Gfx._ieb) &&
        (TJ >= _dom.Gfx._jsb && TJ <= _dom.Gfx._jeb)) {
      s_flag_u[sc] = flag_u[GFX_LOC(TI, TJ, TK, _dom.Gfx.s1b, _dom.Gfx.s2b)];
    }

    // flag_v
    if ((TI >= _dom.Gfy._isb && TI <= _dom.Gfy._ieb) &&
        (TJ >= _dom.Gfy._jsb && TJ <= _dom.Gfy._jeb)) {
      s_flag_v[sc] = flag_v[GFY_LOC(TI, TJ, TK, _dom.Gfy.s1b, _dom.Gfy.s2b)];
    }

    // flag_w
    if ((TI >= _dom.Gfz._isb && TI <= _dom.Gfz._ieb) &&
        (TJ >= _dom.Gfz._jsb && TJ <= _dom.Gfz._jeb)) {
      // Reload memory from shared, not global, when possible
      //s_flag_w_b[sc] = s_flag_w_t[sc];
      s_flag_w_t[sc] = flag_w[GFZ_LOC(TI, TJ, TK+1, _dom.Gfz.s1b,_dom.Gfz.s2b)];
      s_flag_w_b[sc] = flag_w[GFZ_LOC(TI, TJ, TK, _dom.Gfz.s1b, _dom.Gfz.s2b)];
    }

    /* Ensure shared memory load is complete */
    __syncthreads();

    /* Compute A*pb_q into global */
    if ((TI >= _dom.Gcc._is && TI <= _dom.Gcc._ie) &&
        (TJ >= _dom.Gcc._js && TJ <= _dom.Gcc._je) &&
        (si > 0 && si < (blockDim.x-1)) && (sj > 0 && sj < (blockDim.y-1))) {
      cc = GCC_LOC(TI - DOM_BUF, TJ - DOM_BUF, TK - DOM_BUF,
                  _dom.Gcc.s1, _dom.Gcc.s2);

      /* Matrix-free multiplication */
      // If solving Ax = b
      //Apb_q[cc] = idx2 * (s_flag_u[se]*s_flag_u[se]*(s_pb_q[se] - s_pb_q[sc])
      //                 - s_flag_u[sc]*s_flag_u[sc]*(s_pb_q[sc] - s_pb_q[sw])) +
      //            idy2 * (s_flag_v[sn]*s_flag_v[sn]*(s_pb_q[sn] - s_pb_q[sc])
      //                 - s_flag_v[sc]*s_flag_v[sc]*(s_pb_q[sc] - s_pb_q[ss])) +
      //            idz2 * (s_flag_w_t[sc]*s_flag_w_t[sc]*(s_pb_q_t[sc] - s_pb_q[sc])
      //                 - s_flag_w_b[sc]*s_flag_w_b[sc]*(s_pb_q[sc] - s_pb_q_b[sc]));

      // If solve -Ax = -b
      Apb_q[cc] = -idx2 * (s_flag_u[se]*s_flag_u[se]*(s_pb_q[se] - s_pb_q[sc])
                       - s_flag_u[sc]*s_flag_u[sc]*(s_pb_q[sc] - s_pb_q[sw])) -
                  idy2 * (s_flag_v[sn]*s_flag_v[sn]*(s_pb_q[sn] - s_pb_q[sc])
                       - s_flag_v[sc]*s_flag_v[sc]*(s_pb_q[sc] - s_pb_q[ss])) -
                  idz2 * (s_flag_w_t[sc]*s_flag_w_t[sc]*(s_pb_q_t[sc] - s_pb_q[sc])
                       - s_flag_w_b[sc]*s_flag_w_b[sc]*(s_pb_q[sc] - s_pb_q_b[sc]));
      
    }

    /* Ensure computation is complete */
    __syncthreads();
  }
}

__global__ void PP_spmv_shared_load(int *flag_u, int *flag_v, int *flag_w,
  real *pb_q, real *Apb_q, int *phase)
{
  // num_kn_s, dim_kn_s

  /* Prefactors */
  real idx2 = 1./(_dom.dx * _dom.dx);
  real idy2 = 1./(_dom.dy * _dom.dy);
  real idz2 = 1./(_dom.dz * _dom.dz);

  /* Create shared memory */
  // pb_q and flag_w need 3 and 2 planes, respectively
  __shared__ int s_flag_u[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_v[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_w_t[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_w_b[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ real s_pb_q_t[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ real s_pb_q_m[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ real s_pb_q_b[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_phase_t[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_phase_m[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_phase_b[MAX_THREADS_DIM * MAX_THREADS_DIM];

  // Another thought -- do a pointer swap to the location, rather than
  // read/write
  //real *ptmp;
  //real *pt = s_pb_q_top;
  //real *pm = s_pb_q_mid;
  //real *pb = s_pb_q_bot;
  //
  //*(pt + sc) = pb_q[TI + TJ*_dom.Gcc.s1b + _dom.Gcc._ks*_dom.Gcc.s2b];
  //*(pm + sc) = pb_q[TI + TJ*_dom.Gcc.s1b + (_dom.Gcc._ks - 1)*_dom.Gcc.s2b];
  //ptmp = pt;
  //pt = pb;
  //pb = pm;
  //pm = ptmp;

  /* Shared memory indices */
  int si = threadIdx.x;         // x-position in 2D grid
  int sj = threadIdx.y;         // y-position in 2D grid
  int sc = si + sj*blockDim.x;  // indices in 1D array
  int sw = sc - 1;
  int se = sc + 1;
  int ss = sc - blockDim.x;
  int sn = sc + blockDim.x;

  /* s3b subdomain indices */
  // -2*BlockIdx.{x,y} corrects for shared memory overlap 
  int TI = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
  int TJ = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;
  int TK;                                 // calculate in loop

  /* s3 subdomain location */
  int cc;

  /* Load initial shared memory planes */
  if ((TI >= _dom.Gcc._isb && TI <= _dom.Gcc._ieb) &&
      (TJ >= _dom.Gcc._jsb && TJ <= _dom.Gcc._jeb)) {
    s_pb_q_t[sc] = pb_q[GCC_LOC(TI, TJ, _dom.Gcc._ks, _dom.Gcc.s1b, _dom.Gcc.s2b)];
    s_pb_q_m[sc] = pb_q[GCC_LOC(TI, TJ, _dom.Gcc._ksb, _dom.Gcc.s1b, _dom.Gcc.s2b)];

    s_phase_t[sc] = phase[GCC_LOC(TI, TJ, _dom.Gcc._ks, _dom.Gcc.s1b, _dom.Gcc.s2b)];
    s_phase_m[sc] = phase[GCC_LOC(TI, TJ, _dom.Gcc._ksb, _dom.Gcc.s1b, _dom.Gcc.s2b)];
  }
  if ((TI >= _dom.Gfz._isb && TI <= _dom.Gfz._ieb) &&
      (TJ >= _dom.Gfz._jsb && TJ <= _dom.Gfz._jeb)) {
    s_flag_w_t[sc] = flag_w[GFZ_LOC(TI, TJ, _dom.Gfz._ks, _dom.Gfz.s1b, _dom.Gfz.s2b)];
  }

  /* Ensure initial load is done */
  __syncthreads();

  /* Loop over z planes and load shared memory */
  for (TK = _dom.Gcc._ks; TK <= _dom.Gcc._ke; TK++) {
    // pb_q -- top, middle, and bottom planes
    if ((TI >= _dom.Gcc._isb && TI <= _dom.Gcc._ieb) &&
        (TJ >= _dom.Gcc._jsb && TJ <= _dom.Gcc._jeb)) {
        // Reload bottom and middle from shared
        // Load top from global
        // 1. initial load of top <- ks
        //                    mid <- ksb
        // 2. thereafter, bot <- mid
        //                mid <- top
        //                top <- load
       s_pb_q_b[sc] = s_pb_q_m[sc];
       s_pb_q_m[sc] = s_pb_q_t[sc];
       s_pb_q_t[sc] = pb_q[GCC_LOC(TI, TJ, (TK + 1), _dom.Gcc.s1b, _dom.Gcc.s2b)];
       s_phase_b[sc] = s_phase_m[sc];
       s_phase_m[sc] = s_phase_t[sc];
       s_phase_t[sc] = phase[GCC_LOC(TI, TJ, (TK + 1), _dom.Gcc.s1b, _dom.Gcc.s2b)];
     }

    // flag_u
    if ((TI >= _dom.Gfx._isb && TI <= _dom.Gfx._ieb) &&
        (TJ >= _dom.Gfx._jsb && TJ <= _dom.Gfx._jeb)) {
      s_flag_u[sc] = flag_u[GFX_LOC(TI, TJ, TK, _dom.Gfx.s1b, _dom.Gfx.s2b)];
    }

    // flag_v
    if ((TI >= _dom.Gfy._isb && TI <= _dom.Gfy._ieb) &&
        (TJ >= _dom.Gfy._jsb && TJ <= _dom.Gfy._jeb)) {
      s_flag_v[sc] = flag_v[GFY_LOC(TI, TJ, TK, _dom.Gfy.s1b, _dom.Gfy.s2b)];
    }

    // flag_w
    if ((TI >= _dom.Gfz._isb && TI <= _dom.Gfz._ieb) &&
        (TJ >= _dom.Gfz._jsb && TJ <= _dom.Gfz._jeb)) {
      // Reload memory from shared, not global, when possible
      s_flag_w_b[sc] = s_flag_w_t[sc];
      s_flag_w_t[sc] = flag_w[GFZ_LOC(TI, TJ, (TK + 1), _dom.Gfz.s1b, _dom.Gfz.s2b)];
    }

    /* Ensure shared memory load is complete */
    __syncthreads();

    /* Compute A*pb_q into global */
    if ((TI >= _dom.Gcc._is && TI <= _dom.Gcc._ie) &&
        (TJ >= _dom.Gcc._js && TJ <= _dom.Gcc._je) &&
        (si > 0 && si < (blockDim.x-1)) && (sj > 0 && sj < (blockDim.y-1))) {
      cc = GCC_LOC(TI - DOM_BUF, TJ - DOM_BUF, TK - DOM_BUF,
                  _dom.Gcc.s1, _dom.Gcc.s2);

      // abs(flag) = flag*flag. Do this here so it stays in register
      /* If solving Ax = b */
      //Apb_q[cc] = idx2 * (s_flag_u[se]*s_flag_u[se]*(s_pb_q_m[se] - s_pb_q_m[sc])
      //                  - s_flag_u[sc]*s_flag_u[sc]*(s_pb_q_m[sc] - s_pb_q_m[sw])) +
      //            idy2 * (s_flag_v[sn]*s_flag_v[sn]*(s_pb_q_m[sn] - s_pb_q_m[sc])
      //                  - s_flag_v[sc]*s_flag_v[sc]*(s_pb_q_m[sc] - s_pb_q_m[ss])) +
      //            idz2 * (s_flag_w_t[sc]*s_flag_w_t[sc]*(s_pb_q_t[sc] - s_pb_q_m[sc])
      //                  - s_flag_w_b[sc]*s_flag_w_b[sc]*(s_pb_q_m[sc] - s_pb_q_b[sc]));


      /* Particle Modifications:
       *  Main diagonal:
       *  * If (phase > -1):
       *    - if (phase_shell == 0) (physalis bc applied)
       *    - if (phase_shell == 1) (zero     bc applied) 
       *    Boundary condition is known, set coeff = 1.
       *    - if (phase > -1), multiply diagonal by -dx^2/6 (or y, or z). 
       *    - - (+)dx^2/6 if (Ax = b)
       *    - if (phase == -1), multiply diagonal by 1
       *    Flag logic is:
       *    - pfc_x = -(_dom.dx * _dom.dx)/6. * (phase > -1) + (phase == -1)
       *
       *  Off-diagonals:
       *  * If (phase > -1), zero off-diagonals
       *  * If (phase == -1) and adjacent (phase > -1), zero off-diagonals
       *    - if (phase > -1), multiply by zero
       *    - if (phase == -1 && phase[adj] > -1), multiply by zero
       *    - else, multiply by 1
       *    Flag logic is
       *    - pfc_adj = (phase == -1) * !(phase == -1 && phase[adj] > -1)
       */


      /* Particle modification flag */
      real pfx = -(_dom.dx * _dom.dx)/6. * (s_phase_m[sc] > -1)  // set to -dx^2/6
                  + (real) (s_phase_m[sc] == -1);                // set to 1
      real pfy = -(_dom.dy * _dom.dy)/6. * (s_phase_m[sc] > -1)
                  + (real) (s_phase_m[sc] == -1);
      real pfz = -(_dom.dz * _dom.dz)/6. * (s_phase_m[sc] > -1)
                  + (real) (s_phase_m[sc] == -1);

      int pfe = (s_phase_m[sc] == -1) * !(s_phase_m[sc] == -1 && s_phase_m[se] > -1);
      int pfw = (s_phase_m[sc] == -1) * !(s_phase_m[sc] == -1 && s_phase_m[sw] > -1);
      int pfn = (s_phase_m[sc] == -1) * !(s_phase_m[sc] == -1 && s_phase_m[sn] > -1);
      int pfs = (s_phase_m[sc] == -1) * !(s_phase_m[sc] == -1 && s_phase_m[ss] > -1);
      int pft = (s_phase_m[sc] == -1) * !(s_phase_m[sc] == -1 && s_phase_t[sc] > -1);
      int pfb = (s_phase_m[sc] == -1) * !(s_phase_m[sc] == -1 && s_phase_b[sc] > -1);
      
      /* If solving -Ax = -b */
      Apb_q[cc] = -idx2 * (
                 s_flag_u[se]*s_flag_u[se]*(s_pb_q_m[se]*pfe - pfx*s_pb_q_m[sc])
               - s_flag_u[sc]*s_flag_u[sc]*(s_pb_q_m[sc]*pfx - pfw*s_pb_q_m[sw]));
      Apb_q[cc] += -idy2 * (
                 s_flag_v[sn]*s_flag_v[sn]*(s_pb_q_m[sn]*pfn - pfy*s_pb_q_m[sc])
               - s_flag_v[sc]*s_flag_v[sc]*(s_pb_q_m[sc]*pfy - pfs*s_pb_q_m[ss]));
      Apb_q[cc] += -idz2 * (
                 s_flag_w_t[sc]*s_flag_w_t[sc]*(s_pb_q_t[sc]*pft - pfz*s_pb_q_m[sc])
               - s_flag_w_b[sc]*s_flag_w_b[sc]*(s_pb_q_m[sc]*pfz - pfb*s_pb_q_b[sc]));
    }

    /* Ensure computation is complete */
    __syncthreads();
  }
}

__global__ void PP_spmv_shared_load_noparts(int *flag_u, int *flag_v,
  int *flag_w, real *pb_q, real *Apb_q)
{
  // num_kn_s, dim_kn_s

  /* Prefactors */
  real idx2 = 1./(_dom.dx * _dom.dx);
  real idy2 = 1./(_dom.dy * _dom.dy);
  real idz2 = 1./(_dom.dz * _dom.dz);

  /* Create shared memory */
  // pb_q and flag_w need 3 and 2 planes, respectively
  __shared__ int s_flag_u[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_v[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_w_t[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ int s_flag_w_b[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ real s_pb_q_t[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ real s_pb_q_m[MAX_THREADS_DIM * MAX_THREADS_DIM];
  __shared__ real s_pb_q_b[MAX_THREADS_DIM * MAX_THREADS_DIM];

  /* Shared memory indices */
  int si = threadIdx.x;         // x-position in 2D grid
  int sj = threadIdx.y;         // y-position in 2D grid
  int sc = si + sj*blockDim.x;  // indices in 1D array
  int sw = sc - 1;
  int se = sc + 1;
  int ss = sc - blockDim.x;
  int sn = sc + blockDim.x;

  /* s3b subdomain indices */
  // -2*BlockIdx.{x,y} corrects for shared memory overlap 
  int TI = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
  int TJ = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;
  int TK;                                 // calculate in loop

  /* s3 subdomain location */
  int cc;

  /* Load initial shared memory planes */
  if ((TI >= _dom.Gcc._isb && TI <= _dom.Gcc._ieb) &&
      (TJ >= _dom.Gcc._jsb && TJ <= _dom.Gcc._jeb)) {
    s_pb_q_t[sc] = pb_q[GCC_LOC(TI, TJ, _dom.Gcc._ks, _dom.Gcc.s1b, _dom.Gcc.s2b)];
    s_pb_q_m[sc] = pb_q[GCC_LOC(TI, TJ, _dom.Gcc._ksb, _dom.Gcc.s1b, _dom.Gcc.s2b)];
  }
  if ((TI >= _dom.Gfz._isb && TI <= _dom.Gfz._ieb) &&
      (TJ >= _dom.Gfz._jsb && TJ <= _dom.Gfz._jeb)) {
    s_flag_w_t[sc] = flag_w[GFZ_LOC(TI, TJ, _dom.Gfz._ks, _dom.Gfz.s1b, _dom.Gfz.s2b)];
  }

  /* Ensure initial load is done */
  __syncthreads();

  /* Loop over z planes and load shared memory */
  for (TK = _dom.Gcc._ks; TK <= _dom.Gcc._ke; TK++) {
    // pb_q -- top, middle, and bottom planes
    if ((TI >= _dom.Gcc._isb && TI <= _dom.Gcc._ieb) &&
        (TJ >= _dom.Gcc._jsb && TJ <= _dom.Gcc._jeb)) {
        // Reload bottom and middle from shared
        // Load top from global
        // 1. initial load of top <- ks
        //                    mid <- ksb
        // 2. thereafter, bot <- mid
        //                mid <- top
        //                top <- load
       s_pb_q_b[sc] = s_pb_q_m[sc];
       s_pb_q_m[sc] = s_pb_q_t[sc];
       s_pb_q_t[sc] = pb_q[GCC_LOC(TI, TJ, (TK + 1), _dom.Gcc.s1b, _dom.Gcc.s2b)];
     }

    // flag_u
    if ((TI >= _dom.Gfx._isb && TI <= _dom.Gfx._ieb) &&
        (TJ >= _dom.Gfx._jsb && TJ <= _dom.Gfx._jeb)) {
      s_flag_u[sc] = flag_u[GFX_LOC(TI, TJ, TK, _dom.Gfx.s1b, _dom.Gfx.s2b)];
    }

    // flag_v
    if ((TI >= _dom.Gfy._isb && TI <= _dom.Gfy._ieb) &&
        (TJ >= _dom.Gfy._jsb && TJ <= _dom.Gfy._jeb)) {
      s_flag_v[sc] = flag_v[GFY_LOC(TI, TJ, TK, _dom.Gfy.s1b, _dom.Gfy.s2b)];
    }

    // flag_w
    if ((TI >= _dom.Gfz._isb && TI <= _dom.Gfz._ieb) &&
        (TJ >= _dom.Gfz._jsb && TJ <= _dom.Gfz._jeb)) {
      // Reload memory from shared, not global, when possible
      s_flag_w_b[sc] = s_flag_w_t[sc];
      s_flag_w_t[sc] = flag_w[GFZ_LOC(TI, TJ, (TK + 1), _dom.Gfz.s1b, _dom.Gfz.s2b)];
    }

    /* Ensure shared memory load is complete */
    __syncthreads();

    /* Compute A*pb_q into global */
    if ((TI >= _dom.Gcc._is && TI <= _dom.Gcc._ie) &&
        (TJ >= _dom.Gcc._js && TJ <= _dom.Gcc._je) &&
        (si > 0 && si < (blockDim.x-1)) && (sj > 0 && sj < (blockDim.y-1))) {
      cc = GCC_LOC(TI - DOM_BUF, TJ - DOM_BUF, TK - DOM_BUF,
                  _dom.Gcc.s1, _dom.Gcc.s2);

      // abs(flag) = flag*flag. Do this here so it stays in register
      // If solving Ax = b
      //Apb_q[cc] = idx2 * (s_flag_u[se]*s_flag_u[se]*(s_pb_q_m[se] - s_pb_q_m[sc])
      //                  - s_flag_u[sc]*s_flag_u[sc]*(s_pb_q_m[sc] - s_pb_q_m[sw])) +
      //            idy2 * (s_flag_v[sn]*s_flag_v[sn]*(s_pb_q_m[sn] - s_pb_q_m[sc])
      //                  - s_flag_v[sc]*s_flag_v[sc]*(s_pb_q_m[sc] - s_pb_q_m[ss])) +
      //            idz2 * (s_flag_w_t[sc]*s_flag_w_t[sc]*(s_pb_q_t[sc] - s_pb_q_m[sc])
      //                  - s_flag_w_b[sc]*s_flag_w_b[sc]*(s_pb_q_m[sc] - s_pb_q_b[sc]));

      // If solving -Ax = -b
      Apb_q[cc] = -idx2 * (s_flag_u[se]*s_flag_u[se]*(s_pb_q_m[se] - s_pb_q_m[sc])
                         - s_flag_u[sc]*s_flag_u[sc]*(s_pb_q_m[sc] - s_pb_q_m[sw]))
                  -idy2 * (s_flag_v[sn]*s_flag_v[sn]*(s_pb_q_m[sn] - s_pb_q_m[sc])
                         - s_flag_v[sc]*s_flag_v[sc]*(s_pb_q_m[sc] - s_pb_q_m[ss]))
                  -idz2 * (s_flag_w_t[sc]*s_flag_w_t[sc]*(s_pb_q_t[sc] - s_pb_q_m[sc])
                         - s_flag_w_b[sc]*s_flag_w_b[sc]*(s_pb_q_m[sc] - s_pb_q_b[sc]));

    }

    /* Ensure computation is complete */
    __syncthreads();
  }
}

__global__ void PP_update_soln_resid(real *phi, real *p_q, real *r_q,
  real *Apb_q, real *z_q, real alpha, real *invM)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;
  int c, C;

  if (ti < _dom.Gcc.in && tj < _dom.Gcc.jn) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      c = GCC_LOC(ti, tj, k, _dom.Gcc.s1, _dom.Gcc.s2);
      C = GCC_LOC(ti + DOM_BUF, tj + DOM_BUF, k + DOM_BUF,
                  _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Update solution: phi += alpha*p_q  
      phi[C] += alpha*p_q[c];

      // Update residual: r_q -= alpha*Apb_q
      r_q[c] -= alpha*Apb_q[c]; 

      // Update aux vector: z_q = invM*r_q
      z_q[c] = r_q[c]*invM[c];
    }
  }
}

__global__ void PP_update_solution(real *phi, real *p_q, real alpha)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;
  int c, C;

  if (ti < _dom.Gcc.in && tj < _dom.Gcc.jn) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      c = GCC_LOC(ti, tj, k, _dom.Gcc.s1, _dom.Gcc.s2);
      C = GCC_LOC(ti + DOM_BUF, tj + DOM_BUF, k + DOM_BUF,
                  _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Update solution: phi += alpha*p_q  
      phi[C] += alpha*p_q[c];
    }
  }
}

__global__ void PP_update_residual(real *r_q, real *rhs, real *Apb_q,
  real *z_q, real *invM)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;
  int c, C;

  if (ti < _dom.Gcc.in && tj < _dom.Gcc.jn) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      c = GCC_LOC(ti, tj, k, _dom.Gcc.s1, _dom.Gcc.s2);
      C = GCC_LOC(ti + DOM_BUF, tj + DOM_BUF, k + DOM_BUF,
                  _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Update residual: r_q = (-b) - (-A)*phi
      r_q[c] = rhs[C] - Apb_q[c];

      // Update aux vector: z_q = invM*r_q
      z_q[c] = r_q[c]*invM[c];
    }
  }
}

__global__ void PP_update_search(real *p_q, real *pb_q, real *z_q,
  real beta)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  real new_p_q;

  if (ti < _dom.Gcc.in && tj < _dom.Gcc.jn) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      int cc = GCC_LOC(ti, tj, k, _dom.Gcc.s1, _dom.Gcc.s2);
      int CC = GCC_LOC(ti + DOM_BUF, tj + DOM_BUF, k + DOM_BUF,
                      _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Update solution: p_q = invM*z_q1 + beta*p_q
      new_p_q = z_q[cc] + beta*p_q[cc];
      p_q[cc] = new_p_q;
      pb_q[CC] = new_p_q;
    }
  }
}

__global__ void PP_bcgs_init(real *r_q, real *rs_0, real *p_q, real *pb_q,
  real *phi, real *rhs)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int c, C;
  real tmp;

  // Loop over z-planes
  if ((ti < _dom.Gcc.in) & (tj < _dom.Gcc.jn)) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      c = GCC_LOC(ti, tj , k, _dom.Gcc.s1, _dom.Gcc.s2);
      C = GCC_LOC(ti + DOM_BUF, tj + DOM_BUF, k + DOM_BUF, 
                  _dom.Gcc.s1b, _dom.Gcc.s2b);

      tmp = rhs[C];
      r_q[c] = tmp;
      rs_0[c] = tmp;
      p_q[c] = tmp;
      pb_q[C] = tmp;
      phi[c] = 0.;
    }
  }
}

__global__ void PP_find_second_search(real *s_q, real *sb_q, real *r_q, 
  real alpha, real *Apb_q)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;
  int c,C;
  real tmp;

  if (ti < _dom.Gcc.in && tj < _dom.Gcc.jn) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      c = GCC_LOC(ti, tj, k, _dom.Gcc.s1, _dom.Gcc.s2);
      C = GCC_LOC(ti + DOM_BUF, tj + DOM_BUF, k + DOM_BUF, 
                  _dom.Gcc.s1b, _dom.Gcc.s2b);

      // s_q = r_q - alpha * Apb_q
      tmp = r_q[c] - alpha * Apb_q[c];
      s_q[c] = tmp;
      sb_q[C] = tmp;
    }
  }
}

__global__ void PP_bcgs_update_soln_resid(real *phi, real alpha, real *p_q,
  real omega, real *s_q, real *r_q, real *Asb_q)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;
  int c;

  if (ti < _dom.Gcc.in && tj < _dom.Gcc.jn) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      c = GCC_LOC(ti, tj, k, _dom.Gcc.s1, _dom.Gcc.s2);

      // Update solution: phi += alpha*p_q + omega*s_q
      phi[c] += alpha*p_q[c] + omega*s_q[c];

      // Update residual: r_qp1 = s_q - omega*Asb_q
      r_q[c] = s_q[c] - omega*Asb_q[c];
    }
  }
}

__global__ void PP_bcgs_update_search(real *p_q, real *pb_q, real *r_q,
  real beta, real omega, real *Apb_q)
{
  // num_kn, dim_kn
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  real new_p_q;
  int cc, CC;

  if (ti < _dom.Gcc.in && tj < _dom.Gcc.jn) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      cc = GCC_LOC(ti, tj, k, _dom.Gcc.s1, _dom.Gcc.s2);
      CC = GCC_LOC(ti + DOM_BUF, tj + DOM_BUF, k + DOM_BUF,
                    _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Update solution: p_qp1 = r_qp1 + beta*(p_q - omega*Apb_q))
      new_p_q = r_q[cc] + beta*(p_q[cc] - omega*Apb_q[cc]); 
      p_q[cc] = new_p_q;
      pb_q[CC] = new_p_q;
    }
  }
}

__global__ void PP_subtract_mean(real *phi, real phi_mean)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  if (ti < _dom.Gcc.in && tj < _dom.Gcc.jn) {
    for (int k = 0; k < _dom.Gcc.kn; k++) {
      int cc = GCC_LOC(ti, tj, k, _dom.Gcc.s1, _dom.Gcc.s2);

      phi[cc] -= phi_mean;
    }
  }
}
