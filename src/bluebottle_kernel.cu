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

#include "cuda_bluebottle.h"

__global__ void BC_p_W_N(real *p)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    p[GCC_LOC(_dom.Gcc._isb, tj + 1, tk + 1, s1b, s2b)] = 
      p[GCC_LOC(_dom.Gcc._is, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_p_E_N(real *p)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((tj < _dom.Gcc.jn) && (tk < _dom.Gcc.kn))
    p[GCC_LOC(_dom.Gcc._ieb, tj + 1, tk + 1, s1b, s2b)] = 
      p[GCC_LOC(_dom.Gcc._ie, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_p_S_N(real *p)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    p[GCC_LOC(ti + 1, _dom.Gcc._jsb, tk + 1, s1b, s2b)] = 
      p[GCC_LOC(ti + 1, _dom.Gcc._js, tk + 1, s1b, s2b)];
}

__global__ void BC_p_N_N(real *p)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tk < _dom.Gcc.kn))
    p[GCC_LOC(ti + 1, _dom.Gcc._jeb, tk + 1, s1b, s2b)] = 
      p[GCC_LOC(ti + 1, _dom.Gcc._je, tk + 1, s1b, s2b)];
}

__global__ void BC_p_B_N(real *p)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b; 

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    p[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ksb, s1b, s2b)] = 
      p[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ks, s1b, s2b)];
}

__global__ void BC_p_T_N(real *p)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gcc.s1b;
  int s2b = _dom.Gcc.s2b;

  if ((ti < _dom.Gcc.in) && (tj < _dom.Gcc.jn))
    p[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._keb, s1b, s2b)] = 
      p[GCC_LOC(ti + 1, tj + 1, _dom.Gcc._ke, s1b, s2b)];
}

__global__ void BC_u_W_D(real *u, real bc)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((tj < _dom.Gfx.jn) && (tk < _dom.Gfx.kn)) {
    u[GFX_LOC(_dom.Gfx._isb, tj + 1, tk + 1, s1b, s2b)] = 
                 2.*bc - u[GFX_LOC(_dom.Gfx._is + 1, tj + 1, tk + 1, s1b, s2b)];
    u[GFX_LOC(_dom.Gfx._is, tj + 1, tk + 1, s1b, s2b)] = bc;
  }
}

__global__ void BC_u_E_D(real *u, real bc)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((tj < _dom.Gfx.jn) && (tk < _dom.Gfx.kn)) {
    u[GFX_LOC(_dom.Gfx._ieb, tj + 1, tk + 1, s1b, s2b)] = 
                 2.*bc - u[GFX_LOC(_dom.Gfx._ie - 1, tj + 1, tk + 1, s1b, s2b)];
    u[GFX_LOC(_dom.Gfx._ie, tj + 1, tk + 1, s1b, s2b)] = bc;
  }
}

__global__ void BC_u_N_D(real *u, real bc)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((ti < _dom.Gfx.in) && (tk < _dom.Gfx.kn)) {
    u[GFX_LOC(ti + 1, _dom.Gfx._jeb, tk + 1, s1b, s2b)] = 
          8./3.*bc - 2.*u[GFX_LOC(ti + 1, _dom.Gfx._je, tk + 1, s1b, s2b)]
                + 1./3.*u[GFX_LOC(ti + 1, _dom.Gfx._je - 1, tk + 1, s1b ,s2b)];
  }
}

__global__ void BC_u_S_D(real *u, real bc)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((ti < _dom.Gfx.in) && (tk < _dom.Gfx.kn)) {
    u[GFX_LOC(ti + 1, _dom.Gfx._jsb, tk + 1, s1b, s2b)] = 
          8./3.*bc - 2.*u[GFX_LOC(ti + 1, _dom.Gfx._js, tk + 1, s1b, s2b)]
                + 1./3.*u[GFX_LOC(ti + 1, _dom.Gfx._js + 1, tk + 1, s1b, s2b)];
  }
}

__global__ void BC_u_T_D(real *u, real bc)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((ti < _dom.Gfx.in) && (tj < _dom.Gfx.jn))
    u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._keb, s1b, s2b)] = 
          8./3.*bc - 2.*u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._ke, s1b, s2b)]
                + 1./3.*u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._ke - 1, s1b, s2b)];
}

__global__ void BC_u_B_D(real *u, real bc)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((ti < _dom.Gfx.in) && (tj < _dom.Gfx.jn))
    u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._ksb, s1b, s2b)] = 
          8./3.*bc - 2.*u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._ks, s1b, s2b)]
                + 1./3.*u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._ks + 1, s1b, s2b)];
}

__global__ void BC_u_W_N(real *u)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((tj < _dom.Gfx.jn) && (tk < _dom.Gfx.kn))
    u[GFX_LOC(_dom.Gfx._isb, tj + 1, tk + 1, s1b, s2b)] = 
      u[GFX_LOC(_dom.Gfx._is, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_u_E_N(real *u)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((tj < _dom.Gfx.jn) && (tk < _dom.Gfx.kn))
    u[GFX_LOC(_dom.Gfx._ieb, tj + 1, tk + 1, s1b, s2b)] = 
      u[GFX_LOC(_dom.Gfx._ie, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_u_S_N(real *u)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((ti < _dom.Gfx.in) && (tk < _dom.Gfx.kn))
    u[GFX_LOC(ti + 1, _dom.Gfx._jsb, tk + 1, s1b, s2b)] = 
      u[GFX_LOC(ti + 1, _dom.Gfx._js, tk + 1, s1b, s2b)];
}

__global__ void BC_u_N_N(real *u)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((ti < _dom.Gfx.in) && (tk < _dom.Gfx.kn))
    u[GFX_LOC(ti + 1, _dom.Gfx._jeb, tk + 1, s1b, s2b)] = 
      u[GFX_LOC(ti + 1, _dom.Gfx._je, tk + 1, s1b, s2b)];
}

__global__ void BC_u_B_N(real *u)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((ti < _dom.Gfx.in) && (tj < _dom.Gfx.jn))
    u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._ksb, s1b, s2b)] = 
      u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._ks, s1b, s2b)];
}

__global__ void BC_u_T_N(real *u)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfx.s1b;
  int s2b = _dom.Gfx.s2b;

  if ((ti < _dom.Gfx.in) && (tj < _dom.Gfx.jn))
    u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._keb, s1b, s2b)] = 
      u[GFX_LOC(ti + 1, tj + 1, _dom.Gfx._ke, s1b, s2b)];
}

__global__ void BC_v_W_D(real *v, real bc)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((tj < _dom.Gfy.jn) && (tk < _dom.Gfy.kn)) {
    v[GFY_LOC(_dom.Gfy._isb, tj + 1, tk + 1, s1b, s2b)] =
          8./3.*bc - 2.*v[GFY_LOC(_dom.Gfy._is, tj + 1, tk + 1, s1b, s2b)]
                + 1./3.*v[GFY_LOC(_dom.Gfy._is + 1, tj + 1, tk + 1, s1b, s2b)];
  }
}

__global__ void BC_v_E_D(real *v, real bc)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((tj < _dom.Gfy.jn) && (tk < _dom.Gfy.kn)) {
    v[GFY_LOC(_dom.Gfy._ieb, tj + 1, tk + 1, s1b, s2b)] = 
          8./3.*bc - 2.*v[GFY_LOC(_dom.Gfy._ie, tj + 1, tk + 1, s1b, s2b)]
                + 1./3.*v[GFY_LOC(_dom.Gfy._ie - 1, tj + 1, tk + 1, s1b, s2b)];
  }
}

__global__ void BC_v_S_D(real *v, real bc)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((ti < _dom.Gfy.in) && (tk < _dom.Gfy.kn)) {
    v[GFY_LOC(ti + 1, _dom.Gfy._jsb, tk + 1, s1b, s2b)] = 
                 2.*bc - v[GFY_LOC(ti + 1, _dom.Gfy._js + 1, tk + 1, s1b, s2b)];
    v[GFY_LOC(ti + 1, _dom.Gfy._js, tk + 1, s1b, s2b)] = bc;
  }
}

__global__ void BC_v_N_D(real *v, real bc)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((ti < _dom.Gfy.in) && (tk < _dom.Gfy.kn)) {
    v[GFY_LOC(ti + 1, _dom.Gfy._jeb, tk + 1, s1b, s2b)] = 
                 2.*bc - v[GFY_LOC(ti + 1, _dom.Gfy._je - 1, tk + 1, s1b, s2b)];
    v[GFY_LOC(ti + 1, _dom.Gfy._je, tk + 1, s1b, s2b)] = bc;
  }
}

__global__ void BC_v_B_D(real *v, real bc)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((ti < _dom.Gfy.in) && (tj < _dom.Gfy.jn))
    v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._ksb, s1b, s2b)] = 
          8./3.*bc - 2.*v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._ks, s1b, s2b)]
                + 1./3.*v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._ks + 1, s1b, s2b)];
}

__global__ void BC_v_T_D(real *v, real bc)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((ti < _dom.Gfy.in) && (tj < _dom.Gfy.jn))
    v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._keb, s1b, s2b)] = 
          8./3.*bc - 2.*v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._ke, s1b, s2b)]
                + 1./3.*v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._ke - 1, s1b, s2b)];
}

__global__ void BC_v_W_N(real *v)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((tj < _dom.Gfy.jn) && (tk < _dom.Gfy.kn))
    v[GFY_LOC(_dom.Gfy._isb, tj + 1, tk + 1, s1b, s2b)] = 
      v[GFY_LOC(_dom.Gfy._is, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_v_E_N(real *v)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((tj < _dom.Gfy.jn) && (tk < _dom.Gfy.kn))
    v[GFY_LOC(_dom.Gfy._ieb, tj + 1, tk + 1, s1b, s2b)] = 
      v[GFY_LOC(_dom.Gfy._ie, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_v_S_N(real *v)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((ti < _dom.Gfy.in) && (tk < _dom.Gfy.kn))
    v[GFY_LOC(ti + 1, _dom.Gfy._jsb, tk + 1, s1b, s2b)] = 
      v[GFY_LOC(ti + 1, _dom.Gfy._js, tk + 1, s1b, s2b)];
}

__global__ void BC_v_N_N(real *v)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((ti < _dom.Gfy.in) && (tk < _dom.Gfy.kn))
    v[GFY_LOC(ti + 1, _dom.Gfy._jeb, tk + 1, s1b, s2b)] = 
      v[GFY_LOC(ti + 1, _dom.Gfy._je, tk + 1, s1b, s2b)];
}

__global__ void BC_v_B_N(real *v)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((ti < _dom.Gfy.in) && (tj < _dom.Gfy.jn))
    v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._ksb, s1b, s2b)] = 
      v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._ks, s1b, s2b)];
}

__global__ void BC_v_T_N(real *v)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfy.s1b;
  int s2b = _dom.Gfy.s2b;

  if ((ti < _dom.Gfy.in) && (tj < _dom.Gfy.jn))
    v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._keb, s1b, s2b)] = 
      v[GFY_LOC(ti + 1, tj + 1, _dom.Gfy._ke, s1b, s2b)];
}

__global__ void BC_w_W_D(real *w, real bc)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((tj < _dom.Gfz.jn) && (tk < _dom.Gfz.kn))
    w[GFZ_LOC(_dom.Gfz._isb, tj + 1, tk + 1, s1b, s2b)] = 
          8./3.*bc - 2.*w[GFZ_LOC(_dom.Gfz._is, tj + 1, tk + 1, s1b, s2b)]
                + 1./3.*w[GFZ_LOC(_dom.Gfz._is + 1, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_w_E_D(real *w, real bc)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((tj < _dom.Gfz.jn) && (tk < _dom.Gfz.kn))
    w[GFZ_LOC(_dom.Gfz._ieb, tj + 1, tk + 1, s1b, s2b)] = 
          8./3.*bc - 2.*w[GFZ_LOC(_dom.Gfz._ie, tj + 1, tk + 1, s1b, s2b)]
                + 1./3.*w[GFZ_LOC(_dom.Gfz._ie - 1, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_w_S_D(real *w, real bc)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((ti < _dom.Gfz.in) && (tk < _dom.Gfz.kn))
    w[GFZ_LOC(ti + 1, _dom.Gfz._jsb, tk + 1, s1b, s2b)] = 
         8./3.*bc - 2.*w[GFZ_LOC(ti + 1, _dom.Gfz._js, tk + 1, s1b, s2b)]
               + 1./3.*w[GFZ_LOC(ti + 1, _dom.Gfz._js + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_w_N_D(real *w, real bc)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((ti < _dom.Gfz.in) && (tk < _dom.Gfz.kn))
    w[GFZ_LOC(ti + 1, _dom.Gfz._jeb, tk + 1, s1b, s2b)] = 
          8./3.*bc - 2.*w[GFZ_LOC(ti + 1, _dom.Gfz._je, tk + 1, s1b, s2b)]
                + 1./3.*w[GFZ_LOC(ti + 1, _dom.Gfz._je - 1, tk + 1, s1b, s2b)];
}

__global__ void BC_w_B_D(real *w, real bc)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((ti < _dom.Gfz.in) && (tj < _dom.Gfz.jn)) {
    w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._ksb, s1b, s2b)] = 
                2.*bc - w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._ks + 1, s1b, s2b)];
    w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._ks, s1b, s2b)] = bc;
  }
}

__global__ void BC_w_T_D(real *w, real bc)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((ti < _dom.Gfz.in) && (tj < _dom.Gfz.jn)) {
    w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._keb, s1b, s2b)] = 
                2.*bc - w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._ke - 1, s1b, s2b)];
    w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._ke, s1b, s2b)] = bc;
  }
}

__global__ void BC_w_W_N(real *w)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((tj < _dom.Gfz.jn) && (tk < _dom.Gfz.kn))
    w[GFZ_LOC(_dom.Gfz._isb, tj + 1, tk + 1, s1b, s2b)] = 
      w[GFZ_LOC(_dom.Gfz._is, tj + 1, tk + 1, s1b, s2b)];
}

__global__ void BC_w_E_N(real *w)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((tj < _dom.Gfz.jn) && (tk < _dom.Gfz.kn))
    w[GFZ_LOC(_dom.Gfz._ieb, tj + 1, tk + 1, s1b, s2b)] = 
      w[GFZ_LOC(_dom.Gfz._ie, tj + 1, tk + 1, s1b, s2b)]; 
}

__global__ void BC_w_S_N(real *w)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((ti < _dom.Gfz.in) && (tk < _dom.Gfz.kn))
    w[GFZ_LOC(ti + 1, _dom.Gfz._jsb, tk + 1, s1b, s2b)] = 
      w[GFZ_LOC(ti + 1, _dom.Gfz._js, tk + 1, s1b, s2b)];
}

__global__ void BC_w_N_N(real *w)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((ti < _dom.Gfz.in) && (tk < _dom.Gfz.kn))
    w[GFZ_LOC(ti + 1, _dom.Gfz._jeb, tk + 1, s1b, s2b)] = 
      w[GFZ_LOC(ti + 1, _dom.Gfz._je, tk + 1, s1b, s2b)];
}

__global__ void BC_w_B_N(real *w)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((ti < _dom.Gfz.in) && (tj < _dom.Gfz.jn))
    w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._ksb, s1b, s2b)] = 
      w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._ks, s1b, s2b)];
}

__global__ void BC_w_T_N(real *w)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = _dom.Gfz.s1b;
  int s2b = _dom.Gfz.s2b;

  if ((ti < _dom.Gfz.in) && (tj < _dom.Gfz.jn))
    w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._keb, s1b, s2b)] = 
      w[GFZ_LOC(ti + 1, tj + 1, _dom.Gfz._ke, s1b, s2b)];
}

__global__ void self_exchange_Gcc_i(real *array)
{
  // cuda threads -- add one since we don't exchange ghost cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  // ce -> bw
  // cw -> be
  int ce, cw;   // east/west computational index
  int be, bw;   // east/west ghost index

  if ((tj <= _dom.Gcc.jn) && (tk <= _dom.Gcc.kn)) {
    ce = GCC_LOC(_dom.Gcc._ie, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    cw = GCC_LOC(_dom.Gcc._is, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

    be = GCC_LOC(_dom.Gcc._ieb, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    bw = GCC_LOC(_dom.Gcc._isb, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

    // Transfer eastern computational to western ghost cell
    // ce -> bw
    array[bw] = array[ce];

    // Transfer western computational to eastern ghost cell
    // wc -> be
    array[be] = array[cw];
  }
}

__global__ void self_exchange_Gcc_j(real *array)
{
  // cuda threads -- add one since we don't exchange ghost cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  // cn -> bs
  // cs -> bn
  int cn, cs;   // north/south computational index
  int bn, bs;   // north/south ghost index

  if ((tk <= _dom.Gcc.kn) && (ti <= _dom.Gcc.in)) {
    cn = GCC_LOC(ti, _dom.Gcc._je, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    cs = GCC_LOC(ti, _dom.Gcc._js, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

    bn = GCC_LOC(ti, _dom.Gcc._jeb, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    bs = GCC_LOC(ti, _dom.Gcc._jsb, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

    // Transfer northern computational to southern ghost cell
    // cn -> bs
    array[bs] = array[cn];

    // Transfer southern computational to northern ghost cell
    // cs -> bn
    array[bn] = array[cs];
  }
}

__global__ void self_exchange_Gcc_k(real *array)
{
  // cuda threads -- add one since we don't exchange ghost cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  // ct -> bb
  // cb -> bt
  int ct, cb;   // top/bottom computational index
  int bt, bb;   // top/bottom ghost index

  if ((ti <= _dom.Gcc.in) && (tj <= _dom.Gcc.jn)) {
    ct = GCC_LOC(ti, tj, _dom.Gcc._ke, _dom.Gcc.s1b, _dom.Gcc.s2b);
    cb = GCC_LOC(ti, tj, _dom.Gcc._ks, _dom.Gcc.s1b, _dom.Gcc.s2b);

    bt = GCC_LOC(ti, tj, _dom.Gcc._keb, _dom.Gcc.s1b, _dom.Gcc.s2b);
    bb = GCC_LOC(ti, tj, _dom.Gcc._ksb, _dom.Gcc.s1b, _dom.Gcc.s2b);

    // Transfer top computational to bottom ghost cell
    // ct -> bb
    array[bb] = array[ct];

    // Transfer bottom computational to top ghost cell
    // cb -> bt
    array[bt] = array[cb];
  }
}

__global__ void pack_planes_Gcc_east(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tj <= _dom.Gcc._je) && (tk <= _dom.Gcc._ke)) {
    cc = GCC_LOC(_dom.Gcc._ie, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (tj - 1) + _dom.Gcc.jn * (tk - 1);

    package[pp] = contents[cc];
  }
}

__global__ void pack_planes_Gcc_west(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tj <= _dom.Gcc._je) && (tk <= _dom.Gcc._ke)) {
    cc = GCC_LOC(_dom.Gcc._is, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (tj - 1) + _dom.Gcc.jn * (tk - 1);
    package[pp] = contents[cc];
  }
}

__global__ void pack_planes_Gcc_north(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gcc._ke) && (ti <= _dom.Gcc._ie)) {
    cc = GCC_LOC(ti, _dom.Gcc._je, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (tk - 1) + _dom.Gcc.kn * (ti - 1);
    package[pp] = contents[cc];
  }
}

__global__ void pack_planes_Gcc_south(real *contents, real *package)
{
  // cuda threads -- add one sto get to computational cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gcc._ke) && (ti <= _dom.Gcc._ie)) {
    cc = GCC_LOC(ti, _dom.Gcc._js, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (tk - 1) + _dom.Gcc.kn * (ti - 1);
    package[pp] = contents[cc];
  }
}

__global__ void pack_planes_Gcc_top(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gcc._ie) && (tj <= _dom.Gcc._je)) {
    cc = GCC_LOC(ti, tj, _dom.Gcc._ke, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (ti - 1) + _dom.Gcc.in * (tj - 1);
    package[pp] = contents[cc];

  }
}

__global__ void pack_planes_Gcc_bottom(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gcc._ie) && (tj <= _dom.Gcc._je)) {
    cc = GCC_LOC(ti, tj, _dom.Gcc._ks, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (ti - 1) + _dom.Gcc.in * (tj - 1);
    package[pp] = contents[cc];
  }
}

__global__ void pack_planes_Gfx_east(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  // ie - 1 -> isb
  if ((tj <= _dom.Gfx._je) && (tk <= _dom.Gfx._ke)) {
    cfx = GFX_LOC(_dom.Gfx._ie - 1, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (tj - 1) + _dom.Gfx.jn * (tk - 1);

    package[pp] = contents[cfx];
  }
}

__global__ void pack_planes_Gfx_west(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  // is + 1 -> ieb
  if ((tj <= _dom.Gfx._je) && (tk <= _dom.Gfx._ke)) {
    cfx = GFX_LOC(_dom.Gfx._is + 1, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (tj - 1) + _dom.Gfx.jn * (tk - 1);
    package[pp] = contents[cfx];
  }
}

__global__ void pack_planes_Gfx_north(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfx._ke) && (ti <= _dom.Gfx._ie)) {
    cfx = GFX_LOC(ti, _dom.Gfx._je, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (tk - 1) + _dom.Gfx.kn * (ti - 1);
    package[pp] = contents[cfx];
  }
}

__global__ void pack_planes_Gfx_south(real *contents, real *package)
{
  // cuda threads -- add one sto get to computational cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfx._ke) && (ti <= _dom.Gfx._ie)) {
    cfx = GFX_LOC(ti,  _dom.Gfx._js, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (tk - 1) + _dom.Gfx.kn * (ti - 1);
    package[pp] = contents[cfx];
  }
}

__global__ void pack_planes_Gfx_top(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfx._ie) && (tj <= _dom.Gfx._je)) {
    cfx = GFX_LOC(ti, tj, _dom.Gfx._ke, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (ti - 1) + _dom.Gfx.in * (tj - 1);
    package[pp] = contents[cfx];

  }
}

__global__ void pack_planes_Gfx_bottom(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfx._ie) && (tj <= _dom.Gfx._je)) {
    cfx = GFX_LOC(ti, tj, _dom.Gfx._ks, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (ti - 1) + _dom.Gfx.in * (tj - 1);
    package[pp] = contents[cfx];
  }
}

__global__ void pack_planes_Gfy_east(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tj <= _dom.Gfy._je) && (tk <= _dom.Gfy._ke)) {
    cfy = GFY_LOC(_dom.Gfy._ie, tj, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (tj - 1) + _dom.Gfy.jn * (tk - 1);

    package[pp] = contents[cfy];
  }
}

__global__ void pack_planes_Gfy_west(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tj <= _dom.Gfy._je) && (tk <= _dom.Gfy._ke)) {
    cfy = GFY_LOC(_dom.Gfy._is, tj, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (tj - 1) + _dom.Gfy.jn * (tk - 1);

    package[pp] = contents[cfy];
  }
}

__global__ void pack_planes_Gfy_north(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  // je - 1 -> isb
  if ((tk <= _dom.Gfy._ke) && (ti <= _dom.Gfy._ie)) {
    cfy = GFY_LOC(ti, _dom.Gfy._je - 1, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (tk - 1) + _dom.Gfy.kn * (ti - 1);
    package[pp] = contents[cfy];
  }
}

__global__ void pack_planes_Gfy_south(real *contents, real *package)
{
  // cuda threads -- add one sto get to computational cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  // js + 1 -> ieb
  if ((tk <= _dom.Gfy._ke) && (ti <= _dom.Gfy._ie)) {
    cfy = GFY_LOC(ti,  _dom.Gfy._js + 1, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (tk - 1) + _dom.Gfy.kn * (ti - 1);
    package[pp] = contents[cfy];
  }
}

__global__ void pack_planes_Gfy_top(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfy._ie) && (tj <= _dom.Gfy._je)) {
    cfy = GFY_LOC(ti, tj, _dom.Gfy._ke, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (ti - 1) + _dom.Gfy.in * (tj - 1);
    package[pp] = contents[cfy];

  }
}

__global__ void pack_planes_Gfy_bottom(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfy._ie) && (tj <= _dom.Gfy._je)) {
    cfy = GFY_LOC(ti, tj, _dom.Gfy._ks, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (ti - 1) + _dom.Gfy.in * (tj - 1);
    package[pp] = contents[cfy];
  }
}

__global__ void pack_planes_Gfz_east(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tj <= _dom.Gfz._je) && (tk <= _dom.Gfz._ke)) {
    cfz = GFZ_LOC(_dom.Gfz._ie, tj, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (tj - 1) + _dom.Gfz.jn * (tk - 1);

    package[pp] = contents[cfz];
  }
}

__global__ void pack_planes_Gfz_west(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tj <= _dom.Gfz._je) && (tk <= _dom.Gfz._ke)) {
    cfz = GFZ_LOC(_dom.Gfz._is, tj, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (tj - 1) + _dom.Gfz.jn * (tk - 1);
    package[pp] = contents[cfz];
  }
}

__global__ void pack_planes_Gfz_north(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfz._ke) && (ti <= _dom.Gfz._ie)) {
    cfz = GFZ_LOC(ti, _dom.Gfz._je, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (tk - 1) + _dom.Gfz.kn * (ti - 1);
    package[pp] = contents[cfz];
  }
}

__global__ void pack_planes_Gfz_south(real *contents, real *package)
{
  // cuda threads -- add one sto get to computational cells
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfz._ke) && (ti <= _dom.Gfz._ie)) {
    cfz = GFZ_LOC(ti,  _dom.Gfz._js, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (tk - 1) + _dom.Gfz.kn * (ti - 1);
    package[pp] = contents[cfz];
  }
}

__global__ void pack_planes_Gfz_top(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  // ke - 1 -> ksb
  if ((ti <= _dom.Gfz._ie) && (tj <= _dom.Gfz._je)) {
    cfz = GFZ_LOC(ti, tj, _dom.Gfz._ke - 1, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (ti - 1) + _dom.Gfz.in * (tj - 1);
    package[pp] = contents[cfz];

  }
}

__global__ void pack_planes_Gfz_bottom(real *contents, real *package)
{
  // cuda threads -- add one to get to computational cells
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  // ks + 1 -> keb
  if ((ti <= _dom.Gfz._ie) && (tj <= _dom.Gfz._je)) {
    cfz = GFZ_LOC(ti, tj, _dom.Gfz._ks + 1, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (ti - 1) + _dom.Gfz.in * (tj - 1);
    package[pp] = contents[cfz];
  }
}

__global__ void unpack_planes_Gcc_east(real *contents, real *package)
{
  // cuda threads -- add one in j,k because no transfer there
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // contigous contents index
  int pp;   // discontigous package index

  if ((tj <= _dom.Gcc._je) && (tk <= _dom.Gcc._ke)) {
    cc = GCC_LOC(_dom.Gcc._ieb, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (tj - 1) + _dom.Gcc.jn * (tk - 1);
    contents[cc] = package[pp];
  }
}

__global__ void unpack_planes_Gcc_west(real *contents, real *package)
{
  // cuda threads -- add one in j,k because no transfer there
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // contigous contents index
  int pp;   // discontigous package index

  if ((tj <= _dom.Gcc._je) && (tk <= _dom.Gcc._ke)) {
    cc = GCC_LOC(_dom.Gcc._isb, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (tj - 1) + _dom.Gcc.jn * (tk - 1);
    contents[cc] = package[pp];
  }
}

__global__ void unpack_planes_Gcc_north(real *contents, real *package)
{
  // cuda threads -- add one in k,i because no transfer there
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gcc._ke) && (ti <= _dom.Gcc._ie)) {
    cc = GCC_LOC(ti, _dom.Gcc._jeb, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (tk - 1) + _dom.Gcc.kn * (ti - 1);
    contents[cc] = package[pp];
  }
}

__global__ void unpack_planes_Gcc_south(real *contents, real *package)
{
  // cuda threads -- add one in k,i because no transfer there
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gcc._ke) && (ti <= _dom.Gcc._ie)) {
    cc = GCC_LOC(ti, _dom.Gcc._jsb, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (tk - 1) + _dom.Gcc.kn * (ti - 1);
    contents[cc] = package[pp];
  }
}

__global__ void unpack_planes_Gcc_top(real *contents, real *package)
{
  // cuda threads -- add one in j,i because no transfer there
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gcc._ie) && (tj <= _dom.Gcc._je)) {
    cc = GCC_LOC(ti, tj, _dom.Gcc._keb, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (ti - 1) + _dom.Gcc.in * (tj - 1);
    contents[cc] = package[pp];
  }
}

__global__ void unpack_planes_Gcc_bottom(real *contents, real *package)
{
  // cuda threads -- add one in j,i because no transfer there
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cc;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gcc._ie) && (tj <= _dom.Gcc._je)) {
    cc = GCC_LOC(ti, tj, _dom.Gcc._ksb, _dom.Gcc.s1b, _dom.Gcc.s2b);
    pp = (ti - 1) + _dom.Gcc.in * (tj - 1);
    contents[cc] = package[pp];
  }
}

__global__ void unpack_planes_Gfx_east(real *contents, real *package)
{
  // cuda threads -- add one in j,k because no transfer there
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // contigous contents index
  int pp;   // discontigous package index

  if ((tj <= _dom.Gfx._je) && (tk <= _dom.Gfx._ke)) {
    cfx = GFX_LOC(_dom.Gfx._ieb, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (tj - 1) + _dom.Gfx.jn * (tk - 1);
    contents[cfx] = package[pp];
  }
}

__global__ void unpack_planes_Gfx_west(real *contents, real *package)
{
  // cuda threads -- add one in j,k because no transfer there
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // contigous contents index
  int pp;   // discontigous package index

  if ((tj <= _dom.Gfx._je) && (tk <= _dom.Gfx._ke)) {
    cfx = GFX_LOC(_dom.Gfx._isb, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (tj - 1) + _dom.Gfx.jn * (tk - 1);
    contents[cfx] = package[pp];
  }
}

__global__ void unpack_planes_Gfx_north(real *contents, real *package)
{
  // cuda threads -- add one in k,i because no transfer there
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfx._ke) && (ti <= _dom.Gfx._ie)) {
    cfx = GFX_LOC(ti, _dom.Gfx._jeb, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (tk - 1) + _dom.Gfx.kn * (ti - 1);
    contents[cfx] = package[pp];
  }
}

__global__ void unpack_planes_Gfx_south(real *contents, real *package)
{
  // cuda threads -- add one in k,i because no transfer there
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfx._ke) && (ti <= _dom.Gfx._ie)) {
    cfx = GFX_LOC(ti, _dom.Gfx._jsb, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (tk - 1) + _dom.Gfx.kn * (ti - 1);
    contents[cfx] = package[pp];
  }
}

__global__ void unpack_planes_Gfx_top(real *contents, real *package)
{
  // cuda threads -- add one in j,i because no transfer there
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfx._ie) && (tj <= _dom.Gfx._je)) {
    cfx = GFX_LOC(ti, tj, _dom.Gfx._keb, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (ti - 1) + _dom.Gfx.in * (tj - 1);
    contents[cfx] = package[pp];
  }
}

__global__ void unpack_planes_Gfx_bottom(real *contents, real *package)
{
  // cuda threads -- add one in j,i because no transfer there
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfx;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfx._ie) && (tj <= _dom.Gfx._je)) {
    cfx = GFX_LOC(ti, tj, _dom.Gfx._ksb, _dom.Gfx.s1b, _dom.Gfx.s2b);
    pp = (ti - 1) + _dom.Gfx.in * (tj - 1);
    contents[cfx] = package[pp];
  }
}

__global__ void unpack_planes_Gfy_east(real *contents, real *package)
{
  // cuda threads -- add one in j,k because no transfer there
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // contigous contents index
  int pp;   // discontigous package index

  if ((tj <= _dom.Gfy._je) && (tk <= _dom.Gfy._ke)) {
    cfy = GFY_LOC(_dom.Gfy._ieb, tj, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (tj - 1) + _dom.Gfy.jn * (tk - 1);
    contents[cfy] = package[pp];
  }
}

__global__ void unpack_planes_Gfy_west(real *contents, real *package)
{
  // cuda threads -- add one in j,k because no transfer there
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // contigous contents index
  int pp;   // discontigous package index

  if ((tj <= _dom.Gfy._je) && (tk <= _dom.Gfy._ke)) {
    cfy = GFY_LOC(_dom.Gfy._isb, tj, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (tj - 1) + _dom.Gfy.jn * (tk - 1);

    contents[cfy] = package[pp];
  }
}

__global__ void unpack_planes_Gfy_north(real *contents, real *package)
{
  // cuda threads -- add one in k,i because no transfer there
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfy._ke) && (ti <= _dom.Gfy._ie)) {
    cfy = GFY_LOC(ti, _dom.Gfy._jeb, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (tk - 1) + _dom.Gfy.kn * (ti - 1);
    contents[cfy] = package[pp];
  }
}

__global__ void unpack_planes_Gfy_south(real *contents, real *package)
{
  // cuda threads -- add one in k,i because no transfer there
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfy._ke) && (ti <= _dom.Gfy._ie)) {
    cfy = GFY_LOC(ti, _dom.Gfy._jsb, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (tk - 1) + _dom.Gfy.kn * (ti - 1);
    contents[cfy] = package[pp];
  }
}

__global__ void unpack_planes_Gfy_top(real *contents, real *package)
{
  // cuda threads -- add one in j,i because no transfer there
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfy._ie) && (tj <= _dom.Gfy._je)) {
    cfy = GFY_LOC(ti, tj, _dom.Gfy._keb, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (ti - 1) + _dom.Gfy.in * (tj - 1);
    contents[cfy] = package[pp];
  }
}

__global__ void unpack_planes_Gfy_bottom(real *contents, real *package)
{
  // cuda threads -- add one in j,i because no transfer there
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfy;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfy._ie) && (tj <= _dom.Gfy._je)) {
    cfy = GFY_LOC(ti, tj, _dom.Gfy._ksb, _dom.Gfy.s1b, _dom.Gfy.s2b);
    pp = (ti - 1) + _dom.Gfy.in * (tj - 1);
    contents[cfy] = package[pp];
  }
}

__global__ void unpack_planes_Gfz_east(real *contents, real *package)
{
  // cuda threads -- add one in j,k because no transfer there
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // contigous contents index
  int pp;   // discontigous package index

  if ((tj <= _dom.Gfz._je) && (tk <= _dom.Gfz._ke)) {
    cfz = GFZ_LOC(_dom.Gfz._ieb, tj, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (tj - 1) + _dom.Gfz.jn * (tk - 1);
    contents[cfz] = package[pp];
  }
}

__global__ void unpack_planes_Gfz_west(real *contents, real *package)
{
  // cuda threads -- add one in j,k because no transfer there
  int tj = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tk = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // contigous contents index
  int pp;   // discontigous package index

  if ((tj <= _dom.Gfz._je) && (tk <= _dom.Gfz._ke)) {
    cfz = GFZ_LOC(_dom.Gfz._isb, tj, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (tj - 1) + _dom.Gfz.jn * (tk - 1);
    contents[cfz] = package[pp];
  }
}

__global__ void unpack_planes_Gfz_north(real *contents, real *package)
{
  // cuda threads -- add one in k,i because no transfer there
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfz._ke) && (ti <= _dom.Gfz._ie)) {
    cfz = GFZ_LOC(ti, _dom.Gfz._jeb, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (tk - 1) + _dom.Gfz.kn * (ti - 1);
    contents[cfz] = package[pp];
  }
}

__global__ void unpack_planes_Gfz_south(real *contents, real *package)
{
  // cuda threads -- add one in k,i because no transfer there
  int tk = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int ti = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((tk <= _dom.Gfz._ke) && (ti <= _dom.Gfz._ie)) {
    cfz = GFZ_LOC(ti, _dom.Gfz._jsb, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (tk - 1) + _dom.Gfz.kn * (ti - 1);
    contents[cfz] = package[pp];
  }
}

__global__ void unpack_planes_Gfz_top(real *contents, real *package)
{
  // cuda threads -- add one in j,i because no transfer there
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfz._ie) && (tj <= _dom.Gfz._je)) {
    cfz = GFZ_LOC(ti, tj, _dom.Gfz._keb, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (ti - 1) + _dom.Gfz.in * (tj - 1);
    contents[cfz] = package[pp];
  }
}

__global__ void unpack_planes_Gfz_bottom(real *contents, real *package)
{
  // cuda threads -- add one in j,i because no transfer there
  int ti = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  int tj = 1 + blockDim.y * blockIdx.y + threadIdx.y;

  int cfz;   // discontiguous contents index
  int pp;   // contiguous package index

  if ((ti <= _dom.Gfz._ie) && (tj <= _dom.Gfz._je)) {
    cfz = GFZ_LOC(ti, tj, _dom.Gfz._ksb, _dom.Gfz.s1b, _dom.Gfz.s2b);
    pp = (ti - 1) + _dom.Gfz.in * (tj - 1);
    contents[cfz] = package[pp];
  }
}

__global__ void forcing_reset_x(real *fx)
{
  int i;
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  if (tj < _dom.Gfx.jnb && tk < _dom.Gfx.knb) {
    for (i = _dom.Gfx._isb; i <= _dom.Gfx._ieb; i++) {
      fx[GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)] = 0.;
    }
  }
}

__global__ void forcing_reset_y(real *fy)
{
  int j;
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  if (tk < _dom.Gfy.knb && ti < _dom.Gfy.inb) {
    for (j = _dom.Gfy._jsb; j <= _dom.Gfy._jeb; j++) {
      fy[GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b)] = 0.;
    }
  }
}

__global__ void forcing_reset_z(real *fz)
{
  int k;
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  if (ti < _dom.Gfz.inb && tj < _dom.Gfz.jnb) {
    for (k = _dom.Gfz._ksb; k <= _dom.Gfz._keb; k++) {
      fz[GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b)] = 0.;
    }
  }
}

__global__ void forcing_add_c_const(real val, real *cc)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tj = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;
  int k;

  if (ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for (k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      cc[GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b)] += val;
    }
  }
}

__global__ void forcing_add_x_const(real val, real *fx)
{
  int i;
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  if (tj < _dom.Gfx.jnb && tk < _dom.Gfx.knb) {
    for (i = _dom.Gfx._isb; i <= _dom.Gfx._ieb; i++) {
      fx[GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)] += val;
    }
  }
}

__global__ void forcing_add_y_const(real val, real *fy)
{
  int j;
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  if (tk < _dom.Gfy.knb && ti < _dom.Gfy.inb) {
    for (j = _dom.Gfy._jsb; j <= _dom.Gfy._jeb; j++) {
      fy[GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b)] += val;
    }
  }
}

__global__ void forcing_add_z_const(real val, real *fz)
{
  int k;
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  if (ti < _dom.Gfz.inb && tj < _dom.Gfz.jnb) {
    for (k = _dom.Gfz._ksb; k <= _dom.Gfz._keb; k++) {
      fz[GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b)] += val;
    }
  }
}

__global__ void forcing_add_x_field(real scale, real *val, real *fx)
{
  int i;
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  if (tj < _dom.Gfx.jnb && tk < _dom.Gfx.knb) {
    for (i = _dom.Gfx._isb; i <= _dom.Gfx._ieb; i++) {
      int c = GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b); 
      fx[c] += scale * val[c];
    }
  }
}

__global__ void forcing_add_y_field(real scale, real *val, real *fy)
{
  int j;
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  if (tk < _dom.Gfy.knb && ti < _dom.Gfy.inb) {
    for (j = _dom.Gfy._jsb; j <= _dom.Gfy._jeb; j++) {
      int c = GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
      fy[c] += scale * val[c];
    }
  }
}

__global__ void forcing_add_z_field(real scale, real *val, real *fz)
{
  int k;
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  if (ti < _dom.Gfz.inb && tj < _dom.Gfz.jnb) {
    for (k = _dom.Gfz._ksb; k <= _dom.Gfz._keb; k++) {
      int c = GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
      fz[c] += scale * val[c];
    }
  }
}

__global__ void pull_wdot(real *wdot, part_struct *parts, int *bin_start,
  int *bin_count, int *part_ind)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int c;          // bin index (s3)
  int cbin;       // bin index (s3b)
  int pp;         // particle index

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;
  
  int s1 = _bins.Gcc.jn;
  int s2 = s1 * _bins.Gcc.kn;

  if (tj < _bins.Gcc.jn && tk < _bins.Gcc.kn) {
    for (int i = _bins.Gcc._is; i <= _bins.Gcc._ie; i++) {
      cbin = GFX_LOC(i, tj + DOM_BUF, tk + DOM_BUF, s1b, s2b);

      c = GFX_LOC(i - DOM_BUF, tj, tk, s1, s2);
      wdot[c] = 0.;

      // Loop through each bin's particles and add to send_parts
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];

        wdot[c] += parts[pp].wdot;
      }
    }
  }
}

__global__ void calc_u_star(real rho_f, real nu, real *u0, real *v0, real *w0, 
  real *p, real *f, real *diff0, real *conv0, real *diff, real *conv, 
  real *u_star, real dt0, real dt, int *phase)
{
  /* create shared memory */
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_u0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // u back
  __shared__ real s_u1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // u center
  __shared__ real s_u2[MAX_THREADS_DIM * MAX_THREADS_DIM];      // u forward
  __shared__ real s_v01[MAX_THREADS_DIM * MAX_THREADS_DIM];     // v back
  __shared__ real s_v12[MAX_THREADS_DIM * MAX_THREADS_DIM];     // v forward
  __shared__ real s_w01[MAX_THREADS_DIM * MAX_THREADS_DIM];     // w back
  __shared__ real s_w12[MAX_THREADS_DIM * MAX_THREADS_DIM];     // w forward
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv
  __shared__ real s_u_star[MAX_THREADS_DIM * MAX_THREADS_DIM];  // solution

  /* Shared memory indices */
  int tj = threadIdx.x;
  int tk = threadIdx.y;
  int sc = tj + tk*blockDim.x;

  /* working constants */
  real ab0 = 0.5 * dt / dt0;   // for Adams-Bashforth stepping
  real ab = 1. + ab0;          // for Adams-Bashforth stepping
  real ddx = 1. / _dom.dx;     // to limit the number of divisions needed
  real ddy = 1. / _dom.dy;     // to limit the number of divisions needed
  real ddz = 1. / _dom.dz;     // to limit the number of divisions needed

  /* s3b subdomain indices */
  // the extra 2*blockIdx.X terms implement the necessary overlapping of shmem
  int j = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
  int k = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;

  /* loop over u-planes and load shared memory */
  for (int i = _dom.Gfx._is; i <= _dom.Gfx._ie; i++) {
    // u
    if ((k >= _dom.Gfx._ksb && k <= _dom.Gfx._keb) &&
        (j >= _dom.Gfx._jsb && j <= _dom.Gfx._jeb)) {
      s_u0[sc] = u0[GFX_LOC(i - 1, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)];
      s_u1[sc] = u0[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)];
      s_u2[sc] = u0[GFX_LOC(i + 1, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)];
    }

    // v
    if ((k >= _dom.Gfy._ksb && k <= _dom.Gfy._keb) &&
        (j >= _dom.Gfy._jsb && j <= _dom.Gfy._jeb)) {
      s_v01[sc] = v0[GFY_LOC(i - 1, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)];
      s_v12[sc] = v0[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)];
    }

    // w
    if ((k >= _dom.Gfz._ksb && k <= _dom.Gfz._keb) &&
        (j >= _dom.Gfz._jsb && j <= _dom.Gfz._jeb)) {
      s_w01[sc] = w0[GFZ_LOC(i - 1, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)];
      s_w12[sc] = w0[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)];
    }

    s_u_star[sc] = 0.0;

    // make sure all threads complete shared memory copy
    __syncthreads();

    /* compute right-hand side */
    // if off the shared memory block boundary
    if ((tj > 0 && tj < blockDim.x-1) && (tk > 0 && tk < blockDim.y-1) &&
        (k >= _dom.Gfx._ks && k <= _dom.Gfx._ke) &&
        (j >= _dom.Gfx._js && j <= _dom.Gfx._je)) {

      // pressure gradient
      s_u_star[sc] = (p[GCC_LOC(i - 1, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b)]
                 - p[GCC_LOC(i, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b)]) * ddx/rho_f;

      // grab the required data points for calculations
      real u011 = s_u0[sc];
      real u111 = s_u1[sc];
      real u211 = s_u2[sc];

      real u101 = s_u1[(tj - 1) + tk*blockDim.x];
      real u121 = s_u1[(tj + 1) + tk*blockDim.x];
      real v011 = s_v01[sc];
      real v111 = s_v12[sc];
      real v021 = s_v01[(tj + 1) + tk*blockDim.x];
      real v121 = s_v12[(tj + 1) + tk*blockDim.x];

      real u110 = s_u1[tj + (tk - 1)*blockDim.x];
      real u112 = s_u1[tj + (tk + 1)*blockDim.x];
      real w011 = s_w01[sc];
      real w111 = s_w12[sc];
      real w012 = s_w01[tj + (tk + 1)*blockDim.x];
      real w112 = s_w12[tj + (tk + 1)*blockDim.x];

      // compute convection term (Adams-Bashforth stepping)
      real duudx = (u211 + u111)*(u211 + u111) - (u111 + u011)*(u111 + u011);
      duudx *= 0.25 * ddx;

      real duvdy = (u121 + u111)*(v121 + v021) - (u111 + u101)*(v111 + v011);
      duvdy *= 0.25 * ddy;

      real duwdz = (u112 + u111)*(w112 + w012) - (u111 + u110)*(w111 + w011);
      duwdz *= 0.25 * ddz;

      s_c[sc] = duudx + duvdy + duwdz;

      // convection term sums into right-hand side
#ifndef STOKESFLOW
      if (dt0 > 0) // Adams-Bashforth
        s_u_star[sc] += (-ab * s_c[sc]
          + ab0 * conv0[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)]);
      else        // forward Euler
        s_u_star[sc] += -s_c[sc];
#endif

      // compute diffusion term (Adams-Bashforth stepping)
      real dud1 = (u211 - u111) * ddx;
      real dud0 = (u111 - u011) * ddx;
      real ddudxx = (dud1 - dud0) * ddx;

      dud1 = (u121 - u111) * ddy;
      dud0 = (u111 - u101) * ddy;
      real ddudyy = (dud1 - dud0) * ddy;

      dud1 = (u112 - u111) * ddz;
      dud0 = (u111 - u110) * ddz;
      real ddudzz = (dud1 - dud0) * ddz;

      s_d[sc] = nu * (ddudxx + ddudyy + ddudzz);

      // diffusive term sums into right-hand side
      if (dt0 > 0) // Adams-Bashforth
        s_u_star[sc] += (ab * s_d[sc]
          - ab0 * diff0[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)]);
      else
        s_u_star[sc] += s_d[sc];

      // add on imposed pressure gradient
      s_u_star[sc] += f[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)];

      // multiply by dt
      s_u_star[sc] *= dt;

      // velocity term sums into right-hand side
      s_u_star[sc] += u111;

      // zero contribution inside particles
      s_u_star[sc] *=
        (phase[GCC_LOC(i-1, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b)] < 0
        && phase[GCC_LOC(i, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b)] < 0);
    }

    // make sure all threads complete computations
    __syncthreads();

    // copy shared memory back to global
    if ((k >= _dom.Gfx._ks && k <= _dom.Gfx._ke) &&
        (j >= _dom.Gfx._js && j <= _dom.Gfx._je) &&
        (tj > 0 && tj < (blockDim.x-1)) && (tk > 0 && tk < (blockDim.y-1))) {

      u_star[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)] = s_u_star[sc];
      conv[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)] = s_c[sc];
      diff[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)] = s_d[sc];
    }
  }
}

__global__ void calc_v_star(real rho_f, real nu, real *u0, real *v0, real *w0, 
  real *p, real *f, real *diff0, real *conv0, real *diff, real *conv, 
  real *v_star, real dt0, real dt, int *phase)
{
  /* create shared memory */
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_v0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // v back
  __shared__ real s_v1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // v center
  __shared__ real s_v2[MAX_THREADS_DIM * MAX_THREADS_DIM];      // v forward
  __shared__ real s_w01[MAX_THREADS_DIM * MAX_THREADS_DIM];     // w back
  __shared__ real s_w12[MAX_THREADS_DIM * MAX_THREADS_DIM];     // w forward
  __shared__ real s_u01[MAX_THREADS_DIM * MAX_THREADS_DIM];     // u back
  __shared__ real s_u12[MAX_THREADS_DIM * MAX_THREADS_DIM];     // u forward
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv
  __shared__ real s_v_star[MAX_THREADS_DIM * MAX_THREADS_DIM];  // solution

  /* shared memory indices */
  int tk = threadIdx.x;
  int ti = threadIdx.y;
  int sc = tk + ti*blockDim.x;

  /* working constants */
  real ab0 = 0.5 * dt / dt0;   // for Adams-Bashforth stepping
  real ab = 1. + ab0;          // for Adams-Bashforth stepping
  real ddx = 1. / _dom.dx;     // to limit the number of divisions needed
  real ddy = 1. / _dom.dy;     // to limit the number of divisions needed
  real ddz = 1. / _dom.dz;     // to limit the number of divisions needed

  /* subdomain indices */
  // the extra 2*blockIdx.X terms implement the necessary overlapping of shmem
  int k = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;

  /* loop over v-planes and load shmem */
  for (int j = _dom.Gfy._js; j <= _dom.Gfy._je; j++) {
    // v
    if ((i >= _dom.Gfy._isb && i <= _dom.Gfy._ieb) &&
        (k >= _dom.Gfy._ksb && k <= _dom.Gfy._keb)) {
      s_v0[sc] = v0[GFY_LOC(i, j - 1, k, _dom.Gfy.s1b, _dom.Gfy.s2b)];
      s_v1[sc] = v0[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)];
      s_v2[sc] = v0[GFY_LOC(i, j + 1, k, _dom.Gfy.s1b, _dom.Gfy.s2b)];
    }
    
    // w
    if ((i >= _dom.Gfz._isb && i <= _dom.Gfz._ieb) &&
        (k >= _dom.Gfz._ksb && k <= _dom.Gfz._keb)) {
      s_w01[sc] = w0[GFZ_LOC(i, j - 1, k, _dom.Gfz.s1b, _dom.Gfz.s2b)];
      s_w12[sc] = w0[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)];
    }

    // u
    if ((i >= _dom.Gfx._isb && i <= _dom.Gfx._ieb) &&
        (k >= _dom.Gfx._ksb && k <= _dom.Gfx._keb)) {
      s_u01[sc] = u0[GFX_LOC(i, j - 1, k, _dom.Gfx.s1b, _dom.Gfx.s2b)];
      s_u12[sc] = u0[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)];
    }

    s_v_star[sc] = 0.0;

    // make sure all threads complete shared memory copy
    __syncthreads();

    /* compute right-hand side */
    // if off the shared memory block boundary
    if ((tk > 0 && tk < blockDim.x-1) && (ti > 0 && ti < blockDim.y-1) &&
        (k >= _dom.Gfy._ksb && k <= _dom.Gfy._keb) &&
        (i >= _dom.Gfy._isb && i <= _dom.Gfy._ieb)) {

      // pressure gradient
      s_v_star[sc] = (p[GCC_LOC(i, j - 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b)]
                 - p[GCC_LOC(i, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b)]) * ddy/rho_f;

      // grab the required data points for calculations
      real v101 = s_v0[sc];
      real v111 = s_v1[sc];
      real v121 = s_v2[sc];

      real v110 = s_v1[(tk - 1) + ti*blockDim.x];
      real v112 = s_v1[(tk + 1) + ti*blockDim.x];
      real w101 = s_w01[sc];
      real w111 = s_w12[sc];
      real w102 = s_w01[(tk + 1) + ti*blockDim.x];
      real w112 = s_w12[(tk + 1) + ti*blockDim.x];

      real v011 = s_v1[tk + (ti - 1)*blockDim.x];
      real v211 = s_v1[tk + (ti + 1)*blockDim.x];
      real u101 = s_u01[sc];
      real u111 = s_u12[sc];
      real u201 = s_u01[tk + (ti + 1)*blockDim.x];
      real u211 = s_u12[tk + (ti + 1)*blockDim.x];

      // compute convection term (Adams-Bashforth stepping)
      real dvudx = (v211 + v111)*(u211 + u201) - (v111 + v011)*(u111 + u101);
      dvudx *= 0.25 * ddx;

      real dvvdy = (v121 + v111)*(v121 + v111) - (v111 + v101)*(v111 + v101);
      dvvdy *= 0.25 * ddy;

      real dvwdz = (v112 + v111)*(w112 + w102) - (v111 + v110)*(w111 + w101);
      dvwdz *= 0.25 * ddz;

      s_c[sc] = dvudx + dvvdy + dvwdz;

      // convection term sums into right-hand side
#ifndef STOKESFLOW
      if (dt0 > 0) // Adams-Bashforth
        s_v_star[sc] += (-ab * s_c[sc]
          + ab0 * conv0[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)]);
      else
        s_v_star[sc] += -s_c[sc];
#endif

      // compute diffusive term
      real dvd1 = (v211 - v111) * ddx;
      real dvd0 = (v111 - v011) * ddx;
      real ddvdxx = (dvd1 - dvd0) * ddx;

      dvd1 = (v121 - v111) * ddy;
      dvd0 = (v111 - v101) * ddy;
      real ddvdyy = (dvd1 - dvd0) * ddy;

      dvd1 = (v112 - v111) * ddz;
      dvd0 = (v111 - v110) * ddz;
      real ddvdzz = (dvd1 - dvd0) * ddz;

      s_d[sc] = nu * (ddvdxx + ddvdyy + ddvdzz);

      // diffusive term sums into right-hand side
      if (dt0 > 0) // Adams-Bashforth
        s_v_star[sc] += (ab * s_d[sc]
          - ab0 * diff0[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)]);
      else
        s_v_star[sc] += s_d[sc];

      // add on imposed pressure gradient
      s_v_star[sc] += f[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)];

      // multiply by dt
      s_v_star[sc] *= dt;

      // velocity term sums into right-hand side
      s_v_star[sc] += v111;

      // zero contribution inside particles
      s_v_star[sc] *=
        (phase[GCC_LOC(i, j-1, k, _dom.Gcc.s1b, _dom.Gcc.s2b)] < 0
        && phase[GCC_LOC(i, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b)] < 0);
    }

    // make sure all threads complete computations
    __syncthreads();

    // copy shared memory back to global
    if ((i >= _dom.Gfy._is && i <= _dom.Gfy._ie) &&
        (k >= _dom.Gfy._ks && k <= _dom.Gfy._ke) &&
        (tk > 0 && tk < (blockDim.x-1)) && (ti > 0 && ti < (blockDim.y-1))) {

      v_star[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)] = s_v_star[sc];
      conv[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)] = s_c[sc];
      diff[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)] = s_d[sc];
    }
  }
}

__global__ void calc_w_star(real rho_f, real nu, real *u0, real *v0, real *w0, 
  real *p, real *f, real *diff0, real *conv0, real *diff, real *conv, 
  real *w_star, real dt0, real dt, int *phase)
{
  /* create shared memory */
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_w0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w back
  __shared__ real s_w1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w center
  __shared__ real s_w2[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w forward
  __shared__ real s_u01[MAX_THREADS_DIM * MAX_THREADS_DIM];     // u back
  __shared__ real s_u12[MAX_THREADS_DIM * MAX_THREADS_DIM];     // u forward
  __shared__ real s_v01[MAX_THREADS_DIM * MAX_THREADS_DIM];     // v back
  __shared__ real s_v12[MAX_THREADS_DIM * MAX_THREADS_DIM];     // v forward
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff0
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv0
  __shared__ real s_w_star[MAX_THREADS_DIM * MAX_THREADS_DIM];  // solution

  /* Shared memory indices */
  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int sc = ti + tj*blockDim.x;

  // working constants
  real ab0 = 0.5 * dt / dt0;   // for Adams-Bashforth stepping
  real ab = 1. + ab0;          // for Adams-Bashforth stepping
  real ddx = 1. / _dom.dx;     // to limit the number of divisions needed
  real ddy = 1. / _dom.dy;     // to limit the number of divisions needed
  real ddz = 1. / _dom.dz;     // to limit the number of divisions needed

  /* subdomain indices */
  // the extra 2*blockIdx.X terms implement the necessary overlapping of shmem
  int i = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;

  /* loop over w-planes and load shared memory */
  for (int k = _dom.Gfz._ks; k <= _dom.Gfz._ke; k++) {
    // w
    if ((j >= _dom.Gfz._jsb && j <= _dom.Gfz._jeb) &&
        (i >= _dom.Gfz._isb && i <= _dom.Gfz._ieb)) {
      s_w0[sc] = w0[GFZ_LOC(i, j, k - 1, _dom.Gfz.s1b, _dom.Gfz.s2b)];
      s_w1[sc] = w0[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)];
      s_w2[sc] = w0[GFZ_LOC(i, j, k + 1, _dom.Gfz.s1b, _dom.Gfz.s2b)];
    }
     
    // u
    if ((j >= _dom.Gfx._jsb && j <= _dom.Gfx._jeb) &&
        (i >= _dom.Gfx._isb && i <= _dom.Gfx._ieb)) {
      s_u01[sc] = u0[GFX_LOC(i, j, k - 1, _dom.Gfx.s1b, _dom.Gfx.s2b)];
      s_u12[sc] = u0[GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b)];
    }

    // v
    if ((j >= _dom.Gfy._jsb && j <= _dom.Gfy._jeb) &&
        (i >= _dom.Gfy._isb && i <= _dom.Gfy._ieb)) {
      s_v01[sc] = v0[GFY_LOC(i, j, k - 1, _dom.Gfy.s1b, _dom.Gfy.s2b)];
      s_v12[sc] = v0[GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b)];
    }

    s_w_star[sc] = 0.0;

    // make sure all threads complete shared memory copy
    __syncthreads();

    /* compute right-hand side */
    // if off the shared memory block boundary
    if ((ti > 0 && ti < blockDim.x-1) && (tj > 0 && tj < blockDim.y-1) &&
        (j >= _dom.Gfz._js && j <= _dom.Gfz._je) &&
        (i >= _dom.Gfz._is && i <= _dom.Gfz._ie)) {

      // pressure gradient
      s_w_star[sc] = (p[GCC_LOC(i, j, k - 1, _dom.Gcc.s1b, _dom.Gcc.s2b)]
                 - p[GCC_LOC(i, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b)]) * ddz/rho_f;

      // grab the required data points for calculations
      real w110 = s_w0[sc];
      real w111 = s_w1[sc];
      real w112 = s_w2[sc];

      real w011 = s_w1[(ti - 1) + tj*blockDim.x];
      real w211 = s_w1[(ti + 1) + tj*blockDim.x];
      real u110 = s_u01[sc];
      real u111 = s_u12[sc];
      real u210 = s_u01[(ti + 1) + tj*blockDim.x];
      real u211 = s_u12[(ti + 1) + tj*blockDim.x];

      real w101 = s_w1[ti + (tj - 1)*blockDim.x];
      real w121 = s_w1[ti + (tj + 1)*blockDim.x];
      real v110 = s_v01[sc];
      real v111 = s_v12[sc];
      real v120 = s_v01[ti + (tj + 1)*blockDim.x];
      real v121 = s_v12[ti + (tj + 1)*blockDim.x];

      // compute convection term (Adams-Bashforth stepping)
      real dwudx = (w211 + w111)*(u211 + u210) - (w111 + w011)*(u111 + u110);
      dwudx *= 0.25 * ddx;

      real dwvdy = (w121 + w111)*(v121 + v120) - (w111 + w101)*(v111 + v110);
      dwvdy *= 0.25 * ddy;

      real dwwdz = (w112 + w111)*(w112 + w111) - (w111 + w110)*(w111 + w110);
      dwwdz *= 0.25 * ddz;

      s_c[sc] = dwudx + dwvdy + dwwdz;

      // convection term sums into right-hand side
#ifndef STOKESFLOW
      if (dt0 > 0) // Adams-Bashforth
        s_w_star[sc] += (-ab * s_c[sc]
          + ab0 * conv0[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)]);
      else        // forward Euler
        s_w_star[sc] += -s_c[sc];
#endif

      // compute diffusive term
      real dwd1 = (w211 - w111) * ddx;
      real dwd0 = (w111 - w011) * ddx;
      real ddwdxx = (dwd1 - dwd0) * ddx;

      dwd1 = (w121 - w111) * ddy;
      dwd0 = (w111 - w101) * ddy;
      real ddwdyy = (dwd1 - dwd0) * ddy;

      dwd1 = (w112 - w111) * ddz;
      dwd0 = (w111 - w110) * ddz;
      real ddwdzz = (dwd1 - dwd0) * ddz;

      s_d[sc] = nu * (ddwdxx + ddwdyy + ddwdzz);

      // diffusive term sums into right-hand side
      if (dt0 > 0) // Adams-Bashforth
        s_w_star[sc] += (ab * s_d[sc]
          - ab0 * diff0[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)]);
      else        // forward Euler
        s_w_star[sc] += s_d[sc];

      // add on imposed pressure gradient
      s_w_star[sc] += f[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)];

      // multiply by dt
      s_w_star[sc] *= dt;

      // velocity term sums into right-hand side
      s_w_star[sc] += w111;

      // zero contribution inside particles
      s_w_star[sc] *=
        (phase[GCC_LOC(i, j, k-1, _dom.Gcc.s1b, _dom.Gcc.s2b)] < 0
        && phase[GCC_LOC(i, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b)] < 0);
    }

    // make sure all threads complete computations
    __syncthreads();

    // copy shared memory back to global
    if ((j >= _dom.Gfz._js && j <= _dom.Gfz._je) &&
        (i >= _dom.Gfz._is && i <= _dom.Gfz._ie) &&
        (ti > 0 && ti < (blockDim.x-1)) && (tj > 0 && tj < (blockDim.y-1))) {

      w_star[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)] = s_w_star[sc];
      conv[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)] = s_c[sc];
      diff[GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b)] = s_d[sc];
    }
  }
}

__global__ void surf_int_xs(real *u_star, real *u_star_tmp)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;
  int C;
  int Cs;

  if (tj < _dom.Gfx.jn && tk < _dom.Gfx.kn) {
    Cs = GFX_LOC(_dom.Gfx._is, tj + DOM_BUF, tk + DOM_BUF, 
                  _dom.Gfx.s1b, _dom.Gfx.s2b);
    C = tj + tk*_dom.Gfx.jn;

    u_star_tmp[C] = u_star[Cs];
  }
}

__global__ void surf_int_xe(real *u_star, real *u_star_tmp)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;
  int C;
  int Ce;

  if (tj < _dom.Gfx.jn && tk < _dom.Gfx.kn) {
    Ce = GFX_LOC(_dom.Gfx._ie, tj + DOM_BUF, tk + DOM_BUF, 
                  _dom.Gfx.s1b, _dom.Gfx.s2b);
    C = tj + tk*_dom.Gfx.jn;

    u_star_tmp[C] = u_star[Ce];
  }
}

__global__ void surf_int_ys(real *v_star, real *v_star_tmp)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;
  int C;
  int Cs;

  if (tk < _dom.Gfy.kn && ti < _dom.Gfy.in) {
    Cs = GFY_LOC(ti + DOM_BUF, _dom.Gfy._js, tk + DOM_BUF, 
                  _dom.Gfy.s1b, _dom.Gfy.s2b);
    C = ti + tk*_dom.Gfy.in;

    v_star_tmp[C] = v_star[Cs];
  }
}

__global__ void surf_int_ye(real *v_star, real *v_star_tmp)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;
  int C;
  int Ce;

  if (tk < _dom.Gfy.kn && ti < _dom.Gfy.in) {
    Ce = GFY_LOC(ti + DOM_BUF, _dom.Gfy._je, tk + DOM_BUF, 
                  _dom.Gfy.s1b, _dom.Gfy.s2b);
    C = ti + tk*_dom.Gfy.in;

    v_star_tmp[C] = v_star[Ce];
  }
}

__global__ void surf_int_zs(real *w_star, real *w_star_tmp)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;
  int C;
  int Cs;

  if (ti < _dom.Gfz.in && tj < _dom.Gfz.jn) {
    Cs = GFZ_LOC(ti + DOM_BUF, tj + DOM_BUF, _dom.Gfz._ks,
                 _dom.Gfz.s1b, _dom.Gfz.s2b);
    C = ti + tj*_dom.Gfz.in;

    w_star_tmp[C] = w_star[Cs];
  }
}

__global__ void surf_int_ze(real *w_star, real *w_star_tmp)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;
  int C;
  int Ce;

  if (ti < _dom.Gfz.in && tj < _dom.Gfz.jn) {
    Ce = GFZ_LOC(ti + DOM_BUF, tj + DOM_BUF, _dom.Gfz._ke,
                 _dom.Gfz.s1b, _dom.Gfz.s2b);
    C = ti + tj*_dom.Gfz.in;

    w_star_tmp[C] = w_star[Ce];
  }
}

__global__ void plane_eps_x_W(real *u_star, real eps_x)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y; 

  if (tj < _dom.Gfx.jn && tk < _dom.Gfx.kn) {
    int C = GFX_LOC(_dom.Gfx._is, tj + DOM_BUF, tk + DOM_BUF, 
                      _dom.Gfx.s1b, _dom.Gfx.s2b);
    u_star[C] = u_star[C] + eps_x;
  }
}

__global__ void plane_eps_x_E(real *u_star, real eps_x)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y; 

  if (tj < _dom.Gfx.jn && tk < _dom.Gfx.kn) {
    int C = GFX_LOC(_dom.Gfx._ie, tj + DOM_BUF, tk + DOM_BUF, 
                      _dom.Gfx.s1b, _dom.Gfx.s2b);
    u_star[C] = u_star[C] - eps_x;
  }
}

__global__ void plane_eps_y_S(real *v_star, real eps_y)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  if (tk < _dom.Gfy.kn && ti < _dom.Gfy.in) {
    int C = GFY_LOC(ti + DOM_BUF, _dom.Gfy._js, tk + DOM_BUF,
                      _dom.Gfy.s1b, _dom.Gfy.s2b);
    v_star[C] = v_star[C] + eps_y;
  }
}

__global__ void plane_eps_y_N(real *v_star, real eps_y)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  if (tk < _dom.Gfy.kn && ti < _dom.Gfy.in) {
    int C = GFY_LOC(ti + DOM_BUF, _dom.Gfy._je, tk + DOM_BUF, 
                      _dom.Gfy.s1b, _dom.Gfy.s2b);
    v_star[C] = v_star[C] - eps_y;
  }
}

__global__ void plane_eps_z_B(real *w_star, real eps_z)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  if (ti < _dom.Gfz.in && tj < _dom.Gfz.jn) {
    int C = GFZ_LOC(ti + DOM_BUF, tj + DOM_BUF, _dom.Gfz._ks,
                      _dom.Gfz.s1b, _dom.Gfz.s2b);
    w_star[C] = w_star[C] + eps_z;
  }
}

__global__ void plane_eps_z_T(real *w_star, real eps_z)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  if (ti < _dom.Gfz.in && tj < _dom.Gfz.jn) {
    int C =  GFZ_LOC(ti + DOM_BUF, tj + DOM_BUF, _dom.Gfz._ke, 
                      _dom.Gfz.s1b, _dom.Gfz.s2b);
    w_star[C] = w_star[C] - eps_z;
  }
}

__global__ void project_u(real *u_star, real *p, real rho_f, real dt,
  real *u, real ddx, int *flag_u)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if (tj <= _dom.Gfx._je && tk <= _dom.Gfx._ke) {
    for (int i = _dom.Gfx._is; i <= _dom.Gfx._ie; i++) {
      int cfx = GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
      int cc = GCC_LOC(i, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int cw = GCC_LOC(i-1, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      real gradPhi = abs(flag_u[cfx]) * ddx * (p[cc] - p[cw]);
      u[cfx] = (u_star[cfx] - dt / rho_f * gradPhi);
    }
  }
}

__global__ void project_v(real *v_star, real *p, real rho_f, real dt,
  real *v, real ddy, int *flag_v)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int ti = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if(tk <= _dom.Gfy._ke && ti <= _dom.Gfy._ie) {
    for(int j = _dom.Gfy._js; j <= _dom.Gfy._je; j++) {
      int cfy = GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
      int cc = GCC_LOC(ti, j, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int cs = GCC_LOC(ti, j-1, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      real gradPhi = abs(flag_v[cfy]) * ddy * (p[cc] - p[cs]);
      v[cfy] = (v_star[cfy] - dt / rho_f * gradPhi);
    }
  }
}

__global__ void project_w(real *w_star, real *p, real rho_f, real dt,
  real *w, real ddz, int *flag_w)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tj = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if(ti <= _dom.Gfz._ie && tj <= _dom.Gfz._je) {
    for(int k = _dom.Gfz._ks; k <= _dom.Gfz._ke; k++) {
      int cfz = GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
      int cc = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int cb = GCC_LOC(ti, tj, k-1, _dom.Gcc.s1b, _dom.Gcc.s2b);

      real gradPhi = abs(flag_w[cfz]) * ddz * (p[cc] - p[cb]);
      w[cfz] = (w_star[cfz] - dt / rho_f * gradPhi);
    }
  }
}

__global__ void update_p_laplacian(real *Lp, real *p)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tj = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  real iddx = 1./(_dom.dx * _dom.dx);
  real iddy = 1./(_dom.dy * _dom.dy);
  real iddz = 1./(_dom.dz * _dom.dz);

  if(ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for(int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      int C = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int W = GCC_LOC(ti - 1, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int E = GCC_LOC(ti + 1, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int S = GCC_LOC(ti, tj - 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int N = GCC_LOC(ti, tj + 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int B = GCC_LOC(ti, tj, k - 1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int T = GCC_LOC(ti, tj, k + 1, _dom.Gcc.s1b, _dom.Gcc.s2b);

      real ddpdxx = (p[E] - 2.*p[C] + p[W]) * iddx;
      real ddpdyy = (p[N] - 2.*p[C] + p[S]) * iddy;
      real ddpdzz = (p[T] - 2.*p[C] + p[B]) * iddz;

      Lp[C] = ddpdxx + ddpdyy + ddpdzz;
    }
  }
}

__global__ void update_p(real *Lp, real *p0, real *p, real *phi,
  real nu, real dt, int *phase)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tj = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if(ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for(int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      int C = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // pressure correction
      p[C] = (phase[C] < 0) * (p0[C] + phi[C]);

      // see also particle_kernel:part_BC_p
      //p[C] = (phase[C] < 0) * (p0[C] + phi[C] - 0.5*nu*dt*Lp[C]);
    }
  }
}

__global__ void copy_p_p_noghost(real *p_noghost, real *p_ghost)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tj = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  int cc, CC;

  if (ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for (int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      cc = GCC_LOC(ti - DOM_BUF, tj - DOM_BUF, k - DOM_BUF, _dom.Gcc.s1, _dom.Gcc.s2);
      CC = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      p_noghost[cc] = p_ghost[CC];
    }
  }
}

__global__ void copy_p_noghost_p(real *p_noghost, real *p_ghost)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tj = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  int cc, CC;


  if (ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for (int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      cc = GCC_LOC(ti - DOM_BUF, tj - DOM_BUF, k - DOM_BUF, _dom.Gcc.s1, _dom.Gcc.s2);
      CC = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      p_ghost[CC] = p_noghost[cc];
    }
  }
}

__global__ void copy_u_square_noghost(real *u, real *utmp)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if (tj <= _dom.Gfx._je && tk <= _dom.Gfx._ke) {
    for (int ti = _dom.Gfx._is; ti <= _dom.Gfx._ie; ti++) {
      int CC = GFX_LOC(ti, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
      int cc = GFX_LOC(ti - DOM_BUF, tj - DOM_BUF, tk - DOM_BUF,
                        _dom.Gfx.s1, _dom.Gfx.s2);

      utmp[cc] = u[CC]*u[CC];
    }
  }
}

__global__ void copy_v_square_noghost(real *v, real *vtmp)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int ti = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if (tk <= _dom.Gfy._ke && ti <= _dom.Gfy._ie) {
    for (int tj = _dom.Gfy._js; tj <= _dom.Gfy._je; tj++) {
      int CC = GFY_LOC(ti, tj, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
      int cc = GFY_LOC(ti - DOM_BUF, tj - DOM_BUF, tk - DOM_BUF,
                        _dom.Gfy.s1, _dom.Gfy.s2);

      vtmp[cc] = v[CC]*v[CC];
    }
  }
}

__global__ void copy_w_square_noghost(real *w, real *wtmp)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tj = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if (ti <= _dom.Gfz._ie && tj <= _dom.Gfz._je) {
    for (int tk = _dom.Gfz._ks; tk <= _dom.Gfz._ke; tk++) {
      int CC = GFZ_LOC(ti, tj, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
      int cc = GFZ_LOC(ti - DOM_BUF, tj - DOM_BUF, tk - DOM_BUF,
                        _dom.Gfz.s1, _dom.Gfz.s2);

      wtmp[cc] = w[CC]*w[CC];
    }
  }
}

__global__ void copy_u_noghost(real *u, real *utmp)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if (tj <= _dom.Gfx._je && tk <= _dom.Gfx._ke) {
    for (int ti = _dom.Gfx._is; ti <= _dom.Gfx._ie; ti++) {
      int CC = GFX_LOC(ti, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
      int cc = GFX_LOC(ti - DOM_BUF, tj - DOM_BUF, tk - DOM_BUF,
                        _dom.Gfx.s1, _dom.Gfx.s2);

      utmp[cc] = u[CC];
    }
  }
}

__global__ void copy_v_noghost(real *v, real *vtmp)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int ti = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if (tk <= _dom.Gfy._ke && ti <= _dom.Gfy._ie) {
    for (int tj = _dom.Gfy._js; tj <= _dom.Gfy._je; tj++) {
      int CC = GFY_LOC(ti, tj, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
      int cc = GFY_LOC(ti - DOM_BUF, tj - DOM_BUF, tk - DOM_BUF,
                        _dom.Gfy.s1, _dom.Gfy.s2);

      vtmp[cc] = v[CC];
    }
  }
}

__global__ void copy_w_noghost(real *w, real *wtmp)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tj = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if (ti <= _dom.Gfz._ie && tj <= _dom.Gfz._je) {
    for (int tk = _dom.Gfz._ks; tk <= _dom.Gfz._ke; tk++) {
      int CC = GFZ_LOC(ti, tj, tk, _dom.Gfz.s1b, _dom.Gfz.s2b);
      int cc = GFZ_LOC(ti - DOM_BUF, tj - DOM_BUF, tk - DOM_BUF,
                        _dom.Gfz.s1, _dom.Gfz.s2);

      wtmp[cc] = w[CC];
    }
  }
}

__global__ void calc_dissipation(real *u, real *v, real *w, real *eps)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  real idx = 1. / _dom.dx;
  real idy = 1. / _dom.dy;
  real idz = 1. / _dom.dz;

  if (tj <= _dom.Gcc._je && tk <= _dom.Gcc._ke) {
    for (int ti = _dom.Gcc._is; ti <= _dom.Gcc._ie; ti++) {
      real dudx = idx* 
                  (u[GFX_LOC(ti + 1, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)]
                 - u[GFX_LOC(ti    , tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)]);
      real dudy = idy*0.25*
                  (u[GFX_LOC(ti + 1, tj + 1, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)]
                 + u[GFX_LOC(ti    , tj + 1, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)]
                 - u[GFX_LOC(ti + 1, tj - 1, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)]
                 - u[GFX_LOC(ti    , tj - 1, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)]);
      real dudz = idz*0.25*
                  (u[GFX_LOC(ti + 1, tj, tk + 1, _dom.Gfx.s1b, _dom.Gfx.s2b)]
                 + u[GFX_LOC(ti    , tj, tk + 1, _dom.Gfx.s1b, _dom.Gfx.s2b)]
                 - u[GFX_LOC(ti + 1, tj, tk - 1, _dom.Gfx.s1b, _dom.Gfx.s2b)]
                 - u[GFX_LOC(ti    , tj, tk - 1, _dom.Gfx.s1b, _dom.Gfx.s2b)]);

      real dvdx = idx*0.25* 
                  (v[GFY_LOC(ti + 1, tj + 1, tk, _dom.Gfy.s1b, _dom.Gfy.s2b)]
                 + v[GFY_LOC(ti + 1, tj    , tk, _dom.Gfy.s1b, _dom.Gfy.s2b)]
                 - v[GFY_LOC(ti - 1, tj + 1, tk, _dom.Gfy.s1b, _dom.Gfy.s2b)]
                 - v[GFY_LOC(ti - 1, tj    , tk, _dom.Gfy.s1b, _dom.Gfy.s2b)]);
      real dvdy = idy*
                  (v[GFY_LOC(ti, tj + 1, tk, _dom.Gfy.s1b, _dom.Gfy.s2b)]
                 - v[GFY_LOC(ti, tj    , tk, _dom.Gfy.s1b, _dom.Gfy.s2b)]);
      real dvdz = idz*0.25*
                  (v[GFY_LOC(ti, tj + 1, tk + 1, _dom.Gfy.s1b, _dom.Gfy.s2b)]
                 + v[GFY_LOC(ti, tj    , tk + 1, _dom.Gfy.s1b, _dom.Gfy.s2b)]
                 - v[GFY_LOC(ti, tj + 1, tk - 1, _dom.Gfy.s1b, _dom.Gfy.s2b)]
                 - v[GFY_LOC(ti, tj    , tk - 1, _dom.Gfy.s1b, _dom.Gfy.s2b)]);

      real dwdx = idx*0.25*
                  (w[GFZ_LOC(ti + 1, tj, tk + 1, _dom.Gfz.s1b, _dom.Gfz.s2b)]
                 + w[GFZ_LOC(ti + 1, tj, tk    , _dom.Gfz.s1b, _dom.Gfz.s2b)]
                 - w[GFZ_LOC(ti - 1, tj, tk + 1, _dom.Gfz.s1b, _dom.Gfz.s2b)]
                 - w[GFZ_LOC(ti - 1, tj, tk    , _dom.Gfz.s1b, _dom.Gfz.s2b)]);
      real dwdy = idy*0.25*
                  (w[GFZ_LOC(ti, tj + 1, tk + 1, _dom.Gfz.s1b, _dom.Gfz.s2b)]
                 + w[GFZ_LOC(ti, tj + 1, tk    , _dom.Gfz.s1b, _dom.Gfz.s2b)]
                 - w[GFZ_LOC(ti, tj - 1, tk + 1, _dom.Gfz.s1b, _dom.Gfz.s2b)]
                 - w[GFZ_LOC(ti, tj - 1, tk    , _dom.Gfz.s1b, _dom.Gfz.s2b)]);
      real dwdz = idz*
                  (w[GFZ_LOC(ti, tj, tk + 1, _dom.Gfz.s1b, _dom.Gfz.s2b)]
                 - w[GFZ_LOC(ti, tj, tk    , _dom.Gfz.s1b, _dom.Gfz.s2b)]);

      int cc = GCC_LOC(ti - DOM_BUF, tj - DOM_BUF, tk - DOM_BUF,
                        _dom.Gcc.s1, _dom.Gcc.s2);
      eps[cc] = dudx*dudx + dudy*dudy + dudz*dudz 
              + dvdx*dvdx + dvdy*dvdy + dvdz*dvdz
              + dwdx*dwdx + dwdy*dwdy + dwdz*dwdz;
    }
  }
}

__global__ void calc_dudy(real *u, real *dudy, int j)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  if (tk <= _dom.Gfx.kn && ti <= _dom.Gfx.in) {
    int k = tk + DOM_BUF;
    int i = ti + DOM_BUF;
    int CC1 = GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b);
    int CC2 = GFX_LOC(i, j + 1, k, _dom.Gfx.s1b, _dom.Gfx.s2b);
    int cc = tk + ti * _dom.Gfx.kn;

    dudy[cc] = (u[CC2] - u[CC1])/_dom.dy;
  }
}

__global__ void update_vel_BC(BC *bc, real v_bc_tdelay, real ttime)
{
  real delta = ttime - v_bc_tdelay;
  
  if (delta >= 0) {
    // uWD
    if (bc->uWDa == 0) {
      bc->uWD = bc->uWDm;
    } else if (fabs(delta*bc->uWDa) > fabs(bc->uWDm)) {
      bc->uWD = bc->uWDm;
    } else {
      bc->uWD = delta*bc->uWDa;
    }
    // uED
    if (bc->uEDa == 0) {
      bc->uED = bc->uEDm;
    } else if (fabs(delta*bc->uEDa) > fabs(bc->uEDm)) {
      bc->uED = bc->uEDm;
    } else {
      bc->uED = delta*bc->uEDa;
    }
    // uSD
    if (bc->uSDa == 0) {
      bc->uSD = bc->uSDm;
    } else if (fabs(delta*bc->uSDa) > fabs(bc->uSDm)) {
      bc->uSD = bc->uSDm;
    } else {
      bc->uSD = delta*bc->uSDa;
    }
    // uND
    if (bc->uNDa == 0) {
      bc->uND = bc->uNDm;
    } else if (fabs(delta*bc->uNDa) > fabs(bc->uNDm)) {
      bc->uND = bc->uNDm;
    } else {
      bc->uND = delta*bc->uNDa;
    }
    // uBD
    if (bc->uBDa == 0) {
      bc->uBD = bc->uBDm;
    } else if (fabs(delta*bc->uBDa) > fabs(bc->uBDm)) {
      bc->uBD = bc->uBDm;
    } else {
      bc->uBD = delta*bc->uBDa;
    }
    // uTD
    if (bc->uTDa == 0) {
      bc->uTD = bc->uTDm;
    } else if (fabs(delta*bc->uTDa) > fabs(bc->uTDm)) {
      bc->uTD = bc->uTDm;
    } else {
      bc->uTD = delta*bc->uTDa;
    }
    // vWD
    if (bc->vWDa == 0) {
      bc->vWD = bc->vWDm;
    } else if (fabs(delta*bc->vWDa) > fabs(bc->vWDm)) {
      bc->vWD = bc->vWDm;
    } else {
      bc->vWD = delta*bc->vWDa;
    }
    // vED
    if (bc->vEDa == 0) {
      bc->vED = bc->vEDm;
    } else if (fabs(delta*bc->vEDa) > fabs(bc->vEDm)) {
      bc->vED = bc->vEDm;
    } else {
      bc->vED = delta*bc->vEDa;
    }
    // vSD
    if (bc->vSDa == 0) {
      bc->vSD = bc->vSDm;
    } else if (fabs(delta*bc->vSDa) > fabs(bc->vSDm)) {
      bc->vSD = bc->vSDm;
    } else {
      bc->vSD = delta*bc->vSDa;
    }
    // vND
    if (bc->vNDa == 0) {
      bc->vND = bc->vNDm;
    } else if (fabs(delta*bc->vNDa) > fabs(bc->vNDm)) {
      bc->vND = bc->vNDm;
    } else {
      bc->vND = delta*bc->vNDa;
    }
    // vBD
    if (bc->vBDa == 0) {
      bc->vBD = bc->vBDm;
    } else if (fabs(delta*bc->vBDa) > fabs(bc->vBDm)) {
      bc->vBD = bc->vBDm;
    } else {
      bc->vBD = delta*bc->vBDa;
    }
    // vTD
    if (bc->vTDa == 0) {
      bc->vTD = bc->vTDm;
    } else if (fabs(delta*bc->vTDa) > fabs(bc->vTDm)) {
      bc->vTD = bc->vTDm;
    } else {
      bc->vTD = delta*bc->vTDa;
    }
    // wWD
    if (bc->wWDa == 0) {
      bc->wWD = bc->wWDm;
    } else if (fabs(delta*bc->wWDa) > fabs(bc->wWDm)) {
      bc->wWD = bc->wWDm;
    } else {
      bc->wWD = delta*bc->wWDa;
    }
    // wED
    if (bc->wEDa == 0) {
      bc->wED = bc->wEDm;
    } else if (fabs(delta*bc->wEDa) > fabs(bc->wEDm)) {
      bc->wED = bc->wEDm;
    } else {
      bc->wED = delta*bc->wEDa;
    }
    // wSD
    if (bc->wSDa == 0) {
      bc->wSD = bc->wSDm;
    } else if (fabs(delta*bc->wSDa) > fabs(bc->wSDm)) {
      bc->wSD = bc->wSDm;
    } else {
      bc->wSD = delta*bc->wSDa;
    }
    // wND
    if (bc->wNDa == 0) {
      bc->wND = bc->wNDm;
    } else if (fabs(delta*bc->wNDa) > fabs(bc->wNDm)) {
      bc->wND = bc->wNDm;
    } else {
      bc->wND = delta*bc->wNDa;
    }
    // wBD
    if (bc->wBDa == 0) {
      bc->wBD = bc->wBDm;
    } else if (fabs(delta*bc->wBDa) > fabs(bc->wBDm)) {
      bc->wBD = bc->wBDm;
    } else {
      bc->wBD = delta*bc->wBDa;
    }
    // wTD
    if (bc->wTDa == 0) {
      bc->wTD = bc->wTDm;
    } else if (fabs(delta*bc->wTDa) > fabs(bc->wTDm)) {
      bc->wTD = bc->wTDm;
    } else {
      bc->wTD = delta*bc->wTDa;
    }
  }
  //cuda_update_bc();
}


