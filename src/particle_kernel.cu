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

#include "cuda_particle.h"

__global__ void print_parts(part_struct *parts, int nparts) {
  for (int i = 0; i < nparts; i++) {
    printf("N%d >> parts[%d].%d (of %d) at [%.1lf, %.1lf, %.1lf]\n", _dom.rank, i,
      parts[i].N, nparts - 1, parts[i].x, parts[i].y, parts[i].z);
  }
}

__global__ void print_count(int *bin_count)
{
  for (int k = _bins.Gcc._ks; k <= _bins.Gcc._ke; k++) {
    for (int j = _bins.Gcc._js; j <= _bins.Gcc._je; j++) {
      for (int i = _bins.Gcc._is; i <= _bins.Gcc._ie; i++) {
        // Custom GFZ indices
        int s1b = _bins.Gcc.inb;
        int s2b = s1b * _bins.Gcc.jnb;
        int c = GFZ_LOC(i, j, k, s1b, s2b);

        printf("N%d >> bin_count[%d,%d,%d, (%d)] = %+d\n",
          _dom.rank, i, j, k, c, bin_count[c]);
      }
    }
  }
}

__global__ void print_bins(int *bin_start, int *bin_end) {
  for (int k = _bins.Gcc._ks; k <= _bins.Gcc._ke; k++) {
    for (int j = _bins.Gcc._js; j <= _bins.Gcc._je; j++) {
      for (int i = _bins.Gcc._is; i <= _bins.Gcc._ie; i++) {

  // Custom GFX strides
  //int s1b = _bins.Gcc.jnb;
  //int s2b = s1b * _bins.Gcc.knb;
        //int c = GFX_LOC(i, j, k, s1b, s2b);

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.knb;
        int c = GFZ_LOC(i, j, k, s1b, s2b);

        printf("N%d >> bin_start[%d,%d,%d] = %+d | bin_end[%d,%d,%d] = %+d\n",
          _dom.rank, i, j, k, bin_start[c], i, j, k, bin_end[c]);
      }
    }
  }
}

__global__ void print_int(int *a, int n, int rank, int line)
{
  for (int i = 0; i < n; i++) {
    printf("Line %d -- Rank %d: a[%d] = %d\n", line, rank, i, a[i]);
  }
}

__global__ void reset_flag_u(int *flag_u)
{
  int i;    // iterator
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  // flag everything as fluid
  if ((tj < _dom.Gfx.jnb) && (tk < _dom.Gfx.knb)) {
    for (i = _dom.Gfx._isb; i <= _dom.Gfx._ieb; i++) {
      flag_u[GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)] = 1;
    }
  }
}

__global__ void reset_flag_v(int *flag_v)
{
  int j;    // iterator
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  // flag everything as fluid
  if ((tk < _dom.Gfy.knb) && (ti < _dom.Gfy.inb)) {
    for (j = _dom.Gfy._jsb; j <= _dom.Gfy._jeb; j++) {
      flag_v[GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b)] = 1;
    }
  }
}

__global__ void reset_flag_w(int *flag_w)
{
  int k;    // iterator
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  // flag everything as fluid
  if ((ti < _dom.Gfz.inb) && (tj < _dom.Gfz.jnb)) {
    for (k = _dom.Gfz._ksb; k <= _dom.Gfz._keb; k++) {
      flag_w[GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b)] = 1;
    }
  }
}

__global__ void reset_phases(int *phase, int *phase_shell)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  if (ti < _dom.Gcc.inb && tj < _dom.Gcc.jnb) {
    for (int k = _dom.Gcc._ksb; k <= _dom.Gcc._keb; k++) {
      int c = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      phase[c] = -1;        // fluid
      phase_shell[c] = 1;   // fluid
    }
  }
}

__global__ void cage_setup(part_struct *parts, int n, int *cage_dim)
{
  // Add 4? cells to ensure cage is centered in bounding box
  cage_dim[0] = (int) (2. * ceil(parts[n].r / _dom.dx)) + 2;
  cage_dim[1] = (int) (2. * ceil(parts[n].r / _dom.dy)) + 2;
  cage_dim[2] = (int) (2. * ceil(parts[n].r / _dom.dz)) + 2;

  // Remove a cell from cage for odd number of cells in domain
  cage_dim[0] -= (_dom.xn % 2);
  cage_dim[1] -= (_dom.yn % 2);
  cage_dim[2] -= (_dom.zn % 2);
}

__global__ void build_phase(part_struct *parts, int n, int *cage_dim,
  int *phase, int *phase_shell, dom_struct *DOM, BC *bc)
{
  // Precalculate constants
  real idx = 1./_dom.dx;
  real idy = 1./_dom.dy;
  real idz = 1./_dom.dz;
  real irad = 1./parts[n].r;

  // Compute start and end cells of cage that contains particle
  int is = (int) (round((parts[n].x - _dom.xs) *idx) - 0.5*cage_dim[0]+DOM_BUF);
  int js = (int) (round((parts[n].y - _dom.ys) *idy) - 0.5*cage_dim[1]+DOM_BUF);
  int ks = (int) (round((parts[n].z - _dom.zs) *idz) - 0.5*cage_dim[2]+DOM_BUF);

  int ie = is + cage_dim[0];
  int je = js + cage_dim[1];
  int ke = ks + cage_dim[2];

  // For PERIODIC and inner subdomains, fill _isb, _ieb
  // For !PERIODIC edge subdomains, fill _is, _ie
  // This if-statement is non-branching (no thread divergence) since
  // all possible threads are coming from the same subdomain 
  int IS, IE;
  int JS, JE;
  int KS, KE;

  if (_dom.I == DOM->Is && bc->pW != PERIODIC) {  // (west edge) && !(PERIODIC)
    IS = _dom.Gcc._is;
  } else {                      // (is not global west edge) or (is periodic)
    IS = _dom.Gcc._isb;
  }
  if (_dom.I == DOM->Ie && bc->pE != PERIODIC) {  // (east edge) && !(PERIODIC)
    IE = _dom.Gcc._ie;
  } else {                      // (is not global east edge) or (is periodic)
    IE = _dom.Gcc._ieb;
  }

  if (_dom.J == DOM->Js && bc->pS != PERIODIC) {
    JS = _dom.Gcc._js;
  } else {
    JS = _dom.Gcc._jsb;
  }
  if (_dom.J == DOM->Je && bc->pN != PERIODIC) {
    JE = _dom.Gcc._je;
  } else {
    JE = _dom.Gcc._jeb;
  }

  if (_dom.K == DOM->Ks && bc->pB != PERIODIC) {
    KS = _dom.Gcc._ks;
  } else {
    KS = _dom.Gcc._ksb;
  }
  if (_dom.K == DOM->Ke && bc->pT != PERIODIC) {
    KE = _dom.Gcc._ke;
  } else {
    KE = _dom.Gcc._keb;
  }

  // Ensure that start and end of cage are inside subdomain
  //is = is * (is >= IS && is <= IE)  // <- full logic, simplified below
  //      + IS * (is < IS)          
  //      + IE * (is > IE);         
  is += (IS - is) * (is < IS)
      + (IE - is) * (is > IE);
  js += (JS - js) * (js < JS)
      + (JE - js) * (js > JE);
  ks += (KS - ks) * (ks < KS)
      + (KE - ks) * (ks > KE);

  ie += (IS - ie) * (ie < IS)
      + (IE - ie) * (ie > IE);
  je += (JS - je) * (je < JS)
      + (JE - je) * (je > JE);
  ke += (KS - ke) * (ke < KS)
      + (KE - ke) * (ke > KE);

  // Initialize thread positions
  int ti = blockDim.x * blockIdx.x + threadIdx.x + is;
  int tj = blockDim.y * blockIdx.y + threadIdx.y + js;
  int tk = blockDim.z * blockIdx.z + threadIdx.z + ks;

  if ((ti <= ie && is != ie) && 
      (tj <= je && js != je) && 
      (tk <= ke && ks != ke)) {
    // Distance from cell center to particle center

    /* Calculate x position for stencil */
    real xx = (ti - 0.5)*_dom.dx - (parts[n].x - _dom.xs);
    real yy = (tj - 0.5)*_dom.dy - (parts[n].y - _dom.ys);
    real zz = (tk - 0.5)*_dom.dz - (parts[n].z - _dom.zs);

    real d = sqrt(xx*xx + yy*yy + zz*zz);

    int C = GCC_LOC(ti, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

    /* Cage cutoff for phase */
    int cutoff = floor(d * irad) < 1;

    /* Calculate phase for current cell */
    // If cell is inside cage, set  phase equal to local part index
    // Else, leave it be (-1)

    phase[C] += cutoff * (n - phase[C]);
  }
}

__global__ void build_phase_shell(part_struct *parts, int n, int *cage_dim,
  int *phase, int *phase_shell, dom_struct *DOM, BC *bc)
{
  // Precalculate constants
  real idx = 1./_dom.dx;
  real idy = 1./_dom.dy;
  real idz = 1./_dom.dz;
  real irad = 1./parts[n].r;

  // Compute start and end cells of cage that contains particle
  int is = (int) (round((parts[n].x - _dom.xs) *idx) - 0.5*cage_dim[0]+DOM_BUF);
  int js = (int) (round((parts[n].y - _dom.ys) *idy) - 0.5*cage_dim[1]+DOM_BUF);
  int ks = (int) (round((parts[n].z - _dom.zs) *idz) - 0.5*cage_dim[2]+DOM_BUF);

  int ie = is + cage_dim[0];
  int je = js + cage_dim[1];
  int ke = ks + cage_dim[2];

  // For PERIODIC and inner subdomains, fill _isb, _ieb
  // For !PERIODIC edge subdomains, fill _is, _ie
  // This if-statement is non-branching (no thread divergence) since
  // all possible threads are coming from the same subdomain 
  int IS, IE;
  int JS, JE;
  int KS, KE;

  if (_dom.I == DOM->Is && bc->pW != PERIODIC) {  // (west edge) && !(PERIODIC)
    IS = _dom.Gcc._is;
  } else {                      // (is not global west edge) or (is periodic)
    IS = _dom.Gcc._isb;
  }
  if (_dom.I == DOM->Ie && bc->pE != PERIODIC) {  // (east edge) && !(PERIODIC)
    IE = _dom.Gcc._ie;
  } else {                      // (is not global east edge) or (is periodic)
    IE = _dom.Gcc._ieb;
  }

  if (_dom.J == DOM->Js && bc->pS != PERIODIC) {
    JS = _dom.Gcc._js;
  } else {
    JS = _dom.Gcc._jsb;
  }
  if (_dom.J == DOM->Je && bc->pN != PERIODIC) {
    JE = _dom.Gcc._je;
  } else {
    JE = _dom.Gcc._jeb;
  }

  if (_dom.K == DOM->Ks && bc->pB != PERIODIC) {
    KS = _dom.Gcc._ks;
  } else {
    KS = _dom.Gcc._ksb;
  }
  if (_dom.K == DOM->Ke && bc->pT != PERIODIC) {
    KE = _dom.Gcc._ke;
  } else {
    KE = _dom.Gcc._keb;
  }

  // Ensure that start and end of cage are inside subdomain
  //is = is * (is >= IS && is <= IE)  // <- full logic, simplified below
  //      + IS * (is < IS)          
  //      + IE * (is > IE);         
  is += (IS - is) * (is < IS)
      + (IE - is) * (is > IE);
  js += (JS - js) * (js < JS)
      + (JE - js) * (js > JE);
  ks += (KS - ks) * (ks < KS)
      + (KE - ks) * (ks > KE);

  ie += (IS - ie) * (ie < IS)
      + (IE - ie) * (ie > IE);
  je += (JS - je) * (je < JS)
      + (JE - je) * (je > JE);
  ke += (KS - ke) * (ke < KS)
      + (KE - ke) * (ke > KE);


  // Initialize thread positions
  int ti = blockDim.x * blockIdx.x + threadIdx.x + is;
  int tj = blockDim.y * blockIdx.y + threadIdx.y + js;
  int tk = blockDim.z * blockIdx.z + threadIdx.z + ks;

  if ((ti <= ie && is != ie) && 
      (tj <= je && js != je) && 
      (tk <= ke && ks != ke)) {

    int C = GCC_LOC(ti, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

    /* Calculate x position for stencil */
    real xx_w = (ti - 1 - 0.5)*_dom.dx - (parts[n].x - _dom.xs);
    real xx = (ti - 0.5)*_dom.dx - (parts[n].x - _dom.xs);
    real xx_e = (ti + 1 - 0.5)*_dom.dx - (parts[n].x - _dom.xs);

    real yy_s = (tj - 1 - 0.5)*_dom.dy - (parts[n].y - _dom.ys);
    real yy = (tj - 0.5)*_dom.dy - (parts[n].y - _dom.ys);
    real yy_n = (tj + 1 - 0.5)*_dom.dy - (parts[n].y - _dom.ys);
               
    real zz_b = (tk - 1 - 0.5)*_dom.dz - (parts[n].z - _dom.zs);
    real zz = (tk - 0.5)*_dom.dz - (parts[n].z - _dom.zs);
    real zz_t = (tk + 1 - 0.5)*_dom.dz - (parts[n].z - _dom.zs);

    /* Calculate distance */
    real d_w = sqrt(xx_w*xx_w + yy*yy + zz*zz);
    real d_e = sqrt(xx_e*xx_e + yy*yy + zz*zz);
    real d_s = sqrt(xx*xx + yy_s*yy_s + zz*zz);
    real d_n = sqrt(xx*xx + yy_n*yy_n + zz*zz);
    real d_b = sqrt(xx*xx + yy*yy + zz_b*zz_b);
    real d_t = sqrt(xx*xx + yy*yy + zz_t*zz_t);

    /* Calculate stencil phase */
    int phase_w = floor(d_w * irad) < 1;  // floor is 0 in particle
    int phase_e = floor(d_e * irad) < 1;  // floor is >=1 in fluid
    int phase_s = floor(d_s * irad) < 1;  // 0: fluid, 1: solid
    int phase_n = floor(d_n * irad) < 1;  
    int phase_b = floor(d_b * irad) < 1;
    int phase_t = floor(d_t * irad) < 1;

    /* Calculate phase_shell */
    // phase_shell = 1 (fluid), 0 (solid)
    // phase_shell[C] = 0 iff phase[C] == solid and:
    //  (phase_w == fluid) or (phase_e == fluid) or
    //  (phase_s == fluid) or (phase_n == fluid) or
    //  (phase_t == fluid) or (phase_b == fluid) 
    //phase_shell[C] *= 1 - ((phase[C] == -1) && ...
    phase_shell[C] *= 1 - ((phase[C] == n) && 
                           (phase_w == 0 || phase_e == 0 ||
                            phase_s == 0 || phase_n == 0 ||
                            phase_b == 0 || phase_t == 0));

    /* Find phase for other particles*/
    //int oob_w = ti < _dom.Gcc._isb;
    //int oob_e = ti > _dom.Gcc._ieb;
    //int oob_s = tj < _dom.Gcc._jsb;
    //int oob_n = tj > _dom.Gcc._jeb;
    //int oob_b = tk < _dom.Gcc._ksb;
    //int oob_t = tk > _dom.Gcc._keb;

    //// Correct indices for out of bound
    //int ic = ti;
    //int jc = tj;
    //int kc = tk;

    //ic += (_dom.Gcc._isb - ti) * (ti < _dom.Gcc._isb);
    //jc += (_dom.Gcc._jsb - tj) * (tj < _dom.Gcc._jsb);
    //kc += (_dom.Gcc._ksb - tk) * (tk < _dom.Gcc._ksb);
    //ic += (_dom.Gcc._ieb - ti) * (ti > _dom.Gcc._ieb);
    //jc += (_dom.Gcc._jeb - tj) * (tj > _dom.Gcc._jeb);
    //kc += (_dom.Gcc._keb - tk) * (tk > _dom.Gcc._keb);

    ///* Neighbor indices */
    //int E = GCC_LOC(ic + 1, jc, kc, _dom.Gcc.s1b, _dom.Gcc.s2b);
    //int W = GCC_LOC(ic - 1, jc, kc, _dom.Gcc.s1b, _dom.Gcc.s2b);
    //int N = GCC_LOC(ic, jc + 1, kc, _dom.Gcc.s1b, _dom.Gcc.s2b);
    //int S = GCC_LOC(ic, jc - 1, kc, _dom.Gcc.s1b, _dom.Gcc.s2b);
    //int T = GCC_LOC(ic, jc, kc + 1, _dom.Gcc.s1b, _dom.Gcc.s2b);
    //int B = GCC_LOC(ic, jc, kc - 1, _dom.Gcc.s1b, _dom.Gcc.s2b);

    ///* Get phase_shell */
    //real a = parts[n].r;

    //phase_shell[C] *= 1 - (phase[C] > -1 && (
    //  ( (oob_w + (1 - oob_w)*(phase[W] == -1)) && (1 - oob_w + oob_w*(d_w >= a)) ) ||
    //  ( (oob_e + (1 - oob_e)*(phase[E] == -1)) && (1 - oob_e + oob_e*(d_e >= a)) ) ||
    //  ( (oob_s + (1 - oob_s)*(phase[S] == -1)) && (1 - oob_s + oob_s*(d_s >= a)) ) ||
    //  ( (oob_n + (1 - oob_n)*(phase[N] == -1)) && (1 - oob_n + oob_n*(d_n >= a)) ) ||
    //  ( (oob_b + (1 - oob_b)*(phase[B] == -1)) && (1 - oob_b + oob_b*(d_b >= a)) ) ||
    //  ( (oob_t + (1 - oob_t)*(phase[T] == -1)) && (1 - oob_t + oob_t*(d_t >= a)) ) 
    //) );

  }
}

__global__ void phase_shell_x(part_struct *parts, int *phase, int *phase_shell)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tk = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;

  if (tj <= _dom.Gcc._je && tk <= _dom.Gcc._ke) {
    for (int i = _dom.Gcc._is; i <= _dom.Gcc._ieb; i++) {
      int W = GCC_LOC(i-1, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int E = GCC_LOC(i, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // fluid -> solid
      phase_shell[E] *= (1 - (phase[W] < 0 && phase[E] > -1));
      // solid -> fluid
      phase_shell[W] *= (1 - (phase[W] > -1 && phase[E] < 0));
    }
  }
}

__global__ void phase_shell_y(part_struct *parts, int *phase, int *phase_shell)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int ti = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;

  if (tk <= _dom.Gcc._ke && ti <= _dom.Gcc._ie) {
    for (int j = _dom.Gcc._js; j <= _dom.Gcc._jeb; j++) {
      int S = GCC_LOC(ti, j-1, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int N = GCC_LOC(ti, j, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // if phase changes from fluid to solid
      phase_shell[N] *= (1 - (phase[S] < 0 && phase[N] > -1));
      // if phase changes from solid to fluid
      phase_shell[S] *= (1 - (phase[S] > -1 && phase[N] < 0));
    }
  }
}

__global__ void phase_shell_z(part_struct *parts, int *phase, int *phase_shell)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tj = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;

  if (ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for (int k = _dom.Gcc._ks; k <= _dom.Gcc._keb; k++) {
      int B = GCC_LOC(ti, tj, k-1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int T = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // if phase changes from fluid to solid
      phase_shell[T] *= (1 - (phase[B] < 0 && phase[T] > -1));
      // if phase changes from solid to fluid
      phase_shell[B] *= (1 - (phase[B] > -1 && phase[T] < 0));
    }
  }
}

__global__ void cage_flag_u(int *flag_u, int *phase, int *phase_shell)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  // NOTE: This doesn't fill in flag_u[_isb, _ieb] since we don't use those 
  // values. 
  if (tj < _dom.Gfx.jnb && tk < _dom.Gfx.knb) {
    for (int i = _dom.Gfx._is; i <= _dom.Gfx._ie; i++) {
      int Cfx = GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
      int CE = GCC_LOC(i, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int CW = GCC_LOC(i-1, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      flag_u[Cfx] = 1 - 2*((phase[CW] < 0 && phase[CE] > -1) ||
                           (phase[CW] > -1 && phase[CE] < 0) ||
                           (phase_shell[CE] < 1 && phase_shell[CW] < 1));
    }
  }
}

__global__ void cage_flag_v(int *flag_v, int *phase, int *phase_shell)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  // NOTE: This doesn't fill in flag_v[_jsb, _jeb] since we don't use those 
  // values.
  if (tk < _dom.Gfy.knb && ti < _dom.Gfy.inb) {
    for (int j = _dom.Gfy._js; j <= _dom.Gfy._je; j++) {
      int Cfy = GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
      int CN = GCC_LOC(ti, j, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int CS = GCC_LOC(ti, j-1, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      flag_v[Cfy] = 1 - 2*((phase[CS] < 0 && phase[CN] > -1) ||
                           (phase[CS] > -1 && phase[CN] < 0) ||
                           (phase_shell[CN] < 1 && phase_shell[CS] < 1));
    }
  }
}

__global__ void cage_flag_w(int *flag_w, int *phase, int *phase_shell)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  // NOTE: This doesn't fill in flag_w[_isb, _ieb] since we don't use those 
  // values.
  if (ti < _dom.Gfz.inb && tj < _dom.Gfz.jnb) {
    for (int k = _dom.Gfz._ks; k <= _dom.Gfz._ke; k++) {
      int Cfz = GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
      int CT = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int CB = GCC_LOC(ti, tj, k-1, _dom.Gcc.s1b, _dom.Gcc.s2b);

      flag_w[Cfz] = 1 - 2*((phase[CB] < 0 && phase[CT] > -1) ||
                           (phase[CB] > -1 && phase[CT] < 0) ||
                           (phase_shell[CT] < 1 && phase_shell[CB] < 1));
    }
  }
}

__global__ void flag_external_u(int *flag_u, int x_loc)
{
  
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;
  
  // Flag edge of computational _domain
  if ((tj < _dom.Gfx.jnb) && (tk < _dom.Gfx.knb)) {
    flag_u[GFX_LOC(x_loc, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b)] = 0.;
  }
}

__global__ void flag_external_v(int *flag_v, int y_loc)
{
  
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;
  
  // Flag edge of computation _domain
  if ((tk < _dom.Gfy.knb) && (ti < _dom.Gfy.inb)) {
    flag_v[GFY_LOC(ti, y_loc, tk, _dom.Gfy.s1b, _dom.Gfy.s2b)] = 0;
  }
}

__global__ void flag_external_w(int *flag_w, int z_loc)
{
  
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;
  
  // Flag edge of computation _domain
  if ((ti < _dom.Gfz.inb) && (tj < _dom.Gfz.jnb)) {
    flag_w[GFZ_LOC(ti, tj, z_loc, _dom.Gfz.s1b, _dom.Gfz.s2b)] = 0;
  }
}

__global__ void bin_fill_i(int *part_ind, int *part_bin, part_struct *parts,
  int nparts, dom_struct *DOM)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index

  int cb, ib, jb, kb;  // bin indices

  real xsb = _bins.xs - _bins.dx; // to detect correct bins for particles in
  real ysb = _bins.ys - _bins.dy; //  ghost bins
  real zsb = _bins.zs - _bins.dz;

  // GFX_LOC == j + k*jnb + i*jnb*knb
  // We need to change the strides for GFX indexing
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (pp < nparts) {
    real arg_x = (parts[pp].x - xsb) / _bins.dx;
    real arg_y = (parts[pp].y - ysb) / _bins.dy;
    real arg_z = (parts[pp].z - zsb) / _bins.dz;

    /* Deal with floating point errors in position so we don't lose particles */
    // If round(arg_i) != floor(arg_i), we might have a problem
    // If the error in the round and the argument is less than machine epsilon,
    // ignore the floor and use round
    // This doesn't happen often, so ignore branching behavior for now

    ib = floor(arg_x);
    jb = floor(arg_y);
    kb = floor(arg_z);

    int round_x = lrint(arg_x); // round to nearest integer
    int round_y = lrint(arg_y);
    int round_z = lrint(arg_z);

    // Don't do this with an abs, split it into two booleans
    // Also don't do it with a square, because what happens if you square dbl_eps?
    if ((round_x != ib) && (abs(round_x - arg_x) <= 10.*DBL_EPSILON)) {
      ib = round_x;
    }
    if ((round_y != jb) && (abs(round_y - arg_y) <= 10.*DBL_EPSILON)) {
      jb = round_y;
    }
    if ((round_z != kb) && (abs(round_z - arg_z) <= 10.*DBL_EPSILON)) {
      kb = round_z;
    }

    // If particle is outside of ghost indices, it needs to be removed
    // Since all ghost particles are re-added from the adjacent subdomains 
    // anyways, we can just make the bin index to be the ghost bin index and it
    // will get removed
    ib = _bins.Gcc._isb * (ib < _bins.Gcc._isb)
       + ib * (ib >= _bins.Gcc._isb && ib <= _bins.Gcc._ieb) 
       + _bins.Gcc._ieb * (ib > _bins.Gcc._ieb);

    jb = _bins.Gcc._jsb * (jb < _bins.Gcc._jsb)
       + jb * (jb >= _bins.Gcc._jsb && jb <= _bins.Gcc._jeb) 
       + _bins.Gcc._jeb * (jb > _bins.Gcc._jeb);

    kb = _bins.Gcc._ksb * (kb < _bins.Gcc._jsb)
       + kb * (kb >= _bins.Gcc._ksb && kb <= _bins.Gcc._keb) 
       + _bins.Gcc._keb * (kb > _bins.Gcc._keb);

    cb = GFX_LOC(ib, jb, kb, s1b, s2b);

    part_ind[pp] = pp; // particle index
    part_bin[pp] = cb; // bin index
  }
}

__global__ void bin_fill_j(int *part_ind, int *part_bin, part_struct *parts,
  int nparts, dom_struct *DOM)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index

  int cb, ib, jb, kb;  // bin indices

  real xsb = _bins.xs - _bins.dx; // to detect correct bins for particles in
  real ysb = _bins.ys - _bins.dy; //  ghost bins
  real zsb = _bins.zs - _bins.dz;

  // GFY_LOC == k + i*knb + j*knb*inb
  // We need to change the strides for GFY indexing
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (pp < nparts) {
    real arg_x = (parts[pp].x - xsb) / _bins.dx;
    real arg_y = (parts[pp].y - ysb) / _bins.dy;
    real arg_z = (parts[pp].z - zsb) / _bins.dz;

    ib = floor(arg_x);
    jb = floor(arg_y);
    kb = floor(arg_z);

    int round_x = lrint(arg_x);
    int round_y = lrint(arg_y);
    int round_z = lrint(arg_z);

    /* Deal with floating point erros in position */
    if ((round_x != ib) && (abs(round_x - arg_x) <= 10.*DBL_EPSILON)) {
      ib = round_x;
    }
    if ((round_y != jb) && (abs(round_y - arg_y) <= 10.*DBL_EPSILON)) {
      jb = round_y;
    }
    if ((round_z != kb) && (abs(round_z - arg_z) <= 10.*DBL_EPSILON)) {
      kb = round_z;
    }

    ib = _bins.Gcc._isb * (ib < _bins.Gcc._isb)
       + ib * (ib >= _bins.Gcc._isb && ib <= _bins.Gcc._ieb) 
       + _bins.Gcc._ieb * (ib > _bins.Gcc._ieb);

    jb = _bins.Gcc._jsb * (jb < _bins.Gcc._jsb)
       + jb * (jb >= _bins.Gcc._jsb && jb <= _bins.Gcc._jeb) 
       + _bins.Gcc._jeb * (jb > _bins.Gcc._jeb);

    kb = _bins.Gcc._ksb * (kb < _bins.Gcc._jsb)
       + kb * (kb >= _bins.Gcc._ksb && kb <= _bins.Gcc._keb) 
       + _bins.Gcc._keb * (kb > _bins.Gcc._keb);

    // If particle is located on the domain ends (DOM->xe, DOM->ye, DOM->ze),
    // place it in the last non-ghost grid
    //ib = ib * (parts[pp].x != DOM->xe) + 
    //      _bins.Gcc._ie * (parts[pp].x == DOM->xe);
    //jb = jb * (parts[pp].y != DOM->ye) + 
    //      _bins.Gcc._je * (parts[pp].y == DOM->ye);
    //kb = kb * (parts[pp].z != DOM->ze) + 
    //      _bins.Gcc._ke * (parts[pp].z == DOM->ze);

    cb = GFY_LOC(ib, jb, kb, s1b, s2b);

    part_ind[pp] = pp;  // particle index
    part_bin[pp] = cb;  // bin index
  }
}

__global__ void bin_fill_k(int *part_ind, int *part_bin, part_struct *parts,
  int nparts, dom_struct *DOM)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index

  int cb, ib, jb, kb;  // bin indices

  real xsb = _bins.xs - _bins.dx; // to detect correct bins for particles in
  real ysb = _bins.ys - _bins.dy; //  ghost bins
  real zsb = _bins.zs - _bins.dz;

  // GFZ_LOC == i + j*inb + k*inb*jnb
  // We need to change the strides for GFZ indexing
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;


  if (pp < nparts) {
    real arg_x = (parts[pp].x - xsb) / _bins.dx;
    real arg_y = (parts[pp].y - ysb) / _bins.dy;
    real arg_z = (parts[pp].z - zsb) / _bins.dz;

    /* Deal with floating point errors in position so we don't lose particles */
    // If round(arg_z) != floor(arg_z), we might have a problem
    // If the error in the round and the argument is less than machine epsilon,
    // ignore the floor and use round
    // This doesn't happen often, so ignore branching behavior for now

    ib = floor(arg_x);
    jb = floor(arg_y);
    kb = floor(arg_z);

    int round_x = lrint(arg_x);
    int round_y = lrint(arg_y);
    int round_z = lrint(arg_z);

    // Better way? no-ifstatement or abs?
    if ((round_x != ib) && (abs(round_x - arg_x) <= 10.*DBL_EPSILON)) {
      ib = round_x;
    }
    if ((round_y != jb) && (abs(round_y - arg_y) <= 10.*DBL_EPSILON)) {
      jb = round_y;
    }
    if ((round_z != kb) && (abs(round_z - arg_z) <= 10.*DBL_EPSILON)) {
      kb = round_z;
    }

    ib = _bins.Gcc._isb * (ib < _bins.Gcc._isb)
       + ib * (ib >= _bins.Gcc._isb && ib <= _bins.Gcc._ieb) 
       + _bins.Gcc._ieb * (ib > _bins.Gcc._ieb);

    jb = _bins.Gcc._jsb * (jb < _bins.Gcc._jsb)
       + jb * (jb >= _bins.Gcc._jsb && jb <= _bins.Gcc._jeb) 
       + _bins.Gcc._jeb * (jb > _bins.Gcc._jeb);

    kb = _bins.Gcc._ksb * (kb < _bins.Gcc._jsb)
       + kb * (kb >= _bins.Gcc._ksb && kb <= _bins.Gcc._keb) 
       + _bins.Gcc._keb * (kb > _bins.Gcc._keb);

    // If particle is located on the domain ends (DOM->xe, DOM->ye, DOM->ze),
    // place it in the last non-ghost grid
    //ib = ib * (parts[pp].x != DOM->xe) + 
    //      _bins.Gcc._ie * (parts[pp].x == DOM->xe);
    //jb = jb * (parts[pp].y != DOM->ye) + 
    //      _bins.Gcc._je * (parts[pp].y == DOM->ye);
    //kb = kb * (parts[pp].z != DOM->ze) + 
    //      _bins.Gcc._ke * (parts[pp].z == DOM->ze);

    cb = GFZ_LOC(ib, jb, kb, s1b, s2b);

    part_ind[pp] = pp;  // particle index
    part_bin[pp] = cb;  // bin index
  }
}

__global__ void find_bin_start_end(int *bin_start, int *bin_end, int *part_bin,
  int nparts)
{
  // Adapted from NVIDIA CUDA sample "5_Simulations/particles" and 
  // "Particle Simulation using CUDA"
  // http://developer.download.nvidia.com/assets/cuda/files/particles.pdf

  extern __shared__ int shared_part_bin[]; // blockSize + 1
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // part index

  // For particle in part_ind, store the previous particle's bin index in
  //  shared_part_bin
  if (pp < nparts) {
    // Load bin index into shared memory. This enables looking at neighboring
    // particle's bin without loading two bin indices per thread
    shared_part_bin[threadIdx.x + 1] = part_bin[pp];

    // First thread in block must load neighboring particle bin
    if (pp > 0 && threadIdx.x == 0) {
      shared_part_bin[0] = part_bin[pp - 1];
    }
  }
  __syncthreads();

  // Find start and end
  if (pp < nparts) {
    // If current particle has a different bin than the previous particle,
    // it must be the first particle in a bin. Store this.
    // The previous particle is then the end of the previous bin.

    if (pp == 0 || part_bin[pp] != shared_part_bin[threadIdx.x]) {

      bin_start[part_bin[pp]] = pp;

      if (pp > 0) {
        bin_end[shared_part_bin[threadIdx.x]] = pp - 1;
      }
    }
    if (pp == (nparts - 1)) {
      bin_end[part_bin[pp]] = pp + 1 - 1; // -1 so can loop <= bin_end
    }
  }
}

__global__ void count_bin_parts_i(int *bin_start, int *bin_end, int *bin_count)
{
  int i;
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  // Custom GFX strides
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;
  
  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (i = 0; i < _bins.Gcc.inb; i++) {
      int c = GFX_LOC(i, tj, tk, s1b, s2b);

      // +1 since bin_end is inclusive
      bin_count[c] = bin_end[c] + 1 - bin_start[c];

      // Boolean checks to make sure that particle start/end is > -1, i.e.
      // that bin actually has a particle
      bin_count[c] *= (bin_end[c] >= 0 && bin_start[c] >= 0);
    }
  }
}

__global__ void count_bin_parts_j(int *bin_start, int *bin_end, int *bin_count)
{
  int j;
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  // Custom GFY strides
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;
  
  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (j = 0; j < _bins.Gcc.jnb; j++) {
      int c = GFY_LOC(ti, j, tk, s1b, s2b);

      // +1 since bin_end is inclusive
      bin_count[c] = bin_end[c] + 1 - bin_start[c];

      // Boolean checks to make sure that particle start/end is > -1, i.e.
      // that bin actually has a particle
      bin_count[c] *= (bin_end[c] >= 0 && bin_start[c] >= 0);
    }
  }
}

__global__ void count_bin_parts_k(int *bin_start, int *bin_end, int *bin_count)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;
  int k;

  // Custom GFZ strides
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;
  
  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (k = 0; k < _bins.Gcc.knb; k++) {
      int c = GFZ_LOC(ti, tj, k, s1b, s2b);

      // +1 since bin_end is inclusive
      bin_count[c] = bin_end[c] + 1 - bin_start[c];

      // Boolean checks to make sure that particle start/end is > -1, i.e.
      // that bin actually has a particle
      bin_count[c] *= (bin_end[c] >= 0 && bin_start[c] >= 0);
    }
  }
}

__global__ void pack_parts_e(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    cbin = GFX_LOC(_bins.Gcc._ie, tj, tk, s1b, s2b);
    c2b = tj + tk * _bins.Gcc.jnb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_parts_w(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    cbin = GFX_LOC(_bins.Gcc._is, tj, tk, s1b, s2b);
    c2b = tj + tk * _bins.Gcc.jnb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_parts_n(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    cbin = GFY_LOC(ti, _bins.Gcc._je, tk, s1b, s2b);
    c2b = tk + ti * _bins.Gcc.knb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_parts_s(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    cbin = GFY_LOC(ti, _bins.Gcc._js, tk, s1b, s2b);
    c2b = tk + ti * _bins.Gcc.knb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_parts_t(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    cbin = GFZ_LOC(ti, tj, _bins.Gcc._ke, s1b, s2b);
    c2b = ti + tj * _bins.Gcc.inb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void pack_parts_b(part_struct *send_parts, part_struct *parts,
  int *offset, int *bin_start, int *bin_count, int *part_ind)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    cbin = GFZ_LOC(ti, tj, _bins.Gcc._ks, s1b, s2b);
    c2b = ti + tj * _bins.Gcc.inb;

    // Loop through each bin's particles and add to send_parts
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      send_parts[offset[c2b] + i] = parts[pp];
    }
  }
}

__global__ void copy_central_bin_parts_i(part_struct *tmp_parts,
  part_struct *parts, int *bin_start, int *bin_count, int *part_ind,
  int *offset)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;             // bin index
  int pp;               // particle index
  int dest;             // destination in tmp_parts

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    // Loop over i-planes
    for (int i = _bins.Gcc._is; i <= _bins.Gcc._ie; i++) {
      cbin = GFX_LOC(i, tj, tk, s1b, s2b);


      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];
        dest = offset[cbin] + n;

        tmp_parts[dest] = parts[pp];
      }
    }
  }
}

__global__ void copy_central_bin_parts_j(part_struct *tmp_parts,
  part_struct *parts, int *bin_start, int *bin_count, int *part_ind,
  int *offset)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;             // bin index
  int pp;               // particle index
  int dest;             // destination in tmp_parts

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    // Loop over j-planes
    for (int j = _bins.Gcc._js; j <= _bins.Gcc._je; j++) {
      cbin = GFY_LOC(ti, j, tk, s1b, s2b);

      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];
        dest = offset[cbin] + n;

        tmp_parts[dest] = parts[pp];
      }
    }
  }
}

__global__ void copy_central_bin_parts_k(part_struct *tmp_parts,
  part_struct *parts, int *bin_start, int *bin_count, int *part_ind,
  int *offset)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;             // bin index
  int pp;               // particle index
  int dest;             // destination in tmp_parts

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    // Loop over j-planes
    for (int k = _bins.Gcc._ks; k <= _bins.Gcc._ke; k++) {
      cbin = GFZ_LOC(ti, tj, k, s1b, s2b);

      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];
        dest = offset[cbin] + n;

        tmp_parts[dest] = parts[pp];
      }
    }
  }
}

__global__ void copy_ghost_bin_parts(part_struct *tmp_parts,
  part_struct *recv_parts, int nparts_recv, int offset, int plane, dom_struct *DOM)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index
  int dest;

  if (pp < nparts_recv) {
    dest = offset + pp;
    tmp_parts[dest] = recv_parts[pp];

    // Correct particle positions for periodicity
    int e_chk = (plane == EAST && _dom.I == DOM->Ie && _dom.e != MPI_PROC_NULL);
    int w_chk = (plane == WEST && _dom.I == DOM->Is && _dom.w != MPI_PROC_NULL);
    int n_chk = (plane == NORTH && _dom.J == DOM->Je && _dom.n != MPI_PROC_NULL);
    int s_chk = (plane == SOUTH && _dom.J == DOM->Js && _dom.s != MPI_PROC_NULL);
    int t_chk = (plane == TOP && _dom.K == DOM->Ke && _dom.t != MPI_PROC_NULL); 
    int b_chk = (plane == BOTTOM && _dom.K == DOM->Ks && _dom.b != MPI_PROC_NULL);

    real x = tmp_parts[dest].x;
    real y = tmp_parts[dest].y;
    real z = tmp_parts[dest].z;

    x = x * (!e_chk && !w_chk)
      + (x - DOM->xl) * w_chk
      + (x + DOM->xl) * e_chk;

    y = y * (!n_chk && !s_chk)
      + (y - DOM->yl) * s_chk
      + (y + DOM->yl) * n_chk;

    z = z * (!t_chk && !b_chk)
      + (z - DOM->zl) * b_chk
      + (z + DOM->zl) * t_chk;

    tmp_parts[dest].x = x;
    tmp_parts[dest].y = y;
    tmp_parts[dest].z = z;
    
    // use PLANE as input
    // if (PLANE == EAST && _dom.I == DOM->Ie && _dom.e != MPI_PROC_NULL) {
    //   tmp_parts[offset + pp] += DOM->xl  
    // if (PLANE == WEST && ...
    //   tmp_parts[offset + pp] -= DOM->xl
  }
}

__global__ void correct_periodic_boundaries_i(part_struct *parts,
  int offset, int nparts_added, BC *bc, dom_struct *DOM)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index

  // if (parts[pp].x > _bins._ieb) && (bc->pW == PERIODIC)
  //    parts[pp].x -= DOM->xl
  // if (parts[pp].x < _bins._isb) && (bc->pE == PERIODIC)
  //    parts[pp].x += DOM->xl
  // else, nothing

  real xsb = _bins.xs - _bins.dx;
  real xeb = _bins.xe + _bins.dx;

  if (pp < nparts_added) {
    real x = parts[pp + offset].x;
    
    x = x * (x >= xsb && x <= xeb)
      + (x - DOM->xl) * (x > xeb)
      + (x + DOM->xl) * (x < xsb);

    parts[pp + offset].x = x;
  }
}

__global__ void correct_periodic_boundaries_j(part_struct *parts,
  int offset, int nparts_added, BC *bc, dom_struct *DOM)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index

  // if (parts[pp].y > _bins._jeb) && (bc->pS == PERIODIC)
  //    parts[pp].y -= DOM->yl
  // if (parts[pp].y < _bins._jsb) && (bc->pN == PERIODIC)
  //    parts[pp].y += DOM->yl
  // else, nothing

  real ysb = _bins.ys - _bins.dy;
  real yeb = _bins.ye + _bins.dy;

  if (pp < nparts_added) {
    real y = parts[pp + offset].y;
    
    y = y * (y >= ysb && y <= yeb)
      + (y - DOM->yl) * (y > yeb)
      + (y + DOM->yl) * (y < ysb);

    parts[pp + offset].y = y;
  }
}

__global__ void correct_periodic_boundaries_k(part_struct *parts,
  int offset, int nparts_added, BC *bc, dom_struct *DOM)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index

  // if (parts[pp].y > _bins._jeb) && (bc->pS == PERIODIC)
  //    parts[pp].y -= DOM->yl
  // if (parts[pp].y < _bins._jsb) && (bc->pN == PERIODIC)
  //    parts[pp].y += DOM->yl
  // else, nothing

  real zsb = _bins.zs - _bins.dz;
  real zeb = _bins.ze + _bins.dz;

  if (pp < nparts_added) {
    real z = parts[pp + offset].z;
    
    z = z * (z >= zsb && z <= zeb)
      + (z - DOM->zl) * (z > zeb)
      + (z + DOM->zl) * (z < zsb);
  }
}

__global__ void zero_ghost_bins_i(int *bin_count)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  // GFZ_LOC == i + j*inb + k*inb*jnb
  // We need to change the strides for GFZ indexing
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    int ce = GFZ_LOC(_bins.Gcc._ieb, tj, tk, s1b, s2b);
    int cw = GFZ_LOC(_bins.Gcc._isb, tj, tk, s1b, s2b);

    bin_count[ce] = 0;
    bin_count[cw] = 0;
  }
}

__global__ void zero_ghost_bins_j(int *bin_count)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  // GFZ_LOC == i + j*inb + k*inb*jnb
  // We need to change the strides for GFZ indexing
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    int cn = GFZ_LOC(ti, _bins.Gcc._jeb, tk, s1b, s2b);
    int cs = GFZ_LOC(ti, _bins.Gcc._jsb, tk, s1b, s2b);

    bin_count[cn] = 0;
    bin_count[cs] = 0;
  }
}

__global__ void zero_ghost_bins_k(int *bin_count)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  // GFZ_LOC == i + j*inb + k*inb*jnb
  // We need to change the strides for GFZ indexing
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    int cb = GFZ_LOC(ti, tj, _bins.Gcc._ksb, s1b, s2b);
    int ct = GFZ_LOC(ti, tj, _bins.Gcc._keb, s1b, s2b);

    bin_count[cb] = 0;
    bin_count[ct] = 0;
  }
}

__global__ void copy_subdom_parts(part_struct *tmp_parts, part_struct *parts,
  int *bin_start, int *bin_count, int *part_ind, int *bin_offset)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;             // bin index
  int pp;               // particle index
  int dest;             // destination in tmp_parts

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.in && tj < _bins.Gcc.jn) {
    for (int k = _bins.Gcc._ks; k <= _bins.Gcc._ke; k++) {
      cbin = GFZ_LOC(ti + 1, tj + 1, k, s1b, s2b);

      for (int n = 0; n < bin_count[cbin]; n++) {
        pp = part_ind[bin_start[cbin] + n];
        dest = bin_offset[cbin] + n;

        tmp_parts[dest] = parts[pp];
      }
    }
  }
}

__global__ void part_BC_u(real *u, int *phase, int *flag_u, part_struct *parts,
  real nu, int nparts)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int C, CE, CW;
  real x, y, z;         // velocity node location (Cartesian)
  real Xp, Yp, Zp;      // particle position
  real r, theta, phi;   // velocity node location (spherical)
  real Ux, Uy, Uz;      // Cartesian pressure gradients
  int P, PP, PW, PE;    // particle number
  real a;               // part radius
  int order;            // particle order
  real oy, oz;          // particle angular velocity
  real oydot, ozdot;    // particle angular acceleration
  real uu;              // particle velocity

  if (tj < _dom.Gfx.jnb && tk < _dom.Gfx.knb) {
    for (int i = _dom.Gfx._is; i <= _dom.Gfx._ie; i++) {
      // Location of current thread
      C = GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
      CW = GCC_LOC(i-1, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CE = GCC_LOC(i, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Position of current thread
      x = (i-DOM_BUF) * _dom.dx + _dom.xs;
      y = (tj-0.5) * _dom.dy + _dom.ys;
      z = (tk-0.5) * _dom.dz + _dom.zs;

      // Particle number
      PW = phase[CW];
      PE = phase[CE];

      if (PW > -1) {        // particle to the west
        P = PW;
        PP = PW;
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        order = parts[P].order;
        oy = parts[P].oy;
        oz = parts[P].oz;
        oydot = parts[P].oydot;
        ozdot = parts[P].ozdot;
        uu = parts[P].u;
      } else if (PE > -1) { // particle to the east
        P = PE;
        PP = PE;
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        order = parts[P].order;
        oy = parts[P].oy;
        oz = parts[P].oz;
        oydot = parts[P].oydot;
        ozdot = parts[P].ozdot;
        uu = parts[P].u;
      } else {              // no particle
        P = 0;
        PP = -1;
        a = (_dom.dx + _dom.dy + _dom.dz) / 3.;
        Xp = (i - DOM_BUF) * _dom.dx + _dom.xs + a;
        Yp = (tj - 0.5) * _dom.dy + _dom.ys + a;
        Zp = (tk - 0.5) * _dom.dz + _dom.zs + a;
        order = 0;
        oy = 0;
        oz = 0;
        oydot = 0;
        ozdot = 0;
        uu = 0;
      }

      // Calculate position of velocity node in particle frame
      x -= Xp;
      y -= Yp;
      z -= Zp;
      xyz2rtp(x, y, z, &r, &theta, &phi);

      lamb_vel(order, a, r, theta, phi, parts, nu, P, &Ux, &Uy, &Uz);

      real ocrossr_x = oy*z - oz*y;
      real odotcrossr_x = oydot*z - ozdot*y;
      Ux += uu + ocrossr_x;
      Ux += 0.1/nu *(r*r*r*r*r-a*a*a*a*a)/(r*r*r) * odotcrossr_x;

      // boolean check if this is an analytically-posed node
      int check = (flag_u[C] < 1) && (PP > -1);
      u[C] = check * Ux + (1 - check) * u[C];
    }
  }
}

__global__ void part_BC_v(real *v, int *phase, int *flag_v, part_struct *parts,
  real nu, int nparts)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int C, CN, CS;
  real x, y, z;         // velocity node location (Cartesian)
  real Xp, Yp, Zp;      // particle position
  real r, theta, phi;   // velocity node location (spherical)
  real Ux, Uy, Uz;      // Cartesian pressure gradients
  int P, PP, PS, PN;    // particle number
  real a;               // part radius
  int order;            // particle order
  real oz, ox;          // particle angular velocity
  real ozdot, oxdot;    // particle angular acceleration
  real vv;              // particle velocity

  if (tk < _dom.Gfy.knb && ti < _dom.Gfy.inb) {
    for (int j = _dom.Gfy._js; j <= _dom.Gfy._je; j++) {
      // Location of current thread
      C = GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
      CS = GCC_LOC(ti, j-1, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CN = GCC_LOC(ti, j, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Position of current thread
      x = (ti-0.5) * _dom.dx + _dom.xs;
      y = (j-DOM_BUF) * _dom.dy + _dom.ys;
      z = (tk-0.5) * _dom.dz + _dom.zs;

      // Particle number
      PS = phase[CS];
      PN = phase[CN];

      if (PS > -1) {        // particle to the west
        P = PS;
        PP = PS;
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        order = parts[P].order;
        oz = parts[P].oz;
        ox = parts[P].ox;
        ozdot = parts[P].ozdot;
        oxdot = parts[P].oxdot;
        vv = parts[P].v;
      } else if (PN > -1) { // particle to the east
        P = PN;
        PP = PN;
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        order = parts[P].order;
        oz = parts[P].oz;
        ox = parts[P].ox;
        ozdot = parts[P].ozdot;
        oxdot = parts[P].oxdot;
        vv = parts[P].v;
      } else {              // no particle
        P = 0;
        PP = -1;
        a = (_dom.dx + _dom.dy + _dom.dz) / 3.;
        Xp = (ti-0.5) * _dom.dx + _dom.xs + a;
        Yp = (j-DOM_BUF) * _dom.dy + _dom.ys + a;
        Zp = (tk-0.5) * _dom.dz + _dom.zs + a;
        order = 0;
        oz = 0;
        ox = 0;
        ozdot = 0;
        oxdot = 0;
        vv = 0;
      }

      // Calculate position of velocity node
      x -= Xp;
      y -= Yp;
      z -= Zp;
      xyz2rtp(x, y, z, &r, &theta, &phi);

      lamb_vel(order, a, r, theta, phi, parts, nu, P, &Ux, &Uy, &Uz);
  
      // switch reference frame and set boundary condition
      real ocrossr_y = -(ox*z - oz*x);
      real odotcrossr_y = -(oxdot*z - ozdot*x);
      Uy += vv + ocrossr_y;
      Uy += 0.1/nu *(r*r*r*r*r-a*a*a*a*a)/(r*r*r) * odotcrossr_y;

      // boolean check if this is an analytically-posed node
      int check = (flag_v[C] < 1) && (PP > -1);
      v[C] = check * Uy + (1 - check) * v[C];
    }
  }
}
__global__ void part_BC_w(real *w, int *phase, int *flag_w, part_struct *parts,
  real nu, int nparts)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int C, CT, CB;
  real x, y, z;         // velocity node location (Cartesian)
  real Xp, Yp, Zp;      // particle position
  real r, theta, phi;   // velocity node location (spherical)
  real Ux, Uy, Uz;      // Cartesian pressure gradients
  int P, PP, PB, PT;    // particle number
  real a;               // part radius
  int order;            // particle order
  real ox, oy;          // particle angular velocity
  real oxdot, oydot;    // particle angular acceleration
  real ww;              // particle velocity

  if (ti < _dom.Gfz.inb && tj < _dom.Gfz.jnb) {
    for (int k = _dom.Gfz._ks; k <= _dom.Gfz._ke; k++) {
      // Location of current thread
      C = GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
      CB = GCC_LOC(ti, tj, k-1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      CT = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Position of current thread
      x = (ti-0.5) * _dom.dx + _dom.xs;
      y = (tj-0.5) * _dom.dy + _dom.ys;
      z = (k-DOM_BUF) * _dom.dz + _dom.zs;

      // Particle number
      PB = phase[CB];
      PT = phase[CT];

      if (PB > -1) {        // particle to the west
        P = PB;
        PP = PB;
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        order = parts[P].order;
        ox = parts[P].ox;
        oy = parts[P].oy;
        oxdot = parts[P].oxdot;
        oydot = parts[P].oydot;
        ww = parts[P].w;
      } else if (PT > -1) { // particle to the east
        P = PT;
        PP = PT;
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        order = parts[P].order;
        ox = parts[P].ox;
        oy = parts[P].oy;
        oxdot = parts[P].oxdot;
        oydot = parts[P].oydot;
        ww = parts[P].w;
      } else {              // no particle
        P = 0;
        PP = -1;
        a = (_dom.dx + _dom.dy + _dom.dz) / 3.;
        Xp = (ti-0.5) * _dom.dx + _dom.xs + a;
        Yp = (tj-0.5) * _dom.dy + _dom.ys + a;
        Zp = (k-DOM_BUF) * _dom.dz + _dom.zs + a;
        order = 0;
        ox = 0;
        oy = 0;
        oxdot = 0;
        oydot = 0;
        ww = 0;
      }

      // Calculate position of velocity node
      x -= Xp;
      y -= Yp;
      z -= Zp;
      xyz2rtp(x, y, z, &r, &theta, &phi);

      lamb_vel(order, a, r, theta, phi, parts, nu, P, &Ux, &Uy, &Uz);
  
      // switch reference frame and set boundary condition
      real ocrossr_z = ox*y - oy*x;
      real odotcrossr_z = oxdot*y - oydot*x;
      Uz += ww + ocrossr_z;
      Uz += 0.1/nu *(r*r*r*r*r-a*a*a*a*a)/(r*r*r) * odotcrossr_z;
      // boolean check if this is an analytically-posed node
      int check = (flag_w[C] < 1) && (PP > -1);
      w[C] = check * Uz + (1 - check) * w[C];
    }
  }
}

__global__ void part_BC_p(real *p, real *p_rhs, int *phase, int *phase_shell,
  part_struct *parts, real mu, real nu, real dt, real dt0, gradP_struct gradP,
  real rho_f, int nparts, real s_beta, real s_ref, g_struct g)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tj = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;
  int CC;
  real x, y, z;         // pressure node location Cartesian
  real Xp, Yp, Zp;      // particle position
  real r, theta, phi;   // velocity node location spherical
  real pp_tmp;//, pp_tmp00;// temporary pressure
  int P;                // particle number
  real a;               // particle radius
  int order;            // particle order
  real ox, oy, oz;      // particle angular velocity
  real udot, vdot, wdot;// particle acceleration

  if (ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for (int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      CC = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Position of current thread
      x = (ti-0.5) * _dom.dx + _dom.xs;
      y = (tj-0.5) * _dom.dy + _dom.ys;
      z = (k-0.5) * _dom.dz + _dom.zs;

      // get particle number
      P = phase[CC];

      if(P > -1) {
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        order = parts[P].order;
        ox = parts[P].ox;
        oy = parts[P].oy;
        oz = parts[P].oz;
        udot = parts[P].udot;
        vdot = parts[P].vdot;
        wdot = parts[P].wdot;
      } else {
        P = 0;
        a = (_dom.dx + _dom.dy + _dom.dz) / 3.;
        Xp = (ti-0.5) * _dom.dx + _dom.xs + a;
        Yp = (tj-0.5) * _dom.dy + _dom.ys + a;
        Zp = (k-0.5) * _dom.dz + _dom.zs + a;
        order = 0;
        ox = 0;
        oy = 0;
        oz = 0;
        udot = 0;
        vdot = 0;
        wdot = 0;
      }

      // Position in particle frame
      x -= Xp;
      y -= Yp;
      z -= Zp;
      xyz2rtp(x, y, z, &r, &theta, &phi);

      // calculate analytic solution
      real ar = a / r;
      real ra = r / a;
      pp_tmp = X_pn(0, theta, phi, P, parts);
      for(int n = 1; n <= order; n++) {
        pp_tmp += (1.-0.5*n*(2.*n-1.)/(n+1.)*pow(ar,2.*n+1.))*pow(ra,n)
          * X_pn(n, theta, phi, P, parts);
        pp_tmp -= n*(2.*n-1.)*(2.*n+1.)/(n+1.)*pow(ar,n+1.)
          * X_phin(n, theta, phi, P, parts);

        //pp_tmp00 += (1.-0.5*n*(2.*n-1.)/(n+1.)*pow(ar,2.*n+1.))*pow(ra,n)
          //* X_pn(n, theta, phi, pnm_re00, pnm_im00, P, stride);
        //pp_tmp00 -= n*(2.*n-1.)*(2.*n+1.)/(n+1.)*pow(ar,n+1.)
          //* X_phin(n, theta, phi, phinm_re00, phinm_im00, P, stride);
      }
      pp_tmp *= mu*nu/(a*a);
      //pp_tmp00 *= mu*nu/(a*a);
      real ocrossr2 = (oy*z - oz*y) * (oy*z - oz*y);
      ocrossr2 += (ox*z - oz*x) * (ox*z - oz*x);
      ocrossr2 += (ox*y - oy*x) * (ox*y - oy*x);
      real bousiq_x = -s_beta*(parts[P].s - s_ref)*g.x;
      real bousiq_y = -s_beta*(parts[P].s - s_ref)*g.y;
      real bousiq_z = -s_beta*(parts[P].s - s_ref)*g.z;
      real rhoV = rho_f;
      real accdotr = (-gradP.x/rhoV - udot + bousiq_x)*x +
                     (-gradP.y/rhoV - vdot + bousiq_y)*y +
                     (-gradP.z/rhoV - wdot + bousiq_z)*z;
      pp_tmp += 0.5 * rho_f * ocrossr2 + rho_f * accdotr;
      //pp_tmp00 += 0.5 * rho_f * ocrossr2 + rho_f * accdotr;

      // write BC if flagged, otherwise leave alone
      p_rhs[CC] = (real) phase_shell[CC] * p_rhs[CC]
        - (real) (1 - phase_shell[CC]) * (pp_tmp - p[CC]);
      // subtract second half because solving -Ax = -b
      // + 0.5*mu*p_rhs[CC]); // see also cuda_bluebottle:update_p

      p_rhs[CC] = (real) (phase[CC] < 0 && phase_shell[CC]) * p_rhs[CC];
    }
  }
}

__global__ void part_BC_p_fill(real *p, int *phase, part_struct *parts,
  real mu, real nu, real rho_f, gradP_struct gradP, int nparts,
  real s_beta, real s_ref, g_struct g)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tj = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;
  int CC;
  real x, y, z;         // pressure node location Cartesian
  real Xp, Yp, Zp;         // particle position
  real r, theta, phi;   // velocity node location spherical
  real pp_tmp;//, pp_tmp00;// temporary pressure
  int P;                // particle number
  real a;               // particle radius
  real ox, oy, oz;      // particle angular velocity
  real udot, vdot, wdot;// particle acceleration

  if(ti <= _dom.Gcc._ie && tj <= _dom.Gcc._je) {
    for(int k = _dom.Gcc._ks; k <= _dom.Gcc._ke; k++) {
      CC = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      // Position of current thread
      x = (ti-0.5) * _dom.dx + _dom.xs;
      y = (tj-0.5) * _dom.dy + _dom.ys;
      z = (k-0.5) * _dom.dz + _dom.zs;

      // get particle number
      P = phase[CC];
      if(P > -1) {
        a = parts[P].r;
        Xp = parts[P].x;
        Yp = parts[P].y;
        Zp = parts[P].z;
        ox = parts[P].ox;
        oy = parts[P].oy;
        oz = parts[P].oz;
        udot = parts[P].udot;
        vdot = parts[P].vdot;
        wdot = parts[P].wdot;
      } else {
        P = 0;
        a = (_dom.dx + _dom.dy + _dom.dz) / 3.;
        Xp = (ti-0.5) * _dom.dx + _dom.xs + a;
        Yp = (tj-0.5) * _dom.dy + _dom.ys + a;
        Zp = (k-0.5) * _dom.dz + _dom.zs + a;
        ox = 0;
        oy = 0;
        oz = 0;
        udot = 0;
        vdot = 0;
        wdot = 0;
      }

      // Position in particle frame
      x -= Xp;
      y -= Yp;
      z -= Zp;
      xyz2rtp(x, y, z, &r, &theta, &phi);

      // calculate analytic solution
      pp_tmp = X_pn(0, theta, phi, P, parts);
      pp_tmp *= mu*nu/(a*a);
      real ocrossr2 = (oy*z - oz*y) * (oy*z - oz*y);
      ocrossr2 += (ox*z - oz*x) * (ox*z - oz*x);
      ocrossr2 += (ox*y - oy*x) * (ox*y - oy*x);
      real rhoV = rho_f;
      real bousiq_x = -s_beta*(parts[P].s - s_ref)*g.x;
      real bousiq_y = -s_beta*(parts[P].s - s_ref)*g.y;
      real bousiq_z = -s_beta*(parts[P].s - s_ref)*g.z;
      real accdotr = (-gradP.x/rhoV - udot + bousiq_x)*x +
                     (-gradP.y/rhoV - vdot + bousiq_y)*y +
                     (-gradP.z/rhoV - wdot + bousiq_z)*z;
      pp_tmp += 0.5 * rho_f * ocrossr2 + rho_f * accdotr;

      // write BC if inside particle, otherwise leave alone
      p[CC] = (real) (phase[CC] > -1) * pp_tmp
        + (1 - (phase[CC] > -1)) * p[CC];

    }
  }
}

__device__ void lamb_vel(int order, real a, real r, real theta, real phi,
  part_struct *parts, real nu, int p_ind, real *Ux, real *Uy, real *Uz)
{
  real ia = 1. / a;
  real ar = a / r;
  real ra = r * ia;

  real ur = 0.;
  real ut = 0.5*ra*Y_pn(0, theta, phi, p_ind, parts);
  real up = 0.5*ra*Z_pn(0, theta, phi, p_ind, parts);

  for(int n = 1; n <= order; n++) {
    real powranm1 = pow(ra,n-1.);
    real powranp1 = powranm1*ra*ra;

    real powarn = pow(ar,n);
    real powarnp1 = powarn*ar;
    real powarnp2 = powarnp1*ar;

    real od2np3 = 1./(2.*n+3.);
    real odnp1 = 1./(n+1.);

    ur += (0.5*n*od2np3*powranp1
      + 0.25*n*((2.*n+1.)*od2np3*ar*ar-1.)*powarn)
      * X_pn(n, theta, phi, p_ind, parts);
    ur += (n*powranm1
      + 0.5*n*(2.*n-1.-(2.*n+1.)*ra*ra)*powarnp2)
      * X_phin(n, theta, phi, p_ind, parts);

    ut += (0.5*(n+3.)*odnp1*od2np3*powranp1
      + 0.25*odnp1*(n-2.-n*(2.*n+1.)*od2np3*ar*ar)*powarn)
      * Y_pn(n, theta, phi, p_ind, parts);
    ut += (powranm1
      + 0.5*odnp1*((n-2.)*(2.*n+1.)*ra*ra-n*(2.*n-1.))*powarnp2)
      * Y_phin(n, theta, phi, p_ind, parts);
    ut += (powranm1
      - powarnp1)
      * Z_chin(n, theta, phi, p_ind, parts);
      
    up += (0.5*(n+3.)*odnp1*od2np3*powranp1
      + 0.25*odnp1*(n-2.-n*(2.*n+1.)*od2np3*ar*ar)*powarn)
      * Z_pn(n, theta, phi, p_ind, parts);
    up += (powranm1
      + 0.5*odnp1*((n-2.)*(2.*n+1.)*ra*ra-n*(2.*n-1.))*powarnp2)
      * Z_phin(n, theta, phi, p_ind, parts);
    up += (-powranm1
      + powarnp1)
      * Y_chin(n, theta, phi, p_ind, parts);
  }
  ur *= nu * ia;
  ut *= nu * ia;
  up *= nu * ia;

  real st = sin(theta);
  real ct = cos(theta);
  real sp = sin(phi);
  real cp = cos(phi);

  *Ux = ur*st*cp + ut*ct*cp - up*sp;
  *Uy = ur*st*sp + ut*ct*sp + up*cp;
  *Uz = ur*ct - ut*st;
}

__device__ void xyz2rtp(real x, real y, real z, real *r, real *theta, real *phi)
{
  real XY = x*x + y*y;
  real XYZ = XY + z*z;
  // We calculate the coefficients everywhere in space. If a particle is
  // centered at the center of a cell, XYZ will be zero.
  if (XYZ >= 0 && XYZ < DIV_ST) XYZ = DIV_ST;
  *r = sqrt(XYZ);
  *theta = acos(z / *r);
  // Note that XY cannot be set equal to one, because the values are used.
  if (XY >= 0 && XY < DIV_ST) XY = DIV_ST;
  *phi = acos(x / sqrt(XY));
  if (y < 0.) *phi = 2.*PI - *phi;
}

__device__ real Nnm(int n, int m)
{
  real fact_top = 1;
  real fact_bot = 1;

  for(int i = 1; i <= (n-m); i++) fact_top *= (real)i;
  for(int i = 1; i <= (n+m); i++) fact_bot *= (real)i;

  return sqrt((2.*n+1.) / 4. / PI * fact_top / fact_bot);
}

__device__ real Pnm(int n, int m, real theta)
{
  real x = cos(theta);
  real y = sin(theta);

  switch(n) {
    case 0: return 1;
    case 1:
      switch(m) {
        //case -1: return 0.5*y;
        case 0: return x;
        case 1: return -y;
      }
    case 2:
      switch(m) {
        //case -2: return 0.125*y*y;
        //case -1: return 0.5*x*y;
        case 0: return 0.5*(3.*x*x - 1.);
        case 1: return -3.*x*y;
        case 2: return 3.*y*y;
      }
    case 3:
      switch(m) {
        //case -3: return 0.02083333333333*y*y*y;
        //case -2: return 0.125*x*y*y;
        //case -1: return -0.125*(1. - 5.*x*x)*y;
        case 0: return 0.5*x*(5.*x*x - 3.);
        case 1: return -1.5*(5.*x*x - 1.)*y;
        case 2: return 15.*x*y*y;
        case 3: return -15.*y*y*y;
      }
    case 4:
      switch(m) {
        //case -4: return .002604166666667*y*y*y*y;
        //case -3: return 0.02083333333333*x*y*y*y*y;
        //case -2: return 0.02083333333333*(7.*x*x - 1.)*y*y;
        //case -1: return -0.125*x*(3. - 7.*x*x)*y;
        case 0: return 0.125*(35.*x*x*x*x - 30.*x*x + 3.);
        case 1: return -2.5*(7.*x*x - 3.)*x*y;
        case 2: return 7.5*(7.*x*x - 1.)*y*y;
        case 3: return -105.*x*y*y*y;
        case 4: return 105.*y*y*y*y;
      }
    case 5:
      switch(m) {
        //case -5: return 0.000260416666667*y*y*y*y*y;
        //case -4: return 0.002604166666667*x*y*y*y*y;
        //case -3: return -0.002604166666667*y*y*y*(9.*x*x - 1.);
        //case -2: return 0.0625*x*y*y*(3.*x*x - 1.);
        //case -1: return -0.0625*(21.*x*x*x*x - 14.*x*x + 1.);
        case 0: return 0.125*x*(63.*x*x*x*x - 70.*x*x + 15.);
        case 1: return -1.875*y*(21.*x*x*x*x - 14.*x*x + 1.);
        case 2: return 52.5*x*y*y*(3.*x*x - 1.);
        case 3: return -52.5*y*y*y*(9.*x*x - 1.);
        case 4: return 945.*x*y*y*y*y;
        case 5: return -945.*y*y*y*y*y;
      }
  }
  return 0; // this should never be reached
}

__device__ real X_pn(int n, real theta, real phi,
  int pp, part_struct *parts)
{
  int coeff = 0;
  for(int j = 0; j < n; j++) coeff += j+1;

  int m = 0;
  real sum = Nnm(n,m)*Pnm(n,m,theta)*parts[pp].pnm_re[coeff];

  for(m = 1; m <= n; m++) {
    coeff++;
    sum += 2.*Nnm(n,m)*Pnm(n,m,theta)
      *(parts[pp].pnm_re[coeff]*cos(m*phi)
      - parts[pp].pnm_im[coeff]*sin(m*phi));
  }

  return sum;
}

__device__ real X_phin(int n, real theta, real phi,
  int pp, part_struct *parts)
{
  int coeff = 0;
  for(int j = 0; j < n; j++) coeff += j+1;

  int m = 0;
  real sum = Nnm(n,m)*Pnm(n,m,theta)*parts[pp].phinm_re[coeff];

  for(m = 1; m <= n; m++) {
    coeff++;
    sum += 2.*Nnm(n,m)*Pnm(n,m,theta)
      *(parts[pp].phinm_re[coeff]*cos(m*phi)
      - parts[pp].phinm_im[coeff]*sin(m*phi));
  }

  return sum;
}

__device__ real Y_pn(int n, real theta, real phi, int pp, part_struct *parts)
{
  int coeff = 0;
  real ct = cos(theta);
  real st = sin(theta);
  if (st >= 0 && st < DIV_ST) st = DIV_ST;
  else if (st < 0 && st > -DIV_ST) st = -DIV_ST;

  for(int j = 0; j < n; j++) coeff += j+1;

  int m = 0;
  real sum = Nnm(n,m)
    *(-(n+1)*ct/st*Pnm(n,m,theta)+(n-m+1)/st*Pnm(n+1,m,theta))
    *parts[pp].pnm_re[coeff];

  for(m = 1; m <= n; m++) {
    coeff++;
    sum += 2.*Nnm(n,m)
      *(-(n+1)*ct/st*Pnm(n,m,theta)+(n-m+1)/st*Pnm(n+1,m,theta))
      *(parts[pp].pnm_re[coeff]*cos(m*phi)
      - parts[pp].pnm_im[coeff]*sin(m*phi));
  }

  return sum;
}


__device__ real Y_phin(int n, real theta, real phi, int pp, part_struct *parts)
{
  int coeff = 0;
  real ct = cos(theta);
  real st = sin(theta);
  if(st >= 0 && st < DIV_ST) st = DIV_ST;
  else if(st < 0 && st > -DIV_ST) st = -DIV_ST;

  for(int j = 0; j < n; j++) coeff += j+1;

  int m = 0;
  real sum = Nnm(n,m)
    *(-(n+1)*ct/st*Pnm(n,m,theta)+(n-m+1)/st*Pnm(n+1,m,theta))
    *parts[pp].phinm_re[coeff];

  for(m = 1; m <= n; m++) {
    coeff++;
    sum += 2.*Nnm(n,m)
      *(-(n+1)*ct/st*Pnm(n,m,theta)+(n-m+1)/st*Pnm(n+1,m,theta))
      *(parts[pp].phinm_re[coeff]*cos(m*phi)
      - parts[pp].phinm_im[coeff]*sin(m*phi));
  }

  return sum;
}

__device__ real Y_chin(int n, real theta, real phi, int pp, part_struct *parts)
{
  int coeff = 0;
  real ct = cos(theta);
  real st = sin(theta);
  if(st >= 0 && st < DIV_ST) st = DIV_ST;
  else if(st < 0 && st > -DIV_ST) st = -DIV_ST;

  for(int j = 0; j < n; j++) coeff += j+1;

  int m = 0;
  real sum = Nnm(n,m)
    *(-(n+1)*ct/st*Pnm(n,m,theta)+(n-m+1)/st*Pnm(n+1,m,theta))
    *parts[pp].chinm_re[coeff];

  for(m = 1; m <= n; m++) {
    coeff++;
    sum += 2.*Nnm(n,m)
      *(-(n+1)*ct/st*Pnm(n,m,theta)+(n-m+1)/st*Pnm(n+1,m,theta))
      *(parts[pp].chinm_re[coeff]*cos(m*phi)
      - parts[pp].chinm_im[coeff]*sin(m*phi));
  }

  return sum;
}

__device__ real Z_pn(int n, real theta, real phi, int pp, part_struct *parts)
{
  int coeff = 0;
  real st = sin(theta);
  if(st >= 0 && st < DIV_ST) st = DIV_ST;
  else if(st < 0 && st > -DIV_ST) st = -DIV_ST;

  for(int j = 0; j < n; j++) coeff += j+1;

  int m = 0;
  real sum = 0.;

  for(m = 1; m <= n; m++) {
    coeff++;
    sum += -2.*m/st*Nnm(n,m)*Pnm(n,m,theta)
      *(parts[pp].pnm_re[coeff]*sin(m*phi)
      + parts[pp].pnm_im[coeff]*cos(m*phi));
  }

  return sum;
}

__device__ real Z_phin(int n, real theta, real phi, int pp, part_struct *parts)
{
  int coeff = 0;
  real st = sin(theta);
  if(st >= 0 && st < DIV_ST) st = DIV_ST;
  else if(st < 0 && st > -DIV_ST) st = -DIV_ST;

  for(int j = 0; j < n; j++) coeff += j+1;

  int m = 0;
  real sum = 0.;

  for(m = 1; m <= n; m++) {
    coeff++;
    sum += -2.*m/st*Nnm(n,m)*Pnm(n,m,theta)
      *(parts[pp].phinm_re[coeff]*sin(m*phi)
      + parts[pp].phinm_im[coeff]*cos(m*phi));
  }

  return sum;
}

__device__ real Z_chin(int n, real theta, real phi, int pp, part_struct *parts)
{
  int coeff = 0;
  real st = sin(theta);
  if(st >= 0 && st < DIV_ST) st = DIV_ST;
  else if(st < 0 && st > -DIV_ST) st = -DIV_ST;

  for(int j = 0; j < n; j++) coeff += j+1;

  int m = 0;
  real sum = 0.;

  for(m = 1; m <= n; m++) {
    coeff++;
    sum += -2.*m/st*Nnm(n,m)*Pnm(n,m,theta)
      *(parts[pp].chinm_re[coeff]*sin(m*phi)
      + parts[pp].chinm_im[coeff]*cos(m*phi));
  }

  return sum;
}

__global__ void internal_u(real *u, part_struct *parts, int *flag_u, int *phase, int nparts)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tk = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;

  real y, z;     // node position
  real Yp, Zp;  // particle position

  if (tj <= _dom.Gfx._je && tk <= _dom.Gfx._ke) {
    for (int i = _dom.Gfx._is; i <= _dom.Gfx._ie; i++) {
      int C = GFX_LOC(i, tj, tk, _dom.Gfx.s1b, _dom.Gfx.s2b);
      int W = GCC_LOC(i-1, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int E = GCC_LOC(i, tj, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      int pw = phase[W];
      int pe = phase[E];
      int f = flag_u[C];

      int p = (pw > -1 && pe > -1) * phase[E];

      // Position of current thread
      //x = (i-DOM_BUF) * _dom.dx + _dom.xs;
      y = (tj-0.5) * _dom.dy + _dom.ys;
      z = (tk-0.5) * _dom.dz + _dom.zs;

      //Xp = parts[p].x;
      Yp = parts[p].y;
      Zp = parts[p].z;

      //real rx = (i - DOM_BUF) * _dom.dx + _dom.xs - parts[p].x;
      real ry = y - Yp;
      real rz = z - Zp;

      real ocrossr_x = parts[p].oy*rz - parts[p].oz*ry;

      u[C] = (pw == -1 || pe == -1 || f == -1) * u[C]
        + (pw > -1 && pe > -1 && f != -1) * (ocrossr_x + parts[p].u);
    }
  }
}

__global__ void internal_v(real *v, part_struct *parts, int *flag_v, int *phase, int nparts)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int ti = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;

  real x, z;     // node position
  real Xp, Zp;  // particle position

  if (tk <= _dom.Gfy._ke && ti <= _dom.Gfy._ie) {
    for (int j = _dom.Gfy._js; j <= _dom.Gfy._je; j++) {
      int C = GFY_LOC(ti, j, tk, _dom.Gfy.s1b, _dom.Gfy.s2b);
      int S = GCC_LOC(ti, j-1, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int N = GCC_LOC(ti, j, tk, _dom.Gcc.s1b, _dom.Gcc.s2b);

      int ps = phase[S];
      int pn = phase[N];
      int f = flag_v[C];

      int p = (ps > -1 && pn > -1) * phase[N];

      // Position of current thread
      x = (ti-0.5) * _dom.dx + _dom.xs;
      z = (tk-0.5) * _dom.dz + _dom.zs;

      Xp = parts[p].x;
      Zp = parts[p].z;

      real rx = x - Xp;
      real rz = z - Zp;

      real ocrossr_y = parts[p].oz*rx - parts[p].ox*rz;

      v[C] = (ps == -1 || pn == -1 || f == -1) * v[C]
        + (ps > -1 && pn > -1 && f != -1) * (ocrossr_y + parts[p].v);
    }
  }
}

__global__ void internal_w(real *w, part_struct *parts, int *flag_w, int *phase, int nparts)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x + DOM_BUF;
  int tj = blockDim.y*blockIdx.y + threadIdx.y + DOM_BUF;

  real x, y;     // node position
  real Xp, Yp;  // particle position

  if (ti <= _dom.Gfz._ie && tj <= _dom.Gfz._je) {
    for (int k = _dom.Gfz._ks; k <= _dom.Gfz._ke; k++) {
      int C = GFZ_LOC(ti, tj, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
      int B = GCC_LOC(ti, tj, k-1, _dom.Gcc.s1b, _dom.Gcc.s2b);
      int T = GCC_LOC(ti, tj, k, _dom.Gcc.s1b, _dom.Gcc.s2b);

      int pb = phase[B];
      int pt = phase[T];
      int f = flag_w[C];

      int p = (pb > -1 && pt > -1) * phase[T];

      // Position of current thread
      x = (ti-0.5) * _dom.dx + _dom.xs;
      y = (tj-0.5) * _dom.dy + _dom.ys;

      Xp = parts[p].x;
      Yp = parts[p].y;

      real rx = x - Xp;
      real ry = y - Yp;
      //real rz = (k - DOM_BUF) * _dom.dz + _dom.zs - parts[p].z;

      real ocrossr_z = parts[p].ox*ry - parts[p].oy*rx;

      w[C] = (pb == -1 || pt == -1 || f == -1) * w[C]
        + (pb > -1 && pt > -1 && f != -1) * (ocrossr_z + parts[p].w);
    }
  }
}

__global__ void collision_init(part_struct *parts, int nparts)
{
  int j = threadIdx.x + blockIdx.x*blockDim.x;
  if(j < nparts) {
    parts[j].iFx = 0.;
    parts[j].iFy = 0.;
    parts[j].iFz = 0.;
    parts[j].iLx = 0.;
    parts[j].iLy = 0.;
    parts[j].iLz = 0.;
  }
}

__global__ void spring_parts(part_struct *parts, int nparts, dom_struct *DOM)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  if(i < nparts && parts[i].spring_k > 0.) {
    // Correct spring positions for periodicity
    real x = parts[i].x;
    real y = parts[i].y;
    real z = parts[i].z;

    x += DOM->xl * ((x < DOM->xs) - (x > DOM->xe));
    y += DOM->yl * ((y < DOM->ys) - (y > DOM->ye));
    z += DOM->zl * ((z < DOM->zs) - (z > DOM->ze));

    real nx = x - parts[i].spring_x;
    real ny = y - parts[i].spring_y;
    real nz = z - parts[i].spring_z;
    real n = sqrt(nx*nx+ny*ny+nz*nz);
    real nhatx = nx / n;
    real nhaty = ny / n;
    real nhatz = nz / n;

    real lx = parts[i].spring_l * nhatx;
    real ly = parts[i].spring_l * nhaty;
    real lz = parts[i].spring_l * nhatz;

    // If particle center is at spring location, n will blow up and l{x,y,z}
    // will be nan
    // If position is "close enough", just set these to zero
    // If position is "far enough" (abs val > 1e16), use the regular result
    lx *= (nx*nx > 1e-16*1e-16);
    ly *= (ny*ny > 1e-16*1e-16);
    lz *= (nz*nz > 1e-16*1e-16);

    real dx = x - parts[i].spring_x - lx;
    real dy = y - parts[i].spring_y - ly;
    real dz = z - parts[i].spring_z - lz;

    parts[i].kFx = - parts[i].spring_k * dx;
    parts[i].kFy = - parts[i].spring_k * dy;
    parts[i].kFz = - parts[i].spring_k * dz;
  }
}

__global__ void collision_walls(part_struct *parts, int nparts, BC *bc, real eps,
  real mu, real rhof, real nu, int interaction_length_ratio, real dt,
  dom_struct *DOM)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int q;  // iterator

  if(i < nparts) {
    real dx = 0;
    real dy = 0;
    real dz = 0;
    real Un, Utx, Uty, Utz;
    real omx, omy, omz;

    real ai = parts[i].r;
    real h = 0;
    real hN = interaction_length_ratio * parts[i].r;
    real ah, lnah;

    real Fnx, Fny, Fnz, Ftx, Fty, Ftz;
    real Lox, Loy, Loz;

    int isTrue = 0;

    // west wall
    dx = parts[i].x - (DOM->xs + bc->dsW); // center-wall distance
    h = fabs(dx) - ai;                    // edge-wall distance

    // collision force applied ifTrue
    isTrue = (bc->pW == NEUMANN && _dom.I == DOM->Is);

    if(dx > 0 && h < hN && h > 0) {   // Lubrication
      // remove from contact list if it is there
      q = 0;
      while(parts[i].iSt[q] != -10 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(parts[i].iSt[q] == -10) {
        parts[i].iSt[q] = -1;
        parts[i].St[q] = 0.;
      }

      if(h < eps*parts[i].r) h = eps*parts[i].r;
      ah = ai/h - ai/hN;
      lnah = log(hN/h);

      Un = parts[i].u - bc->uWD;
      Utx = 0.;
      Uty = parts[i].v - bc->vWD;
      Utz = parts[i].w - bc->wWD;
      omx = parts[i].ox;
      omy = parts[i].oy;
      omz = parts[i].oz;

      Fnx = -6.*PI*mu*ai*Un*ah;
      Fny = 0.;
      Fnz = 0.;

      Ftx = 0.;
      Fty = -6.*PI*mu*ai*Uty*8./15.*lnah;
      Ftz = -6.*PI*mu*ai*Utz*8./15.*lnah;
      Ftx += 0.;
      Fty += 8.*PI*mu*ai*ai*omz*1./10.*lnah;
      Ftz += -8.*PI*mu*ai*ai*omy*1./10.*lnah;

      Lox = 0.;
      Loy = -8.*PI*mu*ai*ai*Utz*1./10.*lnah;
      Loz = 8.*PI*mu*ai*ai*Uty*1./10.*lnah;
      Lox += 0.;
      Loy += -8.*PI*mu*ai*ai*ai*omy*2./5.*lnah;
      Loz += -8.*PI*mu*ai*ai*ai*omz*2./5.*lnah;

      parts[i].iFx += isTrue * (Fnx + Ftx);
      parts[i].iFy += isTrue * (Fny + Fty);
      parts[i].iFz += isTrue * (Fnz + Ftz);
      parts[i].iLx += isTrue * Lox;
      parts[i].iLy += isTrue * Loy;
      parts[i].iLz += isTrue * Loz;
    }
    if(dx > 0 && h < 0) {       // Wall contact
      Un = parts[i].u - bc->uWD;
      real Uty = 0.5*(parts[i].v+parts[i].v0) - bc->vSD;
      real Utz = 0.5*(parts[i].w+parts[i].w0) - bc->wSD;

      // determine whether this is a new contact
      q = 0;
      while(parts[i].iSt[q] != -10 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(q == MAX_NEIGHBORS) {
        q = 0;
        while(parts[i].iSt[q] != -1) {
          q++;
        }
        parts[i].iSt[q] = -10;
        parts[i].St[q] = 1./9.*parts[i].rho/rhof*2.*parts[i].r*fabs(Un)/nu;
        parts[i].ncoll_wall += 1;
      }

      omx = 0.5*(parts[i].ox+parts[i].ox0);
      omy = 0.5*(parts[i].oy+parts[i].oy0);
      omz = 0.5*(parts[i].oz+parts[i].oz0);

      real Vy = Uty - (ai + 0.5*h)*omz;
      real Vz = Utz + (ai + 0.5*h)*omx;

      // Mindlin's theory for tangential stiffness
      real Hi = 0.5*parts[i].E/(1.+parts[i].sigma);
      real kt = 8./((1.-parts[i].sigma*parts[i].sigma)/Hi
        +(1.-parts[i].sigma*parts[i].sigma)/Hi)/sqrt(1./ai)*sqrt(-h);

      real sy = Vy * dt;
      real sz = Vz * dt;

      lnah = 0;
      real k = 4./3./((1.-parts[i].sigma*parts[i].sigma)/parts[i].E
        + (1.-parts[i].sigma*parts[i].sigma)/parts[i].E)/sqrt(1./ai);

      // estimate damping coefficient
      real xcx0 = 1.e-4;
      real e = parts[i].e_dry + (1.+parts[i].e_dry)/parts[i].St[q]*log(xcx0);
      if(e < 0) e = 0;
      real alpha = -2.263*pow(e,0.3948)+2.22;

      real eta = alpha*sqrt(4./3.*PI*ai*ai*ai*parts[i].rho*k*sqrt(-h));

      // use the same coeff_fric for particle and wall
      real coeff_fric = parts[i].coeff_fric;
      //real Fty = parts[i].ty - k * sy;
      //real Ftz = parts[i].tz - k * sz;
      real Fty = - k * sy;
      real Ftz = - k * sz;
      real Ft = sqrt(Fty*Fty + Ftz*Ftz);
      if(Ft > fabs(coeff_fric * (sqrt(-h*h*h)*k - eta*Un))) {
        Fty = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Fty / Ft;
        Ftz = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Ftz / Ft;
      }

      parts[i].iFx += isTrue * (sqrt(-h*h*h)*k - eta*Un);
      parts[i].iFy += isTrue * Fty;
      parts[i].iFz += isTrue * Ftz;

      parts[i].iLy += isTrue * (ai+0.5*h) * Ftz;
      parts[i].iLz -= isTrue * (ai+0.5*h) * Fty;
    }

    // east wall
    dx = parts[i].x - (DOM->xe - bc->dsE);
    h = fabs(dx) - ai;
    isTrue = (bc->pE == NEUMANN && _dom.I == DOM->Ie);
    if(dx < 0 && h < hN && h > 0) {
      // remove from contact list if it is there
      q = 0;
      while(parts[i].iSt[q] != -11 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(parts[i].iSt[q] == -11) {
        parts[i].iSt[q] = -1;
        parts[i].St[q] = 0.;
      }

      if(h < eps*parts[i].r) h = eps*parts[i].r;
      ah = ai/h - ai/hN;
      lnah = log(hN/h);

      Un = parts[i].u - bc->uED;
      Utx = 0.;
      Uty = parts[i].v - bc->vED;
      Utz = parts[i].w - bc->wED;
      omx = parts[i].ox;
      omy = parts[i].oy;
      omz = parts[i].oz;

      Fnx = -6.*PI*mu*ai*Un*ah;
      Fny = 0.;
      Fnz = 0.;

      Ftx = 0.;
      Fty = -6.*PI*mu*ai*Uty*8./15.*lnah;
      Ftz = -6.*PI*mu*ai*Utz*8./15.*lnah;
      Ftx += 0.;
      Fty += -8.*PI*mu*ai*ai*omz*1./10.*lnah;
      Ftz += 8.*PI*mu*ai*ai*omy*1./10.*lnah;

      Lox = 0.;
      Loy = 8.*PI*mu*ai*ai*Utz*1./10.*lnah;
      Loz = -8.*PI*mu*ai*ai*Uty*1./10.*lnah;
      Lox += 0.;
      Loy += -8.*PI*mu*ai*ai*ai*omy*2./5.*lnah;
      Loz += -8.*PI*mu*ai*ai*ai*omz*2./5.*lnah;

      parts[i].iFx += isTrue * (Fnx + Ftx);
      parts[i].iFy += isTrue * (Fny + Fty);
      parts[i].iFz += isTrue * (Fnz + Ftz);
      parts[i].iLx += isTrue * Lox;
      parts[i].iLy += isTrue * Loy;
      parts[i].iLz += isTrue * Loz;
    } 
    if(dx < 0 && h < 0) {
      Un = -(parts[i].u - bc->uED);
      real Uty = 0.5*(parts[i].v+parts[i].v0) - bc->vSD;
      real Utz = 0.5*(parts[i].w+parts[i].w0) - bc->wSD;

      // determine whether this is a new contact
      q = 0;
      while(parts[i].iSt[q] != -11 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(q == MAX_NEIGHBORS) {
        q = 0;
        while(parts[i].iSt[q] != -1) {
          q++;
        }
        parts[i].iSt[q] = -11;
        parts[i].St[q] = 1./9.*parts[i].rho/rhof*2.*parts[i].r*fabs(Un)/nu;
        parts[i].ncoll_wall += 1;
      }

      omx = 0.5*(parts[i].ox+parts[i].ox0);
      omy = 0.5*(parts[i].oy+parts[i].oy0);
      omz = 0.5*(parts[i].oz+parts[i].oz0);

      real Vy = -(Uty - (ai + 0.5*h)*omz);
      real Vz = -(Utz + (ai + 0.5*h)*omx);

      real Hi = 0.5*parts[i].E/(1.+parts[i].sigma);
      real kt = 8./((1.-parts[i].sigma*parts[i].sigma)/Hi
        +(1.-parts[i].sigma*parts[i].sigma)/Hi)/sqrt(1./ai)*sqrt(-h);

      real sy = Vy * dt;
      real sz = Vz * dt;

      lnah = 0;
      real k = 4./3./((1.-parts[i].sigma*parts[i].sigma)/parts[i].E
        + (1.-parts[i].sigma*parts[i].sigma)/parts[i].E)/sqrt(1./ai);

      // estimate damping coefficient
      real xcx0 = 1.e-4;
      real e = parts[i].e_dry + (1.+parts[i].e_dry)/parts[i].St[q]*log(xcx0);
      if(e < 0) e = 0;
      real alpha = -2.263*pow(e,0.3948)+2.22;

      real eta = alpha*sqrt(4./3.*PI*ai*ai*ai*parts[i].rho*k*sqrt(-h));

      // use the same coeff_fric for particle and wall
      real coeff_fric = parts[i].coeff_fric;
      real Fty = -kt * sy;
      real Ftz = -kt * sz;
      real Ft = sqrt(Fty*Fty + Ftz*Ftz);
      if(Ft > fabs(coeff_fric * (sqrt(-h*h*h)*k - eta*Un))) {
        Fty = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Fty / Ft;
        Ftz = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Ftz / Ft;
      }

      parts[i].iFx -= isTrue * (sqrt(-h*h*h)*k - eta*Un);
      parts[i].iFy += isTrue * Fty;
      parts[i].iFz += isTrue * Ftz;

      parts[i].iLy -= isTrue * (ai+0.5*h) * Ftz;
      parts[i].iLz += isTrue * (ai+0.5*h) * Ftx;
    }

    // south wall
    dy = parts[i].y - (DOM->ys + bc->dsS);
    h = fabs(dy) - ai;
    isTrue = (bc->pS == NEUMANN && _dom.J == DOM->Js);
    if(dy > 0 && h < hN && h > 0) {
      // remove from contact list if it is there
      q = 0;
      while(parts[i].iSt[q] != -12 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(parts[i].iSt[q] == -12) {
        parts[i].iSt[q] = -1;
        parts[i].St[q] = 0.;
      }

      if(h < eps*parts[i].r) h = eps*parts[i].r;
      ah = ai/h - ai/hN;
      lnah = log(hN/h);

      Un = parts[i].v - bc->vSD;
      Utx = parts[i].u - bc->uSD;
      Uty = 0.;
      Utz = parts[i].w - bc->wSD;
      omx = parts[i].ox;
      omy = parts[i].oy;
      omz = parts[i].oz;

      Fnx = 0.;
      Fny = -6.*PI*mu*ai*Un*ah;
      Fnz = 0.;

      Ftx = -6.*PI*mu*ai*Utx*8./15.*lnah;
      Fty = 0.;
      Ftz = -6.*PI*mu*ai*Utz*8./15.*lnah;
      Ftx += -8.*PI*mu*ai*ai*omz*1./10.*lnah;
      Fty += 0.;
      Ftz += 8.*PI*mu*ai*ai*omx*1./10.*lnah;

      Lox = 8.*PI*mu*ai*ai*Utz*1./10.*lnah;
      Loy = 0.;
      Loz = -8.*PI*mu*ai*ai*Utx*1./10.*lnah;
      Lox += -8.*PI*mu*ai*ai*ai*omx*2./5.*lnah;
      Loy += 0.;
      Loz += -8.*PI*mu*ai*ai*ai*omz*2./5.*lnah;

      parts[i].iFx += isTrue * (Fnx + Ftx);
      parts[i].iFy += isTrue * (Fny + Fty);
      parts[i].iFz += isTrue * (Fnz + Ftz);
      parts[i].iLx += isTrue * Lox;
      parts[i].iLy += isTrue * Loy;
      parts[i].iLz += isTrue * Loz;
    }
    if(dy > 0 && h < 0) {
      Un = parts[i].v - bc->vSD;
      real Utx = 0.5*(parts[i].u+parts[i].u0) - bc->uSD;
      real Utz = 0.5*(parts[i].w+parts[i].w0) - bc->wSD;

      // determine whether this is a new contact
      q = 0;
      while(parts[i].iSt[q] != -12 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(q == MAX_NEIGHBORS) {
        q = 0;
        while(parts[i].iSt[q] != -1) {
          q++;
        }
        parts[i].iSt[q] = -12;
        parts[i].St[q] = 1./9.*parts[i].rho/rhof*2.*parts[i].r*fabs(Un)/nu;
        parts[i].ncoll_wall += 1;
      }

      omx = 0.5*(parts[i].ox+parts[i].ox0);
      omy = 0.5*(parts[i].oy+parts[i].oy0);
      omz = 0.5*(parts[i].oz+parts[i].oz0);

      real Vx = Utx + (ai + 0.5*h)*omz;
      real Vz = Utz - (ai + 0.5*h)*omx;

      real Hi = 0.5*parts[i].E/(1.+parts[i].sigma);
      real kt = 8./((1.-parts[i].sigma*parts[i].sigma)/Hi
        +(1.-parts[i].sigma*parts[i].sigma)/Hi)/sqrt(1./ai)*sqrt(-h);

      real sx = Vx * dt;
      real sz = Vz * dt;

      lnah = 0;
      real k = 4./3./((1.-parts[i].sigma*parts[i].sigma)/parts[i].E
        + (1.-parts[i].sigma*parts[i].sigma)/parts[i].E)/sqrt(1./ai);

      // estimate damping coefficient
      real xcx0 = 1.e-4;
      real e = parts[i].e_dry + (1.+parts[i].e_dry)/parts[i].St[q]*log(xcx0);
      if(e < 0) e = 0;
      real alpha = -2.263*pow(e,0.3948)+2.22;

      real eta = alpha*sqrt(4./3.*PI*ai*ai*ai*parts[i].rho*k*sqrt(-h));

      // use the same coeff_fric for particle and wall
      real coeff_fric = parts[i].coeff_fric;
      real Ftx = -kt * sx;
      real Ftz = -kt * sz;
      real Ft = sqrt(Ftx*Ftx + Ftz*Ftz);
      if(Ft > fabs(coeff_fric * (sqrt(-h*h*h)*k - eta*Un))) {
        Ftx = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Ftx / Ft;
        Ftz = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Ftz / Ft;
      }

      parts[i].iFx += isTrue * Ftx;
      parts[i].iFy += isTrue * (sqrt(-h*h*h)*k - eta*Un);
      parts[i].iFz += isTrue * Ftz;

      parts[i].iLx -= isTrue * (ai+0.5*h) * Ftz;
      parts[i].iLz += isTrue * (ai+0.5*h) * Ftx;
    }

    // north wall
    dy = parts[i].y - (DOM->ye - bc->dsN);
    h = fabs(dy) - ai;
    isTrue = (bc->pN == NEUMANN && _dom.J == DOM->Je);
    if(dy < 0 && h < hN && h > 0) {
      // remove from contact list if it is there
      q = 0;
      while(parts[i].iSt[q] != -13 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(parts[i].iSt[q] == -13) {
        parts[i].iSt[q] = -1;
        parts[i].St[q] = 0.;
      }

      if(h < eps*parts[i].r) h = eps*parts[i].r;
      ah = ai/h - ai/hN;
      lnah = log(hN/h);

      Un = parts[i].v - bc->vND;
      Utx = parts[i].u - bc->uND;
      Uty = 0.;
      Utz = parts[i].w - bc->wND;
      omx = parts[i].ox;
      omy = parts[i].oy;
      omz = parts[i].oz;

      Fnx = 0.;
      Fny = -6.*PI*mu*ai*Un*ah;
      Fnz = 0.;

      Ftx = -6.*PI*mu*ai*Utx*8./15.*lnah;
      Fty = 0.;
      Ftz = -6.*PI*mu*ai*Utz*8./15.*lnah;
      Ftx += 8.*PI*mu*ai*ai*omz*1./10.*lnah;
      Fty += 0.;
      Ftz += -8.*PI*mu*ai*ai*omx*1./10.*lnah;

      Lox = -8.*PI*mu*ai*ai*Utz*1./10.*lnah;
      Loy = 0.;
      Loz = 8.*PI*mu*ai*ai*Utx*1./10.*lnah;
      Lox += -8.*PI*mu*ai*ai*ai*omx*2./5.*lnah;
      Loy += 0.;
      Loz += -8.*PI*mu*ai*ai*ai*omz*2./5.*lnah;

      parts[i].iFx += isTrue * (Fnx + Ftx);
      parts[i].iFy += isTrue * (Fny + Fty);
      parts[i].iFz += isTrue * (Fnz + Ftz);
      parts[i].iLx += isTrue * Lox;
      parts[i].iLy += isTrue * Loy;
      parts[i].iLz += isTrue * Loz;
    }
    if(dy < 0 && h < 0) {
      Un = -(parts[i].v - bc->vND);
      real Utx = 0.5*(parts[i].u+parts[i].u0) - bc->uSD;
      real Utz = 0.5*(parts[i].w+parts[i].w0) - bc->wSD;

      // determine whether this is a new contact
      q = 0;
      while(parts[i].iSt[q] != -13 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(q == MAX_NEIGHBORS) {
        q = 0;
        while(parts[i].iSt[q] != -1) {
          q++;
        }
        parts[i].iSt[q] = -13;
        parts[i].St[q] = 1./9.*parts[i].rho/rhof*2.*parts[i].r*fabs(Un)/nu;
        parts[i].ncoll_wall += 1;
      }

      omx = 0.5*(parts[i].ox+parts[i].ox0);
      omy = 0.5*(parts[i].oy+parts[i].oy0);
      omz = 0.5*(parts[i].oz+parts[i].oz0);

      real Vx = -(Utx + (ai + 0.5*h)*omz);
      real Vz = -(Utz - (ai + 0.5*h)*omx);

      real Hi = 0.5*parts[i].E/(1.+parts[i].sigma);
      real kt = 8./((1.-parts[i].sigma*parts[i].sigma)/Hi
        +(1.-parts[i].sigma*parts[i].sigma)/Hi)/sqrt(1./ai)*sqrt(-h);

      real sx = Vx * dt;
      real sz = Vz * dt;

      lnah = 0;
      real k = 4./3./((1.-parts[i].sigma*parts[i].sigma)/parts[i].E
        + (1.-parts[i].sigma*parts[i].sigma)/parts[i].E)/sqrt(1./ai);

      // estimate damping coefficient
      real xcx0 = 1.e-4;
      real e = parts[i].e_dry + (1.+parts[i].e_dry)/parts[i].St[q]*log(xcx0);
      if(e < 0) e = 0;
      real alpha = -2.263*pow(e,0.3948)+2.22;

      real eta = alpha*sqrt(4./3.*PI*ai*ai*ai*parts[i].rho*k*sqrt(-h));

      // use the same coeff_fric for particle and wall
      real coeff_fric = parts[i].coeff_fric;
      real Ftx = -kt * sx;
      real Ftz = -kt * sz;
      real Ft = sqrt(Ftx*Ftx + Ftz*Ftz);
      if(Ft > fabs(coeff_fric * (sqrt(-h*h*h)*k - eta*Un))) {
        Ftx = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Ftx / Ft;
        Ftz = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Ftz / Ft;
      }

      parts[i].iFx += isTrue * Ftx;
      parts[i].iFy -= isTrue * (sqrt(-h*h*h)*k - eta*Un);
      parts[i].iFz += isTrue * Ftz;

      parts[i].iLx += isTrue * (ai+0.5*h) * Ftz;
      parts[i].iLz -= isTrue * (ai+0.5*h) * Ftx;
    }

    // bottom wall
    dz = parts[i].z - (DOM->zs + bc->dsB);
    h = fabs(dz) - ai;
    isTrue = (bc->pB == NEUMANN && _dom.K == DOM->Ks);
    if(dz > 0 && h < hN && h > 0) {
      // remove from contact list if it is there
      q = 0;
      while(parts[i].iSt[q] != -14 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(parts[i].iSt[q] == -14) {
        parts[i].iSt[q] = -1;
        parts[i].St[q] = 0.;
      }

      if(h < eps*parts[i].r) h = eps*parts[i].r;
      ah = ai/h - ai/hN;
      lnah = log(hN/h);

      Un = parts[i].w - bc->wBD;
      Utx = parts[i].u - bc->uBD;
      Uty = parts[i].v - bc->vBD;
      Utz = 0.;
      omx = parts[i].ox;
      omy = parts[i].oy;
      omz = parts[i].oz;

      Fnx = 0.;
      Fny = 0.;
      Fnz = -6.*PI*mu*ai*Un*ah;

      Ftx = -6.*PI*mu*ai*Utx*8./15.*lnah;
      Fty = -6.*PI*mu*ai*Uty*8./15.*lnah;
      Ftz = 0.;
      Ftx += 8.*PI*mu*ai*ai*omy*1./10.*lnah;
      Fty += -8.*PI*mu*ai*ai*omx*1./10.*lnah;
      Ftz += 0.;

      Lox = -8.*PI*mu*ai*ai*Uty*1./10.*lnah;
      Loy = 8.*PI*mu*ai*ai*Utx*1./10.*lnah;
      Loz = 0.;
      Lox += -8.*PI*mu*ai*ai*ai*omx*2./5.*lnah;
      Loy += -8.*PI*mu*ai*ai*ai*omy*2./5.*lnah;
      Loz += 0.;

      parts[i].iFx += isTrue * (Fnx + Ftx);
      parts[i].iFy += isTrue * (Fny + Fty);
      parts[i].iFz += isTrue * (Fnz + Ftz);
      parts[i].iLx += isTrue * Lox;
      parts[i].iLy += isTrue * Loy;
      parts[i].iLz += isTrue * Loz;
    }
    if(dz > 0 && h < 0) {
      Un = parts[i].w - bc->wBD;
      real Utx = 0.5*(parts[i].u+parts[i].u0) - bc->uSD;
      real Uty = 0.5*(parts[i].v+parts[i].v0) - bc->vSD;

      // determine whether this is a new contact
      q = 0;
      while(parts[i].iSt[q] != -14 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(q == MAX_NEIGHBORS) {
        q = 0;
        while(parts[i].iSt[q] != -1) {
          q++;
        }
        parts[i].iSt[q] = -14;
        parts[i].St[q] = 1./9.*parts[i].rho/rhof*2.*parts[i].r*fabs(Un)/nu;
        parts[i].ncoll_wall += 1;
      }

      omx = 0.5*(parts[i].ox+parts[i].ox0);
      omy = 0.5*(parts[i].oy+parts[i].oy0);
      omz = 0.5*(parts[i].oz+parts[i].oz0);

      real Vx = Utx - (ai + 0.5*h)*omy;
      real Vy = Uty + (ai + 0.5*h)*omx;

      real Hi = 0.5*parts[i].E/(1.+parts[i].sigma);
      real kt = 8./((1.-parts[i].sigma*parts[i].sigma)/Hi
        +(1.-parts[i].sigma*parts[i].sigma)/Hi)/sqrt(1./ai)*sqrt(-h);

      real sx = Vx * dt;
      real sy = Vy * dt;

      lnah = 0;
      real k = 4./3./((1.-parts[i].sigma*parts[i].sigma)/parts[i].E
        + (1.-parts[i].sigma*parts[i].sigma)/parts[i].E)/sqrt(1./ai);

      // estimate damping coefficient
      real xcx0 = 1.e-4;
      real e = parts[i].e_dry + (1.+parts[i].e_dry)/parts[i].St[q]*log(xcx0);
      if(e < 0) e = 0;
      real alpha = -2.263*pow(e,0.3948)+2.22;

      real eta = alpha*sqrt(4./3.*PI*ai*ai*ai*parts[i].rho*k*sqrt(-h));

      // use the same coeff_fric for particle and wall
      real coeff_fric = parts[i].coeff_fric;
      real Ftx = -kt * sx;
      real Fty = -kt * sy;
      real Ft = sqrt(Ftx*Ftx + Fty*Fty);
      if(Ft > fabs(coeff_fric * (sqrt(-h*h*h)*k - eta*Un))) {
        Ftx = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Ftx / Ft;
        Fty = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Fty / Ft;
      }

      parts[i].iFx += isTrue * Ftx;
      parts[i].iFy += isTrue * Fty;
      parts[i].iFz += isTrue * (sqrt(-h*h*h)*k - eta*Un);

      parts[i].iLx += isTrue * (ai+0.5*h) * Fty;
      parts[i].iLy -= isTrue * (ai+0.5*h) * Ftx;
    }

    // top wall
    dz = parts[i].z - (DOM->ze - bc->dsT);
    h = fabs(dz) - ai;
    isTrue = (bc->pT == NEUMANN && _dom.K == DOM->Ke);
    if(dz < 0 && h < hN && h > 0) {
      // remove from contact list if it is there
      q = 0;
      while(parts[i].iSt[q] != -15 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(parts[i].iSt[q] == -15) {
        parts[i].iSt[q] = -1;
        parts[i].St[q] = 0.;
      }

      if(h < eps*parts[i].r) h = eps*parts[i].r;
      ah = ai/h - ai/hN;
      lnah = log(hN/h);

      Un = parts[i].w - bc->wTD;
      Utx = parts[i].u - bc->uTD;
      Uty = parts[i].v - bc->vTD;
      Utz = 0.;
      omx = parts[i].ox;
      omy = parts[i].oy;
      omz = parts[i].oz;

      Fnx = 0.;
      Fny = 0.;
      Fnz = -6.*PI*mu*ai*Un*ah;

      Ftx = -6.*PI*mu*ai*Utx*8./15.*lnah;
      Fty = -6.*PI*mu*ai*Uty*8./15.*lnah;
      Ftz = 0.;
      Ftx += -8.*PI*mu*ai*ai*omy*1./10.*lnah;
      Fty += 8.*PI*mu*ai*ai*omx*1./10.*lnah;
      Ftz += 0.;

      Lox = 8.*PI*mu*ai*ai*Uty*1./10.*lnah;
      Loy = -8.*PI*mu*ai*ai*Utx*1./10.*lnah;
      Loz = 0.;
      Lox += -8.*PI*mu*ai*ai*ai*omx*2./5.*lnah;
      Loy += -8.*PI*mu*ai*ai*ai*omy*2./5.*lnah;
      Loz += 0.;

      parts[i].iFx += isTrue * (Fnx + Ftx);
      parts[i].iFy += isTrue * (Fny + Fty);
      parts[i].iFz += isTrue * (Fnz + Ftz);
      parts[i].iLx += isTrue * Lox;
      parts[i].iLy += isTrue * Loy;
      parts[i].iLz += isTrue * Loz;
    }
    if(dz < 0 && h < 0) {
      Un = -(parts[i].w - bc->wTD);
      real Utx = 0.5*(parts[i].u+parts[i].u0) - bc->uSD;
      real Uty = 0.5*(parts[i].v+parts[i].v0) - bc->vSD;

      // determine whether this is a new contact
      q = 0;
      while(parts[i].iSt[q] != -15 && q < MAX_NEIGHBORS) {
        q++;
      }
      if(q == MAX_NEIGHBORS) {
        q = 0;
        while(parts[i].iSt[q] != -1) {
          q++;
        }
        parts[i].iSt[q] = -15;
        parts[i].St[q] = 1./9.*parts[i].rho/rhof*2.*parts[i].r*fabs(Un)/nu;
        parts[i].ncoll_wall += 1;
      }

      omx = 0.5*(parts[i].ox+parts[i].ox0);
      omy = 0.5*(parts[i].oy+parts[i].oy0);
      omz = 0.5*(parts[i].oz+parts[i].oz0);

      real Vx = -(Utx - (ai + 0.5*h)*omy);
      real Vy = -(Uty + (ai + 0.5*h)*omx);

      real Hi = 0.5*parts[i].E/(1.+parts[i].sigma);
      real kt = 8./((1.-parts[i].sigma*parts[i].sigma)/Hi
        +(1.-parts[i].sigma*parts[i].sigma)/Hi)/sqrt(1./ai)*sqrt(-h);

      real sx = Vx * dt;
      real sy = Vy * dt;

      lnah = 0;
      real k = 4./3./((1.-parts[i].sigma*parts[i].sigma)/parts[i].E
        + (1.-parts[i].sigma*parts[i].sigma)/parts[i].E)/sqrt(1./ai);

      // estimate damping coefficient
      real xcx0 = 1.e-4;
      real e = parts[i].e_dry + (1.+parts[i].e_dry)/parts[i].St[q]*log(xcx0);
      if(e < 0) e = 0;
      real alpha = -2.263*pow(e,0.3948)+2.22;

      real eta = alpha*sqrt(4./3.*PI*ai*ai*ai*parts[i].rho*k*sqrt(-h));

      // use the same coeff_fric for particle and wall
      real coeff_fric = parts[i].coeff_fric;
      real Ftx = -kt * sx;
      real Fty = -kt * sy;
      real Ft = sqrt(Ftx*Ftx + Fty*Fty);
      if(Ft > fabs(coeff_fric * (sqrt(-h*h*h)*k - eta*Un))) {
        Ftx = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Ftx / Ft;
        Fty = coeff_fric * (sqrt(-h*h*h)*k - eta*Un) * Fty / Ft;
      }

      parts[i].iFx += isTrue * Ftx;
      parts[i].iFy += isTrue * Fty;
      parts[i].iFz -= isTrue * (sqrt(-h*h*h)*k - eta*Un);

      parts[i].iLx -= isTrue * (ai+0.5*h) * Fty;
      parts[i].iLy += isTrue * (ai+0.5*h) * Ftx;
    }
  }
}

__global__ void collision_parts(part_struct *parts, int nparts,
  real eps, real mu, real rhof, real nu, BC *bc, int *bin_start,
  int *bin_end, int *part_bin, int *part_ind,
  int interaction_length_ratio, real dt)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;

  if (index < nparts) {

    int i = part_ind[index];
    int bin = part_bin[index];
    int j;                        // target indices
    int q;                                // iterator

    int kb = floorf(bin/_bins.Gcc.s2b);
    int jb = floorf((bin - kb*_bins.Gcc.s2b) / _bins.Gcc.s1b);
    int ib = bin - kb*_bins.Gcc.s2b - jb*_bins.Gcc.s1b;

    // loop over adjacent bins
    if ((ib >= _bins.Gcc._is && ib <= _bins.Gcc._ie) &&
        (jb >= _bins.Gcc._js && jb <= _bins.Gcc._je) &&
        (kb >= _bins.Gcc._ks && kb <= _bins.Gcc._ke)) {
      for (int n = -1; n <= 1; n++) {
        for (int m = -1; m <= 1; m++) {
          for (int l = -1; l <= 1; l++) {
            int tbin = GCC_LOC(ib + l, jb + m, kb + n,
                                  _bins.Gcc.s1b, _bins.Gcc.s2b);

            if (bin_start[tbin] != -1) {    // if bin is not empty
              for (int targ = bin_start[tbin]; targ <= bin_end[tbin]; targ++) {
                j = part_ind[targ];
                if (j != i) {               // if its not original part

                  // calculate forces
                  real ai = parts[i].r;
                  real aj = parts[j].r;
                  real B = aj / ai;
                  real hN = interaction_length_ratio * parts[i].r;

                  real ux, uy, uz;
                  real rx, ry, rz, r;
                  real h, ah, lnah;
                  real nx, ny, nz, udotn;
                  real unx, uny, unz, utx, uty, utz, ut;
                  real tx, ty, tz, t, bx, by, bz, b;
                  real omegax, omegay, omegaz, omega;
                  real ocrossnx, ocrossny, ocrossnz;
                  real utcrossnx, utcrossny, utcrossnz;
                  real opB;
                  real Fnx, Fny, Fnz, Ftx, Fty, Ftz, Lox, Loy, Loz;

                  rx = parts[i].x - parts[j].x;
                  ry = parts[i].y - parts[j].y;
                  rz = parts[i].z - parts[j].z;

                  ux = 0.5*((parts[i].u - parts[j].u)
                          + (parts[i].u0 - parts[j].u0));
                  uy = 0.5*((parts[i].v - parts[j].v)
                               + (parts[i].v0 - parts[j].v0));
                  uz = 0.5*((parts[i].w - parts[j].w)
                               + (parts[i].w0 - parts[j].w0));

                  r = sqrt(rx*rx + ry*ry + rz*rz);

                  omegax = 0.5*((parts[i].ox + parts[j].ox)
                                   + (parts[i].ox0 + parts[j].ox0));
                  omegay = 0.5*((parts[i].oy + parts[j].oy)
                                   + (parts[i].oy0 + parts[j].oy0));
                  omegaz = 0.5*((parts[i].oz + parts[j].oz)
                                   + (parts[i].oz0 + parts[j].oz0));

                  omega = sqrt(omegax*omegax + omegay*omegay + omegaz*omegaz);

                  h = r - ai - aj;

                  nx = rx / r;
                  ny = ry / r;
                  nz = rz / r;

                  udotn = ux * nx + uy * ny + uz * nz;

                  unx = udotn * nx;
                  uny = udotn * ny;
                  unz = udotn * nz;

                  utx = ux - unx;
                  uty = uy - uny;
                  utz = uz - unz;

                  ut = sqrt(utx*utx + uty*uty + utz*utz);

                  if(ut > 0) {
                    tx = utx / ut;
                    ty = uty / ut;
                    tz = utz / ut;

                    bx = ny*tz - nz*ty;
                    by = -nx*tz + nz*tx;
                    bz = nx*ty - ny*tx;

                    b = sqrt(bx*bx + by*by + bz*bz);

                    bx = bx / b;
                    by = by / b;
                    bz = bz / b;

                  } else if(omega > 0) {
                    bx = omegax / omega;
                    by = omegay / omega;
                    bz = omegaz / omega;

                    tx = by*nz - bz*ny;
                    ty = -bx*nz + bz*nx;
                    tz = bx*ny - by*nx;

                    t = sqrt(tx*tx + ty*ty + tz*tz);

                    tx = tx / t;
                    ty = ty / t;
                    tz = tz / t;
                  } else {
                    tx = 1.;
                    ty = 0.;
                    tz = 0.;

                    bx = ny*tz - nz*ty;
                    by = -nx*tz + nz*tx;
                    bz = nx*ty - ny*tx;

                    b = sqrt(bx*bx + by*by + bz*bz);

                    bx = bx / b;
                    by = by / b;
                    bz = bz / b;
                  }

                  opB = 1 + B;

                  ocrossnx = omegay*nz - omegaz*ny;
                  ocrossny = -omegax*nz + omegaz*nx;
                  ocrossnz = omegax*ny - omegay*nx;

                  utcrossnx = uty*nz - utz*ny;
                  utcrossny = -utx*nz + utz*nx;
                  utcrossnz = utx*ny - uty*nx;

                  if(h < hN && h > 0) {
                    // remove contact from list if it is there
                    q = 0;
                    while(parts[i].iSt[q] != parts[j].N && q < MAX_NEIGHBORS) {
                      q++;
                    }
                    if(parts[i].iSt[q] == parts[j].N) {
                      parts[i].iSt[q] = -1;
                      parts[i].St[q] = 0.;
                    }

                    if(h < eps*parts[i].r) h = eps*parts[i].r;
                    ah = ai/h - ai/hN;
                    lnah = log(hN/h);
                    Fnx = -1. * B*B / (opB*opB) * ah
                      - B*(1.+7.*B+B*B)/(5.*opB*opB*opB)*lnah;
                    Fny = Fnx;
                    Fnz = Fnx;
                    Fnx *= 6.*PI*mu*ai*unx;
                    Fny *= 6.*PI*mu*ai*uny;
                    Fnz *= 6.*PI*mu*ai*unz;

                    Ftx = -6.*PI*mu*ai*utx*4.*B*(2.+B+2.*B*B)
                      /(15.*opB*opB*opB)*lnah;
                    Fty = -6.*PI*mu*ai*uty*4.*B*(2.+B+2.*B*B)
                      /(15.*opB*opB*opB)*lnah;
                    Ftz = -6.*PI*mu*ai*utz*4.*B*(2.+B+2.*B*B)
                      /(15.*opB*opB*opB)*lnah;
                    Ftx += 8.*PI*mu*ai*ai*ocrossnx*B*(4.+B)/(10.*opB*opB)*lnah;
                    Fty += 8.*PI*mu*ai*ai*ocrossny*B*(4.+B)/(10.*opB*opB)*lnah;
                    Ftz += 8.*PI*mu*ai*ai*ocrossnz*B*(4.+B)/(10.*opB*opB)*lnah;

                    Lox = -8.*PI*mu*ai*ai*utcrossnx*B*(4.+B)/(10.*opB*opB)*lnah;
                    Loy = -8.*PI*mu*ai*ai*utcrossny*B*(4.+B)/(10.*opB*opB)*lnah;
                    Loz = -8.*PI*mu*ai*ai*utcrossnz*B*(4.+B)/(10.*opB*opB)*lnah;
                    Lox += -8.*PI*mu*ai*ai*ai*omegax*2.*B/(5.*opB)*lnah;
                    Loy += -8.*PI*mu*ai*ai*ai*omegay*2.*B/(5.*opB)*lnah;
                    Loz += -8.*PI*mu*ai*ai*ai*omegaz*2.*B/(5.*opB)*lnah;
                  } else {
                    ah = 0;
                    lnah = 0;
                    Fnx = 0;
                    Fny = 0;
                    Fnz = 0;
                    Ftx = 0;
                    Fty = 0;
                    Ftz = 0;
                    Lox = 0;
                    Loy = 0;
                    Loz = 0;
                  }

                  if(h < 0) {
                    // determine whether this is a new contact
                    q = 0;
                    while(parts[i].iSt[q] != parts[j].N && q < MAX_NEIGHBORS) {
                      q++;
                    }
                    if(q == MAX_NEIGHBORS) {
                      q = 0;
                      while(parts[i].iSt[q] != -1) {
                        q++;
                      }
                      parts[i].iSt[q] = parts[j].N;
                      parts[i].St[q] = 1./9.*parts[i].rho/rhof*2.*parts[i].r
                        *fabs(udotn)/nu;
                      // Increment collision counter -- will increment for each
                      // particle involved in the collision.
                      // Chances of integer overflow here are very slim, since 
                      // this is on a per-particle basis. Could used unsigned 
                      // long long int if really necessary
                      parts[i].ncoll_part += 1;
                    }

                    real Vx = -utx + 0.5*(ai + aj + h)*ocrossnx;
                    real Vy = -uty + 0.5*(ai + aj + h)*ocrossny;
                    real Vz = -utz + 0.5*(ai + aj + h)*ocrossnz;

                    real Hi = 0.5*parts[i].E/(1.+parts[i].sigma);
                    real kt = 8./((1.-parts[i].sigma*parts[i].sigma)/Hi
                      +(1.-parts[i].sigma*parts[i].sigma)/Hi)/sqrt(1./ai)
                      *sqrt(-h);

                    real Vdotn = Vx*nx + Vy*ny + Vz*nz;

                    real sx = (Vx - Vdotn * nx) * dt;
                    real sy = (Vy - Vdotn * ny) * dt;
                    real sz = (Vz - Vdotn * nz) * dt;

                    ah = 0;
                    lnah = 0;
                    real k = 4./3./((1.-parts[i].sigma*parts[i].sigma)/parts[i].E
                      + (1.-parts[j].sigma*parts[j].sigma)/parts[j].E)
                      /sqrt(1./ai + 1./aj);
                    // estimate damping coefficient
                    real xcx0 = 1.e-4;
                    real e = parts[i].e_dry + (1.+parts[i].e_dry)/parts[i].St[q]
                      *log(xcx0);
                    if(e < 0) e = 0;
                    real alpha = -2.263*pow(e,0.3948)+2.22;
                    real eta = alpha*sqrt(4./3.*PI*ai*ai*ai*parts[i].rho
                      *k*sqrt(-h));

                    // normal contact forces
                    Fnx = (sqrt(-h*h*h)*k - eta*udotn)*nx;
                    Fny = (sqrt(-h*h*h)*k - eta*udotn)*ny;
                    Fnz = (sqrt(-h*h*h)*k - eta*udotn)*nz;

                    // tangential contact forces
                    real coeff_fric = 0.5 * (parts[i].coeff_fric
                      + parts[j].coeff_fric);
                    Ftx = -kt * sx;
                    Fty = -kt * sy;
                    Ftz = -kt * sz;
                    real Ftdotn = Ftx*nx + Fty*ny + Ftz*nz;
                    Ftx = Ftx - Ftdotn * nx;
                    Fty = Fty - Ftdotn * ny;
                    Ftz = Ftz - Ftdotn * nz;
                    real Ft = sqrt(Ftx*Ftx + Fty*Fty + Ftz*Ftz);
                    real Fn = sqrt(Fnx*Fnx + Fny*Fny + Fnz*Fnz);
                    if(Ft > coeff_fric * Fn) {
                      Ftx = coeff_fric * Fn * Ftx / Ft;
                      Fty = coeff_fric * Fn * Fty / Ft;
                      Ftz = coeff_fric * Fn * Ftz / Ft;
                    }
                    Lox = -(ai+0.5*h)*((Fny+Fty)*nz-(Fnz+Ftz)*ny);
                    Loy =  (ai+0.5*h)*((Fnx+Ftx)*nz-(Fnz+Ftz)*nx);
                    Loz = -(ai+0.5*h)*((Fnx+Ftx)*ny-(Fny+Fty)*nx);
                  }

                  // assign forces
                  parts[i].iFx += Fnx + Ftx;
                  parts[i].iFy += Fny + Fty;
                  parts[i].iFz += Fnz + Ftz;
                  parts[i].iLx += Lox;
                  parts[i].iLy += Loy;
                  parts[i].iLz += Loz;
                }
              }
            }
          } // end for (l...
        } // end for (m...
      } // end for (n...
    } // end if (ib, jb, kb) are not ghost bins
  }
}

__global__ void move_parts_a(part_struct *parts, int nparts, real dt, 
  g_struct g, gradP_struct gradP, real rho_f)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index
  
  if (pp < nparts) {
    real vol = 4./3. * PI * parts[pp].r*parts[pp].r*parts[pp].r;
    real m = vol * parts[pp].rho;

    if(parts[pp].translating) {
      // update linear accelerations
      parts[pp].udot = (parts[pp].Fx + parts[pp].kFx + parts[pp].iFx
        + parts[pp].aFx - vol*gradP.x) / m
        + (parts[pp].rho - rho_f) / parts[pp].rho * g.x;
      parts[pp].vdot = (parts[pp].Fy + parts[pp].kFy + parts[pp].iFy
        + parts[pp].aFy - vol*gradP.y) / m
        + (parts[pp].rho - rho_f) / parts[pp].rho * g.y;
      parts[pp].wdot = (parts[pp].Fz + parts[pp].kFz + parts[pp].iFz
        + parts[pp].aFz - vol*gradP.z) / m
        + (parts[pp].rho - rho_f) / parts[pp].rho * g.z;

      // update linear velocities
      parts[pp].u = parts[pp].u0 + 0.5*dt*(parts[pp].udot + parts[pp].udot0);
      parts[pp].v = parts[pp].v0 + 0.5*dt*(parts[pp].vdot + parts[pp].vdot0);
      parts[pp].w = parts[pp].w0 + 0.5*dt*(parts[pp].wdot + parts[pp].wdot0);

      // do not update position
    }

    if(parts[pp].rotating) {
      // update angular accelerations
      real I = 0.4 * m * parts[pp].r*parts[pp].r;
      parts[pp].oxdot = (parts[pp].Lx + parts[pp].iLx + parts[pp].aLx) / I;
      parts[pp].oydot = (parts[pp].Ly + parts[pp].iLy + parts[pp].aLy) / I;
      parts[pp].ozdot = (parts[pp].Lz + parts[pp].iLz + parts[pp].aLz) / I;

      // update angular velocities
      parts[pp].ox = parts[pp].ox0 + 0.5*dt*(parts[pp].oxdot + parts[pp].oxdot0);
      parts[pp].oy = parts[pp].oy0 + 0.5*dt*(parts[pp].oydot + parts[pp].oydot0);
      parts[pp].oz = parts[pp].oz0 + 0.5*dt*(parts[pp].ozdot + parts[pp].ozdot0);
    }
  }
}

__global__ void move_parts_b(part_struct *parts, int nparts, real dt, 
  g_struct g, gradP_struct gradP, real rho_f)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle index
  
  if (pp < nparts) {
    if (parts[pp].translating) {
      // update position (trapezoidal rule)
      parts[pp].x += 0.5*dt*(parts[pp].u + parts[pp].u0);
      parts[pp].y += 0.5*dt*(parts[pp].v + parts[pp].v0);
      parts[pp].z += 0.5*dt*(parts[pp].w + parts[pp].w0);

      // store for next time step
      parts[pp].u0 = parts[pp].u;
      parts[pp].v0 = parts[pp].v;
      parts[pp].w0 = parts[pp].w;
      parts[pp].udot0 = parts[pp].udot;
      parts[pp].vdot0 = parts[pp].vdot;
      parts[pp].wdot0 = parts[pp].wdot;

    }

    if(parts[pp].rotating) {
      /* update basis vectors */
      // calculate rotation magnitude (trapezoidal rule)
      real mag = 0.5*sqrt(parts[pp].ox*parts[pp].ox + parts[pp].oy*parts[pp].oy
        + parts[pp].oz*parts[pp].oz);
      mag += 0.5*sqrt(parts[pp].ox0*parts[pp].ox0 + parts[pp].oy0*parts[pp].oy0
        + parts[pp].oz0*parts[pp].oz0);
      // calculate normalized rotation axis
      real X = 0;
      real Y = 0;
      real Z = 0;
      if(mag > 0) {
        X = 0.5 * (parts[pp].ox + parts[pp].ox0) / mag;
        Y = 0.5 * (parts[pp].oy + parts[pp].oy0) / mag;
        Z = 0.5 * (parts[pp].oz + parts[pp].oz0) / mag;
      }
      // calculate rotation quaternion
      real theta = mag * dt;
      real qr = cos(0.5*theta);
      real qi = X * sin(0.5*theta);
      real qj = Y * sin(0.5*theta);
      real qk = Z * sin(0.5*theta);
      // compute quaternion conjugation to apply rotation to basis vectors
      rotate(qr, qi, qj, qk, &parts[pp].axx, &parts[pp].axy, &parts[pp].axz);
      rotate(qr, qi, qj, qk, &parts[pp].ayx, &parts[pp].ayy, &parts[pp].ayz);
      rotate(qr, qi, qj, qk, &parts[pp].azx, &parts[pp].azy, &parts[pp].azz);

      // store for next time step
      parts[pp].ox0 = parts[pp].ox;
      parts[pp].oy0 = parts[pp].oy;
      parts[pp].oz0 = parts[pp].oz;
      parts[pp].oxdot0 = parts[pp].oxdot;
      parts[pp].oydot0 = parts[pp].oydot;
      parts[pp].ozdot0 = parts[pp].ozdot;
    }
  }
}


__device__ void rotate(real qr, real qi, real qj, real qk,
  real *pi, real *pj, real *pk)
{
  real Pr = *pi*qi + *pj*qj + *pk*qk;
  real Pi = *pi*qr - *pj*qk + *pk*qj;
  real Pj = *pi*qk + *pj*qr - *pk*qi;
  real Pk = -*pi*qj + *pj*qi + *pk*qr;

  *pi = qr*Pi + qi*Pr + qj*Pk - qk*Pj;
  *pj = qr*Pj - qi*Pk + qj*Pr + qk*Pi;
  *pk = qr*Pk + qi*Pj - qj*Pi + qk*Pr;
}

__global__ void pack_forces_e(real *force_send_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    cbin = GFX_LOC(_bins.Gcc._ie, tj, tk, s1b, s2b);
    c2b = tj + tk * _bins.Gcc.jnb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      force_send_e[0 + 9*dest] = parts[pp].kFx;
      force_send_e[1 + 9*dest] = parts[pp].kFy;
      force_send_e[2 + 9*dest] = parts[pp].kFz;
      force_send_e[3 + 9*dest] = parts[pp].iFx;
      force_send_e[4 + 9*dest] = parts[pp].iFy;
      force_send_e[5 + 9*dest] = parts[pp].iFz;
      force_send_e[6 + 9*dest] = parts[pp].iLx;
      force_send_e[7 + 9*dest] = parts[pp].iLy;
      force_send_e[8 + 9*dest] = parts[pp].iLz;
    }
  }
}

__global__ void pack_forces_w(real *force_send_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    cbin = GFX_LOC(_bins.Gcc._is, tj, tk, s1b, s2b);
    c2b = tj + tk * _bins.Gcc.jnb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {

      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      force_send_w[0 + 9*dest] = parts[pp].kFx;
      force_send_w[1 + 9*dest] = parts[pp].kFy;
      force_send_w[2 + 9*dest] = parts[pp].kFz;
      force_send_w[3 + 9*dest] = parts[pp].iFx;
      force_send_w[4 + 9*dest] = parts[pp].iFy;
      force_send_w[5 + 9*dest] = parts[pp].iFz;
      force_send_w[6 + 9*dest] = parts[pp].iLx;
      force_send_w[7 + 9*dest] = parts[pp].iLy;
      force_send_w[8 + 9*dest] = parts[pp].iLz;
    }
  }
}

__global__ void pack_forces_n(real *force_send_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    cbin = GFY_LOC(ti, _bins.Gcc._je, tk, s1b, s2b);
    c2b = tk + ti * _bins.Gcc.knb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      force_send_n[0 + 9*dest] = parts[pp].kFx;
      force_send_n[1 + 9*dest] = parts[pp].kFy;
      force_send_n[2 + 9*dest] = parts[pp].kFz;
      force_send_n[3 + 9*dest] = parts[pp].iFx;
      force_send_n[4 + 9*dest] = parts[pp].iFy;
      force_send_n[5 + 9*dest] = parts[pp].iFz;
      force_send_n[6 + 9*dest] = parts[pp].iLx;
      force_send_n[7 + 9*dest] = parts[pp].iLy;
      force_send_n[8 + 9*dest] = parts[pp].iLz;
    }
  }
}

__global__ void pack_forces_s(real *force_send_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    cbin = GFY_LOC(ti, _bins.Gcc._js, tk, s1b, s2b);
    c2b = tk + ti * _bins.Gcc.knb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      force_send_s[0 + 9*dest] = parts[pp].kFx;
      force_send_s[1 + 9*dest] = parts[pp].kFy;
      force_send_s[2 + 9*dest] = parts[pp].kFz;
      force_send_s[3 + 9*dest] = parts[pp].iFx;
      force_send_s[4 + 9*dest] = parts[pp].iFy;
      force_send_s[5 + 9*dest] = parts[pp].iFz;
      force_send_s[6 + 9*dest] = parts[pp].iLx;
      force_send_s[7 + 9*dest] = parts[pp].iLy;
      force_send_s[8 + 9*dest] = parts[pp].iLz;
    }
  }
}

__global__ void pack_forces_t(real *force_send_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    cbin = GFZ_LOC(ti, tj, _bins.Gcc._ke, s1b, s2b);
    c2b = ti + tj * _bins.Gcc.inb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      force_send_t[0 + 9*dest] = parts[pp].kFx;
      force_send_t[1 + 9*dest] = parts[pp].kFy;
      force_send_t[2 + 9*dest] = parts[pp].kFz;
      force_send_t[3 + 9*dest] = parts[pp].iFx;
      force_send_t[4 + 9*dest] = parts[pp].iFy;
      force_send_t[5 + 9*dest] = parts[pp].iFz;
      force_send_t[6 + 9*dest] = parts[pp].iLx;
      force_send_t[7 + 9*dest] = parts[pp].iLy;
      force_send_t[8 + 9*dest] = parts[pp].iLz;
    }
  }
}

__global__ void pack_forces_b(real *force_send_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    cbin = GFZ_LOC(ti, tj, _bins.Gcc._ks, s1b, s2b);
    c2b = ti + tj * _bins.Gcc.inb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      force_send_b[0 + 9*dest] = parts[pp].kFx;
      force_send_b[1 + 9*dest] = parts[pp].kFy;
      force_send_b[2 + 9*dest] = parts[pp].kFz;
      force_send_b[3 + 9*dest] = parts[pp].iFx;
      force_send_b[4 + 9*dest] = parts[pp].iFy;
      force_send_b[5 + 9*dest] = parts[pp].iFz;
      force_send_b[6 + 9*dest] = parts[pp].iLx;
      force_send_b[7 + 9*dest] = parts[pp].iLy;
      force_send_b[8 + 9*dest] = parts[pp].iLz;
    }
  }
}

__global__ void unpack_forces_e(real *force_recv_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    cbin = GFX_LOC(_bins.Gcc._ieb, tj, tk, s1b, s2b);
    c2b = tj + tk * _bins.Gcc.jnb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      parts[pp].kFx = force_recv_e[0 + 9*dest];
      parts[pp].kFy = force_recv_e[1 + 9*dest];
      parts[pp].kFz = force_recv_e[2 + 9*dest];
      parts[pp].iFx = force_recv_e[3 + 9*dest];
      parts[pp].iFy = force_recv_e[4 + 9*dest];
      parts[pp].iFz = force_recv_e[5 + 9*dest];
      parts[pp].iLx = force_recv_e[6 + 9*dest];
      parts[pp].iLy = force_recv_e[7 + 9*dest];
      parts[pp].iLz = force_recv_e[8 + 9*dest];
    }
  }
}

__global__ void unpack_forces_w(real *force_recv_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    cbin = GFX_LOC(_bins.Gcc._isb, tj, tk, s1b, s2b);
    c2b = tj + tk * _bins.Gcc.jnb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      parts[pp].kFx = force_recv_w[0 + 9*dest];
      parts[pp].kFy = force_recv_w[1 + 9*dest];
      parts[pp].kFz = force_recv_w[2 + 9*dest];
      parts[pp].iFx = force_recv_w[3 + 9*dest];
      parts[pp].iFy = force_recv_w[4 + 9*dest];
      parts[pp].iFz = force_recv_w[5 + 9*dest];
      parts[pp].iLx = force_recv_w[6 + 9*dest];
      parts[pp].iLy = force_recv_w[7 + 9*dest];
      parts[pp].iLz = force_recv_w[8 + 9*dest];
    }
  }
}

__global__ void unpack_forces_n(real *force_recv_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    cbin = GFY_LOC(ti, _bins.Gcc._jeb, tk, s1b, s2b);
    c2b = tk + ti * _bins.Gcc.knb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      parts[pp].kFx = force_recv_n[0 + 9*dest];
      parts[pp].kFy = force_recv_n[1 + 9*dest];
      parts[pp].kFz = force_recv_n[2 + 9*dest];
      parts[pp].iFx = force_recv_n[3 + 9*dest];
      parts[pp].iFy = force_recv_n[4 + 9*dest];
      parts[pp].iFz = force_recv_n[5 + 9*dest];
      parts[pp].iLx = force_recv_n[6 + 9*dest];
      parts[pp].iLy = force_recv_n[7 + 9*dest];
      parts[pp].iLz = force_recv_n[8 + 9*dest];
    }
  }
}

__global__ void unpack_forces_s(real *force_recv_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    cbin = GFY_LOC(ti, _bins.Gcc._jsb, tk, s1b, s2b);
    c2b = tk + ti * _bins.Gcc.knb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      parts[pp].kFx = force_recv_s[0 + 9*dest];
      parts[pp].kFy = force_recv_s[1 + 9*dest];
      parts[pp].kFz = force_recv_s[2 + 9*dest];
      parts[pp].iFx = force_recv_s[3 + 9*dest];
      parts[pp].iFy = force_recv_s[4 + 9*dest];
      parts[pp].iFz = force_recv_s[5 + 9*dest];
      parts[pp].iLx = force_recv_s[6 + 9*dest];
      parts[pp].iLy = force_recv_s[7 + 9*dest];
      parts[pp].iLz = force_recv_s[8 + 9*dest];
    }
  }
}

__global__ void unpack_forces_t(real *force_recv_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    cbin = GFZ_LOC(ti, tj, _bins.Gcc._keb, s1b, s2b);
    c2b = ti + tj * _bins.Gcc.inb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      parts[pp].kFx = force_recv_t[0 + 9*dest];
      parts[pp].kFy = force_recv_t[1 + 9*dest];
      parts[pp].kFz = force_recv_t[2 + 9*dest];
      parts[pp].iFx = force_recv_t[3 + 9*dest];
      parts[pp].iFy = force_recv_t[4 + 9*dest];
      parts[pp].iFz = force_recv_t[5 + 9*dest];
      parts[pp].iLx = force_recv_t[6 + 9*dest];
      parts[pp].iLy = force_recv_t[7 + 9*dest];
      parts[pp].iLz = force_recv_t[8 + 9*dest];
    }
  }
}

__global__ void unpack_forces_b(real *force_recv_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, part_struct *parts)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    cbin = GFZ_LOC(ti, tj, _bins.Gcc._ksb, s1b, s2b);
    c2b = ti + tj * _bins.Gcc.inb;

    // Loop through each bin's particles 
    // Each bin is offset by offset[cbin] (from excl. prefix scan)
    // Each particle is then offset from that
    for (int i = 0; i < bin_count[cbin]; i++) {
      pp = part_ind[bin_start[cbin] + i];
      dest = offset[c2b] + i;

      // 0: kFx  1: kFy  2: kFz
      // 3: iFx  4: iFy  5: iFz
      // 6: iLx  7: iLy  8: iLz
      parts[pp].kFx = force_recv_b[0 + 9*dest];
      parts[pp].kFy = force_recv_b[1 + 9*dest];
      parts[pp].kFz = force_recv_b[2 + 9*dest];
      parts[pp].iFx = force_recv_b[3 + 9*dest];
      parts[pp].iFy = force_recv_b[4 + 9*dest];
      parts[pp].iFz = force_recv_b[5 + 9*dest];
      parts[pp].iLx = force_recv_b[6 + 9*dest];
      parts[pp].iLy = force_recv_b[7 + 9*dest];
      parts[pp].iLz = force_recv_b[8 + 9*dest];
    }
  }
}
