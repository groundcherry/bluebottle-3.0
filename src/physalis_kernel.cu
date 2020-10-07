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

#include "cuda_physalis.h"

__device__ void rtp2xyz(real r, real theta, real phi, real *x, real *y, real *z)
{
  *x = r * sin(theta) * cos(phi);
  *y = r * sin(theta) * sin(phi);
  *z = r * cos(theta);
}

__device__ void cart2sphere(real u, real v, real w, real theta, real phi,
  real *ur, real *ut, real *up)
{
  real st = sin(theta);
  real ct = cos(theta);
  real sp = sin(phi);
  real cp = cos(phi);

  *ur = st * (u * cp + v * sp) + w * ct;
  *ut = ct * (u * cp + v * sp) - w * st;
  *up = -u * sp + v * cp;
}

__device__ real nnm(int n, int m)
{
  real fact_top = 1;
  real fact_bot = 1;

  for (int i = 1; i <= (n - m); i++) fact_top *= (real) i;
  for (int i = 1; i <= (n + m); i++) fact_bot *= (real) i;

  return sqrt((2.*n + 1.) / 4. / PI * fact_top / fact_bot);
}

__device__ real pnm(int n, int m, real theta)
{
  real x = cos(theta);
  real y = sin(theta);

  switch(n) {
    case 0: return 1;
    case 1:
      switch(m) {
        //case -1: return -0.5*y;
        case 0: return x;
        case 1: return -y;
      }
    case 2:
      switch(m) {
        //case -2: return 0.125*y*y;
        //case -1: return -0.5*x*y;
        case 0: return 0.5*(3.*x*x - 1.);
        case 1: return -3.*x*y;
        case 2: return 3.*y*y;
      }
    case 3:
      switch(m) {
        //case -3: return -0.02083333333333*y*y*y;
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
        //case -3: return -0.02083333333333*x*y*y*y*y;
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
        //case -5: return -0.000260416666667*y*y*y*y*y;
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

__global__ void check_nodes(int nparts, part_struct *parts, BC *bc,
  dom_struct *DOM)
{
  int node = threadIdx.x;
  int part = blockIdx.x;

  /* Convert node (r, theta, phi) to (x, y, z) */
  real xp, yp, zp;  // Cartesian radial vector
  real x, y, z;     // Cartesian location of node
  rtp2xyz(parts[part].rs, _node_t[node], _node_p[node], &xp, &yp, &zp);

  /* shift from particle center */
  x = xp + parts[part].x;
  y = yp + parts[part].y;
  z = zp + parts[part].z;

  // start off with all -1's
  parts[part].nodes[node] = -1;

  /* check if the node is interfered with by a wall */
  // compute distance between node and walls
  // set equal to some number to identify which wall is interfering

  // We use <= for E,N,T and > for W,S,B -- allows us to do [start,end) on all 
  // subdomains regardless of bc
  parts[part].nodes[node] += (WEST_WALL_D + 1) *    // set equal to WEST_WALL_D...
              (x - _dom.xs < 0) *                 // if outside domain &
              (_dom.I == DOM->Is) *                // if edge domain & DIRICHLET
              (bc->uW == DIRICHLET || bc->vW == DIRICHLET || bc->wW == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (WEST_WALL_N + 1) *
              (x - _dom.xs < 0) *
              (_dom.I == DOM->Is) *
              (bc->uW == NEUMANN || bc->vW == NEUMANN || bc->wW == NEUMANN)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (EAST_WALL_D + 1) *
              (x - _dom.xe >= 0) *
              (_dom.I == DOM->Ie) *
              (bc->uE == DIRICHLET || bc->vE == DIRICHLET || bc->wE == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (EAST_WALL_N + 1) *
              (x - _dom.xe >= 0) *
              (_dom.I == DOM->Ie) *
              (bc->uE == NEUMANN || bc->vE == NEUMANN || bc->wE == NEUMANN)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (SOUTH_WALL_D + 1) *
              (y - _dom.ys < 0) *
              (_dom.J == DOM->Js) *
              (bc->uS == DIRICHLET || bc->vS == DIRICHLET || bc->wS == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (SOUTH_WALL_N + 1) *
              (y - _dom.ys < 0) *
              (_dom.J == DOM->Js) *
              (bc->uS == NEUMANN || bc->vS == NEUMANN || bc->wS == NEUMANN)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (NORTH_WALL_D + 1) *
              (y - _dom.ye >= 0) *
              (_dom.J == DOM->Je) *
              (bc->uN == DIRICHLET || bc->vN == DIRICHLET || bc->wN == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (NORTH_WALL_N + 1) *
              (y - _dom.ye >= 0) *
              (_dom.J == DOM->Je) *
              (bc->uN == NEUMANN || bc->vN == NEUMANN || bc->wN == NEUMANN)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (BOTTOM_WALL_D + 1) *
              (z - _dom.zs < 0) *
              (_dom.K == DOM->Ks) *
              (bc->uB == DIRICHLET || bc->vB == DIRICHLET || bc->wB == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (BOTTOM_WALL_N + 1) *
              (z - _dom.zs < 0) *
              (_dom.K == DOM->Ks) *
              (bc->uB == NEUMANN || bc->vB == NEUMANN || bc->wB == NEUMANN)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (TOP_WALL_D + 1) *
              (z - _dom.ze >= 0) *
              (_dom.K == DOM->Ke) *
              (bc->uT == DIRICHLET || bc->vT == DIRICHLET || bc->wT == DIRICHLET)*
              (parts[part].nodes[node] == -1);

  parts[part].nodes[node] += (TOP_WALL_N + 1) *
              (z - _dom.ze >= 0) *
              (_dom.K == DOM->Ke) *
              (bc->uT == NEUMANN || bc->vT == NEUMANN || bc->wT == NEUMANN)*
              (parts[part].nodes[node] == -1);
}

__global__ void interpolate_nodes(real *p, real *u, real *v, real *w,
  real rho_f, real nu, gradP_struct gradP, part_struct *parts, real *pp,
  real *ur, real *ut, real *up, BC *bc, real s_beta, real s_ref, g_struct g)
{
  int node = threadIdx.x;
  int part = blockIdx.x;

  real ddx = 1. / _dom.dx;
  real ddy = 1. / _dom.dy;
  real ddz = 1. / _dom.dz;

  real irho_f = 1. / rho_f;
  real inu = 1. / nu;

  real ox = parts[part].ox;
  real oy = parts[part].oy;
  real oz = parts[part].oz;
  real oxdot = parts[part].oxdot;
  real oydot = parts[part].oydot;
  real ozdot = parts[part].ozdot;
  real udot = parts[part].udot;
  real vdot = parts[part].vdot;
  real wdot = parts[part].wdot;

  real rs2 = parts[part].rs * parts[part].rs;
  real rs3 = rs2 * parts[part].rs;
  real rs5 = rs3 * rs2;
  real irs3 = 1./rs3;
  real a5 = parts[part].r * parts[part].r;  // r^2
  a5 *= a5 * parts[part].r;                 // r^5

  real uu, vv, ww;  // temporary nodes for Cartesian result of interpolation
  real uuwalli, uuwallj, uuwallk;
  real vvwalli, vvwallj, vvwallk;
  real wwwalli, wwwallj, wwwallk;

  int i, j, k;  // index of cells containing node
  int oobi, oobj, oobk, oob;  // out of bounds indicator, 1 if out of bounds else 0
  int C, Ce, Cw, Cn, Cs, Ct, Cb;  // cell indices
  real xx, yy, zz;  // Cartesian location of p,u,v,w

  // convert node (r, theta, phi) to (x, y, z)
  real xp, yp, zp;  // Cartesian radial vector
  real x, y, z;     // Cartesian location of node
  rtp2xyz(parts[part].rs, _node_t[node], _node_p[node], &xp, &yp, &zp);

  // shift from particle center
  x = xp + parts[part].x;
  y = yp + parts[part].y;
  z = zp + parts[part].z;

  /* Find index of cell containing node. */
  // Do this in GLOBAL coordinates so that magnitude of floating point error is
  // the same on each subdomain.
  real arg_x = (x - (_dom.xs - _dom.dx)) * ddx + _dom.Gcc.isb;
  real arg_y = (y - (_dom.ys - _dom.dy)) * ddy + _dom.Gcc.jsb;
  real arg_z = (z - (_dom.zs - _dom.dz)) * ddz + _dom.Gcc.ksb;

  /* Deal with floating point errors in position so we don't lose nodes */
  // Similar to bin_fill_{i,j,k}. If floor != round and round is "close enough"
  // to the nearest integer, use round instead. this ensures that all nodes are
  // accounted for between subdomains
  // Using global indices makes sure that the floating point representation
  // error is the same for each subdomain, since the magnitude of the index will
  // be the similar/the same.

  i = floor(arg_x);
  j = floor(arg_y);
  k = floor(arg_z);

  int round_x = lrint(arg_x);
  int round_y = lrint(arg_y);
  int round_z = lrint(arg_z);

  // Better way to do this? no if-statement... abs?
  if ((round_x != i) && (abs(round_x - arg_x) <= DBL_EPSILON)) {
    i = round_x;
  }
  if ((round_y != j) && (abs(round_y - arg_y) <= DBL_EPSILON)) {
    j = round_y;
  }
  if ((round_z != k) && (abs(round_z - arg_z) <= DBL_EPSILON)) {
    k = round_z;
  }

  // Convert back to LOCAL coodrinates
  i -= _dom.Gcc.isb;
  j -= _dom.Gcc.jsb;
  k -= _dom.Gcc.ksb;

  /* Interpolate Pressure */
  // Find if out-of-bounds -- 1 if oob, 0 if in bounds
  oob = i < _dom.Gcc._is || i > _dom.Gcc._ie
     || j < _dom.Gcc._js || j > _dom.Gcc._je
     || k < _dom.Gcc._ks || k > _dom.Gcc._ke;

  // Correct indices so we don't have out-of-bounds reads
  // If out out bounds, we'll read good info but trash the results
  i += (_dom.Gcc._is - i) * (i < _dom.Gcc._is);
  j += (_dom.Gcc._js - j) * (j < _dom.Gcc._js);
  k += (_dom.Gcc._ks - k) * (k < _dom.Gcc._ks);
  i += (_dom.Gcc._ie - i) * (i > _dom.Gcc._ie);
  j += (_dom.Gcc._je - j) * (j > _dom.Gcc._je);
  k += (_dom.Gcc._ke - k) * (k > _dom.Gcc._ke);

  // Cell-centered indices
  C = GCC_LOC(i, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Ce = GCC_LOC(i + 1, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Cw = GCC_LOC(i - 1, j, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Cn = GCC_LOC(i, j + 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Cs = GCC_LOC(i, j - 1, k, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Ct = GCC_LOC(i, j, k + 1, _dom.Gcc.s1b, _dom.Gcc.s2b);
  Cb = GCC_LOC(i, j, k - 1, _dom.Gcc.s1b, _dom.Gcc.s2b);

  // Cartesian location of center of cell
  xx = (i - 0.5) * _dom.dx + _dom.xs;
  yy = (j - 0.5) * _dom.dy + _dom.ys;
  zz = (k - 0.5) * _dom.dz + _dom.zs;

  // perform tri-linear interpolation
  real dpdx = 0.5*(p[Ce] - p[Cw]) * ddx;
  real dpdy = 0.5*(p[Cn] - p[Cs]) * ddy;
  real dpdz = 0.5*(p[Ct] - p[Cb]) * ddz;
  pp[node + NNODES*part] = p[C] + dpdx*(x - xx) + dpdy*(y - yy) + dpdz*(z - zz);

  // set ppwall equal to
/*  ppwall = (parts[part].nodes[node] == WEST_WALL_D || parts[part].nodes[node] == WEST_WALL_N)*p[Cw]
            + (parts[part].nodes[node] == EAST_WALL_D || parts[part].nodes[node] == EAST_WALL_N)*p[Ce]
            + (parts[part].nodes[node] == SOUTH_WALL_D || parts[part].nodes[node] == SOUTH_WALL_N)*p[Cs]
            + (parts[part].nodes[node] == NORTH_WALL_D || parts[part].nodes[node] == NORTH_WALL_N)*p[Cn]
            + (parts[part].nodes[node] == BOTTOM_WALL_D || parts[part].nodes[node] == BOTTOM_WALL_N)*p[Cb]
            + (parts[part].nodes[node] == TOP_WALL_D || parts[part].nodes[node] == TOP_WALL_N)*p[Ct];
*/
  // switch to particle rest frame
  real ocrossr2 = (oy*zp - oz*yp) * (oy*zp - oz*yp);
  ocrossr2 += (ox*zp - oz*xp) * (ox*zp - oz*xp);
  ocrossr2 += (ox*yp - oy*xp) * (ox*yp - oy*xp);
  real bousiq_x = -s_beta*(parts[part].s - s_ref)*g.x;
  real bousiq_y = -s_beta*(parts[part].s - s_ref)*g.y;
  real bousiq_z = -s_beta*(parts[part].s - s_ref)*g.z;
  real accdotr = (-gradP.x * irho_f - udot + bousiq_x)*xp +
                 (-gradP.y * irho_f - vdot + bousiq_y)*yp +
                 (-gradP.z * irho_f - wdot + bousiq_z)*zp;
  pp[node + NNODES*part] -= 0.5 * rho_f * ocrossr2 + rho_f * accdotr;
//  ppwall -= 0.5 * rho_f * ocrossr2 + rho_f * accdotr;

  // Zero if this node intersects wall or is out of bounds
  pp[node + NNODES*part] = pp[node+part*NNODES] * (1 - oob) *
                              (parts[part].nodes[node] == -1);
//  pp[node + NNODES*part] = ppwall * oob * (parts[part].nodes[node] < -1) +
//     pp[node + NNODES*part] * (1 - oob) * (parts[part].nodes[node] == -1);

  /* Interpolate Velocities */
  // don't work with cell-center anymore; find closest cell face in x-direction

  /* Interpolate u-velocity */
  arg_x = (x - (_dom.xs - _dom.dx)) * ddx + _dom.Gfx.isb;
  arg_y = (y - (_dom.ys - _dom.dy)) * ddy + _dom.Gfx.jsb;
  arg_z = (z - (_dom.zs - _dom.dz)) * ddz + _dom.Gfx.ksb;

  i = floor(arg_x);
  j = floor(arg_y);
  k = floor(arg_z);

  round_x = lrint(arg_x);
  round_y = lrint(arg_y);
  round_z = lrint(arg_z);

  if ((round_x != i) && (abs(round_x - arg_x) <= DBL_EPSILON)) {
    i = round_x;
  }
  if ((round_y != j) && (abs(round_y - arg_y) <= DBL_EPSILON)) {
    j = round_y;
  }
  if ((round_z != k) && (abs(round_z - arg_z) <= DBL_EPSILON)) {
    k = round_z;
  }

  i -= _dom.Gfx.isb;
  j -= _dom.Gfx.jsb;
  k -= _dom.Gfx.ksb;
  //i = round((x - _dom.xs) * ddx) + DOM_BUF;
  //j = floor((y - _dom.ys) * ddy) + DOM_BUF;
  //k = floor((z - _dom.zs) * ddz) + DOM_BUF;

  // Find if out-of-bounds -- 1 if oob, 0 if in bounds
  // Use >= so domain is [start, end)
  oobi = i < _dom.Gcc._is || i > _dom.Gcc._ie;
  oobj = j < _dom.Gcc._js || j > _dom.Gcc._je;
  oobk = k < _dom.Gcc._ks || k > _dom.Gcc._ke;

  // Correct indices so we don't have out-of-bounds reads
  // If out out bounds, we'll read good info but trash the results
  i += (_dom.Gfx._is - i) * (i < _dom.Gfx._is);
  j += (_dom.Gfx._js - j) * (j < _dom.Gfx._js);
  k += (_dom.Gfx._ks - k) * (k < _dom.Gfx._ks);
  i += (_dom.Gfx._ie - i) * (i >= _dom.Gfx._ie);
  j += (_dom.Gfx._je - j) * (j > _dom.Gfx._je);
  k += (_dom.Gfx._ke - k) * (k > _dom.Gfx._ke);

  // Face-centered indices
  C = GFX_LOC(i, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b);
  Ce = GFX_LOC(i + 1, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b);
  Cw = GFX_LOC(i - 1, j, k, _dom.Gfx.s1b, _dom.Gfx.s2b);
  Cn = GFX_LOC(i, j + 1, k, _dom.Gfx.s1b, _dom.Gfx.s2b);
  Cs = GFX_LOC(i, j - 1, k, _dom.Gfx.s1b, _dom.Gfx.s2b);
  Ct = GFX_LOC(i, j, k + 1, _dom.Gfx.s1b, _dom.Gfx.s2b);
  Cb = GFX_LOC(i, j, k - 1, _dom.Gfx.s1b, _dom.Gfx.s2b);

  // Cartesian location of face
  xx = (i - DOM_BUF) * _dom.dx + _dom.xs;
  yy = (j - 0.5) * _dom.dy + _dom.ys;
  zz = (k - 0.5) * _dom.dz + _dom.zs;

  // Tri-linear interpolation
  real dudx = 0.5*(u[Ce] - u[Cw]) * ddx;
  real dudy = 0.5*(u[Cn] - u[Cs]) * ddy;
  real dudz = 0.5*(u[Ct] - u[Cb]) * ddz;

  uu = u[C] + dudx * (x - xx) + dudy * (y - yy) + dudz * (z - zz);

  // set uuwall equal to interfering wall u-velocity
  uuwalli = (parts[part].nodes[node] == WEST_WALL_D)*bc->uWD
          + (parts[part].nodes[node] == EAST_WALL_D)*bc->uED;
  uuwallj = (parts[part].nodes[node] == SOUTH_WALL_D)*bc->uSD
          + (parts[part].nodes[node] == NORTH_WALL_D)*bc->uND;
  uuwallk = (parts[part].nodes[node] == BOTTOM_WALL_D)*bc->uBD
          + (parts[part].nodes[node] == TOP_WALL_D)*bc->uTD;

  // switch to particle rest frame
  real ocrossr_x = oy*zp - oz*yp;
  real odotcrossr_x = oydot*zp - ozdot*yp;
  real tmp_u = parts[part].u + ocrossr_x +
                0.1 * inu * (rs5 - a5) * irs3 * odotcrossr_x; 
  uu -= tmp_u;
  uuwalli -= tmp_u;
  uuwallj -= tmp_u;
  uuwallk -= tmp_u;

  // set actual node value based on whether it is interfered with
  uu = (1-oobi) * (1-oobj) * (1-oobk) * (parts[part].nodes[node] == -1) * uu
    + oobi * (1-oobj) * (1-oobk) * (parts[part].nodes[node] < -1) * uuwalli
    + (1-oobi) * oobj * (1-oobk) * (parts[part].nodes[node] < -1) * uuwallj
    + (1-oobi) * (1-oobj) * oobk * (parts[part].nodes[node] < -1) * uuwallk;

  /* interpolate v-velocity */
  //i = floor((x - _dom.xs) * ddx) + DOM_BUF;
  //j = round((y - _dom.ys) * ddy) + DOM_BUF;
  //k = floor((z - _dom.zs) * ddz) + DOM_BUF;

  arg_x = (x - (_dom.xs - _dom.dx)) * ddx + _dom.Gfy.isb;
  arg_y = (y - (_dom.ys - _dom.dy)) * ddy + _dom.Gfy.jsb;
  arg_z = (z - (_dom.zs - _dom.dz)) * ddz + _dom.Gfy.ksb;

  i = floor(arg_x);
  j = floor(arg_y);
  k = floor(arg_z);

  round_x = lrint(arg_x);
  round_y = lrint(arg_y);
  round_z = lrint(arg_z);

  if ((round_x != i) && (abs(round_x - arg_x) <= DBL_EPSILON)) {
    i = round_x;
  }
  if ((round_y != j) && (abs(round_y - arg_y) <= DBL_EPSILON)) {
    j = round_y;
  }
  if ((round_z != k) && (abs(round_z - arg_z) <= DBL_EPSILON)) {
    k = round_z;
  }
  i -= _dom.Gfy.isb;
  j -= _dom.Gfy.jsb;
  k -= _dom.Gfy.ksb;

  // Find if out-of-bounds -- 1 if oob, 0 if in bounds
  oobi = i < _dom.Gcc._is || i > _dom.Gcc._ie;
  oobj = j < _dom.Gcc._js || j > _dom.Gcc._je;
  oobk = k < _dom.Gcc._ks || k > _dom.Gcc._ke;

  // Correct indices so we don't have out-of-bounds reads
  // If out out bounds, we'll read good info but trash the results
  i += (_dom.Gfy._is - i) * (i < _dom.Gfy._is);
  j += (_dom.Gfy._js - j) * (j < _dom.Gfy._js);
  k += (_dom.Gfy._ks - k) * (k < _dom.Gfy._ks);
  i += (_dom.Gfy._ie - i) * (i > _dom.Gfy._ie);
  j += (_dom.Gfy._je - j) * (j >= _dom.Gfy._je);
  k += (_dom.Gfy._ke - k) * (k > _dom.Gfy._ke);

  // Face-centered indices
  C = GFY_LOC(i, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b);
  Ce = GFY_LOC(i + 1, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b);
  Cw = GFY_LOC(i - 1, j, k, _dom.Gfy.s1b, _dom.Gfy.s2b);
  Cn = GFY_LOC(i, j + 1, k, _dom.Gfy.s1b, _dom.Gfy.s2b);
  Cs = GFY_LOC(i, j - 1, k, _dom.Gfy.s1b, _dom.Gfy.s2b);
  Ct = GFY_LOC(i, j, k + 1, _dom.Gfy.s1b, _dom.Gfy.s2b);
  Cb = GFY_LOC(i, j, k - 1, _dom.Gfy.s1b, _dom.Gfy.s2b);

  // Cartesian location of face
  xx = (i-0.5) * _dom.dx + _dom.xs;
  yy = (j-DOM_BUF) * _dom.dy + _dom.ys;
  zz = (k-0.5) * _dom.dz + _dom.zs;

  // Tri-linear interpolation
  real dvdx = 0.5*(v[Ce] - v[Cw]) * ddx;
  real dvdy = 0.5*(v[Cn] - v[Cs]) * ddy;
  real dvdz = 0.5*(v[Ct] - v[Cb]) * ddz;
  vv = v[C] + dvdx * (x - xx) + dvdy * (y - yy) + dvdz * (z - zz);

  // set vvwall equal to interfering wall v-velocity
  vvwalli = (parts[part].nodes[node] == WEST_WALL_D)*bc->vWD
          + (parts[part].nodes[node] == EAST_WALL_D)*bc->vED;
  vvwallj = (parts[part].nodes[node] == SOUTH_WALL_D)*bc->vSD
          + (parts[part].nodes[node] == NORTH_WALL_D)*bc->vND;
  vvwallk = (parts[part].nodes[node] == BOTTOM_WALL_D)*bc->vBD
          + (parts[part].nodes[node] == TOP_WALL_D)*bc->vTD;

  // switch to particle rest frame
  real ocrossr_y = -(ox*zp - oz*xp);
  real odotcrossr_y = -(oxdot*zp - ozdot*xp);
  real tmp_v = parts[part].v + ocrossr_y +
                0.1 * inu * (rs5 - a5) * irs3 * odotcrossr_y;

  vv -= tmp_v;
  vvwalli -= tmp_v;
  vvwallj -= tmp_v;
  vvwallk -= tmp_v;

  // set actual node value based on whether it is interfered with
  vv = (1-oobi) * (1-oobj) * (1-oobk) * (parts[part].nodes[node] == -1) * vv
    + oobi * (1-oobj) * (1-oobk) * (parts[part].nodes[node] < -1) * vvwalli
    + (1-oobi) * oobj * (1-oobk) * (parts[part].nodes[node] < -1) * vvwallj
    + (1-oobi) * (1-oobj) * oobk * (parts[part].nodes[node] < -1) * vvwallk;

  /* interpolate w-velocity */
  arg_x = (x - (_dom.xs - _dom.dx)) * ddx + _dom.Gfz.isb;
  arg_y = (y - (_dom.ys - _dom.dy)) * ddy + _dom.Gfz.jsb;
  arg_z = (z - (_dom.zs - _dom.dz)) * ddz + _dom.Gfz.ksb;

  i = floor(arg_x);
  j = floor(arg_y);
  k = floor(arg_z);

  round_x = lrint(arg_x);
  round_y = lrint(arg_y);
  round_z = lrint(arg_z);

  if ((round_x != i) && (abs(round_x - arg_x) <= DBL_EPSILON)) {
    i = round_x;
  }
  if ((round_y != j) && (abs(round_y - arg_y) <= DBL_EPSILON)) {
    j = round_y;
  }
  if ((round_z != k) && (abs(round_z - arg_z) <= DBL_EPSILON)) {
    k = round_z;
  }
  i -= _dom.Gfz.isb;
  j -= _dom.Gfz.jsb;
  k -= _dom.Gfz.ksb;
  //i = floor((x - _dom.xs) * ddx) + DOM_BUF;
  //j = floor((y - _dom.ys) * ddy) + DOM_BUF;
  //k = round((z - _dom.zs) * ddz) + DOM_BUF;

  // Find if out-of-bounds -- 1 if oob, 0 if in bounds
  oobi = i < _dom.Gcc._is || i > _dom.Gcc._ie;
  oobj = j < _dom.Gcc._js || j > _dom.Gcc._je;
  oobk = k < _dom.Gcc._ks || k > _dom.Gcc._ke;

  // Correct indices so we don't have out-of-bounds reads
  // If out out bounds, we'll read good info but trash the results
  i += (_dom.Gfz._is - i) * (i < _dom.Gfz._is);
  j += (_dom.Gfz._js - j) * (j < _dom.Gfz._js);
  k += (_dom.Gfz._ks - k) * (k < _dom.Gfz._ks);
  i += (_dom.Gfz._ie - i) * (i > _dom.Gfz._ie);
  j += (_dom.Gfz._je - j) * (j > _dom.Gfz._je);
  k += (_dom.Gfz._ke - k) * (k >= _dom.Gfz._ke);

  // Face-centered indices
  C = GFZ_LOC(i, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
  Ce = GFZ_LOC(i + 1, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
  Cw = GFZ_LOC(i - 1, j, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
  Cn = GFZ_LOC(i, j + 1, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
  Cs = GFZ_LOC(i, j - 1, k, _dom.Gfz.s1b, _dom.Gfz.s2b);
  Ct = GFZ_LOC(i, j, k + 1, _dom.Gfz.s1b, _dom.Gfz.s2b);
  Cb = GFZ_LOC(i, j, k - 1, _dom.Gfz.s1b, _dom.Gfz.s2b);

  // Cartesian location of face
  xx = (i-0.5) * _dom.dx + _dom.xs;
  yy = (j-0.5) * _dom.dy + _dom.ys;
  zz = (k-DOM_BUF) * _dom.dz + _dom.zs;

  // Tri-linear interpolation
  real dwdx = 0.5*(w[Ce] - w[Cw]) * ddx;
  real dwdy = 0.5*(w[Cn] - w[Cs]) * ddy;
  real dwdz = 0.5*(w[Ct] - w[Cb]) * ddz;
  ww = w[C] + dwdx * (x - xx) + dwdy * (y - yy) + dwdz * (z - zz);

  // set uuwall equal to interfering wall u-velocity
  wwwalli = (parts[part].nodes[node] == WEST_WALL_D)*bc->wWD
          + (parts[part].nodes[node] == EAST_WALL_D)*bc->wED;
  wwwallj = (parts[part].nodes[node] == SOUTH_WALL_D)*bc->wSD
          + (parts[part].nodes[node] == NORTH_WALL_D)*bc->wND;
  wwwallk = (parts[part].nodes[node] == BOTTOM_WALL_D)*bc->wBD
          + (parts[part].nodes[node] == TOP_WALL_D)*bc->wTD;

  // switch to particle rest frame
  real ocrossr_z = ox*yp - oy*xp;
  real odotcrossr_z = oxdot*yp - oydot*xp;
  real tmp_w = parts[part].w + ocrossr_z + 
                 0.1 * inu * (rs5 - a5) * irs3 * odotcrossr_z;
  ww -= tmp_w;
  wwwalli -= tmp_w;
  wwwallj -= tmp_w;
  wwwallk -= tmp_w;

  // set actual node value based on whether it is interfered with
  ww = (1-oobi) * (1-oobj) * (1-oobk) * (parts[part].nodes[node] == -1) * ww
    + oobi * (1-oobj) * (1-oobk) * (parts[part].nodes[node] < -1) * wwwalli
    + (1-oobi) * oobj * (1-oobk) * (parts[part].nodes[node] < -1) * wwwallj
    + (1-oobi) * (1-oobj) * oobk * (parts[part].nodes[node] < -1) * wwwallk;

  // convert (uu, vv, ww) to (u_r, u_theta, u_phi) and write to node arrays
  cart2sphere(uu, vv, ww, _node_t[node], _node_p[node],
    &ur[node+part*NNODES], &ut[node+part*NNODES], &up[node+part*NNODES]);
}

__global__ void lebedev_quadrature(part_struct *parts, int ncoeffs_max,
  real *pp, real *ur, real *ut, real *up,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int node = threadIdx.x;
  int part = blockIdx.x;
  int coeff = blockIdx.y;

  if (coeff < parts[part].ncoeff) {
    /* Calculate integrand at each node */
    int j = part*NNODES*ncoeffs_max + coeff*NNODES + node;

    int n = _nn[coeff];
    int m = _mm[coeff];
    real theta = _node_t[node];
    real phi = _node_p[node];
    real N_nm = nnm(n, m);
    real P_nm = pnm(n, m, theta);
    real P_n1m = pnm(n + 1., m, theta);
    real dPdt = (n - m + 1.)*P_n1m - (n + 1.)*cos(theta)*P_nm;
    real dPdp = m*P_nm;

    // Precalculate things we use more than once
    real isth = 1./sin(theta); 
    real cmphi = cos(m * phi);
    real smphi = sin(m * phi);

    int stride = node + part*NNODES;

    int_Yp_re[j] = N_nm*P_nm*pp[stride]*cmphi;
    int_Yp_im[j] = -N_nm*P_nm*pp[stride]*smphi;

    int_rDYu_re[j] = N_nm*isth*(dPdt * ut[stride] * cmphi 
                              - dPdp * up[stride] * smphi);
    int_rDYu_im[j] = N_nm*isth*(-dPdt * ut[stride] * smphi
                               - dPdp * up[stride] * cmphi);

    int_xXDYu_re[j] = N_nm*isth*(dPdp * ut[stride] * smphi
                               + dPdt * up[stride] * cmphi);
    int_xXDYu_im[j] = N_nm*isth*(dPdp * ut[stride] * cmphi
                               - dPdt * up[stride] * smphi);
    __syncthreads();

    /* Compute partial sum of Lebedev quadrature (scalar product) */
    // put sum into first node position for each coeff for each particle
    if (node == 0) {
      int_Yp_re[j] *= _A1;
      int_Yp_im[j] *= _A1;
      int_rDYu_re[j] *= _A1;
      int_rDYu_im[j] *= _A1;
      int_xXDYu_re[j] *= _A1;
      int_xXDYu_im[j] *= _A1;
      for (int i = 1; i < 6; i++) {
        int_Yp_re[j] += _A1 * int_Yp_re[j+i];
        int_Yp_im[j] += _A1 * int_Yp_im[j+i];
        int_rDYu_re[j] += _A1 * int_rDYu_re[j+i];
        int_rDYu_im[j] += _A1 * int_rDYu_im[j+i];
        int_xXDYu_re[j] += _A1 * int_xXDYu_re[j+i];
        int_xXDYu_im[j] += _A1 * int_xXDYu_im[j+i];
      }
      for (int i = 6; i < 18; i++) {
        int_Yp_re[j] += _A2 * int_Yp_re[j+i];
        int_Yp_im[j] += _A2 * int_Yp_im[j+i];
        int_rDYu_re[j] += _A2 * int_rDYu_re[j+i];
        int_rDYu_im[j] += _A2 * int_rDYu_im[j+i];
        int_xXDYu_re[j] += _A2 * int_xXDYu_re[j+i];
        int_xXDYu_im[j] += _A2 * int_xXDYu_im[j+i];
      }
      for (int i = 18; i < 26; i++) {
        int_Yp_re[j] += _A3 * int_Yp_re[j+i];
        int_Yp_im[j] += _A3 * int_Yp_im[j+i];
        int_rDYu_re[j] += _A3 * int_rDYu_re[j+i];
        int_rDYu_im[j] += _A3 * int_rDYu_im[j+i];
        int_xXDYu_re[j] += _A3 * int_xXDYu_re[j+i];
        int_xXDYu_im[j] += _A3 * int_xXDYu_im[j+i];
      }
    } // if (node == 0)
  }
}

__global__ void compute_lambs_coeffs(part_struct *parts, real relax,
  real mu, real nu, int ncoeffs_max, int nparts,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{

  int coeff = threadIdx.x;
  int part = blockIdx.x;

  // precalculate constants
  real ars = parts[part].r / parts[part].rs;
  real rsa = parts[part].rs / parts[part].r;
  real r2 = parts[part].r * parts[part].r;
  real inu = 1./nu;
  real imunu = inu / mu;

  if (coeff < parts[part].ncoeff && part < nparts) {
    int j = part * NNODES * ncoeffs_max + coeff * NNODES + 0;
    int n = _nn[coeff];

    if (n == 0) {
      parts[part].pnm_re[coeff] = (1. - relax) * parts[part].pnm_re0[coeff] + 
                                    relax * r2 * imunu * int_Yp_re[j];
      parts[part].pnm_im[coeff] = (1. - relax) * parts[part].pnm_im0[coeff] + 
                                    relax * r2 * imunu * int_Yp_im[j];
      parts[part].phinm_re[coeff] = 0.;
      parts[part].phinm_im[coeff] = 0.;
      parts[part].chinm_re[coeff] = 0.;
      parts[part].chinm_im[coeff] = 0.;
    } else { // n != 0
      // Precalculate 
      real pow_ars_np1 = pow(ars, n + 1.);                  // ars^(n+1)
      real pow_ars_2np1 = pow_ars_np1 * pow_ars_np1 * rsa;  // ars^(2n+1)
      real pow_rsa_nm1 = pow(rsa, n - 1.);                  // rsa^(n-1)
      real pow_rsa_n = pow_rsa_nm1 * rsa;                   // rsa^n
      real pow_rsa_np1 = pow_rsa_n * rsa;                   // rsa^(n+1)
      real i_np1 = 1./(n + 1.);
      real i_2np3 = 1./(2.*n + 3.);

      // calculate p_nm and phi_nm
      real A = (1. - 0.5*n*(2.*n - 1.) * i_np1 * pow_ars_2np1) * pow_rsa_n;
      real B = n*(2.*n - 1.)*(2.*n + 1.) * i_np1*pow_ars_np1;
      real C = 0.25*n*(2.*(n + 3.)*i_2np3
        + (n - 2. - n*(2.*n + 1.)*i_2np3*ars*ars)*pow_ars_2np1)*pow_rsa_np1;
      real D = n*(n + 1. + 0.5*((n - 2.)*(2.*n + 1.)*rsa*rsa
        - n*(2.*n - 1.))*pow_ars_2np1)*pow_rsa_nm1;

      real idet = 1./ (A*D + B*C);

      parts[part].pnm_re[coeff] = (r2*imunu*int_Yp_re[j]*D +
                                    parts[part].r*inu*int_rDYu_re[j]*B) * idet;
      parts[part].pnm_im[coeff] = (r2*imunu*int_Yp_im[j]*D +
                                    parts[part].r*inu*int_rDYu_im[j]*B) * idet;

      parts[part].phinm_re[coeff] = (parts[part].r*inu*int_rDYu_re[j]*A -
                                       r2*imunu*int_Yp_re[j]*C) * idet;
      parts[part].phinm_im[coeff] = (parts[part].r*inu*int_rDYu_im[j]*A -
                                       r2*imunu*int_Yp_im[j]*C) * idet;

      // calculate chi_nm
      real E = n*(n + 1.)*(pow_ars_2np1 - 1.)*pow_rsa_n;
      real iE = 1./ E;
      parts[part].chinm_re[coeff] = parts[part].r*inu*int_xXDYu_re[j] * iE;
      parts[part].chinm_im[coeff] = parts[part].r*inu*int_xXDYu_im[j] * iE;

      // apply underrelaxation
      parts[part].pnm_re[coeff] = parts[part].pnm_re0[coeff]*(1. - relax)
        + relax*parts[part].pnm_re[coeff];
      parts[part].pnm_im[coeff] = parts[part].pnm_im0[coeff]*(1. - relax)
        + relax*parts[part].pnm_im[coeff];
      parts[part].phinm_re[coeff] = parts[part].phinm_re0[coeff]*(1. - relax)
        + relax*parts[part].phinm_re[coeff];
      parts[part].phinm_im[coeff] = parts[part].phinm_im0[coeff]*(1. - relax)
        + relax*parts[part].phinm_im[coeff];
      parts[part].chinm_re[coeff] = parts[part].chinm_re0[coeff]*(1. - relax)
        + relax*parts[part].chinm_re[coeff];
      parts[part].chinm_im[coeff] = parts[part].chinm_im0[coeff]*(1. - relax)
        + relax*parts[part].chinm_im[coeff];
    }
  }
}

__global__ void calc_forces(part_struct *parts, int nparts,
  real gradPx, real gradPy, real gradPz, real rho_f, real mu, real nu,
  real s_beta, real s_ref, g_struct g)
{
  int pp = threadIdx.x + blockIdx.x*blockDim.x; // particle number

  real irho_f = 1./ rho_f;

  if(pp < nparts) {
    real vol = 4./3. * PI *  parts[pp].r*parts[pp].r*parts[pp].r;
    real N10 = sqrt(3./4./PI);
    real N11 = sqrt(3./8./PI);

    real bousiq_x = -s_beta*(parts[pp].s - s_ref)*g.x;
    real bousiq_y = -s_beta*(parts[pp].s - s_ref)*g.y;
    real bousiq_z = -s_beta*(parts[pp].s - s_ref)*g.z;

    parts[pp].Fx = rho_f * vol * (parts[pp].udot + gradPx * irho_f - bousiq_x)
      - PI * mu * nu * 2.*N11 * (parts[pp].pnm_re[2]
      + 6.*parts[pp].phinm_re[2]);
    parts[pp].Fy = rho_f * vol * (parts[pp].vdot + gradPy * irho_f - bousiq_y)
      + PI * mu * nu * 2.*N11 * (parts[pp].pnm_im[2]
      + 6.*parts[pp].phinm_im[2]);
    parts[pp].Fz = rho_f * vol * (parts[pp].wdot + gradPz * irho_f - bousiq_z)
      + PI * mu * nu * N10 * (parts[pp].pnm_re[1]
      + 6.*parts[pp].phinm_re[1]);

    parts[pp].Lx = rho_f * vol * parts[pp].r*parts[pp].r * parts[pp].oxdot
      - 8. * PI * mu * nu * 2.*N11 * parts[pp].r * parts[pp].chinm_re[2];
    parts[pp].Ly = rho_f * vol * parts[pp].r*parts[pp].r * parts[pp].oydot
      + 8. * PI * mu * nu * 2.*N11 * parts[pp].r * parts[pp].chinm_im[2];
    parts[pp].Lz = rho_f * vol * parts[pp].r*parts[pp].r * parts[pp].ozdot
      + 8. * PI * mu * nu * N10 * parts[pp].r * parts[pp].chinm_re[1];
  }
}

__global__ void pack_sums_e(real *sum_send_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (int ti = _bins.Gcc._ie; ti <= _bins.Gcc._ieb; ti++) {
      cbin = GFX_LOC(ti, tj, tk, s1b, s2b);
      c2b = tj + tk * s1b + (ti - _bins.Gcc._ie) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          sum_send_e[sp0] = int_Yp_re[psum_ind];
          sum_send_e[sp1] = int_Yp_im[psum_ind];
          sum_send_e[sp2] = int_rDYu_re[psum_ind];
          sum_send_e[sp3] = int_rDYu_im[psum_ind];
          sum_send_e[sp4] = int_xXDYu_re[psum_ind];
          sum_send_e[sp5] = int_xXDYu_im[psum_ind];
        }
      }
    } // loop over ti planes
  }
}

__global__ void pack_sums_w(real *sum_send_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (int ti = _bins.Gcc._isb; ti <= _bins.Gcc._is; ti++) {
      cbin = GFX_LOC(ti, tj, tk, s1b, s2b);
      c2b = tj + tk * s1b + (ti - _bins.Gcc._isb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          sum_send_w[sp0] = int_Yp_re[psum_ind];
          sum_send_w[sp1] = int_Yp_im[psum_ind];
          sum_send_w[sp2] = int_rDYu_re[psum_ind];
          sum_send_w[sp3] = int_rDYu_im[psum_ind];
          sum_send_w[sp4] = int_xXDYu_re[psum_ind];
          sum_send_w[sp5] = int_xXDYu_im[psum_ind];
        }
      }
    } // loop over ti
  }
}

__global__ void pack_sums_n(real *sum_send_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (int tj = _bins.Gcc._je; tj <= _bins.Gcc._jeb; tj++) {
      cbin = GFY_LOC(ti, tj, tk, s1b, s2b);
      c2b = tk + ti * s1b + (tj - _bins.Gcc._je) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          sum_send_n[sp0] = int_Yp_re[psum_ind];

//printf("N%d >> packing int_Yp_re[part %d, coeff %d (%d)] = %lf to sum_send_n[%d]\n",
//      _dom.rank, pp, coeff, psum_ind, int_Yp_re[psum_ind], sp0);

          sum_send_n[sp1] = int_Yp_im[psum_ind];
          sum_send_n[sp2] = int_rDYu_re[psum_ind];
          sum_send_n[sp3] = int_rDYu_im[psum_ind];
          sum_send_n[sp4] = int_xXDYu_re[psum_ind];
          sum_send_n[sp5] = int_xXDYu_im[psum_ind];
        }
      }
    } // loop over tj planes
  }
}

__global__ void pack_sums_s(real *sum_send_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (int tj = _bins.Gcc._jsb; tj <= _bins.Gcc._js; tj++) {
      cbin = GFY_LOC(ti, tj, tk, s1b, s2b);
      c2b = tk + ti * s1b + (tj - _bins.Gcc._jsb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          sum_send_s[sp0] = int_Yp_re[psum_ind];
          sum_send_s[sp1] = int_Yp_im[psum_ind];
          sum_send_s[sp2] = int_rDYu_re[psum_ind];
          sum_send_s[sp3] = int_rDYu_im[psum_ind];
          sum_send_s[sp4] = int_xXDYu_re[psum_ind];
          sum_send_s[sp5] = int_xXDYu_im[psum_ind];
        }
      }
    } // loop over tj planes
  }
}

__global__ void pack_sums_t(real *sum_send_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (int tk = _bins.Gcc._ke; tk <= _bins.Gcc._keb; tk++) {
      cbin = GFZ_LOC(ti, tj, tk, s1b, s2b);
      c2b = ti + tj * s1b + (tk - _bins.Gcc._ke) * s2b;

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          sum_send_t[sp0] = int_Yp_re[psum_ind];
          sum_send_t[sp1] = int_Yp_im[psum_ind];
          sum_send_t[sp2] = int_rDYu_re[psum_ind];
          sum_send_t[sp3] = int_rDYu_im[psum_ind];
          sum_send_t[sp4] = int_xXDYu_re[psum_ind];
          sum_send_t[sp5] = int_xXDYu_im[psum_ind];
        }
      }
    } // loop over tk planes
  }
}

__global__ void pack_sums_b(real *sum_send_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (int tk = _bins.Gcc._ksb; tk <= _bins.Gcc._ks; tk++) {
      cbin = GFZ_LOC(ti, tj, tk, s1b, s2b);
      c2b = ti + tj * s1b + (tk - _bins.Gcc._ksb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          sum_send_b[sp0] = int_Yp_re[psum_ind];
          sum_send_b[sp1] = int_Yp_im[psum_ind];
          sum_send_b[sp2] = int_rDYu_re[psum_ind];
          sum_send_b[sp3] = int_rDYu_im[psum_ind];
          sum_send_b[sp4] = int_xXDYu_re[psum_ind];
          sum_send_b[sp5] = int_xXDYu_im[psum_ind];
        }
      }
    } // loop over tk planes
  }
}

__global__ void unpack_sums_e(real *sum_recv_e, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (int ti = _bins.Gcc._ie; ti <= _bins.Gcc._ieb; ti++) {
      cbin = GFX_LOC(ti, tj, tk, s1b, s2b);
      c2b = tj + tk * s1b + (ti - _bins.Gcc._ie) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          int_Yp_re[psum_ind] += sum_recv_e[sp0];
          int_Yp_im[psum_ind] += sum_recv_e[sp1];
          int_rDYu_re[psum_ind] += sum_recv_e[sp2];
          int_rDYu_im[psum_ind] += sum_recv_e[sp3];
          int_xXDYu_re[psum_ind] += sum_recv_e[sp4];
          int_xXDYu_im[psum_ind] += sum_recv_e[sp5];
        }
      }
    } // loop over ti
  }
}

__global__ void unpack_sums_w(real *sum_recv_w, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFX indices
  int s1b = _bins.Gcc.jnb;
  int s2b = s1b * _bins.Gcc.knb;

  if (tj < _bins.Gcc.jnb && tk < _bins.Gcc.knb) {
    for (int ti = _bins.Gcc._isb; ti <= _bins.Gcc._is; ti++) {
      cbin = GFX_LOC(ti, tj, tk, s1b, s2b);
      c2b = tj + tk * s1b + (ti - _bins.Gcc._isb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          int_Yp_re[psum_ind] += sum_recv_w[sp0];
          int_Yp_im[psum_ind] += sum_recv_w[sp1];
          int_rDYu_re[psum_ind] += sum_recv_w[sp2];
          int_rDYu_im[psum_ind] += sum_recv_w[sp3];
          int_xXDYu_re[psum_ind] += sum_recv_w[sp4];
          int_xXDYu_im[psum_ind] += sum_recv_w[sp5];
        }
      }
    } // loop over ti
  }
}

__global__ void unpack_sums_n(real *sum_recv_n, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (int tj = _bins.Gcc._je; tj <= _bins.Gcc._jeb; tj++) {
      cbin = GFY_LOC(ti, tj, tk, s1b, s2b);
      c2b = tk + ti * s1b + (tj - _bins.Gcc._je) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          int_Yp_re[psum_ind] += sum_recv_n[sp0];
          int_Yp_im[psum_ind] += sum_recv_n[sp1];
          int_rDYu_re[psum_ind] += sum_recv_n[sp2];
          int_rDYu_im[psum_ind] += sum_recv_n[sp3];
          int_xXDYu_re[psum_ind] += sum_recv_n[sp4];
          int_xXDYu_im[psum_ind] += sum_recv_n[sp5];
        }
      }
    } // loop over tj
  }
}

__global__ void unpack_sums_s(real *sum_recv_s, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFY indices
  int s1b = _bins.Gcc.knb;
  int s2b = s1b * _bins.Gcc.inb;

  if (tk < _bins.Gcc.knb && ti < _bins.Gcc.inb) {
    for (int tj = _bins.Gcc._jsb; tj <= _bins.Gcc._js; tj++) {
      cbin = GFY_LOC(ti, tj, tk, s1b, s2b);
      c2b = tk + ti * s1b + (tj - _bins.Gcc._jsb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          int_Yp_re[psum_ind] += sum_recv_s[sp0];

//printf("N%d >> unpacking int_Yp_re[part %d, coeff %d (%d)] = %lf from sum_send_s[%d]\n",
//      _dom.rank, pp, coeff, psum_ind, int_Yp_re[psum_ind], sp0);

          int_Yp_im[psum_ind] += sum_recv_s[sp1];
          int_rDYu_re[psum_ind] += sum_recv_s[sp2];
          int_rDYu_im[psum_ind] += sum_recv_s[sp3];
          int_xXDYu_re[psum_ind] += sum_recv_s[sp4];
          int_xXDYu_im[psum_ind] += sum_recv_s[sp5];
        }
      }
    } // loop over tj
  }
}

__global__ void unpack_sums_t(real *sum_recv_t, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (int tk = _bins.Gcc._ke; tk <= _bins.Gcc._keb; tk++) {
      cbin = GFZ_LOC(ti, tj, tk, s1b, s2b);
      c2b = ti + tj * s1b + (tk - _bins.Gcc._ke) * s2b;

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          int_Yp_re[psum_ind] += sum_recv_t[sp0];
          int_Yp_im[psum_ind] += sum_recv_t[sp1];
          int_rDYu_re[psum_ind] += sum_recv_t[sp2];
          int_rDYu_im[psum_ind] += sum_recv_t[sp3];
          int_xXDYu_re[psum_ind] += sum_recv_t[sp4];
          int_xXDYu_im[psum_ind] += sum_recv_t[sp5];
        }
      }
    } // loop over tk
  }
}

__global__ void unpack_sums_b(real *sum_recv_b, int *offset, int *bin_start,
  int *bin_count, int *part_ind, int ncoeffs_max,
  real *int_Yp_re, real *int_Yp_im,
  real *int_rDYu_re, real *int_rDYu_im,
  real *int_xXDYu_re, real *int_xXDYu_im)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x; // bin index 
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  int cbin;       // bin index
  int c2b;        // bin index in 2-d plane
  int pp;         // particle index
  int dest;       // destination for particle partial sums in packed array
  int sp0, sp1;   // scalar product strides for (Ylm, p)
  int sp2, sp3;   // scalar product strides for (rDYlm, u)
  int sp4, sp5;   // scalar product strides for (x X DYlm, u)
  int psum_ind;   // index of partial sum in each scalar product

  // Custom GFZ indices
  int s1b = _bins.Gcc.inb;
  int s2b = s1b * _bins.Gcc.jnb;

  if (ti < _bins.Gcc.inb && tj < _bins.Gcc.jnb) {
    for (int tk = _bins.Gcc._ksb; tk <= _bins.Gcc._ks; tk++) {
      cbin = GFZ_LOC(ti, tj, tk, s1b, s2b);
      c2b = ti + tj * s1b + (tk - _bins.Gcc._ksb) * s2b; // two planes

      // Loop through each bin's particles 
      // Each bin is offset by offset[cbin] (from excl. prefix scan)
      // Each particle is then offset from that
      for (int i = 0; i < bin_count[cbin]; i++) {
        pp = part_ind[bin_start[cbin] + i];
        dest = offset[c2b] + i;

        for (int coeff = 0; coeff < ncoeffs_max; coeff++) {
          // Packing: part varies slowest, coeff varies quickest, sp middle
          sp0 = coeff + ncoeffs_max*SP_YP_RE + ncoeffs_max*NSP*dest;    // Yp_re
          sp1 = coeff + ncoeffs_max*SP_YP_IM + ncoeffs_max*NSP*dest;    // Yp_im
          sp2 = coeff + ncoeffs_max*SP_RDYU_RE + ncoeffs_max*NSP*dest;  // rDYu_re
          sp3 = coeff + ncoeffs_max*SP_RDYU_IM + ncoeffs_max*NSP*dest;  // rDYu_im
          sp4 = coeff + ncoeffs_max*SP_XXDYU_RE + ncoeffs_max*NSP*dest; // xXDYu_re
          sp5 = coeff + ncoeffs_max*SP_XXDYU_IM + ncoeffs_max*NSP*dest; // xXDYu_im

          // Partial sums: part varies slowest, node quickest, coeff middle
          // Partial sums are stored in index for node = 0
          psum_ind = pp*NNODES*ncoeffs_max + coeff*NNODES;

          int_Yp_re[psum_ind] += sum_recv_b[sp0];
          int_Yp_im[psum_ind] += sum_recv_b[sp1];
          int_rDYu_re[psum_ind] += sum_recv_b[sp2];
          int_rDYu_im[psum_ind] += sum_recv_b[sp3];
          int_xXDYu_re[psum_ind] += sum_recv_b[sp4];
          int_xXDYu_im[psum_ind] += sum_recv_b[sp5];
        }
      }
    } // loop over tk
  }
}

__global__ void compute_error(real lamb_cut, int ncoeffs_max, int nparts,
  part_struct *parts, real *part_errors, int *part_nums)
{
  int part = blockIdx.x;
  int coeff = threadIdx.x;

  real div = 0.;
  real max = DBL_MIN;

  __shared__ real s_coeffs[MAX_COEFFS * NSP];
  __shared__ real s_coeffs0[MAX_COEFFS * NSP];
  __shared__ real s_max[MAX_COEFFS];

  if (part < nparts && coeff < ncoeffs_max) {

    s_coeffs[coeff + ncoeffs_max * 0] = parts[part].pnm_re[coeff];
    s_coeffs[coeff + ncoeffs_max * 1] = parts[part].pnm_im[coeff];
    s_coeffs[coeff + ncoeffs_max * 2] = parts[part].phinm_re[coeff];
    s_coeffs[coeff + ncoeffs_max * 3] = parts[part].phinm_im[coeff];
    s_coeffs[coeff + ncoeffs_max * 4] = parts[part].chinm_re[coeff];
    s_coeffs[coeff + ncoeffs_max * 5] = parts[part].chinm_im[coeff];

    s_coeffs0[coeff + ncoeffs_max * 0] = parts[part].pnm_re0[coeff];
    s_coeffs0[coeff + ncoeffs_max * 1] = parts[part].pnm_im0[coeff];
    s_coeffs0[coeff + ncoeffs_max * 2] = parts[part].phinm_re0[coeff];
    s_coeffs0[coeff + ncoeffs_max * 3] = parts[part].phinm_im0[coeff];
    s_coeffs0[coeff + ncoeffs_max * 4] = parts[part].chinm_re0[coeff];
    s_coeffs0[coeff + ncoeffs_max * 5] = parts[part].chinm_im0[coeff];

    s_max[coeff] = DBL_MIN;

    __syncthreads();
    
    // If coefficient has a large enough magnitude (relative to 0th order coeff)
    //  calculate the error
    for (int i = 0; i < NSP; i++) {
      int c = coeff + ncoeffs_max * i;

      // Determine if current coefficient has large enough value compared to 0th
      // (also, make sure it's large enough so we don't get issues with close-to-zero
      //  errors)
      // (also, if zeroth order is 0, ignore)
      real curr_val = s_coeffs[c];
      real zeroth_val = s_coeffs[0 + ncoeffs_max * i];
      int flag = (fabs(curr_val) > fabs(lamb_cut*zeroth_val)) *
                  (fabs(curr_val) > 1.e-16) *
                  (fabs(zeroth_val) > DBL_MIN);

      // If flag == 1, set scoeff equal to error value
      // If flag == 0, set scoeff equal to zero (no error)
      div = fabs(curr_val);
      div += (1.e-16 - div) * (div < 1.e-16);
      real curr_val0 = s_coeffs0[c];

      s_coeffs[c] = (real) flag * fabs(curr_val - curr_val0) / div;

      // See if current error is the max we've seen so far over all the
      // coefficients of a given order, set if so
      s_max[coeff] += (s_coeffs[c] - s_max[coeff]) * (s_coeffs[c] > s_max[coeff]);
    }

    __syncthreads();

    // We've now calculated the error for each "large enough" coefficients and
    //  found the maximum over all coefficients of a given order. Now, each
    //  order has a maximum, and we need to find the max over these
    if (coeff == 0) {
      for (int i = 0; i < ncoeffs_max; i++) {
        max += (s_max[i] - max) * (s_max[i] > max);
      }
      part_errors[part] = max;
      part_nums[part] = parts[part].N;
    }
  }
}

__global__ void store_coeffs(part_struct *parts, int nparts,
  int ncoeffs_max)
{
  int part = blockIdx.x;
  int coeff = threadIdx.x;
  if (part < nparts && coeff < ncoeffs_max) {
   parts[part].pnm_re0[coeff] = parts[part].pnm_re[coeff];
   parts[part].pnm_im0[coeff] = parts[part].pnm_im[coeff];
   parts[part].phinm_re0[coeff] = parts[part].phinm_re[coeff];
   parts[part].phinm_im0[coeff] = parts[part].phinm_im[coeff];
   parts[part].chinm_re0[coeff] = parts[part].chinm_re[coeff];
   parts[part].chinm_im0[coeff] = parts[part].chinm_im[coeff];
  }
}
