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

#ifdef TEST 
#include <cuda.h>

#include "cuda_testing.h"
#include "cuda_physalis.h"

/* more tests: (for pp/cg)
 * Taylor Green Vortex - domain is [0, 2pi]
 *   p = -0.25*rho*[cos(2x) + cos(2x)]
 *   del^2 p = rho*[cos(2x) + cos(2y)]
 *   real f = cos(2.*x) + cos(2.*y);
 * phi = cos(x)^2 + cos(y)^2 + cos(x)*cos(y)
 *   grad(phi) = (-2cos(x)sin(x) - sin(x)cos(y), -2cos(y)sin(y) - sin(y)cos(x))
 *   del^2 phi = -2(cos(x)^2 + cos(y)^2 - sin(x)^2 - sin(y)^2) - 2cos(x)cos(y)
 *   real f = -2.*(cos(x)*cos(x) + cos(y)*cos(y) 
 *        - sin(x)*sin(x) - sin(y)*sin(y)) - 2.*cos(x)*cos(y);
 *
 * phi = cos(x)^2 + cos(y)^2 + cos(z)^2
 *  del2 = -2(c2x + c2y + c2z - s2x - s2y - s2z) - 3cxcycz
 *  real f = -2.*(cos(x)*cos(x) + cos(y)*cos(y) + cos(z)*cos(z) 
 *           - sin(x)*sin(x) - sin(y)*sin(y) - sin(z)*sin(z)) 
 *       - 3.*cos(x)*cos(y)*cos(z);
 *
 * 2 Deltas functions
 *  real x0 = DOM->xs + 1.5*DOM->dx;
 *  real y0 = DOM->ys + 1.5*DOM->dy;
 *  real x1 = DOM->xe - 1.5*DOM->dx;
 *  real y1 = DOM->ye - 1.5*DOM->dy;
 *  real f = 1. * ((x == x0) && (y == y0))
 *          -1. * ((x == x1) && (y == y1));
 *
 */

/* Variable definitions at bottom of file */

extern "C"
void run_test(void)
{
  #ifdef TEST_EXP
    cuda_U_star_test_exp();
  #elif TEST_SIN
    cuda_U_star_test_sin();
  #elif TEST_BC_PERIODIC
    cuda_BC_test_periodic();
  #elif TEST_BC_DIRICHLET
    cuda_BC_test_dirichlet();
  #elif TEST_BC_NEUMANN
    cuda_BC_test_neumann();
  #elif TEST_LEBEDEV_INTERP
    cuda_quad_interp_test();
  #elif TEST_LAMB
    cuda_lamb_test();
  #endif // TEST_EXP
}

extern "C"
void cuda_U_star_test_init(void)
{
  us_a = (real*) malloc(dom[rank].Gfx.s3b * sizeof(real)); // u_star
  uc_a = (real*) malloc(dom[rank].Gfx.s3b * sizeof(real)); // conv_u
  ud_a = (real*) malloc(dom[rank].Gfx.s3b * sizeof(real)); // diff_u
  vs_a = (real*) malloc(dom[rank].Gfy.s3b * sizeof(real)); // v_star
  vc_a = (real*) malloc(dom[rank].Gfy.s3b * sizeof(real)); // conv_v
  vd_a = (real*) malloc(dom[rank].Gfy.s3b * sizeof(real)); // diff_v
  ws_a = (real*) malloc(dom[rank].Gfz.s3b * sizeof(real)); // w_star
  wc_a = (real*) malloc(dom[rank].Gfz.s3b * sizeof(real)); // conv_w
  wd_a = (real*) malloc(dom[rank].Gfz.s3b * sizeof(real)); // diff_w

  u_star_err_min = FLT_MAX;
  u_star_err_max = FLT_MIN;
  conv_u_err_min = FLT_MAX;
  conv_u_err_max = FLT_MIN;
  diff_u_err_min = FLT_MAX;
  diff_u_err_max = FLT_MIN;

  v_star_err_min = FLT_MAX;
  v_star_err_max = FLT_MIN;
  conv_v_err_min = FLT_MAX;
  conv_v_err_max = FLT_MIN;
  diff_v_err_min = FLT_MAX;
  diff_v_err_max = FLT_MIN;

  w_star_err_min = FLT_MAX;
  w_star_err_max = FLT_MIN;
  conv_w_err_min = FLT_MAX;
  conv_w_err_max = FLT_MIN;
  diff_w_err_min = FLT_MAX;
  diff_w_err_max = FLT_MIN;
}

extern "C"
void cuda_BC_test_init(void)
{
  p_i = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  u_i = (real*) malloc(dom[rank].Gfx.s3b * sizeof(real));
  v_i = (real*) malloc(dom[rank].Gfy.s3b * sizeof(real));
  w_i = (real*) malloc(dom[rank].Gfz.s3b * sizeof(real));

  p_err_min = FLT_MAX;
  p_err_max = FLT_MIN;
  u_err_min = FLT_MAX;
  u_err_max = FLT_MIN;
  v_err_min = FLT_MAX;
  v_err_max = FLT_MIN;
  w_err_min = FLT_MAX;
  w_err_max = FLT_MIN;
}

extern "C"
void cuda_U_star_test_clean(void)
{
  /* clean up */
  free(us_a);
  free(uc_a);
  free(ud_a);
  free(vs_a);
  free(vc_a);
  free(vd_a);
  free(ws_a);
  free(wc_a);
  free(wd_a);
}

extern "C"
void cuda_BC_test_clean(void)
{
  /* clean up */
  free(p_i);
  free(u_i);
  free(v_i);
  free(w_i);
}

extern "C"
void cuda_U_star_test_exp(void)
{
  int i,j,k;  // local iterators
  int c;      // local cell locations

  if (rank == 0) {
    printf("\nIntermediate velocity calculation validation:\n\n");
    printf("  u = exp(x), v = exp(y), w = exp(z)\n\n");
  }
  /* Test with exponentials
   * Global domain: x,y = [3, 4]
   *    NOTE: If the domain is the wrong range, the exact soln goes to zero
   *          and therefore the relative error blows up.
   *          WHAT IS THIS RANGE AND CAN WE BE SMARTER?
   * Boundary conditions: All DIRICHLET 0 0
   * Initial conditions: QUIESCENT
   * Forcing: none
   *  t0 = 0 (dt0 = 0)
   *    u0 = exp(x)
   *    v0 = exp(y)
   *    w0 = exp(z)
   *  t1 = dt
   *    u* = exp(x)*[1 + dt*( nu - 2*exp(x) - exp(y) - exp(z) )]
   *    v* = exp(y)*[1 + dt*( nu - 2*exp(y) - exp(z) - exp(x) )]
   *    w* = exp(z)*[1 + dt*( nu - 2*exp(z) - exp(x) - exp(y) )]
   */

  dt0 = 0.;
  dt = 0.1;
  if (rank == 0) {
    printf("  dt = %lf, nu = %lf\n\n", dt, nu);
  }

  /* Init variables */
  cuda_U_star_test_init();

  /* Expected solution */
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        real x = (i - 1.0)*dom[rank].dx + dom[rank].xs;
        real y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        real z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        u[c] = exp(x);
        conv_u[c] = exp(x)*(2.*exp(x) + exp(y) + exp(z));
        diff_u[c] = nu*exp(x);
        u_star[c] = u[c] + dt*(-conv_u[c] + diff_u[c]);

        us_a[c] = u_star[c];
        ud_a[c] = diff_u[c];
        uc_a[c] = conv_u[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        real x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        real y = (j - 1.0)*dom[rank].dy + dom[rank].ys;
        real z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        v[c] = exp(y);
        conv_v[c] = exp(y)*(exp(x) + 2.*exp(y) + exp(z));
        diff_v[c] = nu*exp(y);
        v_star[c] = v[c] + dt*(-conv_v[c] + diff_v[c]);

        vs_a[c] = v_star[c];
        vd_a[c] = diff_v[c];
        vc_a[c] = conv_v[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        real x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        real y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        real z = (k - 1.0)*dom[rank].dz + dom[rank].zs;

        w[c] = exp(z);
        conv_w[c] = exp(z)*(exp(x) + exp(y) + 2.*exp(z));
        diff_w[c] = nu*exp(z);
        w_star[c] = w[c] + dt*(-conv_w[c] + diff_w[c]);

        ws_a[c] = w_star[c];
        wd_a[c] = diff_w[c];
        wc_a[c] = conv_w[c];
      }
    }
  }

  /* Write expected solution */
  //printf("N%d >> Writing expected solution to:   flow-ghost-%.1lf.cgns...\n", 
  //  rank, 1.);
  //cgns_flow_field_ghost(1.);
  //out_VTK_ghost(1);

  /* initialize input velocity fields */
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        real x = (i - 1.0)*dom[rank].dx + dom[rank].xs;
        u[c] = exp(x);
        u0[c] = u[c];
        u_star[c] = 0.;
        conv_u[c] = 0.;
        diff_u[c] = 0.;
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        real y = (j - 1.0)*dom[rank].dy + dom[rank].ys;
        v[c] = exp(y);
        v0[c] = v[c];
        v_star[c] = 0.;
        conv_v[c] = 0.;
        diff_v[c] = 0.;
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        real z = (k - 1.0)*dom[rank].dz + dom[rank].zs;
        w[c] = exp(z);
        w0[c] = w[c];
        w_star[c] = 0.;
        conv_w[c] = 0.;
        diff_w[c] = 0.;
      }
    }
  }
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        p[c] = 0.;
        p0[c] = 0.;
      }
    }
  }

  /* write initial fields */
  //printf("N%d >> Writing initial fields to:      flow-ghost-%.1lf.cgns...\n", 
  //  rank, 0.);
  //cgns_flow_field_ghost(0.);
  //out_VTK_ghost(0);

  /* push fields to device */
  printf("N%d >> Pushing initial fields to device...\n", rank);
  cuda_dom_push();

  /* Test u_star calculation */
  printf("N%d >> Running cuda_U_star...\n", rank);
  cuda_U_star();

  /* Pull fields back */
  printf("N%d >> Pulling fields back to host...\n", rank);
  cuda_dom_pull();

  /* write computed (as u_star) */
  //printf("N%d >> Writing computed fields to:     flow-ghost-%.1lf.cgns...\n", 
  //  rank, 2.);
  //cgns_flow_field_ghost(2.);
  //out_VTK_ghost(2);

  /* Compute error */
  for (k = dom[rank].Gfx._ks; k <= dom[rank].Gfx._ke; k++) {
    for (j = dom[rank].Gfx._js; j <= dom[rank].Gfx._je; j++) {
      for (i = dom[rank].Gfx._is; i <= dom[rank].Gfx._ie; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        
        /* u_star */
        u_star[c] = (u_star[c] - us_a[c]) / us_a[c];
        if (fabs(u_star[c]) > u_star_err_max) u_star_err_max = fabs(u_star[c]);
        if (fabs(u_star[c]) < u_star_err_min) u_star_err_min = fabs(u_star[c]);

        /* conv_u */
        conv_u[c] = (conv_u[c] - uc_a[c]) / uc_a[c];
        if (fabs(conv_u[c]) > conv_u_err_max) conv_u_err_max = fabs(conv_u[c]);
        if (fabs(conv_u[c]) < conv_u_err_min) conv_u_err_min = fabs(conv_u[c]);

        /* diff */
        diff_u[c] = (diff_u[c] - ud_a[c]) / ud_a[c];
        if (fabs(diff_u[c]) > diff_u_err_max) diff_u_err_max = fabs(diff_u[c]);
        if (fabs(diff_u[c]) < diff_u_err_min) diff_u_err_min = fabs(diff_u[c]);
      }
    }
  }
  for (k = dom[rank].Gfy._ks; k <= dom[rank].Gfy._ke; k++) {
    for (j = dom[rank].Gfy._js; j <= dom[rank].Gfy._je; j++) {
      for (i = dom[rank].Gfy._is; i <= dom[rank].Gfy._ie; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        
        /* v_star */
        v_star[c] = (v_star[c] - vs_a[c]) / vs_a[c];
        if (fabs(v_star[c]) > v_star_err_max) v_star_err_max = fabs(v_star[c]);
        if (fabs(v_star[c]) < v_star_err_min) v_star_err_min = fabs(v_star[c]);

        /* conv_v */
        conv_v[c] = (conv_v[c] - vc_a[c]) / vc_a[c];
        if (fabs(conv_v[c]) > conv_v_err_max) conv_v_err_max = fabs(conv_v[c]);
        if (fabs(conv_v[c]) < conv_v_err_min) conv_v_err_min = fabs(conv_v[c]);

        /* diff */
        diff_v[c] = (diff_v[c] - vd_a[c]) / vd_a[c];
        if (fabs(diff_v[c]) > diff_v_err_max) diff_v_err_max = fabs(diff_v[c]);
        if (fabs(diff_v[c]) < diff_v_err_min) diff_v_err_min = fabs(diff_v[c]);
      }
    }
  }
  for (k = dom[rank].Gfz._ks; k <= dom[rank].Gfz._ke; k++) {
    for (j = dom[rank].Gfz._js; j <= dom[rank].Gfz._je; j++) {
      for (i = dom[rank].Gfz._is; i <= dom[rank].Gfz._ie; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        
        /* w_star */
        w_star[c] = (w_star[c] - ws_a[c]) / ws_a[c];
        if (fabs(w_star[c]) > w_star_err_max) w_star_err_max = fabs(w_star[c]);
        if (fabs(w_star[c]) < w_star_err_min) w_star_err_min = fabs(w_star[c]);

        /* conv_w */
        conv_w[c] = (conv_w[c] - wc_a[c]) / wc_a[c];
        if (fabs(conv_w[c]) > conv_w_err_max) conv_w_err_max = fabs(conv_w[c]);
        if (fabs(conv_w[c]) < conv_w_err_min) conv_w_err_min = fabs(conv_w[c]);

        /* diff_w */
        diff_w[c] = (diff_w[c] - wd_a[c]) / wd_a[c];
        if (fabs(diff_w[c]) > diff_w_err_max) diff_w_err_max = fabs(diff_w[c]);
        if (fabs(diff_w[c]) < diff_w_err_min) diff_w_err_min = fabs(diff_w[c]);
      }
    }
  }

  /* write error */
  //printf("N%d >> Writing error difference to:    flow-ghost-%.1lf.cgns...\n", 
  //  rank, 3.);
  //cgns_flow_field_ghost(3.);
  //out_VTK_ghost(3);

  for (int n = 0; n < DOM.S3; n++) {
    if (rank == n) {
      printf("\nN%d >>  Error summary:\n", rank);
      printf("N%d >>  Velocity component:     minimum error:     maximum error:\n",
        rank);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, u_star_err_min, u_star_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, v_star_err_min, v_star_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, w_star_err_min, w_star_err_max);

      printf("N%d >>  Convective component:   minimum error:     maximum error:\n",
        rank);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, conv_u_err_min, conv_u_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, conv_v_err_min, conv_v_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, conv_w_err_min, conv_w_err_max);

      printf("N%d >>  Diffusive component:    minimum error:     maximum error:\n",
        rank);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, diff_u_err_min, diff_u_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, diff_v_err_min, diff_v_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, diff_w_err_min, diff_w_err_max);
    }
    WAIT();
  }

  /* clean up */
  cuda_U_star_test_clean();
}

extern "C"
void cuda_U_star_test_sin(void)
{
  int i,j,k;  // local iterators
  int c;    // cell locations

  if (rank == 0) {
    printf("\nIntermediate velocity calculation validation:\n\n");
    printf("  Sines\n\n");
    printf("  u = sin(y) + 1, v = sin(z) + 1, w = sin(x) + 1\n\n");
  }

  /* Test with sins
   *  * Global domain: x,y = [0, pi] 
   *  * Boundary conditions: All DIRICHLET 0 0
   *  * Initial conditions: QUIESCENT
   *  * Forcing: none
   *  * Remember to take conservative convective term!
   *  *   bc d_j u_i u_j != u_j d_j u_i unless divergence free
   *  * +1 makes it so soln is nonzero, so no issues with relative error
   *  *   e.g. no divide by zero
   *  t0 = 0 (dt0 = 0)
   *    u0 = sin(y) + 1
   *    v0 = sin(z) + 1
   *    w0 = sin(x) + 1
   *  t1 = dt
   *    u* = sin(y) + 1 - dt*[(sin(z) + 1)*cos(y) + nu*sin(y)]
   *    v* = sin(z) + 1 - dt*[(sin(x) + 1)*cos(z) + nu*sin(z)]
   *    w* = sin(x) + 1 - dt*[(sin(y) + 1)*cos(x) + nu*sin(x)]
   */

  dt0 = 0.;
  dt = 0.1;
  if (rank == 0) {
    printf("  dt = %lf, nu = %lf\n\n", dt, nu);
  }

  /* Init variables */
  cuda_U_star_test_init();

  /* Expected solution */
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        real y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        real z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        u[c] = sin(y) + 1.;
        conv_u[c] = (sin(z) + 1.)*cos(y);
        diff_u[c] = -nu*sin(y);
        u_star[c] = u[c] + dt*(-conv_u[c] + diff_u[c]);

        us_a[c] = u_star[c];
        ud_a[c] = diff_u[c];
        uc_a[c] = conv_u[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        real z = (k - 0.5)*dom[rank].dz + dom[rank].zs;
        real x = (i - 0.5)*dom[rank].dx + dom[rank].xs;

        v[c] = sin(z) + 1.;
        conv_v[c] = (sin(x) + 1.)*cos(z);
        diff_v[c] = -nu*sin(z);
        v_star[c] = v[c] + dt*(-conv_v[c] + diff_v[c]);

        vs_a[c] = v_star[c];
        vd_a[c] = diff_v[c];
        vc_a[c] = conv_v[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        real x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        real y = (j - 0.5)*dom[rank].dy + dom[rank].ys;

        w[c] = sin(x) + 1.;
        conv_w[c] = (sin(y) + 1.)*cos(x);
        diff_w[c] = -nu*sin(x);
        w_star[c] = w[c] + dt*(-conv_w[c] + diff_w[c]);

        ws_a[c] = w_star[c];
        wd_a[c] = diff_w[c];
        wc_a[c] = conv_w[c];
      }
    }
  }

  /* Write expected solution */
  printf("Output turned off! In %s at %d\n", __FILE__, __LINE__);
  //printf("N%d >> Writing expected solution to:   flow-ghost-%.1lf.cgns...\n", 
  //  rank, 1.);
  //cgns_flow_field_ghost(1.);
  //out_VTK_ghost(1);

  /* initialize input velocity fields */
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        real y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        u[c] = sin(y) + 1.;
        u0[c] = u[c];
        u_star[c] = 0.;
        conv_u[c] = 0.;
        diff_u[c] = 0.;
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        real z = (k - 0.5)*dom[rank].dz + dom[rank].zs;
        v[c] = sin(z) + 1.;
        v0[c] = v[c];
        v_star[c] = 0.;
        conv_v[c] = 0.;
        diff_v[c] = 0.;
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        real x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        w[c] = sin(x) + 1.;
        w0[c] = w[c];
        w_star[c] = 0.;
        conv_w[c] = 0.;
        diff_w[c] = 0.;
      }
    }
  }
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        p[c] = 0.;
        p0[c] = 0.;
      }
    }
  }

  /* write initial fields */
  printf("Output turned off! In %s at %d\n", __FILE__, __LINE__);
  //printf("N%d >> Writing initial fields to:      flow-ghost-%.1lf.cgns...\n", 
  //  rank, 0.);
  //cgns_flow_field_ghost(0.);
  //out_VTK_ghost(0);

  /* push fields to device */
  printf("N%d >> Pushing initial fields to device...\n", rank);
  cuda_dom_push();

  /* Test u_star calculation */
  printf("N%d >> Running cuda_U_star...\n", rank);
  cuda_U_star();

  /* Pull fields back */
  printf("N%d >> Pulling fields back to host...\n", rank);
  cuda_dom_pull();

  /* write computed */
  printf("Output turned off! In %s at %d\n", __FILE__, __LINE__);
  //printf("N%d >> Writing computed fields to:     flow-ghost-%.1lf.cgns...\n", 
  //  rank, 2.);
  //cgns_flow_field_ghost(2.);
  //out_VTK_ghost(2);

  /* Compute error */
  for (k = dom[rank].Gfx._ks; k <= dom[rank].Gfx._ke; k++) {
    for (j = dom[rank].Gfx._js; j <= dom[rank].Gfx._je; j++) {
      for (i = dom[rank].Gfx._is; i <= dom[rank].Gfx._ie; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        
        /* u_star */
        u_star[c] = (u_star[c] - us_a[c]) / us_a[c];
        if (fabs(u_star[c]) > u_star_err_max) u_star_err_max = fabs(u_star[c]);
        if (fabs(u_star[c]) < u_star_err_min) u_star_err_min = fabs(u_star[c]);

        /* conv_u */
        conv_u[c] = (conv_u[c] - uc_a[c]) / uc_a[c];
        if (fabs(conv_u[c]) > conv_u_err_max) conv_u_err_max = fabs(conv_u[c]);
        if (fabs(conv_u[c]) < conv_u_err_min) conv_u_err_min = fabs(conv_u[c]);

        /* diff */
        diff_u[c] = (diff_u[c] - ud_a[c]) / ud_a[c];
        if (fabs(diff_u[c]) > diff_u_err_max) diff_u_err_max = fabs(diff_u[c]);
        if (fabs(diff_u[c]) < diff_u_err_min) diff_u_err_min = fabs(diff_u[c]);
      }
    }
  }
  for (k = dom[rank].Gfy._ks; k <= dom[rank].Gfy._ke; k++) {
    for (j = dom[rank].Gfy._js; j <= dom[rank].Gfy._je; j++) {
      for (i = dom[rank].Gfy._is; i <= dom[rank].Gfy._ie; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        
        /* v_star */
        v_star[c] = (v_star[c] - vs_a[c]) / vs_a[c];
        if (fabs(v_star[c]) > v_star_err_max) v_star_err_max = fabs(v_star[c]);
        if (fabs(v_star[c]) < v_star_err_min) v_star_err_min = fabs(v_star[c]);

        /* conv_v */
        conv_v[c] = (conv_v[c] - vc_a[c]) / vc_a[c];
        if (fabs(conv_v[c]) > conv_v_err_max) conv_v_err_max = fabs(conv_v[c]);
        if (fabs(conv_v[c]) < conv_v_err_min) conv_v_err_min = fabs(conv_v[c]);

        /* diff */
        diff_v[c] = (diff_v[c] - vd_a[c]) / vd_a[c];
        if (fabs(diff_v[c]) > diff_v_err_max) diff_v_err_max = fabs(diff_v[c]);
        if (fabs(diff_v[c]) < diff_v_err_min) diff_v_err_min = fabs(diff_v[c]);
      }
    }
  }
  for (k = dom[rank].Gfz._ks; k <= dom[rank].Gfz._ke; k++) {
    for (j = dom[rank].Gfz._js; j <= dom[rank].Gfz._je; j++) {
      for (i = dom[rank].Gfz._is; i <= dom[rank].Gfz._ie; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        
        /* w_star */
        w_star[c] = (w_star[c] - ws_a[c]) / ws_a[c];
        if (fabs(w_star[c]) > w_star_err_max) w_star_err_max = fabs(w_star[c]);
        if (fabs(w_star[c]) < w_star_err_min) w_star_err_min = fabs(w_star[c]);

        /* conv_w */
        conv_w[c] = (conv_w[c] - wc_a[c]) / wc_a[c];
        if (fabs(conv_w[c]) > conv_w_err_max) conv_w_err_max = fabs(conv_w[c]);
        if (fabs(conv_w[c]) < conv_w_err_min) conv_w_err_min = fabs(conv_w[c]);

        /* diff_w */
        diff_w[c] = (diff_w[c] - wd_a[c]) / wd_a[c];
        if (fabs(diff_w[c]) > diff_w_err_max) diff_w_err_max = fabs(diff_w[c]);
        if (fabs(diff_w[c]) < diff_w_err_min) diff_w_err_min = fabs(diff_w[c]);
      }
    }
  }

  /* write error */
  printf("Output turned off! In %s at %d\n", __FILE__, __LINE__);
  //printf("N%d >> Writing error difference to:    flow-ghost-%.1lf.cgns...\n", 
  //  rank, 3.);
  //cgns_flow_field_ghost(3.);
  //out_VTK_ghost(3);

  for (int n = 0; n < DOM.S3; n++) {
    if (rank == n) {
      printf("\nN%d >>  Error summary:\n", rank);
      printf("N%d >>  Velocity component:     minimum error:     maximum error:\n",
        rank);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, u_star_err_min, u_star_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, v_star_err_min, v_star_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, w_star_err_min, w_star_err_max);

      printf("N%d >>  Convective component:   minimum error:     maximum error:\n",
        rank);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, conv_u_err_min, conv_u_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, conv_v_err_min, conv_v_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, conv_w_err_min, conv_w_err_max);

      printf("N%d >>  Diffusive component:    minimum error:     maximum error:\n",
        rank);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, diff_u_err_min, diff_u_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, diff_v_err_min, diff_v_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, diff_w_err_min, diff_w_err_max);
    }
    WAIT();
  }

  /* clean up */
  cuda_U_star_test_clean();
}

extern "C"
void cuda_BC_test_periodic(void)
{
  int i,j,k;  // local iterators
  real x,y,z; // grid position
  int c;      // cell locations

  if (rank == 0) {
    printf("\nBoundary condition application validation:\n");

    /* periodic field (on -1 <= x <= 1, -1 <= y <= 1, -1 <= z <= 1) */
    printf("\n  Periodic boundary conditions:\n");
    printf("    p = cos(pi*x) + cos(pi*y) + cos(pi*z)\n");
    printf("    u = cos(pi*x) + cos(pi*y) + cos(pi*z)\n");
    printf("    v = cos(pi*x) + cos(pi*y) + cos(pi*z)\n");
    printf("    w = cos(pi*x) + cos(pi*y) + cos(pi*z)\n\n");
  }
  /* Test PERIODIC BC
   *  * Global domain: x,y = [-1, 1] 
   *  * Boundary conditions: All PERIODIC
   *  * Initial conditions: QUIESCENT
   *  * Forcing: none
   *  p = cos(pi*x) + cos(pi*y) + cos(pi*z)
   *  u = cos(pi*x) + cos(pi*y) + cos(pi*z)
   *  v = cos(pi*x) + cos(pi*y) + cos(pi*z)
   *  w = cos(pi*x) + cos(pi*y) + cos(pi*z)
   */
  
  /* Init variables */
  cuda_BC_test_init();

  /* Write input fields */
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        p_i[c] = cos(PI*x) + cos(PI*y) + cos(PI*z);
        p[c] = p_i[c];
      }
    }
  }
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        x = (i - 1.0)*dom[rank].dx + dom[rank].xs;
        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        u_i[c] = cos(PI*x) + cos(PI*y) + cos(PI*z);
        u[c] = u_i[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        y = (j - 1.0)*dom[rank].dy + dom[rank].ys;
        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        v_i[c] = cos(PI*x) + cos(PI*y) + cos(PI*z);
        v[c] = v_i[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        z = (k - 1.0)*dom[rank].dz + dom[rank].zs;

        w_i[c] = cos(PI*x) + cos(PI*y) + cos(PI*z);
        w[c] = w_i[c];
      }
    }
  }

  /* Write expected solution */
  //printf("N%d >> Writing expected solution to:   flow-ghost-%.1lf.cgns...\n", 
  //  rank, 0.);
  //cgns_flow_field_ghost(0.);
  //out_VTK_ghost(0);

  /* push fields to device */
  printf("N%d >> Pushing initial fields to device...\n", rank);
  cuda_dom_push();

  /* apply BC */
  printf("N%d >> Running cuda_BC()...\n", rank);
  cuda_dom_BC();
  mpi_cuda_exchange_Gcc(_p);
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);

  /* Pull fields back */
  printf("N%d >> Pulling fields back to host...\n", rank);
  cuda_dom_pull();

  /* write computed */ 
  //printf("N%d >> Writing computed fields to:     flow-ghost-%.1lf.cgns...\n", 
  //  rank, 1.);
  //cgns_flow_field_ghost(1.);
  //out_VTK_ghost(1);

  /* copy results and compute error */
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        p[c] = p[c] - p_i[c];
        if(fabs(p[c]) > p_err_max) p_err_max = fabs(p[c]);
        if(fabs(p[c]) < p_err_min) p_err_min = fabs(p[c]);
      }
    }
  }
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        u[c] = u[c] - u_i[c];
        if(fabs(u[c]) > u_err_max) u_err_max = fabs(u[c]);
        if(fabs(u[c]) < u_err_min) u_err_min = fabs(u[c]);
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        v[c] = v[c] - v_i[c];
        if(fabs(v[c]) > v_err_max) v_err_max = fabs(v[c]);
        if(fabs(v[c]) < v_err_min) v_err_min = fabs(v[c]);
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        w[c] = w[c] - w_i[c];
        if(fabs(w[c]) > w_err_max) w_err_max = fabs(w[c]);
        if(fabs(w[c]) < w_err_min) w_err_min = fabs(w[c]);
      }
    }
  }

  /* write error */
  //printf("N%d >> Writing error difference to:    flow-ghost-%.1lf.cgns...\n", 
  //  rank, 2.);
  //cgns_flow_field_ghost(2.);
  //out_VTK_ghost(2);

  for (int n = 0; n < DOM.S3; n++) {
    if (rank == n) {
      printf("\nN%d >>  Error summary:\n", rank);
      printf("N%d >>  Field component:        minimum error:     maximum error:\n",
        rank);
      printf("N%d >>          p              %12.3e       %12.3e\n",
        rank, p_err_min, p_err_max);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, u_err_min, u_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, v_err_min, v_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, w_err_min, w_err_max);
    }
    WAIT();
  }

  /* clean up */
  cuda_BC_test_clean();
}

extern "C"
void cuda_BC_test_dirichlet(void)
{
  int i,j,k;  // local iterators
  real x,y,z; // grid position
  int c;      // cell locations

  if (rank == 0) {
    printf("\nBoundary condition application validation:\n");

    /* dirichlet field (on -1 <= x <= 1, -1 <= y <= 1, -1 <= z <= 1) */
    printf("\n  Dirichlet boundary conditions:\n");
    //printf("    p = sin(pi*x) * sin(pi*y) * sin(pi*z)\n");
    printf("    u = sin(pi*x) * sin(pi*y) * sin(pi*z)\n");
    printf("    v = sin(pi*x) * sin(pi*y) * sin(pi*z)\n");
    printf("    w = sin(pi*x) * sin(pi*y) * sin(pi*z)\n\n");
  }
  /* Test DIRICHLET BC
   *  * Global domain: x,y = [-1, 1] 
   *  * Boundary conditions: All DIRICHLET 0 0 (neumann pressure)
   *  * Initial conditions: QUIESCENT
   *  * Forcing: none
   *  p = sin(pi*x) * sin(pi*y) * sin(pi*z)
   *  u = sin(pi*x) * sin(pi*y) * sin(pi*z)
   *  v = sin(pi*x) * sin(pi*y) * sin(pi*z)
   *  w = sin(pi*x) * sin(pi*y) * sin(pi*z)
   */
  // For now, don't do pressure -- its being tested in test_neumann
  
  /* Init variables */
  cuda_BC_test_init();

  /* Write input fields */
//  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
//    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
//      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
//        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
//        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
//        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;
//
//        p_i[c] = sin(PI*x) * sin(PI*y) * sin(PI*z);
//        p[c] = p_i[c];
//
//      }
//    }
//  }

//  /* set neumann bc on pressure */
//  /* West face */
//  if (dom[rank].I == DOM.Is) {
//    i = dom[rank].Gcc._isb;
//    int ie = dom[rank].Gcc._is;
//    int ce;
//    for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
//      for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
//        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//        ce = GCC_LOC(ie, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//
//        p_i[c] = p_i[ce];
//        p[c] = p[ce];
//      }
//    }
//  }
//  /* East */
//  if (dom[rank].I == DOM.Ie) {
//    i = dom[rank].Gcc._ieb;
//    int iw = dom[rank].Gcc._ie;
//    int cw;
//    for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
//      for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
//        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//        cw = GCC_LOC(iw, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//
//        p_i[c] = p_i[cw];
//        p[c] = p[cw];
//      }
//    }
//  }
//  /* South */
//  if (dom[rank].J == DOM.Js) {
//    j = dom[rank].Gcc._jsb;
//    int jn = dom[rank].Gcc._js;
//    int cn;
//    for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
//      for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
//        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//        cn = GCC_LOC(i, jn, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//
//        p_i[c] = p_i[cn];
//        p[c] = p[cn];
//
//      }
//    }
//  }
//  /* North */
//  if (dom[rank].J == DOM.Je) {
//    j = dom[rank].Gcc._jeb;
//    int js = dom[rank].Gcc._je;
//    for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
//      for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
//        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//        int cs = GCC_LOC(i, js, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//
//        p_i[c] = p_i[cs];
//        p[c] = p[c];
//      }
//    }
//  }
//  /* Bottom */
//  if (dom[rank].K == DOM.Ks) {
//    k = dom[rank].Gcc._ksb;
//    int ks = dom[rank].Gcc._ks;
//    for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
//      for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
//        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//        int ct = GCC_LOC(i, j, ks, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//
//        p_i[c] = p_i[ct];
//        p[c] = p_i[c];
//      }
//    }
//  }
//  /* top */
//  if (dom[rank].K == DOM.Ke) { // top
//    k = dom[rank].Gcc._keb;
//    int kb = dom[rank].Gcc._ke;
//    for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
//      for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
//        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//        int cb = GCC_LOC(i, j, kb, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//
//        p_i[c] = p_i[cb];
//        p[c] = p_i[c];
//      }
//    }
//  }

  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        x = (i - 1.0)*dom[rank].dx + dom[rank].xs;
        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        u_i[c] = sin(PI*x) * sin(PI*y) * sin(PI*z);
        u[c] = u_i[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        y = (j - 1.0)*dom[rank].dy + dom[rank].ys;
        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        v_i[c] = sin(PI*x) * sin(PI*y) * sin(PI*z);
        v[c] = v_i[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        z = (k - 1.0)*dom[rank].dz + dom[rank].zs;

        w_i[c] = sin(PI*x) * sin(PI*y) * sin(PI*z);
        w[c] = w_i[c];
      }
    }
  }

  /* Write expected solution */
  mpi_cuda_exchange_Gcc(_p);
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);

  //printf("N%d >> Writing expected solution to:   flow-ghost-%.1lf.cgns...\n", 
  //  rank, 0.);
  //cgns_flow_field_ghost(0.);
  //out_VTK_ghost(0);

  /* push fields to device */
  printf("N%d >> Pushing initial fields to device...\n", rank);
  cuda_dom_push();

  /* apply BC */
  printf("N%d >> Running cuda_BC()...\n", rank);
  cuda_dom_BC();
  mpi_cuda_exchange_Gcc(_p);
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);

  /* Pull fields back */
  printf("N%d >> Pulling fields back to host...\n", rank);
  cuda_dom_pull();

  /* write computed */ 
  //printf("N%d >> Writing computed fields to:     flow-ghost-%.1lf.cgns...\n", 
  //  rank, 1.);
  //cgns_flow_field_ghost(1.);
  //out_VTK_ghost(1);

  /* copy results and compute error */
//  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
//    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
//      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
//        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
//      
//        // The domain corners are never updated or used,
//        //  so don't look for error there
//        if (is_corner(i, j, k, dom[rank].Gcc)) {
//          p[c] = 0.;
//        } else {
//          p[c] = p[c] - p_i[c];
//          if(fabs(p[c]) > p_err_max) p_err_max = fabs(p[c]);
//          if(fabs(p[c]) < p_err_min) p_err_min = fabs(p[c]);
//        }
//      }
//    }
//  }

  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        if (is_corner(i, j, k, dom[rank].Gfx)) {
          u[c] = 0.;
        } else {
          u[c] = u[c] - u_i[c];
          if(fabs(u[c]) > u_err_max) u_err_max = fabs(u[c]);
          if(fabs(u[c]) < u_err_min) u_err_min = fabs(u[c]);
        }
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        if (is_corner(i, j, k, dom[rank].Gfy)) {
          v[c] = 0.;
        } else {
          v[c] = v[c] - v_i[c];
          if(fabs(v[c]) > v_err_max) v_err_max = fabs(v[c]);
          if(fabs(v[c]) < v_err_min) v_err_min = fabs(v[c]);
        }
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        if (is_corner(i, j, k, dom[rank].Gfz)) {
          w[c] = 0.;
        } else {
          w[c] = w[c] - w_i[c];
          if (fabs(w[c]) > w_err_max) w_err_max = fabs(w[c]);
          if (fabs(w[c]) < w_err_min) w_err_min = fabs(w[c]);
        }
      }
    }
  }

  /* write error */
  //printf("N%d >> Writing error difference to:    flow-ghost-%.1lf.cgns...\n", 
  //  rank, 2.);
  //cgns_flow_field_ghost(2.);
  //out_VTK_ghost(2);

  for (int n = 0; n < DOM.S3; n++) {
    if (rank == n) {
      printf("\nN%d >>  Error summary:\n", rank);
      printf("N%d >>  Field component:        minimum error:     maximum error:\n",
        rank);
      //printf("N%d >>          p              %12.3e       %12.3e\n",
      //  rank, p_err_min, p_err_max);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, u_err_min, u_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, v_err_min, v_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, w_err_min, w_err_max);
      fflush(stdout);
    }
    WAIT();
  }

  /* clean up */
  cuda_BC_test_clean();
}

extern "C"
void cuda_BC_test_neumann(void)
{
  int i,j,k;  // local iterators
  real x,y,z; // grid position
  int c;      // cell locations

  if (rank == 0) {
    printf("\nBoundary condition application validation:\n");

    /* neumann field (on -1 <= x <= 1, -1 <= y <= 1, -1 <= z <= 1) */
    printf("\n  Neumann boundary conditions:\n");
    printf("    (Note that some cosines are stretched more -- see code)\n");
    printf("    p = cos(pi*x) + cos(pi*y) + cos(pi*z)\n");
    printf("    u = cos(pi*x) + cos(pi*y) + cos(pi*z)\n");
    printf("    v = cos(pi*x) + cos(pi*y) + cos(pi*z)\n");
    printf("    w = cos(pi*x) + cos(pi*y) + cos(pi*z)\n\n");
  }
  /* Test DIRICHLET BC
   *  * Global domain: x,y = [-1, 1] 
   *  * Boundary conditions: All NEUMANN
   *  * Initial conditions: QUIESCENT
   *  * Forcing: none
   *  p = cos(pi*x) + cos(pi*y) + cos(pi*z)
   *  u = cos(pi*x) + cos(pi*y) + cos(pi*z)
   *  v = cos(pi*x) + cos(pi*y) + cos(pi*z)
   *  w = cos(pi*x) + cos(pi*y) + cos(pi*z)
   */
  
  /* Init variables */
  cuda_BC_test_init();

  /* Write input fields */
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        p_i[c] = cos(PI*x) + cos(PI*y) + cos(PI*z);
        p[c] = p_i[c];

      }
    }
  }
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        x = (i - 1.0)*dom[rank].dx + dom[rank].xs;
        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        // In order to be neumann at isb/is and ie/ieb, need to redefine 
        // in the x direction -- maximum between 1+dx and -(1+dx)
        u_i[c] = cos(2.*PI*x/(2. + dom[rank].dx)) + cos(PI*y) + cos(PI*z);
        u[c] = u_i[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        y = (j - 1.0)*dom[rank].dy + dom[rank].ys;
        z = (k - 0.5)*dom[rank].dz + dom[rank].zs;

        v_i[c] = cos(PI*x) + cos(2.*PI*y/(2. + dom[rank].dy)) + cos(PI*z);
        v[c] = v_i[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        x = (i - 0.5)*dom[rank].dx + dom[rank].xs;
        y = (j - 0.5)*dom[rank].dy + dom[rank].ys;
        z = (k - 1.0)*dom[rank].dz + dom[rank].zs;

        w_i[c] = cos(PI*x) + cos(PI*y) + cos(2.*PI*z/(2. + dom[rank].dz));
        w[c] = w_i[c];
      }
    }
  }

  /* Write expected solution */
  mpi_cuda_exchange_Gcc(_p);
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);

  //printf("N%d >> Writing expected solution to:   flow-ghost-%.1lf.cgns...\n", 
  //  rank, 0.);
  //cgns_flow_field_ghost(0.);
  //out_VTK_ghost(0);

  /* Zero the boundary cells to ensure we're not just using the set soln */
  int csb, ceb;
  // x-faces
  for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      csb = GCC_LOC(dom[rank].Gcc._isb, j, k, dom[rank].Gcc.s1b, 
                  dom[rank].Gcc.s2b);
      ceb = GCC_LOC(dom[rank].Gcc._ieb, j, k, dom[rank].Gcc.s1b,
                  dom[rank].Gcc.s2b);
      p[csb] = 0.;
      p[ceb] = 0.;

      csb = GFX_LOC(dom[rank].Gfx._isb, j, k, dom[rank].Gfx.s1b,
                    dom[rank].Gfx.s2b);
      ceb = GFX_LOC(dom[rank].Gfx._ieb, j, k, dom[rank].Gfx.s1b,
                    dom[rank].Gfx.s2b);
      u[csb] = 0.;
      u[ceb] = 0.;

      csb = GFY_LOC(dom[rank].Gfy._isb, j, k, dom[rank].Gfy.s1b,
                    dom[rank].Gfy.s2b);
      ceb = GFY_LOC(dom[rank].Gfy._ieb, j, k, dom[rank].Gfy.s1b,
                    dom[rank].Gfy.s2b);
      v[csb] = 0.;
      v[ceb] = 0.;

      csb = GFZ_LOC(dom[rank].Gfz._isb, j, k, dom[rank].Gfz.s1b,
                    dom[rank].Gfz.s2b);
      ceb = GFZ_LOC(dom[rank].Gfz._ieb, j, k, dom[rank].Gfz.s1b,
                    dom[rank].Gfz.s2b);
      w[csb] = 0.;
      w[ceb] = 0.;
    }
  }
  // y-faces
  for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
    for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
      csb = GCC_LOC(i, dom[rank].Gcc._jsb, k, dom[rank].Gcc.s1b,
                    dom[rank].Gcc.s2b);
      ceb = GCC_LOC(i, dom[rank].Gcc._jeb, k, dom[rank].Gcc.s1b,
                    dom[rank].Gcc.s2b);
      p[csb] = 0.;
      p[ceb] = 0.;

      csb = GFX_LOC(i, dom[rank].Gfx._jsb, k, dom[rank].Gfx.s1b,
                    dom[rank].Gfx.s2b);
      ceb = GFX_LOC(i, dom[rank].Gfx._jeb, k, dom[rank].Gfx.s1b,
                    dom[rank].Gfx.s2b);
      u[csb] = 0.;
      u[ceb] = 0.;

      csb = GFY_LOC(i, dom[rank].Gfy._jsb, k, dom[rank].Gfy.s1b,
                    dom[rank].Gfy.s2b);
      ceb = GFY_LOC(i, dom[rank].Gfy._jeb, k, dom[rank].Gfy.s1b,
                    dom[rank].Gfy.s2b);
      v[csb] = 0.;
      v[ceb] = 0.;

      csb = GFZ_LOC(i, dom[rank].Gfz._jsb, k, dom[rank].Gfz.s1b,
                    dom[rank].Gfz.s2b);
      ceb = GFZ_LOC(i, dom[rank].Gfz._jeb, k, dom[rank].Gfz.s1b,
                    dom[rank].Gfz.s2b);
      w[csb] = 0.;
      w[ceb] = 0.;
    }
  }
  // z-faces
  for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
    for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
      csb = GCC_LOC(i, j, dom[rank].Gcc._ksb, dom[rank].Gcc.s1b,
                    dom[rank].Gcc.s2b);
      ceb = GCC_LOC(i, j, dom[rank].Gcc._keb, dom[rank].Gcc.s1b,
                    dom[rank].Gcc.s2b);
      p[csb] = 0.;
      p[ceb] = 0.;

      csb = GFX_LOC(i, j, dom[rank].Gfx._ksb, dom[rank].Gfx.s1b,
                    dom[rank].Gfx.s2b);
      ceb = GFX_LOC(i, j, dom[rank].Gfx._keb, dom[rank].Gfx.s1b,
                    dom[rank].Gfx.s2b);
      u[csb] = 0.;
      u[ceb] = 0.;

      csb = GFY_LOC(i, j, dom[rank].Gfy._ksb, dom[rank].Gfy.s1b,
                    dom[rank].Gfy.s2b);
      ceb = GFY_LOC(i, j, dom[rank].Gfy._keb, dom[rank].Gfy.s1b,
                    dom[rank].Gfy.s2b);
      v[csb] = 0.;
      v[ceb] = 0.;

      csb = GFZ_LOC(i, j, dom[rank].Gfz._ksb, dom[rank].Gfz.s1b,
                    dom[rank].Gfz.s2b);
      ceb = GFZ_LOC(i, j, dom[rank].Gfz._keb, dom[rank].Gfz.s1b,
                    dom[rank].Gfz.s2b);
      w[csb] = 0.;
      w[ceb] = 0.;
    }
  }

  /* push fields to device */
  printf("N%d >> Pushing initial fields to device...\n", rank);
  cuda_dom_push();

  /* apply BC */
  printf("N%d >> Running cuda_BC()...\n", rank);
  cuda_dom_BC();
  mpi_cuda_exchange_Gcc(_p);
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);

  /* Pull fields back */
  printf("N%d >> Pulling fields back to host...\n", rank);
  cuda_dom_pull();

  /* write computed */ 
  //printf("N%d >> Writing computed fields to:     flow-ghost-%.1lf.cgns...\n", 
  //  rank, 1.);
  //cgns_flow_field_ghost(1.);
  //out_VTK_ghost(1);

  /* copy results and compute error */
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        p[c] = p[c] - p_i[c];
        if(fabs(p[c]) > p_err_max) p_err_max = fabs(p[c]);
        if(fabs(p[c]) < p_err_min) p_err_min = fabs(p[c]);
      }
    }
  }
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        u[c] = u[c] - u_i[c];
        if(fabs(u[c]) > u_err_max) u_err_max = fabs(u[c]);
        if(fabs(u[c]) < u_err_min) u_err_min = fabs(u[c]);
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        v[c] = v[c] - v_i[c];
        if(fabs(v[c]) > v_err_max) v_err_max = fabs(v[c]);
        if(fabs(v[c]) < v_err_min) v_err_min = fabs(v[c]);

      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        w[c] = w[c] - w_i[c];
        if(fabs(w[c]) > w_err_max) w_err_max = fabs(w[c]);
        if(fabs(w[c]) < w_err_min) w_err_min = fabs(w[c]);
      }
    }
  }

  /* write error */
  //printf("N%d >> Writing error difference to:    flow-ghost-%.1lf.cgns...\n", 
  //  rank, 2.);
  //cgns_flow_field_ghost(2.);
  //out_VTK_ghost(2);

  for (int n = 0; n < DOM.S3; n++) {
    if (rank == n) {
      printf("\nN%d >>  Error summary:\n", rank);
      printf("N%d >>  Field component:        minimum error:     maximum error:\n",
        rank);
      printf("N%d >>          p              %12.3e       %12.3e\n",
        rank, p_err_min, p_err_max);
      printf("N%d >>          u              %12.3e       %12.3e\n",
        rank, u_err_min, u_err_max);
      printf("N%d >>          v              %12.3e       %12.3e\n",
        rank, v_err_min, v_err_max);
      printf("N%d >>          w              %12.3e       %12.3e\n\n",
        rank, w_err_min, w_err_max);
    }
    WAIT();
  }

  /* clean up */
  cuda_BC_test_clean();
}

extern "C"
int is_corner(int i, int j, int k, grid_info grid)
{
  // No corner is ever used / updated
  int is_grid =
    ((i == grid._isb) && (j == grid._jsb) && (k == grid._ksb)) ||
    ((i == grid._isb) && (j == grid._jsb) && (k == grid._keb)) ||
    ((i == grid._isb) && (j == grid._jeb) && (k == grid._ksb)) ||
    ((i == grid._isb) && (j == grid._jeb) && (k == grid._keb)) ||
    ((i == grid._ieb) && (j == grid._jsb) && (k == grid._ksb)) ||
    ((i == grid._ieb) && (j == grid._jsb) && (k == grid._keb)) ||
    ((i == grid._ieb) && (j == grid._jeb) && (k == grid._ksb)) ||
    ((i == grid._ieb) && (j == grid._jeb) && (k == grid._keb));

  return is_grid;
}

extern "C"
void cuda_quad_interp_test(void)
{
  // Only implemented for 1 GPU right now. Maybe also one particle?
  // Copied from bluebottle

  int i, j, k;  // iterators
  int C;
  real *p_a = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *u_a = (real*) malloc(dom[rank].Gfx.s3b * sizeof(real));
  real *v_a = (real*) malloc(dom[rank].Gfy.s3b * sizeof(real));
  real *w_a = (real*) malloc(dom[rank].Gfz.s3b * sizeof(real));
  real x, y, z;

  // min and max error search
  real *p_err_min = (real*) malloc(nparts * sizeof(real));
  real *p_err_max = (real*) malloc(nparts * sizeof(real));
  real *u_err_min = (real*) malloc(nparts * sizeof(real));
  real *u_err_max = (real*) malloc(nparts * sizeof(real));
  real *v_err_min = (real*) malloc(nparts * sizeof(real));
  real *v_err_max = (real*) malloc(nparts * sizeof(real));
  real *w_err_min = (real*) malloc(nparts * sizeof(real));
  real *w_err_max = (real*) malloc(nparts * sizeof(real));

  printf("\nLebedev quadrature interpolation validation:\n\n");
  printf("  p = u = v = w = exp(x) + exp(y) + exp(z)\n\n");

  // create analytic result and push to device
  for(k = dom[rank].Gcc.ksb; k <= dom[rank].Gcc.keb; k++) {
    for(j = dom[rank].Gcc.jsb; j <= dom[rank].Gcc.jeb; j++) {
      for(i = dom[rank].Gcc.ksb; i <= dom[rank].Gcc.ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;

        p_a[C] = exp(x) + exp(y) + exp(z);
        p[C] = p_a[C];
      }
    }
  }
  for(k = dom[rank].Gfx.ksb; k <= dom[rank].Gfx.keb; k++) {
    for(j = dom[rank].Gfx.jsb; j <= dom[rank].Gfx.jeb; j++) {
      for(i = dom[rank].Gfx.ksb; i <= dom[rank].Gfx.ieb; i++) {
        C = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        x = (i-1.0)*dom[rank].dx + dom[rank].xs;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;

        u_a[C] = exp(x) + exp(y) + exp(z);
        u[C] = u_a[C];
      }
    }
  }
  for(k = dom[rank].Gfy.ksb; k <= dom[rank].Gfy.keb; k++) {
    for(j = dom[rank].Gfy.jsb; j <= dom[rank].Gfy.jeb; j++) {
      for(i = dom[rank].Gfy.ksb; i <= dom[rank].Gfy.ieb; i++) {
        C = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;
        y = (j-1.0)*dom[rank].dy + dom[rank].ys;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;

        v_a[C] = exp(x) + exp(y) + exp(z);
        v[C] = v_a[C];
      }
    }
  }
  for(k = dom[rank].Gfz.ksb; k <= dom[rank].Gfz.keb; k++) {
    for(j = dom[rank].Gfz.jsb; j <= dom[rank].Gfz.jeb; j++) {
      for(i = dom[rank].Gfz.ksb; i <= dom[rank].Gfz.ieb; i++) {
        C = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;
        z = (k-1.0)*dom[rank].dz + dom[rank].zs;

        w_a[C] = exp(x) + exp(y) + exp(z);
        w[C] = w_a[C];
      }
    }
  }

  // write expected solution
  //rec_paraview_stepnum_out++;
  //printf("  Writing expected solution to: out_%d.pvtr...", rec_paraview_stepnum_out);
  //out_VTK_ghost();
  //printf("done.\n");

  // push fields to device
  printf("\n  Pushing fields to devices...");
  cuda_dom_push();
  printf("done.\n");

  // CPU-side theta, phi
  real PI14 = 0.25 * PI;
  real PI12 = 0.5 * PI;
  real PI34 = 0.75 * PI;
  real PI54 = 1.25 * PI;
  real PI32 = 1.5 * PI;
  real PI74 = 1.75 * PI;
  real alph1 = 0.955316618124509;
  real alph2 = 2.186276035465284;

  // nodes TODO: find a more elegant way of fixing the divide by sin(0)
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

  int nnodes = 26;
  real node_t[nnodes];
  real node_p[nnodes];
  for(i = 0; i < 6; i++) {
    node_t[i] = a1_t[i];
    node_p[i] = a1_p[i];
  }
  for(i = 0; i < 12; i++) {
    node_t[6+i] = a2_t[i];
    node_p[6+i] = a2_p[i];
  }
  for(i = 0; i < 8; i++) {
    node_t[18+i] = a3_t[i];
    node_p[18+i] = a3_p[i];
  }

  // call code to test
  printf("  Running cuda_quad_interp()...");

  // create a place to temporarily store field variables at quadrature nodes
  real *pp = (real*) malloc(nnodes * nparts * sizeof(real));
  real *ur = (real*) malloc(nnodes * nparts * sizeof(real));
  real *ut = (real*) malloc(nnodes * nparts * sizeof(real));
  real *up = (real*) malloc(nnodes * nparts * sizeof(real));
  real *_pp;
  real *_ur;
  real *_ut;
  real *_up;
  cudaMalloc((void**) &_pp, nnodes * nparts * sizeof(real));
  cudaMalloc((void**) &_ur, nnodes * nparts * sizeof(real));
  cudaMalloc((void**) &_ut, nnodes * nparts * sizeof(real));
  cudaMalloc((void**) &_up, nnodes * nparts * sizeof(real));

  // exec config
  dim3 num_parts(nparts);
  dim3 dim_nodes(NNODES);

  check_nodes<<<num_parts, dim_nodes>>>(nparts, _parts, _bc, _DOM);
  interpolate_nodes<<<num_parts, dim_nodes>>>(_p, _u, _v, _w, rho_f, nu, gradP,
    _parts, _pp, _ur, _ut, _up, _bc);
  printf("done.\n");

  // pull fields back to host
  printf("  Pulling fields back to host...");
  cudaMemcpy(pp, _pp, nnodes * nparts * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(ur, _ur, nnodes * nparts * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(ut, _ut, nnodes * nparts * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(up, _up, nnodes * nparts * sizeof(real), cudaMemcpyDeviceToHost);
  printf("done.\n");

  for(i = 0; i < nnodes; i++) {
    printf("xx[%d] = %f yy[%d] = %f zz[%d] = %f\n", i, ur[i], i, ut[i], i, up[i]);
  }

  // write computed solution
  int rec_paraview_stepnum_out = 0;
  printf("\n  Writing summarized solution to: out_%d.interp...", rec_paraview_stepnum_out);
  char path[FILE_NAME_SIZE] = "";
  sprintf(path, "%s/output/out_%d.interp", ROOT_DIR, rec_paraview_stepnum_out);
  FILE *file = fopen(path, "w");
  if(file == NULL) {
    fprintf(stderr, "Could not open file out_%d.interp", rec_paraview_stepnum_out);
    exit(EXIT_FAILURE);
  }
  for(int part = 0; part < nparts; part++) {
    p_err_min[part] = FLT_MAX;
    p_err_max[part] = FLT_MIN;
    u_err_min[part] = FLT_MAX;
    u_err_max[part] = FLT_MIN;
    v_err_min[part] = FLT_MAX;
    v_err_max[part] = FLT_MIN;
    w_err_min[part] = FLT_MAX;
    w_err_max[part] = FLT_MIN;
    fprintf(file, "parts[%d].rs = %f\n", part, parts[part].rs);
    fprintf(file, "%11s%11s%13s%11s%11s%11s%11s\n",
      "theta", "phi", "expected", "p_err", "u_err", "v_err", "w_err");
    for(int n = 0; n < nnodes; n++) {
      real x_tmp = parts[part].rs*sin(node_t[n])*cos(node_p[n]) + parts[part].x;
      real y_tmp = parts[part].rs*sin(node_t[n])*sin(node_p[n]) + parts[part].y;
      real z_tmp = parts[part].rs*cos(node_t[n]) + parts[part].z;
      real pa_tmp = exp(x_tmp) + exp(y_tmp) + exp(z_tmp);
      real u_tmp = ur[n+part*nnodes]*sin(node_t[n])*cos(node_p[n]);
      u_tmp += ut[n+part*nnodes]*cos(node_t[n])*cos(node_p[n]);
      u_tmp -= up[n+part*nnodes]*sin(node_p[n]);
      real v_tmp = ur[n+part*nnodes]*sin(node_t[n])*sin(node_p[n]);
      v_tmp += ut[n+part*nnodes]*cos(node_t[n])*sin(node_p[n]);
      v_tmp += up[n+part*nnodes]*cos(node_p[n]);
      real w_tmp = ur[n+part*nnodes]*cos(node_t[n]);
      w_tmp -= ut[n+part*nnodes]*sin(node_t[n]);

      real p_out = (pa_tmp-pp[n+part*nnodes]) / pa_tmp;
      real u_out = (pa_tmp-u_tmp) / pa_tmp;
      real v_out = (pa_tmp-v_tmp) / pa_tmp;
      real w_out = (pa_tmp-w_tmp) / pa_tmp;

      if(fabs(p_out) < p_err_min[part]) p_err_min[part] = fabs(p_out);
      if(fabs(p_out) > p_err_max[part]) p_err_max[part] = fabs(p_out);
      if(fabs(u_out) < u_err_min[part]) u_err_min[part] = fabs(u_out);
      if(fabs(u_out) > u_err_max[part]) u_err_max[part] = fabs(u_out);
      if(fabs(v_out) < v_err_min[part]) v_err_min[part] = fabs(v_out);
      if(fabs(v_out) > v_err_max[part]) v_err_max[part] = fabs(v_out);
      if(fabs(w_out) < w_err_min[part]) w_err_min[part] = fabs(w_out);
      if(fabs(w_out) > w_err_max[part]) w_err_max[part] = fabs(w_out);

      fprintf(file, "%11.7f%11.7f%13.7f%11.3e%11.3e%11.3e%11.3e\n",
        node_t[n], node_p[n], pa_tmp, p_out, u_out, v_out, w_out);
    }
  }
  fclose(file);
  printf("done.\n");

  printf("\n  Error summary:\n");
  for(int a = 0; a < nparts; a++) {
    printf("  Particle %d\n", a);
    printf("    Field component:     minimum error:     maximum error:\n");
    printf("          p              %12.3e       %12.3e\n",
      p_err_min[a], p_err_max[a]);
    printf("          u              %12.3e       %12.3e\n",
      u_err_min[a], u_err_max[a]);
    printf("          v              %12.3e       %12.3e\n",
      v_err_min[a], v_err_max[a]);
    printf("          w              %12.3e       %12.3e\n\n",
      w_err_min[a], w_err_max[a]);
  }
  free(p_a);
  free(u_a);
  free(v_a);
  free(w_a);
  free(p_err_min);
  free(p_err_max);
  free(u_err_min);
  free(u_err_max);
  free(v_err_min);
  free(v_err_max);
  free(w_err_min);
  free(w_err_max);
  free(pp);
  free(ur);
  free(ut);
  free(up);
  cudaFree(_pp);
  cudaFree(_ur);
  cudaFree(_ut);
  cudaFree(_up);

}

extern "C"
void cuda_lamb_test(void)
{
  int i, j, k;  // iterators
  int C;    // cell locations
  real x, y, z;
  real r, theta, phi;
  real a = parts[0].r;

  real *p_a = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real)); // expected solution
  real *p_c = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real)); // computed solution
  real *p_e = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real)); // error difference
  real *u_a = (real*) malloc(dom[rank].Gfx.s3b * sizeof(real)); // expected solution
  real *u_c = (real*) malloc(dom[rank].Gfx.s3b * sizeof(real)); // computed solution
  real *u_e = (real*) malloc(dom[rank].Gfx.s3b * sizeof(real)); // error difference
  real *v_a = (real*) malloc(dom[rank].Gfy.s3b * sizeof(real)); // expected solution
  real *v_c = (real*) malloc(dom[rank].Gfy.s3b * sizeof(real)); // computed solution
  real *v_e = (real*) malloc(dom[rank].Gfy.s3b * sizeof(real)); // error difference
  real *w_a = (real*) malloc(dom[rank].Gfz.s3b * sizeof(real)); // expected solution
  real *w_c = (real*) malloc(dom[rank].Gfz.s3b * sizeof(real)); // computed solution
  real *w_e = (real*) malloc(dom[rank].Gfz.s3b * sizeof(real)); // error difference

  // min and max error search
  real p_err_min = FLT_MAX;
  real p_err_max = FLT_MIN;
  real u_err_min = FLT_MAX;
  real u_err_max = FLT_MIN;
  real v_err_min = FLT_MAX;
  real v_err_max = FLT_MIN;
  real w_err_min = FLT_MAX;
  real w_err_max = FLT_MIN;

  printf("\nLamb's coefficient calculation validation:\n\n");
  printf("  u = exp(x), v = exp(y), w = exp(z), ");
  printf("p = exp(x) + exp(y) + exp(z)\n\n");

  real U = 0.;
  real V = 0.;
  real W = 0.01;

  // set up expected solution
  for(k = dom[rank].Gcc.ksb; k <= dom[rank].Gcc.keb; k++) {
    for(j = dom[rank].Gcc.jsb; j <= dom[rank].Gcc.jeb; j++) {
      for(i = dom[rank].Gcc.isb; i <= dom[rank].Gcc.ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;// - parts[0].x;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;// - parts[0].y;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;// - parts[0].z;

        r = sqrt(x*x+y*y+z*z);
        theta = acos(z/r);
        phi = acos(x/sqrt(x*x+y*y));
        if(y<0.) phi = 2.*PI-phi;

        real st = sin(theta);
        real ct = cos(theta);
        real sp = sin(phi);
        real cp = cos(phi);

        p_a[C] = -1.5*(U*st*cp + V*st*sp + W*ct)*a/r/r;

        p[C] = p_a[C];
      }
    }
  }
 
  for(k = dom[rank].Gfx.ksb; k <= dom[rank].Gfx.keb; k++) {
    for(j = dom[rank].Gfx.jsb; j <= dom[rank].Gfx.jeb; j++) {
      for(i = dom[rank].Gfx.isb; i <= dom[rank].Gfx.ieb; i++) {
        C = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        x = (i-1.0)*dom[rank].dx + dom[rank].xs;// - parts[0].x;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;// - parts[0].y;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;// - parts[0].z;

        r = sqrt(x*x+y*y+z*z);
        theta = acos(z/r);
        phi = acos(x/sqrt(x*x+y*y));
        if(y<0.) phi = 2.*PI-phi;

        real st = sin(theta);
        real ct = cos(theta);
        real sp = sin(phi);
        real cp = cos(phi);

        u_a[C] = -0.75*a/r*(U + (U*st*cp + V*st*sp + W*ct)*st*cp)
          - 0.25*a*a*a/r/r/r*(U - 3.*(U*st*cp + V*st*sp + W*ct)*st*cp)
          + U;

        u[C] = u_a[C];
      }
    }
  }
  for(k = dom[rank].Gfy.ksb; k <= dom[rank].Gfy.keb; k++) {
    for(j = dom[rank].Gfy.jsb; j <= dom[rank].Gfy.jeb; j++) {
      for(i = dom[rank].Gfy.isb; i <= dom[rank].Gfy.ieb; i++) {
        C = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;// - parts[0].x;
        y = (j-1.0)*dom[rank].dy + dom[rank].ys;// - parts[0].y;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;// - parts[0].z;

        r = sqrt(x*x+y*y+z*z);
        theta = acos(z/r);
        phi = acos(x/sqrt(x*x+y*y));
        if(y<0.) phi = 2.*PI-phi;

        real st = sin(theta);
        real ct = cos(theta);
        real sp = sin(phi);
        real cp = cos(phi);

        v_a[C] = -0.75*a/r*(V + (U*st*cp + V*st*sp + W*ct)*st*sp)
          - 0.25*a*a*a/r/r/r*(V - 3.*(U*st*cp + V*st*sp + W*ct)*st*sp)
          + V;

        v[C] = v_a[C];
      }
    }
  }
  for(k = dom[rank].Gfz.ksb; k <= dom[rank].Gfz.keb; k++) {
    for(j = dom[rank].Gfz.jsb; j <= dom[rank].Gfz.jeb; j++) {
      for(i = dom[rank].Gfz.isb; i <= dom[rank].Gfz.ieb; i++) {
        C = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;// - parts[0].x;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;// - parts[0].y;
        z = (k-1.0)*dom[rank].dz + dom[rank].zs;// - parts[0].z;

        r = sqrt(x*x+y*y+z*z);
        theta = acos(z/r);
        phi = acos(x/sqrt(x*x+y*y));
        if(y<0.) phi = 2.*PI-phi;

        real st = sin(theta);
        real ct = cos(theta);
        real sp = sin(phi);
        real cp = cos(phi);

        w_a[C] = -0.75*a/r*(W + (U*st*cp + V*st*sp + W*ct)*ct)
          - 0.25*a*a*a/r/r/r*(W - 3.*(U*st*cp + V*st*sp + W*ct)*ct)
          + W;

        w[C] = w_a[C];
      }
    }
  }

  // write expected solution
  //rec_paraview_stepnum_out++;
  //printf("  Writing expected solution to: out_%d.pvtr...", rec_paraview_stepnum_out);
  //out_VTK();
  //printf("done.\n");

  // set up expected solution
  for(k = dom[rank].Gcc.ksb; k <= dom[rank].Gcc.keb; k++) {
    for(j = dom[rank].Gcc.jsb; j <= dom[rank].Gcc.jeb; j++) {
      for(i = dom[rank].Gcc.isb; i <= dom[rank].Gcc.ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;// - parts[0].x;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;// - parts[0].y;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;// - parts[0].z;

        r = sqrt(x*x+y*y+z*z);
        theta = acos(z/r);
        phi = acos(x/sqrt(x*x+y*y));
        if(y<0.) phi = 2.*PI-phi;

        real st = sin(theta);
        real ct = cos(theta);
        real sp = sin(phi);
        real cp = cos(phi);

        p_a[C] = -1.5*(U*st*cp + V*st*sp + W*ct)*a/r/r;

        p[C] = p_a[C];
      }
    }
  }
 
  for(k = dom[rank].Gfx.ksb; k <= dom[rank].Gfx.keb; k++) {
    for(j = dom[rank].Gfx.jsb; j <= dom[rank].Gfx.jeb; j++) {
      for(i = dom[rank].Gfx.isb; i <= dom[rank].Gfx.ieb; i++) {
        C = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        x = (i-1.0)*dom[rank].dx + dom[rank].xs;// - parts[0].x;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;// - parts[0].y;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;// - parts[0].z;

        r = sqrt(x*x+y*y+z*z);
        theta = acos(z/r);
        phi = acos(x/sqrt(x*x+y*y));
        if(y<0.) phi = 2.*PI-phi;

        real st = sin(theta);
        real ct = cos(theta);
        real sp = sin(phi);
        real cp = cos(phi);

        u_a[C] = -0.75*a/r*(U + (U*st*cp + V*st*sp + W*ct)*st*cp)
          - 0.25*a*a*a/r/r/r*(U - 3.*(U*st*cp + V*st*sp + W*ct)*st*cp)
          + U;

        u[C] = u_a[C];
      }
    }
  }
  for(k = dom[rank].Gfy.ksb; k <= dom[rank].Gfy.keb; k++) {
    for(j = dom[rank].Gfy.jsb; j <= dom[rank].Gfy.jeb; j++) {
      for(i = dom[rank].Gfy.isb; i <= dom[rank].Gfy.ieb; i++) {
        C = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;// - parts[0].x;
        y = (j-1.0)*dom[rank].dy + dom[rank].ys;// - parts[0].y;
        z = (k-0.5)*dom[rank].dz + dom[rank].zs;// - parts[0].z;

        r = sqrt(x*x+y*y+z*z);
        theta = acos(z/r);
        phi = acos(x/sqrt(x*x+y*y));
        if(y<0.) phi = 2.*PI-phi;

        real st = sin(theta);
        real ct = cos(theta);
        real sp = sin(phi);
        real cp = cos(phi);

        v_a[C] = -0.75*a/r*(V + (U*st*cp + V*st*sp + W*ct)*st*sp)
          - 0.25*a*a*a/r/r/r*(V - 3.*(U*st*cp + V*st*sp + W*ct)*st*sp)
          + V;

        v[C] = v_a[C];
      }
    }
  }
  for(k = dom[rank].Gfz.ksb; k <= dom[rank].Gfz.keb; k++) {
    for(j = dom[rank].Gfz.jsb; j <= dom[rank].Gfz.jeb; j++) {
      for(i = dom[rank].Gfz.isb; i <= dom[rank].Gfz.ieb; i++) {
        C = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        x = (i-0.5)*dom[rank].dx + dom[rank].xs;// - parts[0].x;
        y = (j-0.5)*dom[rank].dy + dom[rank].ys;// - parts[0].y;
        z = (k-1.0)*dom[rank].dz + dom[rank].zs;// - parts[0].z;

        r = sqrt(x*x+y*y+z*z);
        theta = acos(z/r);
        phi = acos(x/sqrt(x*x+y*y));
        if(y<0.) phi = 2.*PI-phi;

        real st = sin(theta);
        real ct = cos(theta);
        real sp = sin(phi);
        real cp = cos(phi);

        w_a[C] = -0.75*a/r*(W + (U*st*cp + V*st*sp + W*ct)*ct)
          - 0.25*a*a*a/r/r/r*(W - 3.*(U*st*cp + V*st*sp + W*ct)*ct)
          + W;

        w[C] = w_a[C];
      }
    }
  }

  // write initial fields (same as expected solution)
  //rec_paraview_stepnum_out++;
  //printf("  Writing initial fields to:    out_%d.pvtr...", rec_paraview_stepnum_out);
  //out_VTK();
  //printf("done.\n");

  // push fields to device
  printf("\n  Pushing fields to devices...");
  cuda_dom_push();
  printf("done.\n");

  // call code to test
  printf("  Running cuda_part_BC()...");
  cuda_lamb();
  cuda_part_BC();
  cuda_part_pull();
  //char nam[FILE_NAME_SIZE] = "lamb.rec";
  //recorder_lamb(nam,0);
  printf("done.\n");

  // pull fields back to host
  printf("  Pulling fields back to host...");
  //cuda_div_U();
  cuda_dom_pull();
  printf("done.\n");

  // write computed solution
  //rec_paraview_stepnum_out++;
  //printf("\n  Writing computed solution to: out_%d.pvtr...", rec_paraview_stepnum_out);
  //out_VTK();
  //printf("done.\n");

  // copy results and compute error
  for(k = dom[rank].Gcc.ksb; k <= dom[rank].Gcc.keb; k++) {
    for(j = dom[rank].Gcc.jsb; j <= dom[rank].Gcc.jeb; j++) {
      for(i = dom[rank].Gcc.isb; i <= dom[rank].Gcc.ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        p_c[C] = p[C];
        //if(p_c[C] != 0)
        //  p_e[C] = (p_c[C] - p_a[C]) / p_c[C];
        //else
          p_e[C] = (p_c[C] - p_a[C]);
        if(fabs(p_e[C]) > p_err_max) p_err_max = fabs(p_e[C]);
        if(fabs(p_e[C]) < p_err_min) p_err_min = fabs(p_e[C]);
        p[C] = p_e[C];
      }
    }
  }
  for(k = dom[rank].Gfx.ksb; k <= dom[rank].Gfx.keb; k++) {
    for(j = dom[rank].Gfx.jsb; j <= dom[rank].Gfx.jeb; j++) {
      for(i = dom[rank].Gfx.isb; i <= dom[rank].Gfx.ieb; i++) {
        C = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        u_c[C] = u[C];
        //if(u_c[C] != 0)
        //  u_e[C] = (u_c[C] - u_a[C]) / u_c[C];
        //else
          u_e[C] = (u_c[C] - u_a[C]);
        if(fabs(u_e[C]) > u_err_max) u_err_max = fabs(u_e[C]);
        if(fabs(u_e[C]) < u_err_min) u_err_min = fabs(u_e[C]);

        u[C] = u_e[C];
      }
    }
  }
  for(k = dom[rank].Gfy.ksb; k <= dom[rank].Gfy.keb; k++) {
    for(j = dom[rank].Gfy.jsb; j <= dom[rank].Gfy.jeb; j++) {
      for(i = dom[rank].Gfy.isb; i <= dom[rank].Gfy.ieb; i++) {
        C = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        v_c[C] = v[C];
        //if(v_c[C] != 0)
        //  v_e[C] = (v_c[C] - v_a[C]) / v_c[C];
        //else
          v_e[C] = (v_c[C] - v_a[C]);
        if(fabs(v_e[C]) > v_err_max) v_err_max = fabs(v_e[C]);
        if(fabs(v_e[C]) < v_err_min) v_err_min = fabs(v_e[C]);

        v[C] = v_e[C];
      }
    }
  }
  for(k = dom[rank].Gfz.ksb; k <= dom[rank].Gfz.keb; k++) {
    for(j = dom[rank].Gfz.jsb; j <= dom[rank].Gfz.jeb; j++) {
      for(i = dom[rank].Gfz.isb; i <= dom[rank].Gfz.ieb; i++) {
        C = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        w_c[C] = w[C];
        //if(w_c[C] != 0)
        //  w_e[C] = (w_c[C] - w_a[C]) / w_c[C];
        //else
          w_e[C] = (w_c[C] - w_a[C]);
        if(fabs(w_e[C]) > w_err_max) w_err_max = fabs(w_e[C]);
        if(fabs(w_e[C]) < w_err_min) w_err_min = fabs(w_e[C]);

        w[C] = w_e[C];
      }
    }
  }

  // write error difference
  //rec_paraview_stepnum_out++;
  //printf("  Writing error difference to:  out_%d.pvtr...", rec_paraview_stepnum_out);
  //out_VTK();
  //printf("done.\n");

  printf("\n  Error summary:\n");
  printf("  Field variable:     minimum error:     maximum error:\n");
  printf("          p              %12.3e       %12.3e\n",
    p_err_min, p_err_max);
  printf("          u              %12.3e       %12.3e\n",
    u_err_min, u_err_max);
  printf("          v              %12.3e       %12.3e\n",
    v_err_min, v_err_max);
  printf("          w              %12.3e       %12.3e\n\n",
    w_err_min, w_err_max);

  // clean up
  free(p_a);
  free(p_c);
  free(p_e);
  free(u_a);
  free(u_c);
  free(u_e);
  free(v_a);
  free(v_c);
  free(v_e);
  free(w_a);
  free(w_c);
  free(w_e);
}
real *us_a;
real *vs_a;
real *ws_a;
real *uc_a;
real *vc_a;
real *wc_a;
real *ud_a;
real *vd_a;
real *wd_a;

real *p_i;
real *u_i;
real *v_i;
real *w_i;

real u_star_err_min;
real u_star_err_max;
real conv_u_err_min;
real conv_u_err_max;
real diff_u_err_min;
real diff_u_err_max;

real v_star_err_min;
real v_star_err_max;
real conv_v_err_min;
real conv_v_err_max;
real diff_v_err_min;
real diff_v_err_max;

real w_star_err_min;
real w_star_err_max;
real conv_w_err_min;
real conv_w_err_max;
real diff_w_err_min;
real diff_w_err_max;

real p_err_min;
real u_err_min;
real v_err_min;
real w_err_min;
real p_err_max;
real u_err_max;
real v_err_max;
real w_err_max;

#endif // TEST
