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

#include "bluebottle.h"
#include "particle.h"

int NPARTS;
int nparts;
int nparts_subdom;
real volume_fraction;
real total_parts_mass;
real total_parts_volume;
real rho_avg;
real interaction_length_ratio;
real bin_size;
int ncoeffs_max;
int *flag_u, *flag_v, *flag_w;
int *_flag_u, *_flag_v, *_flag_w;
int *phase, *phase_shell;
int *_phase, *_phase_shell;
part_struct *parts;
part_struct *_parts;
bin_struct bins;
int *_bin_start;
int *_bin_end;
int *_bin_count;
int *_part_ind;
int *_part_bin;
int nparts_send[6];
int nparts_recv[6];

part_struct *_send_parts_e;
part_struct *_send_parts_w;
part_struct *_send_parts_n;
part_struct *_send_parts_s;
part_struct *_send_parts_t;
part_struct *_send_parts_b;

part_struct *_recv_parts_e;
part_struct *_recv_parts_w;
part_struct *_recv_parts_n;
part_struct *_recv_parts_s;
part_struct *_recv_parts_t;
part_struct *_recv_parts_b;

void parts_read_input(void)
{
  int i;  // iterator
  real rbuf;  // real buffer

  int fret = 0;
  fret = fret; // prevent compiler warning

  // open configuration file for reading
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/%s/part.config", ROOT_DIR, INPUT_DIR);
  FILE *infile = fopen(fname, "r");
  if (infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  // Read number of particles
  fret = fscanf(infile, "n %d\n", &NPARTS);

  // Read interaction length ratio
  fret = fscanf(infile, "(l/a) %lf\n", &interaction_length_ratio);


  // Init variables
  nparts = 0;

  real tmp_x, tmp_y, tmp_z;
  real tmp_u, tmp_v, tmp_w;
  real aFx, aFy, aFz;
  real aLx, aLy, aLz;
  real rho;
  real E;
  real sigma;
  real e_dry;
  real coeff_fric;
  int order;
  real rs_r;
  real spring_k;
  real spring_x, spring_y, spring_z;
  real spring_l;
  int translating;
  int rotating;

  real max_a = -1.;
  int check_x = 0;
  int check_y = 0;
  int check_z = 0;

  volume_fraction = 0.;
  total_parts_mass = 0.;
  total_parts_volume = 0.;
  rho_avg = 0.;

  // Read the config file once and determine nparts (local) and max radius
  if (NPARTS > 0) {

    for (i = 0; i < NPARTS; i++) {
      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "r %lf\n", &rbuf);

      // Maximum radius
      if (rbuf > max_a) max_a = rbuf;

      if (fret < 0) {
        printf("Something went wrong reading particle %d\n", i);
        exit(EXIT_FAILURE);
      }
      fret = fscanf(infile, "(x, y, z) %lf %lf %lf\n", &tmp_x, &tmp_y, &tmp_z);
      fret = fscanf(infile, "(u, v, w) %lf %lf %lf\n", &tmp_u, &tmp_v, &tmp_w);
      //if (fret == 0) {
      //  printf("part %d: (u, v, w) input error\n", i);
      //  exit(EXIT_FAILURE);
      //}
      fret = fscanf(infile, "(aFx, aFy, aFz) %lf %lf %lf\n", &aFx, &aFy, &aFz);
      fret = fscanf(infile, "(aLx, aLy, aLz) %lf %lf %lf\n", &aLx, &aLy, &aLz);
      fret = fscanf(infile, "rho %lf\n", &rho);
      fret = fscanf(infile, "E %lf\n", &E);
      fret = fscanf(infile, "sigma %lf\n", &sigma);
      fret = fscanf(infile, "e_dry %lf\n", &e_dry);
      fret = fscanf(infile, "coeff_fric %lf\n", &coeff_fric);
      fret = fscanf(infile, "order %d\n", &order);
      fret = fscanf(infile, "rs/r %lf\n", &rs_r);
      fret = fscanf(infile, "spring_k %lf\n", &spring_k);
      fret = fscanf(infile, "spring (x, y, z) %lf %lf %lf\n", &spring_x, &spring_y, &spring_z);
      fret = fscanf(infile, "spring_l %lf\n", &spring_l);
      fret = fscanf(infile, "translating %d\n", &translating);
      fret = fscanf(infile, "rotating %d\n", &rotating);

      // Particle total mass and volume
      real tmp = 4./3. * PI * rbuf * rbuf * rbuf;
      total_parts_volume += tmp;
      total_parts_mass += rho*tmp;

      // Check if particle center is within subdomain
      // PERIODIC:
      //  Subdomain range is [start, end)
      // ELSE:
      //  Subdomain range is [start, end) except last, which is [s, e]
      // x
      if (dom[rank].I == DOM.Ie && bc.pE != PERIODIC) {
        check_x = (tmp_x >= dom[rank].xs) && (tmp_x <= dom[rank].xe);
      } else {
        tmp_x -= DOM.xl * (tmp_x == DOM.xe); // flip
        check_x = (tmp_x >= dom[rank].xs) && (tmp_x < dom[rank].xe);
      }
      // y
      if (dom[rank].J == DOM.Je && bc.pN != PERIODIC) {
        check_y = (tmp_y >= dom[rank].ys) && (tmp_y <= dom[rank].ye);
      } else {
        tmp_y -= DOM.yl * (tmp_y == DOM.ye);
        check_y = (tmp_y >= dom[rank].ys) && (tmp_y < dom[rank].ye);
      }
      // z
      if (dom[rank].K == DOM.Ke && bc.pT != PERIODIC) {
        check_z = (tmp_z >= dom[rank].zs) && (tmp_z <= dom[rank].ze);
      } else {
        tmp_z -= DOM.zl * (tmp_z == DOM.ze);
        check_z = (tmp_z >= dom[rank].zs) && (tmp_z < dom[rank].ze);
      }

      if ((check_x == 1) && (check_y == 1) && (check_z == 1)) {
        nparts++;
      }

      // Reset check
      check_x = 0;
      check_y = 0;
      check_z = 0;
    }

    // Particle volume fraction, avg density
    volume_fraction = total_parts_volume / (DOM.xl * DOM.yl * DOM.zl);
    rho_avg = total_parts_mass / total_parts_volume * volume_fraction 
                + rho_f * (1. - volume_fraction);

    // Calculate particle interaction distance
    bin_size = 2.*max_a + interaction_length_ratio*max_a;

    /* Quit if the domain is not big enough
     *  NOTE: this is not a hard limit, but a design constraint such that a 
     *  particles cannot interact with the same particle twice through modeled
     *  hydrodynamic forces, e.g. lubrication. This would occur in a quasi-2-D
     *  domain with periodicity in the 3rd direction.
     *  It is possible to implement this correctly. A 2-D flag in the input flag
     *  would be necessary to specify which direction is the quasi-2D direction.
     *  Then, enforce that no motion or collision models take place in that
     *  direction, to constrain particle motion to the 2-D plane.
     *
     */
    if (rank == 0) {
      if (DOM.xl < 2. * bin_size && (bc.pW == PERIODIC || bc.pE == PERIODIC)) {
        printf("N%d >> Error: The domain size in the x-direction must contain at ", rank);
        printf("least two bins with periodic boundaries. The bin length is %lf\n", bin_size);
        printf("N%d >> Error: Please see %s:%d for information\n",
          rank, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      if (DOM.yl < 2. * bin_size && (bc.pS == PERIODIC || bc.pN == PERIODIC)) {
        printf("N%d >> Error: The domain size in the y-direction must contain at ", rank);
        printf("least two bins with periodic boundaries. The bin length is %lf\n", bin_size);
        printf("N%d >> Error: Please see %s:%d for information\n",
          rank, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
      if (DOM.zl < 2. * bin_size && (bc.pB == PERIODIC || bc.pT == PERIODIC)) {
        printf("N%d >> Error: The domain size in the z-direction must contain at ", rank);
        printf("least two bins with periodic boundaries. The bin length is %lf\n", bin_size);
        printf("N%d >> Error: Please see %s:%d for information\n",
          rank, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
    }


    // Check if number of particles matches that specified
    int local_sum = 0;
    MPI_Allreduce(&nparts, &local_sum, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
    if (local_sum != NPARTS) {
      printf("N%d has %d particles\n", rank, nparts);
      if (rank == 0) {
        printf("Total particles %d does not match parts placed in domains (%d)\n",
          NPARTS, local_sum);
      }
      exit(EXIT_FAILURE);
    }

  } // if (NPARTS > 0)
  fclose(infile);

  // Allocate particle structure on each subdomain
  parts = (part_struct*) malloc(nparts * sizeof(part_struct));
  cpumem += nparts * sizeof(part_struct);

  if (NPARTS > 0) {
    // Reread config file, this time filling information
    int ncount = 0;
    infile = fopen(fname, "r");

    fret = fscanf(infile, "n %d\n", &NPARTS);
    fret = fscanf(infile, "(l/a) %lf\n", &interaction_length_ratio);

    ncoeffs_max = -1;

    for (i = 0; i < NPARTS; i++) {
      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "r %lf\n", &rbuf);
      fret = fscanf(infile, "(x, y, z) %lf %lf %lf\n", &tmp_x, &tmp_y, &tmp_z);
      fret = fscanf(infile, "(u, v, w) %lf %lf %lf\n", &tmp_u, &tmp_v, &tmp_w);
      fret = fscanf(infile, "(aFx, aFy, aFz) %lf %lf %lf\n", &aFx, &aFy, &aFz);
      fret = fscanf(infile, "(aLx, aLy, aLz) %lf %lf %lf\n", &aLx, &aLy, &aLz);
      fret = fscanf(infile, "rho %lf\n", &rho);
      fret = fscanf(infile, "E %lf\n", &E);
      fret = fscanf(infile, "sigma %lf\n", &sigma);
      fret = fscanf(infile, "e_dry %lf\n", &e_dry);
      fret = fscanf(infile, "coeff_fric %lf\n", &coeff_fric);
      fret = fscanf(infile, "order %d\n", &order);
      fret = fscanf(infile, "rs/r %lf\n", &rs_r);
      fret = fscanf(infile, "spring_k %lf\n", &spring_k);
      fret = fscanf(infile, "spring (x, y, z) %lf %lf %lf\n", &spring_x, &spring_y, &spring_z);
      fret = fscanf(infile, "spring_l %lf\n", &spring_l);
      fret = fscanf(infile, "translating %d\n", &translating);
      fret = fscanf(infile, "rotating %d\n", &rotating);

      // Calculate max order (where all procs can get it)
      int ncoeffs = 0;
      for (int j = 0; j <= order; j++) {
        ncoeffs += j + 1;
      }
      if (ncoeffs > ncoeffs_max) ncoeffs_max = ncoeffs;


      // Check if particle center is within subdomain
      // PERIODIC:
      //  Subdomain range is [start, end)
      // ELSE:
      //  Subdomain range is [start, end) except last, which is [s, e]
      // x
      if (dom[rank].I == DOM.Ie && bc.pE != PERIODIC) {
        check_x = (tmp_x >= dom[rank].xs) && (tmp_x <= dom[rank].xe);
      } else {
        tmp_x -= DOM.xl * (tmp_x == DOM.xe); // flip
        check_x = (tmp_x >= dom[rank].xs) && (tmp_x < dom[rank].xe);
      }
      // y
      if (dom[rank].J == DOM.Je && bc.pN != PERIODIC) {
        check_y = (tmp_y >= dom[rank].ys) && (tmp_y <= dom[rank].ye);
      } else {
        tmp_y -= DOM.yl * (tmp_y == DOM.ye);
        check_y = (tmp_y >= dom[rank].ys) && (tmp_y < dom[rank].ye);
      }
      // z
      if (dom[rank].K == DOM.Ke && bc.pT != PERIODIC) {
        check_z = (tmp_z >= dom[rank].zs) && (tmp_z <= dom[rank].ze);
      } else {
        tmp_z -= DOM.zl * (tmp_z == DOM.ze);
        check_z = (tmp_z >= dom[rank].zs) && (tmp_z < dom[rank].ze);
      }

      if ((check_x == 1) && (check_y == 1) && (check_z == 1)) {
        parts[ncount].N = i;
        parts[ncount].r = rbuf;
        parts[ncount].x = tmp_x;
        parts[ncount].y = tmp_y;
        parts[ncount].z = tmp_z;
        parts[ncount].u = tmp_u;
        parts[ncount].v = tmp_v;
        parts[ncount].w = tmp_w;
        parts[ncount].aFx = aFx;
        parts[ncount].aFy = aFy;
        parts[ncount].aFz = aFz;
        parts[ncount].aLx = aLx;
        parts[ncount].aLy = aLy;
        parts[ncount].aLz = aLz;
        parts[ncount].rho = rho;
        parts[ncount].E = E;
        parts[ncount].sigma = sigma;
        parts[ncount].e_dry = e_dry;
        parts[ncount].coeff_fric = coeff_fric;
        parts[ncount].order = order;
        parts[ncount].rs = rs_r;
        parts[ncount].spring_k = spring_k;
        parts[ncount].spring_x = spring_x;
        parts[ncount].spring_y = spring_y;
        parts[ncount].spring_z = spring_z;
        parts[ncount].spring_l = spring_l;
        parts[ncount].translating = translating;
        parts[ncount].rotating = rotating;
        ncount++;
      }

      // Reset check
      check_x = 0;
      check_y = 0;
      check_z = 0;

    }
    fclose(infile);
  } // if (NPARTS > 0)
}

void parts_init(void)
{
  // Reset flag and phase fields
  flags_reset();

  for (int i = 0; i < nparts; i++) {
    parts[i].rs = parts[i].rs * parts[i].r;

    // calculate number of coeffs needed
    parts[i].ncoeff = 0;
    for (int j = 0; j <= parts[i].order; j++) {
      parts[i].ncoeff += j + 1;
    }

    // Init previous velocity
    parts[i].u0 = parts[i].u;
    parts[i].v0 = parts[i].v;
    parts[i].w0 = parts[i].w;

    // Init acceleration to zero (default: QUIESCENT)
    parts[i].udot = 0.;
    parts[i].vdot = 0.;
    parts[i].wdot = 0.;
    parts[i].udot0 = 0.;
    parts[i].vdot0 = 0.;
    parts[i].wdot0 = 0.;

    // Set initial particle reference to match global reference 
    parts[i].axx = 1.;
    parts[i].axy = 0.;
    parts[i].axz = 0.;
    parts[i].ayx = 0.;
    parts[i].ayy = 1.;
    parts[i].ayz = 0.;
    parts[i].azx = 0.;
    parts[i].azy = 0.;
    parts[i].azz = 1.;

    parts[i].ox = 0.;
    parts[i].oy = 0.;
    parts[i].oz = 0.;
    parts[i].ox0 = 0.;
    parts[i].oy0 = 0.;
    parts[i].oz0 = 0.;
    parts[i].oxdot = 0.;
    parts[i].oydot = 0.;
    parts[i].ozdot = 0.;
    parts[i].oxdot0 = 0.;
    parts[i].oydot0 = 0.;
    parts[i].ozdot0 = 0.;

    if (init_cond == SHEAR) { // Init parts for shear flow
      if (parts[i].translating) {
        // Set linear velocity according to previous position
        parts[i].u = (bc.uNDm - bc.uSDm) * (parts[i].y - DOM.ys)/DOM.yl + bc.uSDm;
        parts[i].u += (bc.uTDm - bc.uBDm) * (parts[i].z - DOM.zs)/DOM.zl + bc.uBDm;
        parts[i].u0 = parts[i].u;

        parts[i].v = (bc.vEDm - bc.vWDm) * (parts[i].x - DOM.xs)/DOM.xl + bc.vWDm;
        parts[i].v += (bc.vTDm - bc.vBDm) * (parts[i].z - DOM.zs)/DOM.zl + bc.vBDm;
        parts[i].v0 = parts[i].v;

        parts[i].w = (bc.wEDm - bc.wWDm) * (parts[i].x - DOM.xs)/DOM.xl + bc.wWDm;
        parts[i].w += (bc.wNDm - bc.wSDm) * (parts[i].y - DOM.ys)/DOM.yl + bc.wSDm;
        parts[i].w0 = parts[i].w;
      }
      if (parts[i].rotating) {
        parts[i].ox = -0.5*(bc.vTDm - bc.vBDm)/DOM.zl;
        parts[i].ox += 0.5*(bc.wNDm - bc.wSDm)/DOM.yl;
        parts[i].oy = 0.5*(bc.uTDm - bc.uBDm)/DOM.zl;
        parts[i].oy += -0.5*(bc.wEDm - bc.wWDm)/DOM.xl;
        parts[i].oz = -0.5*(bc.uNDm - bc.uSDm)/DOM.yl;
        parts[i].oz += 0.5*(bc.vEDm - bc.vWDm)/DOM.xl;
        parts[i].ox0 = parts[i].ox;
        parts[i].oy0 = parts[i].oy;
        parts[i].oz0 = parts[i].oz;
      }

    } else if (init_cond == CHANNEL) { // Init for channel flow
      if (parts[i].translating) {
        // set linear velocity according to position
        real x = parts[i].x;
        real y = parts[i].y;
        real z = parts[i].z;
        parts[i].u = 0.5/mu*gradP.xm*(y*y - (DOM.ys + DOM.ye)*y + DOM.ys*DOM.ye)
          * (bc.uS == DIRICHLET)
          + 0.5/mu*gradP.xm*(z*z - (DOM.zs + DOM.ze)*z + DOM.zs*DOM.ze)
          * (bc.uB == DIRICHLET);
        parts[i].v = 0.5/mu*gradP.ym*(x*x - (DOM.xs + DOM.xe)*x + DOM.xs*DOM.xe)
          * (bc.vW == DIRICHLET)
          + 0.5/mu*gradP.ym*(z*z - (DOM.zs + DOM.ze)*z + DOM.zs*DOM.ze)
          * (bc.vB == DIRICHLET);
        parts[i].w = 0.5/mu*gradP.zm*(x*x - (DOM.xs + DOM.xe)*x + DOM.xs*DOM.xe)
          * (bc.wW == DIRICHLET)
          + 0.5/mu*gradP.zm*(y*y - (DOM.ys + DOM.ye)*y + DOM.ys*DOM.ye)
          * (bc.wS == DIRICHLET);

        parts[i].u0 = parts[i].u;
        parts[i].v0 = parts[i].v;
        parts[i].w0 = parts[i].w;
      }
      // initialize no rotation component
    }

    // Forces and moments to zero
    parts[i].Fx = 0.;
    parts[i].Fy = 0.;
    parts[i].Fz = 0.;
    parts[i].Lx = 0.;
    parts[i].Ly = 0.;
    parts[i].Lz = 0.;

    // initialize the particle spring force to zero
    parts[i].kFx = 0.;
    parts[i].kFy = 0.;
    parts[i].kFz = 0.;

    // initialize the particle interaction force to zero
    parts[i].iFx = 0.;
    parts[i].iFy = 0.;
    parts[i].iFz = 0.;
    parts[i].iLx = 0.;
    parts[i].iLy = 0.;
    parts[i].iLz = 0.;

    // initialize nodes array
    for (int j = 0; j < NNODES; j++) {
      parts[i].nodes[j] = -1;
    }

    // initialize Stokes number lists to -1
    for (int j = 0; j < MAX_NEIGHBORS; j++) {
      parts[i].St[j] = 0.;
      parts[i].iSt[j] = -1;
    }

    // initialize Lamb's coefficients
    for (int j = 0; j < MAX_COEFFS; j++) {
      parts[i].pnm_re[j] = 0.;
      parts[i].pnm_im[j] = 0.;
      parts[i].phinm_re[j] = 0.;
      parts[i].phinm_im[j] = 0.;
      parts[i].chinm_re[j] = 0.;
      parts[i].chinm_im[j] = 0.;

      parts[i].pnm_re0[j] = 0.;
      parts[i].pnm_im0[j] = 0.;
      parts[i].phinm_re0[j] = 0.;
      parts[i].phinm_im0[j] = 0.;
      parts[i].chinm_re0[j] = 0.;
      parts[i].chinm_im0[j] = 0.;
    }

    // initialize collision counter
    parts[i].ncoll_part = 0;
    parts[i].ncoll_wall = 0;
  }

  #ifdef DDEBUG
    parts_print();
  #endif // DDEBUG
}

void flags_reset(void)
{
  int i, j, k;
  int C;
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        phase[C] = -1;       // all fluid
        phase_shell[C] = -1; // all fluid
      }
    }
  } 
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        C = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        flag_u[C] = 1;       // all fluid
      }
    }
  } 
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        C = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        flag_v[C] = 1;       // all fluid
      }
    }
  } 
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        C = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        flag_w[C] = 1;       // all fluid
      }
    }
  } 
}

void parts_print(void)
{
  // Prep file for output -- one per rank
  char fname[CHAR_BUF_SIZE];
  sprintf(fname, "%s/rank-%d-parts.debug", ROOT_DIR, rank);
  FILE *outfile = fopen(fname, "w");

  for (int n = 0; n < nparts; n++) {
    fprintf(outfile, "parts[%d].N = %d\n", n, parts[n].N);
    fprintf(outfile, "parts[%d].r = %lf\n", n, parts[n].r);
    fprintf(outfile, "parts[%d].{x,y,z} = (%lf, %lf, %lf)\n", n,
      parts[n].x, parts[n].y, parts[n].z);
    fprintf(outfile, "parts[%d].{u,v,w} = (%lf, %lf, %lf)\n", n,
      parts[n].u, parts[n].v, parts[n].w);
    fprintf(outfile, "parts[%d].{u0,v0,w0} = (%lf, %lf, %lf)\n", n,
      parts[n].u0, parts[n].v0, parts[n].w0);
    fprintf(outfile, "parts[%d].{udot,vdot,wdot} = (%lf, %lf, %lf)\n", n,
      parts[n].udot, parts[n].vdot, parts[n].wdot);
    fprintf(outfile, "parts[%d].{udot0,vdot0,wdot0} = (%lf, %lf, %lf)\n", n,
      parts[n].udot0, parts[n].vdot0, parts[n].wdot0);
    fprintf(outfile, "parts[%d].{axx,axy,axz} = (%lf, %lf, %lf)\n", n,
      parts[n].axx, parts[n].axy, parts[n].axz);
    fprintf(outfile, "parts[%d].{ayx,ayy,ayz} = (%lf, %lf, %lf)\n", n,
      parts[n].ayx, parts[n].ayy, parts[n].ayz);
    fprintf(outfile, "parts[%d].{azx,azy,azz} = (%lf, %lf, %lf)\n", n,
      parts[n].axx, parts[n].azy, parts[n].azz);
    fprintf(outfile, "parts[%d].{ox,oy,oz} = (%lf, %lf, %lf)\n", n,
      parts[n].ox, parts[n].oy, parts[n].oz);
    fprintf(outfile, "parts[%d].{ox0,oy0,oz0} = (%lf, %lf, %lf)\n", n,
      parts[n].ox0, parts[n].oy0, parts[n].oz0);
    fprintf(outfile, "parts[%d].{oxdot,oydot,ozdot} = (%lf, %lf, %lf)\n", n,
      parts[n].oxdot, parts[n].oydot, parts[n].ozdot);
    fprintf(outfile, "parts[%d].{oxdot0,oydot0,ozdot0} = (%lf, %lf, %lf)\n", n,
      parts[n].oxdot0, parts[n].oydot0, parts[n].ozdot0);
    fprintf(outfile, "parts[%d].{Fx,Fy,Fz} = (%lf, %lf, %lf)\n", n,
      parts[n].Fx, parts[n].Fy, parts[n].Fz);
    fprintf(outfile, "parts[%d].{Lx,Ly,Lz} = (%lf, %lf, %lf)\n", n,
      parts[n].Lx, parts[n].Ly, parts[n].Lz);
    fprintf(outfile, "parts[%d].{aFx,aFy,aFz} = (%lf, %lf, %lf)\n", n,
      parts[n].aFx, parts[n].aFy, parts[n].aFz);
    fprintf(outfile, "parts[%d].{aLx,aLy,aLz} = (%lf, %lf, %lf)\n", n,
      parts[n].aLx, parts[n].aLy, parts[n].aLz);
    fprintf(outfile, "parts[%d].{kFx,kFy,kFz} = (%lf, %lf, %lf)\n", n,
      parts[n].kFx, parts[n].kFy, parts[n].kFz);
    fprintf(outfile, "parts[%d].{iFx,iFy,iFz} = (%lf, %lf, %lf)\n", n,
      parts[n].iFx, parts[n].iFy, parts[n].iFz);
    fprintf(outfile, "parts[%d].{iLx,iLy,iLz} = (%lf, %lf, %lf)\n", n,
      parts[n].iLx, parts[n].iLy, parts[n].iLz);
    fprintf(outfile, "parts[%d].nodes:\n\t", n);
    for (int i = 0; i < NNODES; i++) {
      fprintf(outfile, " %d", parts[n].nodes[i]);
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "parts[%d].rho = %lf\n", n, parts[n].rho);
    fprintf(outfile, "parts[%d].E = %lf\n", n, parts[n].E);
    fprintf(outfile, "parts[%d].sigma = %lf\n", n, parts[n].sigma);
    fprintf(outfile, "parts[%d].order = %d\n", n, parts[n].order);
    fprintf(outfile, "parts[%d].rs = %lf\n", n, parts[n].rs);
    fprintf(outfile, "parts[%d].ncoeff = %d\n", n, parts[n].ncoeff);
    fprintf(outfile, "parts[%d].spring_k = %lf\n", n, parts[n].spring_k);
    fprintf(outfile, "parts[%d].spring_l = %lf\n", n, parts[n].spring_l);
    fprintf(outfile, "parts[%d].spring_{x,y,z} = %lf, %lf, %lf\n", n,
      parts[n].spring_x, parts[n].spring_y, parts[n].spring_z);
    fprintf(outfile, "parts[%d].translating = %d\n", n, parts[n].translating);
    fprintf(outfile, "parts[%d].rotating = %d\n", n, parts[n].rotating);
    fprintf(outfile, "parts[%d].St:\n\t", n);
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
      fprintf(outfile, " %lf", parts[n].St[i]);
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "parts[%d].iSt:\n\t", n);
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
      fprintf(outfile, " %d", parts[n].iSt[i]);
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "parts[%d].e_dry = %lf\n", n, parts[n].e_dry);
    fprintf(outfile, "parts[%d].coeff_fric = %lf\n", n, parts[n].coeff_fric);

    fprintf(outfile, "parts[%d].pnm_re:\n\t", n);
    for (int i = 0; i < MAX_COEFFS; i++) {
      fprintf(outfile, " %lf", parts[n].pnm_re[i]);
    }
    fprintf(outfile, "\n");

    fprintf(outfile, "parts[%d].pnm_im:\n\t", n);
    for (int i = 0; i < MAX_COEFFS; i++) {
      fprintf(outfile, " %lf", parts[n].pnm_im[i]);
    }
    fprintf(outfile, "\n");

    fprintf(outfile, "parts[%d].phinm_re:\n\t", n);
    for (int i = 0; i < MAX_COEFFS; i++) {
      fprintf(outfile, " %lf", parts[n].phinm_re[i]);
    }
    fprintf(outfile, "\n");

    fprintf(outfile, "parts[%d].phinm_im:\n\t", n);
    for (int i = 0; i < MAX_COEFFS; i++) {
      fprintf(outfile, " %lf", parts[n].phinm_im[i]);
    }
    fprintf(outfile, "\n");

    fprintf(outfile, "parts[%d].chinm_re:\n\t", n);
    for (int i = 0; i < MAX_COEFFS; i++) {
      fprintf(outfile, " %lf", parts[n].chinm_re[i]);
    }
    fprintf(outfile, "\n");

    fprintf(outfile, "parts[%d].chinm_im:\n\t", n);
    for (int i = 0; i < MAX_COEFFS; i++) {
      fprintf(outfile, " %lf", parts[n].chinm_im[i]);
    }
    fprintf(outfile, "\n");
    
  }
  fclose(outfile);
}

void init_bins(void)
{
  // Fill in bins based on dom
  bins.xs = dom[rank].xs;
  bins.xe = dom[rank].xe;
  bins.xl = dom[rank].xl;
  bins.xn = floor(dom[rank].xl / bin_size);
  if (bins.xn == 0) bins.xn = 1; // avoid divide by zero
  bins.dx = dom[rank].xl / bins.xn;

  bins.ys = dom[rank].ys;
  bins.ye = dom[rank].ye;
  bins.yl = dom[rank].yl;
  bins.yn = floor(dom[rank].yl / bin_size);
  if (bins.yn == 0) bins.yn = 1; // avoid divide by zero
  bins.dy = dom[rank].yl / bins.yn;

  bins.zs = dom[rank].zs;
  bins.ze = dom[rank].ze;
  bins.zl = dom[rank].zl;
  bins.zn = floor(dom[rank].zl / bin_size);
  if (bins.zn == 0) bins.zn = 1; // avoid divide by zero
  bins.dz = dom[rank].zl / bins.zn;

  /* Exit if periodic edge domains aren't large enough. This prevents particles
   * from having more than one periodic image in a given direction, which breaks
   * the Lebedev quadrature partial summation accumalation communication.
   * Additionally, having domains that small is incredibly inefficient.
   */
  if (bins.xn < 2) {
    printf("N%d >> Error: Subdomains must contain at least two bins. The "
           "bin size is %lf\nN%d >>   and the subdomain length is xl = %lf\n",
           rank, bin_size, rank, dom[rank].xl);
    exit(EXIT_FAILURE);
  }

  if (bins.yn < 2) {
    printf("N%d >> Error: Subdomains must contain at least two bins. The "
           "bin size is %lf\nN%d >>   and the subdomain length is yl = %lf\n",
           rank, bin_size, rank, dom[rank].yl);
    exit(EXIT_FAILURE);
  }

  if (bins.zn < 2) {
    printf("N%d >> Error: Subdomains must contain at least two bins. The "
           "bin size is %lf\nN%d >>   and the subdomain length is zl = %lf\n",
           rank, bin_size, rank, dom[rank].zl);
    exit(EXIT_FAILURE);
  }

  // Local bin indexing -- global (probably) not needed
  // Gcc
  bins.Gcc._is = DOM_BUF;
  bins.Gcc._isb = bins.Gcc._is - DOM_BUF;
  bins.Gcc.in = bins.xn;
  bins.Gcc.inb = bins.xn + 2 * DOM_BUF;
  bins.Gcc._ie = bins.Gcc._isb + bins.Gcc.in;
  bins.Gcc._ieb = bins.Gcc._ie + DOM_BUF;

  bins.Gcc._js = DOM_BUF;
  bins.Gcc._jsb = bins.Gcc._js - DOM_BUF;
  bins.Gcc.jn = bins.yn;
  bins.Gcc.jnb = bins.yn + 2 * DOM_BUF;
  bins.Gcc._je = bins.Gcc._jsb + bins.Gcc.jn;
  bins.Gcc._jeb = bins.Gcc._je + DOM_BUF;

  bins.Gcc._ks = DOM_BUF;
  bins.Gcc._ksb = bins.Gcc._ks - DOM_BUF;
  bins.Gcc.kn = bins.zn;
  bins.Gcc.knb = bins.zn + 2 * DOM_BUF;
  bins.Gcc._ke = bins.Gcc._ksb + bins.Gcc.kn;
  bins.Gcc._keb = bins.Gcc._ke + DOM_BUF;

  bins.Gcc.s1 = bins.Gcc.in;
  bins.Gcc.s2 = bins.Gcc.s1 * bins.Gcc.jn;
  bins.Gcc.s3 = bins.Gcc.s2 * bins.Gcc.kn;
  bins.Gcc.s1b = bins.Gcc.inb;
  bins.Gcc.s2b = bins.Gcc.s1b * bins.Gcc.jnb;
  bins.Gcc.s3b = bins.Gcc.s2b * bins.Gcc.knb;

  bins.Gcc.s2_i = bins.Gcc.jn * bins.Gcc.kn;
  bins.Gcc.s2_j = bins.Gcc.in * bins.Gcc.kn;
  bins.Gcc.s2_k = bins.Gcc.in * bins.Gcc.jn;
  bins.Gcc.s2b_i = bins.Gcc.jnb * bins.Gcc.knb;
  bins.Gcc.s2b_j = bins.Gcc.inb * bins.Gcc.knb;
  bins.Gcc.s2b_k = bins.Gcc.inb * bins.Gcc.jnb;

  #ifdef DDEBUG
    bin_write_config();
  #endif // DDEBUG
}

void bin_write_config(void)
{
  char fname[CHAR_BUF_SIZE];
  sprintf(fname, "%s/rank-%d-map.debug", ROOT_DIR, rank);
  FILE *outfile = fopen(fname, "a");
  if (outfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  fprintf(outfile, "\n\nBin Domain:\n");
  fprintf(outfile, "  bin size = %f\n", bin_size);
  fprintf(outfile, "  X: (%f, %f), dx = %f\n", bins.xs, bins.xe, bins.dx);
  fprintf(outfile, "  Y: (%f, %f), dy = %f\n", bins.ys, bins.ye, bins.dy);
  fprintf(outfile, "  Z: (%f, %f), dz = %f\n", bins.zs, bins.ze, bins.dz);
  fprintf(outfile, "  Xn = %d, Yn = %d, Zn = %d\n", bins.xn, bins.yn, bins.zn);
  fprintf(outfile, "Bin Grids:\n");
  fprintf(outfile, "  bins.Gcc:\n");
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", bins.Gcc._is, bins.Gcc._ie,
    bins.Gcc.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", bins.Gcc._isb,
    bins.Gcc._ieb, bins.Gcc.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", bins.Gcc._js, bins.Gcc._je,
    bins.Gcc.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", bins.Gcc._jsb,
    bins.Gcc._jeb, bins.Gcc.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", bins.Gcc._ks, bins.Gcc._ke,
    bins.Gcc.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", bins.Gcc._ksb,
    bins.Gcc._keb, bins.Gcc.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", bins.Gcc.s1, bins.Gcc.s2,
    bins.Gcc.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", bins.Gcc.s1b,
    bins.Gcc.s2b, bins.Gcc.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", bins.Gcc.s2_i,
    bins.Gcc.s2_j, bins.Gcc.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", bins.Gcc.s2b_i,
    bins.Gcc.s2b_j, bins.Gcc.s2b_k);

  fclose(outfile);
}

void part_free(void)
{
  free(parts);
}
