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

#include "seeder.h"

// Variable declaration at bottom of file

int main(int argc, char *argv[])
{
  /* Parse command line inputs */
  parse_cmdline(argc, argv);

  /* Parse part.config file for seeder parameters */
  seeder_read_input();

  /* Parse flow.config file for domain size */
  domain_read_input();

  /* Seed! */
  if (array == 1) {
    seed_array();
  } else {
    seeder();
  }
}

void parse_cmdline(int argc, char *argv[])
{
  // Parse command line arguments -- K&R l117
  rand_u = 0;
  array = 0;
  int argin;
  while (--argc > 0 && (*++argv)[0] == '-') {
    while ((argin = *++argv[0])) {
      switch (argin) {
        case 'u':
          rand_u = 1;
          break;
        case 'a':
          array = 1;
          break;
        default:
          printf("seeder: illegal option %c\n", argin);
          argc = 0;
          break;
      }
    }
  }
  if (argc != 1) {
    printf("Usage: seeder [-u] [-a] ./path/to/part.config\n");
    printf("Randomly seed particles according to the information in "
           "./path/to/part.config\n\n");
    printf("Options");
    printf("  -u              Randomize particle velocity according to the "
      "given\n\t\t\t scale, which is prompted by the program. Note that\n\t\t\t"
      " this is only currently useful for debug, since the\n\t\t\t velocity "
      "isn't necessarily set intelligently\n");
    printf("  -a              Seed particles in an array according to the given"
      " array\n\t\t\tof particles, which is prompted by the program.\n");
    exit(EXIT_FAILURE);
  } else {
    sprintf(part_file, "%s", argv[0]);
  }

  // Prompt for special options
  if (rand_u == 1) {
    printf("  Input velocity scale: ");
    fflush(stdout);
    scanf("%lf", &vel_scale);
    printf("\n");
  } else if (array == 1) {
    printf("  Input particles in x direction : ");
    fflush(stdout);
    scanf("%d", &nx);
    printf("\n");

    printf("  Input particles in y direction : ");
    fflush(stdout);
    scanf("%d", &ny);
    printf("\n");

    printf("  Input particles in z direction : ");
    fflush(stdout);
    scanf("%d", &nz);
    printf("\n");
  }
}

void seeder_read_input(void)
{
  // Read part.config to get parameter set
  int fret = 0;
  fret = fret;  // prevent compiler warning
  
  FILE *infile = fopen(part_file, "r");
  if (infile == NULL) {
    fprintf(stderr, "Could not open file %s on line %d\n", part_file, __LINE__);
    exit(EXIT_FAILURE);
  }

  // Read particle list and get parameters
  fret = fscanf(infile, "n %d\n", &nparts);
  fret = fscanf(infile, "(l/a) %lf\n", &loa);

  fret = fscanf(infile, "r %lf\n", &a);
  fret = fscanf(infile, "(x, y, z) %lf %lf %lf\n", &x, &y, &z);
  fret = fscanf(infile, "(u, v, w) %lf %lf %lf\n", &u, &v, &w);
  fret = fscanf(infile, "(aFx, aFy, aFz) %lf %lf %lf\n", &aFx, &aFy, &aFz);
  fret = fscanf(infile, "(aLx, aLy, aLz) %lf %lf %lf\n", &aLx, &aLy, &aLz);
  fret = fscanf(infile, "rho %lf\n", &rho);
  fret = fscanf(infile, "E %lf\n", &E);
  fret = fscanf(infile, "sigma %lf\n", &sigma);
  fret = fscanf(infile, "e_dry %lf\n", &e_dry);
  fret = fscanf(infile, "coeff_fric %lf\n", &coeff_fric);
  fret = fscanf(infile, "order %d\n", &order);
  fret = fscanf(infile, "rs/r %lf\n", &rs);
  fret = fscanf(infile, "spring_k %lf\n", &spring_k);
  fret = fscanf(infile, "spring (x, y, z) %lf %lf %lf\n", &spring_x, &spring_y, &spring_z);
  fret = fscanf(infile, "spring_l %lf\n", &spring_l);
  fret = fscanf(infile, "translating %d\n", &translating);
  fret = fscanf(infile, "rotating %d\n", &rotating);

  // Check input
  int check = 0;
  if (nparts < 1) {
    printf("Error: nparts must be greater than 1\n");
    check++;
  } else if (a < 0) {
    printf("Error: radius must be greater than 0\n");
    check++;
  }

  if (check > 0) {
    printf("Found %d errors in part.config. Please correct and try again\n",
      check);
    exit(EXIT_FAILURE);
  }

  fclose(infile); 
}

void domain_read_input(void)
{
  int fret = 0;
  fret = fret; // prevent compiler warning

  // Find flow.config from part.config
  char tmp_name[FILE_NAME_SIZE];  // since dirname destroys input
  sprintf(tmp_name, "%s", part_file);
  sprintf(INPUT_DIR, "%s", dirname(tmp_name));

  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/%s", INPUT_DIR, "flow.config");

  /* open flow.config file for reading */
  FILE *infile = fopen(fname, "r");
  if (infile == NULL) {
    fprintf(stderr, "Could not open file %s on line %d\n", fname, __LINE__);
    exit(EXIT_FAILURE);
  }

  /* create trash buffers for info we don't care about */
  double fbuf;
  char sbuf[FILE_NAME_SIZE] = "";  // character read buffer

  /* read global domain */
  fret = fscanf(infile, "GLOBAL DOMAIN\n");
  fret = fscanf(infile, "(Xs, Xe, Xn) %lf %lf %lf\n", &dom.xs, &dom.xe, &fbuf);
  fret = fscanf(infile, "(Ys, Ye, Yn) %lf %lf %lf\n", &dom.ys, &dom.ye, &fbuf);
  fret = fscanf(infile, "(Zs, Ze, Zn) %lf %lf %lf\n", &dom.zs, &dom.ze, &fbuf);
  fret = fscanf(infile, "\n");

  dom.xl = dom.xe - dom.xs;
  dom.yl = dom.ye - dom.ys;
  dom.zl = dom.ze - dom.zs;

  fret = fscanf(infile, "MPI/GPU SUBDOMAIN DECOMPOSITION\n");
  fret = fscanf(infile, "(In, Jn, Kn) %lf %lf %lf\n", &fbuf, &fbuf, &fbuf);
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "PHYSICAL PARAMETERS\n");
  fret = fscanf(infile, "rho_f %lf\n", &fbuf);
  fret = fscanf(infile, "nu %lf\n", &fbuf);
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "SIMULATION PARAMETERS\n");
  fret = fscanf(infile, "duration %lf\n", &fbuf);
  fret = fscanf(infile, "CFL %lf\n", &fbuf);
  fret = fscanf(infile, "pp_max_iter %lf\n", &fbuf);
  fret = fscanf(infile, "pp_residual %lf\n", &fbuf);
  fret = fscanf(infile, "lamb_max_iter %lf\n", &fbuf);
  fret = fscanf(infile, "lamb_residual %lf\n", &fbuf);
  fret = fscanf(infile, "lamb_relax %lf\n", &fbuf);
  fret = fscanf(infile, "lamb_cut %lf\n", &fbuf);
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "BOUNDARY CONDITIONS\n");
  fret = fscanf(infile, "v_bc_tdelay %lf\n", &fbuf);
  fret = fscanf(infile, "PRESSURE\n");
  fret = fscanf(infile, "bc.pW %s", sbuf);

  if (strcmp(sbuf, "PERIODIC") == 0) {
    dom.bc_w = PERIODIC;
    dom.ds_w = 0;
  } else if(strcmp(sbuf, "NEUMANN") == 0) {
    dom.bc_w = NEUMANN;
    fret = fscanf(infile, "%lf", &dom.ds_w);
  } else {
    fprintf(stderr, "flow.config read error (%s:%d).\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.pE %s", sbuf);
  if (strcmp(sbuf, "PERIODIC") == 0) {
    dom.bc_e = PERIODIC;
    dom.ds_e = 0;
  } else if(strcmp(sbuf, "NEUMANN") == 0) {
    dom.bc_e = NEUMANN;
    fret = fscanf(infile, "%lf", &dom.ds_e);
  } else {
    fprintf(stderr, "flow.config read error (%s:%d).\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.pS %s", sbuf);
  if (strcmp(sbuf, "PERIODIC") == 0) {
    dom.bc_s = PERIODIC;
    dom.ds_s = 0;
  } else if(strcmp(sbuf, "NEUMANN") == 0) {
    dom.bc_s = NEUMANN;
    fret = fscanf(infile, "%lf", &dom.ds_s);
  } else {
    fprintf(stderr, "flow.config read error (%s:%d).\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.pN %s", sbuf);
  if (strcmp(sbuf, "PERIODIC") == 0) {
    dom.bc_n = PERIODIC;
    dom.ds_n = 0;
  } else if(strcmp(sbuf, "NEUMANN") == 0) {
    dom.bc_n = NEUMANN;
    fret = fscanf(infile, "%lf", &dom.ds_n);
  } else {
    fprintf(stderr, "flow.config read error (%s:%d).\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.pB %s", sbuf);
  if (strcmp(sbuf, "PERIODIC") == 0) {
    dom.bc_b = PERIODIC;
    dom.ds_b = 0;
  } else if(strcmp(sbuf, "NEUMANN") == 0) {
    dom.bc_b = NEUMANN;
    fret = fscanf(infile, "%lf", &dom.ds_b);
  } else {
    fprintf(stderr, "flow.config read error (%s:%d).\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.pT %s", sbuf);
  if (strcmp(sbuf, "PERIODIC") == 0) {
    dom.bc_t = PERIODIC;
    dom.ds_t = 0;
  } else if(strcmp(sbuf, "NEUMANN") == 0) {
    dom.bc_t = NEUMANN;
    fret = fscanf(infile, "%lf", &dom.ds_t);
  } else {
    fprintf(stderr, "flow.config read error (%s:%d).\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  if (dom.ds_e < 0 || dom.ds_w < 0 || 
      dom.ds_n < 0 || dom.ds_s < 0 || 
      dom.ds_t < 0 || dom.ds_b < 0) {
    fprintf(stderr, "flow.config read error -- screen offsets must be >= 0\n");
    exit(EXIT_FAILURE);
  }

  #ifdef DEBUG
    printf("DOMAIN SIZE\n");
    printf(" (Xs, Xe, Xl) = %lf, %lf, %lf\n", dom.xs, dom.xe, dom.xl);
    printf(" (Ys, Ye, Yl) = %lf, %lf, %lf\n", dom.ys, dom.ye, dom.yl);
    printf(" (Zs, Ze, Zl) = %lf, %lf, %lf\n", dom.zs, dom.ze, dom.zl);
    printf("\n");
    printf("BOUNDARY CONDITIONS\n");
    printf("  (0 == PERIODIC, 1 == DIRICHLET, 2 == NEUMANN)\n");
    printf(" (e, w) = %d, %d\n", dom.bc_e, dom.bc_w);
    printf(" (n, s) = %d, %d\n", dom.bc_n, dom.bc_s);
    printf(" (t, b) = %d, %d\n", dom.bc_t, dom.bc_b);
    printf("SCREEN OFFSETS\n");
    printf(" (e, w) = %lf, %lf\n", dom.ds_e, dom.ds_w);
    printf(" (n, s) = %lf, %lf\n", dom.ds_n, dom.ds_s);
    printf(" (t, b) = %lf, %lf\n", dom.ds_t, dom.ds_b);
  #endif

  fclose(infile);
}

void seed_array(void) 
{
  int fail = 0;

  /* Correct domain size for screen */
  double xs = dom.xs + dom.ds_w;
  //double xe = dom.xe + dom.ds_e;
  double xl = dom.xl - dom.ds_e - dom.ds_w;
  double ys = dom.ys + dom.ds_s;
  //double ye = dom.ye + dom.ds_n;
  double yl = dom.yl - dom.ds_s - dom.ds_n;
  double zs = dom.zs + dom.ds_b;
  //double ze = dom.ze + dom.ds_t;
  double zl = dom.zl - dom.ds_b - dom.ds_t;

  // Volume fraction
  nparts = nx * ny * nz;
  vfrac = nparts * 4./3. * PI * a*a*a / (dom.xl * dom.yl * dom.zl);

  // Spacing
  double dx = xl / nx;
  double dy = yl / ny;
  double dz = zl / nz;

  // Quit if...
  // volume fraction is more than cubic packing max
  if (vfrac >= PI/6.) {
    printf("Volume fraction is %.4lf, max is %.4lf.\n", vfrac, PI/6.);
    printf("Exiting.\n");
    fail = 1;
  } 
  // spacing is closer than two particles
  if (dx < 2.*a) {
    printf("Too many particles in x direction!\n");
    fail = 1;
  }
  if (dy < 2.*a) {
    printf("Too many particles in y direction!\n");
    fail = 1;
  }
  if (dz < 2.*a) {
    printf("Too many particles in z direction!\n");
    fail = 1;
  }

  if (fail == 1) {
    printf("... array seeder failed.\n\n");
    exit(EXIT_FAILURE);
  } else {
    printf("... volume fraction is %.4lf\n\n", vfrac);
  }

  /* Allocate parts */
  parts = (part_struct*) malloc(nparts * sizeof(part_struct));

  // Fill parts
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        int p = i + j*nx + k*(nx*ny);

        parts[p].x = xs + 0.5*dx + i*dx;
        parts[p].y = ys + 0.5*dy + j*dy;
        parts[p].z = zs + 0.5*dz + k*dz;
        parts[p].r = a;
        parts[p].u = 0.;
        parts[p].v = 0.;
        parts[p].w = 0.;
      }
    }
  }

  printf("Writing part_seeder_array.config...");
  fflush(stdout);

  // write to file
  char fname[FILE_NAME_SIZE] = "";
  // open file for writing
  sprintf(fname, "%s/part_seeder_array.config", INPUT_DIR);
  FILE *ofile = fopen(fname, "w");
  if(ofile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  // write the number of particles and compact support length ratio
  fprintf(ofile, "n %d\n", nparts);
  fprintf(ofile, "(l/a) %f\n", loa);

  // write each particle configuration
  for(int i = 0; i < nparts; i++) {
    fprintf(ofile, "\n");
    fprintf(ofile, "r %f\n", parts[i].r);
    fprintf(ofile, "(x, y, z) %f %f %f\n", parts[i].x, parts[i].y, parts[i].z);
    fprintf(ofile, "(u, v, w) %f %f %f\n", parts[i].u, parts[i].v, parts[i].w);
    fprintf(ofile, "(aFx, aFy, aFz) %f %f %f\n", aFx, aFy, aFz);
    fprintf(ofile, "(aLx, aLy, aLz) %f %f %f\n", aLx, aLy, aLz);
    fprintf(ofile, "rho %f\n", rho);
    fprintf(ofile, "E %f\n", E);
    fprintf(ofile, "sigma %f\n", sigma);
    fprintf(ofile, "e_dry %f\n", e_dry);
    fprintf(ofile, "coeff_fric %f\n", coeff_fric);
    fprintf(ofile, "order %d\n", order);
    fprintf(ofile, "rs/r %f\n", rs);
    fprintf(ofile, "spring_k %f\n", spring_k);
    fprintf(ofile, "spring (x, y, z) %f %f %f\n", spring_x, spring_y, spring_z);
    fprintf(ofile, "spring_l %f\n", spring_l);
    fprintf(ofile, "translating %d\n", translating);
    fprintf(ofile, "rotating %d\n", rotating);
  }

  // close the file
  fclose(ofile);
  printf(" done.\n");
  printf("\n... bluebottle array seeder done.\n\n");
  fflush(stdout);

  // Free
  free(parts);
}

void seeder(void)
{
  srand(time(NULL));

  /* Correct domain size for screen */
  double xs = dom.xs + dom.ds_w;
  double xe = dom.xe + dom.ds_e;
  double xl = dom.xl - dom.ds_e - dom.ds_w;
  double ys = dom.ys + dom.ds_s;
  double ye = dom.ye + dom.ds_n;
  double yl = dom.yl - dom.ds_s - dom.ds_n;
  double zs = dom.zs + dom.ds_b;
  double ze = dom.ze + dom.ds_t;
  double zl = dom.zl - dom.ds_b - dom.ds_t;

  /* Some variables */
  int fits = 1;
  int attempts = 1;
  int fail = 0;
  int redo = 1;
  double gap = 1.;
  double xx, yy, zz;

  /* Allocate parts */
  parts = (part_struct*) malloc(nparts * sizeof(part_struct));

  /* Start the seeding */

  // Place the first particle...
  parts[0].r = a;

  // ... in x
  redo = 1;
  while (redo == 1) {
    redo = 0;
    parts[0].x = xl * (rand() / (double) RAND_MAX) + xs;

    // Replace if it's too close to a wall
    if ((dom.bc_w != PERIODIC) && (parts[0].x < (xs + gap*parts[0].r)))
      redo = 1;
    if ((dom.bc_e != PERIODIC) && (parts[0].x > (xe - gap*parts[0].r)))
      redo = 1;
  }

  // ... in y
  redo = 1;
  while (redo == 1) {
    redo = 0;
    parts[0].y = yl * (rand() / (double) RAND_MAX) + ys;

    // Replace if it's too close to a wall
    if ((dom.bc_s != PERIODIC) && (parts[0].y < (ys + gap*parts[0].r)))
      redo = 1;
    if ((dom.bc_n != PERIODIC) && (parts[0].y > (ye - gap*parts[0].r)))
      redo = 1;
  }

  // ... in z
  redo = 1;
  while (redo == 1) {
    redo = 0;
    parts[0].z = zl * (rand() / (double) RAND_MAX) + zs;

    // Replace if it's too close to a wall
    if ((dom.bc_b != PERIODIC) && (parts[0].z < (zs + gap*parts[0].r)))
      redo = 1;
    if ((dom.bc_t != PERIODIC) && (parts[0].z > (ze - gap*parts[0].r)))
      redo = 1;
  }

  // Set rest of part struct
  if (rand_u) {
    parts[0].u = vel_scale * (2.*rand()/ (double) RAND_MAX - 1.);
    parts[0].v = vel_scale * (2.*rand()/ (double) RAND_MAX - 1.);
    parts[0].w = vel_scale * (2.*rand()/ (double) RAND_MAX - 1.);
  } else {
    parts[0].u = 0;
    parts[0].v = 0;
    parts[0].w = 0;
  }
  // Everything else, just ignore and set to the extern'd variables

  /* Place the rest of the particles */
  int i = 0;
  for (i = 1; i < nparts; i++) {
    fits = !fits;
    if (fail) break;
    while (!fits) {
      attempts++;

      // Place the next particle...
      parts[i].r = a;

      // ... in x
      redo = 1;
      while (redo == 1) {
        redo = 0;
        parts[i].x = xl * (rand() / (double) RAND_MAX) + xs;

        // Replace if it's too close to a wall
        if ((dom.bc_w != PERIODIC) && (parts[i].x < (xs + gap*parts[i].r)))
          redo = 1;
        if ((dom.bc_e != PERIODIC) && (parts[i].x > (xe - gap*parts[i].r)))
          redo = 1;
      }

      // ... in y
      redo = 1;
      while (redo == 1) {
        redo = 0;
        parts[i].y = yl * (rand() / (double) RAND_MAX) + ys;

        // Replace if it's too close to a wall
        if ((dom.bc_s != PERIODIC) && (parts[i].y < (ys + gap*parts[i].r)))
          redo = 1;
        if ((dom.bc_n != PERIODIC) && (parts[i].y > (ye - gap*parts[i].r)))
          redo = 1;
      }

      // ... in z
      redo = 1;
      while (redo == 1) {
        redo = 0;
        parts[i].z = zl * (rand() / (double) RAND_MAX) + zs;

        // Replace if it's too close to a wall
        if ((dom.bc_b != PERIODIC) && (parts[i].z < (zs + gap*parts[i].r)))
          redo = 1;
        if ((dom.bc_t != PERIODIC) && (parts[i].z > (ze - gap*parts[i].r)))
          redo = 1;
      }

      // Set rest of part struct
      // NOTE: Just becuase the velocity is set doesn't mean it's set
      // intelligently. Probably best used for debugging particle motion
      // for the time being
      if (rand_u) {
        parts[i].u = vel_scale * (2.*rand()/ (double) RAND_MAX - 1.);
        parts[i].v = vel_scale * (2.*rand()/ (double) RAND_MAX - 1.);
        parts[i].w = vel_scale * (2.*rand()/ (double) RAND_MAX - 1.);
      } else {
        parts[i].u = 0;
        parts[i].v = 0;
        parts[i].w = 0;
     }

      /* Check that particle does not intersect any others */
      fits = !fits;
      for (int j = 0; j < i; j++) {
        xx = parts[i].x - parts[j].x;
        xx = xx * xx;
        yy = parts[i].y - parts[j].y;
        yy = yy * yy;
        zz = parts[i].z - parts[j].z;
        zz = zz * zz;
        if (sqrt(xx + yy + zz) < (gap*parts[i].r + gap*parts[j].r)) {
          fits = !fits;
          break;
        }

        // use a virtual particle to check if particle is too close in a
        // periodic direction

        // x
        if (dom.bc_w == PERIODIC) {
          xx = parts[i].x - parts[j].x;
          if (parts[i].x < (xs + parts[i].r))
            xx = parts[i].x + xl - parts[j].x;
          if (parts[i].x > (xe - parts[i].r))
            xx = parts[i].x - xl - parts[j].x;

          xx = xx * xx;

          yy = parts[i].y - parts[j].y;
          yy = yy * yy;
          zz = parts[i].z - parts[j].z;
          zz = zz * zz;

          if (sqrt(xx + yy + zz) < (gap*parts[i].r + gap*parts[j].r)) {
            fits = !fits;
            break;
          }
        }

        // y
        if (dom.bc_s == PERIODIC) {
          xx = parts[i].x - parts[j].x;
          xx = xx * xx;

          yy = parts[i].y - parts[j].y;
          if (parts[i].y < (ys + parts[i].r))
            yy = parts[i].y + yl - parts[j].y;
          if (parts[i].y > (ye - parts[i].r))
            yy = parts[i].y - yl - parts[j].y;

          yy = yy * yy;

          zz = parts[i].z - parts[j].z;
          zz = zz * zz;

          if (sqrt(xx + yy + zz) < (gap*parts[i].r + gap*parts[j].r)) {
            fits = !fits;
            break;
          }
        }

        // z
        if (dom.bc_s == PERIODIC) {
          xx = parts[i].x - parts[j].x;
          xx = xx * xx;

          yy = parts[i].y - parts[j].y;
          yy = yy * yy;

          zz = parts[i].z - parts[j].z;
          if (parts[i].z < (zs + parts[i].r))
            zz = parts[i].z + zl - parts[j].z;
          if (parts[i].z > (ze - parts[i].r))
            zz = parts[i].z - zl - parts[j].z;

          zz = zz * zz;

          if (sqrt(xx + yy + zz) < (gap*parts[i].r + gap*parts[j].r)) {
            fits = !fits;
            break;
          }
        }

        // x and y
        if(dom.bc_w == PERIODIC && dom.bc_s == PERIODIC) {
          xx = parts[i].x - parts[j].x;
          if(parts[i].x < (xs + parts[i].r))
            xx = parts[i].x + xl - parts[j].x;
          if(parts[i].x > (xe - parts[i].r))
            xx = parts[i].x - xl - parts[j].x;
          xx = xx * xx;

          yy = parts[i].y - parts[j].y;
          if(parts[i].y < (ys + parts[i].r))
            yy = parts[i].y + yl - parts[j].y;
          if(parts[i].y > (ye - parts[i].r))
            yy = parts[i].y - yl - parts[j].y;
          yy = yy * yy;

          zz = parts[i].z - parts[j].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[i].r + gap*parts[j].r)) {
            fits = !fits;
            break;
          }
        }

        // y and z
        if(dom.bc_s == PERIODIC && dom.bc_b == PERIODIC) {
          xx = parts[i].x - parts[j].x;
          xx = xx * xx;

          yy = parts[i].y - parts[j].y;
          if(parts[i].y < (ys + parts[i].r))
            yy = parts[i].y + yl - parts[j].y;
          if(parts[i].y > (ye - parts[i].r))
            yy = parts[i].y - yl - parts[j].y;
          yy = yy * yy;

          zz = parts[i].z - parts[j].z;
          if(parts[i].z < (zs + parts[i].r))
            zz = parts[i].z + zl - parts[j].z;
          if(parts[i].z > (ze - parts[i].r))
            zz = parts[i].z - zl - parts[j].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[i].r + gap*parts[j].r)) {
            fits = !fits;
            break;
          }
        }

        // z and x
        if(dom.bc_w == PERIODIC && dom.bc_b == PERIODIC) {
          xx = parts[i].x - parts[j].x;
          if(parts[i].x < (xs + parts[i].r))
            xx = parts[i].x + xl - parts[j].x;
          if(parts[i].x > (xe - parts[i].r))
            xx = parts[i].x - xl - parts[j].x;
          xx = xx * xx;

          yy = parts[i].y - parts[j].y;
          yy = yy * yy;

          zz = parts[i].z - parts[j].z;
          if(parts[i].z < (zs + parts[i].r))
            zz = parts[i].z + zl - parts[j].z;
          if(parts[i].z > (ze - parts[i].r))
            zz = parts[i].z - zl - parts[j].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[i].r + gap*parts[j].r)) {
            fits = !fits;
            break;
          }
        }

        // x, y, and z
        if(dom.bc_w == PERIODIC && dom.bc_s == PERIODIC && dom.bc_b == PERIODIC) {
          xx = parts[i].x - parts[j].x;
          if(parts[i].x < (xs + parts[i].r))
            xx = parts[i].x + xl - parts[j].x;
          if(parts[i].x > (xe - parts[i].r))
            xx = parts[i].x - xl - parts[j].x;
          xx = xx * xx;

          yy = parts[i].y - parts[j].y;
          if(parts[i].y < (ys + parts[i].r))
            yy = parts[i].y + yl - parts[j].y;
          if(parts[i].y > (ye - parts[i].r))
            yy = parts[i].y - yl - parts[j].y;
          yy = yy * yy;

          zz = parts[i].z - parts[j].z;
          if(parts[i].z < (zs + parts[i].r))
            zz = parts[i].z + zl - parts[j].z;
          if(parts[i].z > (ze - parts[i].r))
            zz = parts[i].z - zl - parts[j].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[i].r + gap*parts[j].r)) {
            fits = !fits;
            break;
          }
        }

        // check both ways
        // x only
        if(dom.bc_w == PERIODIC) {
          xx = parts[j].x - parts[i].x;
          if(parts[j].x < (xs + parts[j].r))
            xx = parts[j].x + xl - parts[i].x;
          if(parts[j].x > (xe - parts[j].r))
            xx = parts[j].x - xl - parts[i].x;
          xx = xx * xx;

          yy = parts[j].y - parts[i].y;
          yy = yy * yy;

          zz = parts[j].z - parts[i].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[j].r + gap*parts[i].r)) {
            fits = !fits;
            break;
          }
        }

        // y only
        if(dom.bc_s == PERIODIC) {
          xx = parts[j].x - parts[i].x;
          xx = xx * xx;

          yy = parts[j].y - parts[i].y;
          if(parts[j].y < (ys + parts[j].r))
            yy = parts[j].y + yl - parts[i].y;
          if(parts[j].y > (ye - parts[j].r))
            yy = parts[j].y - yl - parts[i].y;
          yy = yy * yy;

          zz = parts[j].z - parts[i].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[j].r + gap*parts[i].r)) {
            fits = !fits;
            break;
          }
        }

        // z only
        if(dom.bc_b == PERIODIC) {
          xx = parts[j].x - parts[i].x;
          xx = xx * xx;

          yy = parts[j].y - parts[i].y;
          yy = yy * yy;

          zz = parts[j].z - parts[i].z;
          if(parts[j].z < (zs + parts[j].r))
            zz = parts[j].z + zl - parts[i].z;
          if(parts[j].z > (ze - parts[j].r))
            zz = parts[j].z - zl - parts[i].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[j].r + gap*parts[i].r)) {
            fits = !fits;
            break;
          }
        }

        // x and y
        if(dom.bc_w == PERIODIC && dom.bc_s == PERIODIC) {
          xx = parts[j].x - parts[i].x;
          if(parts[j].x < (xs + parts[j].r))
            xx = parts[j].x + xl - parts[i].x;
          if(parts[j].x > (xe - parts[j].r))
            xx = parts[j].x - xl - parts[i].x;
          xx = xx * xx;

          yy = parts[j].y - parts[i].y;
          if(parts[j].y < (ys + parts[j].r))
            yy = parts[j].y + yl - parts[i].y;
          if(parts[j].y > (ye - parts[j].r))
            yy = parts[j].y - yl - parts[i].y;
          yy = yy * yy;

          zz = parts[j].z - parts[i].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[j].r + gap*parts[i].r)) {
            fits = !fits;
            break;
          }
        }

        // y and z
        if(dom.bc_s == PERIODIC && dom.bc_b == PERIODIC) {
          xx = parts[j].x - parts[i].x;
          xx = xx * xx;

          yy = parts[j].y - parts[i].y;
          if(parts[j].y < (ys + parts[j].r))
            yy = parts[j].y + yl - parts[i].y;
          if(parts[j].y > (ye - parts[j].r))
            yy = parts[j].y - yl - parts[i].y;
          yy = yy * yy;

          zz = parts[j].z - parts[i].z;
          if(parts[j].z < (zs + parts[j].r))
            zz = parts[j].z + zl - parts[i].z;
          if(parts[j].z > (ze - parts[j].r))
            zz = parts[j].z - zl - parts[i].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[j].r + gap*parts[i].r)) {
            fits = !fits;
            break;
          }
        }

        // z and x
        if(dom.bc_w == PERIODIC && dom.bc_b == PERIODIC) {
          xx = parts[j].x - parts[i].x;
          if(parts[j].x < (xs + parts[j].r))
            xx = parts[j].x + xl - parts[i].x;
          if(parts[j].x > (xe - parts[j].r))
            xx = parts[j].x - xl - parts[i].x;
          xx = xx * xx;

          yy = parts[j].y - parts[i].y;
          yy = yy * yy;

          zz = parts[j].z - parts[i].z;
          if(parts[j].z < (zs + parts[j].r))
            zz = parts[j].z + zl - parts[i].z;
          if(parts[j].z > (ze - parts[j].r))
            zz = parts[j].z - zl - parts[i].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[j].r + gap*parts[i].r)) {
            fits = !fits;
            break;
          }
        }

        // x, y, and z
        if(dom.bc_w == PERIODIC && dom.bc_s == PERIODIC && dom.bc_b == PERIODIC) {
          xx = parts[j].x - parts[i].x;
          if(parts[j].x < (xs + parts[j].r))
            xx = parts[j].x + xl - parts[i].x;
          if(parts[j].x > (xe - parts[j].r))
            xx = parts[j].x - xl - parts[i].x;
          xx = xx * xx;

          yy = parts[j].y - parts[i].y;
          if(parts[j].y < (ys + parts[j].r))
            yy = parts[j].y + yl - parts[i].y;
          if(parts[j].y > (ye - parts[j].r))
            yy = parts[j].y - yl - parts[i].y;
          yy = yy * yy;

          zz = parts[j].z - parts[i].z;
          if(parts[j].z < (zs + parts[j].r))
            zz = parts[j].z + zl - parts[i].z;
          if(parts[j].z > (ze - parts[j].r))
            zz = parts[j].z - zl - parts[i].z;
          zz = zz * zz;

          if(sqrt(xx + yy + zz) < (gap*parts[j].r + gap*parts[i].r)) {
            fits = !fits;
            break;
          }
        }
      } // for (int j = 0; j < i; j++)

      if(attempts == 1e5*nparts) {
        fail = !fail;
        break;
      }
    } // while (!fits)...

    printf("Placed %d of %d particles in %d attempts...\r", i-1, nparts, attempts);
  } // for (i = 1; i < nparts; i++)

  if(fail) {
    printf("After %d attempts, the seeder has placed", attempts);
    printf(" %d of %d particles (a = %f).\n\n", i-1, nparts, a);
    printf("...bluebottle seeder done.\n\n");
    exit(EXIT_FAILURE);
  }

  printf("It took %d attempts to place %d", attempts, nparts);
  printf(" particles (a = %f) with no intersections.\n\n", a);
  fflush(stdout);

  // Write
  printf("Writing part_seeder.config...");
  fflush(stdout);
  // write particle configuration to file
  char fname[FILE_NAME_SIZE] = "";
  // open file for writing
  sprintf(fname, "%s/part_seeder.config", INPUT_DIR);
  FILE *ofile = fopen(fname, "w");
  if(ofile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  // write the number of particles and compact support length ratio
  fprintf(ofile, "n %d\n", nparts);
  fprintf(ofile, "(l/a) %f\n", loa);

  // write each particle configuration
  for(int i = 0; i < nparts; i++) {
    fprintf(ofile, "\n");
    fprintf(ofile, "r %f\n", parts[i].r);
    fprintf(ofile, "(x, y, z) %f %f %f\n", parts[i].x, parts[i].y, parts[i].z);
    fprintf(ofile, "(u, v, w) %f %f %f\n", parts[i].u, parts[i].v, parts[i].w);
    fprintf(ofile, "(aFx, aFy, aFz) %f %f %f\n", aFx, aFy, aFz);
    fprintf(ofile, "(aLx, aLy, aLz) %f %f %f\n", aLx, aLy, aLz);
    fprintf(ofile, "rho %f\n", rho);
    fprintf(ofile, "E %f\n", E);
    fprintf(ofile, "sigma %f\n", sigma);
    fprintf(ofile, "e_dry %f\n", e_dry);
    fprintf(ofile, "coeff_fric %f\n", coeff_fric);
    fprintf(ofile, "order %d\n", order);
    fprintf(ofile, "rs/r %f\n", rs);
    fprintf(ofile, "spring_k %f\n", spring_k);
    fprintf(ofile, "spring (x, y, z) %f %f %f\n", spring_x, spring_y, spring_z);
    fprintf(ofile, "spring_l %f\n", spring_l);
    fprintf(ofile, "translating %d\n", translating);
    fprintf(ofile, "rotating %d\n", rotating);
  }

  // close the file
  fclose(ofile);
  printf("done.\n");
  printf("\n...bluebottle seeder done.\n\n");
  fflush(stdout);

  // Free
  free(parts);
}

/* Variable declaration */
int rand_u; // random initial velocity flag
int array;  // array seeder flag
double vel_scale;
int nx;             // nparts in x direction (for array)
int ny;             // nparts in y direction (for array)
int nz;             // nparts in z direction (for array)
double vfrac;

int nparts;
double loa;
double a;      
double x, y, z;
double u, v, w;
double aFx, aFy, aFz;
double aLx, aLy, aLz;
double rho;
double E;
double sigma;
double e_dry;
double coeff_fric;
int order;
double rs;
double spring_k;
double spring_x, spring_y, spring_z;
double spring_l;
int translating;
int rotating;

dom_struct dom;
part_struct *parts;

char part_file[FILE_NAME_SIZE];
char INPUT_DIR[FILE_NAME_SIZE];
