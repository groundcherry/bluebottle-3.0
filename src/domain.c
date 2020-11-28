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

#include <time.h>

#include "bluebottle.h"
#include "domain.h"
#include "rng.h"
#include "scalar.h"

int use_restart;

void parse_cmdline_args(int argc, char *argv[])
{
  // If none given, run normally
  // If -r, read input from restart
  // Else, exit
  // From bluebottle and K&R, p. 117

  use_restart = 0;
  int argin;

  while (--argc > 0 && *(++argv)[0] == '-') {
    while ((argin = *++argv[0])) {
      switch (argin) {
        case 'r':
          use_restart = 1;
          break;
        default:
          use_restart = 2;
          printf("bluebottle: illegal option %c\n", argin);
          argc = 0;
          break;
      }
    }
  }

  // If more than just "bluebottle -r"
  if (use_restart == 1 && argc > 0) {
    printf("Usage: Try 'bluebottle -r' for restart capability\n");
    exit(EXIT_FAILURE);

  // If no recognized arguments
  } else if (use_restart == 2) {
    printf("Usage: bluebottle command line arguments: \n");
    printf("                  --r\n");
    printf("                    Use restart file\n");
    printf("No command line arguments runs bluebottle normally.\n");
  }
}

void domain_read_input(void)
{
  int i;  // iterator

  cpumem = 0;
  gpumem = 0;

  int fret = 0;
  fret = fret; // prevent compiler warning

  /* open flow.config file for reading */
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/%s/flow.config", ROOT_DIR, INPUT_DIR);
  FILE *infile = fopen(fname, "r");
  if (infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  /* open decomp.config file for reading */
  char dname[FILE_NAME_SIZE];
  sprintf(dname, "%s/%s/decomp.config", ROOT_DIR, INPUT_DIR);
  FILE *dfile = fopen(dname, "r");
  if (dfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", dname);
    exit(EXIT_FAILURE);
  }

  char buf[CHAR_BUF_SIZE];  // character read buffer

  /* read global domain */
  fret = fscanf(infile, "GLOBAL DOMAIN\n");
#ifdef DOUBLE
  fret = fscanf(infile, "(Xs, Xe, Xn) %lf %lf %d\n", &DOM.xs, &DOM.xe, &DOM.xn);
  fret = fscanf(infile, "(Ys, Ye, Yn) %lf %lf %d\n", &DOM.ys, &DOM.ye, &DOM.yn);
  fret = fscanf(infile, "(Zs, Ze, Zn) %lf %lf %d\n", &DOM.zs, &DOM.ze, &DOM.zn);
  fret = fscanf(infile, "\n"); // \n
#else // single precision
  fret = fscanf(infile, "(Xs, Xe, Xn) %f %f %d\n", &DOM.xs, &DOM.xe, &DOM.xn);
  fret = fscanf(infile, "(Ys, Ye, Yn) %f %f %d\n", &DOM.ys, &DOM.ye, &DOM.yn);
  fret = fscanf(infile, "(Zs, Ze, Zn) %f %f %d\n", &DOM.zs, &DOM.ze, &DOM.zn);
  fret = fscanf(infile, "\n"); // \n
#endif

  /* read mpi-gpu decomposition */
  fret = fscanf(infile, "MPI/GPU SUBDOMAIN DECOMPOSITION\n");
  fret = fscanf(infile, "(In, Jn, Kn) %d %d %d\n", &DOM.In, &DOM.Jn, &DOM.Kn);
  fret = fscanf(infile, "\n"); // \n

  DOM.S1 = DOM.In;
  DOM.S2 = DOM.S1 * DOM.Jn;
  DOM.S3 = DOM.S2 * DOM.Kn;

  // If number of domains in input and mpirun aren't the same, quit
  if (DOM.S3 != nprocs) {
    fprintf(stderr,"  The number of processes requested with mpirun (%d) does"
           " not\n   match the number specified in the flow.config file (%d)\n",
           nprocs, DOM.S3);
    exit(EXIT_FAILURE);
  }

  // allocate memory for subdomains
  dom = (dom_struct*) malloc(DOM.S3 * sizeof(dom_struct));
    cpumem += DOM.S3 * sizeof(dom_struct);

  /* Read subdomain mapping */
  for(i = 0; i < DOM.S3; i++) {  
    fret = fscanf(dfile, "(I, J, K) %d %d %d\n", &dom[i].I, &dom[i].J, &dom[i].K);

    dom[i].rank = dom[i].I + dom[i].J * DOM.S1 + dom[i].K * DOM.S2;

#ifdef DOUBLE
    fret = fscanf(dfile, "(Xs, Xe, Xn) %lf %lf %d\n", 
                  &dom[i].xs, &dom[i].xe, &dom[i].xn);
    fret = fscanf(dfile, "(Ys, Ye, Yn) %lf %lf %d\n",
                  &dom[i].ys, &dom[i].ye, &dom[i].yn);
    fret = fscanf(dfile, "(Zs, Ze, Zn) %lf %lf %d\n",
                  &dom[i].zs, &dom[i].ze, &dom[i].zn);
#else // single
    fret = fscanf(dfile, "(Xs, Xe, Xn) %f %f %d\n",
                  &dom[i].xs, &dom[i].xe, &dom[i].xn);
    fret = fscanf(dfile, "(Ys, Ye, Yn) %f %f %d\n",
                  &dom[i].ys, &dom[i].ye, &dom[i].yn);
    fret = fscanf(dfile, "(Zs, Ze, Zn) %f %f %d\n",
                  &dom[i].zs, &dom[i].ze, &dom[i].zn);
#endif
    fret = fscanf(dfile, "\n");
  }
  fclose(dfile);

  fret = fscanf(infile, "PHYSICAL PARAMETERS\n");
#ifdef DOUBLE
  fret = fscanf(infile, "rho_f %lf\n", &rho_f);
  fret = fscanf(infile, "nu %lf\n", &nu);
  fret = fscanf(infile, "s_D %lf\n", &s_D);
  fret = fscanf(infile, "s_k %lf\n", &s_k);
  fret = fscanf(infile, "s_beta %lf\n", &s_beta);
  fret = fscanf(infile, "s_ref %lf\n", &s_ref);
#else // single
  fret = fscanf(infile, "rho_f %f\n", &rho_f);
  fret = fscanf(infile, "nu %f\n", &nu);
  fret = fscanf(infile, "s_D %f\n", &s_D);
  fret = fscanf(infile, "s_k %f\n", &s_k);
  fret = fscanf(infile, "s_beta %f\n", &s_beta);
  fret = fscanf(infile, "s_ref %f\n", &s_ref);
#endif
  mu = nu * rho_f;  // set dynamic viscosity
  fret = fscanf(infile, "\n");

  /* Read simulation parameters */
  fret = fscanf(infile, "SIMULATION PARAMETERS\n");
#ifdef DOUBLE
  fret = fscanf(infile, "duration %lf\n", &duration);
  fret = fscanf(infile, "CFL %lf\n", &CFL);
  fret = fscanf(infile, "SCALAR %d\n", &SCALAR);
  fret = fscanf(infile, "pp_max_iter %d\n", &pp_max_iter);
  fret = fscanf(infile, "pp_residual %lf\n", &pp_residual);
  fret = fscanf(infile, "lamb_max_iter %d\n", &lamb_max_iter);
  fret = fscanf(infile, "lamb_residual %lf\n", &lamb_residual);
  fret = fscanf(infile, "lamb_relax %lf\n", &lamb_relax);
  fret = fscanf(infile, "lamb_cut %lf\n", &lamb_cut);
  fret = fscanf(infile, "s_lamb_cut %lf\n", &lamb_cut_scalar);
#else
  fret = fscanf(infile, "duration %f\n", &duration);
  fret = fscanf(infile, "CFL %f\n", &CFL);
  fret = fscanf(infile, "SCALAR %d\n", &SCALAR);
  fret = fscanf(infile, "pp_max_iter %d\n", &pp_max_iter);
  fret = fscanf(infile, "pp_residual %f\n", &pp_residual);
  fret = fscanf(infile, "lamb_max_iter %d\n", &lamb_max_iter);
  fret = fscanf(infile, "lamb_residual %f\n", &lamb_residual);
  fret = fscanf(infile, "lamb_relax %f\n", &lamb_relax);
  fret = fscanf(infile, "lamb_cut %f\n", &lamb_cut);
  fret = fscanf(infile, "s_lamb_cut %f\n", &lamb_cut_scalar);
#endif
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "BOUNDARY CONDITIONS\n");
  /* Read pressure boundary conditions */
#ifdef DOUBLE
  fret = fscanf(infile, "v_bc_tdelay %lf\n", &v_bc_tdelay);
#else  
  fret = fscanf(infile, "v_bc_tdelay %f\n", &v_bc_tdelay);
#endif
  fret = fscanf(infile, "PRESSURE\n");
  fret = fscanf(infile, "bc.pW %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.pW = PERIODIC;
    bc.dsW = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.pW = NEUMANN;
    fret = fscanf(infile, "%lf", &bc.dsW);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pE %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.pE = PERIODIC;
    bc.dsE = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.pE = NEUMANN;
    fret = fscanf(infile, "%lf", &bc.dsE);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pS %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.pS = PERIODIC;
    bc.dsS = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.pS = NEUMANN;
    fret = fscanf(infile, "%lf", &bc.dsS);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pN %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.pN = PERIODIC;
    bc.dsN = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.pN = NEUMANN;
    fret = fscanf(infile, "%lf", &bc.dsN);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pB %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.pB = PERIODIC;
    bc.dsB = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.pB = NEUMANN;
    fret = fscanf(infile, "%lf", &bc.dsB);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pT %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.pT = PERIODIC;
    bc.dsT = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.pT = NEUMANN;
    fret = fscanf(infile, "%lf", &bc.dsT);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  if (bc.dsE < 0 || bc.dsW < 0 || 
      bc.dsN < 0 || bc.dsS < 0 || 
      bc.dsT < 0 || bc.dsB < 0) {
    fprintf(stderr, "flow.config read error -- screen offsets must be >= 0\n");
    exit(EXIT_FAILURE);
  }

  /* Read u-velocity boundary conditions */
  fret = fscanf(infile, "X-VELOCITY\n");

  fret = fscanf(infile, "bc.uW %s", buf);
  if (strcmp(buf, "PERIODIC") == 0)
    bc.uW = PERIODIC;
  else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.uW = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.uWDm, &bc.uWDa);
    bc.uWD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.uW = NEUMANN;
    bc.uWDm = 0;
    bc.uWDa = 0;
    bc.uWD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.uW = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.uWDm, &bc.uWDa);
    bc.uWD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.uE %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.uE = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.uE = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.uEDm, &bc.uEDa);
    bc.uED = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.uE = NEUMANN;
    bc.uEDm = 0;
    bc.uEDa = 0;
    bc.uED = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.uE = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.uEDm, &bc.uEDa);
    bc.uED = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.uS %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.uS = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.uS = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.uSDm, &bc.uSDa);
    bc.uSD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.uS = NEUMANN;
    bc.uSDm = 0;
    bc.uSDa = 0;
    bc.uSD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.uS = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.uSDm, &bc.uSDa);
    bc.uSD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.uN %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.uN = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.uN = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.uNDm, &bc.uNDa);
    bc.uND = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.uN = NEUMANN;
    bc.uNDm = 0;
    bc.uNDa = 0;
    bc.uND = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.uN = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.uNDm, &bc.uNDa);
    bc.uND = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.uB %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.uB = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.uB = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.uBDm, &bc.uBDa);
    bc.uBD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.uB = NEUMANN;
    bc.uBDm = 0;
    bc.uBDa = 0;
    bc.uBD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.uB = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.uBDm, &bc.uBDa);
    bc.uBD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.uT %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.uT = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.uT = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.uTDm, &bc.uTDa);
    bc.uTD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.uT = NEUMANN;
    bc.uTDm = 0;
    bc.uTDa = 0;
    bc.uTD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.uT = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.uTDm, &bc.uTDa);
    bc.uTD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  /* Read v-velocity boundary conditions */
  fret = fscanf(infile, "Y-VELOCITY\n");

  fret = fscanf(infile, "bc.vW %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.vW = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.vW = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.vWDm, &bc.vWDa);
    bc.vWD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.vW = NEUMANN;
    bc.vWDm = 0;
    bc.vWDa = 0;
    bc.vWD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.vW = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.vWDm, &bc.vWDa);
    bc.vWD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.vE %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.vE = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.vE = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.vEDm, &bc.vEDa);
    bc.vED = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.vE = NEUMANN;
    bc.vEDm = 0;
    bc.vEDa = 0;
    bc.vED = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.vE = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.vEDm, &bc.vEDa);
    bc.vED = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.vS %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.vS = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.vS = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.vSDm, &bc.vSDa);
    bc.vSD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.vS = NEUMANN;
    bc.vSDm = 0;
    bc.vSDa = 0;
    bc.vSD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.vS = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.vSDm, &bc.vSDa);
    bc.vSD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.vN %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.vN = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.vN = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.vNDm, &bc.vNDa);
    bc.vND = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.vN = NEUMANN;
    bc.vNDm = 0;
    bc.vNDa = 0;
    bc.vND = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.vN = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.vNDm, &bc.vNDa);
    bc.vND = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.vB %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.vB = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.vB = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.vBDm, &bc.vBDa);
    bc.vBD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.vB = NEUMANN;
    bc.vBDm = 0;
    bc.vBDa = 0;
    bc.vBD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.vB = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.vBDm, &bc.vBDa);
    bc.vBD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.vT %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.vT = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.vT = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.vTDm, &bc.vTDa);
    bc.vTD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.vT = NEUMANN;
    bc.vTDm = 0;
    bc.vTDa = 0;
    bc.vTD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.vT = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.vTDm, &bc.vTDa);
    bc.vTD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  /* Read w-velocity boundary conditions */
  fret = fscanf(infile, "Z-VELOCITY\n");

  fret = fscanf(infile, "bc.wW %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.wW = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.wW = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.wWDm, &bc.wWDa);
    bc.wWD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.wW = NEUMANN;
    bc.wWDm = 0;
    bc.wWDa = 0;
    bc.wWD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.wW = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.wWDm, &bc.wWDa);
    bc.wWD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.wE %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.wE = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.wE = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.wEDm, & bc.wEDa);
    bc.wED = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.wE = NEUMANN;
    bc.wEDm = 0;
    bc.wEDa = 0;
    bc.wED = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.wE = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.wEDm, &bc.wEDa);
    bc.wED = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.wS %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.wS = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.wS = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.wSDm, &bc.wSDa);
    bc.wSD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.wS = NEUMANN;
    bc.wSDm = 0;
    bc.wSDa = 0;
    bc.wSD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.wS = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.wSDm, &bc.wSDa);
    bc.wSD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.wN %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.wN = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.wN = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.wNDm, &bc.wNDa);
    bc.wND = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.wN = NEUMANN;
    bc.wNDm = 0;
    bc.wNDa = 0;
    bc.wND = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.wN = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.wNDm, &bc.wNDa);
    bc.wND = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.wB %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.wB = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.wB = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.wBDm, &bc.wBDa);
    bc.wBD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.wB = NEUMANN;
    bc.wBDm = 0;
    bc.wBDa = 0;
    bc.wBD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.wB = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.wBDm, &bc.wBDa);
    bc.wBD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc.wT %s", buf);
  if (strcmp(buf, "PERIODIC") == 0) {
    bc.wT = PERIODIC;
  } else if (strcmp(buf, "DIRICHLET") == 0) {
    bc.wT = DIRICHLET;
    fret = fscanf(infile, "%lf %lf", &bc.wTDm, &bc.wTDa);
    bc.wTD = 0;
  } else if (strcmp(buf, "NEUMANN") == 0) {
    bc.wT = NEUMANN;
    bc.wTDm = 0;
    bc.wTDa = 0;
    bc.wTD = 0;
  } else if (strcmp(buf, "PRECURSOR") == 0) {
    bc.wT = PRECURSOR;
    fret = fscanf(infile, "%lf %lf", &bc.wTDm, &bc.wTDa);
    bc.wTD = 0;
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

/* Read scalar boundary conditions */
  fret = fscanf(infile, "SCALAR\n");

  fret = fscanf(infile, "bc_s.sW %s", buf);
  if(strcmp(buf, "PERIODIC") == 0) {
    bc_s.sW = PERIODIC;
  } else if(strcmp(buf, "DIRICHLET") == 0) {
    bc_s.sW = DIRICHLET;
    fret = fscanf(infile, "%lf", &bc_s.sWD);
  } else if(strcmp(buf, "NEUMANN") == 0) {
    bc_s.sW = NEUMANN;
    fret = fscanf(infile, "%lf", &bc_s.sWN);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc_s.sE %s", buf);
  if(strcmp(buf, "PERIODIC") == 0) {
    bc_s.sE = PERIODIC;
  } else if(strcmp(buf, "DIRICHLET") == 0) {
    bc_s.sE = DIRICHLET;
    fret = fscanf(infile, "%lf", &bc_s.sED);
  } else if(strcmp(buf, "NEUMANN") == 0) {
    bc_s.sE = NEUMANN;
    fret = fscanf(infile, "%lf", &bc_s.sEN);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc_s.sS %s", buf);
  if(strcmp(buf, "PERIODIC") == 0) {
    bc_s.sS = PERIODIC;
  } else if(strcmp(buf, "DIRICHLET") == 0) {
    bc_s.sS = DIRICHLET;
    fret = fscanf(infile, "%lf", &bc_s.sSD);
  } else if(strcmp(buf, "NEUMANN") == 0) {
    bc_s.sS = NEUMANN;
    fret = fscanf(infile, "%lf", &bc_s.sSN);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc_s.sN %s", buf);
  if(strcmp(buf, "PERIODIC") == 0) {
    bc_s.sN = PERIODIC;
  } else if(strcmp(buf, "DIRICHLET") == 0) {
    bc_s.sN = DIRICHLET;
    fret = fscanf(infile, "%lf", &bc_s.sND);
  } else if(strcmp(buf, "NEUMANN") == 0) {
    bc_s.sN = NEUMANN;
    fret = fscanf(infile, "%lf", &bc_s.sNN);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc_s.sB %s", buf);
  if(strcmp(buf, "PERIODIC") == 0) {
    bc_s.sB = PERIODIC;
  } else if(strcmp(buf, "DIRICHLET") == 0) {
    bc_s.sB = DIRICHLET;
    fret = fscanf(infile, "%lf", &bc_s.sBD);
  } else if(strcmp(buf, "NEUMANN") == 0) {
    bc_s.sB = NEUMANN;
    fret = fscanf(infile, "%lf", &bc_s.sBN);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "bc_s.sT %s", buf);
  if(strcmp(buf, "PERIODIC") == 0) {
    bc_s.sT = PERIODIC;
  } else if(strcmp(buf, "DIRICHLET") == 0) {
    bc_s.sT = DIRICHLET;
    fret = fscanf(infile, "%lf", &bc_s.sTD);
  } else if(strcmp(buf, "NEUMANN") == 0) {
    bc_s.sT = NEUMANN;
    fret = fscanf(infile, "%lf", &bc_s.sTN);
  } else {
    fprintf(stderr, "flow.config read error on line %d.\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "\n");
  /* Read initial flow condition */
  fret = fscanf(infile, "INITIAL CONDITION\n");
  fret = fscanf(infile, "init_cond %s", buf);
  if (strcmp(buf, "QUIESCENT") == 0)
    init_cond = QUIESCENT;
  else if (strcmp(buf, "SHEAR") == 0)
    init_cond = SHEAR;
  else if (strcmp(buf, "CHANNEL") == 0)
    init_cond = CHANNEL;
  else if (strcmp(buf, "TAYLOR_GREEN") == 0)
    init_cond = TAYLOR_GREEN;
  else if (strcmp(buf, "TAYLOR_GREEN_3") == 0)
    init_cond = TAYLOR_GREEN_3;
  else if (strcmp(buf, "TURB_CHANNEL") == 0)
    init_cond = TURB_CHANNEL;
  else if (strcmp(buf, "TURBULENT") == 0)
    init_cond = TURBULENT;
  else {
    fprintf(stderr, "flow.config read error on line %d \"%s\".\n",
      __LINE__, buf);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "init_s %lf %lf\n", &s_init, &s_init_rand);

  fret = fscanf(infile, "\n");
  /* Read outplane */
  fret = fscanf(infile, "SOLVABILITY ENFORCEMENT PLANE\n");
  fret = fscanf(infile, "out_plane %s", buf);
  if(strcmp(buf, "WEST") == 0)
    out_plane = WEST;
  else if(strcmp(buf, "EAST") == 0)
    out_plane = EAST;
  else if(strcmp(buf, "SOUTH") == 0)
    out_plane = SOUTH;
  else if(strcmp(buf, "NORTH") == 0)
    out_plane = NORTH;
  else if(strcmp(buf, "BOTTOM") == 0)
    out_plane = BOTTOM;
  else if(strcmp(buf, "TOP") == 0)
    out_plane = TOP;
  else if(strcmp(buf, "HOMOGENEOUS") == 0)
    out_plane = HOMOGENEOUS;
  else {
    fprintf(stderr, "flow.config read error on line %d \"%s\".\n",
      __LINE__, buf);
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "\n");
  /* Body force application */
  gradP.x = 0.;
  gradP.y = 0.;
  gradP.z = 0.;
  g.x = 0.;
  g.y = 0.;
  g.z = 0.;
  pid_int = 0.;
  pid_back = 0.;
  fret = fscanf(infile, "SIMULATION DRIVING CONDITIONS\n");
#ifdef DOUBLE
  fret = fscanf(infile, "p_bc_tdelay %lf\n", &p_bc_tdelay);
  fret = fscanf(infile, "gradP.x %lf %lf\n", &gradP.xm, &gradP.xa);
  fret = fscanf(infile, "gradP.y %lf %lf\n", &gradP.ym, &gradP.ya);
  fret = fscanf(infile, "gradP.z %lf %lf\n", &gradP.zm, &gradP.za);
  fret = fscanf(infile, "g_bc_tdelay %lf\n", &g_bc_tdelay);
  fret = fscanf(infile, "g.x %lf %lf\n", &g.xm, &g.xa);
  fret = fscanf(infile, "g.y %lf %lf\n", &g.ym, &g.ya);
  fret = fscanf(infile, "g.z %lf %lf\n", &g.zm, &g.za);
#else
  fret = fscanf(infile, "p_bc_tdelay %f\n", &p_bc_tdelay);
  fret = fscanf(infile, "gradP.x %f %f\n", &gradP.xm, &gradP.xa);
  fret = fscanf(infile, "gradP.y %f %f\n", &gradP.ym, &gradP.ya);
  fret = fscanf(infile, "gradP.z %f %f\n", &gradP.zm, &gradP.za);
  fret = fscanf(infile, "g_bc_tdelay %f\n", &g_bc_tdelay);
  fret = fscanf(infile, "g.x %f %f\n", &g.xm, &g.xa);
  fret = fscanf(infile, "g.y %f %f\n", &g.ym, &g.ya);
  fret = fscanf(infile, "g.z %f %f\n", &g.zm, &g.za);
#endif
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "PID CONTROLLER GAINS\n");
  fret = fscanf(infile, "Kp %lf\n", &Kp);
  fret = fscanf(infile, "Ki %lf\n", &Ki);
  fret = fscanf(infile, "Kd %lf\n", &Kd);

  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "TURBULENT FORCING PARAMETERS\n");
  fret = fscanf(infile, "turbA %lf\n", &turbA);
  fret = fscanf(infile, "osci_f %lf\n", &osci_f);

  fclose(infile);

  // Check that subdomains fill up global domain
  //int subI = 0, subJ = 0, subK = 0;
  //for (int i = 0; i < DOM.S3; i++) {
  //  subI += dom[i].I;
  //  subJ += dom[i].J;
  //  subK += dom[i].K;
  //}
  //if ((subI + 1) != DOM.In) {
  //  if (rank == 0) {
  //    printf("  The subdomain indices in x do not match that specified for the"
  //           " global domain\n");
  //  }
  //  exit(EXIT_FAILURE);
  //} else if ((subJ + 1) != DOM.Jn) {
  //  if (rank == 0) {
  //    printf("  The subdomain indices in y do not match that specified for the"
  //           " global domain\n");
  //  }
  //  exit(EXIT_FAILURE);
  //} else if ((subK + 1) != DOM.Kn) {
  //  if (rank == 0) {
  //    printf("  The subdomain indices in z do not match that specified for the"
  //           " global domain\n");
  //  }
  //  exit(EXIT_FAILURE);
  //}
}

void domain_fill(void)
{
  //printf("N%d >> Initializing dom_struct...\n", rank);
  int i,j,k;  // iterators

  /* Global Domain */
  // calculate domain sizes
  DOM.xl = DOM.xe - DOM.xs;
  DOM.yl = DOM.ye - DOM.ys;
  DOM.zl = DOM.ze - DOM.zs;

  // calculate cell sizes
  DOM.dx = DOM.xl / DOM.xn;
  DOM.dy = DOM.yl / DOM.yn;
  DOM.dz = DOM.zl / DOM.zn;

  // set up grids
  // Gcc
  DOM.Gcc.is = DOM_BUF;
  DOM.Gcc.isb = DOM.Gcc.is - DOM_BUF;
  DOM.Gcc.in = DOM.xn;
  DOM.Gcc.inb = DOM.Gcc.in + 2 * DOM_BUF;
  DOM.Gcc.ie = DOM.Gcc.isb + DOM.Gcc.in;
  DOM.Gcc.ieb = DOM.Gcc.ie + DOM_BUF;

  DOM.Gcc.js = DOM_BUF;
  DOM.Gcc.jsb = DOM.Gcc.js - DOM_BUF;
  DOM.Gcc.jn = DOM.yn;
  DOM.Gcc.jnb = DOM.Gcc.jn + 2 * DOM_BUF;
  DOM.Gcc.je = DOM.Gcc.jsb + DOM.Gcc.jn;
  DOM.Gcc.jeb = DOM.Gcc.je + DOM_BUF;

  DOM.Gcc.ks = DOM_BUF;
  DOM.Gcc.ksb = DOM.Gcc.ks - DOM_BUF;
  DOM.Gcc.kn = DOM.zn;
  DOM.Gcc.knb = DOM.Gcc.kn + 2 * DOM_BUF;
  DOM.Gcc.ke = DOM.Gcc.ksb + DOM.Gcc.kn;
  DOM.Gcc.keb = DOM.Gcc.ke + DOM_BUF;

  DOM.Gcc._is = DOM.Gcc.is;
  DOM.Gcc._isb = DOM.Gcc.isb;
  DOM.Gcc._ie = DOM.Gcc.ie;
  DOM.Gcc._ieb = DOM.Gcc.ieb;

  DOM.Gcc._js = DOM.Gcc.js;
  DOM.Gcc._jsb = DOM.Gcc.jsb;
  DOM.Gcc._je = DOM.Gcc.je;
  DOM.Gcc._jeb = DOM.Gcc.jeb;

  DOM.Gcc._ks = DOM.Gcc.ks;
  DOM.Gcc._ksb = DOM.Gcc.ksb;
  DOM.Gcc._ke = DOM.Gcc.ke;
  DOM.Gcc._keb = DOM.Gcc.keb;

  DOM.Gcc.s1 = DOM.Gcc.in;
  DOM.Gcc.s2 = DOM.Gcc.s1 * DOM.Gcc.jn;
  DOM.Gcc.s3 = DOM.Gcc.s2 * DOM.Gcc.kn;
  DOM.Gcc.s1b = DOM.Gcc.inb;
  DOM.Gcc.s2b = DOM.Gcc.s1b * DOM.Gcc.jnb;
  DOM.Gcc.s3b = DOM.Gcc.s2b * DOM.Gcc.knb;

  DOM.Gcc.s2_i = DOM.Gcc.jn * DOM.Gcc.kn;
  DOM.Gcc.s2_j = DOM.Gcc.in * DOM.Gcc.kn;
  DOM.Gcc.s2_k = DOM.Gcc.in * DOM.Gcc.jn;
  DOM.Gcc.s2b_i = DOM.Gcc.jnb * DOM.Gcc.knb;
  DOM.Gcc.s2b_j = DOM.Gcc.inb * DOM.Gcc.knb;
  DOM.Gcc.s2b_k = DOM.Gcc.inb * DOM.Gcc.jnb;

  // Gfx
  DOM.Gfx.is = DOM_BUF;
  DOM.Gfx.isb = DOM.Gfx.is - DOM_BUF;
  DOM.Gfx.in = DOM.xn + 1;
  DOM.Gfx.inb = DOM.Gfx.in + 2 * DOM_BUF;
  DOM.Gfx.ie = DOM.Gfx.isb + DOM.Gfx.in;
  DOM.Gfx.ieb = DOM.Gfx.ie + DOM_BUF;

  DOM.Gfx.js = DOM_BUF;
  DOM.Gfx.jsb = DOM.Gfx.js - DOM_BUF;
  DOM.Gfx.jn = DOM.yn;
  DOM.Gfx.jnb = DOM.Gfx.jn + 2 * DOM_BUF;
  DOM.Gfx.je = DOM.Gfx.jsb + DOM.Gfx.jn;
  DOM.Gfx.jeb = DOM.Gfx.je + DOM_BUF;

  DOM.Gfx.ks = DOM_BUF;
  DOM.Gfx.ksb = DOM.Gfx.ks - DOM_BUF;
  DOM.Gfx.kn = DOM.zn;
  DOM.Gfx.knb = DOM.Gfx.kn + 2 * DOM_BUF;
  DOM.Gfx.ke = DOM.Gfx.ksb + DOM.Gfx.kn;
  DOM.Gfx.keb = DOM.Gfx.ke + DOM_BUF;

  DOM.Gfx._is = DOM.Gfx.is;
  DOM.Gfx._isb = DOM.Gfx.isb;
  DOM.Gfx._ie = DOM.Gfx.ie;
  DOM.Gfx._ieb = DOM.Gfx.ieb;

  DOM.Gfx._js = DOM.Gfx.js;
  DOM.Gfx._jsb = DOM.Gfx.jsb;
  DOM.Gfx._je = DOM.Gfx.je;
  DOM.Gfx._jeb = DOM.Gfx.jeb;

  DOM.Gfx._ks = DOM.Gfx.ks;
  DOM.Gfx._ksb = DOM.Gfx.ksb;
  DOM.Gfx._ke = DOM.Gfx.ke;
  DOM.Gfx._keb = DOM.Gfx.keb;

  DOM.Gfx.s1 = DOM.Gfx.jn;
  DOM.Gfx.s2 = DOM.Gfx.s1 * DOM.Gfx.kn;
  DOM.Gfx.s3 = DOM.Gfx.s2 * DOM.Gfx.in;
  DOM.Gfx.s1b = DOM.Gfx.jnb;
  DOM.Gfx.s2b = DOM.Gfx.s1b * DOM.Gfx.knb;
  DOM.Gfx.s3b = DOM.Gfx.s2b * DOM.Gfx.inb;

  DOM.Gfx.s2_i = DOM.Gfx.jn * DOM.Gfx.kn;
  DOM.Gfx.s2_j = DOM.Gfx.in * DOM.Gfx.kn;
  DOM.Gfx.s2_k = DOM.Gfx.in * DOM.Gfx.jn;
  DOM.Gfx.s2b_i = DOM.Gfx.jnb * DOM.Gfx.knb;
  DOM.Gfx.s2b_j = DOM.Gfx.inb * DOM.Gfx.knb;
  DOM.Gfx.s2b_k = DOM.Gfx.inb * DOM.Gfx.jnb;

  // Gfy
  DOM.Gfy.is = DOM_BUF;
  DOM.Gfy.isb = DOM.Gfy.is - DOM_BUF;
  DOM.Gfy.in = DOM.xn;
  DOM.Gfy.inb = DOM.Gfy.in + 2 * DOM_BUF;
  DOM.Gfy.ie = DOM.Gfy.isb + DOM.Gfy.in;
  DOM.Gfy.ieb = DOM.Gfy.ie + DOM_BUF;

  DOM.Gfy.js = DOM_BUF;
  DOM.Gfy.jsb = DOM.Gfy.js - DOM_BUF;
  DOM.Gfy.jn = DOM.yn + 1;
  DOM.Gfy.jnb = DOM.Gfy.jn + 2 * DOM_BUF;
  DOM.Gfy.je = DOM.Gfy.jsb + DOM.Gfy.jn;
  DOM.Gfy.jeb = DOM.Gfy.je + DOM_BUF;

  DOM.Gfy.ks = DOM_BUF;
  DOM.Gfy.ksb = DOM.Gfy.ks - DOM_BUF;
  DOM.Gfy.kn = DOM.zn;
  DOM.Gfy.knb = DOM.Gfy.kn + 2 * DOM_BUF;
  DOM.Gfy.ke = DOM.Gfy.ksb + DOM.Gfy.kn;
  DOM.Gfy.keb = DOM.Gfy.ke + DOM_BUF;

  DOM.Gfy._is = DOM.Gfy.is;
  DOM.Gfy._isb = DOM.Gfy.isb;
  DOM.Gfy._ie = DOM.Gfy.ie;
  DOM.Gfy._ieb = DOM.Gfy.ieb;

  DOM.Gfy._js = DOM.Gfy.js;
  DOM.Gfy._jsb = DOM.Gfy.jsb;
  DOM.Gfy._je = DOM.Gfy.je;
  DOM.Gfy._jeb = DOM.Gfy.jeb;

  DOM.Gfy._ks = DOM.Gfy.ks;
  DOM.Gfy._ksb = DOM.Gfy.ksb;
  DOM.Gfy._ke = DOM.Gfy.ke;
  DOM.Gfy._keb = DOM.Gfy.keb;

  DOM.Gfy.s1 = DOM.Gfy.kn;
  DOM.Gfy.s2 = DOM.Gfy.s1 * DOM.Gfy.in;
  DOM.Gfy.s3 = DOM.Gfy.s2 * DOM.Gfy.jn;
  DOM.Gfy.s1b = DOM.Gfy.knb;
  DOM.Gfy.s2b = DOM.Gfy.s1b * DOM.Gfy.inb;
  DOM.Gfy.s3b = DOM.Gfy.s2b * DOM.Gfy.jnb;

  DOM.Gfy.s2_i = DOM.Gfy.jn * DOM.Gfy.kn;
  DOM.Gfy.s2_j = DOM.Gfy.in * DOM.Gfy.kn;
  DOM.Gfy.s2_k = DOM.Gfy.in * DOM.Gfy.jn;
  DOM.Gfy.s2b_i = DOM.Gfy.jnb * DOM.Gfy.knb;
  DOM.Gfy.s2b_j = DOM.Gfy.inb * DOM.Gfy.knb;
  DOM.Gfy.s2b_k = DOM.Gfy.inb * DOM.Gfy.jnb;

  // Gfz
  DOM.Gfz.is = DOM_BUF;
  DOM.Gfz.isb = DOM.Gfz.is - DOM_BUF;
  DOM.Gfz.in = DOM.xn;
  DOM.Gfz.inb = DOM.Gfz.in + 2 * DOM_BUF;
  DOM.Gfz.ie = DOM.Gfz.isb + DOM.Gfz.in;
  DOM.Gfz.ieb = DOM.Gfz.ie + DOM_BUF;

  DOM.Gfz.js = DOM_BUF;
  DOM.Gfz.jsb = DOM.Gfz.js - DOM_BUF;
  DOM.Gfz.jn = DOM.yn;
  DOM.Gfz.jnb = DOM.Gfz.jn + 2 * DOM_BUF;
  DOM.Gfz.je = DOM.Gfz.jsb + DOM.Gfz.jn;
  DOM.Gfz.jeb = DOM.Gfz.je + DOM_BUF;

  DOM.Gfz.ks = DOM_BUF;
  DOM.Gfz.ksb = DOM.Gfz.ks - DOM_BUF;
  DOM.Gfz.kn = DOM.zn + 1;
  DOM.Gfz.knb = DOM.Gfz.kn + 2 * DOM_BUF;
  DOM.Gfz.ke = DOM.Gfz.ksb + DOM.Gfz.kn;
  DOM.Gfz.keb = DOM.Gfz.ke + DOM_BUF;

  DOM.Gfz._is = DOM.Gfz.is;
  DOM.Gfz._isb = DOM.Gfz.isb;
  DOM.Gfz._ie = DOM.Gfz.ie;
  DOM.Gfz._ieb = DOM.Gfz.ieb;

  DOM.Gfz._js = DOM.Gfz.js;
  DOM.Gfz._jsb = DOM.Gfz.jsb;
  DOM.Gfz._je = DOM.Gfz.je;
  DOM.Gfz._jeb = DOM.Gfz.jeb;

  DOM.Gfz._ks = DOM.Gfz.ks;
  DOM.Gfz._ksb = DOM.Gfz.ksb;
  DOM.Gfz._ke = DOM.Gfz.ke;
  DOM.Gfz._keb = DOM.Gfz.keb;

  DOM.Gfz.s1 = DOM.Gfz.in;
  DOM.Gfz.s2 = DOM.Gfz.s1 * DOM.Gfz.jn;
  DOM.Gfz.s3 = DOM.Gfz.s2 * DOM.Gfz.kn;
  DOM.Gfz.s1b = DOM.Gfz.inb;
  DOM.Gfz.s2b = DOM.Gfz.s1b * DOM.Gfz.jnb;
  DOM.Gfz.s3b = DOM.Gfz.s2b * DOM.Gfz.knb;

  DOM.Gfz.s2_i = DOM.Gfz.jn * DOM.Gfz.kn;
  DOM.Gfz.s2_j = DOM.Gfz.in * DOM.Gfz.kn;
  DOM.Gfz.s2_k = DOM.Gfz.in * DOM.Gfz.jn;
  DOM.Gfz.s2b_i = DOM.Gfz.jnb * DOM.Gfz.knb;
  DOM.Gfz.s2b_j = DOM.Gfz.inb * DOM.Gfz.knb;
  DOM.Gfz.s2b_k = DOM.Gfz.inb * DOM.Gfz.jnb;

  /* Configure MPI Subdomains */
  DOM.Is = 0;
  DOM.Ie = DOM.In - 1;
  DOM.Js = 0;
  DOM.Je = DOM.Jn - 1;
  DOM.Ks = 0;
  DOM.Ke = DOM.Kn - 1;

  for (i = 0; i < DOM.S3; i++) {

    // Set adjacent subdomains using domain strides
    // Domain boundary == -1
    if (dom[i].I == DOM.Is) {     // west -- if on the edge, check BC
      if (bc.pW == PERIODIC) {    // if periodic, flip to other side
        dom[i].w = DOM.Ie + dom[i].J * DOM.S1 + dom[i].K * DOM.S2;
      } else {
        dom[i].w = MPI_PROC_NULL; // else, set to MPI_PROC_NULL
      }
    } else {                      // if in middle, is simple
      dom[i].w = (dom[i].I - 1) + dom[i].J * DOM.S1 + dom[i].K * DOM.S2;
    }

    if (dom[i].I == DOM.Ie) {    // east
      if (bc.pE == PERIODIC) {   // periodic flip
        dom[i].e = DOM.Is + dom[i].J * DOM.S1 + dom[i].K * DOM.S2;
      } else {                   // edge
        dom[i].e = MPI_PROC_NULL;
      }
    } else {                     // middle
      dom[i].e = (dom[i].I + 1) + dom[i].J * DOM.S1 + dom[i].K * DOM.S2;
    }

    if (dom[i].J == DOM.Js) {    // south
      if (bc.pS == PERIODIC) {
        dom[i].s = dom[i].I + DOM.Je * DOM.S1 + dom[i].K * DOM.S2;
      } else {
        dom[i].s = MPI_PROC_NULL;
      }
    } else {
      dom[i].s = dom[i].I + (dom[i].J - 1) * DOM.S1 + dom[i].K * DOM.S2;
    }

    if (dom[i].J == DOM.Je) {    // north
      if (bc.pN == PERIODIC) {
        dom[i].n = dom[i].I + DOM.Js * DOM.S1 + dom[i].K * DOM.S2;
      } else {
        dom[i].n = MPI_PROC_NULL;
      }
    } else {
      dom[i].n = dom[i].I + (dom[i].J + 1) * DOM.S1 + dom[i].K * DOM.S2;
    }

    if (dom[i].K == DOM.Ks) {    // bottom
      if (bc.pB == PERIODIC) {
        dom[i].b = dom[i].I + dom[i].J * DOM.S1 + DOM.Ke * DOM.S2;
      } else {
        dom[i].b = MPI_PROC_NULL;
      }
    } else {
      dom[i].b = dom[i].I + dom[i].J * DOM.S1 + (dom[i].K - 1) * DOM.S2;
    }

    if (dom[i].K == DOM.Ke) {    // top
      if (bc.pT == PERIODIC) {
        dom[i].t = dom[i].I + dom[i].J * DOM.S1 + DOM.Ks * DOM.S2;
      } else {
        dom[i].t = MPI_PROC_NULL;
      }
    } else {
      dom[i].t = dom[i].I + dom[i].J * DOM.S1 + (dom[i].K + 1) * DOM.S2;
    }
  }

  // Set lengths and indices
  int c;
  for (k = 0; k < DOM.Kn; k++) {
    for (j = 0; j < DOM.Jn; j++) {
      for (i = 0; i < DOM.In; i++) {
        c = GCC_LOC(i, j, k, DOM.S1, DOM.S2);

        dom[c].xl = dom[c].xe - dom[c].xs;
        dom[c].yl = dom[c].ye - dom[c].ys;
        dom[c].zl = dom[c].ze - dom[c].zs;
        dom[c].dx = dom[c].xl / dom[c].xn;
        dom[c].dy = dom[c].yl / dom[c].yn;
        dom[c].dz = dom[c].zl / dom[c].zn;

        // Gcc
        // If not the edge, use neighboring dom to find .is
        if (i == DOM.Is) {
          dom[c].Gcc.is = DOM_BUF; 
        } else {
          dom[c].Gcc.is = dom[dom[i].w].Gcc.ie + 1;
        }
        dom[c].Gcc.isb = dom[c].Gcc.is - DOM_BUF;
        dom[c].Gcc.in = dom[c].xn;
        dom[c].Gcc.inb = dom[c].Gcc.in + 2 * DOM_BUF;
        dom[c].Gcc.ie = dom[c].Gcc.isb + dom[c].Gcc.in;
        dom[c].Gcc.ieb = dom[c].Gcc.ie + DOM_BUF;

        if (j == DOM.Js) {
          dom[c].Gcc.js = DOM_BUF;
        } else {
          dom[c].Gcc.js = dom[dom[c].s].Gcc.je + 1;
        }
        dom[c].Gcc.jsb = dom[c].Gcc.js - DOM_BUF;
        dom[c].Gcc.jn = dom[c].yn;
        dom[c].Gcc.jnb = dom[c].Gcc.jn + 2 * DOM_BUF;
        dom[c].Gcc.je = dom[c].Gcc.jsb + dom[c].Gcc.jn;
        dom[c].Gcc.jeb = dom[c].Gcc.je + DOM_BUF;

        if (k == DOM.Ks) {
          dom[c].Gcc.ks = DOM_BUF;
        } else {
          dom[c].Gcc.ks = dom[dom[c].b].Gcc.ke + 1;
        }
        dom[c].Gcc.ksb = dom[c].Gcc.ks - DOM_BUF;
        dom[c].Gcc.kn = dom[c].zn;
        dom[c].Gcc.knb = dom[c].Gcc.kn + 2 * DOM_BUF;
        dom[c].Gcc.ke = dom[c].Gcc.ksb + dom[c].Gcc.kn;
        dom[c].Gcc.keb = dom[c].Gcc.ke + DOM_BUF;

        // Local indexing
        dom[c].Gcc._is = DOM_BUF;
        dom[c].Gcc._isb = dom[c].Gcc._is - DOM_BUF;
        dom[c].Gcc._ie = dom[c].Gcc._isb + dom[c].Gcc.in;
        dom[c].Gcc._ieb = dom[c].Gcc._ie + DOM_BUF;

        dom[c].Gcc._js = DOM_BUF;
        dom[c].Gcc._jsb = dom[c].Gcc._js - DOM_BUF;
        dom[c].Gcc._je = dom[c].Gcc._jsb + dom[c].Gcc.jn;
        dom[c].Gcc._jeb = dom[c].Gcc._je + DOM_BUF;

        dom[c].Gcc._ks = DOM_BUF;
        dom[c].Gcc._ksb = dom[c].Gcc._ks - DOM_BUF;
        dom[c].Gcc._ke = dom[c].Gcc._ksb + dom[c].Gcc.kn;
        dom[c].Gcc._keb = dom[c].Gcc._ke + DOM_BUF;

        dom[c].Gcc.s1 = dom[c].Gcc.in;
        dom[c].Gcc.s2 = dom[c].Gcc.s1 * dom[c].Gcc.jn;
        dom[c].Gcc.s3 = dom[c].Gcc.s2 * dom[c].Gcc.kn;
        dom[c].Gcc.s1b = dom[c].Gcc.inb;
        dom[c].Gcc.s2b = dom[c].Gcc.s1b * dom[c].Gcc.jnb;
        dom[c].Gcc.s3b = dom[c].Gcc.s2b * dom[c].Gcc.knb;

        dom[c].Gcc.s2_i = dom[c].Gcc.jn * dom[c].Gcc.kn;
        dom[c].Gcc.s2_j = dom[c].Gcc.in * dom[c].Gcc.kn;
        dom[c].Gcc.s2_k = dom[c].Gcc.in * dom[c].Gcc.jn;
        dom[c].Gcc.s2b_i = dom[c].Gcc.jnb * dom[c].Gcc.knb;
        dom[c].Gcc.s2b_j = dom[c].Gcc.inb * dom[c].Gcc.knb;
        dom[c].Gcc.s2b_k = dom[c].Gcc.inb * dom[c].Gcc.jnb;

        // Gfx
        if (i == DOM.Is) {
          dom[c].Gfx.is = DOM_BUF;
        } else {
          dom[c].Gfx.is = dom[dom[c].w].Gfx.ie;
        }
        dom[c].Gfx.isb = dom[c].Gfx.is - DOM_BUF;
        dom[c].Gfx.in = dom[c].xn + 1;
        dom[c].Gfx.inb = dom[c].Gfx.in + 2 * DOM_BUF;
        dom[c].Gfx.ie = dom[c].Gfx.isb + dom[c].Gfx.in;
        dom[c].Gfx.ieb = dom[c].Gfx.ie + DOM_BUF;

        if (j == DOM.Js) {
          dom[c].Gfx.js = DOM_BUF;
        } else {
          dom[c].Gfx.js = dom[dom[c].s].Gfx.je + 1;
        }
        dom[c].Gfx.jsb = dom[c].Gfx.js - DOM_BUF;
        dom[c].Gfx.jn = dom[c].yn;
        dom[c].Gfx.jnb = dom[c].Gfx.jn + 2 * DOM_BUF;
        dom[c].Gfx.je = dom[c].Gfx.jsb + dom[c].Gfx.jn;
        dom[c].Gfx.jeb = dom[c].Gfx.je + DOM_BUF;

        if (k == DOM.Ks) {
          dom[c].Gfx.ks = DOM_BUF;
        } else {
          dom[c].Gfx.ks = dom[dom[c].b].Gfx.ke + 1;
        }
        dom[c].Gfx.ksb = dom[c].Gfx.ks - DOM_BUF;
        dom[c].Gfx.kn = dom[c].zn;
        dom[c].Gfx.knb = dom[c].Gfx.kn + 2 * DOM_BUF;
        dom[c].Gfx.ke = dom[c].Gfx.ksb + dom[c].Gfx.kn;
        dom[c].Gfx.keb = dom[c].Gfx.ke + DOM_BUF;

        dom[c].Gfx._is = DOM_BUF;
        dom[c].Gfx._isb = dom[c].Gfx._is - DOM_BUF;
        dom[c].Gfx._ie = dom[c].Gfx._isb + dom[c].Gfx.in;
        dom[c].Gfx._ieb = dom[c].Gfx._ie + DOM_BUF;

        dom[c].Gfx._js = DOM_BUF;
        dom[c].Gfx._jsb = dom[c].Gfx._js - DOM_BUF;
        dom[c].Gfx._je = dom[c].Gfx._jsb + dom[c].Gfx.jn;
        dom[c].Gfx._jeb = dom[c].Gfx._je + DOM_BUF;

        dom[c].Gfx._ks = DOM_BUF;
        dom[c].Gfx._ksb = dom[c].Gfx._ks - DOM_BUF;
        dom[c].Gfx._ke = dom[c].Gfx._ksb + dom[c].Gfx.kn;
        dom[c].Gfx._keb = dom[c].Gfx._ke + DOM_BUF;

        dom[c].Gfx.s1 = dom[c].Gfx.jn;
        dom[c].Gfx.s2 = dom[c].Gfx.s1 * dom[c].Gfx.kn;
        dom[c].Gfx.s3 = dom[c].Gfx.s2 * dom[c].Gfx.in;
        dom[c].Gfx.s1b = dom[c].Gfx.jnb;
        dom[c].Gfx.s2b = dom[c].Gfx.s1b * dom[c].Gfx.knb;
        dom[c].Gfx.s3b = dom[c].Gfx.s2b * dom[c].Gfx.inb;

        dom[c].Gfx.s2_i = dom[c].Gfx.jn * dom[c].Gfx.kn;
        dom[c].Gfx.s2_j = dom[c].Gfx.in * dom[c].Gfx.kn;
        dom[c].Gfx.s2_k = dom[c].Gfx.in * dom[c].Gfx.jn;
        dom[c].Gfx.s2b_i = dom[c].Gfx.jnb * dom[c].Gfx.knb;
        dom[c].Gfx.s2b_j = dom[c].Gfx.inb * dom[c].Gfx.knb;
        dom[c].Gfx.s2b_k = dom[c].Gfx.inb * dom[c].Gfx.jnb;

        // Gfy
        if (i == DOM.Is) {
          dom[c].Gfy.is = DOM_BUF;
        } else {
          dom[c].Gfy.is = dom[dom[c].w].Gfy.ie + 1;
        }
        dom[c].Gfy.isb = dom[c].Gfy.is - DOM_BUF;
        dom[c].Gfy.in = dom[c].xn;
        dom[c].Gfy.inb = dom[c].Gfy.in + 2 * DOM_BUF;
        dom[c].Gfy.ie = dom[c].Gfy.isb + dom[c].Gfy.in;
        dom[c].Gfy.ieb = dom[c].Gfy.ie + DOM_BUF;

        if (j == DOM.Js) {
          dom[c].Gfy.js = DOM_BUF;
        } else {
          dom[c].Gfy.js = dom[dom[c].s].Gfy.je;
        }
        dom[c].Gfy.jsb = dom[c].Gfy.js - DOM_BUF;
        dom[c].Gfy.jn = dom[c].yn + 1;
        dom[c].Gfy.jnb = dom[c].Gfy.jn + 2 * DOM_BUF;
        dom[c].Gfy.je = dom[c].Gfy.jsb + dom[c].Gfy.jn;
        dom[c].Gfy.jeb = dom[c].Gfy.je + DOM_BUF;

        if (k == DOM.Ks) {
          dom[c].Gfy.ks = DOM_BUF;
        } else {
          dom[c].Gfy.ks = dom[dom[c].b].Gfy.ke + 1;
        }
        dom[c].Gfy.ksb = dom[c].Gfy.ks - DOM_BUF;
        dom[c].Gfy.kn = dom[c].zn;
        dom[c].Gfy.knb = dom[c].Gfy.kn + 2 * DOM_BUF;
        dom[c].Gfy.ke = dom[c].Gfy.ksb + dom[c].Gfy.kn;
        dom[c].Gfy.keb = dom[c].Gfy.ke + DOM_BUF;

        dom[c].Gfy._is = DOM_BUF;
        dom[c].Gfy._isb = dom[c].Gfy._is - DOM_BUF;
        dom[c].Gfy._ie = dom[c].Gfy._isb + dom[c].Gfy.in;
        dom[c].Gfy._ieb = dom[c].Gfy._ie + DOM_BUF;

        dom[c].Gfy._js = DOM_BUF;
        dom[c].Gfy._jsb = dom[c].Gfy._js - DOM_BUF;
        dom[c].Gfy._je = dom[c].Gfy._jsb + dom[c].Gfy.jn;
        dom[c].Gfy._jeb = dom[c].Gfy._je + DOM_BUF;

        dom[c].Gfy._ks = DOM_BUF;
        dom[c].Gfy._ksb = dom[c].Gfy._ks - DOM_BUF;
        dom[c].Gfy._ke = dom[c].Gfy._ksb + dom[c].Gfy.kn;
        dom[c].Gfy._keb = dom[c].Gfy._ke + DOM_BUF;

        dom[c].Gfy.s1 = dom[c].Gfy.kn;
        dom[c].Gfy.s2 = dom[c].Gfy.s1 * dom[c].Gfy.in;
        dom[c].Gfy.s3 = dom[c].Gfy.s2 * dom[c].Gfy.jn;
        dom[c].Gfy.s1b = dom[c].Gfy.knb;
        dom[c].Gfy.s2b = dom[c].Gfy.s1b * dom[c].Gfy.inb;
        dom[c].Gfy.s3b = dom[c].Gfy.s2b * dom[c].Gfy.jnb;

        dom[c].Gfy.s2_i = dom[c].Gfy.jn * dom[c].Gfy.kn;
        dom[c].Gfy.s2_j = dom[c].Gfy.in * dom[c].Gfy.kn;
        dom[c].Gfy.s2_k = dom[c].Gfy.in * dom[c].Gfy.jn;
        dom[c].Gfy.s2b_i = dom[c].Gfy.jnb * dom[c].Gfy.knb;
        dom[c].Gfy.s2b_j = dom[c].Gfy.inb * dom[c].Gfy.knb;
        dom[c].Gfy.s2b_k = dom[c].Gfy.inb * dom[c].Gfy.jnb;

        // Gfz
        if (i == DOM.Is) {
          dom[c].Gfz.is = DOM_BUF;
        } else {
          dom[c].Gfz.is = dom[dom[c].w].Gfz.ie + 1;
        }
        dom[c].Gfz.isb = dom[c].Gfz.is - DOM_BUF;
        dom[c].Gfz.in = dom[c].xn;
        dom[c].Gfz.inb = dom[c].Gfz.in + 2 * DOM_BUF;
        dom[c].Gfz.ie = dom[c].Gfz.isb + dom[c].Gfz.in;
        dom[c].Gfz.ieb = dom[c].Gfz.ie + DOM_BUF;

        if (j == DOM.Js) {
          dom[c].Gfz.js = DOM_BUF;
        } else {
          dom[c].Gfz.js = dom[dom[c].s].Gfz.je + 1;
        }
        dom[c].Gfz.jsb = dom[c].Gfz.js - DOM_BUF;
        dom[c].Gfz.jn = dom[c].yn;
        dom[c].Gfz.jnb = dom[c].Gfz.jn + 2 * DOM_BUF;
        dom[c].Gfz.je = dom[c].Gfz.jsb + dom[c].Gfz.jn;
        dom[c].Gfz.jeb = dom[c].Gfz.je + DOM_BUF;

        if (k == DOM.Ks) {
          dom[c].Gfz.ks = DOM_BUF;
        } else {
          dom[c].Gfz.ks = dom[dom[c].b].Gfz.ke;
        }
        dom[c].Gfz.ksb = dom[c].Gfz.ks - DOM_BUF;
        dom[c].Gfz.kn = dom[c].zn + 1;
        dom[c].Gfz.knb = dom[c].Gfz.kn + 2 * DOM_BUF;
        dom[c].Gfz.ke = dom[c].Gfz.ksb + dom[c].Gfz.kn;
        dom[c].Gfz.keb = dom[c].Gfz.ke + DOM_BUF;

        dom[c].Gfz._is = DOM_BUF;
        dom[c].Gfz._isb = dom[c].Gfz._is - DOM_BUF;
        dom[c].Gfz._ie = dom[c].Gfz._isb + dom[c].Gfz.in;
        dom[c].Gfz._ieb = dom[c].Gfz._ie + DOM_BUF;

        dom[c].Gfz._js = DOM_BUF;
        dom[c].Gfz._jsb = dom[c].Gfz._js - DOM_BUF;
        dom[c].Gfz._je = dom[c].Gfz._jsb + dom[c].Gfz.jn;
        dom[c].Gfz._jeb = dom[c].Gfz._je + DOM_BUF;

        dom[c].Gfz._ks = DOM_BUF;
        dom[c].Gfz._ksb = dom[c].Gfz._ks - DOM_BUF;
        dom[c].Gfz._ke = dom[c].Gfz._ksb + dom[c].Gfz.kn;
        dom[c].Gfz._keb = dom[c].Gfz._ke + DOM_BUF;

        dom[c].Gfz.s1 = dom[c].Gfz.in;
        dom[c].Gfz.s2 = dom[c].Gfz.s1 * dom[c].Gfz.jn;
        dom[c].Gfz.s3 = dom[c].Gfz.s2 * dom[c].Gfz.kn;
        dom[c].Gfz.s1b = dom[c].Gfz.inb;
        dom[c].Gfz.s2b = dom[c].Gfz.s1b * dom[c].Gfz.jnb;
        dom[c].Gfz.s3b = dom[c].Gfz.s2b * dom[c].Gfz.knb;

        dom[c].Gfz.s2_i = dom[c].Gfz.jn * dom[c].Gfz.kn;
        dom[c].Gfz.s2_j = dom[c].Gfz.in * dom[c].Gfz.kn;
        dom[c].Gfz.s2_k = dom[c].Gfz.in * dom[c].Gfz.jn;
        dom[c].Gfz.s2b_i = dom[c].Gfz.jnb * dom[c].Gfz.knb;
        dom[c].Gfz.s2b_j = dom[c].Gfz.inb * dom[c].Gfz.knb;
        dom[c].Gfz.s2b_k = dom[c].Gfz.inb * dom[c].Gfz.jnb;
      }
    }
  }

  #ifdef DDEBUG
    domain_write_config();
  #endif
}

void domain_write_config(void)
{
  printf("N%d >> Writing subdomain map debug file...\n", rank);

  // Prep file for output -- one file for each rank
  char fname[CHAR_BUF_SIZE];
  sprintf(fname, "%s/rank-%d-map.debug", ROOT_DIR, rank);
  FILE *outfile = fopen(fname, "w");

  fprintf(outfile, "Domain:\n");
  fprintf(outfile, "  X: (%f, %f), dX = %f\n", DOM.xs, DOM.xe, DOM.dx);
  fprintf(outfile, "  Y: (%f, %f), dY = %f\n", DOM.ys, DOM.ye, DOM.dy);
  fprintf(outfile, "  Z: (%f, %f), dZ = %f\n", DOM.zs, DOM.ze, DOM.dz);
  fprintf(outfile, "  Xn = %d, Yn = %d, Zn = %d\n", DOM.xn, DOM.yn, DOM.zn);
  
  fprintf(outfile, "Mapping:\n");
  fprintf(outfile, "  (Is, Ie, In) = (%d, %d, %d)\n", DOM.Is, DOM.Ie, DOM.In);
  fprintf(outfile, "  (Js, Je, Jn) = (%d, %d, %d)\n", DOM.Js, DOM.Je, DOM.Jn);
  fprintf(outfile, "  (Ks, Ke, Kn) = (%d, %d, %d)\n", DOM.Ks, DOM.Ke, DOM.Kn);
  fprintf(outfile, "  (S1, S2, S3) = (%d, %d, %d)\n", DOM.S1, DOM.S2, DOM.S3);

  fprintf(outfile, "Domain Grids:\n");
  fprintf(outfile, "  DOM.Gcc:\n");
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", DOM.Gcc.is, DOM.Gcc.ie, DOM.Gcc.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", DOM.Gcc.isb, DOM.Gcc.ieb,
    DOM.Gcc.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", DOM.Gcc.js, DOM.Gcc.je, DOM.Gcc.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", DOM.Gcc.jsb, DOM.Gcc.jeb,
    DOM.Gcc.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", DOM.Gcc.ks, DOM.Gcc.ke, DOM.Gcc.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", DOM.Gcc.ksb, DOM.Gcc.keb,
    DOM.Gcc.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", DOM.Gcc._is, DOM.Gcc._ie,
    DOM.Gcc.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", DOM.Gcc._isb, DOM.Gcc._ieb,
    DOM.Gcc.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", DOM.Gcc._js, DOM.Gcc._je,
    DOM.Gcc.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", DOM.Gcc._jsb, DOM.Gcc._jeb,
    DOM.Gcc.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", DOM.Gcc._ks, DOM.Gcc._ke,
    DOM.Gcc.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", DOM.Gcc._ksb, DOM.Gcc._keb,
    DOM.Gcc.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", DOM.Gcc.s1, DOM.Gcc.s2,
    DOM.Gcc.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", DOM.Gcc.s1b, DOM.Gcc.s2b,
    DOM.Gcc.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", DOM.Gcc.s2_i, DOM.Gcc.s2_j, 
    DOM.Gcc.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", DOM.Gcc.s2b_i, DOM.Gcc.s2b_j, 
    DOM.Gcc.s2b_k);
  fprintf(outfile, "  DOM.Gfx:\n");
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", DOM.Gfx.is, DOM.Gfx.ie, DOM.Gfx.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", DOM.Gfx.isb, DOM.Gfx.ieb,
    DOM.Gfx.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", DOM.Gfx.js, DOM.Gfx.je, DOM.Gfx.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", DOM.Gfx.jsb, DOM.Gfx.jeb,
    DOM.Gfx.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", DOM.Gfx.ks, DOM.Gfx.ke, DOM.Gfx.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", DOM.Gfx.ksb, DOM.Gfx.keb,
    DOM.Gfx.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", DOM.Gfx._is, DOM.Gfx._ie,
    DOM.Gfx.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", DOM.Gfx._isb, DOM.Gfx._ieb,
    DOM.Gfx.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", DOM.Gfx._js, DOM.Gfx._je,
    DOM.Gfx.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", DOM.Gfx._jsb, DOM.Gfx._jeb,
    DOM.Gfx.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", DOM.Gfx._ks, DOM.Gfx._ke,
    DOM.Gfx.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", DOM.Gfx._ksb, DOM.Gfx._keb,
    DOM.Gfx.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", DOM.Gfx.s1, DOM.Gfx.s2,
    DOM.Gfx.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", DOM.Gfx.s1b, DOM.Gfx.s2b,
    DOM.Gfx.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", DOM.Gfx.s2_i, DOM.Gfx.s2_j, 
    DOM.Gfx.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", DOM.Gfx.s2b_i, DOM.Gfx.s2b_j, 
    DOM.Gfx.s2b_k);
  fprintf(outfile, "  DOM.Gfy:\n");
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", DOM.Gfy.is, DOM.Gfy.ie, DOM.Gfy.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", DOM.Gfy.isb, DOM.Gfy.ieb,
    DOM.Gfy.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", DOM.Gfy.js, DOM.Gfy.je, DOM.Gfy.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", DOM.Gfy.jsb, DOM.Gfy.jeb,
    DOM.Gfy.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", DOM.Gfy.ks, DOM.Gfy.ke, DOM.Gfy.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", DOM.Gfy.ksb, DOM.Gfy.keb,
    DOM.Gfy.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", DOM.Gfy._is, DOM.Gfy._ie,
    DOM.Gfy.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", DOM.Gfy._isb, DOM.Gfy._ieb,
    DOM.Gfy.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", DOM.Gfy._js, DOM.Gfy._je,
    DOM.Gfy.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", DOM.Gfy._jsb, DOM.Gfy._jeb,
    DOM.Gfy.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", DOM.Gfy._ks, DOM.Gfy._ke,
    DOM.Gfy.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", DOM.Gfy._ksb, DOM.Gfy._keb,
    DOM.Gfy.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", DOM.Gfy.s1, DOM.Gfy.s2,
    DOM.Gfy.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", DOM.Gfy.s1b, DOM.Gfy.s2b,
    DOM.Gfy.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", DOM.Gfy.s2_i, DOM.Gfy.s2_j, 
    DOM.Gfy.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", DOM.Gfy.s2b_i, DOM.Gfy.s2b_j, 
    DOM.Gfy.s2b_k);
  fprintf(outfile, "  DOM.Gfz:\n");
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", DOM.Gfz.is, DOM.Gfz.ie, DOM.Gfz.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", DOM.Gfz.isb, DOM.Gfz.ieb,
    DOM.Gfz.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", DOM.Gfz.js, DOM.Gfz.je, DOM.Gfz.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", DOM.Gfz.jsb, DOM.Gfz.jeb,
    DOM.Gfz.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", DOM.Gfz.ks, DOM.Gfz.ke, DOM.Gfz.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", DOM.Gfz.ksb, DOM.Gfz.keb,
    DOM.Gfz.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", DOM.Gfz._is, DOM.Gfz._ie,
    DOM.Gfz.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", DOM.Gfz._isb, DOM.Gfz._ieb,
    DOM.Gfz.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", DOM.Gfz._js, DOM.Gfz._je,
    DOM.Gfz.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", DOM.Gfz._jsb, DOM.Gfz._jeb,
    DOM.Gfz.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", DOM.Gfz._ks, DOM.Gfz._ke,
    DOM.Gfz.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", DOM.Gfz._ksb, DOM.Gfz._keb,
    DOM.Gfz.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", DOM.Gfz.s1, DOM.Gfz.s2,
    DOM.Gfz.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", DOM.Gfz.s1b, DOM.Gfz.s2b,
    DOM.Gfz.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", DOM.Gfz.s2_i, DOM.Gfz.s2_j, 
    DOM.Gfz.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", DOM.Gfz.s2b_i, DOM.Gfz.s2b_j, 
    DOM.Gfz.s2b_k);
  fprintf(outfile, "\n");

  // Simulation Parameters
  fprintf(outfile, "Physical Parameters:\n");
  fprintf(outfile, "  rho_f = %e\n", rho_f);
  fprintf(outfile, "  nu = %e\n", nu);
  fprintf(outfile, "  mu = %e\n", mu);
  fprintf(outfile, "  s_D = %e\n", s_D);
  fprintf(outfile, "  s_k = %e\n", s_k);
  fprintf(outfile, "  s_beta = %e\n", s_beta);
  fprintf(outfile, "  s_ref = %e\n", s_ref);
  fprintf(outfile, "\n");

  fprintf(outfile, "Simulation Parameters:\n");
  fprintf(outfile, "  duration = %e\n", duration);
  fprintf(outfile, "  CFL = %e\n", CFL);
  fprintf(outfile, "  SCALAR = %d\n", SCALAR);
  fprintf(outfile, "  pp_max_iter = %d\n", pp_max_iter);
  fprintf(outfile, "  pp_residual = %e\n", pp_residual);
  fprintf(outfile, "  init_cond = %d\n", init_cond);
  fprintf(outfile, "  s_init = %e, s_init_rand = %e\n", s_init, s_init_rand);
  fprintf(outfile, "    (QUIESCENT = 0, SHEAR = 1, CHANNEL = 2, TAYLOR_GREEN ="
                   " 3, TAYLOR_GREEN_3 = 4,\n");
  fprintf(outfile, "    TURB_CHANNEL = 5, TURBULENT = 6\n");
  fprintf(outfile, "   out_plane = %d (W/E/S/N/B/T/HOMOGENEOUS = 0/1/2/3/4/5/10\n",
    out_plane);
  fprintf(outfile, "\n");

  fprintf(outfile, "Boundary Conditions: (0 = PERIODIC, 1 = DIRICHLET, 2 = "
    "NEUMANN)\n");
  fprintf(outfile, "  bc.pW = %d", bc.pW);
  fprintf(outfile, ", bc.pE = %d", bc.pE);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.pS = %d", bc.pS);
  fprintf(outfile, ", bc.pN = %d", bc.pN);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.pB = %d", bc.pB);
  fprintf(outfile, ", bc.pT = %d", bc.pT);
  fprintf(outfile, "\n");

  fprintf(outfile, "  bc.uW = %d", bc.uW);
  if (bc.uW == DIRICHLET) fprintf(outfile, " %f", bc.uWDm);
  fprintf(outfile, ", bc.uE = %d", bc.uE);
  if (bc.uE == DIRICHLET) fprintf(outfile, " %f", bc.uEDm);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.uS = %d", bc.uS);
  if (bc.uS == DIRICHLET) fprintf(outfile, " %f", bc.uSDm);
  fprintf(outfile, ", bc.uN = %d", bc.uN);
  if (bc.uN == DIRICHLET) fprintf(outfile, " %f", bc.uNDm);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.uB = %d", bc.uB);
  if (bc.uB == DIRICHLET) fprintf(outfile, " %f", bc.uBDm);
  fprintf(outfile, ", bc.uT = %d", bc.uT);
  if (bc.uT == DIRICHLET) fprintf(outfile, " %f", bc.uTDm);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.vW = %d", bc.vW);
  if (bc.vW == DIRICHLET) fprintf(outfile, " %f", bc.vWDm);
  fprintf(outfile, ", bc.vE = %d", bc.vE);
  if (bc.vE == DIRICHLET) fprintf(outfile, " %f", bc.vEDm);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.vS = %d", bc.vS);
  if (bc.vS == DIRICHLET) fprintf(outfile, " %f", bc.vSDm);
  fprintf(outfile, ", bc.vN = %d", bc.vN);
  if (bc.vN == DIRICHLET) fprintf(outfile, " %f", bc.vNDm);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.vB = %d", bc.vB);
  if (bc.vB == DIRICHLET) fprintf(outfile, " %f", bc.vBDm);
  fprintf(outfile, ", bc.vT = %d", bc.vT);
  if (bc.vT == DIRICHLET) fprintf(outfile, " %f", bc.vTDm);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.wW = %d", bc.wW);
  if (bc.wW == DIRICHLET) fprintf(outfile, " %f", bc.wWDm);
  fprintf(outfile, ", bc.wE = %d", bc.wE);
  if (bc.wE == DIRICHLET) fprintf(outfile, " %f", bc.wEDm);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.wS = %d", bc.wS);
  if (bc.wS == DIRICHLET) fprintf(outfile, " %f", bc.wSDm);
  fprintf(outfile, ", bc.wN = %d", bc.wN);
  if (bc.wN == DIRICHLET) fprintf(outfile, " %f", bc.wNDm);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.wB = %d", bc.wB);
  if (bc.wB == DIRICHLET) fprintf(outfile, " %f", bc.wBDm);
  fprintf(outfile, ", bc.wT = %d", bc.wT);
  if (bc.wT == DIRICHLET) fprintf(outfile, " %f", bc.wTDm);
  fprintf(outfile, "\n");

  fprintf(outfile, "\n");
  fprintf(outfile, "Scalar Boundary Conditions:\n");
  fprintf(outfile, "  bc_s.sW = %d", bc_s.sW);
  if (bc_s.sW == DIRICHLET) fprintf(outfile, " %f", bc_s.sWD);
  if (bc_s.sW == NEUMANN) fprintf(outfile, " %f", bc_s.sWN);
  fprintf(outfile, ", bc_s.sE = %d", bc_s.sE);
  if (bc_s.sE == DIRICHLET) fprintf(outfile, " %f", bc_s.sED);
  if (bc_s.sE == NEUMANN) fprintf(outfile, " %f", bc_s.sEN);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc_s.sS = %d", bc_s.sS);
  if (bc_s.sS == DIRICHLET) fprintf(outfile, " %f", bc_s.sSD);
  if (bc_s.sS == NEUMANN) fprintf(outfile, " %f", bc_s.sSN);
  fprintf(outfile, ", bc_s.sN = %d", bc_s.sN);
  if (bc_s.sN == DIRICHLET) fprintf(outfile, " %f", bc_s.sND);
  if (bc_s.sN == NEUMANN) fprintf(outfile, " %f", bc_s.sNN);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc_s.sB = %d", bc_s.sB);
  if (bc_s.sB == DIRICHLET) fprintf(outfile, " %f", bc_s.sBD);
  if (bc_s.sB == NEUMANN) fprintf(outfile, " %f", bc_s.sBN);
  fprintf(outfile, ", bc_s.sT = %d", bc_s.sT);
  if (bc_s.sT == DIRICHLET) fprintf(outfile, " %f", bc_s.sTD);
  if (bc_s.sT == NEUMANN) fprintf(outfile, " %f", bc_s.sTN);
  fprintf(outfile, "\n");
  fprintf(outfile, "\n");

  fprintf(outfile, "Screen offsets:\n");
  fprintf(outfile, "  bc.dsW = %f\n", bc.dsW);
  fprintf(outfile, "  bc.dsE = %f\n", bc.dsE);
  fprintf(outfile, "  bc.dsS = %f\n", bc.dsS);
  fprintf(outfile, "  bc.dsN = %f\n", bc.dsN);
  fprintf(outfile, "  bc.dsB = %f\n", bc.dsB);
  fprintf(outfile, "  bc.dsT = %f\n", bc.dsT);
  fprintf(outfile, "\n");

  fprintf(outfile, "Applied Pressure Gradient:\n");
  fprintf(outfile, "  gradP.x = %f\n", gradP.x);
  fprintf(outfile, "  gradP.xm = %f\n", gradP.xm);
  fprintf(outfile, "  gradP.xa = %f\n", gradP.xa);
  fprintf(outfile, "  gradP.y = %f\n", gradP.y);
  fprintf(outfile, "  gradP.ym = %f\n", gradP.ym);
  fprintf(outfile, "  gradP.ya = %f\n", gradP.ya);
  fprintf(outfile, "  gradP.z = %f\n", gradP.z);
  fprintf(outfile, "  gradP.zm = %f\n", gradP.zm);
  fprintf(outfile, "  gradP.za = %f\n", gradP.za);
  fprintf(outfile, "\n");

  fprintf(outfile, "Applied Body Forces:\n");
  fprintf(outfile, "  g.x = %f\n", g.x);
  fprintf(outfile, "  g.xm = %f\n", g.xm);
  fprintf(outfile, "  g.xa = %f\n", g.xa);
  fprintf(outfile, "  g.y = %f\n", g.y);
  fprintf(outfile, "  g.ym = %f\n", g.ym);
  fprintf(outfile, "  g.ya = %f\n", g.ya);
  fprintf(outfile, "  g.z = %f\n", g.z);
  fprintf(outfile, "  g.zm = %f\n", g.zm);
  fprintf(outfile, "  g.za = %f\n", g.za);
  fprintf(outfile, "\n");

  int i = rank;
  fprintf(outfile, "MPI Domain Decomposition:\n");
  fprintf(outfile, "  nMPIdom = %d\n", DOM.S3);
  fprintf(outfile, "  dom[%d].rank = %d\n", i, dom[i].rank);

  fprintf(outfile, "MPI Subdomain %d:\n", i);
  fprintf(outfile, "  X: (%lf, %lf), dX = %lf\n", dom[i].xs, dom[i].xe, dom[i].dx);
  fprintf(outfile, "  Y: (%lf, %lf), dY = %lf\n", dom[i].ys, dom[i].ye, dom[i].dy);
  fprintf(outfile, "  Z: (%lf, %lf), dZ = %lf\n", dom[i].zs, dom[i].ze, dom[i].dz);
  fprintf(outfile, "  Xn = %d, Yn = %d, Zn = %d\n", dom[i].xn, dom[i].yn, dom[i].zn);

  fprintf(outfile, "Connectivity:\n");
    fprintf(outfile, "  w: %d, e: %d, s: %d, n: %d, b: %d, t: %d\n", dom[i].w, dom[i].e,
      dom[i].s, dom[i].n, dom[i].b, dom[i].t);
  fprintf(outfile, "MPI Domain Indices:\n");
  fprintf(outfile, "   (I, J, K) = (%d, %d, %d)\n", dom[i].I, dom[i].J, dom[i].K);

  fprintf(outfile, "Grids:\n");
  fprintf(outfile, "  dom[%d].Gcc:\n", i);
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", dom[i].Gcc.is, dom[i].Gcc.ie,
    dom[i].Gcc.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", dom[i].Gcc.isb, dom[i].Gcc.ieb,
    dom[i].Gcc.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", dom[i].Gcc.js, dom[i].Gcc.je,
    dom[i].Gcc.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", dom[i].Gcc.jsb, dom[i].Gcc.jeb,
    dom[i].Gcc.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", dom[i].Gcc.ks, dom[i].Gcc.ke,
    dom[i].Gcc.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", dom[i].Gcc.ksb, dom[i].Gcc.keb,
    dom[i].Gcc.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", dom[i].Gcc._is, dom[i].Gcc._ie,
    dom[i].Gcc.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", dom[i].Gcc._isb,
    dom[i].Gcc._ieb, dom[i].Gcc.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", dom[i].Gcc._js, dom[i].Gcc._je,
    dom[i].Gcc.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", dom[i].Gcc._jsb,
    dom[i].Gcc._jeb, dom[i].Gcc.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", dom[i].Gcc._ks, dom[i].Gcc._ke,
    dom[i].Gcc.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", dom[i].Gcc._ksb,
    dom[i].Gcc._keb, dom[i].Gcc.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", dom[i].Gcc.s1, dom[i].Gcc.s2,
    dom[i].Gcc.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", dom[i].Gcc.s1b, dom[i].Gcc.s2b,
    dom[i].Gcc.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", dom[i].Gcc.s2_i, dom[i].Gcc.s2_j, 
    dom[i].Gcc.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", dom[i].Gcc.s2b_i, dom[i].Gcc.s2b_j, 
    dom[i].Gcc.s2b_k);

  fprintf(outfile, "  dom[%d].Gfx:\n", i);
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", dom[i].Gfx.is, dom[i].Gfx.ie,
    dom[i].Gfx.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", dom[i].Gfx.isb, dom[i].Gfx.ieb,
    dom[i].Gfx.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", dom[i].Gfx.js, dom[i].Gfx.je,
    dom[i].Gfx.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", dom[i].Gfx.jsb, dom[i].Gfx.jeb,
    dom[i].Gfx.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", dom[i].Gfx.ks, dom[i].Gfx.ke,
    dom[i].Gfx.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", dom[i].Gfx.ksb, dom[i].Gfx.keb,
    dom[i].Gfx.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", dom[i].Gfx._is, dom[i].Gfx._ie,
    dom[i].Gfx.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", dom[i].Gfx._isb,
    dom[i].Gfx._ieb, dom[i].Gfx.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", dom[i].Gfx._js, dom[i].Gfx._je,
    dom[i].Gfx.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", dom[i].Gfx._jsb,
    dom[i].Gfx._jeb, dom[i].Gfx.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", dom[i].Gfx._ks, dom[i].Gfx._ke,
    dom[i].Gfx.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", dom[i].Gfx._ksb,
    dom[i].Gfx._keb, dom[i].Gfx.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", dom[i].Gfx.s1, dom[i].Gfx.s2,
    dom[i].Gfx.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", dom[i].Gfx.s1b, dom[i].Gfx.s2b,
    dom[i].Gfx.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", dom[i].Gfx.s2_i, dom[i].Gfx.s2_j, 
    dom[i].Gfx.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", dom[i].Gfx.s2b_i, dom[i].Gfx.s2b_j, 
    dom[i].Gfx.s2b_k);

  fprintf(outfile, "  dom[%d].Gfy:\n", i);
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", dom[i].Gfy.is, dom[i].Gfy.ie,
    dom[i].Gfy.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", dom[i].Gfy.isb, dom[i].Gfy.ieb,
    dom[i].Gfy.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", dom[i].Gfy.js, dom[i].Gfy.je,
    dom[i].Gfy.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", dom[i].Gfy.jsb, dom[i].Gfy.jeb,
    dom[i].Gfy.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", dom[i].Gfy.ks, dom[i].Gfy.ke,
    dom[i].Gfy.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", dom[i].Gfy.ksb, dom[i].Gfy.keb,
    dom[i].Gfy.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", dom[i].Gfy._is, dom[i].Gfy._ie,
    dom[i].Gfy.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", dom[i].Gfy._isb,
    dom[i].Gfy._ieb, dom[i].Gfy.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", dom[i].Gfy._js, dom[i].Gfy._je,
    dom[i].Gfy.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", dom[i].Gfy._jsb,
    dom[i].Gfy._jeb, dom[i].Gfy.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", dom[i].Gfy._ks, dom[i].Gfy._ke,
    dom[i].Gfy.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", dom[i].Gfy._ksb,
    dom[i].Gfy._keb, dom[i].Gfy.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", dom[i].Gfy.s1, dom[i].Gfy.s2,
    dom[i].Gfy.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", dom[i].Gfy.s1b, dom[i].Gfy.s2b,
    dom[i].Gfy.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", dom[i].Gfy.s2_i, dom[i].Gfy.s2_j, 
    dom[i].Gfy.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", dom[i].Gfy.s2b_i, dom[i].Gfy.s2b_j, 
    dom[i].Gfy.s2b_k);

  fprintf(outfile, "  dom[%d].Gfz:\n", i);
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", dom[i].Gfz.is, dom[i].Gfz.ie,
    dom[i].Gfz.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", dom[i].Gfz.isb, dom[i].Gfz.ieb,
    dom[i].Gfz.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", dom[i].Gfz.js, dom[i].Gfz.je,
    dom[i].Gfz.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", dom[i].Gfz.jsb, dom[i].Gfz.jeb,
    dom[i].Gfz.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", dom[i].Gfz.ks, dom[i].Gfz.ke,
    dom[i].Gfz.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", dom[i].Gfz.ksb, dom[i].Gfz.keb,
    dom[i].Gfz.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", dom[i].Gfz._is, dom[i].Gfz._ie,
    dom[i].Gfz.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", dom[i].Gfz._isb,
    dom[i].Gfz._ieb, dom[i].Gfz.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", dom[i].Gfz._js, dom[i].Gfz._je,
    dom[i].Gfz.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", dom[i].Gfz._jsb,
    dom[i].Gfz._jeb, dom[i].Gfz.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", dom[i].Gfz._ks, dom[i].Gfz._ke,
    dom[i].Gfz.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", dom[i].Gfz._ksb,
    dom[i].Gfz._keb, dom[i].Gfz.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", dom[i].Gfz.s1, dom[i].Gfz.s2,
    dom[i].Gfz.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", dom[i].Gfz.s1b, dom[i].Gfz.s2b,
    dom[i].Gfz.s3b);
  fprintf(outfile, "    s2_i = %d, s2_j = %d, s2_k = %d\n", dom[i].Gfz.s2_i, dom[i].Gfz.s2_j, 
    dom[i].Gfz.s2_k);
  fprintf(outfile, "    s2b_i = %d, s2b_j = %d, s2b_k = %d\n", dom[i].Gfz.s2b_i, dom[i].Gfz.s2b_j, 
    dom[i].Gfz.s2b_k);

  fclose(outfile);
}

void compute_vel_BC()
{
  real delta = ttime - v_bc_tdelay;
  if (delta >= 0) {
    // uWD
    if (bc.uWDa == 0) {
      bc.uWD = bc.uWDm;
    } else if (fabs(delta*bc.uWDa) > fabs(bc.uWDm)) {
      bc.uWD = bc.uWDm;
    } else {
      bc.uWD = delta*bc.uWDa;
    }
    // uED
    if (bc.uEDa == 0) {
      bc.uED = bc.uEDm;
    } else if (fabs(delta*bc.uEDa) > fabs(bc.uEDm)) {
      bc.uED = bc.uEDm;
    } else {
      bc.uED = delta*bc.uEDa;
    }
    // uSD
    if (bc.uSDa == 0) {
      bc.uSD = bc.uSDm;
    } else if (fabs(delta*bc.uSDa) > fabs(bc.uSDm)) {
      bc.uSD = bc.uSDm;
    } else {
      bc.uSD = delta*bc.uSDa;
    }
    // uND
    if (bc.uNDa == 0) {
      bc.uND = bc.uNDm;
    } else if (fabs(delta*bc.uNDa) > fabs(bc.uNDm)) {
      bc.uND = bc.uNDm;
    } else {
      bc.uND = delta*bc.uNDa;
    }
    // uBD
    if (bc.uBDa == 0) {
      bc.uBD = bc.uBDm;
    } else if (fabs(delta*bc.uBDa) > fabs(bc.uBDm)) {
      bc.uBD = bc.uBDm;
    } else {
      bc.uBD = delta*bc.uBDa;
    }
    // uTD
    if (bc.uTDa == 0) {
      bc.uTD = bc.uTDm;
    } else if (fabs(delta*bc.uTDa) > fabs(bc.uTDm)) {
      bc.uTD = bc.uTDm;
    } else {
      bc.uTD = delta*bc.uTDa;
    }
    // vWD
    if (bc.vWDa == 0) {
      bc.vWD = bc.vWDm;
    } else if (fabs(delta*bc.vWDa) > fabs(bc.vWDm)) {
      bc.vWD = bc.vWDm;
    } else {
      bc.vWD = delta*bc.vWDa;
    }
    // vED
    if (bc.vEDa == 0) {
      bc.vED = bc.vEDm;
    } else if (fabs(delta*bc.vEDa) > fabs(bc.vEDm)) {
      bc.vED = bc.vEDm;
    } else {
      bc.vED = delta*bc.vEDa;
    }
    // vSD
    if (bc.vSDa == 0) {
      bc.vSD = bc.vSDm;
    } else if (fabs(delta*bc.vSDa) > fabs(bc.vSDm)) {
      bc.vSD = bc.vSDm;
    } else {
      bc.vSD = delta*bc.vSDa;
    }
    // vND
    if (bc.vNDa == 0) {
      bc.vND = bc.vNDm;
    } else if (fabs(delta*bc.vNDa) > fabs(bc.vNDm)) {
      bc.vND = bc.vNDm;
    } else {
      bc.vND = delta*bc.vNDa;
    }
    // vBD
    if (bc.vBDa == 0) {
      bc.vBD = bc.vBDm;
    } else if (fabs(delta*bc.vBDa) > fabs(bc.vBDm)) {
      bc.vBD = bc.vBDm;
    } else {
      bc.vBD = delta*bc.vBDa;
    }
    // vTD
    if (bc.vTDa == 0) {
      bc.vTD = bc.vTDm;
    } else if (fabs(delta*bc.vTDa) > fabs(bc.vTDm)) {
      bc.vTD = bc.vTDm;
    } else {
      bc.vTD = delta*bc.vTDa;
    }
    // wWD
    if (bc.wWDa == 0) {
      bc.wWD = bc.wWDm;
    } else if (fabs(delta*bc.wWDa) > fabs(bc.wWDm)) {
      bc.wWD = bc.wWDm;
    } else {
      bc.wWD = delta*bc.wWDa;
    }
    // wED
    if (bc.wEDa == 0) {
      bc.wED = bc.wEDm;
    } else if (fabs(delta*bc.wEDa) > fabs(bc.wEDm)) {
      bc.wED = bc.wEDm;
    } else {
      bc.wED = delta*bc.wEDa;
    }
    // wSD
    if (bc.wSDa == 0) {
      bc.wSD = bc.wSDm;
    } else if (fabs(delta*bc.wSDa) > fabs(bc.wSDm)) {
      bc.wSD = bc.wSDm;
    } else {
      bc.wSD = delta*bc.wSDa;
    }
    // wND
    if (bc.wNDa == 0) {
      bc.wND = bc.wNDm;
    } else if (fabs(delta*bc.wNDa) > fabs(bc.wNDm)) {
      bc.wND = bc.wNDm;
    } else {
      bc.wND = delta*bc.wNDa;
    }
    // wBD
    if (bc.wBDa == 0) {
      bc.wBD = bc.wBDm;
    } else if (fabs(delta*bc.wBDa) > fabs(bc.wBDm)) {
      bc.wBD = bc.wBDm;
    } else {
      bc.wBD = delta*bc.wBDa;
    }
    // wTD
    if (bc.wTDa == 0) {
      bc.wTD = bc.wTDm;
    } else if (fabs(delta*bc.wTDa) > fabs(bc.wTDm)) {
      bc.wTD = bc.wTDm;
    } else {
      bc.wTD = delta*bc.wTDa;
    }
  }
  cuda_update_bc();
}

void domain_init_fields() 
{
  // Prevent turbulent forcing from being used unless it's specified
  if (init_cond != TURBULENT) {
    turbA = 0.;
    turbl = 0.;
    turb_k0 = 0.;
  }

  /* QUIESCENT */
  init_quiescent();

  /* SHEAR */
  if (init_cond == SHEAR) {
    init_shear();
  }

  /* CHANNEL */
  if (init_cond == CHANNEL) {
    init_channel();
  }

  /* TURB_CHANNEL */
  if (init_cond == TURB_CHANNEL) {
    init_turb_channel();
  }

  /* TURBULENT -- HIT */
  if (init_cond == TURBULENT) {
    init_hit();
  }

  /* TAYLOR GREEN -- 3D */
  if (init_cond == TAYLOR_GREEN_3) {
    init_tg3();
  }

  // Init some other variables
  // value of dt doesn't matter here, but let's make it something realistic
  // Do this before TG because that sets it's own values
  dt = 2.*nu / (dom[rank].dx * dom[rank].dx);
  dt += 2.*nu / (dom[rank].dy * dom[rank].dy);
  dt += 2.*nu / (dom[rank].dz * dom[rank].dz);
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, mpi_real, MPI_MIN, MPI_COMM_WORLD);
  dt = CFL / dt;
  dt0 = -1.;

  stepnum = 0;
  ttime = 0.;
  rec_vtk_stepnum_out = 0;
  rec_cgns_flow_ttime_out = 0.;
  rec_cgns_part_ttime_out = 0.;
  rec_vtk_ttime_out = 0.;
  //rec_restart_ttime_out = 0.;

  /* TAYLOR GREEN */
  if (init_cond == TAYLOR_GREEN) {
    init_tg();
  }
}

void init_quiescent(void)
{
  int i;

  for (i = 0; i < dom[rank].Gcc.s3b; i++) {
    p[i] = 0.;
    p0[i] = 0.;
    phi[i] = 0.;
  }
  for (i = 0; i < dom[rank].Gfx.s3b; i++) {
    u[i] = 0.;
    u0[i] = 0.;
    conv_u[i] = 0.;
    conv0_u[i] = 0.;
    diff_u[i] = 0.;
    diff0_u[i] = 0.;
    f_x[i] = 0.;
    u_star[i] = 0.;
  }
  for (i = 0; i < dom[rank].Gfy.s3b; i++) {
    v[i] = 0.;
    v0[i] = 0.;
    conv_v[i] = 0.;
    conv0_v[i] = 0.;
    diff_v[i] = 0.;
    diff0_v[i] = 0.;
    f_y[i] = 0.;
    v_star[i] = 0.;
  }
  for (i = 0; i < dom[rank].Gfz.s3b; i++) {
    w[i] = 0.;
    w0[i] = 0.;
    conv_w[i] = 0.;
    conv0_w[i] = 0.;
    diff_w[i] = 0.;
    diff0_w[i] = 0.;
    f_z[i] = 0.;
    w_star[i] = 0.;
  }

}

void init_shear(void)
{
  int _ii, _jj, _kk;
  int c;
  real x, y, z;
  for (_kk = dom[rank].Gfx._ksb; _kk <= dom[rank].Gfx._keb; _kk++) {
    for (_jj = dom[rank].Gfx._jsb; _jj <= dom[rank].Gfx._jeb; _jj++) {
      for (_ii = dom[rank].Gfx._isb; _ii <= dom[rank].Gfx._ieb; _ii++) {
        y = ((_jj - 0.5) * dom[rank].dy) + dom[rank].ys;
        z = ((_kk - 0.5) * dom[rank].dz) + dom[rank].zs;
        c = GFX_LOC(_ii, _jj, _kk, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        u[c] = (bc.uNDm - bc.uSDm)*(y - DOM.ys)/DOM.yl + bc.uSDm;
        u[c] += (bc.uTDm - bc.uBDm)*(z - DOM.zs)/DOM.zl + bc.uBDm;
        u0[c] = u[c];
      }
    }
  }
  for (_kk = dom[rank].Gfy._ksb; _kk <= dom[rank].Gfy._keb; _kk++) {
    for (_jj = dom[rank].Gfy._jsb; _jj <= dom[rank].Gfy._jeb; _jj++) {
      for (_ii = dom[rank].Gfy._isb; _ii <= dom[rank].Gfy._ieb; _ii++) {
        z = ((_kk - 0.5) * dom[rank].dz) + dom[rank].zs;
        x = ((_ii - 0.5) * dom[rank].dx) + dom[rank].xs;
        c = GFY_LOC(_ii, _jj, _kk, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        v[c] = (bc.vEDm - bc.vWDm)*(x - DOM.xs)/DOM.xl + bc.vWDm;
        v[c] += (bc.vTDm - bc.vBDm)*(z - DOM.zs)/DOM.zl + bc.vBDm;
        v0[c] = v[c];
      }
    }
  }
  for (_kk = dom[rank].Gfz._ksb; _kk <= dom[rank].Gfz._keb; _kk++) {
    for (_jj = dom[rank].Gfz._jsb; _jj <= dom[rank].Gfz._jeb; _jj++) {
      for (_ii = dom[rank].Gfz._isb; _ii <= dom[rank].Gfz._ieb; _ii++) {
        x = ((_ii - 0.5) * dom[rank].dx) + dom[rank].xs;
        y = ((_jj - 0.5) * dom[rank].dy) + dom[rank].ys;
        c = GFZ_LOC(_ii, _jj, _kk, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        w[c] = (bc.wEDm - bc.wWDm)*(x - DOM.xs)/DOM.xl + bc.wWDm;
        w[c] += (bc.wNDm - bc.wSDm)*(y - DOM.ys)/DOM.yl + bc.wSDm;
        w0[c] = w[c];
      }
    }
  }
}

void init_channel(void)
{
  int _ii, _jj, _kk;
  int c;
  real x, y, z;
  for (_kk = dom[rank].Gfx._ksb; _kk <= dom[rank].Gfx._keb; _kk++) {
    for (_jj = dom[rank].Gfx._jsb; _jj <= dom[rank].Gfx._jeb; _jj++) {
      for (_ii = dom[rank].Gfx._isb; _ii <= dom[rank].Gfx._ieb; _ii++) {
        y = ((_jj - 0.5) * dom[rank].dy) + dom[rank].ys;
        z = ((_kk - 0.5) * dom[rank].dz) + dom[rank].zs;
        c = GFX_LOC(_ii, _jj, _kk, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        u[c]  = 0.5/mu*gradP.xm*(y*y - (DOM.ys + DOM.ye)*y + DOM.ys*DOM.ye)
          * (bc.uS == DIRICHLET)
          + 0.5/mu*gradP.xm*(z*z - (DOM.zs + DOM.ze)*z + DOM.zs*DOM.ze)
          * (bc.uB == DIRICHLET);
        u0[c] = u[c];
      }
    }
  }
  for (_kk = dom[rank].Gfy._ksb; _kk <= dom[rank].Gfy._keb; _kk++) {
    for (_jj = dom[rank].Gfy._jsb; _jj <= dom[rank].Gfy._jeb; _jj++) {
      for (_ii = dom[rank].Gfy._isb; _ii <= dom[rank].Gfy._ieb; _ii++) {
        z = ((_kk - 0.5) * dom[rank].dz) + dom[rank].zs;
        x = ((_ii - 0.5) * dom[rank].dx) + dom[rank].xs;
        c = GFY_LOC(_ii, _jj, _kk, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        v[c]  = 0.5/mu*gradP.ym*(x*x - (DOM.xs + DOM.xe)*x + DOM.xs*DOM.xe)
          * (bc.vW == DIRICHLET)
          + 0.5/mu*gradP.ym*(z*z - (DOM.zs + DOM.ze)*z + DOM.zs*DOM.ze)
          * (bc.vB == DIRICHLET);

        v0[c] = v[c];
      }
    }
  }
  for (_kk = dom[rank].Gfz._ksb; _kk <= dom[rank].Gfz._keb; _kk++) {
    for (_jj = dom[rank].Gfz._jsb; _jj <= dom[rank].Gfz._jeb; _jj++) {
      for (_ii = dom[rank].Gfz._isb; _ii <= dom[rank].Gfz._ieb; _ii++) {
        x = ((_ii - 0.5) * dom[rank].dx) + dom[rank].xs;
        y = ((_jj - 0.5) * dom[rank].dy) + dom[rank].ys;
        c = GFZ_LOC(_ii, _jj, _kk, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        w[c]  = 0.5/mu*gradP.zm*(x*x - (DOM.xs + DOM.xe)*x + DOM.xs*DOM.xe)
          * (bc.wW == DIRICHLET)
          + 0.5/mu*gradP.zm*(y*y - (DOM.ys + DOM.ye)*y + DOM.ys*DOM.ye)
          * (bc.wS == DIRICHLET);
        w0[c] = w[c];
      }
    }
  }
}

void init_turb_channel(void)
{
  int i, j, k;
  int c;

  // Randomly initialize velocity components for turbulent channel flow
  // Assumes x is streamwise, y wall-normal, z spanwise
  real delta = 0.5*DOM.yl;        // Channel half height
  real tau_w = -delta*gradP.xm;   // Wall shear stress
  real u_tau = tau_w / rho_f;     // Friction velocity
  real d_nu = nu / u_tau;         // Viscous length scale
  real u_max = 0.5 * tau_w * delta /(rho_f * nu);   // centerline velocity
  real u_frac = 0.8;              // Fraction of velocity for added noise

  real kappa = 0.41;              // Turbulent mean profile parameter
  real BB = 5.2;                  // Turbulent mean profile parameter
  real ikappa = 1./kappa;
  real id_nu = 1./d_nu; 
  real y_plus;

  printf("u_max = %lf, Re ~ %lf\n", u_max, u_max * delta / nu);

  // Seed rng
  rng_init(time(NULL));

  // Initialize with random field
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        real y = ((j - 0.5) * dom[rank].dy) + dom[rank].ys;

        // Laminar channel flow
        //u[c] = 0.5/mu*gradP.xm*(y*y - (DOM.ys + DOM.ye)*y + DOM.ys*DOM.ye);

        // Turbulent mean profile
        if (y <= 0.5*(DOM.ys + DOM.ye)) {
          y_plus = y * id_nu;
        } else {
          y_plus = (DOM.ye - y) * id_nu;
        }
        u[c] = u_tau * (ikappa * log(y_plus) + BB);
        
        // Noise, if we want
        //u[c] = ((real) rand() / (real) RAND_MAX - 0.5)*u_max*u_frac;
        //u[c] = (rng_dbl() - 0.5)*u_max*u_frac;

        // If east face is adjacent to another subdomain, set u[_ie, j, k] = 0
        //  to enforce periodicity
        if (dom[rank].e != MPI_PROC_NULL && i == dom[rank].Gfx._ie) {
          //u[c] = 0.;
        } else {
          //u[c] += ((real) rand() / (real) RAND_MAX - 0.5)*u_max*u_frac;
        }
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        //v[c] = ((real) rand() / (real) RAND_MAX - 0.5)*u_max*u_frac;
        v[c] = (rng_dbl() - 0.5)*u_max*u_frac;

        // If south face is adjacent to another subdomain, set v[i, _js, k] =0
        //  to enforce periodicity
        if (dom[rank].s != MPI_PROC_NULL && j == dom[rank].Gfy._js) {
          v[c] = 0.;
        }
        // If north face is adjacent to another subdomain, set v[i, _je, k] =0
        //  to enforce periodicity
        if (dom[rank].n != MPI_PROC_NULL && j == dom[rank].Gfy._je) {
          v[c] = 0.;
        }

        // If south face is a wall, set v[i,_js,k] = 0 to enforce wall-i-ness
        if (dom[rank].J == 0 && j == dom[rank].Gfy._js) {
          v[c] = 0.;
        }
        // If north face is a wall, set v[i,_je,k] = 0 as well
        if (dom[rank].J == DOM.Jn && j == dom[rank].Gfy._je) {
          v[c] = 0.;
        }

      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        //w[c] = ((real) rand() / (real) RAND_MAX - 0.5)*u_max*u_frac;
        w[c] = (rng_dbl() - 0.5)*u_max*u_frac;

        // If bot face is adjacent to another subdomain, set w[i, j, _ks] = 0
        //  to enforce periodicity
        if (dom[rank].b != MPI_PROC_NULL && k == dom[rank].Gfz._ks) {
          w[c] = 0.;
        }

        // If top face is adjacent to another subdomain, set w[i, j, _ke] = 0
        //  to enforce periodicity
        if (dom[rank].t != MPI_PROC_NULL && k == dom[rank].Gfz._ke) {
          w[c] = 0.;
        }
      }
    }
  }

  // Communicate boundaries so we can find correct divergence
  cuda_dom_push();
  cuda_dom_BC();
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);
  cuda_dom_pull();

  // Calculate divergence of u
  int W,E,S,N,B,T;
  real ivol = 1./(DOM.xn*DOM.yn*DOM.zn);
  real idx = 1./dom[rank].dx;
  real idy = 1./dom[rank].dy;
  real idz = 1./dom[rank].dz;

  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        W = GFX_LOC(i  , j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        E = GFX_LOC(i+1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        S = GFY_LOC(i, j  , k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        N = GFY_LOC(i, j+1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        B = GFZ_LOC(i, j, k  , dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        T = GFZ_LOC(i, j, k+1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        p[c] = (u[E] - u[W])*idx + (v[N] - v[S])*idy + (w[T] - w[B])*idz;
        p[c] *= ivol;
      }
    }
  }

  // Communicate p to get correct divergence in shared ghost cells 
  cuda_dom_push();
  mpi_cuda_exchange_Gcc(_p);
  cuda_dom_pull();

  // subtract off divergence of U to make U solenoidal
  real umean = 0.;
  real vmean = 0.;
  real wmean = 0.;
  for (k = dom[rank].Gfx._ks; k <= dom[rank].Gfx._ke; k++) {
    for (j = dom[rank].Gfx._js; j <= dom[rank].Gfx._je; j++) {
      for (i = dom[rank].Gfx._is; i <= dom[rank].Gfx._ie; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        W = GCC_LOC(i-1, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        E = GCC_LOC(i  , j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        u[c] = u[c] - 0.5*(p[W] + p[E]);
      }
    }
  }
  for (k = dom[rank].Gfy._ks; k <= dom[rank].Gfy._ke; k++) {
    for (j = dom[rank].Gfy._js; j <= dom[rank].Gfy._je; j++) {
      for (i = dom[rank].Gfy._is; i <= dom[rank].Gfy._ie; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        S = GCC_LOC(i, j-1, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        N = GCC_LOC(i, j  , k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        v[c] = v[c] - 0.5*(p[S] + p[N]);
      }
    }
  }
  for (k = dom[rank].Gfz._ks; k <= dom[rank].Gfz._ke; k++) {
    for (j = dom[rank].Gfz._js; j <= dom[rank].Gfz._je; j++) {
      for (i = dom[rank].Gfz._is; i <= dom[rank].Gfz._ie; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        B = GCC_LOC(i, j, k-1, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        T = GCC_LOC(i, j, k  , dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        w[c] = w[c] - 0.5*(p[B] + p[T]);
      }
    }
  }

  // find mean ( -1 to not double count)
  for (k = dom[rank].Gfx._ks; k <= dom[rank].Gfx._ke; k++) {
    for (j = dom[rank].Gfx._js; j <= dom[rank].Gfx._je; j++) {
      for (i = dom[rank].Gfx._is; i <= dom[rank].Gfx._ie-1; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        umean += u[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ks; k <= dom[rank].Gfy._ke; k++) {
    for (j = dom[rank].Gfy._js; j <= dom[rank].Gfy._je-1; j++) {
      for (i = dom[rank].Gfy._is; i <= dom[rank].Gfy._ie; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        vmean += v[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ks; k <= dom[rank].Gfz._ke-1; k++) {
    for (j = dom[rank].Gfz._js; j <= dom[rank].Gfz._je; j++) {
      for (i = dom[rank].Gfz._is; i <= dom[rank].Gfz._ie; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        wmean += w[c];
      }
    }
  }

  umean *= ivol;
  vmean *= ivol;
  wmean *= ivol;

  // Reduce to find global mean
  MPI_Allreduce(MPI_IN_PLACE, &umean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vmean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &wmean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  printf("u,v,w mean = %lf, %lf, %lf\n", umean, vmean, wmean);

  // re-scale to give zero mean velocity in each direction
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
//          u[c] = u[c] - umean;
        u0[c] = u[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        v[c] = v[c] - vmean;
        v0[c] = v[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        w[c] = w[c] - wmean;
        w0[c] = w[c];
      }
    }
  }

//    /* Superpose channel flow */
//    for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
//      for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
//        for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
//          y = ((j - 0.5) * dom[rank].dy) + dom[rank].ys;
//          c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
//
//          u[c] += 0.5/mu*gradP.xm*(y*y - (DOM.ys + DOM.ye)*y + DOM.ys*DOM.ye);
//          u0[c] = u[c];
//        }
//      }
//    }

  // Communicate boundaries one last time to get correct scale
  cuda_dom_push();
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);
  cuda_dom_pull();

}

void init_hit(void)
{
  // Look into initializing with gaussian distribution following 
  // Eswaran and Pope

  if (turbA <= 0.) {
    if (rank == 0) printf("N%d >> turbA forcing must be greater than 0.\n", rank);
    exit(EXIT_FAILURE);
  }
  // integral scale -- from Meneveau and Rosales 2005
  turbl = 0.19 * (DOM.xl + DOM.yl + DOM.zl) / 3.;
  turb_k0 = 13.5 * turbA * turbA * turbl * turbl;

  real urms = 3. * turbA * turbl;
  real Re_lambda = sqrt(45. * turbA * turbl*turbl / nu);

  if (rank == 0) {
    printf("N%d >> Turbulent Reynolds number = %.1lf\n", rank, Re_lambda);
    printf("N%d >> Eddy turn-over time = %.2lf\n", rank, 0.5 / turbA);
    printf("N%d >> Target kinetic energy = %.2lf\n", rank, turb_k0);
    printf("N%d >> Target dissipation = %.2lf\n", rank, 
      27. * turbl * turbl * turbA * turbA * turbA);
    printf("N%d >> Target integral length = %.2lf\n", rank, turbl);
  }

  // Randomly initialize velocity components
  int i, j, k, c;
  int seed = 0;
  struct timeval ts;

  for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {

    // Seed random number generator with something adjacent subdomains know
    // so boundaries match
    if (i == dom[rank].Gfx._is) {
      seed = abs(rank + dom[rank].w) + 1;
      rng_init(seed);
    } else if (i == dom[rank].Gfx._ie) {
      seed = abs(rank + dom[rank].e) + 1;
      rng_init(seed);
    } else if (i == dom[rank].Gfx._is + 1) {
      gettimeofday(&ts, 0);
      rng_init(ts.tv_usec);
    }

    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        real tmp = (rng_dbl() - 0.5) * urms;
        u[c] = tmp;
        u0[c] = tmp;
      }
    }
  }
  for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {

    // Seed random number generator with something adjacent subdomains know
    // so boundaries match
    if (j == dom[rank].Gfy._js) {
      seed = abs(rank + dom[rank].s) + 1;
      rng_init(seed);
    } else if (j == dom[rank].Gfy._je) {
      seed = abs(rank + dom[rank].n) + 1;
      rng_init(seed);
    } else if (j == dom[rank].Gfy._js + 1) {
      gettimeofday(&ts, 0);
      rng_init(ts.tv_usec);
    }

    for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        real tmp = (rng_dbl() - 0.5) * urms;
        v[c] = tmp;
        v0[c] = tmp;
      }
    }
  }

  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {

    // Seed random number generator with something adjacent subdomains know
    // so boundaries match
    if (k == dom[rank].Gfz._ks) {
      seed = abs(rank + dom[rank].b) + 1;
      rng_init(seed);
    } else if (k == dom[rank].Gfz._ke) {
      seed = abs(rank + dom[rank].t) + 1;
      rng_init(seed);
    } else if (k == dom[rank].Gfz._ks + 1) {
      gettimeofday(&ts, 0);
      rng_init(ts.tv_usec);
    }

    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        real tmp = (rng_dbl() - 0.5) * urms;
        w[c] = tmp;
        w0[c] = tmp;
      }
    }
  }

  // Communicate so we can find correct divergence
  cuda_dom_push();
  cuda_dom_BC();
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);
  cuda_dom_pull();

  // Calculate divergence of u
  int W,E,S,N,B,T;
  real ivol = 1./(DOM.xn*DOM.yn*DOM.zn);
  real idx = 1./dom[rank].dx;
  real idy = 1./dom[rank].dy;
  real idz = 1./dom[rank].dz;

  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        c = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        W = GFX_LOC(i  , j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        E = GFX_LOC(i+1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        S = GFY_LOC(i, j  , k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        N = GFY_LOC(i, j+1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        B = GFZ_LOC(i, j, k  , dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        T = GFZ_LOC(i, j, k+1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        p[c] = (u[E] - u[W])*idx + (v[N] - v[S])*idy + (w[T] - w[B])*idz;
        p[c] *= ivol;
      }
    }
  }

  // Communicate p to get correct divergence in shared ghost cells 
  cuda_dom_push();
  mpi_cuda_exchange_Gcc(_p);
  cuda_dom_pull();

  // subtract off divergence of U to make U solenoidal
  real umean = 0.;
  real vmean = 0.;
  real wmean = 0.;
  for (k = dom[rank].Gfx._ks; k <= dom[rank].Gfx._ke; k++) {
    for (j = dom[rank].Gfx._js; j <= dom[rank].Gfx._je; j++) {
      for (i = dom[rank].Gfx._is; i <= dom[rank].Gfx._ie; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        W = GCC_LOC(i-1, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        E = GCC_LOC(i  , j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        u[c] = u[c] - 0.5*(p[W] + p[E]);
      }
    }
  }
  for (k = dom[rank].Gfy._ks; k <= dom[rank].Gfy._ke; k++) {
    for (j = dom[rank].Gfy._js; j <= dom[rank].Gfy._je; j++) {
      for (i = dom[rank].Gfy._is; i <= dom[rank].Gfy._ie; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        S = GCC_LOC(i, j-1, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        N = GCC_LOC(i, j  , k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        v[c] = v[c] - 0.5*(p[S] + p[N]);
      }
    }
  }
  for (k = dom[rank].Gfz._ks; k <= dom[rank].Gfz._ke; k++) {
    for (j = dom[rank].Gfz._js; j <= dom[rank].Gfz._je; j++) {
      for (i = dom[rank].Gfz._is; i <= dom[rank].Gfz._ie; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        B = GCC_LOC(i, j, k-1, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        T = GCC_LOC(i, j, k  , dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        w[c] = w[c] - 0.5*(p[B] + p[T]);
      }
    }
  }

  // find mean ( -1 to not double count)
  for (k = dom[rank].Gfx._ks; k <= dom[rank].Gfx._ke; k++) {
    for (j = dom[rank].Gfx._js; j <= dom[rank].Gfx._je; j++) {
      for (i = dom[rank].Gfx._is; i <= dom[rank].Gfx._ie-1; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        umean += u[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ks; k <= dom[rank].Gfy._ke; k++) {
    for (j = dom[rank].Gfy._js; j <= dom[rank].Gfy._je-1; j++) {
      for (i = dom[rank].Gfy._is; i <= dom[rank].Gfy._ie; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        vmean += v[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ks; k <= dom[rank].Gfz._ke-1; k++) {
    for (j = dom[rank].Gfz._js; j <= dom[rank].Gfz._je; j++) {
      for (i = dom[rank].Gfz._is; i <= dom[rank].Gfz._ie; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        wmean += w[c];
      }
    }
  }

  umean *= ivol;
  vmean *= ivol;
  wmean *= ivol;

  // Reduce to find global mean
  MPI_Allreduce(MPI_IN_PLACE, &umean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vmean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &wmean, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);

  // re-scale to give zero mean velocity in each direction
  for (k = dom[rank].Gfx._ksb; k <= dom[rank].Gfx._keb; k++) {
    for (j = dom[rank].Gfx._jsb; j <= dom[rank].Gfx._jeb; j++) {
      for (i = dom[rank].Gfx._isb; i <= dom[rank].Gfx._ieb; i++) {
        c = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        u[c] = u[c] - umean;
        u0[c] = u[c];
      }
    }
  }
  for (k = dom[rank].Gfy._ksb; k <= dom[rank].Gfy._keb; k++) {
    for (j = dom[rank].Gfy._jsb; j <= dom[rank].Gfy._jeb; j++) {
      for (i = dom[rank].Gfy._isb; i <= dom[rank].Gfy._ieb; i++) {
        c = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        v[c] = v[c] - vmean;
        v0[c] = v[c];
      }
    }
  }
  for (k = dom[rank].Gfz._ksb; k <= dom[rank].Gfz._keb; k++) {
    for (j = dom[rank].Gfz._jsb; j <= dom[rank].Gfz._jeb; j++) {
      for (i = dom[rank].Gfz._isb; i <= dom[rank].Gfz._ieb; i++) {
        c = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        w[c] = w[c] - wmean;
        w0[c] = w[c];
      }
    }
  }
  // Communicate boundaries one last time to get correct scale
  cuda_dom_push();
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);
  cuda_dom_pull();

 // Set up recorder file if not running under restart
 if (use_restart == 0) {
   char rname[FILE_NAME_SIZE] = "turb.rec";
   recorder_turb_init(rname);
 }
}

void init_tg3(void)
{
  // http://dept.ku.edu/~cfdku/hiocfd/case_c3.5.pdf
  // u =  v0*sin(x/L)*cos(y/L)*cos(z/L)
  // v = -v0*cos(x/L)*sin(y/L)*cos(z/L)
  // w = 0
  // p = rho*v0^2/16*(cos(2x/L) + cos(2y/L))*(cos(2z/L) + 2)
  // -pi*L <= x,y,z <= pi*L
  // Re = 1600 = rho*v0*L/mu

  // Only works for -pi, pi

  int _ii, _jj, _kk;
  int c;
  real x, y, z;

  real Re = 1600.;
  real L = 1.;
  real vel0 = Re * nu/L;

  if (rank == 0) {
    printf("N%d >> Starting 3-D Taylor Green initialization ", rank);
    printf("with Re = %lf hard-coded\n", Re);
  }

  // u
  for (_kk = dom[rank].Gfx._ksb; _kk <= dom[rank].Gfx._keb; _kk++) {
    for (_jj = dom[rank].Gfx._jsb; _jj <= dom[rank].Gfx._jeb; _jj++) {
      for (_ii = dom[rank].Gfx._isb; _ii <= dom[rank].Gfx._ieb; _ii++) {
        x = ((_ii - 1.0) * dom[rank].dx) + dom[rank].xs;
        y = ((_jj - 0.5) * dom[rank].dy) + dom[rank].ys;
        z = ((_kk - 0.5) * dom[rank].dz) + dom[rank].zs;
        c = GFX_LOC(_ii, _jj, _kk, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        u[c] = vel0*sin(x/L)*cos(y/L)*cos(z/L);
        u0[c] = u[c];
      }
    }
  }
  // v
  for (_kk = dom[rank].Gfy._ksb; _kk <= dom[rank].Gfy._keb; _kk++) {
    for (_jj = dom[rank].Gfy._jsb; _jj <= dom[rank].Gfy._jeb; _jj++) {
      for (_ii = dom[rank].Gfy._isb; _ii <= dom[rank].Gfy._ieb; _ii++) {
        x = ((_ii - 0.5) * dom[rank].dx) + dom[rank].xs;
        y = ((_jj - 1.0) * dom[rank].dy) + dom[rank].ys;
        z = ((_kk - 0.5) * dom[rank].dz) + dom[rank].zs;
        c = GFY_LOC(_ii, _jj, _kk, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
          

        v[c] = -vel0*cos(x/L)*sin(y/L)*cos(z/L);
        v0[c] = v[c];
      }
    }
  }
  // w
  for (_kk = dom[rank].Gfz._ksb; _kk <= dom[rank].Gfz._keb; _kk++) {
    for (_jj = dom[rank].Gfz._jsb; _jj <= dom[rank].Gfz._jeb; _jj++) {
      for (_ii = dom[rank].Gfz._isb; _ii <= dom[rank].Gfz._ieb; _ii++) {
        c = GFZ_LOC(_ii, _jj, _kk, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        w[c]  = 0.;
        w0[c] = 0.;
      }
    }
  }
  // p
  for (_kk = dom[rank].Gcc._ksb; _kk <= dom[rank].Gcc._keb; _kk++) {
    for (_jj = dom[rank].Gcc._jsb; _jj <= dom[rank].Gcc._jeb; _jj++) {
      for (_ii = dom[rank].Gcc._isb; _ii <= dom[rank].Gcc._ieb; _ii++) {
        x = ((_ii - 0.5) * dom[rank].dx) + dom[rank].xs;
        y = ((_jj - 0.5) * dom[rank].dy) + dom[rank].ys;
        z = ((_kk - 0.5) * dom[rank].dz) + dom[rank].zs;
        c = GCC_LOC(_ii, _jj, _kk, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        p0[c] = rho_f*vel0*vel0/16. * (cos(2.*x/L) + cos(2.*y/L))
                                    * (cos(2.*z/L) + 2.);
        p[c] = p0[c];
      }
    }
  }
}

void init_tg(void)
{
  int _ii, _jj, _kk;
  int c;

  // Taylor green constants
  real tau = DOM.xl * DOM.yl / (4. * PI * PI * nu);
  real idx = 1./dom[rank].dx;
  real idy = 1./dom[rank].dy;
  real idz = 1./dom[rank].dz;

  // Start taylor-green at t=dt; dt from first timestep
  dt0 = CFL / (idx + idy + 2.*nu*(idx*idx + idy*idy + idz*idz));
  ttime = dt0;

  if (rank == 0) {
    printf("=== Starting TAYLOR-GREEN initialization at t = %e (tau = %e)\n",
      ttime, tau);
  }

  // additional arguments to trig functions
  real xc = 2.*PI / DOM.xl;
  real yc = 2.*PI / DOM.yl;

  for (_kk = dom[rank].Gfx._ksb; _kk <= dom[rank].Gfx._keb; _kk++) {
    for (_jj = dom[rank].Gfx._jsb; _jj <= dom[rank].Gfx._jeb; _jj++) {
      for (_ii = dom[rank].Gfx._isb; _ii <= dom[rank].Gfx._ieb; _ii++) {
        c = GFX_LOC(_ii, _jj, _kk, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        real x = ((_ii - 1.0) * dom[rank].dx) + dom[rank].xs;
        real y = ((_jj - 0.5) * dom[rank].dy) + dom[rank].ys;

        conv0_u[c] = cos(xc*x)*sin(xc*x)*
                   ((-2.*xc + yc) * sin(yc*y)*sin(yc*y) 
                    - yc          * cos(yc*y)*cos(yc*y));
        diff0_u[c] = (xc*xc + yc*yc)*cos(xc*x)*sin(yc*y);

        u0[c] = -cos(xc*x) * sin(yc*y);
        u[c] = u0[c] * exp(-2.*dt0/tau);
      }
    }
  }
  for (_kk = dom[rank].Gfy._ksb; _kk <= dom[rank].Gfy._keb; _kk++) {
    for (_jj = dom[rank].Gfy._jsb; _jj <= dom[rank].Gfy._jeb; _jj++) {
      for (_ii = dom[rank].Gfy._isb; _ii <= dom[rank].Gfy._ieb; _ii++) {
        real x = ((_ii - 0.5) * dom[rank].dx) + dom[rank].xs;
        real y = ((_jj - 1.0) * dom[rank].dy) + dom[rank].ys;
        c = GFY_LOC(_ii, _jj, _kk, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        conv0_v[c] = cos(yc*y)*sin(yc*y)*
                   ((-2.*yc + xc) * sin(xc*x)*sin(xc*x) 
                    - xc          * cos(xc*x)*cos(xc*x));
        diff0_v[c] = -(xc*xc + yc*yc)*sin(xc*x)*cos(yc*y);

        v0[c] = sin(xc*x) * cos(yc*y);
        v[c] = v0[c] * exp(-2.*dt0/tau);
      }
    }
  }
  for (_kk = dom[rank].Gfz._ksb; _kk <= dom[rank].Gfz._keb; _kk++) {
    for (_jj = dom[rank].Gfz._jsb; _jj <= dom[rank].Gfz._jeb; _jj++) {
      for (_ii = dom[rank].Gfz._isb; _ii <= dom[rank].Gfz._ieb; _ii++) {
        c = GFZ_LOC(_ii, _jj, _kk, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        conv0_w[c] = 0.;
        diff0_w[c] = 0.;
        w0[c] = 0.;
        w[c] = 0.;
      }
    }
  }
  for (_kk = dom[rank].Gcc._ksb; _kk <= dom[rank].Gcc._keb; _kk++) {
    for (_jj = dom[rank].Gcc._jsb; _jj <= dom[rank].Gcc._jeb; _jj++) {
      for (_ii = dom[rank].Gcc._isb; _ii <= dom[rank].Gcc._ieb; _ii++) {
        real x = ((_ii - 0.5) * dom[rank].dx) + dom[rank].xs;
        real y = ((_jj - 0.5) * dom[rank].dy) + dom[rank].ys;
        c = GCC_LOC(_ii, _jj, _kk, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        p0[c] = -0.25 * (cos(2.*xc*x) + cos(2.*yc*y));
        p[c] = p0[c] * exp(-4.*dt0/tau);
      }
    }
  }

  // Exchange initial fields for periodic boundary conditions
  cuda_dom_push();
  mpi_cuda_exchange_Gcc(_p);
  mpi_cuda_exchange_Gfx(_u);
  mpi_cuda_exchange_Gfy(_v);
  mpi_cuda_exchange_Gfz(_w);
  mpi_cuda_exchange_Gcc(_p0);
  mpi_cuda_exchange_Gfx(_u0);
  mpi_cuda_exchange_Gfy(_v0);
  mpi_cuda_exchange_Gfz(_w0);
  mpi_cuda_exchange_Gfx(_conv0_u);
  mpi_cuda_exchange_Gfy(_conv0_v);
  mpi_cuda_exchange_Gfz(_conv0_w);
  mpi_cuda_exchange_Gfx(_diff0_u);
  mpi_cuda_exchange_Gfy(_diff0_v);
  mpi_cuda_exchange_Gfz(_diff0_w);

  // write init
  cuda_dom_pull();
  vtk_recorder_write();
  #ifdef CGNS_OUTPUT
    cgns_recorder_flow_write();
  #endif

  // increment
  stepnum = 1;
  rec_vtk_stepnum_out = 1;
  rec_cgns_flow_ttime_out = dt0;
  rec_cgns_part_ttime_out = dt0;
  rec_vtk_ttime_out = dt0;
}


void out_restart(void)
{
  // create the file for each rank
  int sigfigs;
  if (DOM.S3 == 1) {
    sigfigs = 1;
  } else {
    sigfigs = floor(log10(DOM.S3 - 1)) + 1;  
  }
  char fname[FILE_NAME_SIZE] = "";

  sprintf(fname, "%s/%s/restart.config-%0*d", ROOT_DIR, INPUT_DIR, sigfigs,
    rank);

  FILE *rest = fopen(fname, "w");
  if (rest == NULL) {
    fprintf(stderr, "File %s could not be opened.\n", fname);
    exit(EXIT_FAILURE);
  }

  fwrite(&ttime, sizeof(real), 1, rest);
  fwrite(&dt0, sizeof(real), 1, rest);
  fwrite(&dt, sizeof(real), 1, rest);
  fwrite(&stepnum, sizeof(int), 1, rest);
  fwrite(&rec_vtk_stepnum_out, sizeof(int), 1, rest);
  fwrite(&rec_cgns_flow_ttime_out, sizeof(real), 1, rest);
  fwrite(&rec_cgns_part_ttime_out, sizeof(real), 1, rest);
  fwrite(&rec_vtk_ttime_out, sizeof(real), 1, rest);
  //fwrite(&rec_restart_ttime_out, sizeof(real), 1, rest);

  fwrite(u, sizeof(real), dom[rank].Gfx.s3b, rest);
  fwrite(u0, sizeof(real), dom[rank].Gfx.s3b, rest);
  fwrite(diff0_u, sizeof(real), dom[rank].Gfx.s3b, rest);
  fwrite(conv0_u, sizeof(real), dom[rank].Gfx.s3b, rest);
  fwrite(diff_u, sizeof(real), dom[rank].Gfx.s3b, rest);
  fwrite(conv_u, sizeof(real), dom[rank].Gfx.s3b, rest);
  fwrite(u_star, sizeof(real), dom[rank].Gfx.s3b, rest);

  fwrite(v, sizeof(real), dom[rank].Gfy.s3b, rest);
  fwrite(v0, sizeof(real), dom[rank].Gfy.s3b, rest);
  fwrite(diff0_v, sizeof(real), dom[rank].Gfy.s3b, rest);
  fwrite(conv0_v, sizeof(real), dom[rank].Gfy.s3b, rest);
  fwrite(diff_v, sizeof(real), dom[rank].Gfy.s3b, rest);
  fwrite(conv_v, sizeof(real), dom[rank].Gfy.s3b, rest);
  fwrite(v_star, sizeof(real), dom[rank].Gfy.s3b, rest);

  fwrite(w, sizeof(real), dom[rank].Gfz.s3b, rest);
  fwrite(w0, sizeof(real), dom[rank].Gfz.s3b, rest);
  fwrite(diff0_w, sizeof(real), dom[rank].Gfz.s3b, rest);
  fwrite(conv0_w, sizeof(real), dom[rank].Gfz.s3b, rest);
  fwrite(diff_w, sizeof(real), dom[rank].Gfz.s3b, rest);
  fwrite(conv_w, sizeof(real), dom[rank].Gfz.s3b, rest);
  fwrite(w_star, sizeof(real), dom[rank].Gfz.s3b, rest);

  fwrite(p, sizeof(real), dom[rank].Gcc.s3b, rest);
  fwrite(phi, sizeof(real), dom[rank].Gcc.s3b, rest);
  fwrite(p0, sizeof(real), dom[rank].Gcc.s3b, rest);
  fwrite(phase, sizeof(int), dom[rank].Gcc.s3b, rest);
  fwrite(phase_shell, sizeof(int), dom[rank].Gcc.s3b, rest);

  fwrite(flag_u, sizeof(int), dom[rank].Gfx.s3b, rest);
  fwrite(flag_v, sizeof(int), dom[rank].Gfy.s3b, rest);
  fwrite(flag_w, sizeof(int), dom[rank].Gfz.s3b, rest);

  fwrite(&nparts_subdom, sizeof(int), 1, rest);
  fwrite(parts, sizeof(part_struct), nparts_subdom, rest);

  fwrite(&ncoeffs_max, sizeof(int), 1, rest);

  fwrite(&pid_int, sizeof(real), 1, rest);
  fwrite(&pid_back, sizeof(real), 1, rest);
  fwrite(&gradP.z, sizeof(real), 1, rest);

  if (SCALAR >= 1) {
    fwrite(s, sizeof(real), dom[rank].Gcc.s3b, rest);
    fwrite(s0, sizeof(real), dom[rank].Gcc.s3b, rest);
    fwrite(s_conv, sizeof(real), dom[rank].Gcc.s3b, rest);
    fwrite(s_conv0, sizeof(real), dom[rank].Gcc.s3b, rest);
    fwrite(s_diff, sizeof(real), dom[rank].Gcc.s3b, rest);
    fwrite(s_diff0, sizeof(real), dom[rank].Gcc.s3b, rest);
    fwrite(&s_ncoeffs_max, sizeof(int), 1, rest);
  }
  //fwrite(&rec_particle_ttime_out, sizeof(real), 1, rest);
  //fwrite(&rec_prec_ttime_out, sizeof(real), 1, rest);

  // close the file
  fclose(rest);
}

void in_restart(void)
{
  // open the file for each rank
  int sigfigs;
  if (DOM.S3 == 1) {
    sigfigs = 1;
  } else {
    sigfigs = floor(log10(DOM.S3 - 1)) + 1;  
  }
  char fname[FILE_NAME_SIZE] = "";
  sprintf(fname, "%s/%s/restart.config-%0*d", ROOT_DIR, INPUT_DIR, sigfigs,
    rank);

  printf("N%d >> Reading %s\n", rank, fname);

  FILE *infile = fopen(fname, "r");
  if (infile == NULL) {
    fprintf(stderr, "File %s could not be opened.\n", fname);
    exit(EXIT_FAILURE);
  }

  fread(&ttime, sizeof(real), 1, infile);
  fread(&dt0, sizeof(real), 1, infile);
  fread(&dt, sizeof(real), 1, infile);
  fread(&stepnum, sizeof(int), 1, infile);
  fread(&rec_vtk_stepnum_out, sizeof(int), 1, infile);
  fread(&rec_cgns_flow_ttime_out, sizeof(real), 1, infile);
  fread(&rec_cgns_part_ttime_out, sizeof(real), 1, infile);
  fread(&rec_vtk_ttime_out, sizeof(real), 1, infile);
  //fread(&rec_restart_ttime_out, sizeof(real), 1, infile);


  fread(u, sizeof(real), dom[rank].Gfx.s3b, infile);
  fread(u0, sizeof(real), dom[rank].Gfx.s3b, infile);
  fread(diff0_u, sizeof(real), dom[rank].Gfx.s3b, infile);
  fread(conv0_u, sizeof(real), dom[rank].Gfx.s3b, infile);
  fread(diff_u, sizeof(real), dom[rank].Gfx.s3b, infile);
  fread(conv_u, sizeof(real), dom[rank].Gfx.s3b, infile);
  fread(u_star, sizeof(real), dom[rank].Gfx.s3b, infile);

  fread(v, sizeof(real), dom[rank].Gfy.s3b, infile);
  fread(v0, sizeof(real), dom[rank].Gfy.s3b, infile);
  fread(diff0_v, sizeof(real), dom[rank].Gfy.s3b, infile);
  fread(conv0_v, sizeof(real), dom[rank].Gfy.s3b, infile);
  fread(diff_v, sizeof(real), dom[rank].Gfy.s3b, infile);
  fread(conv_v, sizeof(real), dom[rank].Gfy.s3b, infile);
  fread(v_star, sizeof(real), dom[rank].Gfy.s3b, infile);

  fread(w, sizeof(real), dom[rank].Gfz.s3b, infile);
  fread(w0, sizeof(real), dom[rank].Gfz.s3b, infile);
  fread(diff0_w, sizeof(real), dom[rank].Gfz.s3b, infile);
  fread(conv0_w, sizeof(real), dom[rank].Gfz.s3b, infile);
  fread(diff_w, sizeof(real), dom[rank].Gfz.s3b, infile);
  fread(conv_w, sizeof(real), dom[rank].Gfz.s3b, infile);
  fread(w_star, sizeof(real), dom[rank].Gfz.s3b, infile);

  fread(p, sizeof(real), dom[rank].Gcc.s3b, infile);
  fread(phi, sizeof(real), dom[rank].Gcc.s3b, infile);
  fread(p0, sizeof(real), dom[rank].Gcc.s3b, infile);
  fread(phase, sizeof(int), dom[rank].Gcc.s3b, infile);
  fread(phase_shell, sizeof(int), dom[rank].Gcc.s3b, infile);

  fread(flag_u, sizeof(int), dom[rank].Gfx.s3b, infile);
  fread(flag_v, sizeof(int), dom[rank].Gfy.s3b, infile);
  fread(flag_w, sizeof(int), dom[rank].Gfz.s3b, infile);

  fread(&nparts_subdom, sizeof(int), 1, infile);
  nparts = nparts_subdom;

  free(parts); // malloc'd in parts read input
  parts = (part_struct*) malloc(nparts * sizeof(part_struct));

  fread(parts, sizeof(part_struct), nparts_subdom, infile);


  // check if "translating", "rotating" are changed in part.config
  part_struct *PARTS;
  PARTS = (part_struct*) malloc(NPARTS * sizeof(part_struct));

  char ffname[FILE_NAME_SIZE] = "";
  sprintf(ffname, "%s/%s/part.config", ROOT_DIR, INPUT_DIR);
  FILE *partfile = fopen(ffname, "r");
  if (partfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", ffname);
    exit(EXIT_FAILURE);
  }

  if (NPARTS > 0) {
    fscanf(partfile, "n %d\n", &NPARTS);
    fscanf(partfile, "(l/a) %lf\n", &interaction_length_ratio);

    for (int i = 0; i < NPARTS; i++) {
      PARTS[i].N = i;
      fscanf(partfile, "\n");
      fscanf(partfile, "r %lf\n", &PARTS[i].r);
      fscanf(partfile, "(x, y, z) %lf %lf %lf\n", &PARTS[i].x, &PARTS[i].y, &PARTS[i].z);
      fscanf(partfile, "(u, v, w) %lf %lf %lf\n", &PARTS[i].u, &PARTS[i].v, &PARTS[i].w);
      fscanf(partfile, "(aFx, aFy, aFz) %lf %lf %lf\n", &PARTS[i].aFx, &PARTS[i].aFy, &PARTS[i].aFz);
      fscanf(partfile, "(aLx, aLy, aLz) %lf %lf %lf\n", &PARTS[i].aLx, &PARTS[i].aLy, &PARTS[i].aLz);
      fscanf(partfile, "rho %lf\n", &PARTS[i].rho);
      fscanf(partfile, "E %lf\n", &PARTS[i].E);
      fscanf(partfile, "sigma %lf\n", &PARTS[i].sigma);
      fscanf(partfile, "e_dry %lf\n", &PARTS[i].e_dry);
      fscanf(partfile, "coeff_fric %lf\n", &PARTS[i].coeff_fric);
      fscanf(partfile, "order %d\n", &PARTS[i].order);
      fscanf(partfile, "rs/r %lf\n", &PARTS[i].rs);
      fscanf(partfile, "spring_k %lf\n", &PARTS[i].spring_k);
      fscanf(partfile, "spring (x, y, z) %lf %lf %lf\n", &PARTS[i].spring_x, &PARTS[i].spring_y, &PARTS[i].spring_z);
      fscanf(partfile, "spring_l %lf\n", &PARTS[i].spring_l);
      fscanf(partfile, "translating %d\n", &PARTS[i].translating);
      fscanf(partfile, "rotating %d\n", &PARTS[i].rotating);
      fscanf(partfile, "s %lf\n", &PARTS[i].s);
      fscanf(partfile, "update %d\n", &PARTS[i].update);
      fscanf(partfile, "cp %lf\n", &PARTS[i].cp);
      fscanf(partfile, "rs %lf\n", &PARTS[i].srs);
      fscanf(partfile, "s_order %d\n", &PARTS[i].sorder);
    }
  }

  int wng = 1;
  if (nparts > 0) {
    for (int i = 0; i < nparts; i++) {
      if (parts[i].translating != PARTS[parts[i].N].translating || parts[i].rotating != PARTS[parts[i].N].rotating) {
        parts[i].translating = PARTS[parts[i].N].translating;
        parts[i].rotating = PARTS[parts[i].N].rotating;
        if (wng == 1) {
          printf("N%d >> part moving updated\n", rank);
          wng = 0;
        }
      }
      if (abs(parts[i].E - PARTS[parts[i].N].E) > 1e-6) {
        parts[i].E = PARTS[parts[i].N].E;
        if (wng == 1) {
          printf("N%d >> part E updated\n", rank);
          wng = 0;
        }
      }
    }
  }

  free(PARTS);
  fclose(partfile);


  fread(&ncoeffs_max, sizeof(int), 1, infile);

  fread(&pid_int, sizeof(real), 1, infile);
  fread(&pid_back, sizeof(real), 1, infile);
  fread(&gradP.z, sizeof(real), 1, infile);

  if (SCALAR >= 1) {
    fread(s, sizeof(real), dom[rank].Gcc.s3b, infile);
    fread(s0, sizeof(real), dom[rank].Gcc.s3b, infile);
    fread(s_conv, sizeof(real), dom[rank].Gcc.s3b, infile);
    fread(s_conv0, sizeof(real), dom[rank].Gcc.s3b, infile);
    fread(s_diff, sizeof(real), dom[rank].Gcc.s3b, infile);
    fread(s_diff0, sizeof(real), dom[rank].Gcc.s3b, infile);
    fread(&s_ncoeffs_max, sizeof(int), 1, infile);
  }
  //fwrite(phase, sizeof(int), dom[rank].Gcc.s3b, infile);
  //fwrite(phase_shell, sizeof(int), dom[rank].Gcc.s3b, infile);
  //fwrite(&rec_particle_ttime_out, sizeof(real), 1, infile);
  //fwrite(&rec_prec_ttime_out, sizeof(real), 1, infile);

  // close the file
  fclose(infile);
}

void count_mem(void)
{
  cpumem *= 1e-6;   // Convert bytes -> megabytes
  gpumem *= 1e-6;

  // what about cuda structs (threads and blocks), mpi structs....

  // this is fit in bluebottle/prof/mem-usage for 1 gpu
  real gpureal = 0.9835*((real) gpumem) + 118.6961;
  printf("N%d >> Total CPU memory usage...       %5ldMB\n", rank, cpumem);
  printf("N%d >> Estimated GPU memory usage...   %5.lfMB\n", rank, gpureal);
}

void domain_free(void)
{
  //printf("N%d >> Freeing domain information...\n", rank);
  free(dom);
}
