/*******************************************************************************
 ******************************** BLUEBOTTLE ***********************************
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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cgns.h"

void cgns_recorder_init(void)
{
  if (rec_cgns_flow_dt > 0 || rec_cgns_part_dt > 0) {
    #ifdef DDEBUG
      cgns_grid_ghost();
    #else
      cgns_grid();
    #endif // DDEBUG
  }
}

void cgns_recorder_flow_write(void)
{
  if (rec_cgns_flow_dt > 0) {
    if (rec_cgns_flow_ttime_out >= rec_cgns_flow_dt || stepnum == 0) {
      
      // Write
      #ifdef DDEBUG
        cgns_flow_field_ghost(rec_cgns_flow_dt);
      #else
        cgns_flow_field(rec_cgns_flow_dt);
      #endif // DDEBUG

      // Notify output
      char fname[FILE_NAME_SIZE];
      int sigfigs = ceil(log10(1. / rec_cgns_flow_dt));
      if(sigfigs < 1) sigfigs = 1;
      sprintf(fname, "flow-%.*f.cgns", sigfigs, ttime);
      printf("N%d >> Writing %s at t = %e... done.\n", rank, fname, ttime);

      // Don't want to subtract if first output
      if (stepnum != 0) rec_cgns_flow_ttime_out -= rec_cgns_flow_dt;
    }
  }
}

void cgns_recorder_part_write(void)
{
  if (rec_cgns_part_dt > 0) {
    if (rec_cgns_part_ttime_out >= rec_cgns_part_dt || stepnum == 0) {

      // Write
      #ifdef DDEBUG
        // NOTE: currently no debug output for cgns particles
        cgns_particles(rec_cgns_part_dt);
      #else
        cgns_particles(rec_cgns_part_dt);
      #endif

      // Notify output
      char fname[FILE_NAME_SIZE];
      int sigfigs = ceil(log10(1. / rec_cgns_part_dt));
      if(sigfigs < 1) sigfigs = 1;
      sprintf(fname, "part-%.*f.cgns", sigfigs, ttime);
      printf("N%d >> Writing %s at t = %e... done.\n", rank, fname, ttime);

      // Don't want to subtract if first output
      if (stepnum != 0) rec_cgns_part_ttime_out -= rec_cgns_part_dt;
    }
  }
}

void cgns_grid(void)
{
  printf("N%d >> Initializing CGNS grid...\n", rank);

  if (rank == 0) {
    // Create output directory if it doesn't exist
    struct stat st = {0};
    char buf[CHAR_BUF_SIZE];
    sprintf(buf, "%s/%s", ROOT_DIR, OUTPUT_DIR);
    if (stat(buf, &st) == -1) {
      mkdir(buf, 0700);
    }
  }
  
  // create the file
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/%s/%s", ROOT_DIR, OUTPUT_DIR, "grid.cgns");
  int fn;   // file index number
  int bn;   // base index number
  int zn;   // zone index number
  int gn;   // grid "     "
  int cx;   // coord "     "
  int cy;
  int cz;

  // Set MPI communicator for CGNS
  cgp_mpi_comm(MPI_COMM_WORLD);

  // Open file in parallel
  cgp_open(fname, CG_MODE_WRITE, &fn);

  // Write base, zone
  int cell_dim = 3;   // volume cells
  int phys_dim = 3;   // # coords to define a vector
  cg_base_write(fn, "Base", cell_dim, phys_dim, &bn);

  cgsize_t size[9];     // number of vertices and cells
  size[0] = DOM.xn + 1; // number of vertices
  size[1] = DOM.yn + 1;
  size[2] = DOM.zn + 1;
  size[3] = DOM.xn;     // number of cells
  size[4] = DOM.yn;
  size[5] = DOM.zn;
  size[6] = 0;          // 0
  size[7] = 0;
  size[8] = 0;
  cg_zone_write(fn, bn, "Zone0", size, Structured, &zn);

  // grid
  cg_grid_write(fn, bn, zn, "GridCoordinates", &gn);

  // create data nodes for coordinates
  cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateX", &cx);
  cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateY", &cy);
  cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateZ", &cz);

  // number of vertices and range for current rank
  int nverts = (dom[rank].xn + 1)*(dom[rank].yn + 1)*(dom[rank].zn + 1);
  real *x = malloc(nverts * sizeof(real)); 
  real *y = malloc(nverts * sizeof(real)); 
  real *z = malloc(nverts * sizeof(real)); 

  // find position
  int s1 = dom[rank].xn + 1;
  int s2 = (dom[rank].xn + 1)*(dom[rank].yn + 1);
  for (int k = dom[rank].Gcc._ks; k <= (dom[rank].Gcc._ke + 1); k++) {
    for (int j = dom[rank].Gcc._js; j <= (dom[rank].Gcc._je + 1); j++) {
      for (int i = dom[rank].Gcc._is; i <= (dom[rank].Gcc._ie + 1); i++) {
        int C = (i-1) + (j-1)*s1 + (k-1)*s2;
        x[C] = dom[rank].xs + (i-1)*dom[rank].dx;
        y[C] = dom[rank].ys + (j-1)*dom[rank].dy;
        z[C] = dom[rank].zs + (k-1)*dom[rank].dz;
      }
    }
  }

  // start and end index for write (one-indexed) 
  cgsize_t nstart[3];
  nstart[0] = dom[rank].Gcc.is;
  nstart[1] = dom[rank].Gcc.js;
  nstart[2] = dom[rank].Gcc.ks;

  cgsize_t nend[3];
  nend[0] = dom[rank].Gcc.ie + 1;
  nend[1] = dom[rank].Gcc.je + 1;
  nend[2] = dom[rank].Gcc.ke + 1;

  // write the coordinate data in parallel
  cgp_coord_write_data(fn, bn, zn, cx, nstart, nend, x);
  cgp_coord_write_data(fn, bn, zn, cy, nstart, nend, y);
  cgp_coord_write_data(fn, bn, zn, cz, nstart, nend, z);

  free(x);
  free(y);
  free(z);

  cgp_close(fn);
}

void cgns_grid_ghost(void)
{
  printf("N%d >> Initializing CGNS ghost grid...\n", rank);

  if (rank == 0) {
    // Create output directory if it doesn't exist
    struct stat st = {0};
    char buf[CHAR_BUF_SIZE];
    sprintf(buf, "%s/%s", ROOT_DIR, OUTPUT_DIR);
    if (stat(buf, &st) == -1) {
      mkdir(buf, 0700);
    }
  }

  // create the file
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/%s/%s", ROOT_DIR, OUTPUT_DIR, "grid-ghost.cgns");
  int fn;   // cgns indices
  int bn;
  int zn;
  int gn;
  int cx, cy, cz;
  int ierr;

  // set mpi comm
  cgp_mpi_comm(MPI_COMM_WORLD);

  // Open file in parallel
  ierr = cgp_open(fname, CG_MODE_WRITE, &fn);
  if (ierr != 0) {
    printf("N%d >> Line %d, ierr = %d\n", rank, __LINE__, ierr);
    cgp_error_exit();
  }

  // Write base, zone
  int cell_dim = 3;   // volume cells
  int phys_dim = 3;   // # coords to define a vector
  ierr = cg_base_write(fn, "Base", cell_dim, phys_dim, &bn);
  if (ierr != 0) {
    printf("N%d >> Line %d\n", rank, __LINE__);
    cgp_error_exit();
  }

  cgsize_t size[9];          // number of vertices and cells
  size[0] = DOM.Gcc.inb + 1; // number of vertices
  size[1] = DOM.Gcc.jnb + 1;
  size[2] = DOM.Gcc.knb + 1;
  size[3] = DOM.Gcc.inb;     // number of cells
  size[4] = DOM.Gcc.jnb;
  size[5] = DOM.Gcc.knb;
  size[6] = 0;               // 0
  size[7] = 0;
  size[8] = 0;
  if (cg_zone_write(fn, bn, "Zone0", size, Structured, &zn))
    cgp_error_exit();

  // grid
  if (cg_grid_write(fn, bn, zn, "GridCoordinates", &gn))
    cgp_error_exit();

  // create data nodes for coordinates
  if (cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateX", &cx))
    cgp_error_exit();
  if (cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateY", &cy))
    cgp_error_exit();
  if (cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateZ", &cz))
    cgp_error_exit();

  // number of vertices and range for current rank
  //int nvert = (DOM.Gcc.inb + 1)*(DOM.Gcc.jnb + 1)*(DOM.Gcc.knb + 1);
  int nvert = (dom[rank].Gcc.inb + 1)*(dom[rank].Gcc.jnb + 1)*(dom[rank].Gcc.knb + 1);
  real *x = malloc(nvert * sizeof(real));
  real *y = malloc(nvert * sizeof(real));
  real *z = malloc(nvert * sizeof(real));

  // find position
  int s1 = dom[rank].Gcc.inb + 1;
  int s2 = s1*(dom[rank].Gcc.jnb + 1);
  int C;
  for (int k = dom[rank].Gcc._ksb; k <= (dom[rank].Gcc._keb + 1); k++) {
    for (int j = dom[rank].Gcc._jsb; j <= (dom[rank].Gcc._jeb + 1); j++) {
      for (int i = dom[rank].Gcc._isb; i <= (dom[rank].Gcc._ieb + 1); i++) {
        C = GCC_LOC(i, j, k, s1, s2);
        x[C] = dom[rank].xs + (i-1)*dom[rank].dx;
        y[C] = dom[rank].ys + (j-1)*dom[rank].dy;
        z[C] = dom[rank].zs + (k-1)*dom[rank].dz;
      }
    }
  }

  // start and end index for writes (add 1 for one-indexed)
  cgsize_t nstart[3];
  nstart[0] = dom[rank].Gcc.isb + 1;
  nstart[1] = dom[rank].Gcc.jsb + 1;
  nstart[2] = dom[rank].Gcc.ksb + 1;

  cgsize_t nend[3]; // +1 for 1-indexing, +1 for vertices = 2
  nend[0] = dom[rank].Gcc.ieb + 1 + 1;
  nend[1] = dom[rank].Gcc.jeb + 1 + 1;
  nend[2] = dom[rank].Gcc.keb + 1 + 1;

  // write the coordinate data in parallel
  if (cgp_coord_write_data(fn, bn, zn, cx, nstart, nend, x))
    cgp_error_exit();
  if (cgp_coord_write_data(fn, bn, zn, cy, nstart, nend, y))
    cgp_error_exit();
  if (cgp_coord_write_data(fn, bn, zn, cz, nstart, nend, z))
    cgp_error_exit();

  free(x);
  free(y);
  free(z);

  if (cgp_close(fn))
    cgp_error_exit();
}

void cgns_flow_field(real dtout)
{
  // create the solution file names
  char fname[FILE_NAME_SIZE];
  int sigfigs = ceil(log10(1. / dtout));
  if(sigfigs < 1) sigfigs = 1;
  sprintf(fname, "%s/%s/flow-%.*f.cgns", ROOT_DIR, OUTPUT_DIR, sigfigs, ttime);

  char gname[FILE_NAME_SIZE];
  sprintf(gname, "grid.cgns");

  // cgns file indices
  int fn;
  int bn;
  int zn;
  int sn;
  int fnp;
  int fnu;
  int fnv;
  int fnw;

  // set mpi comm
  cgp_mpi_comm(MPI_COMM_WORLD);

  // open file in parallel
  cgp_open(fname, CG_MODE_WRITE, &fn);

  // write base, zone
  int cell_dim = 3;   // volume cells
  int phys_dim = 3;   // # coords to define a vector
  cg_base_write(fn, "Base", cell_dim, phys_dim, &bn);

  cgsize_t size[9];     // number of vertices and cells
  size[0] = DOM.xn + 1; // vertices
  size[1] = DOM.yn + 1;
  size[2] = DOM.zn + 1;
  size[3] = DOM.xn;     // cells
  size[4] = DOM.yn;
  size[5] = DOM.zn;
  size[6] = 0;
  size[7] = 0;
  size[8] = 0;
  cg_zone_write(fn, bn, "Zone0", size, Structured, &zn);
  cg_goto(fn, bn, "Zone_t", zn, "end");

  // check that grid.cgns exists
  /*int fng;
  if(cg_open(gnameall, CG_MODE_READ, &fng) != 0) {
    fprintf(stderr, "CGNS flow field write failure: no grid.cgns\n");
    exit(EXIT_FAILURE);
  } else {
    cg_close(fng);
  }
    cg_close(fng);
*/
  
  // link grid.cgns
  cg_link_write("GridCoordinates", gname, "Base/Zone0/GridCoordinates");

  // create solution
  cg_sol_write(fn, bn, zn, "Solution", CellCenter, &sn);

  // start and end index for write
  cgsize_t nstart[3];
  nstart[0] = dom[rank].Gcc.is;
  nstart[1] = dom[rank].Gcc.js;
  nstart[2] = dom[rank].Gcc.ks;

  cgsize_t nend[3];
  nend[0] = dom[rank].Gcc.ie;
  nend[1] = dom[rank].Gcc.je;
  nend[2] = dom[rank].Gcc.ke;

  // create and write the field data -- pressure
  cgp_field_write(fn, bn, zn, sn, RealDouble, "Pressure", &fnp);

  real *pout = malloc(dom[rank].Gcc.s3 * sizeof(real));
  for (int k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (int j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (int i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        int C = GCC_LOC(i - DOM_BUF, j - DOM_BUF, k - DOM_BUF,
                        dom[rank].Gcc.s1, dom[rank].Gcc.s2);
        int CC = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        pout[C] = p[CC];
      }
    }
  }

  cgp_field_write_data(fn, bn, zn, sn, fnp, nstart, nend, pout);
  free(pout);

  // create and write the field data
  cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityX", &fnu);
  cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityY", &fnv);
  cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityZ", &fnw);

  real *uout = malloc(dom[rank].Gcc.s3 * sizeof(real));
  real *vout = malloc(dom[rank].Gcc.s3 * sizeof(real));
  real *wout = malloc(dom[rank].Gcc.s3 * sizeof(real));
  for (int k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (int j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (int i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        int C = GCC_LOC(i - DOM_BUF, j - DOM_BUF, k - DOM_BUF, 
                          dom[rank].Gcc.s1, dom[rank].Gcc.s2);

        int Cfx_w = GFX_LOC(i - 1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        int Cfx = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        int Cfx_e = GFX_LOC(i + 1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        int Cfx_ee = GFX_LOC(i + 2, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        int Cfy_s =  GFY_LOC(i, j - 1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        int Cfy = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        int Cfy_n =  GFY_LOC(i, j + 1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        int Cfy_nn = GFY_LOC(i, j + 2, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        int Cfz_b = GFZ_LOC(i, j, k - 1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        int Cfz = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        int Cfz_t = GFZ_LOC(i, j, k + 1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        int Cfz_tt = GFZ_LOC(i, j, k + 2, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        uout[C] = -0.0625*u[Cfx_w] + 0.5625*u[Cfx] + 0.5625*u[Cfx_e] 
                    -0.0625*u[Cfx_ee];
        vout[C] = -0.0625*v[Cfy_s] + 0.5625*v[Cfy] + 0.5625*v[Cfy_n] 
                    -0.0625*v[Cfy_nn];
        wout[C] = -0.0625*w[Cfz_b] + 0.5625*w[Cfz] + 0.5625*w[Cfz_t] 
                    -0.0625*w[Cfz_tt];
      }
    }
  }

  cgp_field_write_data(fn, bn, zn, sn, fnu, nstart, nend, uout);
  cgp_field_write_data(fn, bn, zn, sn, fnv, nstart, nend, vout);
  cgp_field_write_data(fn, bn, zn, sn, fnw, nstart, nend, wout);

  free(uout);
  free(vout);
  free(wout);

  // phase
  cgp_field_write(fn, bn, zn, sn, Integer, "Phase", &fnp);
  int *phase_out = malloc(dom[rank].Gcc.s3 * sizeof(int));

  for (int k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (int j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (int i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        int C = GCC_LOC(i - DOM_BUF, j - DOM_BUF, k - DOM_BUF,
                        dom[rank].Gcc.s1, dom[rank].Gcc.s2);
        int CC = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        // mask phase: -1 for fluid, 0 for particle
        // This is because phase is set to the local particle index on the
        // device, which includes ghost particles. When the part_struct is
        // pulled to the host, we don't pull ghost particles, so the indexing
        // in the part_struct changes
        phase_out[C] = phase[CC] * (phase[CC] < 0);

      }
    }
  }

  cgp_field_write_data(fn, bn, zn, sn, fnp, nstart, nend, phase_out);
  free(phase_out);

  // phase_shell
  cgp_field_write(fn, bn, zn, sn, Integer, "PhaseShell", &fnp);
  int *phaseshell_out = malloc(dom[rank].Gcc.s3 * sizeof(int));

  for (int k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (int j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (int i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        int C = GCC_LOC(i - DOM_BUF, j - DOM_BUF, k - DOM_BUF,
                        dom[rank].Gcc.s1, dom[rank].Gcc.s2);
        int CC = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        phaseshell_out[C] = phase_shell[CC];
      }
    }
  }

  cgp_field_write_data(fn, bn, zn, sn, fnp, nstart, nend, phaseshell_out);
  free(phaseshell_out);

  // Misc data -- scalars
  cg_user_data_write("Etc");
  cg_goto(fn, bn, "Zone_t", zn, "Etc", 0, "end");

  int fnt, fnrho, fnnu;
  cgsize_t nele = 1;

  cgsize_t N[1];
  N[0] = 1;
  
  cgp_array_write("Time", RealDouble, 1, &nele, &fnt);
  cgp_array_write("Density", RealDouble, 1, &nele, &fnrho);
  cgp_array_write("KinematicViscosity", RealDouble, 1, &nele, &fnnu);

  cgp_array_write_data(fnt, N, N, &ttime);
  cgp_array_write_data(fnrho, N, N, &rho_f);
  cgp_array_write_data(fnnu, N, N, &nu);

  cgp_close(fn);
}

void cgns_particles(real dtout)
{
  // Determine number of particles in subdomain (NOT ghost particles)
  // Inherited from cuda_part_pull

  // create a new communicator for particles that will write
  // (cg calls are collective; i.e. all procs in communicator need to call)
  int color = MPI_UNDEFINED * (nparts_subdom == 0);
  int key = rank;
  MPI_Comm part_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, key, &part_comm);

  if (NPARTS > 0 && nparts_subdom > 0) {
    // create the solution file names
    char fname[FILE_NAME_SIZE];
    int sigfigs = ceil(log10(1. / dtout));
    if(sigfigs < 1) sigfigs = 1;
    sprintf(fname, "%s/%s/part-%.*f.cgns", ROOT_DIR, OUTPUT_DIR, sigfigs, ttime);

    //printf("N%d >> Starting cgns output...\n", rank);

    // cgns file indices
    int fn;
    int bn;
    int zn;
    int sn;
    int en;
    int cx;
    int cy;
    int cz;
    int fnr;

    // Set cg mpi communicator
    cgp_mpi_comm(part_comm);

    // open file in parallel
    if (cgp_open(fname, CG_MODE_WRITE, &fn)) {
      printf("N%d >> Line %d\n", rank, __LINE__);
      cgp_error_exit();
    }

    // write base, zone
    int cell_dim = 3;   // volume cells
    int phys_dim = 3;   // # coords to define a vector
    if (cg_base_write(fn, "Base", cell_dim, phys_dim, &bn)) {
      printf("N%d >> Line %d\n", rank, __LINE__);
      cgp_error_exit();
    }

    cgsize_t size[3][1];
    size[0][0] = NPARTS;
    size[1][0] = 0;
    size[2][0] = 0;

    if (cg_zone_write(fn, bn, "Zone0", size[0], Unstructured, &zn)) {
      printf("N%d >> Line %d\n", rank, __LINE__);
      cgp_error_exit();
    }
    if (cg_goto(fn, bn, "Zone_t", zn, "end")) {
      cgp_error_exit();
    }

    // MPI_Excan destroys nparts_subdom, so make a tmp nparts that we can
    //  destroy
    int tmp_nparts = nparts_subdom;

    // Get offset in cgns file for current rank
    int offset = 0;
    MPI_Exscan(&tmp_nparts, &offset, 1, MPI_INT, MPI_SUM, part_comm);

    // Start and ending indices for write (indexed from 1)
    cgsize_t nstart[1];
    cgsize_t nend[1];
    nstart[0] = offset + 1;
    nend[0] = offset + nparts_subdom; // + 1 - 1;

    // write particle data to arrays
    real *x = malloc(nparts_subdom * sizeof(real));
    real *y = malloc(nparts_subdom * sizeof(real));
    real *z = malloc(nparts_subdom * sizeof(real));

    real *u = malloc(nparts_subdom * sizeof(real));
    real *v = malloc(nparts_subdom * sizeof(real));
    real *w = malloc(nparts_subdom * sizeof(real));
    real *udot = malloc(nparts_subdom * sizeof(real));  // accel
    real *vdot = malloc(nparts_subdom * sizeof(real));
    real *wdot = malloc(nparts_subdom * sizeof(real));

    real *ox = malloc(nparts_subdom * sizeof(real));  // anngular vel
    real *oy = malloc(nparts_subdom * sizeof(real));
    real *oz = malloc(nparts_subdom * sizeof(real));
    real *oxdot = malloc(nparts_subdom * sizeof(real));
    real *oydot = malloc(nparts_subdom * sizeof(real));
    real *ozdot = malloc(nparts_subdom * sizeof(real));

    real *Fx = malloc(nparts_subdom * sizeof(real));  // total
    real *Fy = malloc(nparts_subdom * sizeof(real));
    real *Fz = malloc(nparts_subdom * sizeof(real));
    real *iFx = malloc(nparts_subdom * sizeof(real)); // interaction
    real *iFy = malloc(nparts_subdom * sizeof(real));
    real *iFz = malloc(nparts_subdom * sizeof(real));
    real *hFx = malloc(nparts_subdom * sizeof(real)); // hydro
    real *hFy = malloc(nparts_subdom * sizeof(real));
    real *hFz = malloc(nparts_subdom * sizeof(real));
    real *kFx = malloc(nparts_subdom * sizeof(real)); // hydro
    real *kFy = malloc(nparts_subdom * sizeof(real));
    real *kFz = malloc(nparts_subdom * sizeof(real));

    real *Lx = malloc(nparts_subdom * sizeof(real));  // total
    real *Ly = malloc(nparts_subdom * sizeof(real));
    real *Lz = malloc(nparts_subdom * sizeof(real));

    real *axx = malloc(nparts_subdom * sizeof(real));
    real *axy = malloc(nparts_subdom * sizeof(real));
    real *axz = malloc(nparts_subdom * sizeof(real));
    real *ayx = malloc(nparts_subdom * sizeof(real));
    real *ayy = malloc(nparts_subdom * sizeof(real));
    real *ayz = malloc(nparts_subdom * sizeof(real));
    real *azx = malloc(nparts_subdom * sizeof(real));
    real *azy = malloc(nparts_subdom * sizeof(real));
    real *azz = malloc(nparts_subdom * sizeof(real));

    real *a = malloc(nparts_subdom * sizeof(real));
    int *N = malloc(nparts_subdom * sizeof(int));
    cgsize_t *conn = malloc(nparts_subdom * sizeof(cgsize_t));

    real *density = malloc(nparts_subdom * sizeof(real));
    real *E = malloc(nparts_subdom * sizeof(real));
    real *sigma = malloc(nparts_subdom * sizeof(real));
    real *coeff_fric = malloc(nparts_subdom * sizeof(real));
    real *e_dry = malloc(nparts_subdom * sizeof(real));
    int *order = malloc(nparts_subdom * sizeof(int));

    int *ncoll_part = malloc(nparts_subdom * sizeof(int));
    int *ncoll_wall = malloc(nparts_subdom * sizeof(int));

    real   *p00_r = malloc(nparts_subdom* sizeof(real));
    real   *p00_i = malloc(nparts_subdom* sizeof(real));
    real *phi00_r = malloc(nparts_subdom* sizeof(real));
    real *phi00_i = malloc(nparts_subdom* sizeof(real));
    real *chi00_r = malloc(nparts_subdom* sizeof(real));
    real *chi00_i = malloc(nparts_subdom* sizeof(real));
    real   *p10_r = malloc(nparts_subdom* sizeof(real));
    real   *p10_i = malloc(nparts_subdom* sizeof(real));
    real *phi10_r = malloc(nparts_subdom* sizeof(real));
    real *phi10_i = malloc(nparts_subdom* sizeof(real));
    real *chi10_r = malloc(nparts_subdom* sizeof(real));
    real *chi10_i = malloc(nparts_subdom* sizeof(real));
    real   *p11_r = malloc(nparts_subdom* sizeof(real));
    real   *p11_i = malloc(nparts_subdom* sizeof(real));
    real *phi11_r = malloc(nparts_subdom* sizeof(real));
    real *phi11_i = malloc(nparts_subdom* sizeof(real));
    real *chi11_r = malloc(nparts_subdom* sizeof(real));
    real *chi11_i = malloc(nparts_subdom* sizeof(real));
    real   *p20_r = malloc(nparts_subdom* sizeof(real));
    real   *p20_i = malloc(nparts_subdom* sizeof(real));
    real *phi20_r = malloc(nparts_subdom* sizeof(real));
    real *phi20_i = malloc(nparts_subdom* sizeof(real));
    real *chi20_r = malloc(nparts_subdom* sizeof(real));
    real *chi20_i = malloc(nparts_subdom* sizeof(real));
    real   *p21_r = malloc(nparts_subdom* sizeof(real));
    real   *p21_i = malloc(nparts_subdom* sizeof(real));
    real *phi21_r = malloc(nparts_subdom* sizeof(real));
    real *phi21_i = malloc(nparts_subdom* sizeof(real));
    real *chi21_r = malloc(nparts_subdom* sizeof(real));
    real *chi21_i = malloc(nparts_subdom* sizeof(real));
    real   *p22_r = malloc(nparts_subdom* sizeof(real));
    real   *p22_i = malloc(nparts_subdom* sizeof(real));
    real *phi22_r = malloc(nparts_subdom* sizeof(real));
    real *phi22_i = malloc(nparts_subdom* sizeof(real));
    real *chi22_r = malloc(nparts_subdom* sizeof(real));
    real *chi22_i = malloc(nparts_subdom* sizeof(real));
    real   *p30_r = malloc(nparts_subdom* sizeof(real));
    real   *p30_i = malloc(nparts_subdom* sizeof(real));
    real *phi30_r = malloc(nparts_subdom* sizeof(real));
    real *phi30_i = malloc(nparts_subdom* sizeof(real));
    real *chi30_r = malloc(nparts_subdom* sizeof(real));
    real *chi30_i = malloc(nparts_subdom* sizeof(real));
    real   *p31_r = malloc(nparts_subdom* sizeof(real));
    real   *p31_i = malloc(nparts_subdom* sizeof(real));
    real *phi31_r = malloc(nparts_subdom* sizeof(real));
    real *phi31_i = malloc(nparts_subdom* sizeof(real));
    real *chi31_r = malloc(nparts_subdom* sizeof(real));
    real *chi31_i = malloc(nparts_subdom* sizeof(real));
    real   *p32_r = malloc(nparts_subdom* sizeof(real));
    real   *p32_i = malloc(nparts_subdom* sizeof(real));
    real *phi32_r = malloc(nparts_subdom* sizeof(real));
    real *phi32_i = malloc(nparts_subdom* sizeof(real));
    real *chi32_r = malloc(nparts_subdom* sizeof(real));
    real *chi32_i = malloc(nparts_subdom* sizeof(real));
    real   *p33_r = malloc(nparts_subdom* sizeof(real));
    real   *p33_i = malloc(nparts_subdom* sizeof(real));
    real *phi33_r = malloc(nparts_subdom* sizeof(real));
    real *phi33_i = malloc(nparts_subdom* sizeof(real));
    real *chi33_r = malloc(nparts_subdom* sizeof(real));
    real *chi33_i = malloc(nparts_subdom* sizeof(real));
    real   *p40_r = malloc(nparts_subdom* sizeof(real));
    real   *p40_i = malloc(nparts_subdom* sizeof(real));
    real *phi40_r = malloc(nparts_subdom* sizeof(real));
    real *phi40_i = malloc(nparts_subdom* sizeof(real));
    real *chi40_r = malloc(nparts_subdom* sizeof(real));
    real *chi40_i = malloc(nparts_subdom* sizeof(real));
    real   *p41_r = malloc(nparts_subdom* sizeof(real));
    real   *p41_i = malloc(nparts_subdom* sizeof(real));
    real *phi41_r = malloc(nparts_subdom* sizeof(real));
    real *phi41_i = malloc(nparts_subdom* sizeof(real));
    real *chi41_r = malloc(nparts_subdom* sizeof(real));
    real *chi41_i = malloc(nparts_subdom* sizeof(real));
    real   *p42_r = malloc(nparts_subdom* sizeof(real));
    real   *p42_i = malloc(nparts_subdom* sizeof(real));
    real *phi42_r = malloc(nparts_subdom* sizeof(real));
    real *phi42_i = malloc(nparts_subdom* sizeof(real));
    real *chi42_r = malloc(nparts_subdom* sizeof(real));
    real *chi42_i = malloc(nparts_subdom* sizeof(real));
    real   *p43_r = malloc(nparts_subdom* sizeof(real));
    real   *p43_i = malloc(nparts_subdom* sizeof(real));
    real *phi43_r = malloc(nparts_subdom* sizeof(real));
    real *phi43_i = malloc(nparts_subdom* sizeof(real));
    real *chi43_r = malloc(nparts_subdom* sizeof(real));
    real *chi43_i = malloc(nparts_subdom* sizeof(real));
    real   *p44_r = malloc(nparts_subdom* sizeof(real));
    real   *p44_i = malloc(nparts_subdom* sizeof(real));
    real *phi44_r = malloc(nparts_subdom* sizeof(real));
    real *phi44_i = malloc(nparts_subdom* sizeof(real));
    real *chi44_r = malloc(nparts_subdom* sizeof(real));
    real *chi44_i = malloc(nparts_subdom* sizeof(real));
    real   *p50_r = malloc(nparts_subdom* sizeof(real));
    real   *p50_i = malloc(nparts_subdom* sizeof(real));
    real *phi50_r = malloc(nparts_subdom* sizeof(real));
    real *phi50_i = malloc(nparts_subdom* sizeof(real));
    real *chi50_r = malloc(nparts_subdom* sizeof(real));
    real *chi50_i = malloc(nparts_subdom* sizeof(real));
    real   *p51_r = malloc(nparts_subdom* sizeof(real));
    real   *p51_i = malloc(nparts_subdom* sizeof(real));
    real *phi51_r = malloc(nparts_subdom* sizeof(real));
    real *phi51_i = malloc(nparts_subdom* sizeof(real));
    real *chi51_r = malloc(nparts_subdom* sizeof(real));
    real *chi51_i = malloc(nparts_subdom* sizeof(real));
    real   *p52_r = malloc(nparts_subdom* sizeof(real));
    real   *p52_i = malloc(nparts_subdom* sizeof(real));
    real *phi52_r = malloc(nparts_subdom* sizeof(real));
    real *phi52_i = malloc(nparts_subdom* sizeof(real));
    real *chi52_r = malloc(nparts_subdom* sizeof(real));
    real *chi52_i = malloc(nparts_subdom* sizeof(real));
    real   *p53_r = malloc(nparts_subdom* sizeof(real));
    real   *p53_i = malloc(nparts_subdom* sizeof(real));
    real *phi53_r = malloc(nparts_subdom* sizeof(real));
    real *phi53_i = malloc(nparts_subdom* sizeof(real));
    real *chi53_r = malloc(nparts_subdom* sizeof(real));
    real *chi53_i = malloc(nparts_subdom* sizeof(real));
    real   *p54_r = malloc(nparts_subdom* sizeof(real));
    real   *p54_i = malloc(nparts_subdom* sizeof(real));
    real *phi54_r = malloc(nparts_subdom* sizeof(real));
    real *phi54_i = malloc(nparts_subdom* sizeof(real));
    real *chi54_r = malloc(nparts_subdom* sizeof(real));
    real *chi54_i = malloc(nparts_subdom* sizeof(real));
    real   *p55_r = malloc(nparts_subdom* sizeof(real));
    real   *p55_i = malloc(nparts_subdom* sizeof(real));
    real *phi55_r = malloc(nparts_subdom* sizeof(real));
    real *phi55_i = malloc(nparts_subdom* sizeof(real));
    real *chi55_r = malloc(nparts_subdom* sizeof(real));
    real *chi55_i = malloc(nparts_subdom* sizeof(real));


    for (int n = 0; n < nparts_subdom; n++) {
      real mass = 4./3.*PI*(parts[n].rho - rho_f) * 
                    parts[n].r * parts[n].r * parts[n].r;
      x[n] = parts[n].x;
      y[n] = parts[n].y;
      z[n] = parts[n].z;

      u[n] = parts[n].u;
      v[n] = parts[n].v;
      w[n] = parts[n].w;
      udot[n] = parts[n].udot;
      vdot[n] = parts[n].vdot;
      wdot[n] = parts[n].wdot;

      ox[n] = parts[n].ox;
      oy[n] = parts[n].oy;
      oz[n] = parts[n].oz;
      oxdot[n] = parts[n].oxdot;
      oydot[n] = parts[n].oydot;
      ozdot[n] = parts[n].ozdot;

      iFx[n] = parts[n].iFx;
      iFy[n] = parts[n].iFy;
      iFz[n] = parts[n].iFz;
      hFx[n] = parts[n].Fx;
      hFy[n] = parts[n].Fy;
      hFz[n] = parts[n].Fz;
      Fx[n] = parts[n].kFx + iFx[n] + hFx[n] + parts[n].aFx + mass*g.x;
      Fy[n] = parts[n].kFy + iFy[n] + hFy[n] + parts[n].aFy + mass*g.y;
      Fz[n] = parts[n].kFz + iFz[n] + hFz[n] + parts[n].aFz + mass*g.z;
      kFx[n] = parts[n].kFx;
      kFy[n] = parts[n].kFy;
      kFz[n] = parts[n].kFz;

      Lx[n] = parts[n].Lx;
      Ly[n] = parts[n].Ly;
      Lz[n] = parts[n].Lz;

      axx[n] = parts[n].axx;
      axy[n] = parts[n].axy;
      axz[n] = parts[n].axz;
      ayx[n] = parts[n].ayx;
      ayy[n] = parts[n].ayy;
      ayz[n] = parts[n].ayz;
      azx[n] = parts[n].azx;
      azy[n] = parts[n].azy;
      azz[n] = parts[n].azz;

      a[n] = parts[n].r;
      N[n] = parts[n].N;

      conn[n] = parts[n].N + 1; // because 0-indexed

      density[n] = parts[n].rho;
      E[n] = parts[n].E;
      sigma[n] = parts[n].sigma;
      coeff_fric[n] = parts[n].coeff_fric;
      e_dry[n] = parts[n].e_dry;
      order[n] = parts[n].order;

      ncoll_part[n] = parts[n].ncoll_part;
      ncoll_wall[n] = parts[n].ncoll_wall;

        p00_r[n] = 0;
        p00_i[n] = 0;
      phi00_r[n] = 0;
      phi00_i[n] = 0;
      chi00_r[n] = 0;
      chi00_i[n] = 0;
        p10_r[n] = 0;
        p10_i[n] = 0;
      phi10_r[n] = 0;
      phi10_i[n] = 0;
      chi10_r[n] = 0;
      chi10_i[n] = 0;
        p11_r[n] = 0;
        p11_i[n] = 0;
      phi11_r[n] = 0;
      phi11_i[n] = 0;
      chi11_r[n] = 0;
      chi11_i[n] = 0;
        p20_r[n] = 0;
        p20_i[n] = 0;
      phi20_r[n] = 0;
      phi20_i[n] = 0;
      chi20_r[n] = 0;
      chi20_i[n] = 0;
        p21_r[n] = 0;
        p21_i[n] = 0;
      phi21_r[n] = 0;
      phi21_i[n] = 0;
      chi21_r[n] = 0;
      chi21_i[n] = 0;
        p22_r[n] = 0;
        p22_i[n] = 0;
      phi22_r[n] = 0;
      phi22_i[n] = 0;
      chi22_r[n] = 0;
      chi22_i[n] = 0;
        p30_r[n] = 0;
        p30_i[n] = 0;
      phi30_r[n] = 0;
      phi30_i[n] = 0;
      chi30_r[n] = 0;
      chi30_i[n] = 0;
        p31_r[n] = 0;
        p31_i[n] = 0;
      phi31_r[n] = 0;
      phi31_i[n] = 0;
      chi31_r[n] = 0;
      chi31_i[n] = 0;
        p32_r[n] = 0;
        p32_i[n] = 0;
      phi32_r[n] = 0;
      phi32_i[n] = 0;
      chi32_r[n] = 0;
      chi32_i[n] = 0;
        p33_r[n] = 0;
        p33_i[n] = 0;
      phi33_r[n] = 0;
      phi33_i[n] = 0;
      chi33_r[n] = 0;
      chi33_i[n] = 0;
        p40_r[n] = 0;
        p40_i[n] = 0;
      phi40_r[n] = 0;
      phi40_i[n] = 0;
      chi40_r[n] = 0;
      chi40_i[n] = 0;
        p41_r[n] = 0;
        p41_i[n] = 0;
      phi41_r[n] = 0;
      phi41_i[n] = 0;
      chi41_r[n] = 0;
      chi41_i[n] = 0;
        p42_r[n] = 0;
        p42_i[n] = 0;
      phi42_r[n] = 0;
      phi42_i[n] = 0;
      chi42_r[n] = 0;
      chi42_i[n] = 0;
        p43_r[n] = 0;
        p43_i[n] = 0;
      phi43_r[n] = 0;
      phi43_i[n] = 0;
      chi43_r[n] = 0;
      chi43_i[n] = 0;
        p44_r[n] = 0;
        p44_i[n] = 0;
      phi44_r[n] = 0;
      phi44_i[n] = 0;
      chi44_r[n] = 0;
      chi44_i[n] = 0;
        p50_r[n] = 0;
        p50_i[n] = 0;
      phi50_r[n] = 0;
      phi50_i[n] = 0;
      chi50_r[n] = 0;
      chi50_i[n] = 0;
        p51_r[n] = 0;
        p51_i[n] = 0;
      phi51_r[n] = 0;
      phi51_i[n] = 0;
      chi51_r[n] = 0;
      chi51_i[n] = 0;
        p52_r[n] = 0;
        p52_i[n] = 0;
      phi52_r[n] = 0;
      phi52_i[n] = 0;
      chi52_r[n] = 0;
      chi52_i[n] = 0;
        p53_r[n] = 0;
        p53_i[n] = 0;
      phi53_r[n] = 0;
      phi53_i[n] = 0;
      chi53_r[n] = 0;
      chi53_i[n] = 0;
        p54_r[n] = 0;
        p54_i[n] = 0;
      phi54_r[n] = 0;
      phi54_i[n] = 0;
      chi54_r[n] = 0;
      chi54_i[n] = 0;
        p55_r[n] = 0;
        p55_i[n] = 0;
      phi55_r[n] = 0;
      phi55_i[n] = 0;
      chi55_r[n] = 0;
      chi55_i[n] = 0;

      switch(parts[n].ncoeff) {
        case(21):
            p55_r[n] = parts[n].pnm_re[0];
            p55_i[n] = parts[n].pnm_im[0];
          phi55_r[n] = parts[n].phinm_re[0];
          phi55_i[n] = parts[n].phinm_im[0];
          chi55_r[n] = parts[n].chinm_re[0];
          chi55_i[n] = parts[n].chinm_im[0];
            p54_r[n] = parts[n].pnm_re[9];
            p54_i[n] = parts[n].pnm_im[9];
          phi54_r[n] = parts[n].phinm_re[9];
          phi54_i[n] = parts[n].phinm_im[9];
          chi54_r[n] = parts[n].chinm_re[9];
          chi54_i[n] = parts[n].chinm_im[9];
            p53_r[n] = parts[n].pnm_re[8];
            p53_i[n] = parts[n].pnm_im[8];
          phi53_r[n] = parts[n].phinm_re[8];
          phi53_i[n] = parts[n].phinm_im[8];
          chi53_r[n] = parts[n].chinm_re[8];
          chi53_i[n] = parts[n].chinm_im[8];
            p52_r[n] = parts[n].pnm_re[7];
            p52_i[n] = parts[n].pnm_im[7];
          phi52_r[n] = parts[n].phinm_re[7];
          phi52_i[n] = parts[n].phinm_im[7];
          chi52_r[n] = parts[n].chinm_re[7];
          chi52_i[n] = parts[n].chinm_im[7];
            p51_r[n] = parts[n].pnm_re[6];
            p51_i[n] = parts[n].pnm_im[6];
          phi51_r[n] = parts[n].phinm_re[6];
          phi51_i[n] = parts[n].phinm_im[6];
          chi51_r[n] = parts[n].chinm_re[6];
          chi51_i[n] = parts[n].chinm_im[6];
            p50_r[n] = parts[n].pnm_re[5];
            p50_i[n] = parts[n].pnm_im[5];
          phi50_r[n] = parts[n].phinm_re[5];
          phi50_i[n] = parts[n].phinm_im[5];
          chi50_r[n] = parts[n].chinm_re[5];
          chi50_i[n] = parts[n].chinm_im[5];
        case(15):
            p44_r[n] = parts[n].pnm_re[4];
            p44_i[n] = parts[n].pnm_im[4];
          phi44_r[n] = parts[n].phinm_re[4];
          phi44_i[n] = parts[n].phinm_im[4];
          chi44_r[n] = parts[n].chinm_re[4];
          chi44_i[n] = parts[n].chinm_im[4];
            p43_r[n] = parts[n].pnm_re[3];
            p43_i[n] = parts[n].pnm_im[3];
          phi43_r[n] = parts[n].phinm_re[3];
          phi43_i[n] = parts[n].phinm_im[3];
          chi43_r[n] = parts[n].chinm_re[3];
          chi43_i[n] = parts[n].chinm_im[3];
            p42_r[n] = parts[n].pnm_re[2];
            p42_i[n] = parts[n].pnm_im[2];
          phi42_r[n] = parts[n].phinm_re[2];
          phi42_i[n] = parts[n].phinm_im[2];
          chi42_r[n] = parts[n].chinm_re[2];
          chi42_i[n] = parts[n].chinm_im[2];
            p41_r[n] = parts[n].pnm_re[1];
            p41_i[n] = parts[n].pnm_im[1];
          phi41_r[n] = parts[n].phinm_re[1];
          phi41_i[n] = parts[n].phinm_im[1];
          chi41_r[n] = parts[n].chinm_re[1];
          chi41_i[n] = parts[n].chinm_im[1];
            p40_r[n] = parts[n].pnm_re[0];
            p40_i[n] = parts[n].pnm_im[0];
          phi40_r[n] = parts[n].phinm_re[0];
          phi40_i[n] = parts[n].phinm_im[0];
          chi40_r[n] = parts[n].chinm_re[0];
          chi40_i[n] = parts[n].chinm_im[0];
        case(10):
            p33_r[n] = parts[n].pnm_re[9];
            p33_i[n] = parts[n].pnm_im[9];
          phi33_r[n] = parts[n].phinm_re[9];
          phi33_i[n] = parts[n].phinm_im[9];
          chi33_r[n] = parts[n].chinm_re[9];
          chi33_i[n] = parts[n].chinm_im[9];
            p32_r[n] = parts[n].pnm_re[8];
            p32_i[n] = parts[n].pnm_im[8];
          phi32_r[n] = parts[n].phinm_re[8];
          phi32_i[n] = parts[n].phinm_im[8];
          chi32_r[n] = parts[n].chinm_re[8];
          chi32_i[n] = parts[n].chinm_im[8];
            p31_r[n] = parts[n].pnm_re[7];
            p31_i[n] = parts[n].pnm_im[7];
          phi31_r[n] = parts[n].phinm_re[7];
          phi31_i[n] = parts[n].phinm_im[7];
          chi31_r[n] = parts[n].chinm_re[7];
          chi31_i[n] = parts[n].chinm_im[7];
            p30_r[n] = parts[n].pnm_re[6];
            p30_i[n] = parts[n].pnm_im[6];
          phi30_r[n] = parts[n].phinm_re[6];
          phi30_i[n] = parts[n].phinm_im[6];
          chi30_r[n] = parts[n].chinm_re[6];
          chi30_i[n] = parts[n].chinm_im[6];
        case( 6):
            p22_r[n] = parts[n].pnm_re[5];
            p22_i[n] = parts[n].pnm_im[5];
          phi22_r[n] = parts[n].phinm_re[5];
          phi22_i[n] = parts[n].phinm_im[5];
          chi22_r[n] = parts[n].chinm_re[5];
          chi22_i[n] = parts[n].chinm_im[5];
            p21_r[n] = parts[n].pnm_re[4];
            p21_i[n] = parts[n].pnm_im[4];
          phi21_r[n] = parts[n].phinm_re[4];
          phi21_i[n] = parts[n].phinm_im[4];
          chi21_r[n] = parts[n].chinm_re[4];
          chi21_i[n] = parts[n].chinm_im[4];
            p20_r[n] = parts[n].pnm_re[3];
            p20_i[n] = parts[n].pnm_im[3];
          phi20_r[n] = parts[n].phinm_re[3];
          phi20_i[n] = parts[n].phinm_im[3];
          chi20_r[n] = parts[n].chinm_re[3];
          chi20_i[n] = parts[n].chinm_im[3];
        case( 3):
            p11_r[n] = parts[n].pnm_re[2];
            p11_i[n] = parts[n].pnm_im[2];
          phi11_r[n] = parts[n].phinm_re[2];
          phi11_i[n] = parts[n].phinm_im[2];
          chi11_r[n] = parts[n].chinm_re[2];
          chi11_i[n] = parts[n].chinm_im[2];
            p10_r[n] = parts[n].pnm_re[1];
            p10_i[n] = parts[n].pnm_im[1];
          phi10_r[n] = parts[n].phinm_re[1];
          phi10_i[n] = parts[n].phinm_im[1];
          chi10_r[n] = parts[n].chinm_re[1];
          chi10_i[n] = parts[n].chinm_im[1];
        case( 1):
            p00_r[n] = parts[n].pnm_re[0];
            p00_i[n] = parts[n].pnm_im[0];
          phi00_r[n] = parts[n].phinm_re[0];
          phi00_i[n] = parts[n].phinm_im[0];
          chi00_r[n] = parts[n].chinm_re[0];
          chi00_i[n] = parts[n].chinm_im[0];
       }
    }

    // create data nodes for coordinates
    if (cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateX", &cx))
      cgp_error_exit();
    if (cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateY", &cy))
      cgp_error_exit();
    if (cgp_coord_write(fn, bn, zn, RealDouble, "CoordinateZ", &cz))
      cgp_error_exit();

    // write the coordinate data in parallel
    if (cgp_coord_write_data(fn, bn, zn, cx, nstart, nend, x)) {
      printf("N%d >> nstart = %zd, nend = %zd\n", rank, nstart[0], nend[0]);
      printf("N%d >> Line %d\n", rank, __LINE__);
      WAIT();
      cgp_error_exit();
    }
    if (cgp_coord_write_data(fn, bn, zn, cy, nstart, nend, y))
      cgp_error_exit();
    if (cgp_coord_write_data(fn, bn, zn, cz, nstart, nend, z))
      cgp_error_exit();

    // create section for particle data
    if (cgp_section_write(fn, bn, zn, "Elements", NODE, 0, NPARTS-1, 0, &en)) {
      printf("N%d >> Line %d\n", rank, __LINE__);
      cgp_error_exit();
    }
    // for some reason, this is 0-indexed
    if (cgp_elements_write_data(fn, bn, zn, en, nstart[0] - 1, nend[0] - 1, conn))
      cgp_error_exit();

//ier = cgp_section_write(int fn, int B, int Z, char *ElementSectionName,
//       ElementType_t type, cgsize_t start, cgsize_t end, int nbndry, int *S);
//ier = cgp_elements_write_data(int fn, int B, int Z, int S, cgsize_t start,
//       cgsize_t end, cgsize_t *Elements);
// 
//ier = cg_section_write(int fn, int B, int Z, char *ElementSectionName,
//       ElementType_t type, cgsize_t start, cgsize_t end, int nbndry,
//       cgsize_t *Elements, int *S);

    // create solution
    if (cg_sol_write(fn, bn, zn, "Solution", Vertex, &sn))
      cgp_error_exit();

    // Create and write the field data
    if (cgp_field_write(fn, bn, zn, sn, RealDouble, "Radius", &fnr))
      cgp_error_exit();
    if (cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, a))
      cgp_error_exit();

    if (cgp_field_write(fn, bn, zn, sn, Integer, "GlobalIndex", &fnr))
      cgp_error_exit();
    if (cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, N))
      cgp_error_exit();

    // Velocity
    cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityX", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, u);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityY", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, v);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityZ", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, w);

    // Acceleration
    cgp_field_write(fn, bn, zn, sn, RealDouble, "AccelerationX", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, udot);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "AccelerationY", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, vdot);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "AccelerationZ", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, wdot);

    // Angular Velocity
    cgp_field_write(fn, bn, zn, sn, RealDouble, "AngularVelocityX", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, ox);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "AngularVelocityY", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, oy);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "AngularVelocityZ", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, oz);

    // Angular Velocity
    cgp_field_write(fn, bn, zn, sn, RealDouble, "AngularVelocityXDot", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, oxdot);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "AngularVelocityYDot", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, oydot);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "AngularVelocityZDot", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, ozdot);

    // Total force
    cgp_field_write(fn, bn, zn, sn, RealDouble, "TotalForceX", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, Fx);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "TotalForceY", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, Fy);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "TotalForceZ", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, Fz);

    // Interaction force
    cgp_field_write(fn, bn, zn, sn, RealDouble, "InteractionForceX", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, iFx);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "InteractionForceY", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, iFy);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "InteractionForceZ", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, iFz);

    // Hydro force
    cgp_field_write(fn, bn, zn, sn, RealDouble, "HydroForceX", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, hFx);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "HydroForceY", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, hFy);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "HydroForceZ", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, hFz);

    // Spring force
    cgp_field_write(fn, bn, zn, sn, RealDouble, "SpringForceX", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, kFx);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "SpringForceY", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, kFy);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "SpringForceZ", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, kFz);

    // Moment
    cgp_field_write(fn, bn, zn, sn, RealDouble, "MomentX", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, Lx);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "MomentY", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, Ly);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "MomentZ", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, Lz);


    cgp_field_write(fn, bn, zn, sn, RealDouble, "Density", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, density);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "YoungsModulus", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, E);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "PoissonRatio", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, sigma);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "DryCoeffRest", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, e_dry);

    cgp_field_write(fn, bn, zn, sn, RealDouble, "FricCoeff", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, coeff_fric);

    cgp_field_write(fn, bn, zn, sn, Integer, "LambOrder", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, order);

    cgp_field_write(fn, bn, zn, sn, Integer, "NParticleCollisions", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, ncoll_part);

    cgp_field_write(fn, bn, zn, sn, Integer, "NWallCollisions", &fnr);
    cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, ncoll_wall);

    // Lamb coeffs
    switch(ncoeffs_max) {
      case(21):
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p55_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p55_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p55_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p55_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi55_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi55_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi55_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi55_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi55_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi55_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi55_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi55_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p54_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p54_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p54_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p54_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi54_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi54_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi54_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi54_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi54_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi54_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi54_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi54_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p53_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p53_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p53_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p53_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi53_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi53_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi53_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi53_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi53_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi53_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi53_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi53_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p52_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p52_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p52_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p52_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi52_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi52_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi52_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi52_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi52_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi52_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi52_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi52_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p51_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p51_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p51_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p51_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi51_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi51_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi51_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi51_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi51_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi51_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi51_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi51_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p50_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p50_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p50_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p50_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi50_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi50_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi50_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi50_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi50_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi50_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi50_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi50_i);
      case(15):
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p44_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p44_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p44_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p44_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi44_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi44_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi44_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi44_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi44_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi44_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi44_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi44_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p43_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p43_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p43_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p43_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi43_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi43_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi43_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi43_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi43_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi43_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi43_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi43_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p42_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p42_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p42_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p42_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi42_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi42_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi42_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi42_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi42_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi42_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi42_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi42_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p41_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p41_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p41_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p41_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi41_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi41_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi41_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi41_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi41_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi41_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi41_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi41_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p40_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p40_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p40_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p40_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi40_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi40_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi40_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi40_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi40_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi40_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi40_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi40_i);
      case(10):
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p33_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p33_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p33_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p33_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi33_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi33_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi33_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi33_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi33_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi33_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi33_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi33_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p32_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p32_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p32_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p32_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi32_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi32_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi32_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi32_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi32_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi32_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi32_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi32_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p31_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p31_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p31_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p31_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi31_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi31_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi31_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi31_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi31_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi31_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi31_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi31_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p30_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p30_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p30_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p30_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi30_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi30_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi30_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi30_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi30_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi30_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi30_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi30_i);
      case( 6):
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p22_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p22_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p22_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p22_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi22_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi22_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi22_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi22_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi22_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi22_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi22_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi22_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p21_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p21_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p21_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p21_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi21_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi21_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi21_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi21_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi21_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi21_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi21_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi21_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p20_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p20_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p20_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p20_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi20_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi20_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi20_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi20_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi20_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi20_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi20_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi20_i);
      case( 3):
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p11_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p11_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p11_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p11_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi11_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi11_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi11_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi11_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi11_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi11_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi11_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi11_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p10_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p10_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p10_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p10_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi10_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi10_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi10_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi10_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi10_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi10_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi10_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi10_i);
      case( 1):
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p00_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p00_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble,   "p00_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend,   p00_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi00_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi00_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "phi00_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, phi00_i);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi00_re", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi00_r);
        cgp_field_write(fn, bn, zn, sn, RealDouble, "chi00_im", &fnr);
        cgp_field_write_data(fn, bn, zn, sn, fnr, nstart, nend, chi00_i);
    }

    // Miscellaneous data - scalars, time
    cgsize_t nele[1];
    nele[0] = 1;

    if (cg_user_data_write("Etc"))
      cgp_error_exit();
    if (cg_goto(fn, bn, "Zone_t", zn, "Etc", 0, "end"))
      cgp_error_exit();

    if (cgp_array_write("Time", RealDouble, 1, nele, &fnr))
      cgp_error_exit();
    if (cgp_array_write_data(fnr, nele, nele, &ttime))
      cgp_error_exit();

    // close and free
    if (cgp_close(fn))
      cgp_error_exit();

    free(x);
    free(y);
    free(z);

    free(u);
    free(v);
    free(w);
    free(udot);
    free(vdot);
    free(wdot);

    free(ox);
    free(oy);
    free(oz);
    free(oxdot);
    free(oydot);
    free(ozdot);

    free(Fx);
    free(Fy);
    free(Fz);
    free(iFx);
    free(iFy);
    free(iFz);
    free(hFx);
    free(hFy);
    free(hFz);
    free(kFx);
    free(kFy);
    free(kFz);

    free(Lx);
    free(Ly);
    free(Lz);

    free(axx);
    free(axy);
    free(axz);
    free(ayx);
    free(ayy);
    free(ayz);
    free(azx);
    free(azy);
    free(azz);

    free(a);
    free(N);
    free(conn);

    free(density);
    free(E);
    free(sigma);
    free(coeff_fric);
    free(e_dry);
    free(order);

    free(ncoll_part);
    free(ncoll_wall);

    free(  p00_r);
    free(  p00_i);
    free(phi00_r);
    free(phi00_i);
    free(chi00_r);
    free(chi00_i);
    free(  p10_r);
    free(  p10_i);
    free(phi10_r);
    free(phi10_i);
    free(chi10_r);
    free(chi10_i);
    free(  p11_r);
    free(  p11_i);
    free(phi11_r);
    free(phi11_i);
    free(chi11_r);
    free(chi11_i);
    free(  p20_r);
    free(  p20_i);
    free(phi20_r);
    free(phi20_i);
    free(chi20_r);
    free(chi20_i);
    free(  p21_r);
    free(  p21_i);
    free(phi21_r);
    free(phi21_i);
    free(chi21_r);
    free(chi21_i);
    free(  p22_r);
    free(  p22_i);
    free(phi22_r);
    free(phi22_i);
    free(chi22_r);
    free(chi22_i);
    free(  p30_r);
    free(  p30_i);
    free(phi30_r);
    free(phi30_i);
    free(chi30_r);
    free(chi30_i);
    free(  p31_r);
    free(  p31_i);
    free(phi31_r);
    free(phi31_i);
    free(chi31_r);
    free(chi31_i);
    free(  p32_r);
    free(  p32_i);
    free(phi32_r);
    free(phi32_i);
    free(chi32_r);
    free(chi32_i);
    free(  p33_r);
    free(  p33_i);
    free(phi33_r);
    free(phi33_i);
    free(chi33_r);
    free(chi33_i);
    free(  p40_r);
    free(  p40_i);
    free(phi40_r);
    free(phi40_i);
    free(chi40_r);
    free(chi40_i);
    free(  p41_r);
    free(  p41_i);
    free(phi41_r);
    free(phi41_i);
    free(chi41_r);
    free(chi41_i);
    free(  p42_r);
    free(  p42_i);
    free(phi42_r);
    free(phi42_i);
    free(chi42_r);
    free(chi42_i);
    free(  p43_r);
    free(  p43_i);
    free(phi43_r);
    free(phi43_i);
    free(chi43_r);
    free(chi43_i);
    free(  p44_r);
    free(  p44_i);
    free(phi44_r);
    free(phi44_i);
    free(chi44_r);
    free(chi44_i);
    free(  p50_r);
    free(  p50_i);
    free(phi50_r);
    free(phi50_i);
    free(chi50_r);
    free(chi50_i);
    free(  p51_r);
    free(  p51_i);
    free(phi51_r);
    free(phi51_i);
    free(chi51_r);
    free(chi51_i);
    free(  p52_r);
    free(  p52_i);
    free(phi52_r);
    free(phi52_i);
    free(chi52_r);
    free(chi52_i);
    free(  p53_r);
    free(  p53_i);
    free(phi53_r);
    free(phi53_i);
    free(chi53_r);
    free(chi53_i);
    free(  p54_r);
    free(  p54_i);
    free(phi54_r);
    free(phi54_i);
    free(chi54_r);
    free(chi54_i);
    free(  p55_r);
    free(  p55_i);
    free(phi55_r);
    free(phi55_i);
    free(chi55_r);
    free(chi55_i);
  }

  // Only free communicator on proc's that have it -- procs not in the
  // communicator are returned MPI_COMM_NULL; freeing a null is disallowed in 1.1
  if (nparts_subdom > 0) {
    MPI_Comm_free(&part_comm);
  }
}

void cgns_flow_field_ghost(real dtout)
{
  // create the solution file names
  char fname[FILE_NAME_SIZE];
  int sigfigs = ceil(log10(1. / dtout));
  if(sigfigs < 1) sigfigs = 1;
  sprintf(fname, "%s/%s/flow-ghost-%.*f.cgns", ROOT_DIR, OUTPUT_DIR, sigfigs, ttime);

  char gname[FILE_NAME_SIZE];
  sprintf(gname, "grid-ghost.cgns");

  // cgns file indices
  int fn;
  int bn;
  int zn;
  int sn;
  int fnp;
  int fnphi;
  int fnu;
  int fnv;
  int fnw;
  int fnu_star;
  int fnv_star;
  int fnw_star;
  int fnu_conv;
  int fnv_conv;
  int fnw_conv;
  int fnu_diff;
  int fnv_diff;
  int fnw_diff;

  cgp_mpi_comm(MPI_COMM_WORLD);
  // open file in parallel
  if (cgp_open(fname, CG_MODE_WRITE, &fn))
    cgp_error_exit();

  // write base, zone
  int cell_dim = 3;   // volume cells
  int phys_dim = 3;   // # coords to define a vector
  if (cg_base_write(fn, "Base", cell_dim, phys_dim, &bn))
    cgp_error_exit();

  cgsize_t size[9];     // number of vertices and cells
  size[0] = DOM.Gcc.inb + 1; // vertices
  size[1] = DOM.Gcc.jnb + 1;
  size[2] = DOM.Gcc.knb + 1;
  size[3] = DOM.Gcc.inb;     // cells
  size[4] = DOM.Gcc.jnb;
  size[5] = DOM.Gcc.knb;
  size[6] = 0;
  size[7] = 0;
  size[8] = 0;
  if (cg_zone_write(fn, bn, "Zone0", size, Structured, &zn))
    cgp_error_exit();
  if (cg_goto(fn, bn, "Zone_t", zn, "end"))
    cgp_error_exit();

  // check that grid.cgns exists
  /*int fng;
  if(cg_open(gnameall, CG_MODE_READ, &fng) != 0) {
    fprintf(stderr, "CGNS flow field write failure: no grid.cgns\n");
    exit(EXIT_FAILURE);
  } else {
    cg_close(fng);
  }
    cg_close(fng);
*/
  
  // link grid.cgns
  if (cg_link_write("GridCoordinates", gname, "Base/Zone0/GridCoordinates"))
    cgp_error_exit();

  // create solution
  if (cg_sol_write(fn, bn, zn, "Solution", CellCenter, &sn))
    cgp_error_exit();

  // start and end index for write (+1 for 1 indexing)
  cgsize_t nstart[3];
  nstart[0] = dom[rank].Gcc.isb + 1;
  nstart[1] = dom[rank].Gcc.jsb + 1;
  nstart[2] = dom[rank].Gcc.ksb + 1;

  cgsize_t nend[3];
  nend[0] = dom[rank].Gcc.ieb + 1;
  nend[1] = dom[rank].Gcc.jeb + 1;
  nend[2] = dom[rank].Gcc.keb + 1;

  // create and write the field data -- pressure
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "Pressure", &fnp))
    cgp_error_exit();

  if (cgp_field_write_data(fn, bn, zn, sn, fnp, nstart, nend, p))
    cgp_error_exit();

  // phi
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "Phi", &fnphi))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnphi, nstart, nend, phi))
    cgp_error_exit();

  /* We can't use four centered points to interpolate the velocities in the
   * ghost cells, so use two points.
   * WARNING: This is lower accuracy!!
   */

  real *uout = malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *vout = malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *wout = malloc(dom[rank].Gcc.s3b * sizeof(real));
  // create and write the velocity fields
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityX", &fnu))
    cgp_error_exit();
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityY", &fnv))
    cgp_error_exit();
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "VelocityZ", &fnw))
    cgp_error_exit();

  for (int k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (int j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (int i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        int C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        int Cfx = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        int Cfx_e = GFX_LOC(i + 1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        int Cfy = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        int Cfy_n = GFY_LOC(i, j + 1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        int Cfz = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        int Cfz_t = GFZ_LOC(i, j, k + 1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);


        uout[C] = 0.5 * (u[Cfx] + u[Cfx_e]);
        vout[C] = 0.5 * (v[Cfy] + v[Cfy_n]);
        wout[C] = 0.5 * (w[Cfz] + w[Cfz_t]);
      }
    }
  }

  if (cgp_field_write_data(fn, bn, zn, sn, fnu, nstart, nend, uout))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnv, nstart, nend, vout))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnw, nstart, nend, wout))
    cgp_error_exit();

  // create and write the velocity star fields
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "UStar", &fnu_star))
    cgp_error_exit();
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "VStar", &fnv_star))
    cgp_error_exit();
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "WStar", &fnw_star))
    cgp_error_exit();

  for (int k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (int j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (int i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        int C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        int Cfx = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        int Cfx_e = GFX_LOC(i + 1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        int Cfy = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        int Cfy_n = GFY_LOC(i, j + 1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        int Cfz = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        int Cfz_t = GFZ_LOC(i, j, k + 1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        uout[C] = 0.5 * (u_star[Cfx] + u_star[Cfx_e]);
        vout[C] = 0.5 * (v_star[Cfy] + v_star[Cfy_n]);
        wout[C] = 0.5 * (w_star[Cfz] + w_star[Cfz_t]);
      }
    }
  }

  if (cgp_field_write_data(fn, bn, zn, sn, fnu_star, nstart, nend, uout))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnv_star, nstart, nend, vout))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnw_star, nstart, nend, wout))
    cgp_error_exit();

  // create and write the convective term fields
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "UConv", &fnu_conv))
    cgp_error_exit();
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "VConv", &fnv_conv))
    cgp_error_exit();
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "WConv", &fnw_conv))
    cgp_error_exit();

  for (int k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (int j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (int i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        int C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        int Cfx = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        int Cfx_e = GFX_LOC(i + 1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        int Cfy = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        int Cfy_n = GFY_LOC(i, j + 1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        int Cfz = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        int Cfz_t = GFZ_LOC(i, j, k + 1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        uout[C] = 0.5 * (conv_u[Cfx] + conv_u[Cfx_e]);
        vout[C] = 0.5 * (conv_v[Cfy] + conv_v[Cfy_n]);
        wout[C] = 0.5 * (conv_w[Cfz] + conv_w[Cfz_t]);
      }
    }
  }

  if (cgp_field_write_data(fn, bn, zn, sn, fnu_conv, nstart, nend, uout))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnv_conv, nstart, nend, vout))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnw_conv, nstart, nend, wout))
    cgp_error_exit();

  // create and write the diffusive term fields
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "UDiff", &fnu_diff))
    cgp_error_exit();
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "VDiff", &fnv_diff))
    cgp_error_exit();
  if (cgp_field_write(fn, bn, zn, sn, RealDouble, "WDiff", &fnw_diff))
    cgp_error_exit();

  for (int k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (int j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (int i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        int C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        int Cfx = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        int Cfx_e = GFX_LOC(i + 1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);

        int Cfy = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        int Cfy_n = GFY_LOC(i, j + 1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);

        int Cfz = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        int Cfz_t = GFZ_LOC(i, j, k + 1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        uout[C] = 0.5 * (diff_u[Cfx] + diff_u[Cfx_e]);
        vout[C] = 0.5 * (diff_v[Cfy] + diff_v[Cfy_n]);
        wout[C] = 0.5 * (diff_w[Cfz] + diff_w[Cfz_t]);
      }
    }
  }

  if (cgp_field_write_data(fn, bn, zn, sn, fnu_diff, nstart, nend, uout))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnv_diff, nstart, nend, vout))
    cgp_error_exit();
  if (cgp_field_write_data(fn, bn, zn, sn, fnw_diff, nstart, nend, wout))
    cgp_error_exit();

  free(uout);
  free(vout);
  free(wout);

  // phase, phase_shell
  cgp_field_write(fn, bn, zn, sn, Integer, "Phase", &fnp);
  cgp_field_write_data(fn, bn, zn, sn, fnp, nstart, nend, phase);
  cgp_field_write(fn, bn, zn, sn, Integer, "PhaseShell", &fnp);
  cgp_field_write_data(fn, bn, zn, sn, fnp, nstart, nend, phase);

  // Misc data -- scalars
  cg_user_data_write("Etc");
  cg_goto(fn, bn, "Zone_t", zn, "Etc", 0, "end");

  int fnt, fnrho, fnnu;
  cgsize_t nele = 1;

  cgsize_t N[1];
  N[0] = 1;
  
  cgp_array_write("Time", RealDouble, 1, &nele, &fnt);
  cgp_array_write("Density", RealDouble, 1, &nele, &fnrho);
  cgp_array_write("KinematicViscosity", RealDouble, 1, &nele, &fnnu);

  cgp_array_write_data(fnt, N, N, &ttime);
  cgp_array_write_data(fnrho, N, N, &rho_f);
  cgp_array_write_data(fnnu, N, N, &nu);

  if (cgp_close(fn))
    cgp_error_exit();
}


