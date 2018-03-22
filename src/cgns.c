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


