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
#include "bluebottle.h"

void vtk_recorder_init(void)
{
  if (rec_vtk_dt > 0) {
    #ifdef DDEBUG
        init_VTK_ghost();
    #else
        init_VTK();
    #endif // DDEBUG
  }
}

void vtk_recorder_write(void)
{
  if (rec_vtk_dt > 0) {
    if (rec_vtk_ttime_out >= rec_vtk_dt || stepnum == 0) {

      // Write
      #ifdef DDEBUG
        out_VTK_ghost();
      #else
        out_VTK();
      #endif // DDEBUG

      // Notify output
      printf("N%d >> Writing ParaView output file %d at t = %e... done.\n", 
        rank, rec_vtk_stepnum_out, ttime);
      rec_vtk_stepnum_out++;

      if (stepnum != 0) rec_vtk_ttime_out -= rec_vtk_dt;
    }
  }
}

void init_VTK(void)
{
  if (rank == 0) {
    // Create output directory if it doesn't exist
    // From stackoverflowc.com/questions/7430248
    struct stat st = {0};
    char buf[CHAR_BUF_SIZE];
    sprintf(buf, "%s/%s", ROOT_DIR, OUTPUT_DIR);
    if (stat(buf, &st) == -1) {
      mkdir(buf, 0700);
    }

    // open PVD file for writing
    char fname[FILE_NAME_SIZE];
    sprintf(fname, "%s/%s/out.pvd", ROOT_DIR, OUTPUT_DIR);
    FILE *outfile = fopen(fname, "w");
    if (outfile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }

    // write PVD file header and footer
    fprintf(outfile, "<VTKFile type=\"Collection\">\n");
    fprintf(outfile, "<Collection>\n");
    fprintf(outfile, "</Collection>\n");
    fprintf(outfile, "</VTKFile>");

    // close the file
    fclose(outfile);
  }
}

void out_VTK(void)
{
  if (rank == 0) {
    char fname_pvd[FILE_NAME_SIZE] = "";  // pvd filename
    char fname_pvtr[FILE_NAME_SIZE] = ""; // pvtr filename
    char fname_pvtp[FILE_NAME_SIZE] = ""; // pvtp filename
    //char fnamenodes_vtp[FILE_NAME_SIZE] = ""; // vtp filename

    sprintf(fname_pvd, "%s/%s/out.pvd", ROOT_DIR, OUTPUT_DIR);
    sprintf(fname_pvtr, "out_%d.pvtr", rec_vtk_stepnum_out);
    sprintf(fname_pvtp, "out_%d.pvtp", rec_vtk_stepnum_out);
    //sprintf(fnamenodes_vtp, "out_nodes_%d.vtp", rec_vtk_stepnum_out);

    FILE *pvdfile= fopen(fname_pvd, "r+");
    if (pvdfile == NULL) {
      init_VTK();
      pvdfile = fopen(fname_pvd, "r+");
    }
    // moves back 2 lines from the end of the file (above the footer)
    fseek(pvdfile, -24, SEEK_END);

    fprintf(pvdfile, "<DataSet timestep=\"%e\" part=\"0\" file=\"%s\"/>\n",
      ttime, fname_pvtr);
    fprintf(pvdfile, "<DataSet timestep=\"%e\" part=\"1\" file=\"%s\"/>\n",
      ttime, fname_pvtp);
    //fprintf(pvdfile, "<DataSet timestep=\"%e\" part=\"2\" file=\"%s\"/>\n",
    //  ttime, fnamenodes_vtp);
    fprintf(pvdfile, "</Collection>\n");
    fprintf(pvdfile, "</VTKFile>");
    fclose(pvdfile);
  }

  dom_out_VTK();
  part_out_VTK();
  //quadnodes_out_VTK();
}

void dom_out_VTK(void)
{
  int i, j, k, l;                      // iterators
  char fname[FILE_NAME_SIZE] = "";     // output filename
  char fname_dom[FILE_NAME_SIZE] = ""; // subdomain filename
  int C;                               // cell center index
  int Cx, Cy, Cz;                      // cell center index for interpolation
  int Cx_e, Cy_n, Cz_t;                // adjacent cell center index

  // only work on pvtr file once
  if (rank == 0) {
    sprintf(fname, "%s/%s/out_%d.pvtr", ROOT_DIR, OUTPUT_DIR, rec_vtk_stepnum_out);
    FILE *outfile = fopen(fname, "w");
    if (outfile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }

    // write Paraview pvtr file
    fprintf(outfile, "<VTKFile type=\"PRectilinearGrid\">\n");
    fprintf(outfile, "<PRectilinearGrid WholeExtent=");
    fprintf(outfile, "\"0 %d 0 %d 0 %d\" GhostLevel=\"0\">\n",
      DOM.xn, DOM.yn, DOM.zn);
    fprintf(outfile, "<PCellData Scalars=\"p phase phase_shell\" Vectors=\"vel\">\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"p\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Int32\" Name=\"phase\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Int32\" Name=\"phase_shell\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"vel\"");
    fprintf(outfile, " NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "</PCellData>\n");
    fprintf(outfile, "<PCoordinates>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"x\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"y\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"z\"/>\n");
    fprintf(outfile, "</PCoordinates>\n");
    for (l = 0; l < nprocs; l++) {
      fprintf(outfile, "<Piece Extent=\"");
      fprintf(outfile, "%d %d ", dom[l].Gcc.is-DOM_BUF, dom[l].Gcc.ie);
      fprintf(outfile, "%d %d ", dom[l].Gcc.js-DOM_BUF, dom[l].Gcc.je);
      fprintf(outfile, "%d %d\" ", dom[l].Gcc.ks-DOM_BUF, dom[l].Gcc.ke);
      sprintf(fname_dom, "out_%d_%d_of_%d.vtr", rec_vtk_stepnum_out, l, nprocs);
      fprintf(outfile, "Source=\"%s\"/>\n", fname_dom);
    }
    fprintf(outfile, "</PRectilinearGrid>\n");
    fprintf(outfile, "</VTKFile>\n");
    fclose(outfile);
  }

  // Interpolate velocities to cell centers (in working arrays)
  real *uu = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *vv = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *ww = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        Cx = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        Cx_e = GFX_LOC(i + 1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        Cy = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        Cy_n = GFY_LOC(i, j + 1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        Cz = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        Cz_t = GFZ_LOC(i, j, k + 1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        // interpolate velocity
        uu[C] = 0.5*(u[Cx] + u[Cx_e]);
        vv[C] = 0.5*(v[Cy] + v[Cy_n]);
        ww[C] = 0.5*(w[Cz] + w[Cz_t]);
      }
    }
  }

  // write subdomain file -- open file for writing
  sprintf(fname, "%s/%s/out_%d_%d_of_%d.vtr", ROOT_DIR, OUTPUT_DIR,
    rec_vtk_stepnum_out, rank, nprocs);
  FILE *outfile = fopen(fname, "w");
  if (outfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  fprintf(outfile, "<VTKFile type=\"RectilinearGrid\">\n");
  fprintf(outfile, "<RectilinearGrid WholeExtent=");
  fprintf(outfile, "\"%d %d ", dom[rank].Gcc.is-DOM_BUF, dom[rank].Gcc.ie);
  fprintf(outfile, "%d %d ", dom[rank].Gcc.js-DOM_BUF, dom[rank].Gcc.je);
  fprintf(outfile, "%d %d\" GhostLevel=\"0\">\n", dom[rank].Gcc.ks-DOM_BUF,
    dom[rank].Gcc.ke);

  fprintf(outfile, "<Piece Extent=\"");
  fprintf(outfile, "%d %d ", dom[rank].Gcc.is-DOM_BUF,
    dom[rank].Gcc.ie);
  fprintf(outfile, "%d %d ", dom[rank].Gcc.js-DOM_BUF,
    dom[rank].Gcc.je);
  fprintf(outfile, "%d %d\">\n", dom[rank].Gcc.ks-DOM_BUF,
    dom[rank].Gcc.ke);
  fprintf(outfile, "<CellData Scalars=\"p phase phase_shell\" Vectors=\"vel\">\n");

  // write pressure
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"p\">\n");
  for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%lf ", p[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write phase for this subdomain
  fprintf(outfile, "<DataArray type=\"Int32\" Name=\"phase\">\n");
  for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

        int tmp_phase = phase[C]*(phase[C] < 0);
        fprintf(outfile, "%d ", tmp_phase);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write phase for this subdomain
  fprintf(outfile, "<DataArray type=\"Int32\" Name=\"phase_shell\">\n");
  for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%d ", phase_shell[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write velocity vector
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"vel\"");
  fprintf(outfile, " NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (k = dom[rank].Gcc._ks; k <= dom[rank].Gcc._ke; k++) {
    for (j = dom[rank].Gcc._js; j <= dom[rank].Gcc._je; j++) {
      for (i = dom[rank].Gcc._is; i <= dom[rank].Gcc._ie; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%lf %lf %lf ", uu[C], vv[C], ww[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  fprintf(outfile, "</CellData>\n");

  fprintf(outfile, "<Coordinates>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"x\">\n");
  for (i = dom[rank].Gcc._is-DOM_BUF; i <= dom[rank].Gcc._ie; i++) {
    fprintf(outfile, "%lf ", i * dom[rank].dx + dom[rank].xs);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"y\">\n");
  for (j = dom[rank].Gcc._js-DOM_BUF; j <= dom[rank].Gcc._je; j++) {
    fprintf(outfile, "%lf ", j * dom[rank].dy + dom[rank].ys);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"z\">\n");
  for (k = dom[rank].Gcc._ks-DOM_BUF; k <= dom[rank].Gcc._ke; k++) {
    fprintf(outfile, "%lf ", k * dom[rank].dz + dom[rank].zs);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");
  fprintf(outfile, "</Coordinates>\n");
  fprintf(outfile, "</Piece>\n");
  fprintf(outfile, "</RectilinearGrid>\n");
  fprintf(outfile, "</VTKFile>\n");
  fclose(outfile);

  // clean up interpolated fields
  free(uu);
  free(vv);
  free(ww);
}

void part_out_VTK(void)
{
  int i;                           // iterator
  char fname[FILE_NAME_SIZE] = ""; // output filename

  // only work on pvtp file once
  if (rank == 0) {
    sprintf(fname, "%s/%s/out_%d.pvtp", ROOT_DIR, OUTPUT_DIR,
      rec_vtk_stepnum_out);
    FILE *outfile = fopen(fname, "w");
    if (outfile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }

    // write paraview vtp file
    fprintf(outfile, "<VTKFile type=\"PPolyData\">\n");
    fprintf(outfile, "<PPolyData GhostLevel=\"0\">\n");
    fprintf(outfile, "<PPointData Scalars=\"n r\" Vectors=\"pvel\">\n");
    fprintf(outfile, "<PDataArray type=\"Int32\" Name=\"n\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"r\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"pvel\" ");
    fprintf(outfile, "NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "</PPointData>\n");
    fprintf(outfile, "<PPoints>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "</PPoints>\n");
    for (int n = 0; n < nprocs; n++) {
      fprintf(outfile, "<Piece Source=\"out_%d_%d_of_%d.vtp\"/>\n", 
        rec_vtk_stepnum_out, n, nprocs);
    }
    fprintf(outfile, "</PPolyData>\n");
    fprintf(outfile, "</VTKFile>\n");

    fclose(outfile);
  } // if (rank == 0)

  // Open vtp files for writing
  sprintf(fname, "%s/%s/out_%d_%d_of_%d.vtp", ROOT_DIR, OUTPUT_DIR,
    rec_vtk_stepnum_out, rank, nprocs);
  FILE *outfile = fopen(fname, "w");
  if (outfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }
  
  // Set up file structure
  fprintf(outfile, "<VTKFile type=\"PolyData\">\n");
  fprintf(outfile, "<PolyData>\n");
  fprintf(outfile, "<Piece NumberOfPoints=\"%d\" NumberOfVerts=\"0\" ", nparts_subdom);
  fprintf(outfile, "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n");

  // Output particle positions
  fprintf(outfile, "<Points>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  // write the locations of the particle centers
  for (i = 0; i < nparts_subdom; i++) {
    fprintf(outfile, "%e %e %e ", parts[i].x, parts[i].y, parts[i].z);
  }
  fprintf(outfile, "\n</DataArray>\n");
  fprintf(outfile, "</Points>\n");

  // Outpart particle data
  fprintf(outfile, "<PointData Scalars=\"n r\" Vectors=\"pvel\">\n");

  // part number
  fprintf(outfile, "<DataArray type=\"Int32\" Name=\"n\" format=\"ascii\">\n");
  for(i = 0; i < nparts_subdom; i++) {
    fprintf(outfile, "%d ", parts[i].N);
  }
  fprintf(outfile, "\n</DataArray>\n");

  // part radius
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"r\" format=\"ascii\">\n");
  for (i = 0; i < nparts_subdom; i++) {
    fprintf(outfile, "%e ", parts[i].r);
  }
  fprintf(outfile, "\n</DataArray>\n");

  // part velocity
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"pvel\" format=\"ascii\" ");
  fprintf(outfile, "NumberOfComponents=\"3\">\n");
  for (i = 0; i < nparts_subdom; i++) {
    fprintf(outfile, "%e %e %e ", parts[i].u, parts[i].v, parts[i].w);
  }
  fprintf(outfile, "\n</DataArray>\n");

  // finish file
  fprintf(outfile, "</PointData>\n");
  fprintf(outfile, "</Piece>\n");
  fprintf(outfile, "</PolyData>\n");
  fprintf(outfile, "</VTKFile>\n");
  fclose(outfile);
}

void init_VTK_ghost(void)
{
  if (rank == 0) {
    // Create output directory if it doesn't exist
    // From stackoverflowc.com/questions/7430248
    struct stat st = {0};
    char buf[CHAR_BUF_SIZE];
    sprintf(buf, "%s/%s", ROOT_DIR, OUTPUT_DIR);
    if (stat(buf, &st) == -1) {
      mkdir(buf, 0700);
    }

    // open PVD file for writing
    char fname[FILE_NAME_SIZE];
    sprintf(fname, "%s/%s/out_ghost.pvd", ROOT_DIR, OUTPUT_DIR);
    FILE *outfile = fopen(fname, "w");
    if (outfile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }

    // write PVD file header and footer
    fprintf(outfile, "<VTKFile type=\"Collection\">\n");
    fprintf(outfile, "<Collection>\n");
    fprintf(outfile, "</Collection>\n");
    fprintf(outfile, "</VTKFile>");

    // close the file
    fclose(outfile);
  }
}

void out_VTK_ghost(void)
{
  if (rank == 0) {
    char fname_pvd[FILE_NAME_SIZE] = "";  // pvd filename
    char fname_pvtr[FILE_NAME_SIZE] = ""; // pvtr filename
    char fname_pvtp[FILE_NAME_SIZE] = ""; // pvtp filename
    //char fnamenodes_vtp[FILE_NAME_SIZE] = ""; // vtp filename

    sprintf(fname_pvd, "%s/%s/out_ghost.pvd", ROOT_DIR, OUTPUT_DIR);
    sprintf(fname_pvtr, "out_ghost_%d.pvtr", rec_vtk_stepnum_out);
    sprintf(fname_pvtp, "out_ghost_%d.pvtp", rec_vtk_stepnum_out);
    //sprintf(fnamenodes_vtp, "out_nodes_%d.vtp", rec_vtk_stepnum_out);

    FILE *pvdfile= fopen(fname_pvd, "r+");
    if (pvdfile == NULL) {
      init_VTK_ghost();
      pvdfile= fopen(fname_pvd, "r+");
    }
    // moves back 2 lines from the end of the file (above the footer)
    fseek(pvdfile, -24, SEEK_END);

    fprintf(pvdfile, "<DataSet timestep=\"%e\" part=\"0\" file=\"%s\"/>\n",
      ttime, fname_pvtr);
    fprintf(pvdfile, "<DataSet timestep=\"%e\" part=\"1\" file=\"%s\"/>\n",
      ttime, fname_pvtp);
    //fprintf(pvdfile, "<DataSet timestep=\"%e\" part=\"2\" file=\"%s\"/>\n",
    //  ttime, fnamenodes_vtp);
    fprintf(pvdfile, "</Collection>\n");
    fprintf(pvdfile, "</VTKFile>");
    fclose(pvdfile);
  }

  dom_out_VTK_ghost();
  part_out_VTK_ghost();
  //part_out_VTK();
}

void dom_out_VTK_ghost(void)
{
  int i, j, k, l;                      // iterators
  char fname[FILE_NAME_SIZE] = "";     // output filename
  char fname_dom[FILE_NAME_SIZE] = ""; // subdomain filename
  int C;                               // cell center index
  int Cx, Cy, Cz;                      // cell center index for interpolation
  int Cx_e, Cy_n, Cz_t;                // adjacent cell center index

  // only work on pvtr file once
  if (rank == 0) {
    sprintf(fname, "%s/%s/out_ghost_%d.pvtr", ROOT_DIR, OUTPUT_DIR, rec_vtk_stepnum_out);
    FILE *outfile = fopen(fname, "w");
    if(outfile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }

    // write Paraview pvtr file
    //  -- WholeExtent and Piece Extent use node/point centered, not cell
    //      centered
    //  -- So, a domain with 4 cells [0, 1, 2, 3] will look like
    //      [0 - 4], with those values located on the faces/nodes/points

    fprintf(outfile, "<VTKFile type=\"PRectilinearGrid\">\n");
    fprintf(outfile, "<PRectilinearGrid WholeExtent=");
    fprintf(outfile, "\"0 %d 0 %d 0 %d\" GhostLevel=\"1\">\n",
      DOM.xn + 2*DOM_BUF, DOM.yn + 2*DOM_BUF, DOM.zn + 2*DOM_BUF);
    //fprintf(outfile, "<PCellData Scalars=\"p phi phase phase_shell\" Vectors=\"vel vel_star flag\">\n");
    fprintf(outfile, "<PCellData Scalars=\"p phi phase phase_shell\" Vectors=");
    fprintf(outfile, "\"vel vel_star conv_vel diff_vel flag\">\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"p\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"phi\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Int32\" Name=\"phase\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Int32\" Name=\"phase_shell\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"vel\"");
    fprintf(outfile, " NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"vel_star\"");
    fprintf(outfile, " NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"conv_vel\"");
    fprintf(outfile, " NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"diff_vel\"");
    fprintf(outfile, " NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"flag\"");
    fprintf(outfile, " NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "</PCellData>\n");
    fprintf(outfile, "<PCoordinates>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"x\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"y\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"z\"/>\n");
    fprintf(outfile, "</PCoordinates>\n");
    for (l = 0; l < nprocs; l++) {
      fprintf(outfile, "<Piece Extent=\""); // add 1 to eb to get node centrd
      fprintf(outfile, "%d %d ", dom[l].Gcc.isb, dom[l].Gcc.ieb + 1);
      fprintf(outfile, "%d %d ", dom[l].Gcc.jsb, dom[l].Gcc.jeb + 1);
      fprintf(outfile, "%d %d\" ", dom[l].Gcc.ksb, dom[l].Gcc.keb + 1);
      sprintf(fname_dom, "out_ghost_%d_%d_of_%d.vtr", rec_vtk_stepnum_out, l, nprocs);
      fprintf(outfile, "Source=\"%s\"/>\n", fname_dom);
    }
    fprintf(outfile, "</PRectilinearGrid>\n");
    fprintf(outfile, "</VTKFile>\n");
    fclose(outfile);
  }

  // interpolate velocities to cell centers
  // cell-center working arrays
  real *uu = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *vv = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *ww = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *uu_star = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *vv_star = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *ww_star = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *flag_uu = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *flag_vv = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *flag_ww = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *conv_uu = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *conv_vv = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *conv_ww = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *diff_uu = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *diff_vv = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));
  real *diff_ww = (real*) malloc(dom[rank].Gcc.s3b * sizeof(real));

  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        Cx = GFX_LOC(i, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        Cx_e = GFX_LOC(i + 1, j, k, dom[rank].Gfx.s1b, dom[rank].Gfx.s2b);
        Cy = GFY_LOC(i, j, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        Cy_n = GFY_LOC(i, j + 1, k, dom[rank].Gfy.s1b, dom[rank].Gfy.s2b);
        Cz = GFZ_LOC(i, j, k, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);
        Cz_t = GFZ_LOC(i, j, k + 1, dom[rank].Gfz.s1b, dom[rank].Gfz.s2b);

        
        //printf("u[%d,%d,%d (%d)] = %lf\n", i, j, k, Cx, u[Cx]);

        // interpolate velocity
        uu[C] = 0.5 * (u[Cx] + u[Cx_e]);
        vv[C] = 0.5 * (v[Cy] + v[Cy_n]);
        ww[C] = 0.5 * (w[Cz] + w[Cz_t]);

        uu_star[C] = 0.5 * (u_star[Cx] + u_star[Cx_e]);
        vv_star[C] = 0.5 * (v_star[Cy] + v_star[Cy_n]);
        ww_star[C] = 0.5 * (w_star[Cz] + w_star[Cz_t]);

        conv_uu[C] = 0.5 * (conv_u[Cx] + conv_u[Cx_e]);
        conv_vv[C] = 0.5 * (conv_v[Cy] + conv_v[Cy_n]);
        conv_ww[C] = 0.5 * (conv_w[Cz] + conv_w[Cz_t]);

        diff_uu[C] = 0.5 * (diff_u[Cx] + diff_u[Cx_e]);
        diff_vv[C] = 0.5 * (diff_v[Cy] + diff_v[Cy_n]);
        diff_ww[C] = 0.5 * (diff_w[Cz] + diff_w[Cz_t]);

        // interpolate flags
        flag_uu[C] = 0.5*(flag_u[Cx] + flag_u[Cx_e]);
        flag_vv[C] = 0.5*(flag_v[Cy] + flag_v[Cy_n]);
        flag_ww[C] = 0.5*(flag_w[Cz] + flag_w[Cz_t]);
      }
    }
  }

  // write each subdomain file -- open file for writing
  sprintf(fname, "%s/%s/out_ghost_%d_%d_of_%d.vtr", ROOT_DIR, OUTPUT_DIR, rec_vtk_stepnum_out,
    rank, nprocs);
  FILE *outfile = fopen(fname, "w");
  if(outfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  fprintf(outfile, "<VTKFile type=\"RectilinearGrid\">\n");
  fprintf(outfile, "<RectilinearGrid WholeExtent=");
  fprintf(outfile, "\"%d %d ", dom[rank].Gcc.isb, dom[rank].Gcc.ieb + 1);
  fprintf(outfile, "%d %d ", dom[rank].Gcc.jsb, dom[rank].Gcc.jeb + 1);
  fprintf(outfile, "%d %d\" ", dom[rank].Gcc.ksb, dom[rank].Gcc.keb + 1);
  fprintf(outfile, "GhostLevel=\"1\">\n");

  fprintf(outfile, "<Piece Extent=\"");
  fprintf(outfile, "%d %d ", dom[rank].Gcc.isb, dom[rank].Gcc.ieb + 1);
  fprintf(outfile, "%d %d ", dom[rank].Gcc.jsb, dom[rank].Gcc.jeb + 1);
  fprintf(outfile, "%d %d\">\n", dom[rank].Gcc.ksb, dom[rank].Gcc.keb + 1);
  fprintf(outfile, "<CellData Scalars=\"p phi phase phase_shell\" ");
  fprintf(outfile, "Vectors=\"vel vel_star conv_vel diff_vel flag\">\n");

  // write pressure for this subdomain
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"p\">\n");
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%e ", p[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write phi for this subdomain
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"phi\">\n");
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%e ", phi[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write phase for this subdomain
  fprintf(outfile, "<DataArray type=\"Int32\" Name=\"phase\">\n");
  if (NPARTS > 0) {
    for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
      for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
        for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
          C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);

          int tmp_phase = phase[C]*(phase[C] < 0);
          fprintf(outfile, "%d ", tmp_phase);
        }
      }
    }
  } else {
    for (i = 0; i < dom[rank].Gcc.s3b; i++) {
      fprintf(outfile, "%e ", -1.);
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write phase_shell for this subdomain
  fprintf(outfile, "<DataArray type=\"Int32\" Name=\"phase_shell\">\n");
  if (NPARTS > 0) {
    for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
      for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
        for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
          C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
          fprintf(outfile, "%d ", phase_shell[C]);
        }
      }
    }
  } else {
    for (i = 0; i < dom[rank].Gcc.s3b; i++) {
      fprintf(outfile, "%e ", -1.);
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write velocity vector
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"vel\"");
  fprintf(outfile, " NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%e %e %e ", uu[C], vv[C], ww[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write velocity star vector
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"vel_star\"");
  fprintf(outfile, " NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%e %e %e ", uu_star[C], vv_star[C], ww_star[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write conv_velocity vector
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"conv_vel\"");
  fprintf(outfile, " NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%e %e %e ", conv_uu[C], conv_vv[C], conv_ww[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write diff_velocity vector
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"diff_vel\"");
  fprintf(outfile, " NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%e %e %e ", diff_uu[C], diff_vv[C], diff_ww[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  // write flag vector
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"flag\"");
  fprintf(outfile, " NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (k = dom[rank].Gcc._ksb; k <= dom[rank].Gcc._keb; k++) {
    for (j = dom[rank].Gcc._jsb; j <= dom[rank].Gcc._jeb; j++) {
      for (i = dom[rank].Gcc._isb; i <= dom[rank].Gcc._ieb; i++) {
        C = GCC_LOC(i, j, k, dom[rank].Gcc.s1b, dom[rank].Gcc.s2b);
        fprintf(outfile, "%lf %lf %lf ", flag_uu[C], flag_vv[C], flag_ww[C]);
      }
    }
  }
  fprintf(outfile, "\n</DataArray>\n");

  fprintf(outfile, "</CellData>\n");

  fprintf(outfile, "<Coordinates>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"x\">\n");
  for (i = dom[rank].Gcc._isb - DOM_BUF; i <= dom[rank].Gcc._ieb; i++) {
    fprintf(outfile, "%lf ", i * dom[rank].dx + dom[rank].xs);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");

  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"y\">\n");
  for (j = dom[rank].Gcc._jsb - DOM_BUF; j <= dom[rank].Gcc._jeb; j++) {
    fprintf(outfile, "%lf ", j * dom[rank].dy + dom[rank].ys);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");

  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"z\">\n");
  for (k = dom[rank].Gcc._ksb - DOM_BUF; k <= dom[rank].Gcc._keb; k++) {
    fprintf(outfile, "%lf ", k * dom[rank].dz + dom[rank].zs);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");

  fprintf(outfile, "</Coordinates>\n");
  fprintf(outfile, "</Piece>\n");
  fprintf(outfile, "</RectilinearGrid>\n");
  fprintf(outfile, "</VTKFile>\n");
  fclose(outfile);
  

  // clean up interpolated fields
  free(uu);
  free(vv);
  free(ww);
  free(uu_star);
  free(vv_star);
  free(ww_star);
  free(conv_uu);
  free(conv_vv);
  free(conv_ww);
  free(diff_uu);
  free(diff_vv);
  free(diff_ww);
  free(flag_uu);
  free(flag_vv);
  free(flag_ww);
}

void part_out_VTK_ghost(void)
{
  int i;                           // iterator
  char fname[FILE_NAME_SIZE] = ""; // output filename

  // only work on pvtp file once
  if (rank == 0) {
    sprintf(fname, "%s/%s/out_ghost_%d.pvtp", ROOT_DIR, OUTPUT_DIR,
      rec_vtk_stepnum_out);
    FILE *outfile = fopen(fname, "w");
    if (outfile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }

    // write paraview vtp file
    fprintf(outfile, "<VTKFile type=\"PPolyData\">\n");
    fprintf(outfile, "<PPolyData GhostLevel=\"0\">\n");
    fprintf(outfile, "<PPointData Scalars=\"n r\" Vectors=\"pvel\">\n");
    fprintf(outfile, "<PDataArray type=\"Int32\" Name=\"n\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"r\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"pvel\" ");
    fprintf(outfile, "NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "</PPointData>\n");
    fprintf(outfile, "<PPoints>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n");
    fprintf(outfile, "</PPoints>\n");
    for (int n = 0; n < nprocs; n++) {
      fprintf(outfile, "<Piece Source=\"out_ghost_%d_%d_of_%d.vtp\"/>\n", 
        rec_vtk_stepnum_out, n, nprocs);
    }
    fprintf(outfile, "</PPolyData>\n");
    fprintf(outfile, "</VTKFile>\n");

    fclose(outfile);
  }

  // Open vtp files for writing
  sprintf(fname, "%s/%s/out_ghost_%d_%d_of_%d.vtp", ROOT_DIR, OUTPUT_DIR,
    rec_vtk_stepnum_out, rank, nprocs);
  FILE *outfile = fopen(fname, "w");
  if (outfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  // Calculate number of actual particles (not ghosts)
  // Inherited from cuda_part_pull

  // Set up file structure
  fprintf(outfile, "<VTKFile type=\"PolyData\">\n");
  fprintf(outfile, "<PolyData>\n");
  fprintf(outfile, "<Piece NumberOfPoints=\"%d\" NumberOfVerts=\"0\" ", nparts_subdom);
  fprintf(outfile, "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n");

  // Output particle positions
  fprintf(outfile, "<Points>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  // write the locations of the particle centers
  for (i = 0; i < nparts_subdom; i++) {
    fprintf(outfile, "%e %e %e ", parts[i].x, parts[i].y, parts[i].z);
  }
  fprintf(outfile, "\n</DataArray>\n");
  fprintf(outfile, "</Points>\n");

  // Outpart particle data
  fprintf(outfile, "<PointData Scalars=\"n r\">\n");

  // part number
  fprintf(outfile, "<DataArray type=\"Int32\" Name=\"n\" format=\"ascii\">\n");
  for(i = 0; i < nparts_subdom; i++) {
    fprintf(outfile, "%d ", parts[i].N);
  }
  fprintf(outfile, "\n</DataArray>\n");

  // part radius
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"r\" format=\"ascii\">\n");
  for (i = 0; i < nparts_subdom; i++) {
    fprintf(outfile, "%e ", parts[i].r);
  }
  fprintf(outfile, "\n</DataArray>\n");

  // part velocity
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"pvel\" format=\"ascii\" ");
  fprintf(outfile, "NumberOfComponents=\"3\">\n");
  for (i = 0; i < nparts_subdom; i++) {
    fprintf(outfile, "%e %e %e ", parts[i].u, parts[i].v, parts[i].w);
  }
  fprintf(outfile, "\n</DataArray>\n");

  // finish file
  fprintf(outfile, "</PointData>\n");
  fprintf(outfile, "</Piece>\n");
  fprintf(outfile, "</PolyData>\n");
  fprintf(outfile, "</VTKFile>\n");
  fclose(outfile);
}

