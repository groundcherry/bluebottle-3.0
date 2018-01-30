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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>

#define FILE_NAME_SIZE 256

int main(int argc, char *argv[]) 
{
  // Variable declaration
  char flow_file[FILE_NAME_SIZE];
  double Xs, Xe, Xl;          // global domain start, end, length
  double Ys, Ye, Yl;
  double Zs, Ze, Zl;          
  int Xn, Yn, Zn;             // global domain grid cells
  int In, Jn, Kn;             // MPI subdomain decomposition
  double *xs, *ys, *zs;       // Subdomain start position
  double *xe, *ye, *ze;       // Subdomain end position
  double *xl, *yl, *zl;       // Subdomain length
  int *xn, *yn, *zn;          // Subdomain grid cells


  // Parse command line args
  if (argc == 2) {
    sprintf(flow_file, "%s", argv[1]);
  } else {
    printf("Usage: ./decomp_reader ./path/to/input/flow.config\n");
    exit(EXIT_FAILURE);
  }

  // Open flow file for reading
  int fret = 0;
  fret = fret;

  FILE *infile = fopen(flow_file, "r");
  if (infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", flow_file);
    exit(EXIT_FAILURE);
  }

  fret = fscanf(infile, "GLOBAL DOMAIN\n");
  fret = fscanf(infile, "(Xs, Xe, Xn) %lf %lf %d\n", &Xs, &Xe, &Xn);
  fret = fscanf(infile, "(Ys, Ye, Yn) %lf %lf %d\n", &Ys, &Ye, &Yn);
  fret = fscanf(infile, "(Zs, Ze, Zn) %lf %lf %d\n", &Zs, &Ze, &Zn);
  fret = fscanf(infile, "\n");

  Xl = Xe - Xs;
  Yl = Ye - Ys;
  Zl = Ze - Zs;

  fret = fscanf(infile, "MPI/GPU SUBDOMAIN DECOMPOSITION\n");
  fret = fscanf(infile, "(In, Jn, Kn) %d %d %d\n", &In, &Jn, &Kn);
  fret = fscanf(infile, "\n");

  printf("Xs, Xe, Xl = %lf, %lf, %lf\n", Xs, Xe, Xl);
  printf("Ys, Ye, Yl = %lf, %lf, %lf\n", Ys, Ye, Yl);
  printf("Zs, Ze, Zl = %lf, %lf, %lf\n", Zs, Ze, Zl);
  printf("Xn, Yn, Zn = %d, %d, %d\n", Xn, Yn, Zn);
  printf("In, Jn, Kn = %d, %d, %d\n", In, Jn, Kn);

  fclose(infile);

  // Ask for precision
  int prec;
  printf("fprintf precision: ");
  scanf("%d", &prec);

  // Calculate subdomain length
  xs = (double*) malloc(In * sizeof(double));
  ys = (double*) malloc(Jn * sizeof(double));
  zs = (double*) malloc(Kn * sizeof(double));
  xe = (double*) malloc(In * sizeof(double));
  ye = (double*) malloc(Jn * sizeof(double));
  ze = (double*) malloc(Kn * sizeof(double));
  xl = (double*) malloc(In * sizeof(double));
  yl = (double*) malloc(Jn * sizeof(double));
  zl = (double*) malloc(Kn * sizeof(double));
  xn = (int*) malloc(In * sizeof(int));
  yn = (int*) malloc(Jn * sizeof(int));
  zn = (int*) malloc(Kn * sizeof(int));

  if (Xn % In) 
    printf("Discretization in X not a multiple of the domain decomposition!\n");
  if (Yn % Jn) 
    printf("Discretization in Y not a multiple of the domain decomposition!\n");
  if (Zn % Kn) 
    printf("Discretization in Z not a multiple of the domain decomposition!\n");

  for (int i = 0; i < In; i++) {
    xl[i] = Xl / In;
    xn[i] = Xn / In;
    xs[i] = Xs + i * xl[i];
    xe[i] = xs[i] + xl[i];
  }
  for (int j = 0; j < Jn; j++) {
    yl[j] = Yl / Jn;
    yn[j] = Yn / Jn;
    ys[j] = Ys + j * yl[j];
    ye[j] = ys[j] + yl[j];
  }
  for (int k = 0; k < Kn; k++) {
    zl[k] = Zl / Kn;
    zn[k] = Zn / Kn;
    zs[k] = Zs + k * zl[k];
    ze[k] = zs[k] + zl[k];
  }

  // Find decomp.config from flow.config
  char tmp_name[FILE_NAME_SIZE];  // since dirname destroys input
  char INPUT_DIR[FILE_NAME_SIZE];
  sprintf(tmp_name, "%s", flow_file);
  sprintf(INPUT_DIR, "%s", dirname(tmp_name));

  char fname[FILE_NAME_SIZE] = "";
  sprintf(fname, "%s/%s", INPUT_DIR, "decomp.config");
  FILE *fout = fopen(fname, "w");

  // Need to loop and stride to print
  for (int k = 0; k < Kn; k++) {
    for (int j = 0; j < Jn; j++) {
      for (int i = 0; i < In; i++) {
        fprintf(fout,"(I, J, K) %d %d %d\n", i, j, k);
        fprintf(fout,"(Xs, Xe, Xn) %.*lf %.*lf %d\n", prec, xs[i], prec, xe[i],
          xn[i]);
        fprintf(fout,"(Ys, Ye, Yn) %.*lf %.*lf %d\n", prec, ys[j], prec, ye[j],
          yn[j]);
        fprintf(fout,"(Zs, Ze, Zn) %.*lf %.*lf %d\n\n", prec, zs[k], prec, ze[k],
          zn[k]);
      }
    }
  }

  fclose(fout);

  free(xs);
  free(ys);
  free(zs);
  free(xe);
  free(ye);
  free(ze);
  free(xl);
  free(yl);
  free(zl);
  free(xn);
  free(yn);
  free(zn);
}
