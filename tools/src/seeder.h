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

#ifndef _SEEDER_H
#define _SEEDER_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <time.h>

#define FILE_NAME_SIZE 256
#define PERIODIC 0
#define DIRICHLET 1
#define NEUMANN 2
#define PI 3.14159265358979323846

/* Variables */
extern int rand_u;         // Flag to randomize particle velocity
extern int array;          // Flag to seed in array
extern double vel_scale;   // Velocity Scale
extern int nx;             // nparts in x direction (for array)
extern int ny;             // nparts in y direction (for array)
extern int nz;             // nparts in z direction (for array)
extern double vfrac;       // Volume fraction
  
extern int nparts;         // Number of particles to seed
extern double loa;         // Interaction length ratio
extern double a;           // Radius
extern double x;           // Position
extern double y;           // Position
extern double z;           // Position
extern double u;           // Velocity
extern double v;           // Velocity
extern double w;           // Velocity
extern double aFx;
extern double aFy;
extern double aFz;
extern double aLx;
extern double aLy;
extern double aLz;
extern double rho;
extern double E;
extern double sigma;
extern double e_dry;
extern double coeff_fric;
extern int order;
extern double rs;
extern double spring_k;
extern double spring_x;
extern double spring_y;
extern double spring_z;
extern double spring_l;
extern int translating;
extern int rotating;
extern double ss;
extern int update;
extern double cp;
extern double srs;
extern int sorder;

extern char part_file[FILE_NAME_SIZE];     // ./path/to/input/part.config
extern char INPUT_DIR[FILE_NAME_SIZE];     // ./path/to/input

/* Functions */
void parse_cmdline(int argc, char *argv[]);
void seeder_read_input(void);
void domain_read_input(void);
void seeder(void);
void seed_array(void);

/* Structures */
typedef struct dom_struct {
  double xs;    // starting location
  double xe;    // ending location
  double xl;    // length
  double ys;
  double ye;
  double yl;
  double zs;
  double ze;
  double zl;
  double ds_e;  // east screen offset
  double ds_w;  // west screen offset
  double ds_n;
  double ds_s;
  double ds_t;
  double ds_b;
  int bc_e;     // east boundary condition, from pressure
  int bc_w;     // west boundary condition, from pressure
  int bc_n;
  int bc_s;
  int bc_t;
  int bc_b;
} dom_struct;

extern dom_struct dom;

typedef struct part_struct {
  double r;
  double x;
  double y;
  double z;
  double u;
  double v; 
  double w;
} part_struct;

extern part_struct *parts;


#endif // _SEEDER_H
