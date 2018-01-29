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

#ifndef _RNG_H
#define _RNG_H

#include <stdlib.h>
#include <math.h>

extern unsigned long long int rng_j; // seed value
extern unsigned long long int rng_v; // random number sequence

// set seed value
void rng_init(unsigned long long int seed);

// random unsigned 64-bit integer
unsigned long long int rng_uint64(void);

// random unsigned 32-bit integer
unsigned int rng_uint32(void);

// random 64-bit integer
long long int rng_int64(void);

// random 32-bit integer
int rng_int32(void);

// random float between 0 and 1
float rng_flt(void);

// random double between 0 and 1
double rng_dbl(void);

// random double gaussian disbribution
double gaussrand(void);

#endif
