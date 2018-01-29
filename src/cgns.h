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

/****h* Bluebottle/cgns
 * NAME
 *  cgns
 * FUNCTION
 *  Write CGNS output files
 ******
 */

#ifndef _CGNS_H
#define _CGNS_H

#include <cgnslib.h>
#include <pcgnslib.h>

#include "bluebottle.h"

/****f* recorder/cgns_recorder_init()
 * NAME
 *  cgns_recorder_init()
 * TYPE
 */
void cgns_recorder_init(void);
/*
 * FUNCTION
 *  Initialize the cgns output recorder.
 ******
 */

/****f* recorder/cgns_recorder_flow_write()
 * NAME
 *  cgns_recorder_flow_write()
 * TYPE
 */
void cgns_recorder_flow_write(void);
/*
 * FUNCTION
 *  Write the cgns flow output.
 ******
 */

/****f* recorder/cgns_recorder_part_write()
 * NAME
 *  cgns_recorder_part_write()
 * TYPE
 */
void cgns_recorder_part_write(void);
/*
 * FUNCTION
 *  Write the cgns part output.
 ******
 */

/****f* recorder/cgns_grid()
 * NAME
 *  cgns_grid()
 * TYPE
 */
void cgns_grid(void);
/*
 * FUNCTION
 *  Write the CGNS grid output file.
 ******
 */

/****f* recorder/cgns_grid_ghost()
 * NAME
 *  cgns_grid_ghost()
 * TYPE
 */
void cgns_grid_ghost(void);
/*
 * FUNCTION
 *  Write the CGNS grid ghost output file.
 ******
 */

/****f* recorder/cgns_flow_field()
 * NAME
 *  cgns_flow_field()
 * TYPE
 */
void cgns_flow_field(real dtout);
/*
 * FUNCTION
 *  Write the CGNS flow_field output file.
 * ARGUMENTS
 *  dtout -- output timestep size
 ******
 */

/****f* recorder/cgns_particles()
 * NAME
 *  cgns_particles()
 * TYPE
 */
void cgns_particles(real dtout);
/*
 * FUNCTION
 *  Write the CGNS particle output file.
 * ARGUMENTS
 *  dtout -- output timestep size
 ******
 */

/****f* recorder/cgns_flow_field_ghost()
 * NAME
 *  cgns_flow_field_ghost()
 * TYPE
 */
void cgns_flow_field_ghost(real dtout);
/*
 * FUNCTION
 *  Write the CGNS flow_field_ghost output file.
 * ARGUMENTS
 *  dtout -- output timestep size
 ******
 */

#endif
