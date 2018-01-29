################################################################################
################################## BLUEBOTTLE ##################################
################################################################################
#
#  Copyright 2012 - 2018 Adam Sierakowski and Daniel Willen,
#                         The Johns Hopkins University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Please contact the Johns Hopkins University to use Bluebottle for
#  commercial and/or for-profit applications.
################################################################################

import sys, os, glob
import h5py as h5
import numpy

# Initialize the reader by passing the directory containing the CGNS files. This
# returns a list containing the rounded time values available for reading.
def init(basedir):
  global base

  base = basedir

  t_read = list()

  files = glob.glob(base + "/part-*.cgns")
  if(len(files) == 0):
    print("cannot find any part-*.cgns files in", base)
    sys.exit()
  else:
    for i in files:
      start = i.find("part-") # this will break on dirs containing "part-"
      t_read.append(i[start+5:-5])

  return sorted(t_read, key=float)

# Open a particular CGNS file using a time value in the list returned by init().
def open(time):
  global f
  global part_index
  global sorted_index

  infile = base + "/part-" + time + ".cgns"

  # Open file
  try:
    f = h5.File(infile, 'r')
  except OSError:
    f = None
    print("file", infile, "does not exist")

  # Pull global indices
  part_index = f["/Base/Zone0/Solution/GlobalIndex/ data"]
  part_index = numpy.squeeze(part_index)

  # Sort global indices, carry along sorting index for later use
  sorted_index = numpy.argsort(part_index)
  part_index = part_index[sorted_index]

  return (f, part_index, sorted_index)

def close():
  f.close()

# Read the time.
def read_time():
  t1 = f["/Base/Zone0/Etc/Time/ data"][0]

  return t1

# Read number of particles
def read_nparts():
  nparts = numpy.squeeze(f["/Base/Zone0/ data"][0])

  return nparts

# Return particle position, sorted according to global index
def read_part_position():
  x1 = numpy.array(f["/Base/Zone0/GridCoordinates/CoordinateX/ data"])
  y1 = numpy.array(f["/Base/Zone0/GridCoordinates/CoordinateY/ data"])
  z1 = numpy.array(f["/Base/Zone0/GridCoordinates/CoordinateZ/ data"])

  x1 = x1[sorted_index]
  y1 = y1[sorted_index]
  z1 = z1[sorted_index]
  
  return (x1,y1,z1)

# Return particle velocities, sorted according to global index
def read_part_velocity():
  u1 = numpy.array(f["/Base/Zone0/Solution/VelocityX/ data"])
  v1 = numpy.array(f["/Base/Zone0/Solution/VelocityY/ data"])
  w1 = numpy.array(f["/Base/Zone0/Solution/VelocityZ/ data"])

  u1 = u1[sorted_index]
  v1 = v1[sorted_index]
  w1 = w1[sorted_index]

  return (u1,v1,w1)

# Return particle accelerations, sorted according to global index
def read_part_acceleration():
  udot = numpy.array(f["/Base/Zone0/Solution/AccelerationX/ data"])
  vdot = numpy.array(f["/Base/Zone0/Solution/AccelerationY/ data"])
  wdot = numpy.array(f["/Base/Zone0/Solution/AccelerationZ/ data"])

  udot = udot[sorted_index]
  vdot = vdot[sorted_index]
  wdot = wdot[sorted_index]

  return (udot,vdot,wdot)

# Return particle total force, sorted according to global index
def read_part_total_forces():
  Fx = numpy.array(f["/Base/Zone0/Solution/TotalForceX/ data"])
  Fy = numpy.array(f["/Base/Zone0/Solution/TotalForceY/ data"])
  Fz = numpy.array(f["/Base/Zone0/Solution/TotalForceZ/ data"])

  Fx = Fx[sorted_index]
  Fy = Fy[sorted_index]
  Fz = Fz[sorted_index]

  return (Fx, Fy, Fz)

# Return particle hydro force, sorted according to global index
def read_part_hydro_forces():
  hFx = numpy.array(f["/Base/Zone0/Solution/HydroForceX/ data"])
  hFy = numpy.array(f["/Base/Zone0/Solution/HydroForceY/ data"])
  hFz = numpy.array(f["/Base/Zone0/Solution/HydroForceZ/ data"])

  hFx = hFx[sorted_index]
  hFy = hFy[sorted_index]
  hFz = hFz[sorted_index]

  return (hFx, hFy, hFz)

# Return particle total force, sorted according to global index
def read_part_interaction_forces():
  iFx = numpy.array(f["/Base/Zone0/Solution/InteractionForceX/ data"])
  iFy = numpy.array(f["/Base/Zone0/Solution/InteractionForceY/ data"])
  iFz = numpy.array(f["/Base/Zone0/Solution/InteractionForceZ/ data"])

  iFx = iFx[sorted_index]
  iFy = iFy[sorted_index]
  iFz = iFz[sorted_index]

  return (iFx, iFy, iFz)

# Return particle radius, sorted according to global index
def read_part_radius():
  a = numpy.array(f["/Base/Zone0/Solution/Radius/ data"])

  a = a[sorted_index]

  return a

# perform periodic flip
def periodic_flip(xi, yi, zi, xs, xe, Lx, ys, ye, Ly, zs, ze, Lz):

  # Note: 1*(boolean) results in an int, False == 0 or True == 1
  # This makes the subtraction correct.
  xi += Lx*(1*(xi < xs) - 1*(xi > xe))
  yi += Ly*(1*(yi < ys) - 1*(yi > ye))
  zi += Lz*(1*(zi < zs) - 1*(zi > ze))

  return (xi, yi, zi)
