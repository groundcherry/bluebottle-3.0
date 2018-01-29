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
def init(sys):
  # Parse command line args
  if len(sys.argv) == 2:
    basedir = sys.argv[1]
  else:
    print("Usage: requires ./path/to/some/output as command-line argument")
    sys.exit()

  if not basedir.endswith("/"):
    basedir = basedir + "/"

  global base

  base = basedir

  t_read = list()

  files = glob.glob(base + "/flow-*.cgns")

  if(len(files) == 0):
    print("cannot find any flow-*.cgns files in", base)
    sys.exit()
  else:
    for i in files:
      start = i.find("flow-") # XXX breaks on dirs with "flow-" in name
      t_read.append(i[start+5:-5])

  return (sorted(t_read, key=float), basedir)

# Open a particular CGNS file using a time value in the list returned by init().
def open(time):
  global f

  infile = base + "/flow-" + time + ".cgns"

  try:
    f = h5.File(infile, 'r')
    return f
  except OSError:
    f = None
    print("file", infile, "does not exist")
    return f

def close():
  f.close()

# Read the time.
def read_time():
  t1 = f["/Base/Zone0/Etc/Time/ data"][0]
  try:
    t2 = g["/Base/Zone0/Etc/Time/ data"][0]
  except NameError:
    return t1
  else:
    return (t1,t2)

# Read flow parameters
def read_flow_params():
  rho_f = numpy.array(f["/Base/Zone0/Etc/Density/ data"]) 
  nu = numpy.array(f["/Base/Zone0/Etc/KinematicViscosity/ data"]) 

  return (rho_f, nu)

# Read grid extents
def read_flow_extents(basedir):
  infile = basedir + "/grid.cgns"
  try:
    gr = h5.File(infile, 'r')
  except OSError:
    gr = None
    print("file", infile, "does not exist")
    sys.exit()

  Nxyz = numpy.array(gr["/Base/Zone0/ data"])
  Nx = Nxyz[1, 2]
  Ny = Nxyz[1, 1]
  Nz = Nxyz[1, 0]

  # These are output as x[k,j,i]
  x = numpy.array(gr["/Base/Zone0/GridCoordinates/CoordinateX/ data"])
  y = numpy.array(gr["/Base/Zone0/GridCoordinates/CoordinateY/ data"])
  z = numpy.array(gr["/Base/Zone0/GridCoordinates/CoordinateZ/ data"])

  xs = numpy.min(x)
  xe = numpy.max(x)
  xl = xe - xs
  ys = numpy.min(y)
  ye = numpy.max(y)
  yl = ye - ys
  zs = numpy.min(z)
  ze = numpy.max(z)
  zl = ze - zs

  return (Nx, Ny, Nz, xs, xe, xl, ys, ye, yl, zs, ze, zl)

# Read the flow positions.
def read_flow_position():
  # Open grid file
  #global gr
  infile = base + "/grid.cgns"
  try:
    gr = h5.File(infile, 'r')
  except OSError:
    gr = None
    print("file", infile, "does not exist")
    sys.exit()

  # These are output as x[k,j,i]
  x = numpy.array(gr["/Base/Zone0/GridCoordinates/CoordinateX/ data"])
  y = numpy.array(gr["/Base/Zone0/GridCoordinates/CoordinateY/ data"])
  z = numpy.array(gr["/Base/Zone0/GridCoordinates/CoordinateZ/ data"])

  x = x[0,0,:]
  y = y[0,:,0]
  z = z[:,0,0]

  return (x,y,z)

# Read the particle velocities.
def read_flow_velocity():
  u1 = numpy.array(f["/Base/Zone0/Solution/VelocityX/ data"])
  v1 = numpy.array(f["/Base/Zone0/Solution/VelocityY/ data"])
  w1 = numpy.array(f["/Base/Zone0/Solution/VelocityZ/ data"])
  try:
    u2 = numpy.array(g["/Base/Zone0/Solution/VelocityX/ data"])
    v2 = numpy.array(g["/Base/Zone0/Solution/VelocityY/ data"])
    w2 = numpy.array(g["/Base/Zone0/Solution/VelocityZ/ data"])
  except NameError:
    return (u1,v1,w1)
  else:
    return ((u1,v1,w1),(u2,v2,w2))
