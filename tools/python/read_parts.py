#!/usr/bin/env python

# bluebottle_particle_reader python module example code

import sys, getopt
import numpy as np
import bluebottle_particle_reader as bbparts

# initialize the reader
times = bbparts.init("~/bluebottle/sim/output")

# visit all outputted time values
for time in times:
  # open the CGNS file for this particular output time
  bbparts.open(time)

  # read the CGNS file
  t = bbparts.read_time()
  n = bbparts.read_nparts()
  (x,y,z) = bbparts.read_part_position()
  (u,v,w) = bbparts.read_part_velocity()

  print("time = ", time, "t =", t, "n =", n)

  print(u)
  sys.exit()

  # close the CGNS file
  bbparts.close()
