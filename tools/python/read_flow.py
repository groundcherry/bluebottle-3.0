#!/usr/bin/env python

# bluebottle_reader python module example code

import sys, getopt
import numpy as np
import bluebottle_flow_reader as bbflow

# initialize the reader
times = bbflow.init("~/scratch/tmp-cgns")

# Pull grid positions
(x,y,z) = bbflow.read_flow_position()

# visit all outputted time values
for time in times:
  # open the CGNS file for this particular output time
  bbflow.open(time)

  # read the CGNS file
  t = bbflow.read_time()
  (u,v,w) = bbflow.read_flow_velocity()

  print("t =", t)

  # close the CGNS file
  bbflow.close()

