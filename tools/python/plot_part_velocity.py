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

#!/usr/bin/env python

# PURPOSE:
#   Pull the particle velocity time series from Bluebottle CGNS output files and
#   plot starting at the given <start_time>. If no <start_time> is given, use all
#   available data.
#
# USAGE:
#   ./plot_part_velocity.py <./path/to/sim/output> <start_time>
#
# OUTPUT
#   <./path/to/sim/output>/img/part_vel.png

# Imports:
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import bluebottle_particle_reader as bbparts

#########

# Parse output directory from commandline
if len(sys.argv) >= 2:    # output directory given
  data_dir = sys.argv[1]

  if len(sys.argv) >= 3:  # start time given
    t_start= sys.argv[2]

else:                     # nothing given
  print("plot_part_velocity error: Invalid commandline arguments.")
  print("Usage: ")
  print("   ./plot_part_velocity.py <./path/to/sim/output> <start_time>")
  print(" or")
  print("   ./plot_part_velocity.py <./path/to/sim/output>")
  sys.exit()

# Init the reader
times = bbparts.init(data_dir)

# Get nparts
bbparts.open(times[0])
nparts = bbparts.read_nparts()
bbparts.close()

# Init data arays
u = np.zeros((len(times), nparts))
v = np.zeros((len(times), nparts))
w = np.zeros((len(times), nparts))
t = np.zeros(len(times))

# Loop over time and pull data
for tt,time in enumerate(times):
  bbparts.open(time)

  t[tt] = bbparts.read_time()

  (u[tt,:], v[tt,:], w[tt,:]) = bbparts.read_part_velocity()

  #print(np.mean(u[tt,:]), np.mean(v[tt,:]), np.mean(w[tt,:]))

  bbparts.close()

# Plot
fig = plt.figure()

ax1 = fig.add_subplot(311)
plt.plot(t, u)
plt.plot(t, np.mean(u, 1), 'ko-')
plt.xlabel("$t$")
plt.ylabel("$u$")

ax1 = fig.add_subplot(312)
plt.plot(t, v)
plt.plot(t, np.mean(v, 1), 'ko-')
plt.xlabel("$t$")
plt.ylabel("$v$")

ax1 = fig.add_subplot(313)
plt.plot(t, w)
plt.plot(t, np.mean(w, 1), 'ko-')
plt.xlabel("$t$")
plt.ylabel("$w$")

# Save figure
img_dir = data_dir + "/img/"
if not os.path.exists(img_dir):
  os.makedirs(img_dir)
plt.savefig(img_dir + "vel.png", bbox_inches='tight', format='png')
