################################################################################
################################# BLUEBOTTLE ###################################
################################################################################
#
#   Copyright 2012 - 2018 Adam Sierakowski and Daniel Willen
#                       	  The Johns Hopkins University
# 
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# 
#   Please contact the Johns Hopkins University to use Bluebottle for
#   commercial and/or for-profit applications.
################################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EDIT: DEPENDENCIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
MPI_DIR = /opt/openmpi-2.1.1
HDF5_DIR = /opt/phdf5/1.8.19
CGNS_DIR = /opt/pcgns/3.3.0
CUDA_DIR = /opt/cuda
CUDA_SDK_DIR = /opt/cuda-7.5/samples
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EDIT: COMPILERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
MPICC = mpicc
NVCC = nvcc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ADVANCED OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Precision:
#	DOUBLE: Use double precision floating point arithmetic (RECOMMENDED)
#	SINGLE: Use single precision floating point arithmetic (UNTESTED)
PREC = DOUBLE

# Preconditioner:
#	JACOBI: Use the diagonal of A as the preconditioner, M=diag(A)
#	NOPRECOND: Use the identity matrix in place of the Jacobi preconditioner
PRECOND = JACOBI

# Output:
# 	TRUE: Compile with cgns libraries for normal runtime
#	FALSE: Compile without cgns libraries for benchmarking of MPI 
#		implementations
CGNS_OUTPUT = TRUE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

SRC_DIR = src
SIM_DIR = sim

COPT += -std=c99 -pedantic -Wall -Wextra -D$(PREC) -D$(PRECOND)
LDINCS = -I $(MPI_DIR)/include

ifeq ($(CGNS_OUTPUT),TRUE)
	COPT += -DCGNS_OUTPUT
	LDINCS += -I $(CGNS_DIR)/include
	LDLIBS = -lm -L $(HDF5_DIR)/lib -L $(CGNS_DIR)/lib -lcgns -lhdf5 
endif

CUDAOPT += -arch=sm_30 -Xcompiler -m64 -D$(PREC) -D$(PRECOND)

CUDAINCS = -I $(CUDA_SDK_DIR)/common/inc
CUDALIBS = -L $(CUDA_DIR)/lib64 -lcudart

SRCC =	bluebottle.c	\
	domain.c	\
	mpi_comm.c	\
	particle.c	\
	recorder.c	\
	rng.c		\
	vtk.c

SRCCUDA = cuda_bluebottle.cu	\
	cuda_solver.cu		\
	cuda_particle.cu	\
	cuda_physalis.cu	\
	bluebottle_kernel.cu	\
	solver_kernel.cu	\
	particle_kernel.cu	\
	physalis_kernel.cu	\
	cuda_testing.cu


EXTRA = Makefile	\
	bluebottle.h	\
	cuda_bluebottle.h	\
	cuda_solver.h	\
	cuda_particle.h	\
	cuda_physalis.h	\
	domain.h	\
	mpi_comm.h	\
	particle.h	\
	physalis.h	\
	recorder.h	\
	rng.h		\
	vtk.h		\
	bluebottle.cuh	\
	cuda_testing.h

ifeq ($(CGNS_OUTPUT),TRUE)
	SRCC += cgns.c
	EXTRA += cgns.h
endif

OBJS = $(addprefix $(SRC_DIR)/, $(addsuffix .o, $(basename $(SRCC))))
OBJSCUDA = $(addprefix $(SRC_DIR)/, $(addsuffix .o, $(basename $(SRCCUDA))))

# compile normally
all: COPT += -O2
all: CUDAOPT += -O2
all: bluebottle

# compile with debug output
debug: COPT += -DDDEBUG -g
debug: CUDAOPT += -DDDEBUG -DTHRUST_DEBUG -g -G
debug: bluebottle

# compile tests -- TODO include cuda_testing for only this rule
test_exp: test
test_exp: COPT += -DTEST_EXP
test_exp: CUDAOPT += -DTEST_EXP

test_sin: test
test_sin: COPT += -DTEST_SIN
test_sin: CUDAOPT += -DTEST_SIN

test_bc_periodic: test
test_bc_periodic: COPT += -DTEST_BC_PERIODIC
test_bc_periodic: CUDAOPT += -DTEST_BC_PERIODIC

test_bc_dirichlet: test
test_bc_dirichlet: COPT += -DTEST_BC_DIRICHLET
test_bc_dirichlet: CUDAOPT += -DTEST_BC_DIRICHLET

test_bc_neumann: test
test_bc_neumann: COPT += -DTEST_BC_NEUMANN
test_bc_neumann: CUDAOPT += -DTEST_BC_NEUMANN

test_interp: test
test_interp: COPT += -DTEST_LEBEDEV_INTERP
test_interp: CUDAOPT += -DTEST_LEBEDEV_INTERP

test_lamb: test
test_lamb: COPT += -DTEST_LAMB
test_lamb: CUDAOPT += -DTEST_LAMB

test: COPT += -DDDEBUG -DTEST -g
test: CUDAOPT += -DDDEBUG -DTEST -g -G
test: bluebottle


$(OBJSCUDA):$(SRC_DIR)/%.o:$(SRC_DIR)/%.cu
	$(NVCC) $(CUDAOPT) -dc $< $(CUDAINCS) $(LDINCS) -o $@

$(OBJS):$(SRC_DIR)/%.o:$(SRC_DIR)/%.c
	$(MPICC) $(COPT) -c $< $(LDINCS) -o $@

$(SRC_DIR)/bblib.o:$(OBJSCUDA)
	$(NVCC) $(CUDAOPT) -dlink $+ -o $(SRC_DIR)/bblib.o $(CUDALIBS)

bluebottle: $(OBJSCUDA) $(SRC_DIR)/bblib.o $(OBJS) 
	$(MPICC) $(COPT) -o $(SIM_DIR)/bluebottle $+ $(LDLIBS) $(CUDALIBS) -lstdc++

clean:
	rm -f $(SRC_DIR)/*.o $(SIM_DIR)/bluebottle
