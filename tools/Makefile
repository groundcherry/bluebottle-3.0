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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EDIT: COMPILERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
CC = gcc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

OUT_DIR = out
SRC_DIR = src
BIN_DIR = bin

COPT = -std=c99 -pedantic -Wall -Wextra

SRCC_DECOMP_READER = decomp_reader.c
SRCC_SEEDER = seeder.c

EXTRA = Makefile

OBJS_DECOMP_READER = $(addprefix $(SRC_DIR)/, $(addsuffix .o, $(basename $(SRCC_DECOMP_READER))))
OBJS_SEEDER = $(addprefix $(SRC_DIR)/, $(addsuffix .o, $(basename $(SRCC_SEEDER))))

# compile normally
all: COPT += -O2
all: seeder decomp_reader

# debug
debug: COPT += -g -DDEBUG
debug: seeder decomp_reader

$(OBJS_DECOMP_READER):$(SRC_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(COPT) -c $< -o $@
$(OBJS_SEEDER):$(SRC_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(COPT) -c $< -o $@

decomp_reader: $(OBJS_DECOMP_READER)
	$(CC) $(COPT) -o $(BIN_DIR)/decomp_reader $+ -lstdc++

seeder: $(OBJS_SEEDER)
	$(CC) $(COPT) -o $(BIN_DIR)/seeder $+ -lstdc++ -lm

clean:
	rm -f $(SRC_DIR)/*.o $(BIN_DIR)/decomp_reader
	rm -f $(SRC_DIR)/*.o $(BIN_DIR)/seeder
