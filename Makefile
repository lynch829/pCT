################################################################################
#
# Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OSX and Linux Platforms)
#
################################################################################

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-7.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
EXTRA_CCFLAGS   ?= -fopenmp

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)
GENCODE_FLAGS   := $(GENCODE_SM20)

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= $(shell echo ${HOME})/sandbox/bin/g++
GCC_BINDIR      := $(shell echo ${HOME})/sandbox/bin

# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) \
                 -I./include/

EXEC_DIR := bin
OBJ_DIR  := obj
SRC_DIR  := src

SRC      := $(SRC_DIR)/pCT_Reconstruction_Data_Segments.cu
OBJ      := $(patsubst %.cu, $(OBJ_DIR)/%.cu.o, $(notdir $(filter %.cu, $(SRC)))) \
            $(patsubst %.c,  $(OBJ_DIR)/%.c.o,  $(notdir $(filter %.c, $(SRC))))
EXEC     := $(EXEC_DIR)/pCT_Reconstruction

# Compile flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32 -arch=sm_20 -O3 #--compiler-bindir $(GCC_BINDIR)
else
      NVCCFLAGS := -m64 -arch=sm_20 -O3 #--compiler-bindir $(GCC_BINDIR)
endif

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH) -O3
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32 -O3
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64 -O3
  endif
endif

# Target rules
all: $(EXEC)

$(EXEC): makedirectory $(OBJ)
	$(NVCC) $(OBJ) -o $@
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	$(NVCC) -v $(NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
$(OBJ_DIR)/%.c.o: $(SRC_DIR)/%.c
	$(NVCC) -v $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
makedirectory:
	mkdir -p $(EXEC_DIR)
	mkdir -p $(OBJ_DIR)
run: $(EXEC)
	./$(EXEC)
clean:
	rm -rf $(OBJ_DIR) $(EXEC_DIR)
test:
	echo "test: "
	$(info OBJ     = $(OBJ))
	$(info LDFLAGS = $(LDFLAGS))
	$(info OSUPPER = $(OSUPPER))
	$(info OSLOWER = $(OSLOWER))
	$(info OS_ARCH = $(OS_ARCH))
	$(info OS_SIZE = $(OS_SIZE))
