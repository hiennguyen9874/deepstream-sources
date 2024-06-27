####################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
####################################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I../../includes -I/usr/local/cuda-$(CUDA_VER)/include

CUFLAGS:= -I../../includes -I/usr/local/cuda-$(CUDA_VER)/include \
        -gencode arch=compute_61,code=compute_61 -gencode arch=compute_70,code=compute_70 \
        -gencode arch=compute_75,code=compute_75 -gencode arch=compute_80,code=compute_80 \
        -gencode arch=compute_86,code=compute_86 -gencode arch=compute_90,code=compute_90

ifeq ($(TARGET_DEVICE),aarch64)
CUFLAGS+= -gencode arch=compute_72,code=compute_72 -gencode arch=compute_87,code=compute_87
else
CUFLAGS+= -gencode arch=compute_60,code=compute_60 -gencode arch=compute_89,code=compute_89
endif

LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= nvdsinfer_yolo_engine.cpp \
           nvdsparsebbox_Yolo.cpp   \
           yoloPlugins.cpp    \
           trt_utils.cpp              \
           yolo.cpp           \
           kernels.cu           \
           NvDsInferParseCustomYolov3_cuda.cu
TARGET_LIB:= libnvdsinfer_custom_impl_Yolo.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $(CUFLAGS) $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB) *.o
