################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= deepstream-mrcnn-test

WITH_OPENCV?=1

CC=g++
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=7.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

SRCS:= $(wildcard *.cpp)

INCS:= $(wildcard *.h)

PKGS+= gstreamer-1.0 gstreamer-video-1.0

ifeq ($(WITH_OPENCV),1)
CFLAGS+= -DWITH_OPENCV
PKGS+= opencv4
endif

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -I../../../includes -std=c++11 \
		-I /usr/local/cuda-$(CUDA_VER)/include \
		-I /usr/include/opencv4

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lrt \
       -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

ifeq ($(WITH_OPENCV),1)
LIBS += -lopencv_imgproc -lopencv_core
endif


all: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)
