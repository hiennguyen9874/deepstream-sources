################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell g++ -dumpmachine | cut -f1 -d -)

CXX:= g++
SRCS:= nvds_dec_parse.cpp nvds_infer_parse.cpp \
       nvds_rest_server.cpp nvds_roi_parse.cpp nvds_stream_parse.cpp \
       nvds_mux_parse.cpp nvds_inferserver_parse.cpp nvds_conv_parse.cpp nvds_enc_parse.cpp \
       nvds_app_instance_parse.cpp nvds_osd_parse.cpp

INCS:= $(wildcard *.h)
LIB:=libnvds_rest_server.so
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
NVDS_VERSION:=6.3

CFLAGS+= -fPIC -DDS_VERSION=\"6.3.0\" \
 	 -I../../includes \
 	 -I/usr/local/cuda-$(CUDA_VER)/include \
	 -I/usr/include/jsoncpp
 
LIBS := -shared -Wl,-no-undefined \
 	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart
 
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_customhelper -ldl\
        -Wl,-rpath,$(LIB_INSTALL_DIR) -lcivetweb

ifeq ($(TARGET_DEVICE),aarch64)
   LIBS+= -L/usr/lib/aarch64-linux-gnu/tegra/ -ljsoncpp
else
   LIBS+= -L$(LIB_INSTALL_DIR) -ljsoncpp
endif

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
