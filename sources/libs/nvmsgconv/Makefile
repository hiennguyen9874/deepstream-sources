###############################################################################
#
# Copyright (c) 2018-2023 NVIDIA CORPORATION.  All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
###############################################################################

CC:= g++

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid

NVDS_VERSION:=6.3

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -Wall -std=c++11 -shared -fPIC

CFLAGS+= -I../../includes -I./deepstream_schema

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -lyaml-cpp -lprotobuf

PROTOBUF_BUILD_DIR:= ./deepstream_schema

SRCFILES:= nvmsgconv.cpp  \
  deepstream_schema/eventmsg_payload.cpp \
  deepstream_schema/dsmeta_payload.cpp \
  deepstream_schema/deepstream_schema.cpp \
  $(PROTOBUF_BUILD_DIR)/schema.pb.cc

TARGET_LIB:= libnvds_msgconv.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB) $(PROTOBUF_BUILD_DIR)/*.pb.*

SCHEMA_PROTOS:=$(wildcard deepstream_schema/*.proto)
$(PROTOBUF_BUILD_DIR)/%.pb.cc: $(SCHEMA_PROTOS) Makefile
	protoc -I./deepstream_schema --cpp_out=$(PROTOBUF_BUILD_DIR)/ $<
