#!/bin/bash
################################################################################
# Copyright (c) 2020-2024 NVIDIA Corporation.  All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

MODEL_REPO_DIR="triton_model_repo"
WITH_INT8=true

if [ -z "${TRTEXEC_BIN}" ]; then
    TRTEXEC_BIN=/usr/src/tensorrt/bin/trtexec
fi

if [ ! -f "${TRTEXEC_BIN}" ]; then
    echo "trtexec binary not found. Set TRTEXEC_BIN"
fi

if [[ $(uname -m) == "aarch64" ]]; then
  TAO_CONVERTER_VERSION="v5.1.0_jp6.0_aarch64"
  export TRT_LIB_PATH=/usr/lib/aarch64-linux-gnu
  export TRT_INC_PATH=/usr/include/aarch64-linux-gnu
else
  TAO_CONVERTER_VERSION="v5.1.0_8.6.3.1_x86"
  export TRT_LIB_PATH="/usr/lib/x86_64-linux-gnu"
  export TRT_INC_PATH="/usr/include/x86_64-linux-gnu"
fi

if [ -z "$TAO_CONVERTER_BIN" ]; then

    TAO_CONVERTER_DIR="$HOME/bin/tao-converter"
    TAO_CONVERTER_BIN="$TAO_CONVERTER_DIR/tao-converter"

    if [ ! -f "$TAO_CONVERTER_BIN" ]; then
        echo "Downloading and installing tao-converter utility into $TAO_CONVERTER_DIR"
        mkdir -p $TAO_CONVERTER_DIR
        wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/$TAO_CONVERTER_VERSION/zip \
            -O $TAO_CONVERTER_DIR/tao-converter_${TAO_CONVERTER_VERSION}.zip
        unzip $TAO_CONVERTER_DIR/tao-converter_${TAO_CONVERTER_VERSION}.zip -d $TAO_CONVERTER_DIR
        chmod +x $TAO_CONVERTER_DIR/tao-converter
        rm -rf $TAO_CONVERTER_DIR/tao-converter_${TAO_CONVERTER_VERSION}.zip
    fi
fi

function updateModelConfig {
    ModelName="$1"
    ConfigFile="$2"
    sed -i -e "s|default_model_filename.*|default_model_filename:\"$ModelName\"|" ${ConfigFile}
}

function buildEngineFromCaffe {
    ModelDir="$1"
    MaxBatch="$2"
    OutputLayers="$3"

    if [[ $# -ne 3 ]]; then
        exit 1
    fi

    echo "Building Model ${ModelDir}..."

    CalibFile=$(ls "models/${ModelDir}"/cal_trt.bin)
    ProtoFile=$(ls "models/${ModelDir}"/*.prototxt)
    ModelFile=$(ls "models/${ModelDir}"/*.caffemodel)
    ModelFileName=$(basename "${ModelFile}")

    if [ "${WITH_INT8}" = true ]; then
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_int8.engine"
    else
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_fp16.engine"
    fi

    mkdir -p $(dirname ${EngineFile})

    TRTEXEC_CMD="${TRTEXEC_BIN} \
        --calib=${CalibFile} \
        --deploy=${ProtoFile} \
        --model=${ModelFile} \
        --maxBatch=${MaxBatch} \
        --saveEngine=${EngineFile} \
        --buildOnly"

    for layer in ${OutputLayers}; do
        TRTEXEC_CMD="${TRTEXEC_CMD} --output=${layer}"
    done

    if [ "${WITH_INT8}" = true ]; then
        TRTEXEC_CMD="${TRTEXEC_CMD} --int8"
    else
        TRTEXEC_CMD="${TRTEXEC_CMD} --fp16"
    fi

    echo "Generating Engine file: ${EngineFile}"

    LOG_FILE="buildModel${ModelDir}.log"
    LOG_FILE=${LOG_FILE//[\/]/_}
    if eval "${TRTEXEC_CMD}" >> "${LOG_FILE}" 2>&1 ; then
        echo "Finished building Model ${ModelDir}"
        rm "${LOG_FILE}"
    else
        echo "ERROR: Failed to build engine for model \"$ModelDir\". Check ${LOG_FILE} for more information."
        return 1;
    fi

    if [ "${WITH_INT8}" != true ]; then
        EngineFileName=$(basename "${EngineFile}")
        updateModelConfig ${EngineFileName} ${MODEL_REPO_DIR}/${ModelDir}/config.pbtxt
    fi
}

function buildEngineFromEtlt {
    ModelDir="$1"
    MaxBatch="$2"
    TltModelKey="$3"
    OutputLayers="$4"
    Dimensions="$5"

    if [[ $# -ne 5 ]]; then
        exit 1
    fi

    echo "Building Model ${ModelDir}..."

    CalibFile=$(ls "models/${ModelDir}"/cal_trt.bin)
    TltEncodedModel=$(ls "models/${ModelDir}"/*.etlt)
    ModelFileName=$(basename "${TltEncodedModel}")

    mkdir -p "${MODEL_REPO_DIR}/${ModelDir}/1/"

    if [ "${WITH_INT8}" = true ]; then
        ModelFormat="int8"
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_int8.engine"
    else
        ModelFormat="fp16"
        EngineFile="${MODEL_REPO_DIR}/${ModelDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_fp16.engine"
    fi

    TAO_CONVERTER_CMD="${TAO_CONVERTER_BIN} \
                ${TltEncodedModel} \
                -c ${CalibFile} \
                -k ${TltModelKey} \
                -b ${MaxBatch} \
                -m ${MaxBatch} \
                -e ${EngineFile}\
                -o ${OutputLayers} \
                -d ${Dimensions} \
                -t ${ModelFormat}"

    echo "Generating Engine file: ${EngineFile}"

    LOG_FILE="buildModel${ModelDir}.log"
    LOG_FILE=${LOG_FILE//[\/]/_}
    if eval "${TAO_CONVERTER_CMD}" >> "${LOG_FILE}" 2>&1 ; then
        echo "Finished building Model ${ModelDir}"
        rm "${LOG_FILE}"
    else
        echo "ERROR: Failed to build engine for model \"$ModelDir\". Check ${LOG_FILE} for more information."
        return 1;
    fi

    if [ "${WITH_INT8}" != true ]; then
        EngineFileName=$(basename "${EngineFile}")
        updateModelConfig ${EngineFileName} ${MODEL_REPO_DIR}/${ModelDir}/config.pbtxt
    fi

}

function buildEngineFromUff {
    ModelDir="$1"
    MaxBatch="$2"
    InputLayer="$3"
    OutputLayers="$4"
    ModelServerDir="$5"

    if [[ $# -ne 5 ]]; then
        exit 1
    fi

    echo "Building Model ${ModelDir}..."

    UffFile=$(ls "models/${ModelDir}"/*.uff)
    ModelFileName=$(basename "${UffFile}")
    EngineFile="${MODEL_REPO_DIR}/${ModelServerDir}/1/${ModelFileName}_b${MaxBatch}_gpu0_fp32.engine"

    mkdir -p $(dirname ${EngineFile})

    TRTEXEC_CMD="${TRTEXEC_BIN} \
        --uff=${UffFile} \
        --maxBatch=${MaxBatch} \
        --saveEngine=${EngineFile} \
        --uffInput=${InputLayer} \
        --buildOnly"

    for layer in ${OutputLayers}; do
        TRTEXEC_CMD="${TRTEXEC_CMD} --output=${layer}"
    done

    echo "Generating Engine file: ${EngineFile}"

    LOG_FILE="buildModel${ModelDir}.log"
    LOG_FILE=${LOG_FILE//[\/]/_}
   if eval "${TRTEXEC_CMD}" >> "${LOG_FILE}" 2>&1 ; then
       echo "Finished building Model ${ModelDir}"
       rm "${LOG_FILE}"
    else
       echo "ERROR: Failed to build engine for model \"$ModelDir\". Check ${LOG_FILE} for more information."
       return 1;
   fi
}

echo "Generating Engine files for CaffeModels provided with the SDK"

echo "Checking for INT8 support..."
if buildEngineFromEtlt "Primary_Detector" 1 "tlt_encode" "output_cov/Sigmoid,output_bbox/BiasAdd" "3,544,960" > /dev/null; then
    echo "Platform supports INT8. Generating engine files using INT8 mode."
else
    echo "Platform does not support INT8. Generating engine files using FP16 mode."
    WITH_INT8=false
fi

buildEngineFromEtlt "Primary_Detector" 30 "tlt_encode" "output_cov/Sigmoid,output_bbox/BiasAdd" "3,544,960" || exit 1
buildEngineFromEtlt "Secondary_VehicleMake" 16 "tlt_encode" "predictions/Softmax" "3,224,224" || exit 1
buildEngineFromEtlt "Secondary_VehicleTypes" 16 "tlt_encode" "predictions/Softmax" "3,224,224" || exit 1

echo "Generating Engine files for segmentation UFF models provided with the SDK"

buildEngineFromUff "Segmentation/semantic" 1 "data,3,512,512" "final_conv/BiasAdd" "Segmentation_Semantic" || exit 1
buildEngineFromUff "Segmentation/industrial" 1 "input_1,1,512,512" "conv2d_19/Sigmoid" "Segmentation_Industrial" || exit 1

echo "Finished generating engine files."

echo "Downloading models from the TensorRT inference server samples"

# Refer to https://github.com/NVIDIA/tensorrt-inference-server/blob/r20.01/docs/examples/fetch_models.sh
# for the source of the models.

Gpu0Instance=$(cat <<-'EOL'

instance_group {
  count: 1
  gpus: 0
  kind: KIND_GPU
}
EOL
)

echo "Downloading TensorFlow inception..."
mkdir -p ${MODEL_REPO_DIR}/inception_graphdef/1
wget -O /tmp/inception_v3_2016_08_28_frozen.pb.tar.gz \
     https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
(cd /tmp && tar xzf inception_v3_2016_08_28_frozen.pb.tar.gz)
mv /tmp/inception_v3_2016_08_28_frozen.pb ${MODEL_REPO_DIR}/inception_graphdef/1/model.graphdef
rm -fr /tmp/inception_v3_2016_08_28_frozen.pb.tar.gz

echo "Downloading ONNX DenseNet..."
mkdir -p ${MODEL_REPO_DIR}/densenet_onnx/1
wget -O ${MODEL_REPO_DIR}/densenet_onnx/1/model.onnx \
https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx

echo "Downloading TensorFlow SSD Coco..."
mkdir -p ${MODEL_REPO_DIR}/ssd_inception_v2_coco_2018_01_28/1
wget -O /tmp/ssd_inception_v2_coco_2018_01_28.tar.gz \
     http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
(cd /tmp && tar xzf ssd_inception_v2_coco_2018_01_28.tar.gz)
mv /tmp/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb \
    ${MODEL_REPO_DIR}/ssd_inception_v2_coco_2018_01_28/1/model.graphdef
rm -fr /tmp/ssd_inception_v2_coco_2018_01_28.tar.gz  /tmp/ssd_inception_v2_coco_2018_01_28

echo "Downloading TensorFlow SSD Mobilenet V1 Coco..."
mkdir -p ${MODEL_REPO_DIR}/ssd_mobilenet_v1_coco_2018_01_28/1
wget -O /tmp/ssd_mobilenet_v1_coco_2018_01_28.tar.gz \
     http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
(cd /tmp && tar xzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
mv /tmp/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb \
    ${MODEL_REPO_DIR}/ssd_mobilenet_v1_coco_2018_01_28/1/
rm -fr /tmp/ssd_mobilenet_v1_coco_2018_01_28.tar.gz  /tmp/ssd_mobilenet_v1_coco_2018_01_28

echo "Downloading TensorFlow Mobilenet V1..."
mkdir -p ${MODEL_REPO_DIR}/mobilenet_v1/1
wget -O /tmp/mobilenet_v1_1.0_224.tgz \
     http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
mkdir -p /tmp/mobilenet_v1_1.0_224
(cd /tmp && tar xzf mobilenet_v1_1.0_224.tgz -C /tmp/mobilenet_v1_1.0_224)
mv /tmp/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb \
    ${MODEL_REPO_DIR}/mobilenet_v1/1/model.graphdef
rm -fr /tmp/mobilenet_v1_1.0_224.tgz /tmp/mobilenet_v1_1.0_224

echo "Adding read permission for downloaded models"
find ${MODEL_REPO_DIR} -name '*.pb' -o -name 'model.*' | xargs chmod a+r

echo "Model repository prepared successfully."
