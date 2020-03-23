

call set TF_OD_PATH=../Tensorflow/models/research/object_detection/
call set TF_OD_PATH_RETURN=../../../../Setup

call set TRAIN_FILE="object_detection/model_main.py"

call set DATA_FOLDER=../Data

call set DATASET_FOLDER=%DATA_FOLDER%/TensorflowDatasets
call set DATA_TF_OUTPUT="%DATA_FOLDER%/TensorflowOutput/"


call set WIP_DIRECTORY=../Projects/wip/
call set WIP_DIRECTORY_RETURN=../../Setup

call set NST_DIR=../Projects/neural-style-master
call set NST_DIR_RETURN=../../Setup


call set LEGACY_TRAIN_FILE="%TF_OD_PATH%legacy/train.py"
call set LEGACY_DATA_PIPELINE_CONFIG_PATH="%DATA_FOLDER%/TensorflowConfig/faster_rcnn_inception_v2_pets.config"
call set LEGACY_DATA_TF_OUTPUT="%DATA_FOLDER%/TensorflowOutput-Legacy/"