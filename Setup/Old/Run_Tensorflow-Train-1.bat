
call Setup_Python_Paths.bat

call Config_Variables.bat


call set img_dir="Data/DataPreperation/octocats"
call set test_dir="Data/DataPreperation/octocats/test"
call set train_dir="Data/DataPreperation/octocats/train"
call set output_path_test="Data/TensorflowRecords/test.record"
call set output_path_train="Data/TensorflowRecords/train.record"
call set output_path_labelmap="Data/TensorflowRecords/labelmap.pbtxt"
call set resize_Images=True

rem call set config_file_path=Data/TensorflowConfig/faster_rcnn_inception_v2_pets
rem call set config_file_path=Data/TensorflowConfig/ssd_mobilenet_v1_coco
rem call set config_file_path=Data/TensorflowConfig/ssd_mobilenet_v1_pets
rem call set config_file_path=Data/TensorflowConfig/ssd_inception_v3_pets

call set config_file_path=Data/TensorflowConfig/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync

rem What folder is our finetune checkpoint in ? that goes here
rem call set FINE_TUNE_CHECKPOINT="Data/TensorflowDatasets/ssd_mobileet_v1_coco_2018_01_28/model.ckpt"
rem call set FINE_TUNE_CHECKPOINT="Data/TensorflowDatasets/ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt"

call set FINE_TUNE_CHECKPOINT="Data/TensorflowDatasets/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/model.ckpt"


cd ..


call python ./Projects/xml_image_folder_to_tfrecord.py --img_dir=%img_dir%  --test_dir=%test_dir%  --train_dir=%train_dir%  --output_path_test=%output_path_test%  --output_path_train=%output_path_train%  --output_path_labelmap=%output_path_labelmap%  --config_file_path="%config_file_path%.template"  --Resize_Images=%resize_Images% --fine_tune_checkpoint=%FINE_TUNE_CHECKPOINT%

cd ./Setup

rem cd ../Data/TensorflowOutput
rem del *.* /S /Q
rem copy /y NUL .KeepMe >NUL
rem cd ../../Setup

cd ../Tensorflow/models/research/

call set FP=../../

call set PIPELINE_CONFIG_PATH="%FP%../%config_file_path%.config"
call set MODEL_DIR=%FP%%DATA_TF_OUTPUT%
call set NUM_TRAIN_STEPS=50000
call set SAMPLE_ONE_OF_N_EVAL_EXAMPLES=1


call python %TRAIN_FILE% --pipeline_config_path=%PIPELINE_CONFIG_PATH% --model_dir=%MODEL_DIR%  --num_train_steps=%NUM_TRAIN_STEPS%  --sample_1_of_n_eval_examples=%SAMPLE_ONE_OF_N_EVAL_EXAMPLES%  --alsologtostderr

cd ../../../Setup