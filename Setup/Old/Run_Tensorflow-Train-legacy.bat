
call Setup_Python_Paths.bat

call Config_Variables.bat

cd ..

call python ./Projects/xml_image_folder_to_tfrecord.py

call set CUDA_VISIBLE_DEVICES=1

cd ./Setup

call python %LEGACY_TRAIN_FILE% --logtostderr --train_dir=%LEGACY_DATA_TF_OUTPUT% --pipeline_config_path=%LEGACY_DATA_PIPELINE_CONFIG_PATH%
