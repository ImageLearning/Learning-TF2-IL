
call Setup_Python_Paths.bat


echo "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md"

cd ../Tensorflow/models/research/object_detection

call python export_tflite_ssd_graph.py --pipeline_config_path="../../../../Data/TensorflowConfig/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config" --trained_checkpoint_prefix="../../../../Data/TensorflowOutput" --output_directory="../../../../Data/TensorflowOutput-TFLite/" --add_postprocessing_op=true

cd ../../../../Setup

cd ../Tensorflow/

bazel run --config=opt tensorflow/lite/toco:toco -- \
--input_file="../Data/TensorflowOutput-TFLite/tflite_graph.pb" \
--output_file="../Data/TensorflowOutput-TFLite/detect.tflite" \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops

cd ../Setup
