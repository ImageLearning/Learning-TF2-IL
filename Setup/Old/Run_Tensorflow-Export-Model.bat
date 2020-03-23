

call Setup_Python_Paths.bat

call Config_Variables.bat

call python ../Tensorflow/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ../Data/TensorflowConfig/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix ../Data/TensorflowOutput/model.ckpt-36374 --output_directory ../Data/TensorFlowFrozenModel

call copy "..\Data\TensorflowRecords\labelmap.pbtxt" "..\Data\TensorFlowFrozenModel\labelmap.pbtxt" /y

