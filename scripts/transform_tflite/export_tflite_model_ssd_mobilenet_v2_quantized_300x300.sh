python export_tflite_ssd_graph.py \
--pipeline_config_path=../../workspace/training_demo/training/ssd_mobilenet_v2_quantized_300x300.config \
--trained_checkpoint_prefix=../../workspace/training_demo/training/model.ckpt-206 \
--output_directory=../../workspace/training_demo/tflite \
--add_postprocessing_op=true 
