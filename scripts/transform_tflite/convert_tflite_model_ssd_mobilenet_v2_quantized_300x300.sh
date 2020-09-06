INPUT_TENSORS='normalized_input_image_tensor'
OUTPUT_TENSORS='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3'
echo "CONVERTING frozen graph to TF Lite file..."
tflite_convert \
  --output_file=../../workspace/training_demo/tflite/output_tflite_graph.tflite \
  --graph_def_file=../../workspace/training_demo/tflite/tflite_graph.pb \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="${INPUT_TENSORS}" \
  --output_arrays="${OUTPUT_TENSORS}" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,300,300,3 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --allow_custom_ops
echo "TFLite graph generated at ${OUTPUT_DIR}/output_tflite_graph.tflite"
