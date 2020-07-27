#python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config
python train.py --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config
