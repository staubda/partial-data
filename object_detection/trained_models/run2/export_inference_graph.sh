INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=./ssd_mobilenet_v1_coco.config
TRAINED_CKPT_PREFIX=./train/model.ckpt-200000
EXPORT_DIR=./inference_graph

python ~/github_repos/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
