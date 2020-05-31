NUM_SHARDS=1  # Set to NUM_GPUS if using the parallel evaluation script above
EVAL_DIR=./evaluate

INPUT_CONFIG_PATH=${EVAL_DIR}/input_config.pbtxt
EVAL_CONFIG_PATH=${EVAL_DIR}/eval_config.pbtxt

mkdir -p ${EVAL_DIR}

# Create input config file
echo "
label_map_path: '../../datasets/partial_v1/tfrecord/label_map.pbtxt'
tf_record_input_reader: { input_path: './test/test_detections.tfrecord@${NUM_SHARDS}' }
" > ${INPUT_CONFIG_PATH}

# Create eval config file
echo "
metrics_set: 'coco_detection_metrics'
" > ${EVAL_CONFIG_PATH}

# Run evaluation script
python ~/github_repos/models/research/object_detection/metrics/offline_eval_map_corloc.py \
  --eval_dir=${EVAL_DIR} \
  --eval_config_path=${EVAL_CONFIG_PATH} \
  --input_config_path=${INPUT_CONFIG_PATH}
