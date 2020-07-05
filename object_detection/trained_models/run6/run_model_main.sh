PIPELINE_CONFIG_PATH=./ssd_mobilenet_v1_coco.config
MODEL_DIR=./train
NUM_TRAIN_STEPS=200000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
RUN_MAIN_DIR=/tf/models/research/object_detection
# RUN_MAIN_DIR=$HOME/github_repos/models/research/object_detection

mkdir -p ${MODEL_DIR}

python ${RUN_MAIN_DIR}/model_main.py \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	--model_dir=${MODEL_DIR} \
	--num_train_steps=${NUM_TRAIN_STEPS} \
	--sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
	--alsologtostderr
