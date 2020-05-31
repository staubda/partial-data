SPLIT=test  # or test
TF_RECORD_FILES=$(ls -1 ../../datasets/partial_v1/tfrecord/complete_${SPLIT}.record-* | tr '\n' ',')

python ~/github_repos/models/research/object_detection/inference/infer_detections.py \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_tfrecord_path=./test/${SPLIT}_detections.tfrecord-00000-of-00001 \
  --inference_graph=inference_graph/frozen_inference_graph.pb \
  --discard_image_pixels
