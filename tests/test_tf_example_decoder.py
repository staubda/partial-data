import os

import pandas as pd

from partial_data.tfrecord import (write_examples_as_tfrecord, encode_object_detection_tf_example,
                                   read_examples_from_tfrecord)
from partial_data.tf_example_decoder import TfExampleDecoder


df_example = pd.DataFrame({
    'image_id': [208, 283],
    'wmin': [[0.00168, 0.0], [0.000537, 0.34]],
    'wmax': [[1.0, 0.475], [0.13969, 1.0]],
    'hmin': [[0.226, 0.313], [0.2228, 0.1047]],
    'hmax': [[0.9910, 0.5419], [0.35775, 0.5563]],
    'category_name': [['sink', 'sink'], ['chair', 'chair']],
    'category_id': [[7, 7], [4, 4]],
    'image_filepath': ['data/000000000208.jpg', 'data/000000000208.jpg'],
    'cats_mask': [[1, 1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
})
examples = [row for _, row in df_example.iterrows()]


def test_load_examples_with_class_mask():
    tmp_filepath = 'tmp.tfrecord'
    write_examples_as_tfrecord(
            examples,
            tmp_filepath,
            encode_object_detection_tf_example,
            num_shards=1
        )
    examples2 = read_examples_from_tfrecord(tmp_filepath, TfExampleDecoder().decode)
    assert examples2[0]['class_mask'].tolist() == df_example['cats_mask'].iloc[0]
    assert examples2[1]['class_mask'].tolist() == df_example['cats_mask'].iloc[1]
    os.remove(tmp_filepath)
