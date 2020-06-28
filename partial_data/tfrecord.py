"""Utils for creating and handling Tensorflow TFRecords.
"""
import os
import io
import contextlib
import hashlib

from PIL import Image

from tqdm import tqdm

import tensorflow as tf

try:
    from tensorflow.python_io import TFRecordWriter
    from tensorflow.gfile import GFile
    from tensorflow import FixedLenFeature, VarLenFeature, parse_single_example, Session
    from tensorflow.data import Iterator, TFRecordDataset
except (AttributeError, ModuleNotFoundError):
    from tensorflow.compat.v1.python_io import TFRecordWriter
    from tensorflow.compat.v1.gfile import GFile
    from tensorflow.compat.v1 import FixedLenFeature, VarLenFeature, parse_single_example, Session
    from tensorflow.compat.v1.data import Iterator, TFRecordDataset


def create_label_map_pbtxt(label_map, output_path):
    """Generates a pbtxt file from a list of label info maps.

    Parameters
    ----------
    label_map: list(dict)
        A list of dictionaries, where each dictionary contains the name,
        display name, and id of a label.
    output_path: str
        Filepath to write the pbtxt file.
    """
    label_map_strs = [
        f"item {{\n  id: {label['id']}\n  name: '{label['name']}'\n  display_name: '{label['display_name']}'\n}}"
        for label in label_map
    ]
    label_map_str = '\n\n'.join(label_map_strs)
    with open(output_path, 'w') as fp:
        fp.write(label_map_str)


def int64_feature(value):
    """
    Note: Copied directly from https://github.com/tensorflow/models/blob/master/research/
        object_detection/utils/dataset_util.py
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    """
    Note: Copied directly from https://github.com/tensorflow/models/blob/master/research/
        object_detection/utils/dataset_util.py
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """
    Note: Copied directly from https://github.com/tensorflow/models/blob/master/research/
        object_detection/utils/dataset_util.py
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    """
    Note: Copied directly from https://github.com/tensorflow/models/blob/master/research/
        object_detection/utils/dataset_util.py
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    """
    Note: Copied directly from https://github.com/tensorflow/models/blob/master/research/
        object_detection/utils/dataset_util.py
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.

    Note: Copied directly from https://github.com/tensorflow/models/blob/master/research/
        object_detection/dataset_tools/tf_record_creation_util.py

    Parameters
    ----------
    exit_stack: context2.ExitStack
        A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
    base_path: str
        The base path for all shards
    num_shards: int
        The number of shards

    Returns
    -------
    tfrecords: list(tf.TFRecord)
        The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]
    tfrecords = [
        exit_stack.enter_context(TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]
    return tfrecords


def encode_object_detection_tf_example(example):
    """Creates a tf.Example proto from image and annotation data.

    Parameters
    ----------
    example: dict-like
        A key/value map containing bounding box annotations and image
        metadata.

    Returns
    -------
    tf_example: tf.Example
        The created tf.Example.
    """
    # Grab values from example
    image_filepath = example['image_filepath']
    image_id = example['image_id']
    xmins = example['wmin']
    xmaxs = example['wmax']
    ymins = example['hmin']
    ymaxs = example['hmax']
    classes_text = example['category_name']
    classes = example['category_id']
    # class_mask = example.get('cats_mask', None)
    class_ids_labeled = example.get('labeled_cat_ids', None)

    # Load image
    with GFile(image_filepath, 'rb') as fp:
        encoded_img = fp.read()
    encoded_img_io = io.BytesIO(encoded_img)
    img = Image.open(encoded_img_io)
    key = hashlib.sha256(encoded_img).hexdigest()

    # Get image related values
    width, height = img.size
    filename = os.path.split(image_filepath)[-1]
    image_format = img.format

    features = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_img),
        'image/format': bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature([txt.encode('utf8') for txt in classes_text]),
        'image/object/class/label': int64_list_feature(classes),
    }
    # if class_mask is not None:
    #     features['image/class_mask'] = int64_list_feature(class_mask)
    if class_ids_labeled is not None:
        features['image/class/labeled_classes'] = int64_list_feature(class_ids_labeled)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    return tf_example


def decode_object_detection_tf_example(example_proto):
    feature_types = {
        'image/height': FixedLenFeature([], tf.int64),
        'image/width': FixedLenFeature([], tf.int64),
        'image/filename': FixedLenFeature([], tf.string),
        'image/source_id': FixedLenFeature([], tf.string),
        'image/key/sha256': FixedLenFeature([], tf.string),
        'image/encoded': FixedLenFeature([], tf.string),
        'image/format': FixedLenFeature([], tf.string),
        # 'image/class_mask': VarLenFeature(tf.int64),
        'image/class/labeled_classes': VarLenFeature(tf.int64),
        'image/object/bbox/xmin': VarLenFeature(tf.float32),
        'image/object/bbox/ymin': VarLenFeature(tf.float32),
        'image/object/bbox/xmax': VarLenFeature(tf.float32),
        'image/object/bbox/ymax': VarLenFeature(tf.float32),
        'image/object/class/text': VarLenFeature(tf.string),
        'image/object/class/label': VarLenFeature(tf.int64),
    }

    example = parse_single_example(example_proto, features=feature_types)

    example = {
        'height': example['image/height'],
        'width': example['image/width'],
        'image_filename': example['image/filename'],
        'image_id': example['image/source_id'],
        'key': example['image/key/sha256'],
        'image_bytes': example['image/encoded'],
        'image_filetype': example['image/format'],
        'labeled_classes': example['image/class/labeled_classes'].values,
        'wmin': example['image/object/bbox/xmin'].values,
        'hmin': example['image/object/bbox/ymin'].values,
        'wmax': example['image/object/bbox/xmax'].values,
        'hmax': example['image/object/bbox/ymax'].values,
        'label_names': example['image/object/class/text'].values,
        'label_ids': example['image/object/class/label'].values,
    }

    return example


def write_examples_as_tfrecord(examples, output_filebase, example_encoder, num_shards=1):
    """Serialize examples as a TFRecord dataset.

    Note: Adapted from https://github.com/tensorflow/models/blob/master/research/
        object_detection/g3doc/using_your_own_dataset.md

    Parameters
    ----------
    examples: list(dict-like)
        A list of key/value maps, each map contains relevant info
        for a single data example.
    output_filebase: str
        The base path for all shards
    example_encoder: func
        A function that encodes an input example as a tf.Example.
    num_shards: int
        The number of shards to divide the examples among. If > 1 multiple
        tfrecord files will be created with names appended with a shard index.
    """
    if num_shards == 1:
        writer = TFRecordWriter(output_filebase)
        for example in tqdm(examples):
            tf_example = example_encoder(example)
            writer.write(tf_example.SerializeToString())
        writer.close()
    else:
        with contextlib.ExitStack() as tf_record_close_stack:
            output_tfrecords = open_sharded_output_tfrecords(
                tf_record_close_stack, output_filebase, num_shards)
            for index, example in tqdm(enumerate(examples), total=len(examples)):
                tf_example = example_encoder(example)
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def read_examples_from_tfrecord(tfrecord_filepath, example_decoder):
    """Load examples from a TFRecord file into memory.

    Examples are loaded as as list of dicts.

    Note: Adapted from https://github.com/tensorflow/models/blob/master/research/
        object_detection/g3doc/using_your_own_dataset.md

    Parameters
    ----------
    tfrecord_filepath: str
        A filepath where a serialized TFRecord is stored.
    example_decoder: func
        A function that decodes a serialized tf.Example.
    """
    dataset = TFRecordDataset(tfrecord_filepath)
    dataset = dataset.map(example_decoder).make_one_shot_iterator()
    example = dataset.get_next()

    decoded_examples = []
    try:
        if tf.executing_eagerly():
            while True:
                decoded_examples.append({k: v.numpy() for k, v in example.items()})
                example = dataset.get_next()
        else:
            with Session() as sess:
                while True:
                    decoded_examples.append(sess.run(example))
    except tf.errors.OutOfRangeError:
        pass

    return decoded_examples
