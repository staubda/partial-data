"""TF Object Detection losses updated to handle per example class masks.
"""

import tensorflow as tf

from object_detection.utils import ops
from object_detection import losses


class WeightedSigmoidClassificationLoss(losses.Loss):
    """Sigmoid cross entropy classification loss function."""

    def _compute_loss(
            self,
            prediction_tensor,
            target_tensor,
            weights,
            class_indices=None,
            class_mask=None
    ):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape, either [batch_size, num_anchors,
            num_classes] or [batch_size, num_anchors, 1]. If the shape is
            [batch_size, num_anchors, 1], all the classses are equally weighted.
          class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.
          class_mask: A boolean tensor of shape [batch_size, num_classes] indicating
            which classes to compute loss for for each example in the batch.


        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        if class_mask is not None:
            class_mask_full = tf.cast(tf.reshape(
                tf.tile(class_mask, [1, prediction_tensor.shape[1]]),
                prediction_tensor.shape
            ), tf.int8)
        elif class_indices is not None:
            class_mask_full = tf.reshape(
                ops.indices_to_dense_vector(class_indices, tf.shape(prediction_tensor)[2]),
                [1, 1, -1]
            )

        if class_mask_full is not None:
            weights = tf.cast(weights * class_mask_full, tf.float64)

        per_entry_cross_ent = (
            tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor)
        )
        weighted_per_entry_cross_ent = per_entry_cross_ent * weights

        return weighted_per_entry_cross_ent
