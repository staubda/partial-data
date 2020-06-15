import sys

import numpy as np

from tensorflow import Session

from partial_data.losses import WeightedSigmoidClassificationLoss

sys.path.append('/home/david/github_repos/models/research/')


def test_per_example_class_mask():
    batch_size = 2
    num_anchors = 3
    num_classes = 7

    np.random.seed(0)
    pred_tensor = 0.3 * np.ones([batch_size, num_anchors, num_classes])
    targ_tensor = 0.7 * np.ones([batch_size, num_anchors, num_classes])
    wts = np.ones([batch_size, num_anchors, 1]).astype(np.float64)
    class_inds = np.array([0, 3, 5])
    class_msk = np.array([[1, 1, 0, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 1]])

    # Class mask should override weights
    loss = WeightedSigmoidClassificationLoss()._compute_loss(pred_tensor, targ_tensor, wts, class_inds, class_msk)
    with Session() as sess:
        loss_vals = sess.run(loss)
    assert (loss_vals[0, 0, :][class_msk[0, :].astype(bool)] == 0).sum() == 0
    assert (loss_vals[1, 0, :][class_msk[1, :].astype(bool)] == 0).sum() == 0
    assert (loss_vals[0, 0, :][~class_msk[0, :].astype(bool)] > 0).sum() == 0
    assert (loss_vals[1, 0, :][~class_msk[1, :].astype(bool)] > 0).sum() == 0
