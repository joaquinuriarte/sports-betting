import tensorflow as tf


def mse_plus_hinge_margin_loss(alpha=1.0):
    def loss_fn(y_true, y_pred):
        # 1) Mean Squared Error
        mse_value = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2) The margins
        actual_margin = y_true[:, 0] - y_true[:, 1]
        pred_margin = y_pred[:, 0] - y_pred[:, 1]

        # 3) Hinge term
        hinge_losses = tf.nn.relu(-actual_margin * pred_margin)
        margin_loss = tf.reduce_mean(hinge_losses)

        return mse_value + alpha * margin_loss
    return loss_fn


def margin_sensitive_hinge_loss(delta=5.0):
    def loss_fn(y_true, y_pred):
        # 1) Mean Squared Error
        mse_value = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2) The margins
        actual_margin = y_true[:, 0] - y_true[:, 1]
        pred_margin = y_pred[:, 0] - y_pred[:, 1]

        product = actual_margin * pred_margin
        # margin-sensitive hinge
        losses = tf.nn.relu(delta - product)
        return mse_value + tf.reduce_mean(losses)
    return loss_fn
