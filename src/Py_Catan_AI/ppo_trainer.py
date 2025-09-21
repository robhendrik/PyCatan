import tensorflow as tf
from tensorflow import keras
import numpy as np


@tf.keras.utils.register_keras_serializable()
def ppo_loss(y_true, y_pred, clip_ratio=0.2, num_actions=241):
    """
    PPO clipped surrogate loss.
    y_true is packed: [one_hot_actions, advantages, old_probs].
    """
    # Split into pieces
    one_hot_actions, advantages, old_probs = tf.split(
        y_true, [num_actions, 1, 1], axis=1
    )

    # Current probs
    new_probs = tf.reduce_sum(one_hot_actions * tf.nn.softmax(y_pred), axis=1, keepdims=True)

    # Ratio
    ratio = new_probs / (old_probs + 1e-8)

    # Clipped surrogate objective
    clipped_adv = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    surrogate = ratio * advantages
    policy_loss = -tf.reduce_mean(tf.minimum(surrogate, clipped_adv))

    return policy_loss


class PPOTrainer:
    def __init__(self, rl_model, clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5, learning_rate=1e-4):
        self.rl_model = rl_model
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.learning_rate = learning_rate


        # Compile using ppo_loss directly (no lambda!)
        self.rl_model.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                "output": ppo_loss,          # decorated, serializable
                "value_output": "mse",
            },
            loss_weights={"output": 1.0, "value_output": self.value_coef},
        )

    def train(self, dataset, epochs=10, batch_size=256):
        """
        Train PPO using the provided dataset.

        Args:
            dataset: dict from to_training_dataset()
            epochs: number of epochs
            batch_size: minibatch size
        """
        x_inputs = dataset["x_inputs"]
        y_policy = dataset["y_policy"]          # [batch, num_actions]
        y_value = dataset["y_value"]            # [batch, 1]
        adv = dataset["adv"]                    # [batch,]
        
        # === Compute old_probs from current policy ===
        old_logits, old_values = self.rl_model.predict_logits_and_value(x_inputs, verbose=0)
        old_probs = np.sum(y_policy * tf.nn.softmax(old_logits).numpy(), axis=1)  # [batch]

        # === Pack policy + adv + old_probs into one tensor ===
        y_true = np.concatenate(
            [
                y_policy,                       # one-hot actions [batch, num_actions]
                adv.reshape(-1, 1),             # [batch, 1]
                old_probs.reshape(-1, 1)        # [batch, 1]
            ],
            axis=1
        )

        # === Fit ===
        history = self.rl_model.model.fit(
            x=x_inputs,
            y={
                "output": y_true,               # PPO custom loss expects [num_actions+2]
                "value_output": y_value,        # standard MSE loss
            },
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=1
        )
        return history



