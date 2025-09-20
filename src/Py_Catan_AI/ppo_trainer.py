import numpy as np
import tensorflow as tf

class PPOTrainer:
    def __init__(self, rl_model, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, learning_rate=1e-4):
        """
        PPO trainer for RLDecisionModel.
        
        Args:
            rl_model: an instance of RLDecisionModel (must have .model attribute)
            clip_epsilon: PPO clipping range
            entropy_coef: weight for entropy bonus
            value_coef: weight for value loss
            learning_rate: Adam learning rate
        """
        self.rl_model = rl_model
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _ppo_loss(self, old_probs, actions, advantages):
        """Build the clipped PPO surrogate loss."""
        def loss_fn(y_true, y_pred):
            # y_true = one-hot actions
            # y_pred = current policy logits
            new_probs = tf.reduce_sum(y_true * tf.nn.softmax(y_pred), axis=1)
            ratio = new_probs / (old_probs + 1e-10)

            unclipped = ratio * advantages
            clipped = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

            entropy = -tf.reduce_mean(tf.nn.softmax(y_pred) * tf.nn.log_softmax(y_pred))
            return policy_loss - self.entropy_coef * entropy
        return loss_fn

    def train(self, dataset, epochs=10, batch_size=512):
        """
        Train the RL model using PPO.
        
        Args:
            dataset: dict from to_training_dataset
            epochs: number of passes over the dataset
            batch_size: minibatch size
        """
        x_inputs = dataset["x_inputs"]
        y_policy = dataset["y_policy"]
        y_value = dataset["y_value"]
        adv = dataset["adv"]

        # === Step 1: compute old action probabilities under current policy ===
        old_logits, old_values = self.rl_model.predict_logits_and_value(x_inputs, verbose=0)

        # Ensure numeric type
        old_logits = np.array(old_logits, dtype=np.float32)

        # Compute old action probabilities under the old policy
        old_probs = np.sum(y_policy * tf.nn.softmax(old_logits, axis=1).numpy(), axis=1)

        # === Step 2: build losses ===
        policy_loss = self._ppo_loss(old_probs, y_policy, adv)
        value_loss = tf.keras.losses.MeanSquaredError()

        self.rl_model.model.compile(
            optimizer=self.optimizer,
            loss={"output": policy_loss, "value_output": value_loss},
            loss_weights={"output": 1.0, "value_output": self.value_coef}
        )

        # === Step 3: fit ===
        history = self.rl_model.model.fit(
            x=x_inputs,
            y={"output": y_policy, "value_output": y_value},
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=1
        )
        return history
