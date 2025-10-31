"""PPO training utilities and Keras-compatible loss functions.

This module provides loss factories and a small trainer wrapper used by the
project's PPO training flow. It includes:

- numerically-stable PPO loss variants that respect legal-action masks,
- an entropy-augmented loss with optional hinge behavior,
- a small `PPOTrainer` class that compiles a Keras model with the PPO loss
    and exposes a `train()` method which performs KL-based early stopping
    and learning-rate backoff.

Author: Rob Hendriks
Version: 1.0.0
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

def _get_lr(opt):
    """Return the current learning rate from an optimizer.

    This helper handles several possible optimizer learning-rate
    representations: plain float, TensorFlow Variables, or schedule
    objects that expose `.numpy()` or require `tf.keras.backend.get_value`.
    """
    lr = opt.learning_rate
    # tf.Variable or tensor-like
    if hasattr(lr, "numpy"):
        return float(lr.numpy())
    try:
        import tensorflow as tf
        return float(tf.keras.backend.get_value(lr))
    except Exception:
        return float(lr)  # plain float

def _set_lr(opt, new_lr):
    """Set a new learning rate on an optimizer.

    Works with Variable-style learning rates (has `.assign`) or plain
    numeric values on the optimizer object.
    """
    lr = opt.learning_rate
    # tf.Variable style
    if hasattr(lr, "assign"):
        lr.assign(new_lr)
    else:
        # plain float / schedule replacement
        opt.learning_rate = float(new_lr)



@tf.keras.utils.register_keras_serializable()
def make_ppo_loss_with_entropy(entropy_coef, 
                               clip_ratio, 
                               num_actions, 
                               use_entropy_hinge=False, 
                               entropy_target=0.0, 
                               hinge_weight=1.0,
                               bc_beta=0.0
                               ):
    """
    y_true is packed as:
      [ one_hot(num_actions), advantage(1), old_action_prob(1), legal_mask(num_actions) ]
    y_pred is probs over all actions (softmax output).
    - Entropy is computed on masked & renormalized probs over *legal* actions.
    - Ratio uses the taken-action prob under new/old *legal* distributions.
    """
    eps = 1e-8

    def loss(y_true, y_pred):
        one_hot, advantages, old_p, legal_mask = tf.split(
            y_true, [num_actions, 1, 1, num_actions], axis=1
        )

        # New policy probs (clip for safety)
        p_all = tf.clip_by_value(y_pred, eps, 1.0)

        # Mask to legal actions and renormalize
        masked = p_all * legal_mask
        z = tf.reduce_sum(masked, axis=1, keepdims=True) + eps
        p_legal = masked / z

        # Prob of the taken action under *new* legal distribution
        new_p = tf.reduce_sum(one_hot * p_legal, axis=1, keepdims=True)

        # PPO ratio: require that old_p is the *behavior* prob of the taken action
        ratio   = new_p / (old_p + eps)
        clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

        # Surrogate (advantages should be z-scored per round & clipped outside)
        surr = tf.minimum(ratio * advantages, clipped * advantages)

        # Entropy over the legal distribution only: H = -sum p log p
        entropy = -tf.reduce_sum(p_legal * tf.math.log(p_legal + eps), axis=1)

        # Optional entropy hinge to actively push up when too low
        if use_entropy_hinge and entropy_target > 0.0:
            shortfall = tf.nn.relu(entropy_target - entropy)
            entropy_bonus = entropy_coef * entropy + hinge_weight * shortfall * shortfall
        else:
            entropy_bonus = entropy_coef * entropy


        # CE(one_hot, p_legal) = -log(new_p)
        bc_loss = -tf.math.log(new_p + eps)  # shape [batch, 1]

        # Minimize: -(surr + entropy_bonus) + β * bc_loss
        total = -(surr + entropy_bonus) + bc_beta * bc_loss
        return tf.reduce_mean(total)

    return loss

@tf.keras.utils.register_keras_serializable()
def ppo_loss_from_probs(y_true, y_pred, clip_ratio=0.05, num_actions=241):
    """
    PPO clipped surrogate loss where y_pred are ALREADY probabilities (masked softmax).
    y_true is packed: [one_hot_actions, advantages, old_action_prob].
    """
    one_hot, advantages, old_p = tf.split(y_true, [num_actions, 1, 1], axis=1)

    # Use probs directly (clip for numerical safety; do not renormalize here)
    new_p = tf.reduce_sum(one_hot * tf.clip_by_value(y_pred, 1e-8, 1.0), axis=1, keepdims=True)

    ratio   = new_p / (old_p + 1e-8)
    clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    obj     = tf.minimum(ratio * advantages, clipped * advantages)
    return -tf.reduce_mean(obj)


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
    """Small wrapper to compile and train a Keras-based RL decision model with PPO.

    This class expects `rl_model` to expose a Keras `model` attribute with two
    outputs: the policy head named `output` and a scalar value head named
    `value_output`. The constructor compiles the model with a PPO-style
    surrogate loss (policy) and an MSE loss for the value head.

    Args:
        rl_model: object exposing a compiled Keras model as `rl_model.model`.
        entropy_coef: coefficient multiplying the entropy bonus.
        value_coef: weight for the value loss in the final loss weights.
        learning_rate: initial learning rate for the Adam optimizer.
        clipnorm: gradient clipping by norm for the optimizer.
        epsilon: Adam numerical epsilon.
        clip_ratio: PPO clip ratio (epsilon for the surrogate objective).
    """
    def __init__(self, rl_model, entropy_coef, value_coef, learning_rate, clipnorm, epsilon, clip_ratio):
        self.rl_model = rl_model
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.epsilon = epsilon
        self.clip_ratio = clip_ratio


        self.rl_model.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, 
                                            clipnorm=self.clipnorm,
                                            epsilon=self.epsilon),
            loss={
                #"output": ppo_loss_from_probs,   # ← consumes probs
                "output": make_ppo_loss_with_entropy(
                    entropy_coef=self.entropy_coef, 
                    clip_ratio=self.clip_ratio, 
                    num_actions=241,
                    bc_beta=0.00  # try 0.05–0.1
                ),
                "value_output": "mse",
            },
            loss_weights={"output": 1.0, "value_output": self.value_coef},
        )


    def train(self, dataset, epochs, batch_size, target_kl, lr_backoff):
        """
        Train PPO using the provided dataset.

        Args:
            dataset: dict from to_training_dataset()
            epochs: number of epochs
            batch_size: minibatch size
        """
        x_inputs = dataset["x_inputs"]          # dataset["x_inputs"] = [x1, x2, x3, masks]
        y_policy = dataset["y_policy"]          # [batch, num_actions]
        y_value = dataset["y_value"]            # [batch, 1]
        adv = dataset["adv"]                    # [batch,]
        old_probs = dataset["old_action_probs"] # [batch,]
        masks = x_inputs[3].astype(np.float32)   # [batch, num_actions]

        # === Pack policy + adv + old_probs into one tensor ===
        y_true = np.concatenate(
            [
                y_policy,                       # one-hot actions [batch, num_actions]
                adv.reshape(-1, 1),             # [batch, 1]
                old_probs.reshape(-1, 1),       # [batch, 1]
                masks                           # [batch, num_actions] mask input
            ],
            axis=1
        ).astype(np.float32)
        
        assert (old_probs >= 0.0).all(), "old action probs has negative values"
        assert (old_probs <= 1.0).all(), "old action probs has values > 1.0"

        # --- Callback to measure KL each epoch and early stop / backoff LR ---
        # --- drop-in replacement for KLEarlyStop inside PPOTrainer.train ---
        class KLEarlyStop(keras.callbacks.Callback):
            """
            Computes cumulative KL( p_old || p_new ) to the *start-of-round model policy*
            over the full batch of states (with legal-action masking if available),
            and early-stops + backs off LR when target is exceeded.
            """
            def __init__(self, model_ref, x, one_hot_actions, p_old, target, backoff, eps=1e-8):
                super().__init__()
                self.model_ref = model_ref
                self.x = x                    # expected: [x1, x2, x3, masks]
                self.target = target
                self.backoff = backoff
                self.eps = eps
                self.tripped = False
                self._p_old = None            # frozen model policy at start of training
                # keep signature compatible; we don't use one_hot_actions or p_old here
                self._unused_one_hot = one_hot_actions
                self._unused_p_old = p_old

            def _predict_policy(self):
                outs = self.model_ref.predict(self.x, verbose=0)
                # handle list/tuple/dict/single-output
                if isinstance(outs, (list, tuple)):
                    policy = outs[0]
                elif isinstance(outs, dict):
                    policy = outs["output"]
                else:
                    policy = outs
                return np.asarray(policy, dtype=np.float64)

            def _ensure_masked_and_normalized(self, p):
                """
                Ensure probabilities are masked to legal actions and row-normalized.
                If masks are provided as x[3], use them. Otherwise assume p already legal.
                """
                try:
                    masks = np.asarray(self.x[3], dtype=np.float64)
                    if masks.ndim == 2 and masks.shape == p.shape:
                        p_masked = np.clip(p, self.eps, 1.0) * masks
                        z = p_masked.sum(axis=1, keepdims=True) + self.eps
                        return p_masked / z
                except Exception:
                    pass
                # fallback: clip & renorm (assume already legal)
                p = np.clip(p, self.eps, 1.0)
                z = p.sum(axis=1, keepdims=True) + self.eps
                return p / z

            def on_train_begin(self, logs=None):
                p = self._predict_policy()
                self._p_old = self._ensure_masked_and_normalized(p)
                print("[KL Callback] init: KL_to_start=0.0000 (frozen p_old)")

            def on_epoch_end(self, epoch, logs=None):
                p_new_raw = self._predict_policy()
                p_new = self._ensure_masked_and_normalized(p_new_raw)

                # KL(p_old || p_new) per sample, then mean over batch
                kl_per = np.sum(self._p_old * (np.log(self._p_old + self.eps) - np.log(p_new + self.eps)), axis=1)
                measured_kl = float(np.mean(kl_per))

                print(f"[KL Callback] epoch {epoch} KL_to_start={measured_kl:.4f} (target {self.target})")

                if measured_kl > self.target and not self.tripped:
                    opt = self.model_ref.optimizer
                    old = _get_lr(opt)
                    new = old * self.backoff
                    _set_lr(opt, new)
                    print(f"[KL Callback] KL>{self.target:.3f} → early stop, LR {old:.3e}→{new:.3e}")
                    self.model_ref.stop_training = True
                    self.tripped = True


        kl_cb = KLEarlyStop(
            model_ref=self.rl_model.model,
            x=x_inputs,
            one_hot_actions=y_policy,
            p_old=old_probs,
            target=target_kl,     # start with 0.1–0.2 if your logged KLs are ~1–2
            backoff=lr_backoff,
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
            verbose=1,
            callbacks=[kl_cb]   # ← add it here
        )
        return history



