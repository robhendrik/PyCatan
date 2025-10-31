"""RL decision model utilities.

This module provides a simple Keras-based policy+value network wrapper used
by the project for inference and weight transfer. The network architecture
matches the original DecisionModel's policy head so weights can be reused.

Primary class
        - RLDecisionModel: builds, queries, and initializes a policy+value Keras model.

Notes
        - The class expects a `structure` object exposing `vector_indices` and
            `mask_space_length` used for shaping inputs/outputs.
        - The model produces two outputs: a masked policy probability vector named
            "output" and a scalar value prediction named "value_output".

Author:
        Rob Hendriks

Version:
        1.0.0
"""

from tabnanny import verbose
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Concatenate, Lambda, Dense, Activation
import tensorflow as tf

class RLDecisionModel:
    """Keras policy+value model wrapper used for RL decisions.

    The wrapper constructs an embedding-based trunk that feeds a policy head
    (masked softmax) and a scalar value head. It exposes helpers for:
    - building/resetting the model (`reset_model_to_new`)
    - selecting an action from a single state (`get_action`)
    - running batched predictions for probabilities/values (`predict`,
      `predict_probabilities`, `predict_logits_and_value`)
    - initializing weights from another compatible model (`init_from_existing`)

    Args:
        structure: Board structure object providing `vector_indices` and
            `mask_space_length` used to shape inputs and outputs.
        explore (bool): Whether to enable exploration behavior in
            `get_action` (defaults to False).
    """
    def __init__(self, structure, explore: bool = False):
        self.structure = structure
        self.model = None
        self.explore = explore
        self.reset_model_to_new()
        self.this_model_has_probs_as_output_and_not_logits = True  # for PPO compatibility
        self.name = "RLDecisionModel_named_by_user"

    def get_model(self):
        """Return the underlying compiled Keras `Model`.

        Returns:
            keras.models.Model: The compiled Keras model with outputs
            `output` (policy probabilities) and `value_output` (scalar value).
        """
        return self.model

    def reset_model_to_new(self):
        """ Create a new untrained model at self.model
        
        Build a policy+value network where the policy branch matches the original DecisionModel,
        so weights can be transferred directly. Adds a value head in parallel.
        """
        # Inputs
        input1_layer = Input(shape=(len(self.structure.vector_indices['nodes']),), dtype='int32', name='input1')
        input2_layer = Input(shape=(len(self.structure.vector_indices['edges']),), dtype='int32', name='input2')
        input3_layer = Input(shape=(len(self.structure.vector_indices['hands']),), dtype='float32', name='input3')

        # Embeddings
        embed1 = Embedding(input_dim=9, output_dim=4, name='embed1')(input1_layer)
        embed2 = Embedding(input_dim=6, output_dim=3, name='embed2')(input2_layer)

        embed1_flat = Reshape((len(self.structure.vector_indices['nodes']) * 4,), name='reshape1')(embed1)
        embed2_flat = Reshape((len(self.structure.vector_indices['edges']) * 3,), name='reshape2')(embed2)

        # Normalize input3
        normalized_input3 = Lambda(lambda x: x / 10.0, name='normalize_input3')(input3_layer)

        # Concatenate
        combined = Concatenate(name='concat')([embed1_flat, embed2_flat, normalized_input3])

        # Shared trunk (same as original)
        dense1 = Dense(128, activation='relu', name="dense_12")(combined)
        dense2 = Dense(64, activation='relu', name="dense_13")(dense1)

        # Policy head (identical naming as old model)
        logits = Dense(self.structure.mask_space_length, name='logits')(dense2)
        mask_input = Input(shape=(self.structure.mask_space_length,), dtype='float32', name='mask_input')
        large_negative = -1e9
        masked_logits = Lambda(lambda x: x[0] + (1.0 - x[1]) * large_negative, name="lambda_6")([logits, mask_input])
        output = Activation('softmax', name='output')(masked_logits)

        # Value head (new)
        value_output = Dense(1, activation='linear', name='value_output')(dense2)

        # Build model
        self.model = Model(
            inputs=[input1_layer, input2_layer, input3_layer, mask_input],
            outputs={"output": output, "value_output": value_output}   # dict instead of list
        )
        # model outputs are a dict: outputs={"output": output, "value_output": value_output}
        self.model.compile(
            optimizer='adam',
            loss={'output': 'categorical_crossentropy', 'value_output': 'mse'},
            loss_weights={'output': 1.0, 'value_output': 0.5}
        )
    
    def predict(self, vector_dataframe, mask_dataframe):
        """
        Returns both action probabilities and value estimate for a batch.
        """
        if not self.this_model_has_probs_as_output_and_not_logits:
            raise ValueError("Model must output probabilities, not logits, for this method.")
        x1 = vector_dataframe.iloc[:, self.structure.vector_indices['nodes']].values.astype(np.int32)
        x2 = vector_dataframe.iloc[:, self.structure.vector_indices['edges']].values.astype(np.int32)
        x3 = vector_dataframe.iloc[:, self.structure.vector_indices['hands']].values.astype(np.float32)
        mask = mask_dataframe.values.astype(np.float32)

        preds = self.model.predict([x1, x2, x3, mask], verbose=0)
        policy_probs = preds["output"]         # (B, A) masked softmax probs
        state_values = preds["value_output"]   # (B, 1)
        return policy_probs, state_values

    def get_action(self, vector_row, mask_row, include_best_action=False, include_old_action_prob=False) -> tuple:
        """
        Returns a tuple with action, probabilities and values given one board state and mask.
        If include_best_action is True, also returns the best action (argmax prob).

        Args:
            vector_row: np.ndarray
            mask_row: np.ndarray
            include_best_action: bool

        Returns:
            If include_best_action is False:
                (action: int, probs: np.ndarray, value: float)
            If include_best_action is True:
                (action: int, probs: np.ndarray, value: float, best_action: int)
        """
        explore = self.explore
        if not self.this_model_has_probs_as_output_and_not_logits:
            raise ValueError("Model must output probabilities, not logits, for this method.")
        # --- Ensure inputs are numeric numpy arrays ---
        vector_row = np.array(vector_row, dtype=np.float32)
        mask_row = np.array(mask_row, dtype=np.float32)

        # Extract inputs for the model
        x1 = np.expand_dims(vector_row[self.structure.vector_indices['nodes']], axis=0)
        x2 = np.expand_dims(vector_row[self.structure.vector_indices['edges']], axis=0)
        x3 = np.expand_dims(vector_row[self.structure.vector_indices['hands']], axis=0)
        mask = np.expand_dims(mask_row.astype(np.float32), axis=0)

        # preds: dict outputs (policy probs already masked+softmaxed, value)
        preds = self.model.predict([x1, x2, x3, mask], verbose=0)
        policy_probs = preds["output"]         # shape (B, A) — probabilities
        value_pred   = preds["value_output"]   # shape (B, 1)

        # Expect batch size 1 during action selection
        if policy_probs.ndim != 2 or policy_probs.shape[0] != 1:
            raise ValueError(f"Expected policy_probs shape (1, A), got {policy_probs.shape}")
        if value_pred.ndim != 2 or value_pred.shape != (1, 1):
            raise ValueError(f"Expected value_pred shape (1, 1), got {value_pred.shape}")

        # Value for return statement
        value = float(value_pred[0, 0])

        probs = np.asarray(policy_probs[0], dtype=np.float32)   # (A,)

        # Strict, NaN-safe mask
        m = np.asarray(mask_row, dtype=float)
        mask_vec = np.isfinite(m) & (m > 0.5)

        # --- checks you already have ---
        if not np.isfinite(probs).all():
            raise ValueError("Probabilities contain NaN/Inf.")
        if (probs < -1e-12).any():
            raise ValueError("Probabilities contain negative values.")
        illegal_mass = float(np.where(~mask_vec, probs, 0.0).sum())
        if illegal_mass > 1e-8:
            raise ValueError(f"Illegal actions have non-zero probability mass: {illegal_mass:.3e}")
        if not np.isclose(float(probs.sum()), 1.0, atol=1e-6):
            raise ValueError("Probabilities must sum to ~1.")

        legal_actions = np.where(mask_vec)[0]
        if legal_actions.size == 0:
            raise ValueError("No legal actions in mask.")

        def prob_from_full(a: int) -> float:
            p = float(probs[a])
            if not np.isfinite(p):
                raise ValueError("Selected action prob is NaN/Inf.")
            return max(p, 1e-8)  # clamp for downstream stability

        if len(legal_actions) == 1:
            action = int(legal_actions[0])
            best_action = action
            old_action_prob = 1.0

        elif len(legal_actions) == 2:
            la = legal_actions
            p2 = probs[la].astype(float).copy()
            s2 = p2.sum()
            # must renormalize these two; guard zero/NaN sum
            if not np.isfinite(s2) or s2 <= 0.0:
                p2[:] = 0.5
            else:
                p2 /= s2

            best_action = int(la[np.argmax(p2)])
            if explore:
                # (optional) exploration tweak on the two-entry, already normalized p2
                if p2.max() > 0.90:
                    i = int(np.argmax(p2))
                    p2[i] = 0.8
                    p2[1 - i] = 0.2
                action = int(np.random.choice(la, p=p2))
            else:
                action = best_action

            old_action_prob = prob_from_full(action)

        else:
            la = legal_actions
            if explore:
                pL = probs[la].astype(float).copy()
                sL = pL.sum()
                if not np.isfinite(sL) or sL <= 0.0:
                    pL[:] = 1.0 / len(la)
                else:
                    pL /= sL
                action = int(np.random.choice(la, p=pL))
                best_action = int(la[np.argmax(pL)])
            else:
                j = int(np.argmax(probs[la]))
                action = int(la[j])
                best_action = action

            old_action_prob = prob_from_full(action)

        # final legality check
        if not mask_vec[action]:
            raise ValueError(f"Chosen action {action} is illegal under mask.")

        # Quick check
        assert mask_vec[action] == True, "Chosen action must be legal."

        # assert old_action_prob is not NaN or Inf
        if not np.isfinite(old_action_prob):
            print(f"old_action_prob: {old_action_prob}")
            print(f"probs: {probs}")
            print(f"mask_vec: {mask_vec}")
            raise ValueError("Old action probability is NaN/Inf.")
        
        # Return(s)
        if not include_old_action_prob:
            if not include_best_action:
                return action, probs, value
            else:
                return action, probs, value, best_action
        else:
            if not include_best_action:
                return action, probs, value, old_action_prob
            else:
                return action, probs, value, best_action, old_action_prob



    def init_from_existing(self, existing_model, verbose=False):
        """
        Initialize this model's weights from an existing Keras model
        (e.g., a previously trained DecisionModel or RLDecisionModel).
        Tries to match by layer name.
        """
        existing_layers = {layer.name: layer for layer in existing_model.layers}
        transferred, skipped = [], []

        for layer in self.model.layers:
            if layer.name in existing_layers:
                try:
                    layer.set_weights(existing_layers[layer.name].get_weights())
                    transferred.append(layer.name)
                except Exception:
                    skipped.append(layer.name)
            else:
                skipped.append(layer.name)

        # Print summary of 
        if verbose:
            for name in transferred:
                print(f"✅ Transferred weights for layer {name}")
            for name in skipped:
                print(f"⚠️ Skipping layer {name} (no matching layer in source model)")

        return self
    
    def predict_logits_and_value(self, x_inputs, verbose=0):
        """
        Runs the model and returns (policy_logits, value_preds), where:
        - If the model outputs probabilities, we convert to masked log-probs (valid logits).
        - If the model outputs logits, we pass them through.
        This keeps PPO callers correct in both cases.

        Args:
            x_inputs: [x1, x2, x3, mask]
        Returns:
            policy_logits: (B, A) float32
            value_preds:   (B, 1) float32
        """

        preds = self.model.predict(x_inputs, verbose=verbose)

        # Unpack dict or list
        if isinstance(preds, dict):
            policy_out = preds["output"]
            value_preds = preds["value_output"]
        elif isinstance(preds, (list, tuple)) and len(preds) == 2:
            policy_out, value_preds = preds
        else:
            raise ValueError(f"Unexpected predict() output type: {type(preds)}")

        policy_out = np.asarray(policy_out, dtype=np.float32)   # (B, A)
        value_preds = np.asarray(value_preds, dtype=np.float32) # (B, 1)

        # Get mask from inputs (B, A). No re-masking; used only for checks/transform.
        mask_arr = np.asarray(x_inputs[-1], dtype=np.float32)
        if mask_arr.shape != policy_out.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} != policy shape {policy_out.shape}")
        mask_bool = mask_arr > 0.5

        # Detect if model output is probs (row sums ~ 1) or logits
        row_sums = policy_out.sum(axis=1)
        in_01 = (policy_out >= -1e-6) & (policy_out <= 1.0 + 1e-6)
        is_probs = np.all(in_01) and np.allclose(row_sums, 1.0, atol=1e-5)

        if is_probs:
            # Fail fast if illegal actions have probability mass
            illegal_mass = float((policy_out * (~mask_bool)).sum())
            if illegal_mass > 1e-8:
                raise ValueError(f"Illegal actions have non-zero probability mass: {illegal_mass:.3e}")

            # Convert probs -> masked logits (log-probs with -1e9 for illegal)
            eps = 1e-8
            policy_logits = np.where(
                mask_bool,
                np.log(np.clip(policy_out, eps, 1.0)),   # log-probs; softmax(log p) == p
                -1e9
            ).astype(np.float32)
        else:
            if self.this_model_has_probs_as_output_and_not_logits:
                raise ValueError("Model claims to output probabilities, but they do not sum to 1.0.")
            # Already logits; just pass through
            policy_logits = policy_out

        return policy_logits, value_preds
    
    def predict_probabilities(self, x_inputs, verbose=0, eps_illegal=1e-8, eps_sum=1e-6):
        """Run the model and return validated action probabilities and values.

        This helper accepts either logits or probability outputs from the model.
        If logits are provided they are converted to probabilities with a
        numerically stable softmax. The function performs strict validations
        (shape, normalization and illegal-action checks) and raises on errors.

        Args:
            x_inputs (list|tuple): Model inputs [x1, x2, x3, mask].
            verbose (int): Forward-pass verbosity forwarded to Keras.
            eps_illegal (float): Tolerance for illegal-action mass.
            eps_sum (float): Tolerance for row-sum normalization.

        Returns:
            (np.ndarray, np.ndarray): (probs, value_preds) where `probs` has
                shape (B, A) and `value_preds` has shape (B, 1).
        """

        import numpy as np
        # confirm mask is ok
        mask = x_inputs[-1]
        if not np.all(np.isfinite(mask)):
            raise ValueError("Mask is invalid.")
        if not np.all((mask >= -1e-6) & (mask <= 1.0 + 1e-6)):
            raise ValueError("Mask entries must be in [0,1].")
        # run the model
        preds = self.model.predict(x_inputs, verbose=verbose)
        if isinstance(preds, dict):
            policy_out = preds["output"]
            value_preds = preds["value_output"]
        elif isinstance(preds, (list, tuple)) and len(preds) == 2:
            policy_out, value_preds = preds
        else:
            raise ValueError(f"Unexpected predict() output type: {type(preds)}")

        policy_out = np.asarray(policy_out, dtype=np.float32)   # (B, A)
        value_preds = np.asarray(value_preds, dtype=np.float32) # (B, 1)

        # Basic shape checks
        if policy_out.ndim != 2:
            raise ValueError(f"Expected 2D policy output, got {policy_out.shape}")
        if value_preds.ndim != 2 or value_preds.shape[1] != 1:
            raise ValueError(f"Expected value shape (B,1), got {value_preds.shape}")

        # Mask for checks only (do NOT re-mask)
        mask_arr = np.asarray(x_inputs[-1], dtype=np.float32)
        if mask_arr.shape != policy_out.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} != policy shape {policy_out.shape}")
        mask_bool = mask_arr > 0.5

        # Detect probs vs logits
        row_sums = policy_out.sum(axis=1)
        in_01 = (policy_out >= -1e-6) & (policy_out <= 1.0 + 1e-6)
        is_probs = np.all(in_01) and np.allclose(row_sums, 1.0, atol=1e-5)

        if not is_probs:
            print("Warning: Model outputs logits, should not be the case.")

        probs = policy_out

        # Strict validations (do not modify probs)
        if not np.isfinite(probs).all() or np.any(probs < -2e-6):
            print(f"Warning! Probabilities contain NaN/Inf/negatives. {np.any(probs < -2e-6)}, {np.any(probs > 1 + 2e-6)}.")
        bad_sum = np.where(~np.isclose(probs.sum(axis=1), 1.0, atol=eps_sum))[0]
        if bad_sum.size:
            raise ValueError(f"Rows not normalized to 1 (examples: {bad_sum[:10].tolist()}).")
        illegal_mass = (probs * (~mask_bool)).sum(axis=1)
        bad_illegal = np.where(illegal_mass > eps_illegal)[0]
        if bad_illegal.size:
            raise ValueError(
                f"Probability mass on illegal actions (rows {bad_illegal[:10].tolist()}), "
                f"max illegal mass={illegal_mass[bad_illegal].max():.3e}"
            )

        return probs.astype(np.float32), value_preds.astype(np.float32)


