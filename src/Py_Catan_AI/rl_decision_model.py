# rl_decision_model.py
from tabnanny import verbose
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Concatenate, Lambda, Dense, Activation
import tensorflow as tf

class RLDecisionModel:
    def __init__(self, structure):
        self.structure = structure
        self.model = None
        self.reset_model_to_new()

    def get_model(self):
        return self.model

    def reset_model_to_new(self):
        """
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

        self.model.compile(
            optimizer='adam',
            loss={'output': 'categorical_crossentropy', 'value_output': 'mse'},
            loss_weights={'output': 1.0, 'value_output': 0.5}
        )

    

    def predict(self, vector_dataframe, mask_dataframe):
        """
        Returns both action probabilities and value estimate for a batch.
        """
        x1 = vector_dataframe.iloc[:, self.structure.vector_indices['nodes']].values.astype(np.int32)
        x2 = vector_dataframe.iloc[:, self.structure.vector_indices['edges']].values.astype(np.int32)
        x3 = vector_dataframe.iloc[:, self.structure.vector_indices['hands']].values.astype(np.float32)
        mask = mask_dataframe.values.astype(np.float32)

        policy_probs, state_values = self.model.predict([x1, x2, x3, mask], verbose=0)
        return policy_probs, state_values

    def get_action(self, vector_row, mask_row, explore=True):
        """
        Returns a single action given one board state and mask.
        - vector_row: 1D row from vector DataFrame or numpy array
        - mask_row: 1D row from mask DataFrame or numpy array
        - explore=True: sample from distribution; if False: pick greedy action
        """
        # --- Ensure inputs are numeric numpy arrays ---
        vector_row = np.array(vector_row, dtype=np.float32)
        mask_row = np.array(mask_row, dtype=np.float32)

        # Extract inputs for the model
        x1 = np.expand_dims(vector_row[self.structure.vector_indices['nodes']], axis=0)
        x2 = np.expand_dims(vector_row[self.structure.vector_indices['edges']], axis=0)
        x3 = np.expand_dims(vector_row[self.structure.vector_indices['hands']], axis=0)
        mask = np.expand_dims(mask_row.astype(np.float32), axis=0)

        # --- Forward pass (dict outputs now) ---
        preds = self.model.predict([x1, x2, x3, mask], verbose=0)
        policy_logits = preds["output"]
        value_pred = preds["value_output"]

        # --- Ensure probs are numeric and normalized ---
        probs = policy_logits.flatten().astype(np.float32)
        probs = np.maximum(probs, 1e-8)
        probs = probs / probs.sum()

        # --- Choose action ---
        if explore:
            action = int(np.random.choice(len(probs), p=probs))
        else:
            action = int(np.argmax(probs))

        value = float(value_pred.flatten()[0])

        return action, probs, value



    def init_from_existing(self, existing_model):
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

        # Print summary of transfer
        for name in transferred:
            print(f"✅ Transferred weights for layer {name}")
        for name in skipped:
            print(f"⚠️ Skipping layer {name} (no matching layer in source model)")

        return self
    
    def predict_logits_and_value(self, x_inputs, verbose=0):
        """
        Convenience wrapper: runs the model and returns (policy_logits, value_preds).
        Args:
            x_inputs: [x1, x2, x3, mask] as prepared by to_training_dataset
            verbose: passed to model.predict
        Returns:
            policy_logits: np.array of shape (batch, num_actions)
            value_preds: np.array of shape (batch, 1)
        """
        preds = self.model.predict(x_inputs, verbose=verbose)

        # Case 1: Keras returns a dict (likely here)
        if isinstance(preds, dict):
            policy_logits = preds.get("output")
            value_preds = preds.get("value_output")
        # Case 2: Keras returns a list/tuple
        elif isinstance(preds, (list, tuple)) and len(preds) == 2:
            policy_logits, value_preds = preds
        else:
            raise ValueError(f"Unexpected predict() output type: {type(preds)}")

        return np.array(policy_logits, dtype=np.float32), np.array(value_preds, dtype=np.float32)




