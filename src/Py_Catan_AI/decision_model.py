from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from keras.models import Model
from keras.models import load_model, clone_model
from keras.layers import Input, Embedding, Reshape, Concatenate, Lambda, Dense, Activation
import matplotlib.pyplot as plt
from importlib.resources import files
import pandas as pd
import numpy as np
import sys  
import os
import warnings
from Py_Catan_AI.default_structure import default_structure
sys.path.append("../src")

class PlayerDecisionModelTypes:
    """
    Class to handle different decision based player model types.
    """
    DEFAULT_STRUCTURE = default_structure

    # Path for data files belonging to source code
    ORIGINAL_DEAULT_PATH = 'Py_Catan_AI.data'
    DEFAULT_PATH_TO_MODELS = 'Py_Catan_AI.models'
    DEFAULT_PATH_TO_DATA = 'Py_Catan_AI.data'
    

    # Default mode, usually overwritten in child classes
    PATH_TO_DEFAULT_MODEL = files(DEFAULT_PATH_TO_MODELS).joinpath('decision_model_gameplay.keras')
    PATH_TO_DEFAULT_MODEL_GAMEPLAY = files(DEFAULT_PATH_TO_MODELS).joinpath('decision_model_gameplay.keras')
    PATH_TO_DEFAULT_MODEL_SETUP = files(DEFAULT_PATH_TO_MODELS).joinpath('decision_model_setup.keras')
    PATH_TO_DEFAULT_MODEL_TRADE_RESPONSE = files(DEFAULT_PATH_TO_MODELS).joinpath('decision_model_trade_response.keras')
 
    # Data with randomness setting 0.0
    DEFAULT_TRAINING_DATA_RANDOMNESS_000 = [
        files('Py_Catan_AI.data').joinpath(f'data_from_tournament_randomness_000_{identifier}.csv') for identifier in
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj',
         'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az']
        ]
    DEFAULT_TRAINING_DATA_RANDOMNESS_000_MASKS = [
        files('Py_Catan_AI.data').joinpath(f'data_from_tournament_randomness_000_action_and_mask_{id}_masks.csv') for id in
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj',
         'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az']
        ]
    DEFAULT_TRAINING_DATA_RANDOMNESS_000_BEST_ACTION = [
        files('Py_Catan_AI.data').joinpath(f'data_from_tournament_randomness_000_action_and_mask_{id}_best_action_index.csv') for id in
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj',
         'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az']
        ]
    DEFAULT_TRAINING_DATA_RANDOMNESS_000_TEST = files('Py_Catan_AI.data').joinpath('data_from_tournament_randomness_000_test.csv')
    DEFAULT_TRAINING_DATA_RANDOMNESS_000_TEST_MASKS = files('Py_Catan_AI.data').joinpath('data_from_tournament_randomness_000_action_and_mask_test_masks.csv')
    DEFAULT_TRAINING_DATA_RANDOMNESS_000_TEST_BEST_ACTION = files('Py_Catan_AI.data').joinpath('data_from_tournament_randomness_000_action_and_mask_test_best_action_index.csv')

    # Data with randomness setting 0.1
    DEFAULT_TRAINING_DATA_RANDOMNESS_010 = [
        files('Py_Catan_AI.data').joinpath(f'data_from_tournament_randomness_010_{identifier}.csv') for identifier in
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj',
         'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az']
        ]
    DEFAULT_TRAINING_DATA_RANDOMNESS_010_TEST = files('Py_Catan_AI.data').joinpath('data_from_tournament_randomness_010_test.csv')

    # Data with randomness setting 0.35
    DEFAULT_TRAINING_DATA_RANDOMNESS_035 = [
        files('Py_Catan_AI.data').joinpath(f'data_from_tournament_randomness_035_{identifier}.csv') for identifier in 
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        ]
    DEFAULT_TRAINING_DATA_RANDOMNESS_035_MASKS = [
        files('Py_Catan_AI.data').joinpath(f'data_from_tournament_randomness_035_action_and_mask_{identifier}_masks.csv') for identifier in 
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        ]

    DEFAULT_TRAINING_DATA_RANDOMNESS_035_BEST_ACTION = [
        files('Py_Catan_AI.data').joinpath(f'data_from_tournament_randomness_035_action_and_mask_{identifier}_best_action_index.csv') for identifier in 
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        ]

    DEFAULT_TRAINING_DATA_RANDOMNESS_035_TEST = files('Py_Catan_AI.data').joinpath('data_from_tournament_randomness_035_test.csv')
    DEFAULT_TRAINING_DATA_RANDOMNESS_035_TEST_MASKS = files('Py_Catan_AI.data').joinpath('data_from_tournament_randomness_035_action_and_mask_test_masks.csv')
    DEFAULT_TRAINING_DATA_RANDOMNESS_035_TEST_BEST_ACTION = files('Py_Catan_AI.data').joinpath('data_from_tournament_randomness_035_action_and_mask_test_best_action_index.csv')


    # Data that can be used for training, including a test file generated in same way as the training data.
    DEFAULT_TRAINING_DATA = DEFAULT_TRAINING_DATA_RANDOMNESS_000.copy()
    DEFAULT_TRAINING_DATA_MASKS = DEFAULT_TRAINING_DATA_RANDOMNESS_000_MASKS.copy()
    DEFAULT_TRAINING_DATA_BEST_ACTION = DEFAULT_TRAINING_DATA_RANDOMNESS_000_BEST_ACTION.copy()
    
    # Standard test set used for correlation in all child classes
    DEFAULT_TEST_DATA_FILE = DEFAULT_TRAINING_DATA_RANDOMNESS_000_TEST
    DEFAULT_TEST_MASK_FILE = DEFAULT_TRAINING_DATA_RANDOMNESS_000_TEST_MASKS
    DEFAULT_TEST_BEST_ACTION_FILE = DEFAULT_TRAINING_DATA_RANDOMNESS_000_TEST_BEST_ACTION

    def __init__(self, structure, model_file_name: str = ''):
        '''
        Model has to be stored as Keras file in directory Py_Catan_AI.data

        Args:
        - model_file_name (str): Name of the Keras model file to load. If empty, uses the default model.
        '''
        self.structure = structure 
        self.model = None
        # === Model ===
        if model_file_name == '':
            # === Try loading the default model ===
            model_file_name = self.PATH_TO_DEFAULT_MODEL              
   
        self.load_model_from_file(model_file_name)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
        # === Test Data ===
        self.path_to_test_data = files(self.ORIGINAL_DEAULT_PATH).joinpath(self.DEFAULT_TEST_DATA_FILE)
        self.path_to_test_masks = files(self.ORIGINAL_DEAULT_PATH).joinpath(self.DEFAULT_TEST_MASK_FILE)
        self.path_to_test_best_action = files(self.ORIGINAL_DEAULT_PATH).joinpath(self.DEFAULT_TEST_BEST_ACTION_FILE)

        self.data_for_test_and_correlation = pd.read_csv(self.path_to_test_data, header=0)
        self.masks_for_test_and_correlation = pd.read_csv(self.path_to_test_masks, header=0)
        self.best_action_for_test_and_correlation = pd.read_csv(self.path_to_test_best_action, header=0)


    def read_test_data_from_file(self, 
                                 file_name_for_data: str, 
                                 file_name_for_masks: str, 
                                 file_name_for_best_action: str
                                 ) -> None:
        """
        Only used if you want to deviate from the default data.
        
        Reads data from a CSV file and updates the internal dataset:
        - 'data_for_test_and_correlation' attribute.
        - 'masks_for_test_and_correlation' attribute.
        - 'best_action_for_test_and_correlation' attribute.

        Args:
        - file_name_for_data (str): Path to the CSV file containing test data.
        - file_name_for_masks (str): Path to the CSV file containing test masks.
        - file_name_for_best_action (str): Path to the CSV file containing best action data.

        Raises:
        - FileNotFoundError: If any of the specified files do not exist.

        Returns:
        - None
        """
        # Try to resolve the path either as a package resource or a direct file path
        try:
            path = files(file_name_for_data)
            if not path.exists():
                raise FileNotFoundError
        except Exception:
            path = file_name_for_data
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file {file_name_for_data} not found.")
        self.data_for_test_and_correlation = pd.read_csv(path, header=0)

        try:
            path = files(file_name_for_masks)
            if not path.exists():
                raise FileNotFoundError
        except Exception:
            path = file_name_for_masks
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file {file_name_for_masks} not found.")
        self.masks_for_test_and_correlation = pd.read_csv(path, header=0)

        try:
            path = files(file_name_for_best_action)
            if not path.exists():
                raise FileNotFoundError
        except Exception:
            path = file_name_for_best_action
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file {file_name_for_best_action} not found.")
        self.best_action_for_test_and_correlation = pd.read_csv(path, header=0)

    def reset_model_to_new(self) -> None:
        """
        Creates a new Keras model with the same structure as the current model.
        """
  
        # Inputs
        input1_layer = Input(shape=(len(self.structure.vector_indices['nodes']),), dtype='int32', name='input1')
        input2_layer = Input(shape=(len(self.structure.vector_indices['edges']),), dtype='int32', name='input2')
        input3_layer = Input(shape=(len(self.structure.vector_indices['hands']),), dtype='float32', name='input3')

        # Embeddings
        embed1 = Embedding(input_dim=9, output_dim=4, name='embed1')(input1_layer)  # shape: (None, nodes, 4)
        embed2 = Embedding(input_dim=6, output_dim=3, name='embed2')(input2_layer)  # shape: (None, edges, 3)

        embed1_flat = Reshape((len(self.structure.vector_indices['nodes']) * 4,), name='reshape1')(embed1)
        embed2_flat = Reshape((len(self.structure.vector_indices['edges']) * 3,), name='reshape2')(embed2)

        # Normalize input3
        normalized_input3 = Lambda(lambda x: x / 10.0, name='normalize_input3')(input3_layer)

        # Concatenate
        combined = Concatenate(name='concat')([embed1_flat, embed2_flat, normalized_input3])

        # Add some Dense layers for representation (you can tune this)
        dense1 = Dense(128, activation='relu')(combined)
        dense2 = Dense(64, activation='relu')(dense1)

        # Final logits layer (before masking)
        logits = Dense(self.structure.mask_space_length, name='logits')(dense2)  # No activation here

        # Binary mask input (mask shape = (action_space.length_of_action_space,))
        mask_input = Input(shape=(self.structure.mask_space_length,), dtype='float32', name='mask_input')

        # Apply mask to logits - set logits of forbidden actions to a large negative number so softmax zeroes them
        large_negative = -1e9
        masked_logits = Lambda(lambda x: x[0] + (1.0 - x[1]) * large_negative)([logits, mask_input])

        # Softmax output over masked logits
        output = Activation('softmax', name='output')(masked_logits)

        # Define the model with inputs and outputs
        model = Model(inputs=[input1_layer, input2_layer, input3_layer, mask_input], outputs=output)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        return
    
    def load_model_from_file(self, path_to_keras_model: str) -> None:
        """
        Loads a Keras model from a specified file and updates the internal model attribute.
        path_keras_model can be a string or a files(string) object.
        """
        # Try to resolve the path either as a package resource or a direct file path
        try:
            path = files(path_to_keras_model)
            if not path.exists():
                raise FileNotFoundError
        except Exception:
            path = path_to_keras_model
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file {path_to_keras_model} not found.")
        new_model = load_model(path, safe_mode=False , compile=True)
        self.model = new_model
        return
    
    def get_model(self) -> Model:
        """
        Returns the current Keras model.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please load or create a model first.")
        return self.model

    def save_model_to_file(self, path_to_keras_model: str) -> None:
        """
        Saves the current model to a specified file.
        """
        # Try to resolve the path either as a package resource or a direct file path
        # Check if the directory exists and is writable
        dir_path = os.path.dirname(path_to_keras_model)
        if dir_path and not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist for saving model.")
        if dir_path and not os.access(dir_path, os.W_OK):
            raise PermissionError(f"Directory {dir_path} is not writable for saving model.")
        path = path_to_keras_model
        self.model.save(path, include_optimizer=True)
        return

    def load_dataframe_from_csv(self, file_path_csv: any) -> pd.DataFrame:
        """
        Loads a vector, best_action or mask dataframe from a CSV file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded vector dataframe.
        """
        if type(file_path_csv) is not list:
            file_path_csv = [file_path_csv]

        file_path_csv = [str(f) for f in file_path_csv]

        for i, data_file_csv in enumerate(file_path_csv):
            # convert path to right format and check extension

            data_file_csv = str(data_file_csv)
            if type(data_file_csv) is not str or not data_file_csv.endswith('.csv'):
                print(f"  ⚠️ Invalid file path: {data_file_csv}, skipping...")
                continue

            # Check if file exists
            data_file_exists = os.path.exists(data_file_csv)
            if not (data_file_exists):
                print(f"data_file_csv: {data_file_csv}  ⚠️ File does not exist, skipping...")
                continue
            current_data = pd.read_csv(data_file_csv)

            if i == 0:
                # For the first file, create the dataframes
                combined_data_dataframe = current_data.copy()
            else:
                # For consecutive files, concatenate the dataframes
                combined_data_dataframe = pd.concat([combined_data_dataframe, current_data], 
                                                            ignore_index=True)

        return combined_data_dataframe

    def generate_X_data_from_catan_vector_format_dataframe(self, vector_dataframe: pd.DataFrame) -> list[np.ndarray]:
        """
        Generates training input data from a dataframe in Catan vector format.

        Returns a list of NumPy arrays representing the input data in the form [input1, input2, input3] which can be used
        as input for model training or creating model predictions.

        Example usage:
        ```
        x_data = self.generate_X_data_from_catan_vector_format_dataframe(vector_dataframe)
        y_data = self.generate_Y_data_from_best_action_dataframe(best_action_dataframe)
        mask_data = self.generate_mask_data_from_dataframe(mask_dataframe)
        model.fit(
            x=[x_data[0], x_data[1], x_data[2], mask_data],
            y=y_data)
        ```
        """
        # === Extract inputs (from column 6 onward) ===
        # input1: first edges
        input1 = vector_dataframe.iloc[:, self.structure.vector_indices['nodes']].values.astype(np.int32)
        # input2: next nodes
        input2 = vector_dataframe.iloc[:, self.structure.vector_indices['edges']].values.astype(np.int32)
        # input3: final 4 hands
        input3 = vector_dataframe.iloc[:, self.structure.vector_indices['hands']].values.astype(np.int32)
        return [input1, input2, input3]

    def generate_mask_data_from_dataframe(self, mask_dataframe: pd.DataFrame) -> np.ndarray:
        """
        Generates mask data from a dataframe in Catan vector format.

        Example usage:
        ```
        x_data = self.generate_X_data_from_catan_vector_format_dataframe(vector_dataframe)
        y_data = self.generate_Y_data_from_best_action_dataframe(best_action_dataframe)
        mask_data = self.generate_mask_data_from_dataframe(mask_dataframe)
        model.fit(
            x=[x_data[0], x_data[1], x_data[2], mask_data],
            y=y_data)
        ```

        Args:
            mask_dataframe (pd.DataFrame, optional): DataFrame containing the mask data. Defaults to None.

        Returns:
            np.ndarray: Array containing the mask data.
        """
        # generate masks data from mask dataframe
        mask_data = mask_dataframe.values.astype(np.int32)
        return mask_data
    
    def generate_Y_data_from_best_action_dataframe(self, best_action_dataframe: pd.DataFrame) -> np.ndarray:
        """
        Generates Y data from a dataframe containing the best action information as index.

        Example usage:
        ```
        x_data = self.generate_X_data_from_catan_vector_format_dataframe(vector_dataframe)
        y_data = self.generate_Y_data_from_best_action_dataframe(best_action_dataframe)
        mask_data = self.generate_mask_data_from_dataframe(mask_dataframe)
        model.fit(
            x=[x_data[0], x_data[1], x_data[2], mask_data],
            y=y_data)
        ```

        Args:
            best_action_dataframe (pd.DataFrame): DataFrame containing the best action information.

        Returns:
            np.ndarray: Array containing the Y data.
        """
        # generate data from with y-values from best action dataframe
        y_train = best_action_dataframe.values.astype(np.int32)
        y_train = y_train.flatten()
        # transform y_train from an index to one_hot_encoded
        y_train = np.identity(self.structure.mask_space_length)[y_train]
        return y_train

    def train_the_model(self, 
                        vector_dataframe: pd.DataFrame,
                        mask_dataframe: pd.DataFrame,
                        best_action_dataframe: pd.DataFrame,
                        batch_size: int = 32, 
                        epochs: int = 128,
                        verbose: int = 0) -> None:
        """
        Trains the model with the provided training data.

        Arguments:
        - data: DataFrame containing the training data as a list of numpy vectors. If None, uses self.data.
        - batch_size: Number of samples per gradient update.
        - epochs: Number of epochs to train the model.

        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please load or create a model first.")

        x_train = self.generate_X_data_from_catan_vector_format_dataframe(vector_dataframe)
        y_train = self.generate_Y_data_from_best_action_dataframe(best_action_dataframe)
        mask_data = self.generate_mask_data_from_dataframe(mask_dataframe)

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(
            x=[x_train[0], x_train[1], x_train[2], mask_data],
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose = verbose,
            validation_split=0.2,
            )
        return


    def create_prediction_from_model(self, 
                                    vector_dataframe: pd.DataFrame = None,
                                    mask_dataframe: pd.DataFrame = None) -> np.ndarray:
        """
        Creates a prediction from the model using the provided dataframes. The prediction is returned
        as a NumPy array containing the predicted class indices for 'best action'.

        Args:
            vector_dataframe (pd.DataFrame, optional): DataFrame containing the vector data. Defaults to None.
            mask_dataframe (pd.DataFrame, optional): DataFrame containing the mask data. Defaults to None.
        """
        x_data = self.generate_X_data_from_catan_vector_format_dataframe(vector_dataframe)
        mask_data = self.generate_mask_data_from_dataframe(mask_dataframe)
        y_pred_probs = self.model.predict(
            x=[x_data[0], x_data[1], x_data[2], mask_data],
            verbose=0
        )
        # Convert predictions from probabilities to class indices
        y_pred = np.argmax(y_pred_probs, axis=1)
        return y_pred

    def create_prediction_from_vector_and_mask(self, vector: np.ndarray, mask: np.ndarray) -> np.int32:
        """
        Creates a prediction from the model using the provided vector and mask. The function will generate the mask
        from the vector before generating a prediction.

        The prediction is returned as an index in ActionSpace, pointing to the 'best_action'
        """
        x_vector = vector
        mask_vector = mask


        df_x = pd.DataFrame([x_vector], columns=self.structure.vector_space_header)
        df_mask = pd.DataFrame([mask_vector], columns=self.structure.mask_space_header)
        y_pred = self.create_prediction_from_model(df_x, df_mask)
        return y_pred[0]

    def create_plot_for_correlations(   self,
                                            vector_dataframe: pd.DataFrame = None,
                                            mask_dataframe: pd.DataFrame = None,
                                            y_reference_dataframe: pd.DataFrame = None
                                            ) -> None:


        x_data = self.generate_X_data_from_catan_vector_format_dataframe(vector_dataframe)
        mask_data = self.generate_mask_data_from_dataframe(mask_dataframe)
        y_true = self.generate_Y_data_from_best_action_dataframe(y_reference_dataframe)
        y_true = np.argmax(y_true, axis=1)
        y_pred_probs = self.model.predict(
            x=[x_data[0], x_data[1], x_data[2], mask_data],
            verbose=0
        )
        
        # Convert predictions from probabilities to class indices
        y_pred = np.argmax(y_pred_probs, axis=1)

        # get statistics
        max_probs = np.max(y_pred_probs, axis=1)
        errors = y_pred - y_true
        # Create visualization
        plt.figure(figsize=(15, 10))

        # Plot 1: Predictions vs True values scatter plot
        plt.subplot(2, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, s=1)
        plt.plot([0, self.structure.mask_space_length-1], [0, self.structure.mask_space_length-1], 'r--', label='Perfect prediction')
        plt.xlabel('True Action Index')
        plt.ylabel('Predicted Action Index')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Distribution of true vs predicted actions
        plt.subplot(2, 3, 2)
        bins = np.arange(self.structure.mask_space_length + 1) - 0.5
        plt.hist(y_true, bins=bins, alpha=0.7, label='True', density=True)
        plt.hist(y_pred, bins=bins, alpha=0.7, label='Predicted', density=True)
        plt.xlabel('Action Index')
        plt.ylabel('Density')
        plt.title('Distribution of Actions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Prediction confidence distribution
        plt.subplot(2, 3, 3)
        plt.hist(max_probs, bins=50, alpha=0.7)
        plt.xlabel('Maximum Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.grid(True, alpha=0.3)

        # Plot 4: Confusion matrix (for a subset if too large)
        plt.subplot(2, 3, 4)
        if self.structure.mask_space_length <= 20:
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
        else:
            # Show confusion matrix for top 10 most frequent classes
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            top_classes = unique_true[np.argsort(counts_true)[-10:]]
            
            # Filter data for top classes
            mask_top = np.isin(y_true, top_classes)
            y_true_top = y_true[mask_top]
            y_pred_top = y_pred[mask_top]
            
            cm_top = confusion_matrix(y_true_top, y_pred_top, labels=top_classes)
            sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues', square=True,
                        xticklabels=top_classes, yticklabels=top_classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix (Top 10 Classes)')

        # Plot 5: Error analysis - difference between predicted and true
        plt.subplot(2, 3, 5)
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel('Prediction Error (Pred - True)')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.axvline(x=0, color='red', linestyle='--', label='Perfect prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 6: Action-wise accuracy
        plt.subplot(2, 3, 6)
        action_accuracy = []
        action_labels = []
        for action_idx in range(self.structure.mask_space_length):
            mask = y_true == action_idx
            if np.sum(mask) > 0:  # Only include actions that appear in test set
                acc = np.mean(y_pred[mask] == y_true[mask])
                action_accuracy.append(acc)
                action_labels.append(action_idx)

        if len(action_accuracy) > 0:
            plt.bar(range(len(action_accuracy)), action_accuracy)
            plt.xlabel('Action Index')
            plt.ylabel('Accuracy')
            plt.title('Per-Action Accuracy')
            plt.xticks(range(len(action_labels)), action_labels, rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return 

    def print_summary_statistics(   self,
                                    vector_dataframe: pd.DataFrame = None,
                                    mask_dataframe: pd.DataFrame = None,
                                    y_reference_dataframe: pd.DataFrame = None
                                    ) -> None:
        """
        Print summary statistics for model performance.
        """
        x_data = self.generate_X_data_from_catan_vector_format_dataframe(vector_dataframe)
        mask_data = self.generate_mask_data_from_dataframe(mask_dataframe)
        y_true = self.generate_Y_data_from_best_action_dataframe(y_reference_dataframe)
        y_true = np.argmax(y_true, axis=1)
        y_pred_probs = self.model.predict(
            x=[x_data[0], x_data[1], x_data[2], mask_data],
            verbose=0
        )
        
        # Convert predictions from probabilities to class indices
        y_pred = np.argmax(y_pred_probs, axis=1)

        # get statistics
        max_probs = np.max(y_pred_probs, axis=1)
        errors = y_pred - y_true

        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Mean prediction confidence: {np.mean(max_probs):.4f}")
        print(f"Std prediction confidence: {np.std(max_probs):.4f}")
        print(f"Mean absolute error: {np.mean(np.abs(errors)):.4f}")
        print(f"Std absolute error: {np.std(np.abs(errors)):.4f}")

        # Action distribution comparison
        print(f"\nAction Distribution Comparison:")
        print("Action | True Count | Pred Count | True % | Pred %")
        print("-" * 55)
        for action_idx in range(min(100, self.structure.mask_space_length)):  # Show first 100 actions
            true_count = np.sum(y_true == action_idx)
            pred_count = np.sum(y_pred == action_idx)
            true_pct = (true_count / len(y_true)) * 100
            pred_pct = (pred_count / len(y_pred)) * 100
            print(f"{action_idx:6d} | {true_count:10d} | {pred_count:10d} | {true_pct:6.2f} | {pred_pct:6.2f}")

        if self.structure.mask_space_length > 100:
            print("... (showing first 100 actions only)")

        return





    

# ----------------------------------------------------------------------
class BlankDecisionModel(PlayerDecisionModelTypes):

    def __init__(self,structure):
        """
        Untrained model cloned from the default model. Model has to be trained by the user:
        - Use 'model_type.train_the_model()' to train the model with data.

        After this you can plot correlations, save the model, make predictions, etc.

        The test data is an empty DataFrame, so no correlations can be plotted.
        """
        super().__init__(structure,model_file_name='')
        # this is a blank model, so everything has to be filled by user
        self.reset_model_to_new()
        self.path_to_test_data = ""
        self.path_to_keras_model = ""
        
        # === Do not load specific test data, use default ===
        self.data_for_test_and_correlation = pd.DataFrame()

        return
    

# ----------------------------------------------------------------------    
class TrainedDecisionModelGamePlay(PlayerDecisionModelTypes):
    
    def __init__(self):
        """
        Model trained on tournament data without randomeness.
        """
        super().__init__(self.DEFAULT_STRUCTURE,model_file_name=self.PATH_TO_DEFAULT_MODEL_GAMEPLAY)
# ----------------------------------------------------------------------    
class TrainedDecisionModelGameSetup(PlayerDecisionModelTypes):
    
    def __init__(self):
        """
        Model trained on tournament data without randomeness.
        """
        super().__init__(self.DEFAULT_STRUCTURE,model_file_name=self.PATH_TO_DEFAULT_MODEL_SETUP)
# ----------------------------------------------------------------------    
class TrainedDecisionModelTradeResponse(PlayerDecisionModelTypes):
    
    def __init__(self):
        """
        Model trained on tournament data without randomeness.
        """
        super().__init__(self.DEFAULT_STRUCTURE,model_file_name=self.PATH_TO_DEFAULT_MODEL_TRADE_RESPONSE)

    
 
