import yaml
from typing import List, Dict
from sklearn.model_selection import train_test_split
import uproot
import os
import glob
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


class BinaryDecisionTree:
    def __init__(self, yaml_file: str):
        """
        Initialize with configurations from a YAML file.
        :param yaml_file: Path to the YAML configuration file.
        """
        self.input_paths = None
        self.samples = {}
        self.variables = []
        self.hyperparameters = {}
        self.model = None
        self.dataframes = {"background": None, "signal": None}
        self._parse_yaml(yaml_file)
        self._load_data()

    def _parse_yaml(self, yaml_file: str):
        try:
            with open(yaml_file, 'r') as file:
                config = yaml.safe_load(file)

            self.input_paths = config['input_path']
            self.samples[1] = config.get(1, [])
            self.samples[0] = config.get(0, [])
            self.variables = config.get('variables', [])
            self.hyperparameters = config.get('hyperparameters', {})

        except Exception as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _load_data(self):
        for sample_type, key in [("background", 0), ("signal", 1)]:
            sample_list = self.samples[key]
            dataframes = []

            for sample in sample_list:
                files = glob.glob(os.path.join(f"{self.input_paths}/{sample}/", "*.root"))
                for file in files:
                    tree = uproot.open(file)['analysis']
                    df = tree.arrays(self.variables, library="pd")
                    df['sample'] = sample
                    df['label'] = key
                    dataframes.append(df)

            combined_df = pd.concat(dataframes, ignore_index=True).dropna()
            self.dataframes[sample_type] = combined_df

    def print_config(self):
        """Prints the current configuration."""
        print(f"Input Paths: {self.input_paths}")
        print(f"Samples: {self.samples}")
        print(f"Variables: {self.variables}")
        print(f"Hyperparameters: {self.hyperparameters}")

    def prepare_data(self):
        background_df = self.dataframes["background"]
        signal_df = self.dataframes["signal"]

        combined_df = pd.concat([background_df, signal_df], ignore_index=True)
        X = combined_df[self.variables]  # Use only the specified variables
        y = combined_df['label']

        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self, model_path="xgboost_model.json", importance_plot_path=None):
        X_train, X_test, y_train, y_test = self.prepare_data()

        dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns.tolist())
        dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=X_test.columns.tolist())

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": self.hyperparameters.get("learning_rate", 0.1),
            "max_depth": self.hyperparameters.get("max_depth", 3),
            "random_state": 42
        }

        evals = [(dtrain, "train"), (dtest, "test")]
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.hyperparameters.get("n_estimators", 100),
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=True
        )

        y_pred_prob = self.model.predict(dtest)
        y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

        print(f"Accuracy: {accuracy_score(y_test, y_pred_binary):.2%}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_prob):.2f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_binary):.2f}")

        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")

        self.plot_feature_importance(save_path=importance_plot_path)
        return self.model

    def find_optimal_bdt_threshold(self):
        if not hasattr(self, 'model'):
            raise AttributeError("The model has not been trained. Train the model before calculating S/sqrt(B).")

        X_train, X_test, y_train, y_test = self.prepare_data()

        # Combine train and test data for score calculation
        X_combined = pd.concat([X_train, X_test], ignore_index=True)
        y_combined = pd.concat([y_train, y_test], ignore_index=True)

        # Generate scores using the trained model
        dmatrix = xgb.DMatrix(data=X_combined, feature_names=X_combined.columns.tolist())
        bdt_scores = self.model.predict(dmatrix)

        # Add scores and labels to a combined DataFrame
        combined_df = pd.DataFrame({
            'bdt_score': bdt_scores,
            'Label': y_combined
        })

        # Initialize variables to track the optimal threshold
        max_significance = 0
        optimal_threshold = 0

        for threshold in np.arange(0, 1.01, 0.01):
            signal_count = combined_df[(combined_df['bdt_score'] > threshold) & (combined_df['Label'] == 1)].shape[0]
            background_count = combined_df[(combined_df['bdt_score'] > threshold) & (combined_df['Label'] == 0)].shape[0]

            if background_count > 0:
                significance = signal_count / np.sqrt(background_count)
            else:
                significance = 0

            if significance > max_significance:
                max_significance = significance
                optimal_threshold = threshold

        print(f"Maximum S/sqrt(B) = {max_significance:.2f} at BDT score threshold = {optimal_threshold:.2f}")
        return optimal_threshold
    
    def plot_feature_importance(self, feature_names=None, save_path=None):
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(self.model, importance_type='gain')
        plt.title("Feature Importance by Gain")
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
    
        plt.show()
        
    def plot_bdt_score(self, save_path=None):
        if not hasattr(self, 'model'):
            raise AttributeError("The model has not been trained. Train the model before plotting BDT scores.")
    
        # Combine preloaded data
        background_df = self.dataframes["background"]
        signal_df = self.dataframes["signal"]
    
        combined_df = pd.concat([background_df, signal_df], ignore_index=True)
        X = combined_df[self.variables]
        y = combined_df['label']
    
        # Generate scores using the trained model
        dmatrix = xgb.DMatrix(data=X, feature_names=self.variables)
        bdt_scores = self.model.predict(dmatrix)
    
        # Add scores and labels to a combined DataFrame
        combined_df['BDT Score'] = bdt_scores
    
        # Separate background and signal
        background_scores = combined_df[combined_df['label'] == 0]['BDT Score']
        signal_scores = combined_df[combined_df['label'] == 1]['BDT Score']
    
        # Normalize each group to unity
        background_weights = [1 / len(background_scores)] * len(background_scores) if len(background_scores) > 0 else []
        signal_weights = [1 / len(signal_scores)] * len(signal_scores) if len(signal_scores) > 0 else []
    
        # Plot histograms
        plt.figure(figsize=(10, 6))
        plt.hist(background_scores, bins=50, weights=background_weights, alpha=0.7, label='Background (Label 0)', color='blue')
        plt.hist(signal_scores, bins=50, weights=signal_weights, alpha=0.7, label='Signal (Label 1)', color='red')
    
        plt.xlabel("BDT Score")
        plt.ylabel("Normalized Density")
        plt.title("Normalized BDT Score Distribution (Background and Signal)")
        plt.legend(loc='best')
        plt.tight_layout()
    
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"BDT score plot saved to {save_path}")
    
        plt.show()
        
    def dump_samples_to_root(self, output_dir="categorized_arrays", bdt_score_threshold=0.5):
        if not hasattr(self, 'model'):
            raise AttributeError("The model has not been trained. Train the model before dumping samples.")
    
        # Combine preloaded data
        background_df = self.dataframes["background"]
        signal_df = self.dataframes["signal"]
    
        combined_df = pd.concat([background_df, signal_df], ignore_index=True)
        X = combined_df[self.variables]
        y = combined_df['label']
    
        # Generate scores using the trained model
        dmatrix = xgb.DMatrix(data=X, feature_names=self.variables)
        bdt_scores = self.model.predict(dmatrix)
    
        # Add scores and labels to a combined DataFrame
        combined_df['bdt_score'] = bdt_scores
    
        # Apply the BDT score threshold
        selected_df = combined_df[combined_df['bdt_score'] > bdt_score_threshold].copy()
    
        # Ensure the required variables exist in the DataFrame
        required_columns = ['bdt_score', 'b_mZstar_jj', 'sample']
        if not all(col in selected_df.columns for col in required_columns):
            raise KeyError(f"The following required columns are missing: {set(required_columns) - set(selected_df.columns)}")
    
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
        # Group by sample name and save each group to a separate ROOT file
        for sample_name, sample_df in selected_df.groupby('sample'):
            output_file = os.path.join(output_dir, f"{sample_name}.root")
    
            # Save to ROOT file
            with uproot.recreate(output_file) as root_file:
                root_file["tree"] = {
                    "bdt_score": sample_df['bdt_score'].to_numpy(dtype=np.float32),
                    "b_mZstar_jj": sample_df['b_mZstar_jj'].to_numpy(dtype=np.float32),
                }
            print(f"Saved ROOT file for Sample {sample_name} at: {output_file}")

if __name__=='__main__':
    bdt = BinaryDecisionTree("config.yaml")
    model=bdt.train_model()
    bdt.plot_bdt_score()
    thresh=bdt.find_optimal_bdt_threshold()
    bdt.dump_samples_to_root(output_dir="categorized_minitrees", bdt_score_threshold=thresh)
