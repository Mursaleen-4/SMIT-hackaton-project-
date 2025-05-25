import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from typing import Tuple, Dict, Any

class LaunchPredictor:
    def __init__(self, model_dir: str = "data"):
        self.model_dir = model_dir
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_names = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        print("\n[DEBUG] Starting model training...")
        print("[DEBUG] X shape:", X.shape)
        print("[DEBUG] y shape:", y.shape)
        print("[DEBUG] y values:", y.value_counts())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        print("[DEBUG] Feature names:", self.feature_names)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("[DEBUG] Training set shape:", X_train.shape)
        print("[DEBUG] Test set shape:", X_test.shape)
        print("[DEBUG] Training set target distribution:", y_train.value_counts())
        print("[DEBUG] Test set target distribution:", y_test.value_counts())
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("[DEBUG] Model metrics:", metrics)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions and return probabilities."""
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet")
        
        # Create a DataFrame with all required features, initialized to 0
        X_final = pd.DataFrame(0, index=X.index, columns=self.feature_names)
        
        # Copy values from input DataFrame to final DataFrame
        for col in X.columns:
            if col in self.feature_names:
                X_final[col] = X[col]
        
        # Make predictions
        predictions = self.model.predict(X_final)
        probabilities = self.model.predict_proba(X_final)
        
        return predictions, probabilities
    
    def save_model(self, filename: str = "launch_predictor.joblib"):
        """Save the trained model to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_model(self, filename: str = "launch_predictor.joblib"):
        """Load a trained model from disk."""
        filepath = os.path.join(self.model_dir, filename)
        saved_model = joblib.load(filepath)
        self.model = saved_model['model']
        self.feature_names = saved_model['feature_names']
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)

def main():
    """Main function to test model training."""
    from preprocessing import DataPreprocessor
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_training_data()
    
    # Train model
    predictor = LaunchPredictor()
    metrics = predictor.train(X, y)
    
    # Print metrics
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print feature importance
    importance = predictor.get_feature_importance()
    print("\nFeature Importance:")
    print(importance)

if __name__ == "__main__":
    main() 