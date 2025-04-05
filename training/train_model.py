import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import pickle
import os
import logging
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Set up and return necessary file paths"""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "semi_labeled_expenses.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    return data_path, models_dir

def load_and_explore_data(data_path):
    """Load dataset and perform initial exploration"""
    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Count missing values
    missing_values = df['label'].isna().sum()
    logger.info(f"Missing values: {missing_values}")
    logger.info(f"Percentage of missing labels: {(missing_values/len(df))*100:.2f}%")
    
    # Get label distribution
    label_dist = df['label'].value_counts()
    logger.info(f"Labels distribution: {label_dist}")
    
    # Extract only labeled data
    labeled_df = df.dropna(subset=['label']).copy()
    logger.info(f"Labeled data shape: {labeled_df.shape}")
    
    # Check for class imbalance
    class_ratio = labeled_df['label'].value_counts(normalize=True)
    logger.info(f"Class distribution: {class_ratio}")
    
    # Check for duplicate entries
    duplicates = labeled_df.duplicated().sum()
    logger.info(f"Number of duplicate entries: {duplicates}")
    
    return df, labeled_df

def preprocess_data(labeled_df):
    """Preprocess data for model training"""
    logger.info("Preprocessing data...")
    
    # Convert amount to float if it's not already
    labeled_df['amount'] = labeled_df['amount'].apply(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else float(x))
    
    # Check for outliers in amount
    amount_stats = labeled_df['amount'].describe()
    logger.info(f"Amount statistics: {amount_stats}")
    
    # Remove extreme outliers (beyond 3 standard deviations)
    mean = labeled_df['amount'].mean()
    std = labeled_df['amount'].std()
    labeled_df = labeled_df[(labeled_df['amount'] >= mean - 3*std) & (labeled_df['amount'] <= mean + 3*std)]
    logger.info(f"Data shape after removing outliers: {labeled_df.shape}")
    
    # Log distributions
    logger.info("\nCategory distribution:")
    logger.info(labeled_df['category'].value_counts())
    
    # Encode categorical variables
    X = labeled_df.drop('label', axis=1)
    y = labeled_df['label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder

def engineer_features(X):
    """Engineer features for model training"""
    # Extract day of week from date
    X['date'] = pd.to_datetime(X['date'], errors='coerce')
    X['day_of_week'] = X['date'].dt.day_name()
    X['month'] = X['date'].dt.month_name()
    X['day_of_month'] = X['date'].dt.day
    X['is_weekend'] = X['date'].dt.dayofweek >= 5
    X['quarter'] = X['date'].dt.quarter
    X['is_end_of_month'] = X['day_of_month'] >= 25
    X['is_start_of_month'] = X['day_of_month'] <= 5
    
    # Create amount-based features using robust scaling
    X['amount_scaled'] = (X['amount'] - X['amount'].median()) / (X['amount'].quantile(0.75) - X['amount'].quantile(0.25))
    X['amount_per_age'] = X['amount'] / X['age']
    X['amount_per_age_scaled'] = (X['amount_per_age'] - X['amount_per_age'].median()) / (X['amount_per_age'].quantile(0.75) - X['amount_per_age'].quantile(0.25))
    
    # Create temporal amount features
    X['weekend_amount'] = X['is_weekend'] * X['amount']
    X['end_month_amount'] = X['is_end_of_month'] * X['amount']
    X['start_month_amount'] = X['is_start_of_month'] * X['amount']
    
    # Create age-based features
    X['age_scaled'] = (X['age'] - X['age'].median()) / (X['age'].quantile(0.75) - X['age'].quantile(0.25))
    
    # Create interaction features
    X['age_amount_interaction'] = X['age_scaled'] * X['amount_scaled']
    X['weekend_age_interaction'] = X['is_weekend'] * X['age_scaled']
    X['end_month_age_interaction'] = X['is_end_of_month'] * X['age_scaled']
    
    # Convert categorical columns to numeric codes with noise
    categorical_cols = ['day_of_week', 'month', 'gender', 'quarter']
    for col in categorical_cols:
        if col in X.columns:
            # Add small random noise to prevent perfect correlation
            X[col] = pd.Categorical(X[col]).codes + np.random.normal(0, 0.01, len(X))
    
    # Drop unnecessary columns
    X = X.drop(['date', 'item', 'category'], axis=1)
    
    # Check for highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        logger.info(f"Dropping highly correlated features: {to_drop}")
        X = X.drop(to_drop, axis=1)
    
    return X

def split_data(X, y):
    """Split data into training and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def perform_cross_validation(X_train, y_train):
    """Perform cross validation to evaluate model performance"""
    logger.info("Performing cross-validation...")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Define XGBoost model with conservative parameters
    model = xgb.XGBClassifier(
        n_estimators=50,
        learning_rate=0.01,
        max_depth=2,
        subsample=0.5,
        colsample_bytree=0.5,
        min_child_weight=5,
        reg_alpha=2.0,
        reg_lambda=5.0,
        gamma=1.0,
        scale_pos_weight=class_weights[1] if len(class_weights) > 1 else 1.0,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    
    # Use StratifiedKFold for better cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, 
        X_train, 
        y_train, 
        cv=cv, 
        scoring=['accuracy', 'roc_auc', 'f1', 'precision', 'recall'],
        return_train_score=True
    )
    
    # Extract and print CV scores
    for metric in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        logger.info(f"Mean CV {metric}: {test_scores.mean():.4f} (±{test_scores.std():.4f})")
        logger.info(f"Mean Train {metric}: {train_scores.mean():.4f} (±{train_scores.std():.4f})")
    
    # Check for overfitting
    overfitting_score = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
    logger.info(f"Overfitting score (train - test): {overfitting_score:.4f}")
    
    return model

def select_features(model, X_train, y_train, X_test):
    """Select important features using the model"""
    logger.info("Performing feature selection...")
    
    # Fit the model to get feature importance
    model.fit(X_train, y_train)
    
    # Use SelectFromModel to select features
    selector = SelectFromModel(model, prefit=True, threshold='mean')
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()]
    logger.info(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
    logger.info(f"Selected features: {list(selected_features)}")
    
    return X_train_selected, X_test_selected, selected_features

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, models_dir):
    """Train the final model and evaluate its performance"""
    logger.info("Training final model...")
    
    # Perform feature selection with a more lenient threshold
    selector = SelectFromModel(model, prefit=False, threshold='mean')
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()]
    logger.info(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
    logger.info(f"Selected features: {list(selected_features)}")
    
    # Split training data to create a validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_selected, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create new model instance with conservative parameters
    final_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=2,
        subsample=0.5,
        colsample_bytree=0.5,
        min_child_weight=5,
        reg_alpha=2.0,
        reg_lambda=5.0,
        gamma=1.0,
        scale_pos_weight=model.get_params()['scale_pos_weight'],
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train with early stopping
    eval_set = [(X_val, y_val)]
    final_model.fit(
        X_train_final,
        y_train_final,
        eval_set=eval_set,
        verbose=True
    )
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test_selected)
    y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test ROC-AUC: {roc_auc:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Need', 'Want'], 
                yticklabels=['Need', 'Want'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = models_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    return final_model, selected_features

def analyze_feature_importance(model, X_train, selected_features, models_dir):
    """Analyze and visualize feature importance"""
    logger.info("Analyzing feature importance...")
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Display top 10 features
    top_features = importance_df.head(10)
    logger.info(f"Top 10 important features:\n{top_features}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(15))
    plt.title('Feature Importance')
    plt.tight_layout()
    fi_path = models_dir / "feature_importance.png"
    plt.savefig(fi_path)
    logger.info(f"Feature importance plot saved to {fi_path}")

def save_model(model, label_encoder, selected_features, models_dir):
    """Save the model, label encoder, and selected features"""
    logger.info("Saving model...")
    
    # Save XGBoost model
    model_path = models_dir / "xgb_classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save label encoder
    encoder_path = models_dir / "label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save selected features
    features_path = models_dir / "selected_features.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)
    
    logger.info(f"Model saved to {model_path}")

def main():
    """Main function to orchestrate the training process"""
    logger.info("Starting model training process...")
    
    # Setup paths
    data_path, models_dir = setup_paths()
    
    # Load and explore data
    df, labeled_df = load_and_explore_data(data_path)
    logger.info(f"Total labeled data points: {len(labeled_df)}")
    
    # Preprocess data
    X, y_encoded, label_encoder = preprocess_data(labeled_df)
    
    # Engineer features
    X_encoded = engineer_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_encoded, y_encoded)
    
    # Perform cross-validation
    model = perform_cross_validation(X_train, y_train)
    
    # Train and evaluate final model
    final_model, selected_features = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, models_dir)
    
    # Analyze feature importance
    analyze_feature_importance(final_model, X_train, selected_features, models_dir)
    
    # Save model and encoder
    save_model(final_model, label_encoder, selected_features, models_dir)
    
    logger.info("Training process completed!")

if __name__ == "__main__":
    main()