import pandas as pd
from pathlib import Path

def merge_features_and_predictions(features_file: Path, predictions_file: Path, output_file: Path) -> pd.DataFrame:
    """
    Merges extracted features with CNN model predictions based on filenames.

    Args:
        features_file (str): Path to the CSV file containing extracted features.
        predictions_file (str): Path to the CSV file containing CNN model predictions.
        output_file (str): Path to save the merged CSV file.

    Returns:
        None
    """
    # Load data
    features_df = pd.read_csv(features_file)
    predictions_df = pd.read_csv(predictions_file)

    # Merge on filename
    merged_df = features_df.merge(predictions_df, on='filename')

    # Save merged data
    merged_df.to_csv(output_file, index=False)
    return merged_df

def merge_test_features_and_predictions() -> pd.DataFrame:
    test_features_path = Path("Predictions/test_features.csv")
    test_predictions_path = Path("Predictions/test_predictions.csv")
    output_path = Path("Predictions/test_features_and_predictions.csv")
    
    return merge_features_and_predictions(test_features_path, test_predictions_path, output_path)

def merge_train_features_and_predictions() -> pd.DataFrame:
    train_features_path = Path("Predictions/train_features.csv")
    train_predictions_path = Path("Predictions/train_predictions.csv")
    output_path = Path("Predictions/train_features_and_predictions.csv")
    
    return merge_features_and_predictions(train_features_path, train_predictions_path, output_path)

if __name__ == "__main__":
    merge_test_features_and_predictions()
    merge_train_features_and_predictions()