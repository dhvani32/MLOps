from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Function to load the Iris dataset
def load_data():
    # Load the Iris dataset as a pandas DataFrame
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # Ensure the target column is of integer type for consistency
    df['target'] = df['target'].astype(int)  # Just to be safe
    
    # Return the DataFrame containing the Iris data (features and target)
    return df

# Function to prepare data by splitting into training and testing sets
def prepare_data(df, test_size=0.2, random_state=42):
    # Separate the features (X) and the target variable (y)
    X = df.drop(columns=["target"])  # All columns except 'target' are features
    y = df["target"]  # The 'target' column is the label/target variable
    
    # Split the data into training and testing sets, while keeping the class distribution balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Combine the features and target for both training and testing sets into DataFrames
    df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # Return the prepared training and testing DataFrames
    return df_train, df_test