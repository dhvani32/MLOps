from metaflow import FlowSpec, step
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import pandas as pd

# Define the Metaflow flow for scoring a pre-trained RandomForest model
class RandomForestScoringFlow(FlowSpec):

    @step
    def start(self):
        # Load the Iris dataset with 'as_frame=True' to use it as a pandas DataFrame
        iris = load_iris(as_frame=True)  # Load the Iris dataset

        # Assign the data and target to class variables
        self.data = iris['data']  # Feature data (X)
        self.target = iris['target']  # Target labels (y)
        self.feature_names = iris['feature_names']  # Feature names (e.g., petal length)
        self.model = None  # Placeholder for the loaded model
        self.score = None  # Placeholder for the model's accuracy score
        
        # Move to the next step: data splitting
        self.next(self.split_data)

    @step
    def split_data(self):
        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=42
        )
        
        # Store the training and testing data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Move to the next step: loading the trained model
        self.next(self.load_model)

    @step
    def load_model(self):
        # Load the registered model from MLflow (version 1 of the model)
        self.model = mlflow.sklearn.load_model("models:/iris-rf-model/1")  # Model version 1
        
        # Move to the next step: scoring the model
        self.next(self.score_model)

    @step
    def score_model(self):
        # Score the model on the test set and compute the accuracy
        self.score = self.model.score(self.X_test, self.y_test)
        
        # Print the model's accuracy on the test set
        print(f"Model Test Accuracy: {self.score}")
        
        # Move to the final step
        self.next(self.end)

    @step
    def end(self):
        # Final step, prints a success message and the test accuracy
        print("Scoring completed successfully.")
        print(f"Test Accuracy: {self.score}")

# Run the Metaflow flow when this script is executed
if __name__ == '__main__':
    RandomForestScoringFlow()