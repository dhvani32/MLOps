from metaflow import FlowSpec, step
import os

# Define the Metaflow flow for training the RandomForest model
class RandomForestTrainFlow(FlowSpec):

    @step
    def start(self):
        # Import the preprocessing functions
        from preprocessing import load_data, prepare_data
        
        # Load and prepare the data
        data = load_data()  # Load the dataset
        train_df, test_df = prepare_data(data)  # Split into train and test datasets

        # Split the training and test sets into features and target
        self.X_train = train_df.drop(columns=["target"])
        self.y_train = train_df["target"]
        self.X_test = test_df.drop(columns=["target"])
        self.y_test = test_df["target"]

        # Print dataset shape for verification
        print(f"Data loaded with shape: {data.shape}")
        
        # Move to the next step: model training
        self.next(self.train_model)

    @step
    def train_model(self):
        # Import necessary libraries for training and evaluation
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import ParameterSampler
        from sklearn.metrics import accuracy_score
        import numpy as np

        # Set up the MLflow tracking URI and experiment
        mlflow.set_tracking_uri(f"file://{os.path.abspath('./mlruns')}")
        mlflow.set_experiment("iris-rf-experiment")

        # Define the hyperparameter grid for the RandomForest model
        param_dist = {
            "n_estimators": [50, 100, 150],  # Number of trees
            "max_depth": [3, 5, 10, None],  # Maximum depth of each tree
            "min_samples_split": [2, 5],  # Minimum samples to split a node
            "min_samples_leaf": [1, 2],  # Minimum samples to be a leaf node
        }

        # Sample a set of parameters for random search
        param_list = list(ParameterSampler(param_dist, n_iter=10, random_state=42))

        best_score = -1  # Initialize the best accuracy score
        best_model = None  # Placeholder for the best model
        best_run_id = None  # Placeholder for the best run ID

        # Iterate over the parameter combinations and train the model
        for i, params in enumerate(param_list):
            with mlflow.start_run(run_name=f"run_{i}") as run:
                # Train a RandomForestClassifier with the current hyperparameters
                clf = RandomForestClassifier(random_state=42, **params)
                clf.fit(self.X_train, self.y_train)

                # Calculate the accuracy on the test set
                acc = accuracy_score(self.y_test, clf.predict(self.X_test))

                # Log the parameters and accuracy to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)

                # Check if the current model has the best accuracy so far
                if acc > best_score:
                    best_score = acc
                    best_model = clf
                    best_run_id = run.info.run_id

        # Store the best model and associated run ID and accuracy
        self.best_model = best_model
        self.best_run_id = best_run_id
        self.best_score = best_score
        
        # Move to the next step: model registration
        self.next(self.register_model)

    @step
    def register_model(self):
        # Import MLflow and Sklearn for model registration
        import mlflow
        import mlflow.sklearn

        # Set the MLflow tracking URI and experiment
        mlflow.set_tracking_uri(f"file://{os.path.abspath('./mlruns')}")
        mlflow.set_experiment("iris-rf-experiment")

        # Register the best model under the best run ID
        with mlflow.start_run(run_id=self.best_run_id):
            mlflow.sklearn.log_model(
                sk_model=self.best_model,  # Best trained model
                artifact_path="model",  # Location for storing the model
                registered_model_name="iris-rf-model"  # Name for the registered model
            )

        # Print out the registered model's run ID
        print(f"Model registered under run ID: {self.best_run_id}")
        
        # Proceed to the final step: end of the flow
        self.next(self.end)

    @step
    def end(self):
        # Print out the final message and the best accuracy achieved
        print("Training and registration completed.")
        print(f"Best accuracy: {self.best_score:.4f}")

# Run the Metaflow flow when this script is executed
if __name__ == '__main__':
    RandomForestTrainFlow()