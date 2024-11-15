import pandas as pd
import numpy as np
import json
from collections import defaultdict


# Load the dataset from a CSV file
def load_data(file_path):
    # Read the data from the specified CSV file and return it as a DataFrame
    print(f"\n--- Loading Data from {file_path} ---\n")
    
    # Using pandas to read the CSV file and load it into a DataFrame
    df = pd.read_csv(file_path)
    
    # Print the DataFrame to show the loaded data in the console
    print(df)
    
    # Open the output file and write the DataFrame to it
    with open('D:/Github Projects/Machine Learning/Naive-Bayes-Classifier/Dataset/input_data.txt', 'w') as f:
        # Writing the column headers
        f.write(f"--- Loaded Data from {file_path} ---\n")
        f.write(f"\n{df}\n")
    
    print("\nData loaded successfully and written to output file.\n")
    
    # Return the DataFrame containing the loaded data
    return df


# Calculate the prior probabilities for each class in the target variable
def calculate_prior_probabilities(df):
    print("\n--- Calculating Prior Probabilities ---\n")
    # Count the number of instances for each class ('Yes' and 'No') in the 'PlayTennis' column
    class_counts = df['PlayTennis'].value_counts().to_dict()
    # Total number of instances in the dataset
    total_count = len(df)
    
    # Calculate the prior probability for each class
    priors = {cls: count / total_count for cls, count in class_counts.items()}
    
    # Print each prior probability with an explanation
    for cls, prob in priors.items():
        print(f"Prior Probability of class '{cls}' (P({cls})) = {prob:.6f}")
    
    print("\nPrior probabilities calculated successfully.\n")
    # Return the dictionary of prior probabilities
    return priors

# Calculate likelihoods with Laplace smoothing for each feature value given a class
def calculate_likelihoods(df):
    print("\n--- Calculating Likelihoods with Laplace Smoothing ---\n")
    # Use a nested dictionary to store likelihoods in the format: likelihoods[feature][value][class]
    likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    # Count the number of instances for each class in the target variable
    class_counts = df['PlayTennis'].value_counts().to_dict()
    
    # Iterate over each feature (excluding the target variable 'PlayTennis')
    for feature in df.columns[:-1]:
        print(f"Processing feature: {feature}")
        # Get unique values of the feature
        feature_values = df[feature].unique()
        
        # For each unique value of the feature, calculate the likelihood for each class
        for value in feature_values:
            for cls in class_counts:
                # Count the occurrences of the feature value for the given class
                count = len(df[(df[feature] == value) & (df['PlayTennis'] == cls)])
                # Apply Laplace smoothing: (count + 1) / (class_count + number of unique feature values)
                smoothed_likelihood = (count + 1) / (class_counts[cls] + len(feature_values))
                likelihoods[feature][value][cls] = smoothed_likelihood  # Store the likelihood
                
                # Print the calculation with a detailed explanation
                print(f"  P({feature}={value} | {cls}) = (count + 1) / (class_count + num_values)")
                print(f"  = ({count} + 1) / ({class_counts[cls]} + {len(feature_values)}) = {smoothed_likelihood:.6f}")
        # Print a blank line for better readability
        print()
    
    print("\nLikelihoods calculated successfully.\n")

    # Return the dictionary of likelihoods
    return likelihoods

# Train the Naive Bayes classifier and save the model
def train_naive_bayes(file_path):
    print("\n--- Training Naive Bayes Classifier ---\n")
    # Load the dataset
    df = load_data(file_path)
    
    # Calculate prior probabilities and likelihoods
    priors = calculate_prior_probabilities(df)
    likelihoods = calculate_likelihoods(df)
    
    # Convert dictionary keys to strings for JSON compatibility
    priors = {str(k): v for k, v in priors.items()}
    likelihoods = {
        feature: {str(value): {str(cls): prob for cls, prob in class_probs.items()} 
                  for value, class_probs in values.items()}
        for feature, values in likelihoods.items()
    }
    
    # Create the model as a dictionary containing priors and likelihoods
    model = {"priors": priors, "likelihoods": likelihoods}
    
    # Save the model to a JSON file for future use
    with open('D:/Github Projects/Machine Learning/Naive-Bayes-Classifier/Output/naive_bayes_model.json', 'w') as file:
        # Save with formatted JSON for readability
        json.dump(model, file, indent=4)  
    
    print("Model trained and saved to naive_bayes_model.json.\n")
    # Return the model for use in predictions
    return model

def predict(instance):
    print("\n--- Making Prediction ---\n")
    # Load the model from the JSON file
    with open('D:/Github Projects/Machine Learning/Naive-Bayes-Classifier/Output/naive_bayes_model.json', 'r') as file:
        model = json.load(file)
    
    # Extract prior probabilities and likelihoods from the model
    priors = model["priors"]
    likelihoods = model["likelihoods"]
    
    print(f"Instance to predict: {instance}\n")
    # List to store calculation steps for the log file
    calculation_steps = []
    calculation_steps.append("\n--- Making Prediction ---\n")
    calculation_steps.append(f"Instance to predict: {instance}\n")
    
    # Dictionary to store the log scores for each class
    scores = {}
    
    # Initialize the scores with the log of the prior probabilities
    for cls, prior in priors.items():
        # Use log to avoid underflow with small probabilities
        scores[cls] = np.log(prior)
        explanation = f"Initial score for class '{cls}' (Log(P({cls}))) = {scores[cls]:.6f}"
        print(explanation)
        calculation_steps.append(explanation + "\n")
    
    # Update the scores using the likelihoods for each feature
    for feature, value in instance.items():
        print(f"\nProcessing feature: {feature} | Value: {value}")
        calculation_steps.append(f"\nProcessing feature: {feature} | Value: {value}\n")
        
        if feature in likelihoods and value in likelihoods[feature]:
            # If the feature value exists in the model, use the likelihood
            for cls in scores:
                # Get the likelihood for the feature value given the class
                # 1e-6 to handle missing values in the likelihoods
                likelihood = likelihoods[feature][value].get(cls, 1e-6)
                explanation = (
                    f"  P({feature}={value} | {cls}) = {likelihood:.6f}\n"
                    f"  Updated score for class '{cls}' = {scores[cls] + np.log(likelihood):.6f}"
                )
                print(explanation)
                calculation_steps.append(explanation + "\n")
                
                # Update the score using log-likelihood
                scores[cls] += np.log(likelihood)
        else:
            # If the feature value is not found, print a message about using smoothing
            message = f"  Value '{value}' not found in likelihoods. Using smoothing.\n"
            print(message)
            calculation_steps.append(message)
    
    # Determine the class with the highest score
    predicted_class = max(scores, key=scores.get)
    print("\nFinal Scores:")
    calculation_steps.append("\nFinal Scores:\n")
    
    for cls, score in scores.items():
        result = f"  Class '{cls}': {score:.6f}"
        print(result)
        calculation_steps.append(result + "\n")
    
    final_result = f"\nPredicted class: {predicted_class}\n"
    print(final_result)
    calculation_steps.append(final_result)
    
    # Write all calculation steps to the calculations file
    with open('D:/Github Projects/Machine Learning/Naive-Bayes-Classifier/Output/naive_bayes_calculations.txt', 'a') as calc_file:
        calc_file.writelines(calculation_steps)
    
    # Return the predicted class and scores for further analysis
    return predicted_class, scores

def cross_validate(file_path):
    print("\n--- Cross-Validation ---\n")
    # Load the dataset
    df = load_data(file_path)

    # Initialize counters for the confusion matrix
    TP = 0  # True Positives
    TN = 0  # True Negatives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    # Counter for the number of correct predictions
    correct_predictions = 0
    # Total number of instances
    total_instances = len(df)
    
    # Prepare the header for the log file
    log_lines = []
    log_lines.append("\n--------------------------------------------------")
    log_lines.append("| Instance |   Actual   | Predicted  | Correct  |")
    log_lines.append("--------------------------------------------------")
    
    # Clear the calculations file before writing new content
    with open('D:/Github Projects/Machine Learning/Naive-Bayes-Classifier/Output/naive_bayes_calculations.txt', 'w') as calc_file:
        calc_file.write("Detailed Calculation Steps for Each Instance\n")
        calc_file.write("===========================================\n")
    
    # Iterate over each instance in the dataset
    for index, row in df.iterrows():
        print(f"\n--- Instance {index + 1} ---")

        # Remove the current instance from the dataset to train the model
        # Use all other instances for training
        train_df = df.drop(index)
        
        # Convert the row to a dictionary for prediction
        test_instance = row.to_dict()

        # Extract the actual class label for the test instance - Remove the actual class label from the test instance
        actual_class = test_instance.pop('PlayTennis')
        
        print("\nTraining model on remaining instances...\n")
        # Train the model on the training data
        priors = calculate_prior_probabilities(train_df)
        likelihoods = calculate_likelihoods(train_df)
        model = {"priors": priors, "likelihoods": likelihoods}
        
        # Make a prediction for the test instance and log calculation steps
        predicted_class, _ = predict(test_instance)
        # Check if the prediction is correct and print the result
        correct = (predicted_class == actual_class)
        print(f"Actual class: {actual_class} | Predicted class: {predicted_class} | Correct: {correct}\n")
        
        # Format the results for the log
        log_line = f"| {index + 1:^8} | {actual_class:^10} | {predicted_class:^10} | {str(correct):^8} |"
        log_lines.append(log_line)
        
        # Update the confusion matrix counters
        if actual_class == "Yes" and predicted_class == "Yes":
            TP += 1
        elif actual_class == "No" and predicted_class == "No":
            TN += 1
        elif actual_class == "No" and predicted_class == "Yes":
            FP += 1
        elif actual_class == "Yes" and predicted_class == "No":
            FN += 1

        if correct:
            # Increment the correct predictions counter
            correct_predictions += 1
    
    # Calculate and print the overall accuracy
    accuracy = correct_predictions / total_instances
    log_lines.append("--------------------------------------------------")
    log_lines.append(f"\nOverall Accuracy: {accuracy:.2f}\n")
    
    # Append the formatted confusion matrix to the log
    log_lines.append("\nConfusion Matrix:")
    log_lines.append("--------------------------------------------------")
    log_lines.append("                  Predicted")
    log_lines.append("            |   Yes   |    No   |")
    log_lines.append("------------|---------|---------|")
    log_lines.append(f"Actual Yes  |  {TP:^6} |  {FN:^6} |")
    log_lines.append(f"Actual No   |  {FP:^6} |  {TN:^6} |")
    log_lines.append("--------------------------------------------------")

    log_lines.append("--------------------------------------------------")
    log_lines.append(f"True Positives (TP): {TP}")
    log_lines.append(f"True Negatives (TN): {TN}")
    log_lines.append(f"False Positives (FP): {FP}")
    log_lines.append(f"False Negatives (FN): {FN}")
    log_lines.append("--------------------------------------------------")

    # Print the log lines to the console
    for line in log_lines:
        print(line)
    
    # Write the log lines to the log file
    with open('D:/Github Projects/Machine Learning/Naive-Bayes-Classifier/Output/naive_bayes_log.txt', 'w') as log_file:
        log_file.write("\n".join(log_lines))
    
    print("\nResults logged in Output/naive_bayes_log.txt\n")
    # Return the accuracy
    return accuracy


# Main function to run the training and evaluation
if __name__ == "__main__":
    # Train the model and evaluate it using cross-validation

    # Train the model on the "Play Tennis" dataset
    train_naive_bayes('D:/Github Projects/Machine Learning/Naive-Bayes-Classifier/Dataset/play_tennis.csv')

    # Cross-validate the model on the "Play Tennis" dataset and log the results
    cross_validate('D:/Github Projects/Machine Learning/Naive-Bayes-Classifier/Dataset/play_tennis.csv')