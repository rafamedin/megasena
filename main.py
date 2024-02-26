import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Attention, Embedding, Flatten, concatenate, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

# Load historical lottery data (replace 'lottery_data.csv' with the actual dataset)
lottery_data = pd.read_csv('lottery_data.csv')

# Assuming the dataset contains columns 'draw_date' and 'winning_numbers', and each row represents a single draw

# Generate all possible combinations of lottery numbers
possible_combinations = list(combinations(range(1, 61), 6))

# Convert winning numbers to sets for faster lookup
winning_sets = set(map(frozenset, lottery_data['winning_numbers']))

# Create a DataFrame to store features (previous winning combinations) and labels (next winning combination)
features = []
labels = []

# Iterate through historical data to generate training samples
for i in range(len(lottery_data) - 1):
    previous_winning_numbers = lottery_data.loc[i, 'winning_numbers']
    next_winning_numbers = lottery_data.loc[i + 1, 'winning_numbers']
    
    # Convert winning numbers to sets
    previous_winning_set = set(previous_winning_numbers)
    
    # Generate negative samples (combinations that have not appeared in previous draws)
    negative_samples = [comb for comb in possible_combinations if not comb.intersection(previous_winning_set)]
    
    # Add positive and negative samples to features and labels
    features.extend([list(comb) for comb in previous_winning_numbers])
    labels.extend([1] * len(previous_winning_numbers))  # 1 for positive samples
    
    features.extend(negative_samples)
    labels.extend([0] * len(negative_samples))  # 0 for negative samples

# Convert lists to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Feature selection using SelectKBest
k_best = SelectKBest(score_func=f_classif, k=10)
X_selected = k_best.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Perform oversampling to address class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define a function to create a deep learning model
def create_model(optimizer='adam', dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(128, input_dim=10, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Wrap the Keras model as a scikit-learn estimator
keras_model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameters for tuning
param_grid = {
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.2, 0.3, 0.4]
}

# Perform randomized search cross-validation for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=keras_model, param_distributions=param_grid, n_iter=10, cv=StratifiedKFold(n_splits=3), verbose=2)
random_search.fit(X_train_resampled, y_train_resampled)

# Best hyperparameters
best_params = random_search.best_params_

# Train the best model
best_model = create_model(optimizer=best_params['optimizer'], dropout_rate=best_params['dropout_rate'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
best_model.fit(X_train_resampled, y_train_resampled, epochs=best_params['epochs'], batch_size=best_params['batch_size'],
               validation_split=0.2, callbacks=[early_stopping], verbose=2)

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Function to generate the next lottery combination
def generate_next_combination(previous_winning_numbers):
    previous_winning_set = set(previous_winning_numbers)
    possible_next_combinations = [comb for comb in possible_combinations if not comb.intersection(previous_winning_set)]
    next_combination = best_model.predict(possible_next_combinations)
    return possible_next_combinations[next_combination[0]]

# Example usage
previous_winning_numbers = [4, 8, 15, 16, 23, 42]  # Replace with actual previous winning numbers
next_combination = generate_next_combination(previous_winning_numbers)
print("Next Winning Combination:", next_combination)
