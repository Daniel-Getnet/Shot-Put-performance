import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importing dataset
dani_file = pd.read_csv('dani_dataf.csv')

# Create a separate LabelEncoder object for each categorical column
dani_gender = LabelEncoder()

# Fit the label encoder and transform each categorical column individually
dani_file["Gender"] = dani_gender.fit_transform(dani_file["Gender"])

# Handling missing values by replacing them with the mean of each feature
X = dani_file.iloc[:, :-1]
X.fillna(X.mean(), inplace=True) 

y = dani_file.iloc[:, -1]
y.fillna(y.mean(), inplace=True)  

# Splitting dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 42)
# Instantiate Random Forest Regressor
gb_regressor = GradientBoostingRegressor(
          n_estimators= 103, # [100, 500] step of 1
        learning_rate= 0.3,  # [0.01, 0.3] step 0f 0.01
        min_samples_split= 10,  # [2, 10] step 0f 1
        subsample= 0.6,   # (0.1, 1.0] step of 0.1
        random_state=42  # Seed for the random number generator

)

# Fit to training set
gb_regressor.fit(train_X, train_y)

# Predict on test set
pred_y = gb_regressor.predict(test_X)

# Model evaluation 
print(f"r_square_for_the_test_dataset : {r2_score(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")

print('**************************************')

# Initialize LabelEncoders for categorical variables
le_dani_gender = LabelEncoder()

# Fit the LabelEncoders with possible categories
possible_dani_gender = ['F', 'M']

# Fit the LabelEncoders with possible categories
le_dani_gender.fit(possible_dani_gender)

# Function to prompt user for input with handling missing values
def prompt_user_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input.strip():  # Check if input is not empty after stripping whitespace
            return user_input
        else:
            print("Missing value detected. Filling with mean value.")
            return np.nan  # Return NaN for missing values

# Prompt the user for input for each feature
Gender = prompt_user_input ("Enter the Gender: ")
Height = float(prompt_user_input("Enter Height: "))
Body_Mass = float(prompt_user_input("Enter Body Mass: "))
Release_Velocity = float(prompt_user_input("Enter Release Velocity: "))
Release_Height = float(prompt_user_input("Enter Release Height: "))
Reach_Over_SB = float(prompt_user_input("Enter Reach Over SB: "))
FB_TL_at_Release = float(prompt_user_input("Enter FB TL at Release: "))
LR_TL_at_Release  = float(prompt_user_input("Enter LR TL at Release : "))
Shot_Path_Length = float(prompt_user_input("Enter Shot Path Length: "))
Power_Position_Distance = float(prompt_user_input("Enter Power Position Distance: "))


# Create a dictionary with the provided values
data = {
    'Gender': [Gender],
    'Height (m)': [Height],
    'Body Mass (kg)': [Body_Mass],
    'Release Velocity ( m/s)': [Release_Velocity],
    'Release Height (m)': [Release_Height],
    'Reach Over SB (m)': [Reach_Over_SB],
    'FB TL at Release (°)': [FB_TL_at_Release],
    'LR TL at Release (°)': [LR_TL_at_Release],
    'Shot Path Length (m)': [Shot_Path_Length],
    'Power Position Distance (m)': [Power_Position_Distance]
    }

# Create the DataFrame
new_data = pd.DataFrame(data)

# Handle missing values by replacing them with the mean of each feature
new_data.fillna(X.mean(), inplace=True)

# If there are categorical variables, encode them
new_data["Gender"] = le_dani_gender.transform(new_data["Gender"])

# Predict on new data
new_predictions = gb_regressor.predict(new_data)

print('**************************************')

print("The estimated distance is:", new_predictions)
