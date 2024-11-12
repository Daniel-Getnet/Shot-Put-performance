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
xgb = XGBRegressor(
        n_estimators=101,  # [100, 500] step of 1
        learning_rate=0.3,  # [0.01, 0.3] step of 0.01
        max_depth=2,  # [1, 10] step of 1
        subsample=0.89,   # [0.5, 1] step of 0.1
        random_state=42  # Seed for the random number generator
    )

# Fit to training set
xgb.fit(train_X, train_y)

# Predict on test set
pred_y = xgb.predict(test_X)

# Compute feature importance
feature_importance = xgb.feature_importances_

# Get feature names
feature_names = dani_file.columns[:-1]

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(10, 8))  # Set figure size for better readability
plt.scatter(test_y, pred_y, color='#00a053', alpha=0.7, label='Data Points', edgecolor='black')  # Added edge color for data points

# Fit line
m, b = np.polyfit(test_y, pred_y, 1)
plt.plot(test_y, m * test_y + b, color='#febd15', linewidth=2.0, label='Fit Line')  # Increased line width for emphasis

# R-squared value
r2_test = r2_score(test_y, pred_y)
plt.title(f'Scatter Plot of Actual vs. Predicted Values\nR² = {r2_test:.4f}', fontsize=16, fontweight='bold')  # Title with larger font
plt.xlabel('Actual (m)', fontsize=14, fontweight='bold')  # X-axis label with formatting
plt.ylabel('Predicted (m)', fontsize=14, fontweight='bold')  # Y-axis label with formatting

# Add grid
plt.grid(color='gray', linestyle='--', alpha=0.7)  # Light grid for better visualization

# Adjust legend and tick parameters
plt.legend(fontsize=12)
plt.xticks(fontsize=12)  # X-axis tick labels
plt.yticks(fontsize=12)  # Y-axis tick labels

plt.tight_layout()  # Ensure layout fits well within the figure
plt.show()  # Display the plot
# Plot feature importance
plt.figure(figsize=(10, 8))

# Define a custom color gradient
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']

# Plot with a color cycle for variety in each bar
plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors * (len(importance_df) // len(colors) + 1), edgecolor='black')

plt.xlabel('Importance', fontsize=14, fontweight='bold')
plt.title('Feature Importance', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.yticks(fontsize=12)  # Increase the font size for y-axis labels
plt.tight_layout()
plt.show()

# Model evaluation 
print(f"r_square_for_the_test_dataset: {r2_score(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")