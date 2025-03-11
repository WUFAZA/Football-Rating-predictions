# Football-Rating-Predictions
My First ML Project

# Football Player Rating Prediction

This project predicts the ratings of football players for the upcoming year using a Linear Regression model. Linear Regression is a supervised machine learning technique ideal for regression challenges, where it predicts continuous target values based on input features.

---

## How the Code Works

1. **Data Collection:**
   - The dataset includes historical performance data of football players and their past ratings.
   - Data is sourced from: [Football Player Dataset 2022 - CLEAN FIFA 23 Official Data](https://raw.githubusercontent.com/falaprodavi/Football-Player-Dataset2022/refs/heads/main/datasets/CLEAN_FIFA23_official_data.csv).

2. **Data Preprocessing:**
   - Key target variable: `Potential`.
   - Features used include:
     - `Overall`
     - `Contract Valid Until`
     - `Height(cm.)`
     - `Weight(lbs.)`
     - `Release Clause (£)`
   - Data was split into training and testing sets and standardized for consistent scaling.

3. **Model Training:**
   - A Linear Regression model was trained on historical player data to predict potential future ratings.

4. **Prediction:**
   - The trained Linear Regression model was used to predict player ratings for test data.

5. **Evaluation:**
   - The model's performance was evaluated using:
     - **Mean Squared Error (MSE):** To measure the variance between actual and predicted ratings.
     - **R² Score:** To assess how well the model explains the variance in the data.

---

## Example Code Snippet

```python
# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib as plt

# Loading the dataset
df = pd.read_csv('https://raw.githubusercontent.com/falaprodavi/Football-Player-Dataset2022/refs/heads/main/datasets/CLEAN_FIFA23_official_data.csv')
print(df.head())  # Displaying the first few rows of the dataset

# Separating the target variable and features
y = df['Potential']
columns_to_add = ['Overall', 'Contract Valid Until', 'Height(cm.)', 'Weight(lbs.)', 'Release Clause(£)']
x = df[columns_to_add]

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
print(x_train.head())  # Displaying the first few rows of the training features

# Training the model using Linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

# Testing the model
y_old_pred = lr.predict(x_train)
y_new_pred = lr.predict(x_test)

# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score

# Calculating training performance
lr_train_mse = mean_squared_error(y_train, y_old_pred)
r2_score_train = r2_score(y_train, y_old_pred)

# Calculating testing performance
lr_test_mse = mean_squared_error(y_test, y_new_pred)
r2_score_test = r2_score(y_test, y_new_pred)

# Displaying evaluation results
print("LR Train Mean Squared Error: ", lr_train_mse)
print("LR Train R² Score: ", r2_score_train)
print("LR Test Mean Squared Error: ", lr_test_mse)
print("LR Test R² Score: ", r2_score_test)

## Dependencies

The following libraries are required to run the notebook:

- numpy
- pandas
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn
```

## Data Collection

The dataset used in this project was obtained from: Football Player Dataset 2022 - CLEAN FIFA 23 Official Data(https://raw.githubusercontent.com/falaprodavi/Football-Player-Dataset2022/refs/heads/main/datasets/CLEAN_FIFA23_official_data.csv).

It contains detailed information about football players, with features such as:

Overall, Potential, Height(cm.), Weight(lbs.), and more (see the full feature set in the dataset).

## Data Preprocessing

The data preprocessing steps include:

Extracting the target variable Potential and selecting key features (Overall, Contract Valid Until, Height(cm.), etc.).

Standardizing the data for consistent scaling.

Splitting the dataset into training and testing sets using train_test_split.
## Model Training

The model is trained using Linear Regression:

Import the Linear Regression model from scikit-learn.

Train it on the training data to learn patterns.

Use the trained model to make predictions.

## Model Evaluation

The model's performance is evaluated using:

Mean Squared Error (MSE): Measures the variance between predicted and actual values.

R² Score: Indicates how well the model explains the data variance.

## Usage

To run the project:

Open Player_Rating_Prediction.ipynb in Jupyter Notebook or Google Colab.

Install the required dependencies using pip.

Execute the code cells sequentially.
## Acknowledgements
The dataset used in this project is sourced from: FIFA 23 Official Data. Special thanks to the contributors for providing clean and detailed data.
