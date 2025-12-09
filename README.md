# House Price Prediction with LightGBM

This project focuses on predicting house prices using machine learning techniques. It includes data cleaning, preprocessing, visualization, and model training using the LightGBM regression algorithm.

## Project Files

The repository contains the following main files:

- data_cleaning_&_visual.ipynb  
  This notebook handles data cleaning, preprocessing, encoding, missing value handling, and outlier detection.

- model_training_&_visual.ipynb  
  This notebook trains and evaluates machine learning models, with LightGBM used as the main regression model.

- data_visual.py  
  This script automatically generates visual graphs and saves them as image files inside the `visualizations` folder.

- visualizations/  
  This folder contains all generated graphs and plots.

## Data Processing Details

The dataset was processed using the following steps:

- Most missing (null) values were filled with the label "Bilinmiyor"
- Rows containing 4 or more missing values were removed
- Outliers were detected and handled
- Categorical variables were encoded using One-Hot Encoding

## Model

The main model used in this project is:

- LightGBM (LGBMRegressor)

The model was evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## How to Run the Project

1. Clone the repository:

   git clone <your-repo-url>

2. Navigate into the project folder:

   cd <your-project-folder>

3. Install the required libraries:

   pip install -r requirements.txt

4. Run the notebooks in this order:

   data_cleaning_&_visual.ipynb  
   model_training_&_visual.ipynb

5. Generate visualizations by running:

   python data_visual.py

## Notes

This project uses One-Hot Encoding for categorical features. Missing values are mostly replaced with "Bilinmiyor", with a few exceptions. Rows with 4 or more missing values are removed to improve data quality.
