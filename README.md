# Shot Put Performance Prediction: Machine Learning Models

This project analyzes the key biomechanical and body composition features that influence shot put performance. We used machine learning models to predict shot distances based on these features. The models implemented include Random Forest, Gradient Boosting, CatBoost, and XGBoost, which were tested and optimized to achieve the best possible predictive accuracy.

## Project Overview

The purpose of this study is to identify the most influential factors in shot put performance and predict shot distances using machine learning. By leveraging a dataset of elite shot put athletes and assessing various biomechanical and body composition features, this study provide insights that can help athletes and coaches optimize performance.

## Key Findings

- **Top Predictors**: The feature selection process identified that release velocity, gender, shot path length, and body mass are the most influential predictors of shot put performance.
- **Model Performance**: Gradient Boosting (GB) emerged as the best model, achieving an R² of 0.8248, MAE of 0.4474, and RMSE of 0.6500.
- **Evaluation Metrics**: Models were evaluated using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Dataset

The dataset consists of 37 elite shot putters, with 15 biomechanical features obtained from the World Athletics reports of the 2017 and 2018 world championships. The dataset includes both male and female athletes and has been preprocessed to fill missing values with the mean of each feature.

### Features

| Feature | Unit | Feature | Unit |
|---------|------|---------|------|
| Gender | - | Forward Backward Trunk Lean (FB TL) at Release | ° |
| Height | m | Left Right Trunk Lean (LR TL) at Release | ° |
| Body Mass | kg | Shot Path Length | m |
| Technique | - | Shot Release Height | m |
| Release Velocity | m/s | Glide Flight (G/F) Distance | m |
| Angle of Release | ° | Power Position Distance | m |
| Release Height | m | Shoulder Hip Separation Angle (SH SA) at Release | ° |
| Reach Over Step Board (SB) | m | - | - |

### Target Variable
 Distance
 ## 6. Conclusions

Analysis of feature importance in the Random Forest (RF) model identified **release velocity**, **gender**, **shot path length**, and **body mass** as the most influential predictors of shot put distance. These features significantly improved the model's accuracy, highlighting key factors in shot put performance.

Multiple machine learning models, including RF, CatBoost (CB), Gradient Boosting (GB), and XGBoost (XGB), were evaluated. The GB model outperformed others in accuracy and error metrics, making it the top performer.

The GB model's strong performance on unseen data underscores its effectiveness as a predictive tool, offering valuable insights for athletes and coaches. By applying the model with relevant data, shot put performance can be forecasted more accurately, enabling data-driven training and performance optimization.






