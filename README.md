# Machine Learning Scenarios Project

This project explores various Machine Learning scenarios through multiple datasets. It demonstrates key techniques in ML, including regression, classification, and polynomial feature engineering.

## Datasets and Scenarios

The project includes analysis and models for the following datasets and objectives:

1. **CarPrice Dataset**  
   - **Objective:** Predict car prices based on various features.
   - **Techniques Used:** Linear regression, polynomial regression, feature engineering.
   - **Description:** This dataset includes car attributes such as engine size, horsepower, and weight to help build predictive models for car pricing.

2. **Iris Dataset**  
   - **Objective:** Classify iris species based on petal and sepal measurements.
   - **Techniques Used:** K-Nearest Neighbors (KNN), Decision Trees, Support Vector Machines (SVM).
   - **Description:** The classic Iris dataset, which includes measurements of iris flowers, is used to classify species based on sepal and petal features.

3. **Polynomial Multifeatures Dataset**  
   - **Objective:** Analyze the impact of polynomial features on model performance.
   - **Techniques Used:** Polynomial feature transformation, regression, cross-validation.
   - **Description:** This analysis examines how introducing polynomial features affects the accuracy and flexibility of regression models.

## Project Structure

- `data/` – Contains datasets used in this project.
- `notebooks/` – Jupyter notebooks with code for each scenario.
- `models/` – Saved models for each scenario.
- `README.md` – Project documentation.
- `requirements.txt` – List of required libraries.

## Exploratory Data Analysis (EDA)

Before diving into model development, each dataset undergoes a thorough Exploratory Data Analysis (EDA) to understand its structure, features, and potential correlations. This section provides an overview of EDA objectives and techniques for each dataset, focusing on identifying the most relevant features and metrics for model performance.

### EDA Goals

The main goals of EDA in this project include:

1. **Identifying Key Features**: Understanding which features are most relevant for prediction or classification in each dataset.
2. **Detecting Outliers and Missing Values**: Identifying any outliers, missing values, or data inconsistencies that may affect model performance.
3. **Visualizing Feature Distributions and Relationships**: Using visualizations to reveal patterns, distributions, and correlations within the data.
4. **Determining Evaluation Metrics**: Selecting appropriate metrics (e.g., MAE, accuracy, F1-score) based on the type of model and dataset.

### EDA Techniques

The following techniques are commonly applied in the EDA phase for each dataset:

- **Statistical Summaries**: Generating summary statistics to understand distributions, central tendency, and variability in each feature.
- **Correlation Analysis**: Calculating correlation coefficients to examine relationships between features, especially for regression tasks.
- **Feature Histograms and Box Plots**: Plotting feature distributions to identify skewness, outliers, and data ranges.
- **Pair Plots and Scatter Plots**: Visualizing relationships between features for potential multivariate relationships.

### Key Insights from EDA by Dataset

#### 1. CarPrice Dataset
   - **Key Features**: Engine size, horsepower, weight, fuel type, and make.
   - **Notable Findings**: Engine size and horsepower show a strong correlation with car price, while categorical features like fuel type are essential for subgroup analysis.
   - **Metric**: Mean Absolute Error (MAE) for regression.

#### 2. Iris Dataset
   - **Key Features**: Sepal length, sepal width, petal length, and petal width.
   - **Notable Findings**: Petal measurements have distinct clustering patterns among iris species, making them key features for classification.
   - **Metric**: Accuracy and F1-score for classification.

#### 3. Polynomial Multifeatures
   - **Key Features**: Derived polynomial features created from the original set to enhance model complexity and fit.
   - **Notable Findings**: Polynomial features increase model flexibility but may lead to overfitting if not regularized.
   - **Metric**: Root Mean Squared Error (RMSE) and R² for regression.

## Model Evaluation Metrics

Each model is evaluated based on the specific goals of the dataset scenario:

- **Mean Absolute Error (MAE)**: Used in regression tasks like CarPrice to measure average prediction error.
- **Accuracy**: A primary metric for classification tasks, especially in the Iris dataset.
- **F1-score**: Provides a balanced measure of precision and recall for classification.
- **Root Mean Squared Error (RMSE)** and **R² Score**: Used to assess the fit and generalization of regression models, especially in polynomial scenarios.

The results of these metrics are documented within each notebook for a transparent comparison of model performance.

---

This section on EDA provides context on how feature selection, data distribution analysis, and metric choice shape the Machine Learning pipeline in the project.

## Model Selection and Use Cases

Each dataset in this project presents unique challenges, requiring different models tailored to specific tasks. Here’s a breakdown of the models used, along with their specific applications and strengths in various scenarios.

### 1. CarPrice Dataset – Regression Models

   - **Linear Regression**: Provides a straightforward, interpretable approach to predicting car prices based on linear relationships with features like engine size and horsepower.
   - **Polynomial Regression**: Extends the linear model by introducing polynomial features, enabling a better fit for complex, non-linear relationships observed in car pricing.
   - **Ridge and Lasso Regression**: These regularized regression models are used to address overfitting by penalizing large coefficients, especially useful when polynomial features introduce high model complexity.

   **Use Case**: Regression models in the CarPrice dataset focus on accurately predicting car prices, allowing for insights into feature importance and ensuring that predictions remain stable across various car attributes.

### 2. Iris Dataset – Classification Models

   - **K-Nearest Neighbors (KNN)**: This simple yet effective model is used for its ability to classify data points based on similarity, ideal for datasets with clear class clusters, as seen in the Iris dataset.
   - **Decision Tree**: Offers interpretable classification with a clear, step-by-step decision-making structure, making it suitable for understanding the importance of features like petal width and length in iris classification.
   - **Support Vector Machine (SVM)**: Utilizes hyperplanes to separate classes with maximum margin, especially useful when classes are not easily separable, enhancing the accuracy of species classification.

   **Use Case**: Classification models applied to the Iris dataset allow us to assign species labels with high accuracy, leveraging patterns in petal and sepal measurements to differentiate between classes.

### 3. Polynomial Multifeatures – Complex Regression Models

   - **Polynomial Regression with Cross-Validation**: This model investigates the impact of polynomial features on regression performance. Cross-validation ensures model reliability by testing on multiple subsets of the data, minimizing overfitting risks.
   - **Regularized Polynomial Regression (Ridge and Lasso)**: These models handle high-dimensional polynomial features by penalizing large coefficients, balancing model complexity with prediction accuracy.

   **Use Case**: These models are applied to examine the effects of complex feature transformations (polynomial expansions) on model performance, providing insights into how added complexity can enhance or hinder predictive accuracy.

### Summary of Model Uses

- **Regression Models**: Applied primarily for continuous target variables, where predictions benefit from understanding the impact of each feature. Regularization techniques are employed to improve generalization.
- **Classification Models**: Used for categorical outcomes, focusing on models that excel with small datasets and clear feature-based separability, like in the Iris dataset.
- **Polynomial Models**: Used to test the impact of high-dimensional feature spaces on regression tasks, with an emphasis on cross-validation to manage potential overfitting.

---

This section describes the practical applications of each model type in the project, providing insight into how the choice of algorithm aligns with each dataset's characteristics and the project's analytical goals.

## Installation

Install the dependencies with:
```bash
pip install -r requirements.txt
```

## Disclaimer

The datasets used in this project are publicly available and have been obtained from open sources for educational and demonstration purposes only. No proprietary or sensitive data is used, and all analyses comply with public data usage policies. Each dataset—such as CarPrice, Iris, and Polynomial Multifeatures—serves to illustrate different Machine Learning techniques and is provided to foster learning and experimentation in data science.