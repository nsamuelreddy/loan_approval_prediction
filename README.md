# Loan Approval Prediction

A Machine Learning classification project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and demographic information.

The project uses a **Decision Tree Classifier** with **RandomizedSearchCV** for hyperparameter tuning and evaluates the final model using accuracy, classification metrics, and a confusion matrix.

## Features

* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Categorical variable encoding using Label Encoding
* Removal of unnecessary features
* Train-test data splitting
* Decision Tree classification
* Hyperparameter tuning using RandomizedSearchCV
* 5-fold cross-validation
* Model evaluation using accuracy, precision, recall, and F1-score
* Confusion matrix analysis
* Feature importance analysis
* Trained model saved using Joblib

## Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **Scikit-learn** – Machine Learning and model evaluation
* **SciPy** – Hyperparameter distributions for RandomizedSearchCV
* **Joblib** – Model serialization
* **KaggleHub** – Dataset loading
* **Jupyter Notebook** – Development environment

## Dataset

The project uses the **Loan Approval Prediction Dataset** containing **4,269 records**.

After preprocessing, the dataset contains **11 input features** and one target variable, `loan_status`. The `loan_id` column was removed because it was not required for prediction.

### Input Features

| Feature                    | Description                 |
| -------------------------- | --------------------------- |
| `no_of_dependents`         | Number of dependents        |
| `education`                | Applicant education status  |
| `self_employed`            | Self-employment status      |
| `income_annum`             | Annual income               |
| `loan_amount`              | Requested loan amount       |
| `loan_term`                | Loan repayment term         |
| `cibil_score`              | Applicant CIBIL score       |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value`  | Value of commercial assets  |
| `luxury_assets_value`      | Value of luxury assets      |
| `bank_asset_value`         | Value of bank assets        |

### Target

`loan_status`

* `0` → Rejected
* `1` → Approved

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded using KaggleHub and Pandas.

### 2. Data Preprocessing

The project:

* Removes unnecessary whitespace from column names
* Removes the `loan_id` column
* Converts categorical columns into numerical values using `LabelEncoder`
* Checks for missing values
* Separates features and target

The dataset contains no missing values after loading.

### 3. Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split** with `random_state=42`.

### 4. Model Training

A **Decision Tree Classifier** is used for binary loan approval classification.

### 5. Hyperparameter Tuning

`RandomizedSearchCV` is used with **5-fold cross-validation** to search for effective Decision Tree hyperparameters.

The selected parameters were:

```text
criterion = entropy
max_depth = 19
min_samples_leaf = 6
min_samples_split = 8
```

## Model Performance

The tuned Decision Tree achieved:

**Test Accuracy: 98.59%**

The best cross-validation score obtained during hyperparameter tuning was approximately **98.13%**.

### Classification Performance

| Class    | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| Rejected |      0.99 |   0.99 |     0.99 |
| Approved |      0.98 |   0.98 |     0.98 |

The overall accuracy on the test set was approximately **99%**.

## 🔍 Feature Importance

The trained Decision Tree identified **CIBIL score** as the most influential feature, followed by loan term and loan amount.

The recorded feature importance for CIBIL score was approximately **0.773**, indicating its strong contribution to the model's predictions.

## Project Structure

```text
Loan-Approval-Prediction/
│
├── loan_approval_prediction.ipynb
├── loan_approval_decision_tree.pkl
└── README.md
```

## Run the Project

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Loan-Approval-Prediction
```

Install the required libraries:

```bash
pip install pandas scikit-learn scipy joblib kagglehub
```

Open the notebook:

```bash
jupyter notebook
```

Run the cells in order to perform data preprocessing, model training, hyperparameter tuning, and evaluation.

## Saved Model

The final tuned Decision Tree model is saved as:

```text
loan_approval_decision_tree.pkl
```

using Joblib.

## Key Takeaways

* Performed complete data preprocessing and exploratory analysis.
* Converted categorical variables into numerical representations.
* Used a Decision Tree for binary classification.
* Applied RandomizedSearchCV with 5-fold cross-validation.
* Achieved **98.59% test accuracy**.
* Analyzed model performance using classification metrics and a confusion matrix.
* Identified feature importance, with CIBIL score being the dominant feature.


## Author

**Narravula Samuel Reddy**

B.Tech – Computer Science and Engineering
Rajiv Gandhi University of Knowledge Technologies (RGUKT), RK Valley

```
```
