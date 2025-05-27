# Data-Cleaning
#  Data Cleaning & Preprocessing for Machine Learning

This project demonstrates the essential steps to clean and prepare raw data for Machine Learning using the Titanic dataset.

## Objective

To learn how to clean, preprocess, and explore a dataset before feeding it into a machine learning model.

## 🛠 Tools & Libraries

- Python
- Pandas
- NumPy
- Seaborn / Matplotlib
- Scikit-learn (for scaling)

##  Dataset

We used the **Titanic Dataset**, which contains data about passengers aboard the Titanic, including survival status, age, class, fare, and more.

You can download the dataset from: [Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)

---

## Steps Performed

### 1. Importing and Exploring the Data
- Loaded the dataset using `pandas`.
- Used `df.info()` and `df.describe()` to understand structure, types, and missing values.

### 2. Handling Missing Values
- Filled missing `Age` values with the **mean**.
- Filled missing `Embarked` values with the **mode**.

```python
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

**
 Day 2: Exploratory Data Analysis (EDA)**
✅ Objective:
Understand the dataset using statistics and visualizations.

📊 Tools Used:
Pandas for data manipulation

Matplotlib & Seaborn for visualization

Plotly for interactive plots (optional)

🔍 Key Steps:
Generated summary statistics using df.describe() and df.info().

Visualized distributions using:

Histograms

Boxplots

Analyzed relationships using:

Pairplot

Correlation matrix
