Day 1: Data Cleaning & Preprocessing with Titanic Dataset
🎯 Objective
To learn how to clean, preprocess, and explore a dataset before feeding it into a machine learning model.

🛠 Tools & Libraries
Python

Pandas

NumPy

Seaborn / Matplotlib

Scikit-learn (for scaling)

📁 Dataset
We used the Titanic Dataset, which includes data about passengers aboard the Titanic—such as survival status, age, class, fare, etc.
You can download it from Titanic Dataset on Kaggle.

🔧 Steps Performed
1. Importing and Exploring the Data
Loaded the dataset using pandas.

Used df.info() and df.describe() to understand structure, data types, and missing values.

2. Handling Missing Values
python
Copy
Edit
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
Filled missing Age values with mean.

Filled missing Embarked values with mode.

🗓 Day 2: Exploratory Data Analysis (EDA) on Titanic Dataset
🎯 Objective
To understand the dataset better using statistics and visualizations.

🛠 Tools Used
Pandas for data manipulation

Matplotlib & Seaborn for static visualizations

Plotly (optional) for interactive visualizations

🔍 Key Steps Performed
Generated summary statistics using:

python
Copy
Edit
df.describe()
df.info()
Visualized distributions using:

Histograms

Boxplots

Analyzed relationships using:

seaborn.pairplot()

Correlation heatmap with sns.heatmap(df.corr(), annot=True)

🗓 Day 3: Housing Price Prediction using Multiple Linear Regression
🎯 Objective
Predict house prices based on multiple features using Multiple Linear Regression.

📁 Project Folder Structure
bash
Copy
Edit
house-price-prediction/
├── housing_data.csv             # dataset file
├── housing_regression.ipynb     # Google Colab / Jupyter notebook
├── README.md                    # project documentation
├── requirements.txt             # Python packages used
└── .gitignore                   # to ignore unnecessary files
🛠 Tools & Libraries
Python

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn

🔧 Key Steps
1. Data Preprocessing
Removed price column as the target variable (y)

Converted categorical columns using pd.get_dummies()

2. Train-Test Split
python
Copy
Edit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
3. Model Training
python
Copy
Edit
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
4. Model Evaluation
python
Copy
Edit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
5. Coefficients & Intercept
python
Copy
Edit
print("Intercept:", model.intercept_)
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")



