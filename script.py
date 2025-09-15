import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Task 1: Load and Explore
try:
    print("=== Task 1: Load and Explore the Dataset ===")

    # Load Iris dataset from sklearn and convert to DataFrame
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame

    # Show first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Check structure (data types, nulls)
    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Clean dataset (Iris has no missing values, but just in case)
    df = df.dropna()

except FileNotFoundError:
    print("Error: Dataset not found.")
except Exception as e:
    print(f"Unexpected error: {e}")


# Task 2: Basic Data Analysis
print("\n=== Task 2: Basic Data Analysis ===")

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# Grouping: Mean petal length per species
grouped = df.groupby("target")["petal length (cm)"].mean()
print("\nAverage Petal Length per Species:")
print(grouped)

# Patterns / Interesting findings
print("\nInsights:")
print("- Setosa species generally have much shorter petals compared to Versicolor and Virginica.")
print("- Virginica species tend to have the largest petals overall.")


# Task 3: Data Visualization
print("\n=== Task 3: Data Visualization ===")

sns.set(style="whitegrid")  # for prettier plots

# 1. Line chart (we'll fake a "time series" using row index for demo)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.plot(df.index, df["petal length (cm)"], label="Petal Length")
plt.title("Line Chart: Sepal vs Petal Length across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart: Avg petal length per species
plt.figure(figsize=(6, 5))
sns.barplot(x="target", y="petal length (cm)", data=df, estimator="mean", errorbar=None)
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species (0=Setosa, 1=Versicolor, 2=Virginica)")
plt.ylabel("Avg Petal Length (cm)")
plt.show()

# 3. Histogram: Distribution of Sepal Width
plt.figure(figsize=(6, 5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot: Sepal Length vs Petal Length
plt.figure(figsize=(6, 5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
