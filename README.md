🌸 Iris Dataset Analysis

This project demonstrates data loading, exploration, analysis, and visualization using Python libraries such as Pandas, Seaborn, Matplotlib, and Scikit-learn. The dataset used is the classic Iris flower dataset, which is often used for machine learning and data science practice.

📂 Project Structure

Task 1: Load and Explore the Dataset

Loads the Iris dataset from sklearn.datasets

Converts it into a Pandas DataFrame

Displays the first few rows of the dataset

Checks dataset info, data types, and missing values

Cleans the dataset (removes missing values if any)

Task 2: Basic Data Analysis

Computes summary statistics (.describe())

Groups data by species (target) and computes average petal length

Provides insights into differences among Iris species

Task 3: Data Visualization

Line Chart – Sepal vs. Petal Length across samples

Bar Chart – Average Petal Length per Species

Histogram – Distribution of Sepal Width

Scatter Plot – Sepal Length vs. Petal Length, color-coded by species

🛠 Libraries Used

Pandas → Data handling and analysis

Seaborn → High-level plotting with attractive styles

Matplotlib → Customizable plots and visualizations

Scikit-learn → Provides the Iris dataset

Install dependencies with:

pip install pandas seaborn matplotlib scikit-learn

▶️ How to Run

Clone this repository or copy the script into a .py file.

Run the script:

python iris_analysis.py


Follow the printed output and view the generated charts.

📊 Example Visualizations

Line Chart: Sepal vs Petal length across samples

Bar Chart: Average petal length per species

Histogram: Sepal width distribution

Scatter Plot: Sepal length vs Petal length (species highlighted)

🌍 Insights

Setosa has the shortest petals on average.

Virginica species generally have the largest petals.

The dataset shows clear separability between species, making it useful for classification tasks.
