# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Uncomment the following line if you're using a Jupyter Notebook
# %matplotlib inline

# ----------------------------
# Task 1: Load and Explore the Dataset
# ----------------------------

try:
    # Load the dataset (ensure 'iris.csv' exists in your working directory)
    df = pd.read_csv("iris.csv")
except FileNotFoundError:
    print("Error: The file 'iris.csv' was not found. Please check the file name and path.")
    exit()  # Exit the program if file is not found

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset: data types and missing values
print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values by Column:")
print(df.isnull().sum())

# Clean the dataset: drop rows with missing values (or use fillna() as needed)
df_clean = df.dropna()

# ----------------------------
# Task 2: Basic Data Analysis
# ----------------------------

# Compute basic statistics (already partly shown by describe())
# Group the data by a categorical column ('species') and compute the mean of a numerical column ('sepal_length')
if 'species' in df_clean.columns:
    species_group = df_clean.groupby('species')['sepal_length'].mean()
    print("\nAverage Sepal Length by Species:")
    print(species_group)
else:
    print("\nThe column 'species' was not found in the dataset for grouping analysis.")

# ----------------------------
# Task 3: Data Visualization
# ----------------------------

# Visualization 1: Line chart showing trends over time
# (Since Iris doesn't have a time column, we simulate 'time' using the DataFrame index)
plt.figure(figsize=(8, 4))
plt.plot(df_clean.index, df_clean['sepal_length'], marker='o', linestyle='-', color='blue')
plt.title("Line Chart: Sepal Length Trend (Simulated Time)")
plt.xlabel("Index (Simulated Time)")
plt.ylabel("Sepal Length")
plt.grid(True)
plt.show()

# Visualization 2: Bar chart showing average petal length per species
if 'species' in df_clean.columns and 'petal_length' in df_clean.columns:
    species_avg = df_clean.groupby('species')['petal_length'].mean().reset_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(data=species_avg, x='species', y='petal_length', palette='viridis')
    plt.title("Bar Chart: Average Petal Length by Species")
    plt.xlabel("Species")
    plt.ylabel("Average Petal Length")
    plt.show()
else:
    print("Columns 'species' and/or 'petal_length' not found for bar chart.")

# Visualization 3: Histogram of a numerical column (sepal_width)
plt.figure(figsize=(8, 4))
plt.hist(df_clean['sepal_width'], bins=20, color='green', edgecolor='black')
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.show()

# Visualization 4: Scatter plot to visualize the relationship between sepal_length and petal_length
plt.figure(figsize=(8, 4))
plt.scatter(df_clean['sepal_length'], df_clean['petal_length'], alpha=0.7, color='purple')
plt.title("Scatter Plot: Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.grid(True)
plt.show()

# ----------------------------
# Findings / Observations
# ----------------------------
print("\nObservations:")
print("- The dataset shows variation in sepal and petal dimensions across different species.")
print("- The bar chart indicates that the average petal length differs among species.")
print("- The scatter plot suggests a positive correlation between sepal length and petal length.")

# End of assignment code
