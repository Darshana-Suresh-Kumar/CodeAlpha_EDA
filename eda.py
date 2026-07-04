# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print("\nFirst 5 Rows")
print(df.head())

# Dataset information
print("\nDataset Information")
print(df.info())

# Statistical summary
print("\nStatistical Summary")
print(df.describe())

# Check missing values
print("\nMissing Values")
print(df.isnull().sum())

# Count passengers who survived
print("\nSurvival Count")
print(df["Survived"].value_counts())

# Plot Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Passenger class distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()

# Gender distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"].dropna(), bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully!")