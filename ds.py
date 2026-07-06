# ===========================================================
# LOAN APPLICATION DATA ANALYSIS
# Dataset : loan_prediction.csv
# Libraries Used : Pandas, Matplotlib
# ===========================================================

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------

df = pd.read_csv("loan_prediction.csv")

# -----------------------------------------------------------
# Display Dataset Information
# -----------------------------------------------------------

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# -----------------------------------------------------------
# Applicant Details
# -----------------------------------------------------------

print("\nEducation Count")
print(df["education"].value_counts())

print("\nSelf Employed Count")
print(df["self_employed"].value_counts())

print("\nLoan Status Count")
print(df["loan_status"].value_counts())

print("\nAverage Income :", df["income_annum"].mean())

print("Maximum Income :", df["income_annum"].max())

print("Minimum Income :", df["income_annum"].min())

print("\nAverage Loan Amount :", df["loan_amount"].mean())

print("Average CIBIL Score :", df["cibil_score"].mean())

# ===========================================================
# CHART 1 : Education Distribution
# ===========================================================

df["education"].value_counts().plot(kind="bar")

plt.title("Education Distribution")
plt.xlabel("Education")
plt.ylabel("Count")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 2 : Self Employed Distribution
# ===========================================================

df["self_employed"].value_counts().plot(kind="bar")

plt.title("Self Employed Distribution")
plt.xlabel("Self Employed")
plt.ylabel("Count")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 3 : Loan Status Distribution
# ===========================================================

df["loan_status"].value_counts().plot(kind="bar")

plt.title("Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 4 : Loan Status Pie Chart
# ===========================================================

df["loan_status"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Loan Status Percentage")
plt.ylabel("")

plt.show()

# ===========================================================
# CHART 5 : Income Distribution
# ===========================================================

plt.hist(df["income_annum"], bins=20)

plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 6 : Loan Amount Distribution
# ===========================================================

plt.hist(df["loan_amount"], bins=20)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 7 : CIBIL Score Distribution
# ===========================================================

plt.hist(df["cibil_score"], bins=20)

plt.title("CIBIL Score Distribution")
plt.xlabel("CIBIL Score")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 8 : Education vs Loan Status
# ===========================================================

pd.crosstab(
    df["education"],
    df["loan_status"]
).plot(kind="bar")

plt.title("Education vs Loan Status")
plt.xlabel("Education")
plt.ylabel("Count")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 9 : Self Employed vs Loan Status
# ===========================================================

pd.crosstab(
    df["self_employed"],
    df["loan_status"]
).plot(kind="bar")

plt.title("Self Employed vs Loan Status")
plt.xlabel("Self Employed")
plt.ylabel("Count")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 10 : CIBIL Score vs Loan Amount
# ===========================================================

plt.scatter(
    df["cibil_score"],
    df["loan_amount"]
)

plt.title("CIBIL Score vs Loan Amount")
plt.xlabel("CIBIL Score")
plt.ylabel("Loan Amount")
plt.grid(True)

plt.show()

# ===========================================================
# CHART 11 : Correlation Heatmap
# ===========================================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

correlation = numeric_df.corr()

plt.imshow(correlation, cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.show()

print("\n===================================")
print("Loan Application Data Analysis Completed Successfully!")
print("===================================")