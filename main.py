# ================================
# TITANIC DATA CLEANING + EDA
# ================================

# Import libraries used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset (from URL)
# -------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("First 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# -------------------------------
# 2. Data Cleaning
# -------------------------------

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].astype('category').cat.codes

# Drop unnecessary columns
df.drop(columns=['Name', 'Ticket'], inplace=True)

print("\nCleaned Data:")
print(df.head())


# -------------------------------
# 3. Exploratory Data Analysis
# -------------------------------

sns.set(style="whitegrid")

# --- Survival Count ---
plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.show()


# --- Gender vs Survival ---
plt.figure()
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Gender vs Survival (0 = Male, 1 = Female)")
plt.show()


# --- Passenger Class vs Survival ---
plt.figure()
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.show()


# --- Age Distribution ---
plt.figure()
df['Age'].hist(bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# --- Correlation Heatmap ---
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
