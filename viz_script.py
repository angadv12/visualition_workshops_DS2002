# Name: Angad Brar, Computing ID: zqq4hx

import matplotlib.pyplot as plt
import seaborn as sns

# Load real-world datasets
iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")

# Set up a seaborn style for the visuals
sns.set_theme(style="whitegrid")

# 1. Iris Dataset Visualization: Pairplot
plt.figure(figsize=(10, 6))
sns.pairplot(
    iris,
    hue="species",
    diag_kind="kde",
    palette="muted"
)
plt.suptitle("Iris Dataset: Pairplot of Features", y=1.02)
plt.show()

# 2. Titanic Dataset Visualization: Survival Rate by Class and Gender
plt.figure(figsize=(10, 6))
sns.barplot(
    x="class",
    y="survived",
    hue="sex",
    data=titanic,
    ci=None,
    palette="coolwarm"
)
plt.title("Titanic Survival Rate by Class and Gender")
plt.ylabel("Survival Rate")
plt.xlabel("Class")
plt.legend(title="Gender")
plt.show()

# 3. Titanic Age Distribution by Survival Status
plt.figure(figsize=(12, 6))
sns.histplot(
    titanic,
    x="age",
    hue="survived",
    kde=True,
    bins=30,
    palette="pastel",
    element="step"
)
plt.title("Titanic Age Distribution by Survival Status")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend(title="Survived", labels=["Not Survived", "Survived"])
plt.show()

# 4. Iris Dataset Visualization: Boxplot for Sepal Dimensions
plt.figure(figsize=(10, 6))
sns.boxplot(
    x="species",
    y="sepal_length",
    data=iris,
    palette="Set2"
)
plt.title("Sepal Length Distribution Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.show()

# 5. Titanic Correlation Heatmap
plt.figure(figsize=(10, 6))
corr = titanic.corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Titanic Dataset: Correlation Heatmap")
plt.show()