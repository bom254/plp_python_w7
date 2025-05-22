import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Display basic information about the dataset
print("Dataset Overview:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Set up the plotting style
plt.style.use('seaborn')
fig = plt.figure(figsize=(15, 10))

# 1. Line chart showing trends
plt.subplot(2, 2, 1)
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.plot(species_data.index, species_data['sepal length (cm)'], 
             label=species, marker='o')
plt.title('Sepal Length by Species')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()

# 2. Bar chart comparing means
plt.subplot(2, 2, 2)
species_means = df.groupby('species')['petal length (cm)'].mean()
species_means.plot(kind='bar')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.xticks(rotation=45)

# 3. Histogram
plt.subplot(2, 2, 3)
plt.hist(df['sepal width (cm)'], bins=15, edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# 4. Scatter plot
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='sepal length (cm)', 
                y='sepal width (cm)', 
                hue='species',
                style='species')
plt.title('Sepal Length vs Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

plt.tight_layout()
plt.show()

# Calculate basic statistics for numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Group by species and calculate means
print("\nMean measurements by species:")
print(df.groupby('species').mean())

# Calculate correlations between features
print("\nCorrelation matrix:")
print(df.corr())
