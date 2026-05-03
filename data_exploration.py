import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('creditcard.csv')

# Basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check fraud vs normal
print("\nClass Distribution:")
print(df['Class'].value_counts())

# Visualize fraud vs normal
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Normal Transactions')
plt.xlabel('0 = Normal, 1 = Fraud')
plt.ylabel('Count')
plt.show()