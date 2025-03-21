### Meta-variables

# Heatmap script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
np.random.seed(42)
variable_A = np.arange(1, 11)  # 10 values for Variable A
variable_B = np.arange(1, 11)  # 10 values for Variable B
accuracy_values = np.random.rand(10, 10)  # Random accuracy values between 0 and 1

# Creating DataFrame
df = pd.DataFrame(accuracy_values, index=variable_B, columns=variable_A)

# Plotting heatmap
plt.figure(figsize=(8, 6))
plt.imshow(df, cmap='viridis', aspect='auto')

# Adding color bar
cbar = plt.colorbar()
cbar.set_label('Accuracy')

# Labeling axes
plt.xticks(ticks=np.arange(len(variable_A)), labels=variable_A)
plt.yticks(ticks=np.arange(len(variable_B)), labels=variable_B)
plt.xlabel('Variable A')
plt.ylabel('Variable B')
plt.title('Heatmap of Accuracy')

# Adding text annotations
for i in range(len(variable_B)):
    for j in range(len(variable_A)):
        plt.text(j, i, f"{df.iloc[i, j]:.2f}", ha='center', va='center', color='white')

plt.show()