### Meta-variables

# Heatmap script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
# task:
0: RQ2 (Heatmap)
1: RQ2 (Table)
-: tsfresh
"""


TASK = 1

if TASK == 0:
    """
    # Sample data
    np.random.seed(42)
    variable_A = np.arange(1, 11)  # 10 values for Variable A
    variable_B = np.arange(1, 11)  # 10 values for Variable B
    accuracy_values = np.random.rand(10, 10)  # Random accuracy values between 0 and 1

    # Creating DataFrame
    df = pd.DataFrame(accuracy_values, index=variable_B, columns=variable_A)
    """

    # Create Dataframe
    df = pd.read_csv('RQ2_short.csv')
    df = df.pivot(index="alphabet_size", columns="n_segments", values="test_acc")

    df = df.iloc[::-1]
    print(df.head())

    # Convert to numeric, forcing errors to NaN
    #df = df.apply(pd.to_numeric, errors='coerce')

    #print(df.head())

    # Plotting heatmap
    plt.figure(figsize=(8, 4))
    plt.imshow(df, cmap='viridis', aspect='auto')

    # Adding color bar
    cbar = plt.colorbar()
    cbar.set_label('Accuracy')

    """ 
    # variable_A = df.index.tolist()
    # variable_A = df.iloc[:, 1].tolist()
    # variable_B = df.iloc[:, 0].tolist() 
    """

    variable_A = df.columns.tolist() #seg
    variable_B = df.index.tolist() # alph

    print(variable_A)
    print(variable_B)

    # Labeling axes
    plt.xticks(ticks=np.arange(len(variable_A)), labels=variable_A)
    plt.yticks(ticks=np.arange(len(variable_B)), labels=variable_B)
    plt.xlabel('Segment size')
    plt.ylabel('Alphabet size')
    plt.title('Heatmap of Accuracy')

    # Adding text annotations
    for i in range(len(variable_B)):
        for j in range(len(variable_A)):
            color = 'white' if df.iat[i,j] < 95 else 'black'
            plt.text(j, i, f"{df.iloc[i, j]}", ha='center', va='center', color=color)
                    #  bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.2'))
            # plt.text(j, i, f"{df.iloc[i, j]:.2f}", ha='center', va='center', color='white')

    plt.show()

    plt.savefig("plot.pdf", transparent=True)

if TASK == 1:



    # Function to truncate text to 38 characters
    def truncate_text(text, max_length=35):
        if isinstance(text, str) and len(text) > max_length:
            return text[:max_length] + ".."
        return text


    # Read the CSV files
    df1 = pd.read_csv("tsf_best_0.csv", sep=";")  # First CSV file
    df2 = pd.read_csv("tsf_worst_1.csv", sep=";")  # Second CSV file

    # Concatenate them column-wise
    df_combined = pd.concat([df1, df2], axis=1)

    df_combined = df_combined.map(truncate_text)

    # Convert to LaTeX table
    latex_table = df_combined.to_latex(index=False, escape=True)

    with open("RQ1_tsf_global.tex", "w") as f:
        f.write(latex_table)
        
    """ 
    # Create Dataframe
    df = pd.read_csv('tsf_best_0.csv')
    df = df.pivot(index="alphabet_size", columns="n_segments", values="test_acc")

    df = df.iloc[::-1]
    print(df.head())

    # Convert to numeric, forcing errors to NaN
    #df = df.apply(pd.to_numeric, errors='coerce')

    #print(df.head())

    # Plotting heatmap
    plt.figure(figsize=(8, 4))
    plt.imshow(df, cmap='viridis', aspect='auto')

    # Adding color bar
    cbar = plt.colorbar()
    cbar.set_label('Accuracy')

    """ 
    # variable_A = df.index.tolist()
    # variable_A = df.iloc[:, 1].tolist()
    # variable_B = df.iloc[:, 0].tolist() 
    """

    variable_A = df.columns.tolist() #seg
    variable_B = df.index.tolist() # alph

    print(variable_A)
    print(variable_B)

    # Labeling axes
    plt.xticks(ticks=np.arange(len(variable_A)), labels=variable_A)
    plt.yticks(ticks=np.arange(len(variable_B)), labels=variable_B)
    plt.xlabel('Segment size')
    plt.ylabel('Alphabet size')
    plt.title('Heatmap of Accuracy')

    # Adding text annotations
    for i in range(len(variable_B)):
        for j in range(len(variable_A)):
            color = 'white' if df.iat[i,j] < 95 else 'black'
            plt.text(j, i, f"{df.iloc[i, j]}", ha='center', va='center', color=color)
                    #  bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.2'))
            # plt.text(j, i, f"{df.iloc[i, j]:.2f}", ha='center', va='center', color='white')

    plt.show()

    plt.savefig("plot.pdf", transparent=True)
 """

elif TASK == 2:
    df = pd.read_csv('RQ2_full.csv')
    latex_code = df.to_latex(
        index=False,
        float_format="%.2f")
    with open("RQ2_full.tex", "w") as file:
        file.write(latex_code)
