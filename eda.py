import pandas as pd

# Load the dataset
file_path = "dataset-miniproject.csv"  
df = pd.read_csv(file_path)


print("Basic Information:")
print(df.info())
