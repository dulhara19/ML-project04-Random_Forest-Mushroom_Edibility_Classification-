from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("jlh/uci-mushrooms")

# Check available splits
print(dataset)

df = pd.DataFrame(dataset ['train'])
print(df)