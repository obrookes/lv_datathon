# -- Initial testing/validation split

# Libraries
import pandas as pd

# Read data
full_data = pd.read_csv("data/new_train.csv")

# 20% of the training data is going to be a validation set (shuffled)
train = full_data.sample(frac = 0.8, random_state = 1)
val   = full_data.drop(train.index).sample(frac = 1, random_state = 2)

# save to csv
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)