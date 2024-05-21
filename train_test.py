import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original CSV file
data = pd.read_csv("/home/sojitra_2211mc15/Daivik/TextTime/fact-update-final.csv")

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and test sets to separate CSV files
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)
