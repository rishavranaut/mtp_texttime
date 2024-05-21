import pandas as pd
from datasets import Dataset

# Step 1: Read data from CSV
data = pd.read_csv("/home/sojitra_2211mc15/Daivik/TextTime/FLAN-T5/Fact-Update/Augmented_Balanced_Fact_updates_train.csv")

for i in range(len(data['text'])):
    data['version1'][i] = data['text'][i].split("<eod>\n<sod>")[0].split("<sod>")[1]
    data['version2'][i] = data['text'][i].split("<eod>\n<sod>")[1].split("<eod>")[0]

data.drop('id', axis=1, inplace=True)
data.drop('text', axis=1, inplace=True)

print(data.head())
# Step 2: Create Hugging Face Dataset
# hf_dataset = Dataset.from_pandas(data)

# # Step 3: Push dataset to Hugging Face Hub
# dataset_name = "Daivik1911/Fact-Updates-versions"
# #dataset_description = "This is a dataset of fact updates in wikipedia revisions having two versions of pages of wekipedia for binary classifiaction of fact updates."

# # Push dataset to Hugging Face Hub
# hf_dataset.push_to_hub(dataset_name)