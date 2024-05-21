import pandas as pd
import torch

# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="Daivik1911/roberta-base_fact_updates")
# Load model directly


tokenizer = RobertaTokenizer.from_pretrained("Daivik1911/roberta-base_fact_updates")
model = RobertaModel.from_pretrained("Daivik1911/roberta-base_fact_updates")
torch.cuda.set_device(0)
device = torch.device("cuda:0")  # Set the device


# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model = model.to(device) 
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

df = pd.read_csv('/home/sojitra_2211mc15/Daivik/TextTime/fact-update-test-over.csv',header=None)
df = df.iloc[1:]
print(df.head())
print(df.values.tolist()[0])


labels=[]
responselist=[]

for i in range(len(df.values.tolist())):
    print('curr sample: {}'.format(i))
    if i==10:
        break
    # old_content=df.values.tolist()[i][1]
    # old_date=df.values.tolist()[i][2]
    # new_content=df.values.tolist()[i][3]
    # new_time=df.values.tolist()[i][4]
    text=df.values.tolist()[i][0]
    label=df.values.tolist()[i][1]

    #Given two versions of the text namely Version1 and Version2 having generation date namely Version1Date and Version2Date respectively. Your task is to find out weather the given Version2, Version1 is obsolete or not and needs to be updated. If provided Version2, Version1 is obsolete then return Yes otherwise retuen No as answer.
    prompt= """Given two versions of the text separated by <sod> <eod>. Your task is to find out weather there is update in numbers, dates, scores,  episodes or status in Version2 compared to Version1 text it means there is a fact update. If there is a fact update in Version2 compared to Version1 then return Yes otherwise return No.
text:"""+str(text)+"""
Answer:"""

    # prompt = str(text)

    end_sequence="###"
    print('***********PROMPT**************')
    print(prompt)
    try:
        # text = "Kederis proclaims innocence Olympic champion Kostas Kederis today left hospital ahead of his date with IOC inquisitors claiming his innocence and vowing: quot;After the crucifixion comes the resurrection. quot; .."
        inputs = tokenizer(prompt, return_tensors='pt')
        print(inputs)
        input('enter')
        response = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        print(response)
        input('enter')
        # result = classifier(prompt)

        # response = result[0]["label"]
        print(f"Predicted label: {response}")
        # print(response)
        print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
        #response=response.split('Final Label:')[1]
        print(response)
        print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')

        print('**************')
        if 'yes' in response.lower():
            print('res: {}'.format(response))
            responselist.append(1)
        else:
            print('res: {}'.format(response))
            responselist.append(0)

        print(responselist)
        labels.append(label)
        print(label)
    except:
        continue



import pickle
with open('TT_T_Full_flan.pkl', 'wb') as f:
	pickle.dump(responselist, f)

with open('TT_Tru_Full_flan.pkl', 'wb') as f:
    pickle.dump(labels, f)

print(labels)
labelssk=[]

for i in range(len(labels)):
	print(labels[i])
	if labels[i] == '0':
		labelssk.append(0)		
	elif labels[i] == '1':
		labelssk.append(1)
		

print(responselist)
print(labelssk)

# print(f1_score(labelssk, responselist, average='macro'))

# print(f1_score(labelssk, responselist, average='micro'))

# print(f1_score(labelssk, responselist, average='weighted'))

# accuracy = accuracy_score(responselist, labelssk)
# print(f"Accuracy: {accuracy:.4f}")
