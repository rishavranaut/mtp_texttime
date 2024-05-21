import pandas as pd
import torch

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


torch.cuda.set_device(0)
device = torch.device("cuda:0")  # Set the device


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model = model.to(device) 
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

df = pd.read_csv('/home/sojitra_2211mc15/Daivik/TextTime/test.csv',header=None)
df = df.iloc[1:]
print(df.head())
print(df.values.tolist()[0])


labels=[]
responselist=[]

for i in range(len(df.values.tolist())):
    print('curr sample: {}'.format(i))
    # if i==30:
    #     break
    old_content=df.values.tolist()[i][1]
    old_date=df.values.tolist()[i][2]
    new_content=df.values.tolist()[i][3]
    new_time=df.values.tolist()[i][4]
    label=df.values.tolist()[i][5]

    #Given two versions of the text namely Version1 and Version2 having generation date namely Version1Date and Version2Date respectively. Your task is to find out weather the given Version2, Version1 is obsolete or not and needs to be updated. If provided Version2, Version1 is obsolete then return Yes otherwise retuen No as answer.
    prompt= """Given two versions of the text namely Version1 and Version2 having generation date namely Version1Date and Version2Date respectively. Your task is to find out weather the given Version2, Version1 is obsolete or not and needs to be updated. If provided Version2, Version1 is obsolete then return Yes otherwise retuen No as answer.
Version1:"""+str(old_content)+"""
Version1Date:"""+str(old_date)+"""
Version2:"""+str(new_content)+"""
Version2Date:"""+str(new_time)+"""
Answer:"""

    end_sequence="###"
    print('***********PROMPT**************')
    print(prompt)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs,max_length=700)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
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

print(f1_score(labelssk, responselist, average='macro'))

print(f1_score(labelssk, responselist, average='micro'))

print(f1_score(labelssk, responselist, average='weighted'))

accuracy = accuracy_score(responselist, labelssk)
print(f"Accuracy: {accuracy:.4f}")
