import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv

path = "/home/sojitra_2211mc15/Daivik/TextTime/edit_intention_dataset.csv"
df = pd.read_csv(path,header=None)
df = df.iloc[1:]

filepathV2 = "fact-update-final.csv"
f = csv.writer(open(filepathV2, "w+"))

f.writerow(["wiki-ref","old_content","old_time","new_content","new_time","Ground-Truth"])

for line in range(len(df.values.tolist())):

    ref = str(df.values.tolist()[line][0])

    # URL of the Wikipedia page
    url = "https://en.wikipedia.org/wiki/WP:Labels?diff=" + ref

    # Send a GET request to the URL
    response = requests.get(url)

    try:
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the content div
            old_time = soup.find(id="mw-diff-otitle1")
            old_timestamp = old_time.find(class_="mw-diff-timestamp")
            old_time = old_timestamp['data-timestamp']
            new_time = soup.find(id="mw-diff-ntitle1")
            new_timestamp = new_time.find(class_="mw-diff-timestamp")
            new_time = new_timestamp['data-timestamp']

            old_content = soup.find(class_="diff-deletedline diff-side-deleted").get_text()

            new_content = soup.find(class_="diff-addedline diff-side-added").get_text()
            
            # if content_div:
            #     # Extract the text content
            #     text_content = content_div.get_text()
                
            #     # Print the extracted content
            #     print(text_content)
            if old_time:
                print(old_time)
                print("#######@@@@@")
                print(new_time)
                print("#######@@@@@")
                print(old_content)
                print("#######@@@@@")
                print(new_content)
                if '1' in str(df.values.tolist()[line][1]):
                    f.writerow([ref,old_content,old_time,new_content,new_time,1])
                else:
                    f.writerow([ref,old_content,old_time,new_content,new_time,0])
            else:
                print("Content div not found.")
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)
    except:
        continue
