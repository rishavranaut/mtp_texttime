import requests
import random

def get_random_revisions(start_date, end_date, sample_size):
    # Define the URL for the MediaWiki API to get recent changes
    url = f"https://www.mediawiki.org/w/api.php?action=query&format=json&list=recentchanges&rctype=edit&rclimit=500&rcstart={start_date}&rcend={end_date}"

    # Make a request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        # Extract recent changes
        changes = data['query']['recentchanges']
        # Randomly sample from the changes
        sampled_changes = random.sample(changes, min(sample_size, len(changes)))
        for change in sampled_changes:
            title = change['title']
            timestamp = change['timestamp']
            user = change['user']
            comment = change['comment']
            print(f"Title: {title}, Timestamp: {timestamp}, User: {user}, Comment: {comment}")
    else:
        print("Error occurred while fetching data")

# Example usage:
start_date = "20230601000000"  # June 1, 2023
end_date = "20240229000000"    # February 29, 2024
sample_size = 5000  # Number of revisions to sample
get_random_revisions(start_date, end_date, sample_size)
