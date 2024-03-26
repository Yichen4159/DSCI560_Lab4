import threading
import time
import random
import pandas as pd
import hashlib
import re
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from selenium import webdriver
from bs4 import BeautifulSoup
from rake_nltk import Rake
import csv
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from PIL import Image
from io import BytesIO
import pytesseract
import mysql.connector
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=5, random_state=42)
model = Doc2Vec(vector_size=20, min_count=2, epochs=50)

# Function to hash a given value using SHA-256
def hash_post_id(value):
    # Convert the value to a string and encode it to bytes, then hash it using SHA-256
    return hashlib.sha256(str(value).encode()).hexdigest()

def clean_html_special_chars(text):
    # Remove HTML tags using a regular expression
    text_clean = re.sub(r'<[^>]+>', '', text)
    # Remove special characters using a regular expression
    # You can adjust the pattern to keep any characters you don't want to remove
    text_clean = re.sub(r'[^\w\s]', '', text_clean)
    return text_clean

# Transfor Objects to lists
def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []

# Data cleaning
def clean_keywords(keyword_list):   
    # Create an empty list
    cleaned_keywords = []
    stop_words = set(stopwords.words('english'))
    for phrase in keyword_list:
        # Regural expression
        words = re.findall(r'\w+', phrase)
        # Filter out stop words
        cleaned_words = [word for word in words if word.lower() not in stop_words]
        # Extend the clean words list
        cleaned_keywords.extend(cleaned_words)
    return cleaned_keywords

# Function to extract text from image using pytesseract
def extract_text_from_image(image_content):
    try:
        image = Image.open(BytesIO(image_content))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None
# Function to scrape the image and extract text for each post
def scrape_and_extract_text(row):
    post_link = row['post_link']
    time.sleep(1)

    
    # Send a request to the post link and parse the HTML content
    response = requests.get(post_link)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the image with id 'post-image'
    post_image = soup.find('img', {'id': 'post-image'})
    
    if post_image:
        # Get the image source URL
        image_url = post_image.get('src')
        
        # Fetch the image content directly
        image_response = requests.get(image_url)
        image_content = image_response.content
        # print(image_content)
        
        # Extract text from the image content using pytesseract
        text_from_image = extract_text_from_image(image_content)

        
        # Return the extracted text
        return str(text_from_image)
    return None


def update_database(interval_minutes):
    #while True:
        print("Fetching the data.....")

        # Set up the Selenium WebDriver
        driver = webdriver.Chrome()  # You can configure the driver as needed

        # URL of the webpage
        url = "https://www.reddit.com/r/tech/"

        # Open the webpage with Selenium
        driver.get(url)

        # Initialize variables for scrolling
        scroll_js = "window.scrollTo(0, document.body.scrollHeight);"
        scroll_interval = 1  # Time interval between scrolls (in seconds)
        scroll_attempts = 0
        time_limit = 10  # Time limit for scrolling in seconds

        # Get the start time
        start_time = time.time()

        # Scroll down the webpage until the time limit is reached
        while True:
            driver.execute_script(scroll_js)
            time.sleep(scroll_interval)
            scroll_attempts += 1
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Check if the time limit is reached
            if elapsed_time >= time_limit:
                break

        # Get the updated page source after scrolling
        page_source = driver.page_source

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(page_source, "html.parser")

        # Initialize a list to store post names
        post_names_tech = []
        post_id_tech = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/r/tech/comments/'):
                # Split the URL by "/" and get the last part
                url_parts = href.split("/")
                if len(url_parts) > 3:
                    comment_part = url_parts[5]  # Get the part after "/r/tech/comments/"
                    id_part = url_parts[4]
                    if comment_part not in post_names_tech:
                        post_names_tech.append(comment_part)
                        post_id_tech.append(id_part)


        # Return the number of posts
        num_posts = len(post_names_tech)
        print("Number of Posts:", num_posts)

        # Close the WebDriver
        driver.quit()
        print("Data is been successfully fetched from the Web.....")
        print("Preprocessing data.....")

        timestamp_elements = soup.find_all(attrs={"created-timestamp": True})

        # Extract and print the attribute values
        timestamps = [element["created-timestamp"] for element in timestamp_elements]

        post_links = [f"https://www.reddit.com/r/tech/comments/{post_id}/{name}/" for post_id, name in zip(post_id_tech, post_names_tech)]
        post_title_elements = soup.find_all('a', id=lambda x: x and x.startswith('post-title-'))
        post_titles = [element.text.strip() for element in post_title_elements]


        rake_nltk_var = Rake()
        keyword_lst = []
        for title in post_titles:
            rake_nltk_var.extract_keywords_from_text(title)
            keyword_extracted = rake_nltk_var.get_ranked_phrases()
            keyword_lst.append(keyword_extracted)


        data = [{'post_id_tech': post_id, 'formatted_post_title': title, 'formatted_timestamps': timestamp, 'post_link': link, 'keyword': keyword}
                for post_id, title, timestamp, link, keyword in zip(post_id_tech, post_titles, timestamps, post_links, keyword_lst)]

        # Define CSV file path
        csv_file = 'posts.csv'
        # Write data to CSV file
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['post_id_tech', 'formatted_post_title', 'formatted_timestamps', 'post_link', 'keyword']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        #print(f"CSV file '{csv_file}' has been created with the data.")


        df = pd.read_csv('posts.csv')
        df['post_id_tech'] = df['post_id_tech'].apply(hash_post_id)
        df['formatted_post_title'] = df['formatted_post_title'].apply(clean_html_special_chars)



        # Apply the function to each row and create a new column 'extracted_text'
        df['extracted_text'] = df.apply(scrape_and_extract_text, axis=1)
        #df['formatted_extracted_text'] = df['extracted_text'].apply(clean_html_special_chars)
        # Create a program and attach shaders
        df['keyword'] = df['keyword'].apply(string_to_list)

        # Apply cleaning function to keyword column
        df['keyword'] = df['keyword'].apply(clean_keywords)

        keywords_list = []


        for text in df['extracted_text']:
            rake_nltk_var.extract_keywords_from_text(str(text))
            keywords = rake_nltk_var.get_ranked_phrases()
            keywords_list.append(str(keywords))

        # Add the list of keywords to a new column 'image_keywords'
        df['image_keywords'] = keywords_list

        df['image_keywords'] = df['image_keywords'].apply(string_to_list)
        df['image_keywords'] = df['image_keywords'].apply(clean_keywords)
        # Merge short sentences from the list into a long string, as TfidfVector requires string input
        df['keyword_joined'] = df['keyword'].apply(lambda x: ' '.join(x)) + ' ' + df['image_keywords'].apply(lambda x: ' '.join(x))

        # Initialize TFIDF vectorizer
        tfidf = TfidfVectorizer()

        # Applying TFIDF Vectorizer to Keyword_Joined Column
        tfidf_matrix = tfidf.fit_transform(df['keyword_joined'])

        # Retrieve the index of the maximum TFIDF value for each document
        max_tfidf_indices = tfidf_matrix.argmax(axis=1)

        # Extract the word corresponding to the maximum TFIDF value of each row from the feature names of the TFIDF vectorizer
        topics = [tfidf.get_feature_names_out()[max_tfidf_indices[i, 0]] for i in range(tfidf_matrix.shape[0])]

        # Add the extracted keywords to the DataFrame
        df['topic'] = topics

        # Apply the function to each row and create a new column 'extracted_text'
        df['extracted_text'] = df.apply(scrape_and_extract_text, axis=1)

        df.to_csv('processed_data.csv', index=False)

        print(f"Data has been successfully preprocessed!!!")
        display_clusters()

        #Insert Records to DB
        

        for index, row in df.iterrows():
            post_id = row['post_id_tech']
            select_query = f"SELECT * FROM Posts WHERE post_id = '{post_id}'"
            cursor.execute(select_query)
            existing_record = cursor.fetchone()
            if existing_record:
            # If the record exists, update it
                update_query = f"UPDATE Posts SET post_title = '{row['formatted_post_title']}', extract_text_from_image = '{row['extracted_text']}', keywords = '{row['keyword_joined']}', topic = '{row['topic']}' WHERE post_id = '{post_id}'"
                cursor.execute(update_query)
            else:
            # If the record doesn't exist, insert a new one
                insert_query = f"INSERT INTO Posts (post_id, post_title, extract_text_from_image, keywords, topic) VALUES ('{post_id}', '{row['formatted_post_title']}', '{row['extracted_text']}', '{row['keyword_joined']}', '{row['topic']}')"
                cursor.execute(insert_query)
		#print(f"New record with post_id {post_id} inserted.")

	    # Commit the changes
            connection.commit()

        print("Values are successfully updated in the database.")
        time.sleep(60 * interval_minutes)  # Convert minutes to seconds


def display_clusters():
    # Display clusters and messages
    # Load the CSV file
    global kmeans, model,X_r
    df = pd.read_csv('processed_data.csv')

    # Assuming 'formatted_post_title' is the column with text data
    data = df['formatted_post_title'].tolist()

    # Preprocess the documents and create TaggedDocuments, using post_id_tech as tags if needed
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(data)]

    # Train the Doc2Vec model (adjust parameters as needed)
    #model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Infer vectors for the original dataset (or any new data)
    document_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in data]

    # Convert Doc to numpy array
    X = np.array(document_vectors)

    # Set up k means classifier
    #kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    # Set up labels
    labels = kmeans.labels_

    # Attach Labels
    clustered_data = pd.DataFrame(data, columns=['document'])
    clustered_data['cluster'] = labels
    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # Print Cluster
    print(clustered_data)

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Deploy PCA
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)

    # Create matplotlib graphs
    plt.figure(figsize=(12, 8))

    for i in range(5):
        plt.scatter(X_r[labels == i, 0], X_r[labels == i, 1], label=f'Cluster {i}')

    plt.legend()

    plt.title('K-Means Clustering')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.show()

    # # Export csv 
    clustered_data.to_csv('clustered_data.csv', index=False)
    #return X_r

def input_thread(interval_minutes):
    global kmeans,model, X_r
    try:
        while True:
            user_input = input().strip()
            
            if user_input.lower() == "quit":
            	if connection.is_connected():
            	    cursor.close()
            	    connection.close()
            	os._exit(0)
            else:
                # Assume user input is a message or keywords
                # Preprocess the input text
                #X_r = display_clusters()
                labels = kmeans.labels_
                clustered_data = pd.read_csv('clustered_data.csv')
                cleaned_input = clean_html_special_chars(user_input)

                tokenized_input = word_tokenize(cleaned_input.lower())
                input_vector = model.infer_vector(tokenized_input)
                input_vector = input_vector.astype(np.float64)
                kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)
                # Predict the cluster for the input
                input_cluster = kmeans.predict([input_vector])[0]


                # Filter DataFrame to get messages from the selected cluster
                selected_cluster_data = clustered_data[clustered_data['cluster'] == input_cluster]['document']

                # Display messages from the selected cluster
                print("Predicted Cluster:", input_cluster)
                print(f"Messages from Cluster {input_cluster}:\n")
                for message in selected_cluster_data:
                    print(message)

                # Display graphical representation
                plt.figure(figsize=(12, 8))
                for i in range(5):
                    plt.scatter(X_r[labels == i, 0], X_r[labels == i, 1], label=f'Cluster {i}')
                plt.scatter(input_vector[0], input_vector[1], marker='x', s=100, color='red', label='Input')
                plt.legend()
                plt.title('K-Means Clustering')
                plt.xlabel('PCA Feature 1')
                plt.ylabel('PCA Feature 2')
                plt.show()
                print("Input processed successfully!")
                
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    import sys
    db_credentials = {
        'host': 'localhost',
        'user': 'root',
        'password': 'your_password',
        'database': 'post_db'
    }
	 # Connect to MySQL database
    connection = mysql.connector.connect(**db_credentials)
    cursor = connection.cursor()
    nltk.download('stopwords')
    nltk.download('punkt')

    if len(sys.argv) != 2:
        print("Usage: python filename.py <interval_minutes>")
        sys.exit(1)

    try:
        interval_minutes = int(sys.argv[1])
        print(interval_minutes)

        # Start the background thread for database updates
        update_thread = threading.Thread(target=update_database, args=(interval_minutes,))
        update_thread.start()

        # Start the input thread for user commands
        input_thread = threading.Thread(target=input_thread, args=(interval_minutes,))
        input_thread.start()

        # Wait for both threads to finish
        update_thread.join()
        input_thread.join()

    except ValueError:
        print("Error: Interval must be a valid integer.")
        sys.exit(1)



