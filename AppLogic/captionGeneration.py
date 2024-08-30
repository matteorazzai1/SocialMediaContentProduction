import pandas as pd

from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from AppLogic.dbManager import retrieveKMostSimilarPost


def performRequest(prompt):
    client = Groq(
        api_key='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'
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content":
                    f"{prompt}",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

def generateCaption(firm_name,field_name):

    caption = performRequest(f"Can you suggest me a caption for a post related to" + field_name + " for my firm called" + firm_name + "?"
                            "But not a list, a single caption that present the field and the firm in the best way possible, without talking about the caption, I want only the caption in response from you"
                            "And these are some post example:"+retrieveKMostSimilarPost(firm_name,field_name))
    return caption


#def generate_actions(df,index_name):
#    for i, message in df.iterrows():
#        yield {
#            "_index": index_name,
#            "_id": i,
#            "_source": {
#                "status_message": message['status_message']
#            }
#        }
#
#def retrieveKMostSimilarPost(firm_name,field_name):
##    # Load the CSV file
##    df = pd.read_csv("../dataset/output_status_messages.csv")
##
##    # Initialize the Sentence Transformer model
##    model = SentenceTransformer('all-MiniLM-L6-v2')
##
##    # Generate embeddings for the status messages
##    embeddings = model.encode(df['status_message'].tolist())
##
##    # Function to retrieve the top K most similar posts
##
##    # Create a search query combining firm_name and field_name
##    query = f"Post related to {field_name} for the firm called {firm_name}"
##
##    # Generate an embedding for the query
##    query_embedding = model.encode([query])
##
##    # Compute cosine similarity between the query embedding and all message embeddings
##    similarities = cosine_similarity(query_embedding, embeddings)
#
#    # Get the indices of the top K most similar messages
#    k=5
#    top_k_indices = np.argsort(similarities[0])[-k:][::-1]
#
#    # Retrieve the top K most similar messages
#    retrieved_messages = df.iloc[top_k_indices]['status_message'].tolist()
#
#    print(retrieved_messages)
#    return retrieved_messages
#