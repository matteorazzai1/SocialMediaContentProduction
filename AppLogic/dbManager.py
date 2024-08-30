import os
from supabase import create_client, Client
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import vecs
from vecs import IndexMethod, IndexMeasure

url: str = "https://cjiypdsqhtnfdzshiral.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNqaXlwZHNxaHRuZmR6c2hpcmFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjQ5NTIyNzIsImV4cCI6MjA0MDUyODI3Mn0.Bg6w5UcVLB4lM8e_1FfijZceDxCDsAijjqFYKjrqXek"
DB_CONNECTION: str = "postgresql://postgres.cjiypdsqhtnfdzshiral:3zFR6lPgV4Jp3PUJ@aws-0-eu-central-2.pooler.supabase.com:6543/postgres"


def insertData():
    vx = vecs.create_client(DB_CONNECTION)
    vx.delete_collection(name='posts')
    docs=vx.get_or_create_collection('posts',dimension=384)

    docs.create_index(
        method=IndexMethod.auto,
        measure=IndexMeasure.cosine_distance
    )

    df = pd.read_csv("../dataset/output_status_messages.csv")
    iterator = df.iterrows()


    for index,row in iterator:
        print(index)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(row["status_message"])
        docs.upsert([(index,embedding,{"message": row["status_message"]})])

def retrieveKMostSimilarPost(firm_name,field_name):

    vx = vecs.create_client(DB_CONNECTION)
    docs = vx.get_or_create_collection(name="posts", dimension=384)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(firm_name+" "+ field_name)

    results = docs.query(
        data=embedding,
        limit=3,
        include_metadata=True
    )

    return handleResults(results)

def handleResults(results):

    string = ""

    for result in results:
        string += result[1].get("message") +"\n"

    return string


#insertData()
#print(retrieveKMostSimilar("I want to upload my beatiful book collection on Internet"))
