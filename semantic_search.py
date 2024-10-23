import re
import pandas as pd
import string
import os
import time
import faiss
import string
import pickle
import numpy as np
import pandas as pd

from typing import List
from collections import Counter

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report,confusion_matrix
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data_df = pd.read_csv("20_newsgroup.csv")

def clean(text):
    text = str(text)
    text=text.lower()
    url_removed=re.sub(r'https\S+','',text,flags=re.MULTILINE)
    text=re.sub("[^a-zA-Z]"," ",url_removed)
    text=re.sub("\.+"," ",text)
    text=[word for word in text if word not in string.punctuation]
    text="".join(text).strip()
    text=re.sub("\s\s+", " ", text)
    return "".join(text).strip()


data_df["cleaned_data"]=data_df["text"].apply(lambda x:clean(x) if x!=None else x)

train,test= train_test_split(data_df,stratify=data_df[["title"]],test_size=0.2,random_state=0)
train.reset_index(drop=True,inplace=True)
test.reset_index(drop=True,inplace=True)

train.to_csv("train_data.csv",index=False)
test.to_csv("test_data.csv",index=False)


model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = model.to(device)

train_cleaned_texts = train["cleaned_data"].tolist()
train_cleaned_texts = list(map(str, train_cleaned_texts))
import time
import os

def get_embeddings(model, sentences: List[str], parallel: bool = True):
    start = time.time()
    if parallel:
        # Start the multi-process pool on all cores
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pool = model.start_multi_process_pool(target_devices=["cpu"] * 5)
        embeddings = model.encode_multi_process(sentences, pool, batch_size=16)
        model.stop_multi_process_pool(pool)
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        embeddings = model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device
        )

    print(f"Time taken to encode {len(sentences)} items: {round(time.time() - start, 2)}")
    return embeddings.cpu().detach().numpy()



train_embeddings = get_embeddings(model=model, sentences=train_cleaned_texts, parallel=False)
print(train_embeddings.shape)

#save embeddings of idea texts
cleaned_train_texts_embeddings_file = f"data/train_embeddings_all_minilm_l6_v2.pkl"
pickle.dump(train_embeddings, open(cleaned_train_texts_embeddings_file, "wb"))



### Create mapings

#create mappings for index and category this will be later used for faiss
train_category_index_mapping=dict(zip(train.index,train.title))
with open('data/train_category_index.pickle', 'wb') as handle:
    pickle.dump(train_category_index_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

#load the mappings file which is used for printing category as output
def get_mappings(file_name):
    with open(file_name, 'rb') as handle:
        category_mapping_dict = pickle.load(handle)
    return category_mapping_dict


mappings = get_mappings(file_name='data/train_category_index.pickle')



### Load mappings

#read the embeddings created at earlier step
def read_embeddings(file_name):
    cleaned_texts_embeddings_file = file_name
    if os.path.exists(cleaned_texts_embeddings_file):
        with open(cleaned_texts_embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
    temp1=np.asarray(embeddings,dtype="float32")
    return temp1


    return

samples = read_embeddings(file_name=f"data/train_embeddings_all_minilm_l6_v2.pkl")


### creating and saving indices
#save the index once its created
def save_index(index):
    path = os.path.join("data/","news_train_index")
    faiss.write_index(index, path)

# we have used flat Index and with Inner product
def create_index(mappings,samples):
    index = faiss.IndexIDMap(faiss.IndexFlatIP(samples.shape[1]))
    faiss.normalize_L2(samples)  # normalise the embedding
    #index.train(samples)
    index.add_with_ids(samples,np.array(list(mappings.keys())))
    save_index(index)
create_index(mappings=mappings,samples=samples)
# print top 2 categories among top5 nearest neighbours returned from index search
train=pd.read_csv("train_data.csv") # using the file from previous notebook created step
test=pd.read_csv("test_data.csv")


#read the index
index = faiss.read_index("data/news_train_index")

# embeddings for query
def predict_embeddings(query):
    query_embedding=model.encode(query)
    query_embedding=np.asarray([query_embedding],dtype="float32")
    return query_embedding
    

#predict for given query
def predict(query,mappings):
    cleaned_query= clean(query)
    query_embedding=predict_embeddings(cleaned_query)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, 10) # d is the distance and I is the index number 
    D=[np.unique(D)]
    I=[np.unique(I)]
    res_df=[]
    for values in I:
        for val in D:
            details= {'cleaned_text':list(train.iloc[values]["cleaned_data"]),
            'category':list(train.iloc[values]["title"]),
            'score':list(val)
            }
            res_df.append(details)
    return res_df


def most_frequent(result):
    top2 = Counter(result)
    return top2.most_common(1)


def test_predict(query,mappings=mappings):
    cleaned_query= clean(query)
    query_embedding=predict_embeddings(cleaned_query)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, 5) # d is the distance and I is the index number 
    return most_frequent([mappings[id_] for id_ in I[0]])[0][0]


test["predict"]=test["cleaned_data"].astype(str).apply(lambda x:test_predict(x))

test.to_csv("data/results.csv")

print(classification_report(test['title'],test["predict"]))
cm = confusion_matrix(test['title'],test["predict"])
