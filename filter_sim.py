import pickle as pkl
import nltk
from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('punkt')

cachedStopWords = stopwords.words("english")
cachedStopWords += [".", ",", "]", "[", "(", ")"]

df = pd.read_csv("final_test_data_submission_80.csv")
data = pd.read_csv("processed_test_data.csv")


def getSim(s1, s2):
    
    count1 = 0
    count2 = 0
    
    s1 = nltk.word_tokenize(s1)
    s2 = nltk.word_tokenize(s2)

    s1  = [word.lower() for word in s1 if word not in cachedStopWords]
    s2  = [word.lower() for word in s2 if word not in cachedStopWords]
    
    for s in s1:
        if s in s2:
            count1 += 1
    
    for s in s2:
        if s in s1:
            count2 += 1
    
    if len(s1) == 0:
        return 0 
    
    return (count1 + count2)/(len(s1) + len(s2))

sim_list = []
ids = []
for id1, id2 in tqdm(zip(df["id1"], df["id2"])):
    ids.append((id1, id2))
    s1 = data[data["id"] == id1]["text"].tolist()[0]
    s2 = data[data["id"] == id2]["text"].tolist()[0]   
    
    sim = getSim(s1, s2)
    sim_list.append(sim)

df["word_sim"] = sim_list
threshold = 0.59
new_df = df[df["word_sim"] > threshold].iloc[:, :2]

new_df.to_csv(f"testdata_final_submission_80_wordprocess59.csv", index = False)