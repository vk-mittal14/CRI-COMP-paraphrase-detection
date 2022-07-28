import pandas as pd
import os 
import xml.etree.ElementTree as ET

def read_info(file_name):
	xml_data = open(file_name,'r').read()
	root = ET.XML(xml_data)
	data = []
	for child in root:
		data.append([subchild.text] for subchild in child)
	df = pd.DataFrame(data)
	return df

files = os.listdir("indices/")
files = sorted(files)

all_text = []
all_idx = []
for f in files:
    filepath = os.path.join("indices/", f)
    df= read_info(filepath)
    texts = df.iloc[:, 1]
    vals = df.iloc[:, 2].tolist()
    name = f.split(".")[0][7:]
    assert len(texts) == len(vals)
    for i in range(len(texts)):
        t = texts[i][0].replace(";", " ")
        a, b = vals[i][0].split(",")
        a = a[1:]
        b = b[1:-1]
        final_name = f"{name}_{a}_{b}"
        all_text.append(t)
        all_idx.append(final_name)

df = pd.DataFrame(columns = ["id", "text"])
df["id"] = all_idx
df["text" ] = all_text

df.to_csv("processed_test_data.csv", index=False)