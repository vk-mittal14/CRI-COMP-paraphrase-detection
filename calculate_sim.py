import pickle as pkl
import torch
from tqdm import tqdm
import pandas as pd 

torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(f"final_testdata_paraphrase-albert-small-v2.pkl", "rb") as f:
    data =  pkl.load(f)

df = pd.read_csv("processed_test_data.csv")

data = torch.nn.functional.normalize(torch.cat(data), p=2.0, dim=1)
device = torch.device("cuda:0")
data = data.to(device)

values = []
indexs = []
for i in tqdm(range(40)):
    res = torch.matmul(data[i*2500:(i+1)*2500], data.T).cpu()
    value, index = torch.topk(res, k=2, dim=1)
    values.append(value)
    indexs.append(index)

values =  torch.cat(values)
indexs = torch.cat(indexs)
indexs = indexs[:, 1]
final_index = (values[:, 1] > .80).nonzero(as_tuple=True)[0]
y = indexs[final_index].tolist()
x = final_index.tolist()

submission = pd.DataFrame(columns= ["id1", "id2"])

id1 = df["id"][x].tolist()
id2 = df["id"][y].tolist()

new_id1 = []
new_id2 = []
for i, j in zip(id1, id2):
    if i != j:
        new_id1.append(i)
        new_id2.append(j)

submission["id1"] = new_id1
submission["id2"] = new_id2

submission.to_csv("final_test_data_submission_80.csv", index=False)
