import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import pickle as pkl

data = pd.read_csv("processed_test_data.csv")
text_data = data["text"].tolist()

torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
device = torch.device("cuda:0")

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-albert-small-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-albert-small-v2').to(device)

embeds  = []


for i in tqdm(range(980)):
    use = text_data[100*i: 100*(i+1)]
    encoded_input = tokenizer(use, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu()
    embeds.append(sentence_embeddings)


with open(f"final_testdata_paraphrase-albert-small-v2.pkl", "wb") as f:
    pkl.dump(embeds, f)