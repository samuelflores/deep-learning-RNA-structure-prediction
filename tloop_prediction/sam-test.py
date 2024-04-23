import torch
print("check .5")
from transformers import AutoTokenizer, AutoModel
print("check 1")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
print("check 2")

model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
print("check 3")
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
print("check 4")
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
print("check 5")
hidden_states = model(inputs)[0] # [1, sequence_length, 768]
print("check 6")

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print("check 7")
print(embedding_mean.shape) # expect to be 768

# embedding with max pooling
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print("check 8")
print(embedding_max.shape) # expect to be 768
