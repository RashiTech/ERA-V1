import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split, DataLoader
from transformers import AutoProcessor, AutoTokenizer
import pickle
import requests
import pandas as pd
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class llavadataset(Dataset):
  def __init__(self, qa_dataset, phi_model_name, clip_model_name, tokenizer,processor):

    self.tokenizer  = tokenizer
    self.processor  = processor
    self.qa_dataset = qa_dataset

  def __len__(self):
    return self.qa_dataset.shape[0]

  def __getitem__(self, idx):

    img_url = self.qa_dataset.img_url[idx]
    ques    = torch.tensor(np.array(np.matrix(self.qa_dataset.input[idx]))[0])  
    ans     = torch.tensor(np.array(np.matrix(self.qa_dataset.label[idx]))[0])

    # image load
    image_load = Image.open(requests.get(img_url,stream=True).raw)
    image_processed = self.processor(images=image_load, return_tensors="pt") ['pixel_values']
    image_processed = image_processed.squeeze(0)
    #q = self.tokenizer(ques, return_tensors="pt", return_attention_mask=False)['input_ids'].squeeze(0)
    #a = self.tokenizer(ans, return_tensors="pt", return_attention_mask=False)['input_ids'].squeeze(0)
    return(image_processed , ques, ans)
  

def collate_fn(batch):
    image_embeddings, ques,ans = zip(*batch)
    image_embeddings_stacked = torch.stack(image_embeddings, dim=0)
    ques_padded = torch.nn.utils.rnn.pad_sequence(ques, batch_first=True, padding_value=50256)
    ans_padded = torch.nn.utils.rnn.pad_sequence(ans, batch_first=True, padding_value=50256)
    return (image_embeddings_stacked, ques_padded,ans_padded)