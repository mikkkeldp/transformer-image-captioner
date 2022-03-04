from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import torch
import pandas as pd
import math

# bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# loss_fct = torch.nn.CrossEntropyLoss()

# def get_score(sentence):
#     tokenize_input = tokenizer.tokenize(sentence)
#     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#     predictions = bertMaskedLM(tensor_input)
#     loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data
#     return math.exp(loss)

# import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device("cuda")

def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

def get_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    input_ids = torch.tensor(tokenizer.encode(tokenize_input)).unsqeeze(0).to(device)
    with torch.no_grad():
        loss = model(input_ids, labels = input_ids)[0]
        return math.exp(loss.item())

from lm_scorer.models.auto import AutoLMScorer as LMScorer

scorer = LMScorer.from_pretrained("gpt2", device="cuda", batch_size=1)


def get_score(sentence):
    score = scorer.sentence_score(sentence, log=True)
    return score