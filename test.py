import torch
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import argparse
from data_loader import get_train_loader
from models import caption
import pickle
from configuration import Config
import os
from build_vocab import Vocabulary
from tqdm import tqdm
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
import nltk
import numpy as np
import math
from language_model_rescoring import get_score
from models import utils


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')


parser = argparse.ArgumentParser(description='Image Captioning')

parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()


checkpoint_path = args.checkpoint

config = Config()

with open("./vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(244),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

print("Checking for checkpoint.")
if checkpoint_path is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
    print("Found checkpoint! Loading!")
    model,_ = caption.build_model(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict

def greedy_search(image, caption, cap_mask, id):
    

    for i in range(config.max_position_embeddings - 1): #iterate rows of vocab_size x max_pos_embedding matrix
        predictions = model(image, caption, cap_mask, id)
        
        predictions = predictions[:, i, :]
        
        predicted_id = torch.argmax(predictions, axis=-1)
        
        if predicted_id[0] == 102: # END OF SEQUENCE TOKEN
            return caption
       
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption


"""
Beam search debugging util:

prints all current beams decoded and associated scores
"""
def show_beam(sequences):
    for i in range(len(sequences)):
        caption = sequences[i][0].cpu().detach().numpy()[0]
        index = np.where(caption == 0)[0][0]

        caption = caption[:index]
        length = len(caption) - 1 #discard start token
        caption = tokenizer.decode(caption,skip_special_tokens=True)
         # caption = tokenizer.convert_ids_to_tokens(caption)
        score = sequences[i][1].cpu().detach().numpy()
        complete = sequences[i][3]
        print("c ", i , ":", caption, ". len: ", length, "\t score: ", score, " ", complete)
    print("\n")

'''
Rescore beam search candidates with the aid of a language model

'''
def rescore_sequences(sequences, influence):
    global avg_lm_score
    global avg_loss_score
    for i in range(len(sequences)):
        caption = sequences[i][0].cpu().detach().numpy()[0]
        try:
            index = np.where(caption == 0)[0][0]
            caption = caption[:index]
        except IndexError:
            pass

        caption = tokenizer.decode(caption,skip_special_tokens=True)
        lm_score = get_score(caption)
        avg_lm_score += lm_score
        avg_loss_score += sequences[i][1].cpu().detach().numpy()
        # print(lm_score)
        sequences[i][1] += influence*lm_score
    sequences = sorted(sequences, key=lambda tup:tup[1], reverse=True) 
    return sequences


"""
Beam search implementation, k number of beams
"""
def beam_search(image, caption, cap_mask, k):

    sequences = []
    predictions = model(image, caption, cap_mask)
    predictions = predictions[:, 0, :]
    scores, predicted_ids = torch.topk(predictions, k=k,  dim=-1)
    scores = scores[0]
    predicted_ids = predicted_ids[0]
  
    for j in range(k):
        candidate_seq = torch.clone(caption)
        candidate_seq[:, 1] = predicted_ids[j]
        candidate_cap_mask = cap_mask
        candidate_cap_mask[:, 1] = False
        completed = False
        candidate = [candidate_seq, scores[j], candidate_cap_mask, completed]
        sequences.append(candidate)
  
    decay = 0

    for i in range(1, config.max_position_embeddings -1):
        if i > 7:
            decay = 3
        elif i > 10:
            decay = 5 
        elif i > 12:
            decay = 8

        new_seqs = []
        for j in range(k): #for every beam in sequence list
            new_caption = sequences[j][0].clone()
            new_score = sequences[j][1].clone()
            new_cap_mask = sequences[j][2].clone()
            new_complete = sequences[j][3]


            if new_complete:
                new_candidate = [new_caption.clone(), new_score.clone(), new_cap_mask.clone(), True]
                new_seqs.append(new_candidate)
            else:
                new_predictions = model(image, new_caption, new_cap_mask)
                new_predictions = new_predictions[:, i, :] 
                new_scores, new_predicted_ids = torch.topk(new_predictions, k=k+1,  dim=-1)

                new_scores = new_scores[0]
                new_predicted_ids = new_predicted_ids[0]

                # remove 102 tokens, thats why we get k+1 top tokens, and remove 102 if present
                nx = new_predicted_ids.cpu().numpy()
                index = np.where(nx == 102)[0]
                if index:
                    new_predicted_ids = torch.cat([new_predicted_ids[0:index[0]], new_predicted_ids[index[0]+1:]])

                for l in range(k): #expand each beam 
                    new_caption[:, i+1] = new_predicted_ids[l]
                    new_cap_mask[:, i+1] = False

                    new_score += new_scores[l]
                    new_score -= decay


                    if new_predicted_ids[l] == 1012 or new_predicted_ids[l] == 102:
                        new_candidate = [new_caption.clone(), new_score.clone(), new_cap_mask.clone(), True]
                        new_seqs.append(new_candidate)
                    
                    else:
                        new_candidate = [new_caption.clone(), new_score.clone(), new_cap_mask.clone(), False]
                        new_seqs.append(new_candidate)
                
            
        # sort beams and keep top k beams
        new_seqs = sorted(new_seqs, key=lambda tup:tup[1], reverse=True)
        new_seqs = new_seqs[:k]
        sequences = new_seqs
     
        # show_beam(sequences)
        all_completed = True
        for seq in sequences:
            if seq[3] == False: all_completed = False
        
        if all_completed:
            # show_beam(sequences)
            if config.lm_scoring:
                sequences = rescore_sequences(sequences, config.lm_influence)
            # show_beam(sequences)
            return sequences[0][0]


    print("MAX")
    print(sequences)
    if config.lm_scoring:
        sequences = rescore_sequences(sequences, config.lm_influence)
    
    return sequences[0][0]


        


@torch.no_grad()
def evaluate(image, caption, cap_mask, id):
    model.eval()
    if config.beam_width == 1:
        return greedy_search(image, caption, cap_mask, id)
    else:
        return beam_search(image, caption, cap_mask, config.beam_width, id)
    # return n_beam_search(image, caption, cap_mask, config.beam_width)


#test stuff
avg_lm_score = 0
avg_loss_score = 0


dataset_path = "./dataset/Flickr8k_Dataset"
dataset_test_paths = "./dataset/Flickr_8k.testImages.txt"
caption_path = "./dataset/Flickr8k.token.txt"

f = open(dataset_test_paths, "r")
content = f.read()

testing_imgs = content.splitlines()
f.close()

f = open(caption_path, "r")
content = f.read()

target_captions = content.splitlines()
f.close()

targets_dict = {}
for caption in target_captions:
    
    splits = caption.split("\t")
    id = splits[0][:-2]
    if id in testing_imgs:
        tokens = tokenizer.encode(splits[1])
        if id not in targets_dict.keys():
            targets_dict[id] = [tokens]
        else:
            targets_dict[id].append(tokens)





targets = []
predicted = []
for img in tqdm(testing_imgs):
    id = [img.split(".")[0]]
   
    img_path = dataset_path + "/" + img
    image = Image.open(img_path)        
    image = transform(image)

    image = utils.nested_tensor_from_tensor_list(image)

    image, image_mask = image.tensors.squeeze(0), image.mask.squeeze(0)
    print(image.shape)
    print(image_mask.shape)
    sample = utils.NestedTensor(image, image_mask).to(device)

    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)
    caption = caption.to(device)
    cap_mask = cap_mask.to(device)
    image = image.to(device)

    output = evaluate(sample, caption, cap_mask, id)
    output = output[0].tolist()
   
    try:
        index = output.index(0)
    except ValueError as e:
        index = -1
        print("LONG")
    
    output = output[:index]
    output.append(102)
    target = targets_dict[img]

    predicted.append(output)
    targets.append(target)

print(avg_loss_score/1000*config.batch_size)
print(avg_lm_score/1000*config.batch_size)
scores = get_eval_score(targets, predicted)