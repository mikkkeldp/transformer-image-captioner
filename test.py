import torch

from transformers import BertTokenizer
from PIL import Image
import argparse
from tqdm import tqdm
from zmq import device
from models import caption

from configuration import Config
import os
from data_loader import get_train_loader
from torchvision import transforms
from build_vocab import Vocabulary
import pickle 
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor

transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(244),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

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
parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)

args = parser.parse_args()
checkpoint_path = args.checkpoint

config = Config()

with open("./vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

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
    model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(244),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])



def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template



@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption


dataset_path = "./dataset/Flickr8k_Dataset"
dataset_test_paths = "./dataset/Flickr_8k.testImages.txt"
caption_path = "./dataset/Flickr8k.token.txt"
data_loader_val = get_train_loader("./dataset/Flickr8k_Dataset", "./dataset/Flickr8k.token.txt", "./dataset/Flickr_8k.testImages.txt", vocab,
                             transform, 1,
                             shuffle=False, num_workers=int(2), cpi=5)



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
    img_path = dataset_path + "/" + img
    image = Image.open(img_path)
    image = transform(image)
    image = image.unsqueeze(0)

    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)

    output = evaluate()
    output = output[0].tolist()
    index = output.index(0)
    output = output[:index]
    target = targets_dict[img]
    predicted.append(output)
    targets.append(target)
    # result = tokenizer.decode(output, skip_special_tokens=True)
    
    # retrieve targets 

    
    # result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    # result = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(result.capitalize())

scores = get_eval_score(targets, predicted)