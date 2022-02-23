import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary, Flickr8k
from PIL import ImageChops
from utils import nested_tensor_from_tensor_list, read_json
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class Flickr8kTrainDataset(data.Dataset):

    def __init__(self, image_dir, caption_path, split_path, vocab, transform=None, cpi=5, max_length=33):
        self.image_dir = image_dir
        self.f8k = Flickr8k(caption_path=caption_path)

        with open(split_path, 'r') as f:
            self.train_imgs = f.read().splitlines()

        self.vocab = vocab
        self.transform = transform
        #change this to 10 for augmented captions
        self.cpi = cpi # captions per image

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1 #max_position_embeddings


    def __getitem__(self, index):
   
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        fname = self.train_imgs[index//self.cpi]
        caption = self.f8k.captions[fname][index%self.cpi]
        file_path = self.image_dir + "/" + fname
        id = fname.split(".")[0]


        image = Image.open(file_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))


        caption_encoded = self.tokenizer.encode_plus(caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        
        caption = np.array(caption_encoded['input_ids'])
   
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)
       
        

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask

    def __len__(self):
        return len(self.train_imgs)*self.cpi




def get_train_loader(image_dir, caption_path, train_path, vocab, transform, batch_size, shuffle, num_workers, cpi, max_length):
    """Returns torch.utils.data.DataLoader for custom flickr8k dataset."""
  
    f8k = Flickr8kTrainDataset(image_dir=image_dir,
                       caption_path=caption_path,
                       split_path=train_path,
                       vocab=vocab,
                       transform=transform,
                       cpi=cpi,
                       max_length=max_length)


    sampler_train = torch.utils.data.RandomSampler(f8k)
    

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        f8k, batch_sampler=batch_sampler_train, num_workers=2)
   

    return data_loader_train





class Flickr8kValidationDataset(data.Dataset):

    def __init__(self, image_dir, caption_path, split_path, vocab, transform=None):
        self.image_dir = image_dir
        self.f8k = Flickr8k(caption_path=caption_path)

        with open(split_path, 'r') as f:
            self.val_imgs = f.read().splitlines()

        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        
        vocab = self.vocab
        fname = self.val_imgs[index]
        id = fname.split(".")[0]
        caption = self.f8k.captions[fname][index%self.cpi]


        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
      
        caption_encoded = self.tokenizer.encode_plus(caption, max_length=self.j, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

       
        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask
            
        

    def __len__(self):
        return len(self.val_imgs)



def get_validation_loader(image_dir, caption_path, val_path, vocab, transform, batch_size, num_workers, max_length):
    f8k = Flickr8kValidationDataset(image_dir=image_dir,
                   caption_path=caption_path,
                   split_path=val_path,
                   vocab=vocab,
                   transform=transform)
    
    sampler_val = torch.utils.data.SequentialSampler(f8k)
    


    data_loader_val = DataLoader(sampler_val, batch_size,
                              sampler=sampler_val, drop_last=False, num_workers=2)

    return data_loader_val