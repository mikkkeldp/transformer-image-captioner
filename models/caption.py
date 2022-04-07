import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List
from models.utils import NestedTensor, nested_tensor_from_tensor_list, is_main_process
from .backbone import build_backbone
from .transformer import build_transformer
from torchvision.models._utils import IntermediateLayerGetter
import torchvision.models as models
from .position_encoding import build_position_encoding
from configuration import Config
import pickle 


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Caption(nn.Module):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.config = Config()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

        self.image_encoder = models.resnet101(pretrained=True)  ##maybe try 152/101?
        self.cnn = nn.Sequential(*list(self.image_encoder.children())[:-5])
        self.config = Config()
        self.backbone2 = getattr(models, 'resnet101')(
            replace_stride_with_dilation=[False, False, True],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        self.body = IntermediateLayerGetter(self.backbone2, return_layers={'layer4': "0"})
        self.position_embedding = build_position_encoding(self.config)
        self.linear = nn.Linear(1024, 2048) # change this to 1280
        
        if self.config.ml:
            self.multi_pos_embeding = nn.Parameter(torch.zeros(self.config.batch_size, 361+10, hidden_dim))
            self.multi_pos_embeding_pan = nn.Parameter(torch.zeros(self.config.batch_size, 361+6, hidden_dim))
            
            self.linear_proj = nn.Linear(2048, hidden_dim)
            self.linear = nn.Linear(1024, hidden_dim) # change this to 1280


        with open('panoptic_feature_dict.pickle', 'rb') as f:
            self.panoptic_dict = pickle.load(f)
       

    def forward(self, samples, target, target_mask, ids, testing, ml, ml_type): #target_mask = image_mask
        if ml == False: 
            # TODO fix training dimensions and stuff
            if testing:
                images = samples.tensors.unsqueeze(0)
                m = samples.mask.unsqueeze(0)
            else:
                batch_size = samples.tensors.shape[0]
                images = samples.tensors
                m = samples.mask

            feat_vecs = self.body(images)['0']
            
           
            
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat_vecs.shape[-2:]).to(torch.bool)[0]
        
            feat_vec_mask_pair = NestedTensor(feat_vecs, mask)
            pos_embed = self.position_embedding(feat_vec_mask_pair).to(feat_vec_mask_pair.tensors.dtype)
                
            # project grid features to lower dim: 2048 -> 256
            proj = self.input_proj(feat_vecs)

            # RESHAPING OF VISUAL FEATS, MASKS AND POS_EMBEDDINGS

            proj = proj.flatten(2).permute(2, 0, 1) #  flatten NxCxHxW to HWxNxC
            # print(proj.shape)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
            mask = mask.flatten(1)
            pos_emb = pos_embed
        else:
            # if training 
            images = samples.tensors
            m = samples.mask

            # if testing
            if testing:
                images = samples.tensors.unsqueeze(0)
                batch_size = 1
                m = samples.mask.unsqueeze(0)
            else:
                batch_size = samples.tensors.shape[0]

            assert m is not None
            
        
            # get grid features
            feat_vecs = self.body(images)['0']
            feat_vecs = self.input_proj(feat_vecs)

            
            mask = F.interpolate(m[None].float(), size=feat_vecs.shape[-2:]).to(torch.bool)[0]

            # standard grid features 
            feat_vecs = feat_vecs.flatten(2)
            mask = mask.flatten(1)

            #panoptic features
            if ml_type == "pan":

                object_feat_vecs = []
                for id in ids:
                    feat = self.panoptic_dict[id][0]
                    object_feat_vecs.append(feat) 

                object_feat_vecs = torch.stack(object_feat_vecs).to('cuda')
                object_feat_vecs = torch.squeeze(object_feat_vecs, 1)
                object_feat_vecs = self.linear_proj(object_feat_vecs)
                object_feat_vecs = object_feat_vecs.permute(0,2,1)

                obj_mask = torch.zeros((batch_size,6),dtype=torch.bool).to('cuda')

                feat_vecs = torch.cat((feat_vecs, object_feat_vecs), dim=2)
                feat_vecs = feat_vecs.permute(0,2,1)

                mask = torch.cat((mask, obj_mask), dim=1)
        
                pos_emb = self.multi_pos_embeding_pan
                
                proj = feat_vecs.permute(1,0,2)
             
            # faster rcnn features
            elif ml_type =="faster":

                # get faster rcnn features
                object_feat_vecs = []
                for id in ids:
                        with open("faster_rcnn_extracted_features/" + id + '_features.pickle', 'rb') as f:
                            obj_pickle = pickle.load(f)
                        object_feat_vecs.append(obj_pickle)  

                object_feat_vecs = torch.stack(object_feat_vecs)
                object_feat_vecs = torch.squeeze(object_feat_vecs, 2)   
                object_feat_vecs = self.linear(object_feat_vecs)
                object_feat_vecs = object_feat_vecs.permute(0,2,1)

                obj_mask = torch.zeros((batch_size,10),dtype=torch.bool).to('cuda')
            
            
                feat_vecs = torch.cat((feat_vecs, object_feat_vecs), dim=2)
                feat_vecs = feat_vecs.permute(0,2,1)
        
                mask = torch.cat((mask, obj_mask), dim=1)
        
                pos_emb = self.multi_pos_embeding
                
            

                proj = feat_vecs.permute(1,0,2)

                
            else:
                print("ml_type not supported")

        
            if testing:
                pos_emb = pos_emb[0]
                pos_emb = pos_emb.unsqueeze(0)

            pos_emb = pos_emb.permute(1,0,2)
        
        hs = self.transformer(proj, mask, pos_emb, target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out
    


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)

    model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion


def load_transformer_weights(model, weights, config):

    pretrained_dict = weights
    model_dict = model.state_dict()

    backbone = build_backbone(config)
    backbone_dict = backbone.state_dict()
    
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in backbone_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return model

