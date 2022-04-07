class Config(object):
    def __init__(self):

        # Learning rates
        self.lr_backbone = 1e-5
        self.lr = 0.0001

        # Epochs
        self.epochs = 12
        self.lr_drop = 4
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'learned'
        self.dilation = True
        
        # Basic
        self.device = 'cuda:0'
        self.seed = 42
        self.batch_size = 12
        self.num_workers = 16
        self.checkpoint = './checkpoint_4.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 512
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.25
        self.vocab_size = 30522 

        self.enc_layers = 6
        self.dec_layers = 3
        self.dim_feedforward = 2048
        self.nheads = 2
        self.pre_norm = True
        self.load_pretrained_weights = False

        # Dataset
        self.dir = '../dataset'
        self.limit = -1

        #Improvements 
        self.aug_caps = True
        self.beam_width = 3
        self.lm_scoring = False 
        self.lm_influence = 0.05 #[0.1-3]
        self.ml = True
        self.ml_type ='faster' #faster/pan
