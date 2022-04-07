from configuration import Config
from test import test
from main import train
import os
from build_vocab import Vocabulary
import glob
import numpy as np
config = Config()

#delete current checkpoints
checkpoint_paths =  glob.glob('*.pth')


# for filename in checkpoint_paths:
#     if os.path.exists(filename):
#         os.remove(filename)
#     else:
#         print("The file does not exist")

# # Set parameters
# lrs = [0.0005]
# hidden_dims = [512]

# cnt = 0
# for lr in lrs:
#     for hidden_dim in hidden_dims:
#         config.lr = lr
#         config.hidden_dim = hidden_dim


#         # train with parameters
#         # train(config)
        
#         # evaluate checkpoints, saves checkpoint 3-5
#         for i in [2,3,4,5]:
#             config.checkpoint = "./checkpoint_" + str(i) + ".pth"
#             scores = test(config)

#             print("**************")
#             print("CP: ", config.checkpoint)
#             print("lr: ", config.lr)
#             print("hd: ", config.hidden_dim)
#             print("SCORES: \n")
#             print(scores)



for i in np.arange(3,7):
    config.checkpoint = "./checkpoint_" + str(i) + ".pth"
    scores = test(config)

    # print("**************")
    # print("CP: ", config.checkpoint)
    # print("lr: ", config.lr)
    # print("hd: ", config.hidden_dim)
    # print("SCORES: \n")
    # print(scores)









                    





