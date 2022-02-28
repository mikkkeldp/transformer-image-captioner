# Transformer image captioning 

This project implements a Transformer-based image captioning model. We aim at training an image captioning network in a low-resource regime. We make use of the Flickr8k dataset consisting of 30,000 image-caption pairs. This is still a work in progress and is inspired by the following papers:

<table>
  <tr>
    <td valign="top">[1]</td>
    <td>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. <a href="https://arxiv.org/abs/1612.00563">Attention is all you need</a>. In <i>Advances in neural information processing systems</i>, pages 5998–6008, 2017.</td>
  </tr>
  <tr>
    <td valign="top">[2]</td>
    <td>Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang. <a href="https://arxiv.org/abs/1707.07998">Bottom-up and top-down attention for image captioning and visual question answering</a>. In <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</i>, pages 6077–6086, 2018.</td>
  </tr>
  <tr>
    <tr>
    <td valign="top">[3]</td>
    <td> Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, Rita Cucchiara. <a href="https://arxiv.org/abs/1912.08226">M2: Meshed-memory transformer for image captioning</a>. <i>arXiv preprint</i> arXiv:1912.08226, 2019.</td>
  </tr>
    <td valign="top">[4]</td>
    <td>Steven J. Rennie, Etienne Marcheret, Youssef Mroueh, Jarret Ross, Vaibhava Goel. <a href="https://arxiv.org/abs/1612.00563">Self-critical Sequence Training for Image Captioning</a>. In <i>Computer Vision and Pattern Recognition</i>, pages 1179–1195, 2017.</td>
  </tr>

</table>


## Table of Contents
1. [An Overview of work](#1) 
2. [Model comparison](#2)
3. [Results](#3)
4. [Project TODO](#4)
5. [Usage](#5)
6. [Optimizing Transformers for small datasets](#6)
7. [Self-Critical Sequence Training (SCST)](#7)
8. [Good Resources](#8)

## 1. An Overview of work<a name="1"></a>
We implement an image captioning model that uses a Transformer for both the encoder and decoder. The Transformer encoder will be used for self-attention on visual features, while the Transformer decoder will be used for masked self-attention on caption tokens and for vision-language attention.  

On top of this, we will be incorporating the improvements introduced to [Xu et al.](https://arxiv.org/abs/1502.03044)'s Soft-Attention model, described in our [previous work](www.google.com). Below is a short summary of the improvements:

1. Representing the image at various levels of granularity to the decoder in terms of low-level and high-level regions, providing additional context to the decoder
2. Introducing a pre-trained language model that suppresses semantically unlikely captions during beam-search
3. Improve the vocabulary and expressiveness of the model by augmenting training captions with the aid of a paraphrase generator

## 2. Model comparison<a name="2"></a>
We will be using our previous work's implementation as our base model, that is comprised of the Soft-Attention model along with all 3 improvements listed above. To measure the effectiveness of these improvements on a transformer-based model, we will be implementing the following model variations:

<table>
  </tr>
    <td><b>Model</b></td>
    <td><b>Description</b></td>
  </tr>
  </tr>
    <td>Base Transformer</td>
    <td>The encoder is fed image region embeddings consisting of high-level attention regions achieved through the feature maps of a pre-trained CNN in ResNet.</td>
  </tr>
   </tr>
    <td> Multi-level regions Transformer</td>
    <td>In addition to the high-level attention regions provided in the base transformer, we provide more fine-grained attention regions produced by either PanopticFCN or Faster R-CNN.</td>
  </tr>
  </tr>
   </tr>
    <td> LM rescoring Transformer</td>
    <td>During beam-search, we will use GPT-2 to rescore the caption candidates.</td>
  </tr>
  </tr>
    <td>Caption augmentation Transformer</td>
    <td>We make use of the T5 text-to-text model to augment training captions.</td>
  </tr>
</table>

## 3. Results<a name="3"></a>
Here the *Base model* is the implementation of our previous work - incorporating all above mentioned improvements to the Soft-Attention model.
  <table>
    </tr>
      <td> Model</td>
      <td>B-1</td>
      <td>B-2</td>
      <td>B-3</td>
      <td>B-4</td>
      <td>MTR</td>
      <td>CDR</td>
    </tr>
    </tr >
        <td>Soft-Attention</td>
        <td>67</td>
        <td>44.8</td>
        <td>29.9</td>
        <td>19.5</td>
        <td>18.9</td>
        <td>-</td>
    </tr>
    </tr>
        <td>Base model</td>
        <td>68.6</td>
        <td>48.5</td>
        <td>34.7</td>
        <td>24.5</td>
        <td>23.2</td>
        <td>49.2</td>
    </tr>
        </tr>
        <td>Base Transformer</td>
        <td>68.49</td>
        <td>51.15</td>
        <td>35.82</td>
        <td>25.25</td>
        <td>27.43</td>
        <td>47.79</td>
    </tr>
    </tr>
        <td>MLR Transformer</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tr>
        <td>LM rescoring Transformer</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tr>
        <td>CA Transformer</td>
        <td>68.78</td>
        <td>52.12</td>
        <td>36.45</td>
        <td>25.68</td>
        <td>27.17</td>
        <td>49.83</td>
    </tr>
  </table>

Note that models marked with * have not yet been hyperparameter tuned and are expected to improve. The CA transformer has **6** encoder layers and **3** decoder layers (the opposite of what IC models trained on larger models use) and shows promising results.

## 4. Project TODO <a name="4"></a> 
- [x] Fix tokenizer (30/1/2022)
- [x] Literature review of image captioning papers implementing transformers (1/2/2022)
- [x] Base Transformer (2/2/2022)
- [x] Optimize Transformer for smaller datasets (8/2/2022)
- [x] Use custom vocab instead of Bert (recommended for limited datasets, able to limit vocab) (9/2/2022)
- [ ] MLR Transformer implementation
- [x] Beam search implementation (20/2/2022)
- [ ] LM rescoring Transformer implementation
- [x] CA Transformer implementation
- [ ] Self-Critical Sequence Training (SCST) 

## 5. Usage<a name="5"></a> 
### 5.1 Setup project 
Clone repository:
```
$ git clone https://github.com/mikkkeldp/transformer-image-captioner
```
Install dependencies:
```
$ pip install -r requirements.txt
```

### 5.2 Data preparation

Download and extract Flickr8k Dataset from [here](https://www.kaggle.com/adityajn105/flickr8k/activity) and extract to *dataset* folder

### 5.3 Build vocabulary
We use the Bert Tokenizer to build a vocabulary over the captions within the training captions.

```
$ python3 build_vocab.py
```

### 5.4 Training
Tweak the hyperparameters in ```configuration.py```.

To train on a single GPU, run:
```
$ python3 main.py
```
To train from a checkpoint, run
```
$ python3 main.py --checkpoint /path/to/checkpoint/
```

We train our model with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in for ResNet. 

### 5.5 Testing 
To test the model using a checkpoint, run:
```
$ python3 test.py --checkpoint /path/to/checkpoint/
```

<!-- 
### Predict on sample
To test our model on sample images, run:
```
$ python predict.py --path /path/to/image --v v2  // You can choose between v1, v2, v3 [default is v3]
```
 -->



## 6. Optimizing Transformers for small datasets <a name="6"></a>
It is known that transformers struggle to learn under a low-resource regime. However, there are some works that managed to achieve success under these circumstances. These works focus on the task of machine translation (Seq2Seq), but hopefully carry over to the task of image captioning. Here are the findings of some of these papers:

### 6.1 [Optimizing Transformer for Low-Resource Neural Machine Translation](https://aclanthology.org/2020.coling-main.304.pdf)
Limit the amount of trainable parameters. They found that Transformers under low-resource conditions is highly dependent on the hyper-parameter settings. Deep transformers requires large amounts of training data. Reducing the depth and width, including the number of attention heads, feed-forward dimension, and number of layers along with increasing the rate of different regularization techniques is highly effective (+6 BLEU). The largest improvements are obtained by increasing the dropout rate (+1.4 BLEU), adding layer dropout to the decoder (+1.6 BLEU), and adding word dropout to the target side (+0.8 BLEU). 

| Optimal hyper-parameters | default | 5k      | 10k     | 20k     | 40k     | 80k     |
|--------------------------|---------|---------|---------|---------|---------|---------|
| BPE operations           | 37k     |    5k   | 10k     | 10k     | 12k     | 15k     |
| feed-forward dim         | 2048    | 512     | 1024    | 1024    | 2048    | 2048    |
| att heads                | 8       | 2       | 2       | 2       | 2       | 2       |
| dropout                  | 0.1     | 0.3     | 0.3     | 0.3     | 0.3     | 0.3     |
| layers                   | 6       | 5       | 5       | 5       | 5       | 5       |
| label smoothing          | 0.1     | 0.6     | 0.5     | 0.5     | 0.5     | 0.4     |
| enc/dec layerdrop        | 0.0/0.0 | 0.0/0.3 | 0.0/0.2 | 0.0/0.2 | 0.0/0.1 | 0.0/0.1 |
| src/tgt word dropout     | 0.0/0.0 | 0.0/0.1 | 0.0/0.1 | 0.1/0.1 | 0.1/0.1 | 0.2/0.2 |
| act dropout              | 0.0     |   0.3   | 0.3     | 0.3     | 0.3     | 0       |
| batch size               | 4096    | 4096    | 4096    | 4096    | 4096    | 8192    |


### 6.2 [Low-Resource Neural Machine Translation for Southern African Languages](https://arxiv.org/pdf/2104.00366.pdf)
They found that using large models is detrimental for low-resource language translation, since it makes training more difficult and hurts overall performance. It was found that depth of 6 layers was optimal, opposed to a deep Transformer consisting of 12 layers. They also found that finfing stable learning rates can be very computationally expensive.

### 6.3 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)

This work introduced a novel neural architecture Transformer-XL that was able to perform sequence-to-sequence translation with great success. A key insight was discovered after the code had been released, was that in a small dataset regime, data augmentation is crucial. This in turn regularizes the model. The most dramatic performance gain comes from discrete embedding dropout. That is, you embed as usual, but with a probability *p* you zero the entire word vector. This is akin to masked language modelling but with the goal not to predict the mask - just regular LM with uncertain context. Another important factor is regular input dropout. This is, dropping elements of the embeddings with probability *p*. This is the same as dropping out random pixels from images. The drawback to all these regularization techniques is much longer training times. 

### 6.4 Most common hyper-parameters on image captioning models (mostly on MSCOCO)

Below are the most common transformer hyper-parameters used for image captioning (as found on public github repos). These are mostly tuned for MSCOCO, but give an indicator of adjustments made for the task of image captioning, opposed default parameters for seq2seq tasks.

| Parameter                | default | recommended |
|--------------------------|---------|-------------|
| lr | 0.00005| 0.0003 |
| lr-scheduler | StepLR (10 epoch steps) | inverse_sqrt (8000 iteration steps)|
| criterion| CE | label smoother CE |
| encoder layers | 6 | 3 | <----- Maybe due to using pre-trained CNN's for input to the encoder.
| decoder layers | 6 | 6 | 
| dropout | 0.1 | 0.3 |
| encoder_embed_dim | 128 | 512 | 



## 7. Hyper-parameter testing <a name="6"></a>

| Model                                                 | description                                                                                                                | B1 | B2 | B3 | B4 | MTR | CDR | # epochs |
|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|----|----|----|----|----|-----|-----|
| Default                                               | Base model with default transformer hyper-params. Standard image transforms of resize, normalization.                      |  64.96  |  46.29  | 32.60   |  22.78  |  23.44  |   45.98  |  7 |   
| Image transforms                                      | Base model with default transformer hyper-params. Image transforms: color jitter, random horizontal flip, random rotation. |  63.63  | 45.34   | 31.80   | 22.27   | 23.21   |  47.86   |  12   |  
| Partial optimal suggested hyper-params with image transforms. | enc/dec layers = 5, feedforw dim = 1024, heads =2. dropout=0.3(other regularization techniques to be added)                                                             |  64.57  |  46.30  |  32.98  |   23.29 | 23.09  |  46.45 | 14 |
| Smaller max_position_embedding | max_position_embeddings = 64, image transformers, batch_size = 10, default settings otherwise  |  64.55  |  45.92  |  32.31  |      22.75  | 22.84  | 43.03 | 8 |
| Common IC transformer params, but adjusted for smaller datasets | enc_layers = 3, dropout = 0.2, default image transforms, lr=0.0003,  lr_step at epoch 8       |  65.29  |  46.54  |  32.99  |  23.39  |  23.26  |  45.62 | 8 |

From observing the results achieved through implementing regularization techniques only increases the training time (more epochs) without noticeably increasing the accuracies.

## 8. Self-Critical Sequence Training  <a name="8"></a>
Deep learning models for sequence generation are traditionally trained using supervised learning methods, in which a cross entropy loss is calculated for each output token and average across the entire generated sequence. Such models are often sequence-to-sequence recurrent models, where the model maintains an internal state *h*<sub>t</sub> during the generation of a sequence and outputs a single token *w*<sub>t</sub> corresponding to an input token at each time step *t*.

During training time, a method called "Teacher-Forcing" is often used, where the model is trained with cross entropy to maximize the probability of outputting a token *w*<sub>t</sub> conditioned on the pervious ground truth token *w*<sub>t-1</sub> (in addition to its internal state *h*<sub>t</sub>). Through using cross entropy loss during training, the network is fully differentiable, and thus backpropagation can be used. This, however, creates a schism between training and testting time, as the model's test-time interference algorithm does not have access to the previous ground truth token *w*<sub>t-1</sub> and therefore typically feeds in the previous predicted token *ŵ*<sub>t-1</sub>. This may lead to cascasing errors during inference and is known as exposure bias. 

### 8.1 Policy Gradient Methods
The use of policy gradient methods from reinforcement learning is a relatively new development in the training of sequence generation models. This class of algorithms allow for non-differentiable metrics to be directly optimized and perform the exposure bias to be reduced. In the case of image captioning, you can directly optimize the model to maximize a evaluation model. The most commonly optimized metric is CIDEr, as it's known to lift the performance of all other metrics considerably. 

### 8.2 Self-Critical Sequence Training (SCST)
Recently, reinforcement learning methods such as SCST have emerged to mitigate the weaknesses with policy gradient methods. SCST uses the reward obtained from the model's own test-time inference algorithm as the baseline and combines it with the technique known as REINFORCE. **The cost of SCST is only one additional forward pass, therefor only requiring 1.33x cost versus traditional backpropagation methods.**


<figure>
<img src="https://i.postimg.cc/76nn2TYg/SCST.png" alt="Trulli" style="width:80%">
<figcaption align = "center">Fig.1 - "Self-critical sequence training (SCST). The weight put on words of a sampled sentence from the model is determined by the difference between the reward for the sampled sentence and the reward obtained by the estimated sentence
under the test-time inference procedure (greedy inference depicted). This harmonizes learning with the inference procedure,
and lowers the variance of the gradients, improving the training procedure."</figcaption>
</figure>  





## 9. Good resources <a name="9"></a>
- [Article on Transformers](https://glassboxmedicine.com/2019/08/15/the-transformer-attention-is-all-you-need/)
- [Article on Positional Embeddings](https://medium.com/nlp-trend-and-review-en/positional-embeddings-7b168da36605#:~:text=Sinusoidal%20positional%20embeddings%20generates%20a,to%20learn%20the%20relative%20positions.)
  

