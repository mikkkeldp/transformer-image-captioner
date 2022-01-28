# Transformer image captioning 

This project implements a Transformer-based image captioning model. This word is still a work in progress and is inspired by the following papers:

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

We implement an image captioning model that uses a Transformer for both the encoder and decoder. The Transformer encoder will be used for self-attention on visual features, while the Transformer decoder will be used for masked self-attention on caption tokens and for vision-language attention.  

On top of this, we will be incorporating the improvements introduced to [Xu et al.](https://arxiv.org/abs/1502.03044)'s Soft-Attention model, described in our [previous work](www.google.com). Below is a short summary of the improvements:

1. Representing the image at various levels of granularity to the decoder in terms of low-level and high-level regions, providing additional context to the decoder
2. Introducing a pre-trained language model that suppresses semantically unlikely captions during beam-search
3. Improve the vocabulary and expressiveness of the model by augmenting training captions with the aid of a paraphrase generator

## Model comparison
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

## Results
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
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tr>
        <td>MLR Transformer*</td>
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
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
  </table>

<!-- Note that models marked with * have not yet been hyperparameter tuned and are expected to improve. -->

### Project TODO:
- [x] Fix tokenizer
- [ ] Literature review of image captioning papers implementing transformers
- [ ] Base Transformer
- [ ] MLR Transformer implementation
- [ ] Beam search implementation
- [ ] LM rescoring Transformer implementation
- [ ] CA Transformer implementation
- [ ] Self-Critical Sequence Training (SCST) 

## Usage 

Clone repository:
```
$ git clone https://github.com/mikkkeldp/transformer_ic
```
Install dependencies:
```
$ pip install -r requirements.txt
```

### Data preparation

Download and extract Flickr8k Dataset from [here](https://www.kaggle.com/adityajn105/flickr8k/activity) and extract to *dataset* folder
```


### Training
Tweak the hyperparameters in ```configuration.py```.

To train on a single GPU, run:
```
$ python main.py
```
To train from a checkpoint, run
```
$ python main.py --checkpoint /path/to/checkpoint/
```

We train our model with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in for ResNet. 

### Testing 
To test the moddel using a checkpoint, run:
```
$ python test.py --checkpoint /path/to/checkpoint/
```

<!-- 
### Predict on sample
To test our model on sample images, run:
```
$ python predict.py --path /path/to/image --v v2  // You can choose between v1, v2, v3 [default is v3]
```
 -->

## APPENDIX

### Self-Critical Sequence Training (SCST) 
**TODO**


## Good resources
- [Article on Transformers](https://glassboxmedicine.com/2019/08/15/the-transformer-attention-is-all-you-need/)
- [Article on Positional Embeddings](https://medium.com/nlp-trend-and-review-en/positional-embeddings-7b168da36605#:~:text=Sinusoidal%20positional%20embeddings%20generates%20a,to%20learn%20the%20relative%20positions.)
  

