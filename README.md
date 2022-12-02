## Code release progress:

<strong>Todo:</strong>
- <strong>Instructions for training code</strong>
- <strong>Dataset preparation</strong>
- <strong>Interactive interface</strong>

Done:
- Basic training code
- Environment
- Pretrained model checkpoint
- Sampling code

# *UniColor:* A Unified Framework for Multi-Modal Colorization with Transformer

[Project page](https://luckyhzt.github.io/unicolor) | [arXiv](https://arxiv.org/abs/2209.11223) ([LowRes version](https://luckyhzt.github.io/files/unicolor-lowres.pdf)) | [ACM TOG Paper](https://dl.acm.org/doi/10.1145/3550454.3555471) | [BibTex](#bibtex)

![alt text](figures/teaser.png)

Zhitong Huang $^{1*}$, [Nanxuan Zhao](http://nxzhao.com/) $^{2*}$, [Jing Liao](https://liaojing.github.io/html/) $^{1\dagger}$

<font size="1"> $^1$: City University of Hong Kong, Hong Kong SAR, China &nbsp;&nbsp; $^2$: University of Bath, Bath, United Kingdom </font> \
<font size="1"> $^*$: Both authors contributed equally to this research &nbsp;&nbsp; $^\dagger$: Corresponding author </font>

## Abstract:
We propose the first unified framework <em>UniColor</em> to support colorization in multiple modalities, including both unconditional and conditional ones, such as stroke, exemplar, text, and even a mix of them. Rather than learning a separate model for each type of condition, we introduce a two-stage colorization framework for incorporating various conditions into a single model. In the first stage, multi-modal conditions are converted into a common representation of hint points. Particularly, we propose a novel CLIP-based method to convert the text to hint points. In the second stage, we propose a Transformer-based network composed of <em>Chroma-VQGAN</em> and <em>Hybrid-Transformer</em> to generate diverse and high-quality colorization results conditioned on hint points. Both qualitative and quantitative comparisons demonstrate that our method outperforms state-of-the-art methods in every control modality and further enables multi-modal colorization that was not feasible before. Moreover, we design an interactive interface showing the effectiveness of our unified framework in practical usage, including automatic colorization, hybrid-control colorization, local recolorization, and iterative color editing.

### Two-stage Method:
Our framework consists of two stages. In the first stage, all different conditions (stroke, exemplar, and text) are converted to a common form of hint points. In the second stage, diverse results are generated automatically either from scratch or based on the condition of hint points.
![alt text](figures/unified.png)

## Environments
To setup Anaconda environment:
```
$ conda env create -f environment.yaml
$ conda activate unicolor
```

## Pretrained Models
Download the pretrained models (including both Chroma-VQGAN and Hybrid-Transformer) from [Hugging Face](https://huggingface.co/luckyhzt/unicolor-pretrained-model/tree/main):
- Trained model with ImageNet - put the file `mscoco_step259999.ckpt` under folder `framework/checkpoints/unicolor_imagenet`.
- Trained model with MSCOCO - put the file `imagenet_step142124.ckpt` under folder `framework/checkpoints/unicolor_mscoco`.

To use exemplar-based colorization, download the [pretrained models](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/releases/download/v1.0/colorization_checkpoint.zip) from [Deep-Exemplar-based-Video-Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization), unzip the file and place the files into the corresponding folders:
- `video_moredata_l1` under the `sample/ImageMatch/checkpoints` folder
- `vgg19_conv.pth` and `vgg19_gray.pth` under the `sample/ImageMatch/data` folder

## Sampling
In `sample/sample.ipynb`, we show how to use our framework to perform unconditional, stroke-based, exemplar-based, and text-based colorization.

## Comments
Thanks to the authors who make their code and pretrained models publicly available:
- Our code is built based on "[Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers)".
- In exemplar-based colorization, we rely on code from "[Deep Exemplar-based Video Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization)" to calculate the similarity between grayscale and reference images.
- In text-based colorization, we use "[CLIP (Contrastive Language-Image Pre-Training)](https://github.com/openai/CLIP)" to calculate text-image relevance.

## BibTex
arXiv:
```
@misc{huang2022unicolor
      author = {Huang, Zhitong and Zhao, Nanxuan and Liao, Jing},
      title = {UniColor: A Unified Framework for Multi-Modal Colorization with Transformer},
      year = {2022},
      eprint = {2209.11223},
      archivePrefix = {arXiv},
      primaryClass = {cs.CV}
}
```

ACM Transactions on Graphics:
```
@article{10.1145/3550454.3555471,
      author = {Huang, Zhitong and Zhao, Nanxuan and Liao, Jing},
      title = {UniColor: A Unified Framework for Multi-Modal Colorization with Transformer},
      year = {2022},
      issue_date = {December 2022},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      volume = {41},
      number = {6},
      issn = {0730-0301},
      url = {https://doi.org/10.1145/3550454.3555471},
      doi = {10.1145/3550454.3555471},
      journal = {ACM Trans. Graph.},
      month = {nov},
      articleno = {205},
      numpages = {16},
}
```

