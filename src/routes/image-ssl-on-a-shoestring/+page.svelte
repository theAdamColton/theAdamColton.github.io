<script>
  import MediaBox from '../MediaBox.svelte';
  import AsideBox from '../AsideBox.svelte';
  import InlineFootnote from '../InlineFootnote.svelte';
</script>

<div class="rainbow-text-animated">
	<h1>Image Self Supervised Learning (SSL) on a Shoestring</h1>
</div>

June 6th, 2024

<hr/>

<h2>
	Introduction
</h2>

<p>
	These days there are so many different pretrained models that you can almost always find a particular  model for your exact purposes. Whether it be image encoders, image segmentation, image generation, joint image-text encoders, it seems like every possible type of pretrained model already exists and can be downloaded and used easily. 
</p>

<p>
	My niche requirements were met by no existing AI models. In this blog post I explain how I overcome the compute gap to train a competitive ViT-S from scratch.
</p>

<h2>
	Motivation
</h2>

<p>
	Pretraining an image encoder from scratch is expensive. It requires somewhere in the ballpark of 300 A100 GPU-hours. 1 A100 GPU-hour is a unit which means using one A100 GPU for one hour. If you have 300 A100 GPUs it will take you one hour to spend 300 A100 GPU-hours. 1 A100 GPU-hour costs about $1 on vastai. This only includes the training run and does not include costs encurred from experiments and hyperparameter tuning. For an average bloke without coorperate funding and no research grants these sort of resources are out of reach. 
	</p>

<p>
	Since pretraining a model from scratch is so expensive there are only a few cases where it makes sense.
</p>
<ul>
	<li>
	 Architectural requirements: Architecture is probably less important than we all think and what really matters is how much data you can pump through a network. But if you have a need for a particular network you will probably have to train it from scratch because adapting pretrained weights from an entirely different architecture won't work.
	</li>
	<li>
		License: Many open-weight models these days are released under the condition that you not use the weights for commercial purposes. In order to use the model for commercial purposes you will have to either pay the company a fee or train the model yourself.
	</li>
</ul>


<h3>
High throughput and varying resolution
</h3>

<p>
	I wanted an image encoder that met two specific requirements. Firstly, I wanted to have a very high throughput. A good number would be 1,000 images per second on a 3090 GPU. Secondly, I wanted an encoder that can process images of varying resolution. 
</p>

<p>
Convolutional neural networks don't meet these requirements. CNNs can be applied to varying resolution images but cannot be easily batched. In order to reach high throughputs, images have to be put into uniformly sized batches.
</p>

<p>
	The one prominent method for training transformer based vision networks on varying resolutions is NaViT. NaViT is a design pattern for training transformer models using various sized images. It has high throughput and can process images of varying resolution.
</p>

<p>
	The only architectural difference between a NaViT and a ViT is that the NaViT has factorized learnable position embeddings. ViTs use either 2D sinusoidal, or 2D (absolute) learned positional embeddings. This is only a small change in the overall network but ViTs cannot be used out of the box as NaViTs. 
</p>


<h2>
	Methodology
</h2>

<p>
	There are different pretraining strategies with different trade-offs for compute and quality. Which pretraining strategy is best is one that gives you the best benchmark results for the amount of GPU-hours that is within your budget. There were a few options that I considered as a pretraining objective.
</p>

<ul>
<li>
	<a href="https://arxiv.org/abs/2303.15343">Sigmoid Loss for Language Image Pre-Training</a>: This is a contrastive method which learns a joint embedding space for images and texts. They get 71% zero shot accuracy on imagenet 1k after training with about 1,000 A100-hours <InlineFootnote>This assumes that 1 TPUv4 is roughly equivalent to 1 A100</InlineFootnote>
</li>
<li>
	<a href="https://arxiv.org/abs/2301.08243">Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture</a> This is a method that uses masked latent prediction. They report 62.5% accuracy on imagenet 1k using a linear probe. This is after training with about 700 A100-hours <InlineFootnote>Figure 5, ViT-B/16</InlineFootnote>.
</li>
<li>
	<a href="https://arxiv.org/abs/2310.08584">Is ImageNet worth 1 video? Learning strong image encoders from 1 long unlabelled video</a> This is a method trained on a small number of videos and uses an emergent masking strategy. They get 72% on imagenet 1k after training with 3,200 A100-hours<InlineFootnote>The authors state in their <a href="https://openreview.net/forum?id=Yen1lGns2o&noteId=7om3d7wfTx">openreview comment</a> that the (DORA ViT-S), when using 8 A100 GPUs, takes 6 hours and 48 minutes to train one epoch of ImageNet1k. Table 10 shows a model trained for 60 epochs obtaining 71.9% accuracy using a linear probe.</InlineFootnote>.
</li>
<li>
<a href="https://arxiv.org/abs/2304.07193">DINOv2: Learning Robust Visual Features without Supervision</a>  get 81% accuracy on imagenet after using 4,500 A100-hours.
<InlineFootnote>The authors <a href="https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/MODEL_CARD.md">report</a> that the ViT-S required 4,500 A100 hours to train. It also was not trained from scratch because it requires a larger pretrained model for distillation loss. I can't find where they report the training time for the ViT-S without distillation.
</InlineFootnote>
</li>
</ul>

<MediaBox>
<img src="image-ssl-on-a-shoestring/SSL image pretraining efficiencies.png"
/>
Different self supervised learning pretraining methods, their cost in A100 GPU-hours, and their accuracy when used for ImageNet1K prediction. Sigmoid uses zero-shot prediction, whereas IJEPA, DORA and DINOv2 all train a linear probe.
</MediaBox>

<p>
From the above data it might that the "Sigmoid" method is the best. But this method has some negative aspects. First of all, there has to be a dataset of images and captions. This might not be ideal for certain circumstances where there are no captions available or they are low quality. Additionally, models trained using CLIP tend to have good imagenet scores, but suffer at counting objects or localizing features. On the other hand, DINOv2 has good all round performance.
</p>


<p>
Results from the IJEPA paper suggest that it is slightly better than DINOv2. But there are some caveats with IJEPA. First of all, other researchers have yet to replicate the results of 
</p>

<h2>
	Results
</h2>

<h2>
	Discussion
</h2>


<style>
	p {
	width: 70vw;
	}
	ul {
	width: 70vw;
	}
</style>
