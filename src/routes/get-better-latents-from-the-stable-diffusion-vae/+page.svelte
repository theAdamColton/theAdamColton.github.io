<script>
	import InlineFootnote from "../InlineFootnote.svelte";
	import MediaBox from "../MediaBox.svelte";
</script>

<svelte:head>
	<title>Get Better Latents from the Stable Diffusion VAE</title>
</svelte:head>

<div class="column">
	<div class="row"></div>
</div>

<div class="rainbow-text-animated">
	<h1>
		Get Better Latents from the Stable Diffusion VAE : Inference time
		optimization of encodings
	</h1>
</div>

<h2>Summary</h2>

I show a trick that lets you spend some extra compute to get better image
encodings. The result is reconstructed images with perfect faces and perfect
text. First encode the image to the VAE latent space, then treat the latent
means and variances as optimizable weights. A few optimization steps and a
handmade loss function fixes issues in the reconstructed pixels, without ruining
the interpretability for downstream models.

<h2>Motivation</h2>

<p>
	It is important for VAEs to have proper regularization. That's why it's a VAE
	and not a AE. The VAE outputs a 3D grid of latents. Each latent has a mean and
	variance. That's because we want the latents to behave as if they were
	normally distributed. When the VAE is being trained, there's a KL-divergence
	loss that makes the latents tend towards being normal uniform.
</p>

<p>
	The idea is that if the VAE latents look like they are normal uniform, the
	latent space is simpler, and easier to make sense of for downstream models
	that use (or rip off) the stable diffusion VAE. <InlineFootnote
		>Will any serious research material ever be released on Dalle 3 or Sora or
		GPT 4?</InlineFootnote
	>
</p>

<p>
	So for training, it's important that the VAE latents are nicely behaved and
	uniform looking. But sometimes too much regularization forces the VAE to make
	mistakes. For example, the VAE can produce horrifically mutilated faces. Us
	humans have a knack for facial perception so bad faces really stand out. Also
	text can be ruined by the VAE. How do we fix the VAE so that it can encode
	images to 'better' latents that can be decoded to perfect faces and
	characters? There might be some way to finetune the VAE weights without
	ruining the embeddings for use in models that were trainined on the original
	VAE. But finetuning the VAE is prohibitively expensive. Instead, why not
	finetune the latents themselves?
</p>

<h2>Latent inversion</h2>

<p>
	Say you have a generative model, that can accept latent variables and produce
	data samples. Usually for the forward pass of the model you give the model the
	latents, and you get a generated data point. Inverting a latent means that you
	don't care about generating a data point; you already have the data point.
	Instead you care about finding the latent varibles that when given to the
	generative model produce your data point.
</p>

<p>
	There are many different ways latent inversion can be useful. In the land of
	stable diffusion, there's prompt-inversion <InlineFootnote
		>also known as text inversion https://textual-inversion.github.io/</InlineFootnote
	>, where you can train a prompt embedding that can capture the <i>essence</i>
	of a set of images. Latent inversion have also shown up in GAN research<InlineFootnote
		><a href="https://arxiv.org/abs/1912.04958">
			Analyzing and Improving the Image Quality of StyleGAN</a
		> page 19 section D
	</InlineFootnote>.
</p>

<p>
	Latent inversion can be used to get better VAE embeddings, assuming we already
	have the pixel values of an image we want to embed. MSE loss doesn't work well
	as a pixel-to-pixel reconstrucion loss. There are better metrics that are also
	differentiable such as LPIPS and
</p>

<MediaBox>
	<img src="get-better-latents-from-the-stable-diffusion-vae/diagram.jpg" />
	Simple latent inversion using an encoder and a decoder. <br />
	The image is run through the encoder to get the latents. Then the latents are repeatedly
	decoded and are optimized for some loss function.
</MediaBox>
