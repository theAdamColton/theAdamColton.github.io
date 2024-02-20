<script>
	import MediaBox from "./MediaBox.svelte";
	import ProjectBox from "./ProjectBox.svelte";
</script>

<svelte:head>
	<title>Home</title>
	<meta name="description" content="Adam Colton" />
</svelte:head>

<div class="column">
	<div class="row">
		<img src="me.jpg" alt="me" height="400" />
	</div>
	<div class="row">
		<img src="facepic.gif" alt="My mug" height="400" />
	</div>

	<div id="spinningcube">
		<div style="transform: translate3d(0em, 0em, -1em)">Studious,</div>
		<div style="transform: translate3d(0em, 1em, 0em)">Diligent,</div>
		<div style="transform: translate3d(0em, 2em, 1em)">Brave,</div>
		<div style="transform: translate3d(0em, 3em, 2em)">Peaceful,</div>
		<div style="transform: translate3d(0em, 4em, 3em)">That's me:</div>
		<div style="transform: translate3d(0em, 6em, 4em)">Adam</div>
		<div style="transform: translate3d(0em, 7em, 5em)">Colton</div>
	</div>
</div>

<hr />
<div class="section-title">Projects:</div>

<ProjectBox>
	<a href="/generative-modelling-of-bits-from-compressed-image-files"
		>Generative modelling of compressed bits from image files</a
	>: Do you have issues with achieving GPU saturation because of your
	dataloading load? Don't you wish you could train directly on compressed image
	files? Say no more! I trained llama to directly generate the bits of a lossy image compression file format called spiht. Check out my report! There will be more coming soon.
	<MediaBox>
		<video object-fit="fill" autoplay controls muted loop>
			<source src="spihtter/llama-training-mnist.mp4" type="video/mp4" />
		</video>
	</MediaBox>

	<a href="https://github.com/theAdamColton/spihtter">source code available</a>
</ProjectBox>

<ProjectBox>
	<a href="https://github.com/theAdamColton/spiht-py">spiht-py</a>: An
	implementation of the <a href="https://spiht.com/">SPIHT</a> algorithm in
	Rust, with Python bindings SPIHT is an lossy image compression algorithm. Like
	JPG, it can reduce the amount of bits required to store images. Unlike JPG,
	the SPIHT bitstream can be interrupted at any point and the entire image
	decoded. There are no 'blocks' in the SPIHT algorithm.

	<MediaBox>
		<img
			src="spiht/motorcycle.gif"
			height="300"
			alt="animation of the spiht algorithm"
		/>
		Left: Intermediate decoded image
		<br />
		Right: Intermediate coefficients from the discrete wavelet transform.
		<br />
		Bits per pixel (BPP) are shown in the top left corner
		<br />
		As BPP increases, you can see how the encoder assigns information to coefficients
		at higher frequencies.
	</MediaBox>

	<a href="https://github.com/theAdamColton/spiht-py">source code available</a>
</ProjectBox>

<ProjectBox>
	<a href="/your-vae-sucks">Your VAE Sucks</a>
	A short foray into the Forier transform, JPG, Image Autoencoders, and a new image-autoencoder
	architecture inspired by jpg, that produces latent codes with a left-to-right positional
	bias.
	<MediaBox>
		<img
			src="figures-dct-autoencoder/skijump.gif"
			height="400"
			alt="decoded ski jump animation"
		/>
	</MediaBox>
</ProjectBox>

<ProjectBox>
	<a href="/llms-and-faithfulness-to-reasoning"
		>LLMs and faithfulness to reasoning (blog post)</a
	>: Humans have written a trove of step-by-step explanations and proofs. These
	exist on the internet, and some of them end up in the pre-training data for
	large language models. Thus, some modicum of step-by-stepedness exists in the
	weights of LLMs....
</ProjectBox>

<ProjectBox>
	<a href="https://github.com/theAdamColton/vq-clip">VQ-CLIP</a>: So, you've
	heard about
	<a href="https://arxiv.org/abs/2103.00020">CLIP</a>, but have you heard about
	CLIP but with a
	<div class="rainbow-text-animated" style="display: inline">
		quantized embedding space
	</div>
	??

	<br />
	<br />

	Rather than using a real number vector as the embedding, VQ-CLIP uses a list
	of k integer class IDs. The embedding codes from these models can be used for
	exiting downstream tasks, such as autregressive CLIP embedding generation.

	<br />
	<br />

	You can find code+models
	<a href="https://github.com/theAdamColton/vq-clip">here</a>
</ProjectBox>

<ProjectBox>
	<a href="/a-picture-is-worth-8x8x8-words"
		>Image Retrieval: A Picture is worth 8x8x8 Words</a
	>: SOTA image retrieval models usually use global real-number embeddings,
	obtained from big neural networks. But why is there no love for more
	traditional information retrieval techniques such as BoW? Given a 256x256
	image, we encode it into a 8x8x8 matrix of discrete integer tokens. We do this
	using a ResNet model trained with a learned
	<a href="https://github.com/lucidrains/vector-quantize-pytorch"
		>vector quantization</a
	>
	layer. Using these tokens, we use
	<div class="rainbow-text-animated" style="display: inline">2D Kgrams</div>
	to obtain global vectors containing term frequencies.

	<div class="column">
		<MediaBox>
			<img src="2d_kgrams.png" alt="2d kgrams diagram" height="300" />
			<div class="video-description">
				<br />
				Given an arbitrarily sized 3D input tensor of token IDs, we can obtain a
				vector of gram frequencies, also called a bag-of-words (BoW). For tokens
				obtained from CNNs with vector quantization, this BoW can contain rich information
				about the activation at this layer.
			</div>
		</MediaBox>
		<MediaBox>
			<img
				src="image-ret-poster.png"
				alt="Our group's poster. Thanks for voting for us!"
				height="300"
			/>
		</MediaBox>
	</div>
</ProjectBox>

<ProjectBox>
	<a href="https://1anza.github.io/dataviscourse-pr-gravitymarket/"
		>Gravity Market</a
	>:
	<br />

	A D3 javascript web app which uses a physics simulation to display the percent
	change in value of various stocks from the S&P 500. This project was made for
	Vis for Data Science 2022 taught by Dr. Alexander Lex at the University of
	Utah. Source code to be added soon!
</ProjectBox>

<ProjectBox>
	<a href="songsfromukraine.pdf"> Songs from Ukraine </a>:

	<br />
	I scraped several hundred gigabytes of social media posts and analyzed the music
	found in them. Videos were scraped from the Telegram social media platform. Music
	metadata was retrieved from the videos using Shazam. <br /><br />
	Videos were retrieved over the course of months. The scraping process was scheduled
	using SystemD units on a linux server. CSV metrics on the data were updated asyncronously
	via file changes. The code for the scraper and data processing was written in Python.
</ProjectBox>

<ProjectBox>
	<a href="scivispreport.pdf">
		Visualization of a Computationally Derived Fentanyl Binding Protein
	</a>:

	<br />
	I used PyMol to create an animation of a binder enzyme as it transitions to the
	bound state. The program 'Climber' is used to interpolate between the bound and
	apo states.
</ProjectBox>

<ProjectBox>
	<a href="https://github.com/theAdamColton/ascii-unmasked"
		>Ascii Art Latent Masked Transformer Model</a
	>:
	<br />
	I trained a variational quantized autoencoder on ASCII art. The characters in the
	art are represented as one hot encoded vectors at each 'pixel' in the string. The
	discrete latents learned by this model were used to train a bidirectional transformer.
	The transformer was trained to predict masked latent tokens, akin to
	<a href="https://arxiv.org/pdf/2202.04200.pdf">MaskGit</a> by Google research.

	<MediaBox>
		<video object-fit="fill" autoplay controls muted loop>
			<source src="ascii-unmasked-demo.mp4" type="video/mp4" />
		</video>

		<div class="video-description">
			<p>
				This video shows the interpolation of the latent space of the VQ VAE. In
				the top left corner you can see a visualization of the discrete tokens
				which compose the latent space. The left ascii art is the decoded
				representation of the current embedding. On the right is the unaltered
				ascii art that is being interpolated to.
			</p>

			<p>
				First, a new datapoint is randomly pulled from the set of training
				items. This is shown as the ascii art on the right side. Interpolation
				is done over spacial dimensions, truncating or padding with zeros the
				interpolated-from and interpolated-to embedding tensors and linearly
				combining them based on the current step in the interpolation. You can
				observe the dimensions of the embedding space shift with this
				interpolation.
			</p>

			<p>
				Next, the masking step can be observed. Randomly, tokens in the top
				right are masked and turn red. The bidirectional transformer then in one
				pass reconstructs the set of tokens. The reconstructed tokens and
				decoded ascii art are then shown for about a second before again
				starting interpolation to the next data point.
			</p>
		</div>
	</MediaBox>
</ProjectBox>

<ProjectBox>
	<a href="https://github.com/theAdamColton/ethminer-gui">Ethminer GUI</a>:
	<br />
	A simple cross platform GUI app written in Rust for the ethminer CLI program. It
	includes capturing of console output from the program, and asyncronous channel
	communication using Tokio.
</ProjectBox>
