<script>
	import AsideBox from "../AsideBox.svelte";
	import InlineFootnote from "../InlineFootnote.svelte";
	import MediaBox from "../MediaBox.svelte";
</script>

<svelte:head>
	<title>generative modelling of compressed image file bits</title>
</svelte:head>

<h1>Generative modelling of compressed image file bits</h1>

Februrary 20th 2024

<MediaBox>
	<video autoplay controls muted loop style="width: 40vw;">
		<source src="spihtter/llama-training-mnist.mp4" type="video/mp4" />
	</video>
	mnist digit generation as training progresses, 2M parameter llama model
	<a href="https://github.com/theAdamColton/spihtter"
		>https://github.com/theAdamColton/spihtter</a
	>
</MediaBox>

<hr />

<AsideBox>
	<u slot="title"> The holy grail </u>
	<img src="spihtter/holy grail.jpg" style="height: 3em;" slot="image" />
	<p>
		Tales have long told of a great and mythical image-text architecture.
		Stories passed down through the generations called it the holy grail.
		Parents, after a long hard day of prompting, would come home to their
		children and speak of a day when their tedious prompting would be no more.
		It would be the holy grail and would generate images using the direct
		bits/bytes of a compressed file format.
	</p>

	<p>
		Most called it a fools errand but many were desperate to search for the
		grail. Despite much research<InlineFootnote
			>some found that LLMs can use VQ-coded images and use bidirectional
			attention as well as a parallel sampling scheme for image tokens.
			(Chamel3on and <a href="https://arxiv.org/abs/2309.16039"
				>Effective Long-Context Scaling of Foundation Models</a
			>) <br /> others discovered that llms could input and output CLIP
			embeddings (<a href="https://arxiv.org/abs/2301.13823"
				>Grounding Language Models to Images for Multimodal Inputs and Outputs</a
			>)
		</InlineFootnote> to this day the holy grain remains out of reach. Whether it
		was to do with anti-research or proprietary datasets, none could tell.
	</p>
</AsideBox>

<h2>Images and pixels</h2>

<p>
	Images have two dimensions. They have a height and a width. At each position
	in a 2D grid is a pixel which has 3 bytes that are the pixel's colors. The
	pixels themselves are captured by a piece of photosensitive material which
	samples light rebounding off of objects in the real world. The real world is
	where physics happens and makes light bounce around from stuff going on.
</p>

<p>
	So there are two worlds: pixel world and the real world. Pixel world, though
	being just a sample of the real world, can nonetheless still communicate
	complicated information about the real world. Sampling down to 2D doesn't rid
	all of the intracacies of the real world. Images still have to portray scenes
	that maintain coherent portrayal objects moving through time and 3D space.
	Moreover, images aren't restricted to physical 3D objects, they
	also can show written text and numbers. Imagine an entity that lives in 2D
	pixel world and can only ever see and interact in a 2D plane of existance.
	Would this entity have enough information to figure out all of the underlying
	facts of the real world that their 2D existence is sampled from? <InlineFootnote
		>This is an idea explored in the Three-Body-Problem sci-fi series. Also this <a
			href="https://www.physicsforums.com/threads/how-would-a-2-dimensional-intelligence-find-out-the-world-has-3-dimensions.421760/"
			>forum</a
		> has some interesting thoughts on this.</InlineFootnote
	> It turns out that in 2d pixel world life isn't really any more simple than in
	real world.
</p>

<p>
	Images being virtually unrestricted in their expressive power is bad news for
	our generative models. If we train a model on pixels and we don't specify any
	other restrictions, we are in effect expecting the model to not only figure
	out snapshots of the natural world, but also understand all bodies of written
	text and mathematical proofs and also physics and geometry and psychology and
	biology. That is why I think people are so eager for an all-in-one holy grail
	model that does everything. Because in the skillsets demanded of LLMs and
	image gen is the same.
</p>

<p>
	It used to be the case that LMs couldn't generate text with proper grammar and
	a whole bunch of research was done on decoding within the bounds of correct grammatical sytax.
	Well, nowdays our image models can't generate within the bounds of physics. I
	think physics is more complicated than grammar. Why not force our image models
	to generate within some sort of constrained system?
</p>

<p>
	We aren't designing handmade generation constraints and features these days,
	or at not as much as we used to. Models get trained on data that is as close
	to the pure input and output modalities as possible, end-to-end. For images
	this means means no more SIFT features or handmade edge kernels. <InlineFootnote
		>Lex Fridman's interview with Jitendra Malik is very interesting, <a
			href="https://youtu.be/LRYkH-fAVGE?t=3658">here</a
		> he talks about the transition from the 'old-style' computer vision to the more
		end-to-end approach</InlineFootnote
	> This avoids fettering inputs and outputs with any prior assumptions about structure.
	In image generation this idea takes shape in the form of training on raw pixels.
	In video generation, this means using pixels from videos frames. For 3D data, this
	means something like implicit neural representations. I'd say the vast majority
	of the recent sucessful generative models use this end-to-end approach. <InlineFootnote
		>LLMs which use tokenizers which are arguably 'hand-picked' features. The
		less restrictive approach would be training on the direct characters/bytes.</InlineFootnote
	>
</p>

<p>
	There are still some modelling strategies being used that are less expressive
	but impose more structure on the inputs and outputs. The model is not given
	the 'raw' data, it is given a limited version. You're still expecting your
	model to perform well on the raw data but you only are giving it some gimped
	version of the data, which is not ideal. If given a choice, most researchers
	would prefer to use the end to end, but are forced to make concessions for
	stability or performance reasons. Some examples of this for 3D data are
	techniques that don't use nerfs <a href="https://arxiv.org/abs/2111.12480"
		>Octree transformer</a
	>
	where 3D data is generated by autoregressively decoding Octrees, and
	<a href="https://arxiv.org/abs/2311.15475">Mesh GPT</a>
	for autoregressive generation of vector quantized codings of vertices. For
	images and sound, the recent paper
	<a href="https://arxiv.org/abs/2306.00238"
		>Bytes Are All You Need: Transformers Operating Directly On File Bytes</a
	> showed high quality image and sound classification done using the raw bytes of
	encoded files. These strategies tie in with the idea of constrained decoding. It's
	similar to generating structured data (like json) with an LLM. Octree-transformer
	and Mesh GPT during generation have deterministic code than runs in step with the
	model, allowing and disallowing actions.
</p>

<MediaBox>
	<img
		src="spihtter/yann-tweet.jpg"
		alt="https://x.com/ylecun/status/1625127902890151943?s=20"
	/>
</MediaBox>

<p>
	To summarize, there's the idea that we divvy our creations up into two distict
	worlds. The real world, where all the complicated signals are, and the world
	that programs live in. There is a massive chasm between realness and
	virtualness. The trend seems to be that machine learning needs to learn from
	the real world. But I'd argue that the digital world holds a lot, probably
	enough. I think that it shouldn't matter to train image models on full .RAW
	images and decode uncompressed pixels and videos. The algorithms that provided
	the media that fueled the digital boom: html,jpg,mp3, are all handmade and
	intricate and beautiful. Why should we suddenly desert them? By the 90s we had
	given up on all of the raw data formats. These encoding algorithms are
	suboptimal, but they do a fantastic job at linking the real world and the
	digital world with minimal overhead, and they do it more than well enough for
	human perception. It shouldn't matter that our models aren't given the real
	deal.
</p>

<h2>An image as a left-to-right sequence of tokens</h2>

<p>
	Autoregressive decoder-only token-based transformers work pretty well. They
	have an objective that matches well with the idea of max entropy, and they
	scale easily, and can usually generate samples matching their data
	distribution without many tweaks. For images, the landscape of autoregressive
	generation is a bit desolate. Granted, there have been advancements in
	discrete modelling of images as a 2D array of tokens. This is not the same as
	autoregressive generation though, tokens are sampled from a 2D grid of
	probabilities. <InlineFootnote
		>masked tokenized image generation. see: magvit, vqvae, taming transformers,
		Meta's Chamel3on,
	</InlineFootnote> Fewer attempts are made at left-to-right autoregressive conditional
	modelling. <InlineFootnote
		>Using residual VQ you can sort of do part masked image sampling and part
		autoregressive modelling, see <a href="https://arxiv.org/abs/2203.01941"
			>Autoregressive Image Generation using Residual Quantization</a
		></InlineFootnote
	>
</p>

<p>
	Previous works have flattened images in a variety of creative ways. The
	simplest way to flatten an image into a 1D sequence is to use raster ordering
	of all pixels. This is what is experimented with in the classic <a
		href="https://arxiv.org/abs/1601.06759">pixel rnn</a
	>. <a href="https://arxiv.org/abs/2305.07185">MEGABYTE</a> used a slightly fancier
	patch scan order.
</p>

<p>
	Conceptually, raster pixel ordering and even patch scan ordering is a very bad
	way to treat an image as a sequence of tokens. Image asking a painter to
	create an art piece by starting in the top left corner, and from there only
	allowing them to make one brush stroke at every position while moving
	left-right, top-down in the painting. It would make everything much more
	difficult for the painter. A more natural way to add detail to a painting over
	time is to first start with a rough background, then create the overall
	composition of colors, then add particular strong edges and shapes, and then
	move in and add fine details to certain areas.
</p>

<MediaBox>
	<img src="spihtter/bob ross.gif" />
	Bob Ross and other artists do not make paintings by placing brush strokes in a
	raster scan ordering.
</MediaBox>

<p>
	We want to turn an image into a sequence of actions that create the image, in
	a way that is easy to explain and follow. In a way, the image itself is a
	place for planning. The first actions are noncommittal but set the general
	stage for what the image will finally look like. Human artists have a sense of
	what to start with when making a painting. Unless you have a dataset of
	millions of brush stroke histories we can't really mimic this sequential
	behavior. <InlineFootnote>Looking at you adobe</InlineFootnote> But I think that
	image compression algorithms are expert demonstrators of step-by-step image synthesis.
</p>

<MediaBox>
	<img src="spihtter/owl.jpg" />
	Tokenizing an image can be interpreted as translating an image into a sequence
	of expert instructions. The instructions have to be 'intuitive', and when followed,
	actually result in the creation of something close to the original image. When
	we train our LLM to produce the image, we want it to generally mimic these expert
	instructions.
</MediaBox>

<h2>Image compression algorithm</h2>

<p>
	There are lots of image compression algorithms, but we want one with certain
	properties. Firstly, the bit-stream should have a left-to-right bias.
	Secondly, it would be nice if bit stream could be able to be interrupted at
	any point and decoded into an image. This would let us have previews during
	the slow autoregressive generation. Also, we want the coding to be very
	resiliant to noisy tokens. Perhaps most importantly, there should be a way for
	the decoder/encoder of the compression algorithm to provide extra conditioning
	information to the autoregressive model about what is going on. We want this
	property so it removes the burden of the autoregressive model needing to
	reverse engineer in its weights the compression algorithm.
</p>

<MediaBox>
		<img src="spihtter/ideal image compression.jpg" />
		We want to obtain a stream of tokens that describes a sequence of actions
		to construct the image. We also want to get conditioning tokens.
		Conditioning tokens aren't used directly by the encoder/decoder, but are
		given as additional information for the downstream autoregressive
		generative models. After the autoregressive model generates each token, it
		feeds the new sequence to the decoder and gets an additional conditioning
		vector. It is difficult to train both the image compression model and the
		autoregressive model at the same time and pass gradients between them. How
		do we know what information will be useful for the generative model? In
		this project I use handmade features.
</MediaBox>

<p>
	<a href="https://spiht.com/">SPIHT</a> (pdf
	<a href="https://spiht.com/EW_Code/csvt96_sp.pdf">here</a>) is an old image
	compression algorithm from the 90s that satisfies many of our requirements.
	First, it uses the DWT transform to decorrelate RGB pixel values. The
	<a href="https://en.wikipedia.org/wiki/Discrete_wavelet_transform"
		>DWT decomposition</a
	> of an image makes a 2D grid of sparse coefficients. They are placed in a way
	that the coefficients that aren't close to zero are arranged in predictable clusters.
	Importantly, MSE distance is preserved in the DWT.
</p>

<p>
	The goal of the SPIHT encoder is to output a sequence of bits from most
	importance to least importance (where importance is measured by MSE between
	the output image and the current encoded representation). DWT coefficients can
	be arranged in a tree structure. SPIHT does a kind of BFS of this tree in an
	ordering of most-to-least significance in the bit plane.
</p>

<p>
	I think SPIHT is better for autoregressive modelling than JPG or JPGXL or HEIC
	or png. At any point, the bitstream can be interrupted and the full image
	decoded. Each bit has a direct relationship to a single coefficient (or it's
	decendents). Each bit is explainable, and clearly attributable to pixels in
	the image. It does not use entropy encoding; one bit is one action <InlineFootnote
		>although entropy coding isn't something that would stop us from training an
		autoregressive model. Entropy coding reminds me a lot of tokenization; how
		to find an efficient sub-word vocab.
	</InlineFootnote>. Each bit has a simple and explainable role in the resulting
	final image. So coming up with a way to produce SPIHT specific token
	conditioning information is not too difficult.
</p>

<MediaBox>
	<video autoplay controls muted loop>
		<source src="spihtter/doggo.mp4" type="video/mp4" />
	</video>
	animation of spiht decoding
	<br />
	BPP stands for bits per pixel. The last image is of size 120 kb
	<br />
	On the right are the DWT coefficients in a packed array
</MediaBox>

<h2>Assisted SPIHT decoding - Conditioning tokens</h2>

<p>
	It's possible to take an AR model, and train it directly on SPIHT bits where
	each bit is a token. This would be wasteful because it completely discards all
	advantage of how explainable each bit of the SPIHT encoding is. By default, by
	just training on the bits themselves, we're basically expecting the NN model
	to fully memorize how the decoding algorithm works and the current variables
	at each step of the SPIHT decoder/encoder. SPIHT is close to being a tree
	decoding algorithm. You can 'tell' the NN where it is in the tree at each next
	token. We can infuse our input tokens with extra information about the current
	state of the SPIHT decoder right as it is consuming the next bit. This is
	information that is available during inference time.
</p>

<p>
	What can be used from the spiht algorithm as conditioning at each bit? There's
	the height and width and channel position of the next coefficient to be
	queried. <InlineFootnote
		>When I say 'queried' I mean that it is the singular coefficient that the
		next bit is answering some information about</InlineFootnote
	> There's also the next coefficient's value in the coefficient array. Also, there's
	the local variable 'n'. Also, there's the coefficient's depth, and it's filter
	array. There's also the type of query being asked of the coefficient; there are
	7 lines in the SPIHT encoder/decoder code where bits are emitted/imitted. So overall,
	8 concrete values.
</p>

<h2>Spihtter</h2>

<p>
	Put altogether I call this model architecture 'Spihtter'. Spihtter inputs and
	outputs a left to right sequence of discrete tokens. It recieves additional
	embeddings per-token, representing the state of the spiht decoder.
	Technically, any model could be used to produce next token logits.
</p>

<p>
	The only tricky part is keeping the model fed with spiht metadata during
	inference. Per generated token, this is a linear time operation. But you have
	to keep track of where you are in the SPIHT decoding algorithm. I implemented
	a spiht encoder/decoder in rust which is used to build a dataset of training
	examples. During inference I use a different python implementation of the
	spiht algorithm. I do this so I can use python's yield feature and access
	spiht bits in a stream.
</p>

<p>
	I place image bits inside of html tags, like this: &lt;spiht h=28 w=28
	n=11101&gt;010101011110111101&lt;/spiht&gt;
	<br />
	Spiht needs some extra information to start decoding. Namely, the height and width
	and starting value of the variable 'n'. I encode these values as html attributes.
</p>

<h2>Experiments</h2>

<i>
	All experiments can be reproduced from the code base here:
	<a href="https://github.com/theAdamColton/spihtter"
		>https://github.com/theAdamColton/spihtter</a
	>
</i>

<p>
	I train different Spihtter models on some small toy datasets. I am currently
	gpu poor and did these small experiments using an M1 macbook air with 8GB of
	memory.
</p>

<p>
	For generation I use greedy search. I did not play around with sampling
	settings. I left hyperparameters at their recommended values
</p>

<h3>MNIST</h3>

<p>
	28x28 greyscale MNIST digits can be encoded in as few as 256 tokens using
	SPIHT. The small llama model can be trained to produce coherent digits in a
	very short wall clock time, about 4 minutes. Mamba fails to produce correct
	mnist digits.
</p>

<MediaBox>
	<div class="column" style="padding: 10px;">
		<div class="rowentry">
			<img src="spihtter/mnist dataset samples.png" style="width: 10vw;" />
		</div>
		Random dataset samples, compressed to 256 bits. These are the images decoded
		from the ground truth bits.
		<div class="rowentry">
			<img src="spihtter/mnist mamba samples.png" style="width:10vw;"/>
		</div>
			digits 0-9 after 1 hour of training, decoded from .3M parameter mamba model,
			batch size 16, 1000 steps
		<div class="rowentry">
			<img src="spihtter/mnist llama 1000.png" style="width:10vw;"/>
		</div>
			digits 0-9 after 10 minutes of training, decoded from 2M parameter llama model,
			batch size 16, 1000 steps
	</div>
</MediaBox>

<MediaBox>
	<img src="spihtter/mamba-vs-llama-mnist.jpg" />
	mamba: purple, vs llama: grey, NLL loss. To be fair, LLaMa had almost 10x the number
	of parameters, but also trained 6x faster.
	<img src="spihtter/llama-metadata-vs-no-metadata-mnist.jpg" />
	The blue loss curve is without any conditioning information from the spiht decoder. The grey loss curve is from giving the LLM embeddings based on the internal programmatic state of the spiht decoder. The conditioning information from the spiht algorithm vastly improves the ability to model the sequence. model used: llama-small.json
</MediaBox>

<p>
	The mamba model didn't match the performance of the llama model. I don't want
	to be overly critical. I'm relying on a hacked together implementation. I
	couldn't train using the 'MPS' device because 4D cumsum is not supported as of
	pytorch 2.2.3. So the training time was much longer. Worryingly, throughout
	training, most of the images generated from the 0-9 digits are exactly the
	same. I checked for bugs and couldn't find any; the mamba model is receiving
	the proper labels, but it seems that the information from the class label at
	the start of the sequence goes completely forgotten.
</p>

<h2>Future work</h2>

<p>
	I figured out that you can use spiht to encode latents from the stable
	diffusion VAE. It make perfect sense why it works. It looks kinda trippy, and
	the reconstruction quality is quite high at around .25 bpp for most images.
	I'm doing more research on neural image compression so I'm going to come back
	to this topic later once I understand the literature better.
</p>

<MediaBox>
	<video autoplay controls muted loop>
		<source src="spihtter/doggo-vae.mp4" type="video/mp4" />
	</video>
	Instead of using spiht to compress pixel values, you can use it to compress latent values. Here is a reconstruction at different bit rates of stable diffusion VAE latents.
	<br />
	There are some artifacts, but you get a good image at around 0.25 bpp
</MediaBox>

<p>
	Using this trick you can get good images at around ~25k bits. It's also
	extremely useful if you are trying to save a huge dataset of VAE embeddings on
	disk. You can just save them as torch tensors, but using spiht as lossy
	compression you can save 60% of the disk space.
</p>

Again, it seems like the quest for the holy grail will have to continue.

<style>
	.rowentry {
		padding: 10px;
	}
</style>
