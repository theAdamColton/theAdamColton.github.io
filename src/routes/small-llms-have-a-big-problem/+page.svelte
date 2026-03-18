<script>
	import AsideBox from "../AsideBox.svelte";
	import InlineFootnote from "../InlineFootnote.svelte";
	import MediaBox from "../MediaBox.svelte";
</script>

<svelte:head>
	<title>Small LLMs have a Big Problem</title>
</svelte:head>

<h1>Small LLMs Have a Big Problem</h1>

<p>
	I discuss and verify a widely know issue with LLM architecture - the token
	embedding and LM head. Replacing these two pieces with a bits-centric LLM
	solves the issues when tested on a toy problem. I demonstrate that the
	bits-centric approach scales to testing on textual and image pretraining.
</p>

March 18th 2026

<hr />

<h3>LLMs Fail at this Simple Toy Problem</h3>

<p>
	I create a simple toy dataset where the first token is sampled randomly from
	the vocab size of the LLM, and the remaining tokens are taken as the same as
	the first. For example, the first token is <code
		>random.randint(0, 1-vocab_size)</code
	> and the remaining tokens in the sequence are the same as the first.
</p>

<p>
	I train a small 70 million parameter LLM on these sequences for 50,000 steps.
	Lo and behold it cannot learn it at ALL.
</p>

<MediaBox>
	<img
		src="small-llms-have-a-big-problem/loss_over_steps_vanilla_toy_problem.webp"
	/>
	Cross entropy training loss of vanilla LLM trained on SpamLang.
</MediaBox>

<p>
	For example, with a prefix of <code
		>117385, 117385, 117385, 117385, 117385, 117385, 117385, 117385</code
	>, the fully trained vanilla model autoregresively generates this
	continuation:
	<code>55571, 55571, 55 571, 55571, 55571, 55571, 55571, 55571, ...</code>
	All it has to do is learn how to copy the tokens in the prompt and repeat them
	a bunch of times! Why can't it learn this simple pattern?
</p>

<h3>Why it fails and one potential solution</h3>

<p>
	My first thought is that its failure is obvious. Consider that I only train
	the model for 50,000 steps at a batch size of 1. That means with its
	vocab_size of 131072 it only sees about 38% of the possible SpamLang samples.
	The remaining 62% of the token_embeddings recieve absolutely zero gradients
	whatsoever. But despite this seeming like a trivial issue resolved through
	more data I feel like it is a fundemental flaw in the way LLMs treat language
	tokens.
</p>

<p>
	More data and more compute can solve anything. But what if we want small
	models trained with small compute? LLM researchers seem not to care much about
	training impoverished models but we can find answers from the field of image
	generation where researchers train much smaller models with much fewer
	training samples. Usually images are generated using continuous latents or
	pixels but there is a less common paradigm of generating <i>discrete</i> image
	tokens in a similar way that LLMs generate <i>discrete</i> language tokens.
	Researchers have noticed that these small image models fail to fit to images
	tokenized with very large vocab sizes. Recently a paper from BitDance
	demonstrated a promising potential solution to this problem.<InlineFootnote
		>BitDance: Scaling Autoregressive Generative Models with Binary Tokens<br
		/><a href="https://arxiv.org/pdf/2602.14041">[arxiv]</a></InlineFootnote
	> Instead of treating discrete image tokens as a classification problem, they instead
	treat the token ids as bits that are diffused using a custom diffusion head.
</p>

<p>
	Simply replacing the bit diffusion head is not enough to solve the toy
	problem. The token embedding still has the issue that 62% of the token
	embeddings are never used. So I made a modified bit-centric LLM with both a
	bit-embedding and a bit-diffusion head. Instead of using a learnable vector
	embedding for each of the <code>m</code>
	possible token ids, I use a linear projection that projects the
	<math>log2(m)</math> bits of the token id into the models hidden size. The bit
	diffusion head doesn't output logits like the vanilla LLM. Instead it takes as
	input the noisy bits of a token_id and gradually denoises them until it
	dicretizes them and converts the bits back to an integer token id.<InlineFootnote
		>I use x-prediction with a v-prediction loss but otherwise keep the flow
		matching as simple as possible.</InlineFootnote
	>
</p>

<p>
	I train this bit-centric model on the SpamLang dataset using the same training
	settings. The differences are stark. Within 20,000 training steps the
	bit-centric LLM completely learns the pattern.
</p>

<MediaBox
	><img
		src="small-llms-have-a-big-problem/vanilla_vs_bit_diffusion_spamlang.webp"
	/>
	Autoregressive accuracy of Vanilla LLM vs Bit-centric LLM. Models are given the
	first 51 repeating tokens and need to autoregressively predict the remaining 205
	tokens.
</MediaBox>

<p>
	To the bit diffusion model this pattern is simple to learn. It doesn't matter
	that it is only every trained on 38% of the possible token ids because it
	learns a generalizable pattern that operates on the <b>bits</b> of the token ids
	instead of directly on the very large space of token ids.
</p>

<p>
	"Awesome!", you might be thinking, "Let's all switch to bit-centric LLMs!".
	Not so fast - I want to first see if this can work on real language tokens
	from real datasets. It may be the case that the Vanilla LLM fails miserably on
	the SpamLang dataset but excels on the real data. After all, bits are not a
	intuitive natural representation of language. Before jumping to any
	conclusions I want to compare the vanilla LLM to the bit-centric LLM on
	FineFineWeb.<InlineFootnote
		>Currently the most downloaded text dataset on <a
			href="https://huggingface.co/datasets/m-a-p/FineFineWeb">[huggingface]</a
		></InlineFootnote
	>
</p>
