<script>
	import "./styles.css";
</script>

<svelte:head>
	<title>
		Image Retrieval: A Picture is worth 8 &#x00D7; 8 &#x00D7; 8 Words
	</title>
	<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
	<meta name="generator" content="TeX4ht (https://tug.org/tex4ht/)" />
	<meta name="originator" content="TeX4ht (https://tug.org/tex4ht/)" />
	<!-- html -->
	<meta name="src" content="main.tex" />
</svelte:head>

<div class="maketitle">
	<h2 class="titleHead">
		Image Retrieval: A Picture is worth 8 &#x00D7; 8 &#x00D7; 8 Words
	</h2>
	<div class="author">
		<span class="cmr-12x-x-120">Adam Colton, Byron Liu, Lane Zaugg</span>
	</div>
	<br />
	<div class="date">
		<span class="cmr-12x-x-120">April 11th, 2023</span>
	</div>
</div>

<h3 class="likesectionHead"><a id="x1-1000"></a>Introduction</h3>
<!--l. 44-->
<p class="noindent">
	We tested and designed an image retrieval system that leverages traditional
	document similarity metrics during retrieval time. Our method first encodes
	input images into a 3D matrix of discrete tokens, and then computes k-grams in
	multiple dimensions. We then index images as a global Bag-of-Words (BoW)
	vector. We assessed the performance of our model on the GPR1200 dataset, and
	experimented with various similarity scoring methods for BoW. Using term
	frequency-inverse document frequency (tf-idf), we achieved a mean average
	precision (mAP) that is 6% worse than the comparable non-quantized method
	reported by the authors of GPR1200.
	<!--l. 49-->
</p>

<p class="noindent"></p>
<h4 class="likesubsectionHead"><a id="x1-2000"></a>Background</h4>
<!--l. 51-->
<p class="noindent">
	A common strategy for information retrieval is to compare the frequencies of
	different terms between different documents. Search engines such as
	<a href="https://docs.rs/tantivy/latest/tantivy/">Tantivy</a> and
	<a href="https://lucene.apache.org/">Lucene</a> retrieve documents using an
	inverted term index. BoW representations fit naturally into these
	implementations, and as a result have been widely implemented in different
	image retrieval solutions [<a id="x1-2001"></a><a
		href="#cite.0@DBLP:journals/corr/ZhouLT17">8</a
	>].
	<!--l. 53-->
</p>

<p class="indent">
	<a id="x1-2002"></a>Cao et al. [<a href="#cite.0@10.5555/3016387.3016390">1</a
	>] obtained image tokens using k-means clustering of the intermediate
	activation of a convolutional network. Other methods have been proposed that
	use end to end training of the token representations for image retrieval [<a
		id="x1-2003"
	></a><a href="#cite.0@liu2017e2bows">4</a>,
	<a href="#cite.0@10.5555/3016387.3016390">1</a>, <a id="x1-2004"></a><a
		href="#cite.0@DBLP:journals/corr/abs-2012-00641">2</a
	>]. Despite training our token representations without much consideration of
	their downstream usage in for image retrieval, our method still obtains
	reasonable benchmark results.
	<!--l. 57-->
</p>

<p class="noindent"></p>
<h3 class="likesectionHead"><a id="x1-3000"></a>Data Exploration</h3>
<!--l. 58-->
<p class="noindent"></p>
<h4 class="likesubsectionHead"><a id="x1-4000"></a>Data Sources</h4>
<!--l. 59-->
<p class="noindent">
	We fine tune our VQ-ResNet models using the
	<a href="https://www.image-net.org/challenges/LSVRC/2012/"
		>ImageNet LSVRC 2012 Training Set</a
	>. We then evaluate the accuracy of VQ-ResNet on the validation set, ImageNet
	LSVRC 2012 Validation Set. Both of these datasets were retrieved from Academic
	Torrents.

	<!--l. 61-->
</p>

<p class="indent">
	We evaluate image retrieval performance on the GPR1200 dataset [<a
		id="x1-4001"
	></a><a href="#cite.0@DBLP:journals/corr/abs-2111-13122">7</a>]. This
	benchmark set contains a diverse collection of general-purpose images, and
	measures the ability of a model to retrieve images in similar categories such
	as objects, places, or people. The authors provided baseline scores for the
	embeddings of various image models, and their reported retrieval mAP (mean
	average presicion) using embeddings from a ResNet101, was 42.8.
	<!--l. 63-->
</p>

<p class="indent">
	In prior reports we were considering the FlickrFaces dataset and Oxford5k
	dataset. Both were ultimately dropped: The FlickrFaces dataset did not provide
	the nececssary image labels to conduct accurate image retrieval; and the
	Oxford5k dataset didn&#8217;t encompass a wide enough image category to prove
	the robustness of our methods. This led to selecting the GPR1200 dataset.
	<!--l. 65-->
</p>

<p class="noindent"></p>
<h3 class="sectionHead">
	<span class="titlemark">1 </span> <a id="x1-50001"></a>Methodology
</h3>
<!--l. 67-->
<p class="noindent"></p>
<h4 class="likesubsectionHead"><a id="x1-6000"></a>Indexing Pipeline</h4>
<!--l. 68-->
<p class="noindent">
	Our unoptimized pipline takes about 60 seconds on a Nvidia 3070 GPU and Ryzen
	3700 CPU to fully index the GPR1200 dataset. This is with an image resolution
	of 256, batch size of 64, and for k grams,
	<span class="cmmi-10">k </span>= 3 with no padding or stride, on our RN50
	model.
	<!--l. 70-->
</p>

<p class="indent">
	In compute limited settings, such as on a 4-core laptop, the indexing time of
	GPR1200 is about 30 minutes, with most of the time being spent running image
	batches through the VQ-Resnet.
	<!--l. 72-->
</p>

<p class="indent">
	Our indexing pipeline is invariant to input image resolution. Our pipeline
	indexes any image of any input resolution, (as long as the embedding shape is
	larger than the k-grams kernel size). The result of the indexing pipeline is
	that all images are stored in memory as a BoW vector.
</p>

<ul class="itemize1">
	<li class="itemize">
		<!--l. 75-->
		<p class="noindent">
			Batching images: We sort images by aspect ratio, and then group images
			into batches. The same aspect ratio is maintained across all images in a
			batch. Different batches of images have different spatial resolutions.
			Images in the same batch were slightly scaled to have the same resolution.
			The max of height/width of the image batches can be modified as a
			hyperparameter. This batching step parallelizes following pipeline
			operations.
		</p>
	</li>

	<li class="itemize">
		<!--l. 77-->
		<p class="noindent">
			VQ-ResNet Model: Image batches are passed through our VQ-ResNet model to
			obtain 3D arrays of tokens, or &#8221;words,&#8221; that encode meaningful
			information about the contents of the images. The shape of these
			embeddings depended on the input batch&#8217;s spatial resolution, and the
			number of codebook heads.
		</p>
	</li>

	<li class="itemize">
		<!--l. 79-->
		<p class="noindent">
			Calculating kgrams: For each embedding, a window of size
			<span class="cmmi-10">k</span><span class="cmsy-10">&#x00D7;</span><span
				class="cmmi-10">k</span
			><span class="cmsy-10">&#x00D7;</span><span class="cmmi-10">c</span>,
			where
			<span class="cmmi-10">c </span>is the number of codebook heads, is slid
			across all possible positions over the batched embeddings. The value
			counts of different tokens within this window are stored, resulting in a
			tensor of shape (k by k by k by the number of possible tokens). This
			tensor is the Bag-of-Words for each image.
		</p>
	</li>
</ul>
<!--l. 83-->
<p class="noindent"></p>
<h4 class="likesubsectionHead"><a id="x1-7000"></a>VQ-ResNet Model</h4>
<!--l. 84-->
<p class="noindent">
	ResNet is a deep learning model widely used in the computer vision task of
	image classification. The intermediate activation&#8217;s contain information
	about the types of objects that are present in an input image, which is useful
	for image retrieval, as images containing similar objects should ideally have
	similar intermediate activation&#8217;s within ResNet.
	<!--l. 86-->
</p>

<p class="indent">
	We took inspiration from learned vector quantization models [<a id="x1-7001"
	></a><a href="#cite.0@DBLP:journals/corr/abs-1711-00937">6</a>], and obtain
	our token representation of images using a learned vector quantization layer.
	We modified a pretrained ResNet50 model from
	<a href="https://pytorch.org/vision/stable/models.html">torchvision</a>,
	placing a Vector Quantization (VQ) layer after the last convolutional layer.
	We froze the first three convolutional blocks of ResNet, and train the model
	for four epochs on the ImageNet LSVRC 2012 training set. Our best performing
	VQ-Resnet reached close to original accuracy@5 (within 5%) and accuracy@1
	(within 3%) on the validation set.
	<!--l. 89-->
</p>

<p class="indent">
	It would have been interesting to compare the performance of a non-learned vs
	learned vector quantization layer. Our learned VQ layer did require training
	the weights of the original model. In particular, freezing all of the ResNet
	layers led to more inaccuracies on ImageNet, but overall, the validation
	accuracy of our best model has shown that the tokens do contain meaningful
	&#8221;words&#8221; that describe the objects classes present in images.

	<!--l. 91-->
</p>

<p class="indent">
	In all of our experiments we use a multi-headed vector quantization layer [<a
		id="x1-7002"
	></a><a href="#cite.0@mama2021nwt">5</a>]. For a single image, the resulting
	tokens from VQ-Resnet are shaped
	<span class="cmmi-10">c</span><span class="cmsy-10">&#x00D7;</span><span
		class="cmmi-10"
		>H
	</span><span class="cmsy-10">&#x00D7;</span><span class="cmmi-10">W</span>,
	where <span class="cmmi-10">c </span>is the number of VQ heads. The codebook
	IDs are integers from 0 to the number of codebooks, which we varied in our
	experiments.
	<!--l. 94-->
</p>

<p class="noindent"></p>
<h4 class="likesubsectionHead">
	<a id="x1-8000"></a>From 3D Token Embeddings to BoW: Multidimensional K Grams
</h4>
<!--l. 96-->
<p class="noindent">
	We obtain Bag-of-Words (BoW) representations of our discrete tokens by using K
	grams. This allows us to obtain a global vector representation of gram
	frequencies. We define a gram as all possible token values in a k by k window
	over the last two dimensions of an input embedding.
	<!--l. 98-->
</p>

<p class="indent">
	Our intuition is that given the fully convolutional architecture of the first
	four blocks of ResNet, the information in the intermediate activations is
	highly localized, meaning much of the spatial information in the intermediate
	activation&#8217;s of a convolutional network is kept to small sub-areas.
	Discarding long-range information in this activation should not have left out
	much detail about how the network interprets different activation&#8217;s, as
	convolutional networks usually have small kernels; at any given single layer,
	long-range information is not able to be interpreted or transmitted.
	<!--l. 100-->
</p>

<p class="indent">
	We extend the notion of K-grams into 2 dimensions; given a 3D input tensor of
	discrete integer class values, 2D K-grams returns the value counts of the
	tokens over all positions of a k by k window over the last two dimensions of
	the input tensor. Conceptually, a k by k window is slid across the input
	tensor, and then the counts of integer classes at each position in the window
	are stored. On an 8 by 8 by 8 input tensor, with integer class ids from 0 to
	63, 2D 3-grams returns an 8 by 3 by 3 by 64 tensor containing the value counts
	of each class id at each position in the 3<sup
		><span class="cmr-7">3</span></sup
	>
	window.
	<!--l. 102-->
</p>

<p class="indent">
	We also experiment with using 3D k grams. 3D K grams applies the kernel along
	the channel dimension, as well as the height and width of the embedded 3D
	tokens. This results in a much smaller total BoW size.
	<!--l. 104-->
</p>

<p class="indent">
	Using 2D k grams, the size of the BoW descriptive vector for a single image is
	equal to <span class="cmmi-10">c</span><span class="cmsy-10">&#x22C5;</span
	><span class="cmmi-10">k </span><span class="cmsy-10">&#x22C5;</span><span
		class="cmmi-10"
		>k
	</span><span class="cmsy-10">&#x22C5;</span><span class="cmmi-10">n</span>,
	where <span class="cmmi-10">c </span>is the number of channels in the 3D
	tensor, and <span class="cmmi-10">n </span><span class="cmsy-10">- </span>1 is
	the max possible integer class id.
	<!--l. 106-->
</p>

<p class="indent">
	Using 3D k grams, the size of the BoW descriptive vector for a single image is
	equal to <span class="cmmi-10">k </span><span class="cmsy-10"
		>&#x22C5;
	</span><span class="cmmi-10">k </span><span class="cmsy-10"
		>&#x22C5;
	</span><span class="cmmi-10">k </span><span class="cmsy-10"
		>&#x22C5;
	</span><span class="cmmi-10">n</span>.
	<!--l. 108-->
</p>

<p class="noindent"></p>
<h4 class="likesubsectionHead">
	<a id="x1-9000"></a>Similarity Calculation: Evaluating Image Similarities
	Using Various Techniques
</h4>
<!--l. 109-->
<p class="noindent">
	In this project, we compare the mAP scores obtained from different similarity
	scorings between image embeddings.
	<!--l. 111-->
</p>

<p class="indent">
	<span class="cmbx-10">Jaccard Similarity:</span>
	<!--l. 113-->
</p>

<p class="indent">
	The choice of using Jaccard Similarity and TF-IDF as similarity measures in
	this project is based on their widespread use and proven effectiveness in
	comparing sets and weighing the importance of tokens, respectively. The
	Jaccard similarity is calculated as the intersection of two sets divided by
	their union:
</p>

<div class="math-display">
	<img
		src="main0x.png"
		alt="         |A&#x2229; B |
J(A,B) = |A&#x222A;-B-|
"
		class="math-display"
	/>
</div>
<!--l. 114-->
<p class="nopar">
	where <span class="cmmi-10">A </span>and
	<span class="cmmi-10">B </span>represent the sets being compared.
	<!--l. 117-->
</p>

<p class="indent">
	The Jaccard similarity is intuitive and easy to compute, though it may not be
	the best choice for our tokens. We experimented with cropping all dataset
	images to the same size, and then measuring similarity based on elementwise
	equality of the tokens. This produced lackluster scores compared to our BoW
	method, probably because it doesn&#8217;t take into account the frequency of
	different terms. The importance of different tokens being equal between two
	embeddings is likely not the same for all tokens.
	<!--l. 119-->
</p>

<p class="indent">
	<span class="cmbx-10">TF-IDF:</span>
	<!--l. 122-->
</p>

<p class="indent">
	The TF-IDF technique takes into account the frequency of tokens in a document
	and their rarity across a collection of documents. The TF-IDF weight for a
	token <span class="cmmi-10">t </span>in a document
	<span class="cmmi-10">d </span>is calculated as:
</p>

<div class="math-display">
	<img
		src="main1x.png"
		alt="TF - IDF (t,d) = TF (t,d)&#x00D7; IDF (t)
"
		class="math-display"
	/>
</div>
<!--l. 123-->
<p class="nopar">
	where TF(<span class="cmmi-10">t,d</span>) is the term frequency of token
	<span class="cmmi-10">t </span>in document
	<span class="cmmi-10">d</span>, and IDF(<span class="cmmi-10">t</span>) = log
	<img src="main2x.png" alt="DFN(t)" class="frac" align="middle" /> is the
	inverse document frequency of token <span class="cmmi-10">t</span>, with
	<span class="cmmi-10">N </span>being the total number of documents and DF(<span
		class="cmmi-10">t</span
	>) representing the number of documents containing the token
	<span class="cmmi-10">t</span>.
	<!--l. 126-->
</p>

<p class="indent">
	<span class="cmbx-10">Similarity Measure Comparison and Selection:</span>
	<!--l. 128-->
</p>

<p class="indent">
	We compare the performance of Jaccard Similarity and TF-IDF using the Mean
	Average Precision (mAP) evaluation metric. Our experiments indicate that the
	TF-IDF applied on our obtained BoW, outperforms Jaccard Similarity applied on
	the VQ-Resnet tokens.

	<!--l. 131-->
</p>

<p class="noindent"></p>
<h3 class="likesectionHead">
	<a id="x1-10000"></a>Experiments and Results
</h3>
<!--l. 133-->
<p class="noindent"></p>
<h4 class="likesubsectionHead">
	<a id="x1-11000"></a>Mean Average Precision: Calculation and Interpretation
</h4>
<!--l. 135-->
<p class="noindent">
	The Mean Average Precision (mAP) metric is calculated by first determining the
	Average Precision (AP) for each query in the dataset. AP is computed as the
	average of the precision values obtained at each relevant item in the ranked
	retrieval results. We analyze the mAP values obtained for various parameter
	configurations and similarity measures on GPR1200.
	<!--l. 138-->
</p>

<p class="noindent"></p>
<h4 class="likesubsectionHead"><a id="x1-12000"></a>Experiment Setup</h4>
<!--l. 139-->
<p class="noindent">
	mAP scores are measure on the GPR1200 dataset.
	<!--l. 141-->
</p>
<p class="indent">
	<span class="cmbx-10">Explanation of model parameters:</span>
</p>

<ul class="itemize1">
	<li class="itemize">
		<!--l. 144-->
		<p class="noindent">
			Heads: Number of codebook attention heads [<a id="x1-12001"></a><a
				href="#cite.0@mama2021nwt">5</a
			>]. All of our experiments used a shared codebook.
		</p>
	</li>

	<li class="itemize">
		<!--l. 145-->
		<p class="noindent">
			Codebook Dim: The z-dimension of the codebook. A linear layer projects
			each input vector of size Dim, to the codebook space.
		</p>
	</li>

	<li class="itemize">
		<!--l. 146-->
		<p class="noindent">
			Codebook Size: The number <span class="cmmi-10">k</span>, or the number of
			codebook vectors.
		</p>
	</li>

	<li class="itemize">
		<!--l. 147-->
		<p class="noindent">
			Commitment Weight: See equation 3 from <a id="x1-12002"></a>Oord, Vinyals,
			and Kavukcuoglu [<a href="#cite.0@DBLP:journals/corr/abs-1711-00937">6</a
			>]
		</p>
	</li>

	<li class="itemize">
		<!--l. 148-->
		<p class="noindent">
			Threshold EMA Dead Code: Randomly reinitializes codes that are not being
			selected by the network.
		</p>
	</li>

	<li class="itemize">
		<!--l. 149-->
		<p class="noindent">
			Dim: The input dimension, which is from the preceding ResNet Conv layer.
		</p>
	</li>
</ul>
<!--l. 152-->
<p class="indent">
	<span class="cmbx-10">Explanation of k-grams parameters:</span>
</p>

<ul class="itemize1">
	<li class="itemize">
		<!--l. 155-->
		<p class="noindent">
			SimilarityMeasurement: Labels what methods was used to measure similarity,
			Jaccard or TF-IDF.
		</p>
	</li>

	<li class="itemize">
		<!--l. 156-->
		<p class="noindent">
			Model: Indicates which model was used from the model table prior to the
			similarity measurements.
		</p>
	</li>

	<li class="itemize">
		<!--l. 157-->
		<p class="noindent">
			Resolution: Indicates the size of the image space across various aspect
			ratios.
		</p>
	</li>

	<li class="itemize">
		<!--l. 158-->
		<p class="noindent">
			Kernel: Dimensions of the k-grams calculation (NAN means Jaccard
			similarity was used)
		</p>
	</li>

	<li class="itemize">
		<!--l. 159-->
		<p class="noindent">
			Padding: Dimensions of the k-grams calculation (NAN means Jaccard
			similarity was used)
		</p>
	</li>

	<li class="itemize">
		<!--l. 160-->
		<p class="noindent">
			should_channel_kgrams: Whether k grams is applied to the channel
			dimension. If this is true, then the k-grams window is applied to the
			channel dimension, or the &#8216;codebook head&#8216; dimension.
		</p>
	</li>

	<li class="itemize">
		<!--l. 161-->
		<p class="noindent">
			GPR1200 mAP (shown as mAP): mean average precision across all categories
		</p>
	</li>

	<li class="itemize">
		<!--l. 162-->
		<p class="noindent">
			Landmarks, IMSketch, iNat, Instre, SOP, Faces: categories of images found
			within the GPR1200 dataset. Each value is the mAP value for the relevant
			category.
		</p>
	</li>
</ul>
<!--l. 166-->
<p class="noindent"></p>
<h4 class="likesubsectionHead"><a id="x1-13000"></a>Results</h4>
<!--l. 167-->
<p class="noindent">
	We evaluate our method&#8217;s mAP score on the GPR1200 benchmark task,
	testing the effect of various k-grams parameters, as well as VQ-ResNet
	parameters.
</p>

<div class="table">
	<!--l. 170-->
	<p class="indent"><a id="x1-130011"></a></p>
	<hr class="float" />
	<div class="float">
		<div class="caption">
			<span class="id">Table&#x00A0;1: </span><span class="content">Models</span
			>
		</div>
		<!--tex4ht:label?: x1-130011 -->
		<!--tex4ht:inline-->
		<div class="tabular">
			<table id="TBL-2" class="tabular">
				<colgroup id="TBL-2-1g">
					<col id="TBL-2-1" />
				</colgroup>
				<colgroup id="TBL-2-2g">
					<col id="TBL-2-2" />
				</colgroup>
				<colgroup id="TBL-2-3g">
					<col id="TBL-2-3" />
				</colgroup>
				<colgroup id="TBL-2-4g">
					<col id="TBL-2-4" />
				</colgroup>
				<colgroup id="TBL-2-5g">
					<col id="TBL-2-5" />
				</colgroup>
				<colgroup id="TBL-2-6g">
					<col id="TBL-2-6" />
				</colgroup>
				<colgroup id="TBL-2-7g">
					<col id="TBL-2-7" />
				</colgroup>
				<colgroup id="TBL-2-8g">
					<col id="TBL-2-8" />
				</colgroup>
				<colgroup id="TBL-2-9g">
					<col id="TBL-2-9" />
				</colgroup>
				<tbody>
				<tr class="hline" style="border-top: 1px solid #000">
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-2-1-">
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-1-1"
						class="td11"
					>
						Model Name
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-1-2"
						class="td11"
					>
						Heads
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-1-3"
						class="td11"
					>
						Codebook Dim
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-1-4"
						class="td11"
					>
						Codebook Size
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-1-5"
						class="td11"
					>
						Commitment Weight
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-1-6"
						class="td11"
					>
						Threshold EMA Dead Code
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-1-7"
						class="td11"
					>
						Dim
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-1-8"
						class="td11"
					>
						ResNet Type
					</td>
				</tr>
				<tr class="hline" style="border-top: 1px solid #000">
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-2-2-">
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-2-1"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-2-2"
						class="td11"
					>
						8
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-2-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-2-4"
						class="td11"
					>
						128
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-2-5"
						class="td11"
					>
						0.0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-2-6"
						class="td11"
					>
						2
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-2-7"
						class="td11"
					>
						2048
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-2-8"
						class="td11"
					>
						50
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-2-3-">
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-3-1"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-3-2"
						class="td11"
					>
						8
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-3-3"
						class="td11"
					>
						128
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-3-4"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-3-5"
						class="td11"
					>
						5.0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-3-6"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-3-7"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-3-8"
						class="td11"
					>
						34
					</td>
				</tr>
				<tr class="hline" style="border-top: 1px solid #000">
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
					<td><hr /></td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-2-4-">
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-2-4-1"
						class="td11"
					></td>
				</tr>
				</tbody>
			</table>
		</div>
	</div>
	<hr class="endfloat" />
</div>
<div class="table">
	<!--l. 189-->
	<p class="indent"><a id="x1-130022"></a></p>
	<hr class="float" />
	<div class="float">
		<div class="caption">
			<span class="id">Table&#x00A0;2: </span><span class="content"
				>SimilarityResults</span
			>
		</div>
		<!--tex4ht:label?: x1-130022 -->
		<!--tex4ht:inline-->
		<div class="tabular">
			<table id="TBL-3" class="tabular">
				<colgroup id="TBL-3-1g">
					<col id="TBL-3-1" />
					<col id="TBL-3-2" />
					<col id="TBL-3-3" />
					<col id="TBL-3-4" />
					<col id="TBL-3-5" />
					<col id="TBL-3-6" />
					<col id="TBL-3-7" />
					<col id="TBL-3-8" />
					<col id="TBL-3-9" />
					<col id="TBL-3-10" />
					<col id="TBL-3-11" />
					<col id="TBL-3-12" />
					<col id="TBL-3-13" />
					<col id="TBL-3-14" />
				</colgroup>
				<tbody>
				<tr style="vertical-align: baseline" id="TBL-3-1-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-1-1"
						class="td11"
					>
						SimilarityMeasurement
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-1-2"
						class="td11"
					>
						Model
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-1-3"
						class="td11"
					>
						Resolution
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-1-4"
						class="td11"
					>
						Kernel
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-1-5"
						class="td11"
					>
						Padding
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-1-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-1-7"
						class="td01"
					>
						should_channel_kgrams
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-1-8"
						class="td11"
					>
						mAP
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-1-9"
						class="td11"
					>
						Landmarks
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-1-10"
						class="td11"
					>
						IMSketch
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-1-11"
						class="td11"
					>
						iNat
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-1-12"
						class="td11"
					>
						Instre
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-1-13"
						class="td11"
					>
						SOP
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-1-14"
						class="td11"
					>
						Faces
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-2-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-2-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-2-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-2-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-2-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-2-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-2-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-2-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-2-8"
						class="td11"
					>
						0.358996
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-2-9"
						class="td11"
					>
						0.58213
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-2-10"
						class="td11"
					>
						0.27893
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-2-11"
						class="td11"
					>
						0.24168
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-2-12"
						class="td11"
					>
						0.28393
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-2-13"
						class="td11"
					>
						0.60545
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-2-14"
						class="td11"
					>
						0.16185
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-3-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-3-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-3-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-3-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-3-4"
						class="td11"
					>
						5
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-3-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-3-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-3-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-3-8"
						class="td11"
					>
						0.358116
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-3-9"
						class="td11"
					>
						0.58626
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-3-10"
						class="td11"
					>
						0.27578
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-3-11"
						class="td11"
					>
						0.23728
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-3-12"
						class="td11"
					>
						0.27722
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-3-13"
						class="td11"
					>
						0.61037
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-3-14"
						class="td11"
					>
						0.16180
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-4-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-4-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-4-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-4-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-4-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-4-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-4-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-4-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-4-8"
						class="td11"
					>
						0.356071
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-4-9"
						class="td11"
					>
						0.58348
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-4-10"
						class="td11"
					>
						0.27313
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-4-11"
						class="td11"
					>
						0.23403
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-4-12"
						class="td11"
					>
						0.27203
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-4-13"
						class="td11"
					>
						0.61197
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-4-14"
						class="td11"
					>
						0.16178
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-5-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-5-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-5-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-5-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-5-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-5-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-5-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-5-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-5-8"
						class="td11"
					>
						0.347656
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-5-9"
						class="td11"
					>
						0.54185
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-5-10"
						class="td11"
					>
						0.26551
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-5-11"
						class="td11"
					>
						0.21882
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-5-12"
						class="td11"
					>
						0.27872
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-5-13"
						class="td11"
					>
						0.62476
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-5-14"
						class="td11"
					>
						0.15628
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-6-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-6-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-6-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-6-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-6-4"
						class="td11"
					>
						5
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-6-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-6-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-6-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-6-8"
						class="td11"
					>
						0.347565
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-6-9"
						class="td11"
					>
						0.54985
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-6-10"
						class="td11"
					>
						0.26217
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-6-11"
						class="td11"
					>
						0.21274
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-6-12"
						class="td11"
					>
						0.27008
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-6-13"
						class="td11"
					>
						0.63362
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-6-14"
						class="td11"
					>
						0.15695
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-7-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-7-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-7-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-7-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-7-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-7-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-7-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-7-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-7-8"
						class="td11"
					>
						0.347015
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-7-9"
						class="td11"
					>
						0.58669
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-7-10"
						class="td11"
					>
						0.25582
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-7-11"
						class="td11"
					>
						0.21834
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-7-12"
						class="td11"
					>
						0.26909
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-7-13"
						class="td11"
					>
						0.59012
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-7-14"
						class="td11"
					>
						0.16203
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-8-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-8-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-8-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-8-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-8-4"
						class="td11"
					>
						5
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-8-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-8-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-8-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-8-8"
						class="td11"
					>
						0.345833
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-8-9"
						class="td11"
					>
						0.58998
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-8-10"
						class="td11"
					>
						0.25283
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-8-11"
						class="td11"
					>
						0.21417
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-8-12"
						class="td11"
					>
						0.26241
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-8-13"
						class="td11"
					>
						0.59373
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-8-14"
						class="td11"
					>
						0.16187
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-9-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-9-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-9-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-9-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-9-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-9-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-9-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-9-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-9-8"
						class="td11"
					>
						0.344792
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-9-9"
						class="td11"
					>
						0.54615
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-9-10"
						class="td11"
					>
						0.25851
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-9-11"
						class="td11"
					>
						0.21024
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-9-12"
						class="td11"
					>
						0.26400
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-9-13"
						class="td11"
					>
						0.63323
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-9-14"
						class="td11"
					>
						0.15663
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-10-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-10-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-10-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-10-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-10-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-10-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-10-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-10-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-10-8"
						class="td11"
					>
						0.343568
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-10-9"
						class="td11"
					>
						0.58732
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-10-10"
						class="td11"
					>
						0.24950
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-10-11"
						class="td11"
					>
						0.21074
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-10-12"
						class="td11"
					>
						0.25619
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-10-13"
						class="td11"
					>
						0.59582
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-10-14"
						class="td11"
					>
						0.16184
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-11-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-11-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-11-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-11-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-11-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-11-5"
						class="td11"
					>
						2
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-11-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-11-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-11-8"
						class="td11"
					>
						0.339198
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-11-9"
						class="td11"
					>
						0.47780
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-11-10"
						class="td11"
					>
						0.35360
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-11-11"
						class="td11"
					>
						0.24893
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-11-12"
						class="td11"
					>
						0.24240
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-11-13"
						class="td11"
					>
						0.56072
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-11-14"
						class="td11"
					>
						0.15174
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-12-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-12-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-12-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-12-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-12-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-12-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-12-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-12-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-12-8"
						class="td11"
					>
						0.339197
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-12-9"
						class="td11"
					>
						0.47687
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-12-10"
						class="td11"
					>
						0.35379
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-12-11"
						class="td11"
					>
						0.24947
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-12-12"
						class="td11"
					>
						0.24325
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-12-13"
						class="td11"
					>
						0.55978
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-12-14"
						class="td11"
					>
						0.15204
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-13-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-13-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-13-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-13-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-13-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-13-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-13-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-13-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-13-8"
						class="td11"
					>
						0.338142
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-13-9"
						class="td11"
					>
						0.55188
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-13-10"
						class="td11"
					>
						0.24478
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-13-11"
						class="td11"
					>
						0.19829
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-13-12"
						class="td11"
					>
						0.26012
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-13-13"
						class="td11"
					>
						0.61759
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-13-14"
						class="td11"
					>
						0.15620
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-14-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-14-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-14-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-14-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-14-4"
						class="td11"
					>
						5
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-14-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-14-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-14-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-14-8"
						class="td11"
					>
						0.337825
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-14-9"
						class="td11"
					>
						0.55749
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-14-10"
						class="td11"
					>
						0.24103
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-14-11"
						class="td11"
					>
						0.19377
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-14-12"
						class="td11"
					>
						0.25305
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-14-13"
						class="td11"
					>
						0.62484
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-14-14"
						class="td11"
					>
						0.15677
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-15-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-15-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-15-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-15-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-15-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-15-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-15-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-15-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-15-8"
						class="td11"
					>
						0.335068
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-15-9"
						class="td11"
					>
						0.55416
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-15-10"
						class="td11"
					>
						0.23722
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-15-11"
						class="td11"
					>
						0.19122
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-15-12"
						class="td11"
					>
						0.24681
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-15-13"
						class="td11"
					>
						0.62444
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-15-14"
						class="td11"
					>
						0.15656
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-16-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-16-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-16-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-16-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-16-4"
						class="td11"
					>
						5
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-16-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-16-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-16-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-16-8"
						class="td11"
					>
						0.330183
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-16-9"
						class="td11"
					>
						0.52225
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-16-10"
						class="td11"
					>
						0.24846
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-16-11"
						class="td11"
					>
						0.21264
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-16-12"
						class="td11"
					>
						0.25831
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-16-13"
						class="td11"
					>
						0.59082
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-16-14"
						class="td11"
					>
						0.14862
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-17-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-17-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-17-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-17-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-17-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-17-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-17-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-17-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-17-8"
						class="td11"
					>
						0.326344
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-17-9"
						class="td11"
					>
						0.50732
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-17-10"
						class="td11"
					>
						0.24662
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-17-11"
						class="td11"
					>
						0.21861
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-17-12"
						class="td11"
					>
						0.26113
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-17-13"
						class="td11"
					>
						0.57731
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-17-14"
						class="td11"
					>
						0.14708
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-18-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-18-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-18-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-18-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-18-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-18-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-18-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-18-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-18-8"
						class="td11"
					>
						0.323291
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-18-9"
						class="td11"
					>
						0.51431
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-18-10"
						class="td11"
					>
						0.24105
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-18-11"
						class="td11"
					>
						0.20902
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-18-12"
						class="td11"
					>
						0.24815
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-18-13"
						class="td11"
					>
						0.58027
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-18-14"
						class="td11"
					>
						0.14695
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-19-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-19-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-19-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-19-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-19-4"
						class="td11"
					>
						5
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-19-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-19-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-19-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-19-8"
						class="td11"
					>
						0.317817
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-19-9"
						class="td11"
					>
						0.53150
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-19-10"
						class="td11"
					>
						0.22737
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-19-11"
						class="td11"
					>
						0.19314
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-19-12"
						class="td11"
					>
						0.23635
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-19-13"
						class="td11"
					>
						0.57025
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-19-14"
						class="td11"
					>
						0.14828
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-20-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-20-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-20-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-20-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-20-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-20-5"
						class="td11"
					>
						2
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-20-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-20-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-20-8"
						class="td11"
					>
						0.315947
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-20-9"
						class="td11"
					>
						0.42983
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-20-10"
						class="td11"
					>
						0.29231
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-20-11"
						class="td11"
					>
						0.21948
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-20-12"
						class="td11"
					>
						0.24301
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-20-13"
						class="td11"
					>
						0.56028
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-20-14"
						class="td11"
					>
						0.15077
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-21-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-21-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-21-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-21-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-21-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-21-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-21-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-21-7"
						class="td01"
					>
						False
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-21-8"
						class="td11"
					>
						0.315619
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-21-9"
						class="td11"
					>
						0.42757
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-21-10"
						class="td11"
					>
						0.29264
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-21-11"
						class="td11"
					>
						0.22086
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-21-12"
						class="td11"
					>
						0.24486
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-21-13"
						class="td11"
					>
						0.55697
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-21-14"
						class="td11"
					>
						0.15081
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-22-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-22-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-22-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-22-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-22-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-22-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-22-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-22-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-22-8"
						class="td11"
					>
						0.314760
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-22-9"
						class="td11"
					>
						0.51752
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-22-10"
						class="td11"
					>
						0.22639
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-22-11"
						class="td11"
					>
						0.19760
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-22-12"
						class="td11"
					>
						0.23944
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-22-13"
						class="td11"
					>
						0.56070
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-22-14"
						class="td11"
					>
						0.14691
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-23-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-23-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-23-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-23-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-23-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-23-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-23-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-23-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-23-8"
						class="td11"
					>
						0.310504
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-23-9"
						class="td11"
					>
						0.52159
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-23-10"
						class="td11"
					>
						0.22073
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-23-11"
						class="td11"
					>
						0.18945
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-23-12"
						class="td11"
					>
						0.22604
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-23-13"
						class="td11"
					>
						0.55865
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-23-14"
						class="td11"
					>
						0.14656
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-24-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-24-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-24-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-24-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-24-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-24-5"
						class="td11"
					>
						2
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-24-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-24-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-24-8"
						class="td11"
					>
						0.304981
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-24-9"
						class="td11"
					>
						0.41032
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-24-10"
						class="td11"
					>
						0.28407
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-24-11"
						class="td11"
					>
						0.21846
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-24-12"
						class="td11"
					>
						0.23638
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-24-13"
						class="td11"
					>
						0.53534
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-24-14"
						class="td11"
					>
						0.14531
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-25-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-25-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-25-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-25-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-25-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-25-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-25-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-25-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-25-8"
						class="td11"
					>
						0.304241
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-25-9"
						class="td11"
					>
						0.40883
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-25-10"
						class="td11"
					>
						0.28296
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-25-11"
						class="td11"
					>
						0.21938
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-25-12"
						class="td11"
					>
						0.23723
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-25-13"
						class="td11"
					>
						0.53142
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-25-14"
						class="td11"
					>
						0.14563
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-26-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-26-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-26-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-26-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-26-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-26-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-26-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-26-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-26-8"
						class="td11"
					>
						0.296027
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-26-9"
						class="td11"
					>
						0.49294
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-26-10"
						class="td11"
					>
						0.21070
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-26-11"
						class="td11"
					>
						0.22187
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-26-12"
						class="td11"
					>
						0.21419
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-26-13"
						class="td11"
					>
						0.48399
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-26-14"
						class="td11"
					>
						0.15247
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-27-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-27-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-27-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-27-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-27-4"
						class="td11"
					>
						5
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-27-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-27-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-27-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-27-8"
						class="td11"
					>
						0.295039
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-27-9"
						class="td11"
					>
						0.49761
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-27-10"
						class="td11"
					>
						0.20775
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-27-11"
						class="td11"
					>
						0.21758
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-27-12"
						class="td11"
					>
						0.20830
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-27-13"
						class="td11"
					>
						0.48800
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-27-14"
						class="td11"
					>
						0.15100
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-28-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-28-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-28-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-28-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-28-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-28-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-28-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-28-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-28-8"
						class="td11"
					>
						0.286522
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-28-9"
						class="td11"
					>
						0.48304
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-28-10"
						class="td11"
					>
						0.19891
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-28-11"
						class="td11"
					>
						0.21354
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-28-12"
						class="td11"
					>
						0.19940
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-28-13"
						class="td11"
					>
						0.47538
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-28-14"
						class="td11"
					>
						0.14887
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-29-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-29-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-29-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-29-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-29-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-29-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-29-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-29-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-29-8"
						class="td11"
					>
						0.286102
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-29-9"
						class="td11"
					>
						0.49331
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-29-10"
						class="td11"
					>
						0.19496
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-29-11"
						class="td11"
					>
						0.20189
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-29-12"
						class="td11"
					>
						0.20796
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-29-13"
						class="td11"
					>
						0.46663
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-29-14"
						class="td11"
					>
						0.15187
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-30-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-30-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-30-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-30-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-30-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-30-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-30-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-30-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-30-8"
						class="td11"
					>
						0.285037
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-30-9"
						class="td11"
					>
						0.41862
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-30-10"
						class="td11"
					>
						0.26744
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-30-11"
						class="td11"
					>
						0.22793
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-30-12"
						class="td11"
					>
						0.18756
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-30-13"
						class="td11"
					>
						0.46522
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-30-14"
						class="td11"
					>
						0.14345
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-31-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-31-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-31-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-31-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-31-4"
						class="td11"
					>
						5
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-31-5"
						class="td11"
					>
						1
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-31-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-31-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-31-8"
						class="td11"
					>
						0.285007
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-31-9"
						class="td11"
					>
						0.49790
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-31-10"
						class="td11"
					>
						0.19262
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-31-11"
						class="td11"
					>
						0.19820
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-31-12"
						class="td11"
					>
						0.20234
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-31-13"
						class="td11"
					>
						0.46900
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-31-14"
						class="td11"
					>
						0.14998
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-32-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-32-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-32-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-32-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-32-4"
						class="td11"
					>
						4
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-32-5"
						class="td11"
					>
						2
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-32-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-32-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-32-8"
						class="td11"
					>
						0.283205
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-32-9"
						class="td11"
					>
						0.41683
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-32-10"
						class="td11"
					>
						0.26473
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-32-11"
						class="td11"
					>
						0.22735
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-32-12"
						class="td11"
					>
						0.18539
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-32-13"
						class="td11"
					>
						0.46198
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-32-14"
						class="td11"
					>
						0.14294
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-33-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-33-1"
						class="td11"
					>
						KGramTF-IDF
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-33-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-33-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-33-4"
						class="td11"
					>
						3
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-33-5"
						class="td11"
					>
						0
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-33-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-33-7"
						class="td01"
					>
						True
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-33-8"
						class="td11"
					>
						0.276272
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-33-9"
						class="td11"
					>
						0.48147
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-33-10"
						class="td11"
					>
						0.18505
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-33-11"
						class="td11"
					>
						0.19359
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-33-12"
						class="td11"
					>
						0.19382
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-33-13"
						class="td11"
					>
						0.45562
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-33-14"
						class="td11"
					>
						0.14808
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-34-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-34-1"
						class="td11"
					>
						CropAndJaccard
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-34-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-34-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-34-4"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-34-5"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-34-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-34-7"
						class="td01"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-34-8"
						class="td11"
					>
						0.261098
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-34-9"
						class="td11"
					>
						0.38146
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-34-10"
						class="td11"
					>
						0.18768
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-34-11"
						class="td11"
					>
						0.18802
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-34-12"
						class="td11"
					>
						0.23603
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-34-13"
						class="td11"
					>
						0.42209
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-34-14"
						class="td11"
					>
						0.15130
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-35-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-35-1"
						class="td11"
					>
						CropAndJaccard
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-35-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-35-3"
						class="td11"
					>
						614
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-35-4"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-35-5"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-35-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-35-7"
						class="td01"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-35-8"
						class="td11"
					>
						0.216076
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-35-9"
						class="td11"
					>
						0.31513
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-35-10"
						class="td11"
					>
						0.15473
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-35-11"
						class="td11"
					>
						0.18724
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-35-12"
						class="td11"
					>
						0.18264
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-35-13"
						class="td11"
					>
						0.30936
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-35-14"
						class="td11"
					>
						0.14735
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-36-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-36-1"
						class="td11"
					>
						CropAndJaccard
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-36-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-36-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-36-4"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-36-5"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-36-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-36-7"
						class="td01"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-36-8"
						class="td11"
					>
						0.209238
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-36-9"
						class="td11"
					>
						0.32666
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-36-10"
						class="td11"
					>
						0.13962
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-36-11"
						class="td11"
					>
						0.14898
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-36-12"
						class="td11"
					>
						0.15798
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-36-13"
						class="td11"
					>
						0.33274
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-36-14"
						class="td11"
					>
						0.14945
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-37-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-37-1"
						class="td11"
					>
						CropAndJaccard
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-37-2"
						class="td11"
					>
						VQ-FrozenResNet34
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-37-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-37-4"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-37-5"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-37-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-37-7"
						class="td01"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-37-8"
						class="td11"
					>
						0.196921
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-37-9"
						class="td11"
					>
						0.30610
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-37-10"
						class="td11"
					>
						0.13622
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-37-11"
						class="td11"
					>
						0.14287
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-37-12"
						class="td11"
					>
						0.14431
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-37-13"
						class="td11"
					>
						0.30527
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-37-14"
						class="td11"
					>
						0.14675
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-38-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-38-1"
						class="td11"
					>
						CropAndJaccard
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-38-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-38-3"
						class="td11"
					>
						512
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-38-4"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-38-5"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-38-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-38-7"
						class="td01"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-38-8"
						class="td11"
					>
						0.165012
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-38-9"
						class="td11"
					>
						0.21407
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-38-10"
						class="td11"
					>
						0.12580
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-38-11"
						class="td11"
					>
						0.14234
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-38-12"
						class="td11"
					>
						0.12543
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-38-13"
						class="td11"
					>
						0.23798
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-38-14"
						class="td11"
					>
						0.14446
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-39-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-39-1"
						class="td11"
					>
						CropAndJaccard
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-39-2"
						class="td11"
					>
						VQ-ResNet50
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-39-3"
						class="td11"
					>
						256
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-39-4"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-39-5"
						class="td11"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: center"
						id="TBL-3-39-6"
						class="td10"
					></td>
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-39-7"
						class="td01"
					>
						NAN
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-39-8"
						class="td11"
					>
						0.154992
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-39-9"
						class="td11"
					>
						0.19104
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-39-10"
						class="td11"
					>
						0.12308
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-39-11"
						class="td11"
					>
						0.13314
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-39-12"
						class="td11"
					>
						0.11553
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-39-13"
						class="td11"
					>
						0.22524
					</td>
					<td
						style="white-space: nowrap; text-align: right"
						id="TBL-3-39-14"
						class="td11"
					>
						0.14192
					</td>
				</tr>
				<tr style="vertical-align: baseline" id="TBL-3-40-">
					<td
						style="white-space: nowrap; text-align: left"
						id="TBL-3-40-1"
						class="td11"
					></td>
				</tr>
				</tbody>
			</table>
		</div>
	</div>
	<hr class="endfloat" />
</div>
<h4 class="likesubsectionHead"><a id="x1-14000"></a>Discussion</h4>
<!--l. 252-->
<p class="noindent">
	Our image retrieval system retrieves relevant images across the GPR1200
	dataset, with reasonable mAP values for different kernel sizes and similarity
	measures. 3D-Kgrams usually performed much worse than 2d-kgrams. Raising the
	input image resolution to 614 did not increase mAP scores. Increasing the
	kernel size past 4, did not improve scores on our best VQ-ResNet50 model. This
	method still needs more refinement, but overall we recommend finetuning
	larger, more accurate VQ-ResNet models for obtaining the tokens, and then
	using 2D k-grams, with a kernel size of 3.
	<!--l. 254-->
</p>

<p class="indent">
	The choice of similarity measure and kernel size significantly impact the
	retrieval performance; in general, the BoW and TF-IDF approach provides better
	results than the Jaccard similarity measure. This shows that multi dimensional
	K-Grams extracted meaningful information from discretized tokens from our
	VQ-Resnet.
	<!--l. 256-->
</p>

<p class="indent">
	The authors of GPR 1200 obtained a mAP score of 42.8 % using kNN from the
	output of the last activation of a ResNet101 V2 [<a id="x1-14001"></a><a
		href="#cite.0@DBLP:journals/corr/abs-2111-13122">7</a
	>]. They obtain higher scores using models with BiT pretraining [<a
		id="x1-14002"
	></a><a href="#cite.0@kolesnikov2020big">3</a>]. Our score from our best
	configuration is 6 % worse. With pretraining of our VQ-ResNet on ImageNet 21k,
	we may be able to improve our scores somewhat. Generally, this reduction in
	score may be acceptable; our indexed features can be inputted into traditional
	text search engines.
	<!--l. 258-->
</p>

<p class="noindent"></p>
<h3 class="likesectionHead"><a id="x1-15000"></a>Conclusion</h3>
<!--l. 259-->
<p class="noindent">
	In this data mining project, we aimed to develop an efficient and accurate
	image retrieval system for large-scale datasets by combining deep learning
	models with traditional data mining techniques. We chose the diverse GPR1200
	dataset and employed a VQ-ResNet model along with Kgrams for image
	preprocessing. Our approach integrated image preprocessing, Bag-of-Words
	representations, and various similarity calculation methods.
	<!--l. 261-->
</p>

<p class="indent">
	We compared different methods, such as Jaccard Similarity and TF-IDF, and
	assessed the performance using the Mean Average Precision (mAP) metric. Our
	results revealed that the VQ-ResNet model, combined with preprocessing,
	Bag-of-Words representation, and TF-IDF similarity measure, delivered a robust
	image retrieval system.
	<!--l. 263-->
</p>

<p class="indent"></p>
<hr class="figure" />
<div class="figure">
	<a id="x1-150011"></a>

	<!--l. 265-->
	<p class="noindent">
		<img src="query69.png" alt="PIC" width="100%" height="100%" /> <br />
	</p>

	<div class="caption">
		<span class="id">Figure&#x00A0;1: </span><span class="content"
			>Example of returned query by VQ-ResNet50, with k-grams kernel size of 3.
			Top 20 closest BoW vectors, as measured from TFIDF similarity. The query
			index from the GPR dataset was item 69.</span
		>
	</div>
	<!--tex4ht:label?: x1-150011 -->

	<!--l. 268-->
	<p class="indent"></p>
</div>

<hr class="endfigure" />
<!--l. 271-->
<p class="indent"></p>

<h3 class="sectionHead"><a id="x1-16000"></a>References</h3>
<!--l. 273-->
<p class="noindent"></p>
<dl class="thebibliography">
	<dt id="X0-10.5555/3016387.3016390" class="thebibliography">[1]</dt>
	<dd id="bib-1" class="thebibliography">
		<!--l. 273-->
		<p class="noindent">
			<a id="cite.0@10.5555/3016387.3016390"></a>Yue Cao et al. &#8220;Deep
			Quantization Network for Efficient Image Retrieval&#8221;. In:
			<span class="cmti-10">Proceedings of the</span>
			<span class="cmti-10"
				>Thirtieth AAAI Conference on Artificial Intelligence</span
			>. AAAI&#8217;16. Phoenix, Arizona: AAAI Press, 2016,
			pp.&#x00A0;3457&#8211;3463.
		</p>
	</dd>

	<dt id="X0-DBLP:journals/corr/abs-2012-00641" class="thebibliography">[2]</dt>
	<dd id="bib-2" class="thebibliography">
		<!--l. 273-->
		<p class="noindent">
			<a id="cite.0@DBLP:journals/corr/abs-2012-00641"></a>Shiv Ram Dubey.
			&#8220;A Decade Survey of Content Based Image Retrieval using Deep
			Learning&#8221;. In:
			<span class="cmti-10">CoRR</span> abs/2012.00641 (2020). arXiv:
			<a href="https://arxiv.org/abs/2012.00641"
				><span class="cmtt-10">2012.00641</span></a
			>.
			<span class="cmcsc-10"
				><span class="small-caps">u</span><span class="small-caps">r</span><span
					class="small-caps">l</span
				></span
			>:
			<a href="https://arxiv.org/abs/2012.00641" class="url"
				><span class="cmtt-10">https://arxiv.org/abs/2012.00641</span></a
			>.
		</p>
	</dd>

	<dt id="X0-kolesnikov2020big" class="thebibliography">[3]</dt>
	<dd id="bib-3" class="thebibliography">
		<!--l. 273-->
		<p class="noindent">
			<a id="cite.0@kolesnikov2020big"></a>Alexander Kolesnikov et al.
			<span class="cmti-10"
				>Big Transfer (BiT): General Visual Representation Learning</span
			>. 2020. arXiv:
			<a href="https://arxiv.org/abs/1912.11370"
				><span class="cmtt-10">1912.11370 [cs.CV]</span></a
			>.
		</p>
	</dd>

	<dt id="X0-liu2017e2bows" class="thebibliography">[4]</dt>
	<dd id="bib-4" class="thebibliography">
		<!--l. 273-->
		<p class="noindent">
			<a id="cite.0@liu2017e2bows"></a>Xiaobin Liu et al.
			<span class="cmti-10">E</span><span class="cmr-7">2</span><span
				class="cmti-10"
				>BoWs: An End-to-End Bag-of-Words Model via Deep Convolutional Neural</span
			>
			<span class="cmti-10">Network</span>. 2017. arXiv:
			<a href="https://arxiv.org/abs/1709.05903"
				><span class="cmtt-10">1709.05903 [cs.CV]</span></a
			>.
		</p>
	</dd>

	<dt id="X0-mama2021nwt" class="thebibliography">[5]</dt>
	<dd id="bib-5" class="thebibliography">
		<!--l. 273-->
		<p class="noindent">
			<a id="cite.0@mama2021nwt"></a>Rayhane Mama et al.
			<span class="cmti-10"
				>NWT: Towards natural audio-to-video generation with representation
				learning</span
			>. 2021. arXiv:
			<a href="https://arxiv.org/abs/2106.04283"
				><span class="cmtt-10">2106.04283 [cs.SD]</span></a
			>.
		</p>
	</dd>

	<dt id="X0-DBLP:journals/corr/abs-1711-00937" class="thebibliography">[6]</dt>
	<dd id="bib-6" class="thebibliography">
		<!--l. 273-->
		<p class="noindent">
			<a id="cite.0@DBLP:journals/corr/abs-1711-00937"></a>A�ron van den Oord,
			Oriol Vinyals, and Koray Kavukcuoglu. &#8220;Neural Discrete
			Representation Learning&#8221;. In:
			<span class="cmti-10">CoRR </span>abs/1711.00937 (2017). arXiv:
			<a href="https://arxiv.org/abs/1711.00937"
				><span class="cmtt-10">1711.00937</span></a
			>.
			<span class="cmcsc-10"
				><span class="small-caps">u</span><span class="small-caps">r</span><span
					class="small-caps">l</span
				></span
			>:
			<a href="http://arxiv.org/abs/1711.00937" class="url"
				><span class="cmtt-10">http://arxiv.org/abs/1711.00937</span></a
			>.
		</p>
	</dd>

	<dt id="X0-DBLP:journals/corr/abs-2111-13122" class="thebibliography">[7]</dt>
	<dd id="bib-7" class="thebibliography">
		<!--l. 273-->
		<p class="noindent">
			<a id="cite.0@DBLP:journals/corr/abs-2111-13122"></a>Konstantin Schall et
			al. &#8220;GPR1200: A Benchmark for General-Purpose Content-Based Image
			Retrieval&#8221;. In:
			<span class="cmti-10">CoRR </span>abs/2111.13122 (2021). arXiv:
			<a href="https://arxiv.org/abs/2111.13122"
				><span class="cmtt-10">2111.13122</span></a
			>.
			<span class="cmcsc-10"
				><span class="small-caps">u</span><span class="small-caps">r</span><span
					class="small-caps">l</span
				></span
			>:
			<a href="https://arxiv.org/abs/2111.13122" class="url"
				><span class="cmtt-10">https://arxiv.org/abs/2111.13122</span></a
			>.
		</p>
	</dd>

	<dt id="X0-DBLP:journals/corr/ZhouLT17" class="thebibliography">[8]</dt>
	<dd id="bib-8" class="thebibliography">
		<!--l. 273-->
		<p class="noindent">
			<a id="cite.0@DBLP:journals/corr/ZhouLT17"></a>Wengang Zhou, Houqiang Li,
			and Qi Tian. &#8220;Recent Advance in Content-based Image Retrieval: A
			Literature Survey&#8221;. In:
			<span class="cmti-10">CoRR </span>abs/1706.06064 (2017). arXiv:
			<a href="https://arxiv.org/abs/1706.06064"
				><span class="cmtt-10">1706 . 06064</span></a
			>.
			<span class="cmcsc-10"
				><span class="small-caps">u</span><span class="small-caps">r</span><span
					class="small-caps">l</span
				></span
			>:
			<a href="http://arxiv.org/abs/1706.06064" class="url"
				><span class="cmtt-10">http://arxiv.org/abs/1706.06064</span></a
			>.
		</p>
	</dd>
</dl>
