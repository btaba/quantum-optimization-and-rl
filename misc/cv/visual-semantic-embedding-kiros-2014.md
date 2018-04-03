
# [Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/pdf/1411.2539.pdf)

They encode sentences and images, and optimize a pairwise ranking loss (images with ground truth captions should rank higher than other captions). The loss is contrastive, similar to word2vec for choosing negative examples.

They do sentence generation using some neural language model that seems non-standard as of 2018.

Their benchmarks before worse than Klein 2014, but they use a different CNN encoder (OxfordNet instead of VGG).

I like how they do a PCA projection of the images and words on the same graph. They demonstrate linguistic similarities embedded in the word+image space.

Questions:

	- what are better neural language models than the one used in this paper? probably LSTM with attention?
	- did the use any pre-trained word vectors in their LSTM encoder? Seems like they didn't
