
# [Fisher Vectors Derived from Hybrid Gaussian-Laplacian Mixture Models for Image Annotation](https://arxiv.org/pdf/1411.7399.pdf)

They take image features from VGG, and text features from word2vec. Then they convert this set of text vectores to Fisher Vectors [1] using different distributions. An alternative is to average the word-vectors, which doesn't work as well. Then they use CCA (Canonical Correlation Analysis) to map text and images to the same embedding space.

Main contribution is different distributions used for Fisher Vectors, performing better than averaging word vectors in image tasks when combined with CCA. The image tasks are:

	- Image search ranking, given a caption rank the images, 
	- Image annotation, given an image find the caption 
	- Sentence mean rank, generate a sentence and calculate the similarity.


[1] A Fisher Vector is a concatenation of gradients of the log-likelihood with respect to parameters of some distribution (i.e. a GMM) fitted on the training set in an unsupervised manner. This allows you to pool vectors into one vector, instead of just averaging word vectors for example.


Question:

	- why is mean rank a lot higher than median rank in the results? does this mean there are a few images that rank very poorly, but most have high rank?
	- why aren't these methods widely used? scalability?
	- can we approximate kernel CCA using some nonlinear function since they didn't use kernel CCA?
