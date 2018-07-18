
# [Learning Deep Structure-Preserving Image-Text Embeddings](http://slazebni.cs.illinois.edu/publications/cvpr16_structure.pdf)


Canonical Correlation Analysis (CCA) maximizes correlations between projected vectors. In 2016, CCA was still state of the art. Extensions of CCA using deep networks didn't perform as well as CCA at the time since covariance estimated in mini-batches was unstable.

Some background papers on multimodal retrieval:

1. [From captions to visual concepts and back](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Fang_From_Captions_to_2015_CVPR_paper.pdf)
    - they use a multimodal model with cosine similarity and logistic loss function. They train it with a fixed number of non-matching pairs, and 1 matching pair of image/text. They use the multi-modal model to rank generated captions.
2. [Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/pdf/1411.2539.pdf)
    - They use contrastive sentences and images with cosine similarity in a max-margin loss. The embeddings are linear.
