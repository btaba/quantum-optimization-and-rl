
# [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/pdf/1607.06520.pdf)

[Code](https://github.com/tolga-b/debiaswe)

Text that NLP models are trained on is biased. Debiased words should be linearly separable from gender specific terms like man/woman, but words like man/woman should retain their gender meaning.

They come up with gender specific words (brother, sister, man, woman), and gender neutral words (homemaker, football, etc.). They want gender neutral words to be equidistant to gender specific words.

## Methodology

To get gender sub-space vector. Take definitional gender words (man/woman, brother/sister). Center each pair. Do PCA on all the gender pair vectors, take the first principal component [link](https://github.com/tolga-b/debiaswe/blob/master/debiaswe/we.py#L235).


For all words **not** in the gender specific set (e.g. brother, sister, businesswoman, businessman), drop the vectors onto the gender subspace (u - v * u.dot(v) / v.dot(v)). [see here](https://github.com/tolga-b/debiaswe/blob/master/debiaswe/debias.py#L24). This ensures that gender neutral words have no cosine similarity to the gender subspace.

For all equality sets (e.g. (brother, sister), (man, woman)), we need them to be equidistant from the center of the gender subspace. We do this by dropping the midpoint of the equality set vectors onto the gender subspace, and re-adding +/- the gender direction. [see here](https://github.com/tolga-b/debiaswe/blob/master/debiaswe/debias.py#L30)
