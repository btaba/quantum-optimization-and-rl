
[Deep Neural Nets for Youtube Recommendation](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)

They do 2-stage recommendations. First they generate video candidates with high recall, then they rank those candidates.

The high-recall candidate generation model is learned similar to word-2-vec models. They embed video IDs and user search tokens, effectively learning video embeddings and word embeddings within the same network. They also add other features, like age/gender. The output of the network is the user vector. They maximize the probability of a user watching a video using cross-entropy loss, where the probability is similar to word2vec (using dot-product of user and video vectors), see section 3.1. They can then do approximate nearest neighbor lookups at test time really quickly to generate candidates.

They filter the candidates videos per user by using a similar network with the embeddings learned previously with a logistic regression to get high precision.
