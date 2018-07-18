[Deep Learning](http://www.deeplearningbook.org)

# Chapter 2 - Linear Algebra

## Linear Equations

Ax = b

- The **span** of a set of vectors is the set of all points obtainable by linear combination of the original vectors.
- Solution exists if b is in the span of columns in A.
- A **singular** matrix is square with linearly dependent columns.

## Norms

A norm is a distance measure that must satisfy these 3 properties:

1. if f(x) = 0 then x = 0
2. triangle inequality: f(x) + f(y) <= f(x) + f(y)
3. for all a, f(ax) = |a|f(x)

**L2 norm** is Euclidean norm. Max norm or **infinity norm** is just the absolute value of the largest value in the vector. **Frobenius norm** is just the square root of the sum of squares for all values in a matrix.

## Eigendecomposition

- An eigenvector of a matrix is a vector that when multiplied by that matrix produces the eigenvector times a scalar. A x = lambda x. lambda is the eigenvalue.
- Every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and eigenvalues. **A = Q diag(lambda) Q^T**.
- A matrix is singular if any of the eigenvalues are zero.
- A matrix with all positive eigenvalues is **positive definite**.
- A matrix with all positive eigenvalues or zero is **positive semi-definite**.

## Singular Value Decomposition

If a matrix is not square, the eigendecomposition is not deﬁned, and we must use a singular value decomposition instead. 

- The SVD of a matrix A is A = UDV^T instead of A = VDV^-1. If A is mXn, then U is mXm and V is nXn. D is a diagonal matrix but not necessarily square.
- The elements along the diagonal of D are known as **singular values**.


## Trace

Sum of diagonals of a matrix.


## Determinant

Product of all eigenvalues of a matrix.

## PCA

Lossy linear compression of m points X, solved by taking getting top-k eigenvectors of X^T X corresponding to top-k eigenvalues - call this matrix D. The compression is c = D^T x, and the decompression is x = Dc.

# Chapter 3 - Probability

Discrete random variables have probability mass functions; P(x) is the probability that the outcome x occurs. Continuous random variables have probability density functions that must integrate to 1; P(x) is the probability that we get a value x within an infinitesimal region dx, which can be greater than 1!

## Chain rule of conditionals

P (a, b, c) = P (a | b, c)P (b | c)P (c)


## Distributions

### Multinoulli

Multinoulli describes a distribution of a single discrete variable that can take on k discrete values for 1 sample. As opposed to the multinomial distribution which is the distribution over k discrete values for n > 1 samples.

### CLT

The **central limit theorem** shows that the sum of many independent random variables is approximately normally distributed.

### Mixture Model

P (x) = sum(P (c = i)P (x | c = i))

where c comes from a multinoulli distribution. c is called a **latent variable**. A Gaussian Mixture model is a universal approximator of densities, meaning we can approximate any smooth density by a GMM with zero error given that we use a sufficient number of components.

## Common Functions

- Logistic sigmoid - the inverse logit: 1 / (1 + exp(-x)) , bounded between 0 and 1, often used to approximate a probability.
- Softplus function: log(1 + exp(x)), this is a soft version of max(0, x).

## Measure Theory

Jacobian Matrix (J) is a matrix of partial derivatives. If y = g(x), we want to preserve |p(x)dx| = |p(g(x))dy|. This roughly generalizes to multiple dimensional spaces as such: p(x) = p(g(x)) det(J). 


## Information Theory

- Self information: I = -log(P(x)), where log = natural logarithm. I(x) has units of nats. 1 nat = probability of observing an event of probabilit 1/e. If we use base 2 logarithms, we have units of bits or shannons.
- Shannon Entropy gives us the amount of uncertainty in an entire probability distribution. H(x) = E[I(x)]. This tells us the average number of bits to needed to encode a symbol drawn from P. For continuous variables, we call this the differential entropy.
- KL divergence: we can measure how different two distributions are. KL(P|Q) = E[log(P) - log(Q)]. This is the additional bits or nats needed to send a message containing symbols drawn from P when using a code designed for distribution Q.
    - KL is non-negative
    - has a minimum of zero
    - not symmetric
    - **If we are fitting Q to P, minimizing KL(P|Q) will blur modes in P, and minimizing KL(Q|P) will seek modes in P.** This is because KL(P|Q) makes us seek high probabilities for Q where P is high in probability. KL(Q|P) makes us seek low probabilities for Q where P is low.
- Cross entropy: H(P, Q) = H(P) + KL(P|Q)
    - when we minimize the cross entropy with respect to Q, we are minimizing the KL.
    - best case is 0, worst case is +infinity


# Chapter 4 - Numerical Computation

- Poor Conditioning: occurs when ratio of max and min eigenvalues are large. the matrix inversion becomes very sensitive to numeric instability.
- Hessian Matrix: is the Jacobian of the gradient of a function f(x).
    - Used in Newton's method to compute gradients to minimize f(x). The Hessian matrix gives us curvature information to know which direction has a better acceleration of descent.
- Lipschitz Continuous: a function is lipschitz continuous if for all x and y, |f(x) - f(y)| <= L |x - y|\_2. where L is a lipschitz constant.


# Chapter 5 - Machine Learning Basics


- MSE can be deconstructed into a bias and variance term. Bias is E[x_hat] - E[x], where x is the true underlying value, and x_hat is the predicted x. Variance is Var[x_hat].
    - Underfitting is akin to high bias and low variance. Overfitting is akin to high variance and low bias.

## Maximum Likelihood Estimation

Consists of maximizing the log probability of the model given the data. This is equivalent to minimizing KL or cross-entropy between the data and model distributions.

For linear regression, maximizing the log probability is equivalent to minimizing mean squared error if the probability distribution of the predictions is assumed to be Gaussian conditional with mean at the prediction.


# Chapter 6 - Deep Feedforward Networks

...


# Chapter 10 - Sequence Modeling: Recurrent and Recursive Nets

Teacher Forcing: recurrent networks that have their outputs leading back into the network at the next timestep. During training time, the true outputs are used. During test, the network outputs are used.

BPTT: back-propagation through time, used for recurrent networks with recurrent connections between hidden units. If the recurrent connection is only between output units, we don't need BPTT (we can just compute in parallel for each output).

Losses of the target and output predictions are usually cross-entropy loss (negative log likelihood), or MSE.



