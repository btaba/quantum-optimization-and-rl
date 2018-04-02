
# [A quantum mechanical algorithm for database search](https://arxiv.org/pdf/quant-ph/9605043.pdf)

Quadratic speedup for unstructured search O(\sqrt N) instead of O(N). Given N elements, find an element which makes $f(x) = 1$ where $f : X \rightarrow \{0, 1\}$.


# [CMU Lecture Notes](https://www.cs.cmu.edu/~odonnell/quantum15/lecture04.pdf)

Algorithm:

    - apply Hadamard gate to all bits, this puts them in a superposition of all n-bit strings
    - Apply oracle gate and Grover diffusion operator repeatedly, until some condition is met

Oracle gate:

Given an n-bit string input x, this gate will flip the sign of the state if f(x) = 1, where f(x) determines which unique record we want to find in the database. This gate is equivalent to adding an extra bit to record whether f(x) = 1 or 0 (proof provided in lecture notes).

After applying the Hadamard gate, the Oracle gate will flip the amplitude of x* (the solution).


Grover diffusion operator:

The Grover diffusion operator increases the amplitude of the flipped x* and decreasese the amplitude of all other states.

Define $\alpha_x$ is amplitude of state x. Define $\mu = \frac{1}{N} \sum \alpha_x$. The Grover diffusion operator does α_x|x⟩ →   (2μ−α_x)|x⟩ for all states.


## Questions

- How is the oracle function physically implemented?
