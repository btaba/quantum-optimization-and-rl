
# (Quantum algorithms: an overview)[https://arxiv.org/pdf/1511.04206.pdf]

TLDR: Quantum computers are designed to be faster than classical computers.

## How to measure speedup

Use computational complexity classes:

* P: can be solved by deterministic classical computer in polynomial time
* BPP: can be solved by a probabilistic classical computer in polynomial time
* BQP: can be solved by a quantum computer in polynomial time
* NP: solution can be checked on deterministic classical comp. in polynomial time
* QMA: solution can be checked on quantum comp. in polynomial time

QMA is larger than NP, BQP is larger than BPP.

## Problems


1. HSP - Hidden Subgroups Problem
    - Shor's algorithm gives faster integer factorisation
2. Search and optimisation
    - Grover's algorithm solves unstructured search problem faster than in classical case
    - Given a function $f: {0, 1}^n$ -> ${0, 1}$ find x such that $f(x) = 1$, can be solved in $O(\sqrt(N))$ instead of $O(N)$ where $N = 2^n$
3. Adiabatic optimisation
    - Use adiabatic theorem in quantum mechanics to constrain ground state over some solution set until we have satisfied all constraints
    - Adiabatic Theorem: states that if conditions are changed slowly enough, the state of a system will remain in a similar eigenstate under a new Hamiltonian
    - D-wave systems has apparently built systems to implement this
4. Quantum Simulation
    - simulate quantum systems better
5. Quantum walks
    - walk boolean trees, determine outcomes of two-player games faster
    - speedup in evaluating Markov chains
6. Solving linear equations - HHL Algo

## Applications

Still super small, factorisation of 21, 2x2 systems of linear equations

## Zero qubit applications

Can prove new limits on classical data structures, can estimate how hard systems are to solve using quantum computational complexity

## Outlook

Finding hidden problem structure for quantum computers to exploit remains an important problem. Then find some application I guess.

Any quantum algorithm may be expressed as the use of quantum fourier transform interspersed with classical processing.

