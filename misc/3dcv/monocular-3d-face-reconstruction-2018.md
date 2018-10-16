
## [State of the Art on Monocular 3D Face Reconstruction](https://web.stanford.edu/~zollhoef/papers/EG18_FaceSTAR/paper.pdf)

### Input Modalities

1. Multi-view setups are calibrated dense camera arrays.
2. Monocular RGB - ill-posed since we expect 3D information to be solely encoded by RGB pixels. these methods rely heavily on simplifications and data-driven priors
3. Monocular RGB-D - depth is captured as well using a stero-camera setup or a camera-projector setup (kinect, etc. the projector is in the infrared domain).

### Face Models and Statistical Priors

1. Blendshape Expression Model - a linear combination of several 3D face models
2. Parametric Face Models - use a learned low-dimensional face subspace from high-resolution scans, and use the subspace to form linear combinations that reconstruct faces

### Estimating facial model parameters

1. Generative: iteratively refine the generated face (energy minimization). Some terms in the minimization are:
    - Sparse feature alignment with facial landmarks
    - Dense photometric alignment using RGB pixels and mean-squared error
    - Geometric alignment of the depth-field using mean-squared error
    - Uses gradient descent
2. Discriminative: parameter regression
    - Model the problem end-to-end, faster inference but provides worse performance as of yet (Section 8.5) - this has some great references


Challenges:
    - monocular face reconstruction accuracy has yet to be improved
    - large head rotations are difficult for monocular face reconstruction
    - monocular face reconstruction is usually restricted to the inner face mask and not the entire head 
