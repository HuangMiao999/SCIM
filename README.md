# Information-Geometry-on-Cell-Manifolds

The datasets used are available from https://cloud.itp.ac.cn/d/19a2175f2f32435ca99f/. The environment file env_fisher.yml is also included.

Please run the calculation notebooks first to generate the encoder and curvature results.

# Parameter Selection Guidance

There are three primary parameters:

1. **The number of neighbors $K_{nei}$ used in constructing the KNN graph**
2. **The maximal neighbor order $K$ considered in neural network loss computation**
3. **The dimensionality $L$ of the Gaussian distribution obtained from the embedding**

The selection of the first two parameters is closely related to the total number of cells $n$. Specifically, for each cell, we aim for the neighbors considered in loss computation to traverse, as comprehensively as possible, all other cells in the dataset. Therefore, the following relationship should be approximately satisfied:

$$\[
n \approx (K_{nei})^K
\]$$

According to the original GE article, loss calculation is approximated via sampling expectations; thus, to ensure convergence, it is necessary for each of the $C_K^2$ possible neighbor order combinations to be covered at least once. Considering that excessively large $K$ values impose a heavy sampling burden, $K$ is typically chosen as 2, 3, or 4. In datasets with more uniform density, a larger $K$ is preferable, whereas for more heterogeneous distributions a smaller $K$ is recommended. After determining $K$, $K_{nei}$ may then be selected in the interval

$$
[n^{1/(K+1)},\ n^{1/K}]
$$

according to the aforementioned relationship.

Regarding the selection of $L$, one may initially choose a relatively large value for $L$, and subsequently increase the number of training epochs to facilitate potential overfitting. As overfitting progresses, the average variance of the embedded Gaussian distribution will increase continually in some dimensions and gradually stabilize in others. Those dimensions with excessively high variance contribute negligibly to the distance metric; hence, the number of dimensions with stable variance can be regarded as the appropriate value for $L$. Based on our empirical observations, $L$ for single-cell data is typically chosen between 6 and 10, though a slightly larger value may be adopted to provide redundancy during training.

For example，

$$n = 2000,K = 3 \to K_{nei} \in [6.68,12.59]$$
