# Boundary Detection

A research prototype for detecting boundary bands in point clouds using Gaussian-weighted neighborhood averages. Input is an N-by-d array: rows are samples, columns are coordinates or flattened image pixels.

**The active method is the fuzz/cutoff classifier, used throughout the notebook.** The log-log slope and fitted boundary-model experiments have been set aside because they performed worse in the examples. No PCA is used.

[Boundary_Detection.ipynb](Boundary_Detection.ipynb) contains synthetic geometry, MNIST, rotated images, and two-link arm poses. [manifold_utils.py](manifold_utils.py) contains the samplers and algorithms.

## Run it

~~~bash
python -m pip install numpy scipy scikit-learn matplotlib pandas jupyterlab
jupyter lab Boundary_Detection.ipynb
~~~

Run cells from the top; the first cell reloads the local module. MNIST is fetched through OpenML and reused by the rotation experiment. Random samplers are unseeded unless you set a NumPy seed.

~~~python
import numpy as np
import manifold_utils as mu

X = mu.sample_hemisphere(N=1000)
labels = mu.classify_boundary_threshold(X, fuzz=0.1, use_knn=False)
print(f"{labels.sum()} boundary candidates out of {len(X)} samples")
~~~

The first cell exposes a shared `FUZZ=0.1` setting, restored for MNIST as well as the synthetic examples.

The first eight examples show two panels: the score at the displayed epsilon and the fuzz classification across adaptive scales. Classification uses dense weighting; the helix and intersecting-plane score panels retain their labeled kNN calculations. MNIST image grids, t-SNE, rotation borders, and pose plots all use cutoff labels.

## 1. The Laplacian is a displacement toward a local average

Write $h=\sqrt{\epsilon}$ for the length scale. The kernel and row-normalized operator are

$$
W_{ij}=e^{-\|x_i-x_j\|^2/\epsilon},\qquad
P_{ij}=\frac{W_{ij}}{\sum_\ell W_{i\ell}},\qquad
L_\epsilon=P-I.
$$

Consequently,

$$
(L_\epsilon X)_i=\sum_jP_{ij}x_j-x_i
$$

points from the query to its weighted neighborhood average. Constants satisfy $L_\epsilon\mathbf 1=0$. Self weights are included.

The sign is opposite to the common random-walk convention $I-P$. For a smooth interior point with uniform sampling, $4L_\epsilon f/\epsilon$ approaches the Laplace-Beltrami operator with sign convention $\operatorname{div}\nabla$. With smooth positive sampling density rho, the leading interior expansion is instead

$$
\frac{4}{\epsilon}L_\epsilon f
\approx \Delta_M f+2\langle\nabla_M\log\rho,\nabla_M f\rangle.
$$

Thus row normalization does not remove drift toward denser regions. Coordinate functions also respond to extrinsic curvature. See [diffusion-operator normalization](https://pmc.ncbi.nlm.nih.gov/articles/PMC1140422/).

Epsilon has units of squared distance; this Gaussian convention has $\epsilon=2\sigma^2$.

## 2. Why b values matter

Normalize the displacement by the length scale:

$$
u_i(\epsilon)=\frac{(L_\epsilon X)_i}{\sqrt{\epsilon}},
\qquad b_i(\epsilon)=\|u_i(\epsilon)\|.
$$

Neighbors on opposite sides approximately cancel in a smooth interior. A boundary removes neighbors on one side, leaving an inward displacement.

For constant density on a flat half-space, the continuum response at distance delta from the boundary is

$$
u(h)=F(\delta/h)n,\qquad
F(t)=\frac{e^{-t^2}}{\sqrt{\pi}\,[1+\operatorname{erf}(t)]}.
$$

Here n is the inward unit vector tangent to the manifold and perpendicular to its boundary. At the boundary, $F(0)=1/\sqrt{\pi}$, approximately **0.5642**. F decreases toward zero as the boundary becomes farther away relative to h.

This is the leading response underlying the [boundary-direction estimator](https://arxiv.org/abs/1912.01391); that paper's bandwidth corresponds to our h. The density and boundary-estimation construction is developed by [Berry and Sauer](https://arxiv.org/abs/1511.08271).

The constant is an ideal limit, not a universal upper bound on sampled scores. Curvature, density gradients, noise, and nonlocal neighborhoods can all increase b.

## 3. The fuzz classifier and its geometric meaning

The implemented rule is

$$
\operatorname{label}_i
=\mathbf 1\!\left\{
\max_{\epsilon\in\mathcal E}b_i(\epsilon)>
1/\sqrt{\pi}-\texttt{fuzz}
\right\}.
$$

With fuzz=0.1 the cutoff is approximately **0.4642**. Increasing fuzz lowers the cutoff and admits more points. Equality returns 0. Labels mean boundary candidate=1 and no boundary evidence=0.

Fuzz has a useful geometric interpretation in the ideal half-space model. For $0<\texttt{fuzz}<1/\sqrt{\pi}$, let $\tau$ solve

$$
F(\tau)=1/\sqrt{\pi}-\texttt{fuzz}.
$$

Then the cutoff at a single scale means $\delta<\tau h$. For fuzz=0.1, tau is approximately **0.16384**. Alternatively, to target a chosen relative band width tau, set

$$
\texttt{fuzz}=F(0)-F(\tau).
$$

This interpretation requires a resolved, approximately flat, uniformly sampled neighborhood. It is not an exact distance estimate on arbitrary data. Taking a maximum across scales also means the largest admissible h can determine the detected band's width.

Negative fuzz raises the threshold above the ideal boundary value. For example, fuzz=-0.8 gives 1.3642. This can select strong image-space displacements, but cannot be explained as a positive-width boundary band in the ideal half-space model. It is not a mathematical repair for unsuitable scales or a poor data metric.

## 4. Scaling with epsilon: useful theory, no slope classifier in the notebook

| Location | Coordinate displacement | Normalized score |
|---|---|---|
| Smooth interior | $O(\epsilon)$ | $O(\sqrt{\epsilon})$ |
| Smooth boundary | Leading order $\sqrt{\epsilon}$ | Approaches $1/\sqrt{\pi}$ |

An interior log-log slope is 1/2 only when its leading coefficient is nonzero. At a fixed positive distance from a boundary, the leading response follows $F(\delta/h)$, not an exact power law. At fixed sample size, very small scales are dominated by discretization and self weights; very large scales mix nonlocal geometry.

These limitations made the slope restrictions unreliable. The fitted boundary model also allowed its drift correction to absorb too much of the observed signal. Both approaches are retired from the workflow. The scaling laws motivate locality and bias analysis, but are not classification conditions.

## 5. Mathematically motivated improvements to investigate

**The following are proposals, not changes silently applied to the current cutoff.** The current baseline and its score normalization are preserved.

A reliable positive-fuzz solution for MNIST has **not** been established: with the current adaptive grid, the dense baseline selected all 2,000 checked nines at fuzz=0.1. A trial that weighted displacement by variance along its direction preserved the ideal boundary constant but introduced many false positives on the no-boundary sphere control; it was rejected. Reducing a detection count is not enough to validate a new score.

### A. Select trustworthy scales before taking a maximum

The bandwidth must resolve enough samples while remaining small relative to curvature radii, density-variation lengths, and separation between nearby sheets.

For normalized weights, effective sample size

$$
N_{\mathrm{eff}}=\frac{1}{\sum_jP_{ij}^2}
$$

helps diagnose under-supported neighborhoods. It does not by itself establish locality. A useful extension would combine support with geometric locality, self-weight, and kNN-tail diagnostics, then apply the existing cutoff only at admissible scales. If no such scale exists, report insufficient resolution.

This should be the first investigation: the current automatic fallback expands the epsilon range when its nominal bounds cross, and that expansion has no locality guarantee. A universal cap such as 20% of N is also unjustified and can conflict with a minimum support count in small clouds.

### B. Account for uncertainty instead of increasing fuzz

Taking a vector norm introduces positive sampling bias:

$$
\mathbb E\|\widehat u\|^2
=\|\mathbb E\widehat u\|^2+\operatorname{tr}\operatorname{Cov}(\widehat u).
$$

If a valid error bound gives $\|\widehat u-u\|\le\eta$, the reverse triangle inequality gives $\|u\|\ge\max(\|\widehat u\|-\eta,0)$. One could threshold this lower bound at $F(0)-\texttt{fuzz}$, retaining the geometric meaning of fuzz.

The bound would need calibration for the weighted ratio estimator and selection over multiple scales. Those scales share samples and are correlated. A naive independent-scale standard error, or an uncalibrated noise subtraction, is not sufficient.

### C. Cancel leading smooth drift with two vector measurements

A direct alternative to fitting a free drift vector is

$$
v_r(h)=\frac{r\,u(h)-u(rh)}{r-1},\qquad r>1.
$$

Under the expansion $u(h)=a+hc+O(h^2)$, this gives $v_r(h)=a+O(h^2)$: it preserves the leading boundary term and cancels the linear smooth drift. In an interior expansion with $a=0$, the leading drift cancels too. The combination must be applied to vectors before taking norms.

This is a promising experiment, not a drop-in fix. Both scales must be local, sampling noise can be amplified, and near a boundary the leading response becomes

$$
\frac{rF(\delta/h)-F(\delta/(rh))}{r-1}\,n.
$$

It can change sign, so its magnitude needs its own band calibration. The original fuzz-to-distance mapping cannot simply be reused.

### D. Correct density only with a boundary-aware estimate

With a reliable estimate of the sampling density, reweighting neighbors by $W_{ij}/\widehat\rho(x_j)$ before row normalization can reduce density drift and recover the locally uniform interpretation.

A raw kernel degree is confounded by missing mass near the boundary: dividing by it can change the very boundary response being detected. A boundary-aware density estimate and a fresh bias calculation are required; see [boundary-corrected density estimation](https://math.gmu.edu/~tsauer/pre/CSDA2017.pdf). This would still leave curvature, noise, and singularities to address.

None of these proposals requires PCA. More nonzero detections would not, by itself, establish an improvement.

## Scales, kNN, and computation

The adaptive grid has 30 log-spaced epsilons. Its lower endpoint is the squared median k-th neighbor distance, floored at 1e-8, using k=max(5,floor(log N)), capped at N-1. The upper endpoint is (0.2 diameter)^2; when it is no larger than the lower endpoint, the code substitutes ten times the lower endpoint.

In kNN mode, the graph is the symmetric union of neighbor selections. Searches include self, and symmetrization can increase a row's degree. k controls available neighbors; epsilon controls their weights. A small k can truncate significant Gaussian mass.

~~~python
labels = mu.classify_boundary_threshold(
    X, fuzz=0.1, use_knn=True, k=min(len(X), 64), batch_size=256
)
~~~

The score/cutoff kNN default is 15; 64 above is an explicit example, not an optimal choice.

Distances or neighbors are cached once per call. Weight calculations run in row blocks, and threshold classification stops evaluating a query after it passes while retaining it as a neighbor. Dense mode retains an N-by-N float64 distance cache: **392 MB for 7,000 points**, plus working arrays. kNN scale selection still uses an exact distance pass in blocks.

## Files, experiments, and interpretation

| Entry point | Purpose |
|---|---|
| `sample_*` | Synthetic point-cloud generators |
| `compute_laplacian`, `compute_laplacian_knn` | Explicit operators for inspection and score plots |
| `choose_adaptive_epsilons` | Shared candidate epsilon grid |
| `compute_b_vals` | Scores for all points and requested epsilons |
| `classify_boundary_threshold` | Active fuzz classifier |
| `classify_boundary_slope`, `classify_boundary_concavity` | Retired experiments; unused by the notebook |

The former classify_boundary_integrated has been removed. The existing tests directory still includes an obsolete check for it and is not a fully current validation suite.

The disk uses a grid, so requested N=1000 yields an approximate sample count. The sphere is a no-boundary control. The square perimeter is a closed curve with corners, while the double cone and intersecting planes contain singularities. Their strong scores need not identify smooth boundaries; first moments alone cannot reliably distinguish all singularities from boundaries.

MNIST classification uses pixel space. A useful-looking selection, or an edge in the t-SNE visualization, is not ground-truth manifold-boundary membership. Class shape, sampling density, noise, and bandwidth all affect results; there is no digit-9-specific guarantee.

The rotation interval has endpoints, provided its image curve is locally smooth and does not identify them through symmetry. A complete periodic rotation would have no parameter boundary. The arm experiment uses joint limits as an interpretable reference for detections made in its four-dimensional observation space.
