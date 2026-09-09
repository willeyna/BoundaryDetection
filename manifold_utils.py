import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

def add_ambient_noise(X, noise=0.0):
    """
    Adds ambient Gaussian noise to the coordinate matrix X.

    Args:
        X: (N, d) array of point coordinates
        noise: standard deviation of Gaussian noise

    Returns:
        X_noisy: (N, d) array of coordinates with noise added
    """
    if noise > 0:
        X = X + np.random.randn(*X.shape) * noise
    return X

def sample_disk(N=1000, radius=1.0, noise=0.0):
    """
    Sample points uniformly distributed on a grid inside a 2D disk (x^2 + y^2 <= radius^2).

    Args:
        N: approximate number of points to sample (actual points will match grid size)
        radius: radius of the disk (default is 1.0)
        noise: ambient noise level

    Returns:
        X: (N_actual, 2) array of coordinates on a Cartesian grid
    """
    # Estimate grid size M to get approximately N points inside the disk (area = pi * radius^2)
    # The ratio of area of disk to square [-radius, radius]^2 is pi/4.
    # Therefore, M^2 * (pi/4) approx N => M = ceil(sqrt(4 * N / pi))
    M = int(np.ceil(np.sqrt(4 * N / np.pi)))
    
    # Generate coordinates on a uniform 2D grid in [-radius, radius] x [-radius, radius]
    x_range = np.linspace(-radius, radius, M)
    y_range = np.linspace(-radius, radius, M)
    xx, yy = np.meshgrid(x_range, y_range)
    X_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Filter points inside the disk
    inside = (X_grid[:, 0]**2 + X_grid[:, 1]**2) <= radius**2
    X = X_grid[inside]
    
    return add_ambient_noise(X, noise)

def sample_cone(N=1000, noise=0.0):
    """
    Sample points uniformly from the surface of a double cone
    (light cone):
        z = ±sqrt(x^2 + y^2), |z| <= 1.

    Args:
        N: number of points to sample
        noise: ambient noise level

    Returns:
        X: (N, 3) array of coordinates
    """
    r = np.sqrt(np.random.rand(N))
    theta = np.random.rand(N) * (2 * np.pi)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Randomly choose upper or lower cone
    sign = np.where(np.random.rand(N) < 0.5, 1.0, -1.0)
    z = sign * r

    X = np.column_stack((x, y, z))
    return add_ambient_noise(X, noise)

def sample_cylinder(N=1000, noise=0.0):
    """
    Sample points uniformly from the surface of a cylinder (x^2 + y^2 = 1, z in [0, 1]).

    Args:
        N: number of points to sample
        noise: ambient noise level

    Returns:
        X: (N, 3) array of coordinates
    """
    theta = np.random.rand(N) * (2 * np.pi)
    z = np.random.rand(N)
    x = np.cos(theta)
    y = np.sin(theta)
    X = np.column_stack((x, y, z))
    return add_ambient_noise(X, noise)

def sample_hemisphere(N=1000, noise=0.0):
    """
    Sample points uniformly from the surface of the unit upper hemisphere (x^2 + y^2 + z^2 = 1, z >= 0).

    Args:
        N: number of points to sample
        noise: ambient noise level

    Returns:
        X: (N, 3) array of coordinates
    """
    z = np.random.rand(N)
    theta = np.random.rand(N) * (2 * np.pi)
    r = np.sqrt(1.0 - z**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    X = np.column_stack((x, y, z))
    return add_ambient_noise(X, noise)

def sample_sphere(N=1000, noise=0.0):
    """
    Sample points uniformly from the surface of the unit sphere in R^3 (x^2 + y^2 + z^2 = 1).

    Args:
        N: number of points to sample
        noise: ambient noise level

    Returns:
        X: (N, 3) array of coordinates
    """
    z = np.random.rand(N) * 2.0 - 1.0  # uniform z in [-1, 1]
    theta = np.random.rand(N) * (2 * np.pi)
    r = np.sqrt(1.0 - z**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    X = np.column_stack((x, y, z))
    return add_ambient_noise(X, noise)

def sample_helix(N=500, noise=0.0):
    """
    Sample points uniformly along a 3D helix (a 1D curve: x = cos(t), y = sin(t), z = 0.1t, t in [0, 4*pi]).

    Args:
        N: number of points to sample
        noise: ambient noise level

    Returns:
        X: (N, 3) array of coordinates
    """
    t = np.random.rand(N) * 4.0 * np.pi  # two full turns
    x = np.cos(t)
    y = np.sin(t)
    z = 0.1 * t
    X = np.column_stack((x, y, z))
    return add_ambient_noise(X, noise)

def sample_plane_intersection(N=1000, noise=0.0, angle=45):
    """
    Sample points uniformly from two intersecting planes.

    Plane 1: z = 0
    Plane 2: Plane obtained by rotating z = 0 about the x-axis by `angle`
             degrees. The two planes intersect along the x-axis.

    Args:
        N: number of points to sample
        noise: ambient noise level
        angle: angle between the planes in degrees (default: 45)

    Returns:
        X: (N, 3) array of coordinates
    """
    N1 = N // 2
    N2 = N - N1

    # Plane 1: z = 0
    x1 = np.random.uniform(-1, 1, N1)
    y1 = np.random.uniform(-1, 1, N1)
    z1 = np.zeros(N1)
    X1 = np.column_stack((x1, y1, z1))

    # Plane 2: rotate z=0 around the x-axis
    theta = np.deg2rad(angle)

    u = np.random.uniform(-1, 1, N2)  # x-coordinate
    v = np.random.uniform(-1, 1, N2)  # in-plane coordinate

    x2 = u
    y2 = v * np.cos(theta)
    z2 = v * np.sin(theta)

    X2 = np.column_stack((x2, y2, z2))

    X = np.vstack((X1, X2))
    return add_ambient_noise(X, noise)

def sample_square(N=1000, noise=0.0):
    """
    Sample points uniformly from the perimeter of a square with side length 2
    in the xy-plane (z = 0).

    Square corners:
        (-1,-1), (1,-1), (1,1), (-1,1)

    Args:
        N: number of points to sample
        noise: ambient noise level

    Returns:
        X: (N, 3) array of coordinates
    """
    side = np.random.randint(0, 4, N)
    t = np.random.uniform(-1, 1, N)

    x = np.empty(N)
    y = np.empty(N)

    # Bottom edge
    mask = side == 0
    x[mask] = t[mask]
    y[mask] = -1

    # Right edge
    mask = side == 1
    x[mask] = 1
    y[mask] = t[mask]

    # Top edge
    mask = side == 2
    x[mask] = t[mask]
    y[mask] = 1

    # Left edge
    mask = side == 3
    x[mask] = -1
    y[mask] = t[mask]

    z = np.zeros(N)

    X = np.column_stack((x, y, z))
    return add_ambient_noise(X, noise)

def compute_laplacian(X, epsilon):
    """
    Computes the dense transition Laplacian L = P - I.

    Args:
        X: (N, d) array of coordinates
        epsilon: bandwidth parameter (epsilon = 2 * sigma^2)

    Returns:
        L: (N, N) dense array, L = P - I
    """
    dists_sq = cdist(X, X, 'sqeuclidean')
    W = np.exp(-dists_sq / epsilon)
    row_sums = np.sum(W, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1e-12, row_sums)
    P = W / row_sums
    L = P - np.eye(X.shape[0])
    return L

def compute_laplacian_knn(X, epsilon, k=15):
    """
    Computes the transition Laplacian L = P - I using a kNN graph.

    Args:
        X: (N, d) array of coordinates
        epsilon: bandwidth parameter (epsilon = 2 * sigma^2)
        k: number of nearest neighbors (default: 15)

    Returns:
        L: (N, N) sparse matrix, L = P - I
    """
    N = X.shape[0]
    
    # Use NearestNeighbors for kNN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Compute Gaussian kernel weights (same kernel)
    weights = np.exp(-(distances**2) / epsilon)
    
    # Create sparse adjacency matrix
    row_indices = np.repeat(np.arange(N), k)
    col_indices = indices.flatten()
    W = sp.csr_matrix((weights.flatten(), (row_indices, col_indices)), shape=(N, N))
    
    # Symmetrize the weight matrix to ensure the graph is undirected
    W = W.maximum(W.T)
    
    # Row normalize
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-12
    
    # Sparse row normalization
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ W
    
    # Compute Laplacian L = P - I
    I = sp.eye(N)
    L = P - I
    
    return L

def _validate_batch_size(batch_size):
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def _adaptive_epsilons(X, num_eps=30, k=None, gamma=0.2,
                       batch_size=256, distances_sq=None):
    """Select the original scales using row blocks, optionally reusing distances."""
    N = X.shape[0]
    if N < 2:
        return np.ones(num_eps)
    if k is None:
        k = max(5, int(np.floor(np.log(N))))
    k = min(k, N - 1)
    kth_distances = np.empty(N)
    diameter_sq = 0.0
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        block = (cdist(X[start:stop], X, 'sqeuclidean')
                 if distances_sq is None else distances_sq[start:stop])
        diameter_sq = max(diameter_sq, float(np.max(block)))
        # Partition a copy so the cached distance matrix stays intact.
        kth_distances[start:stop] = np.sqrt(np.partition(block, k, axis=1)[:, k])
    # Take the median of distances BEFORE squaring, as in the original code.
    eps_min = max(np.median(kth_distances) ** 2, 1e-8)
    eps_max = (gamma * np.sqrt(diameter_sq)) ** 2
    if eps_max <= eps_min:
        eps_max = eps_min * 10.0
    return np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)


def choose_adaptive_epsilons(X, num_eps=30, k=None, gamma=0.2, batch_size=256):
    """Choose the same log-spaced scales without a full distance allocation.

    The lower bound is the squared median k-th neighbor distance; the upper
    bound is (gamma * diameter)**2. Exact distances are processed in row blocks.
    """
    _validate_batch_size(batch_size)
    return _adaptive_epsilons(np.asarray(X), num_eps, k, gamma, batch_size)


def _knn_squared_distances(X, k=15):
    """Cache the symmetric neighbor graph once, including zero-distance edges."""
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    rows = np.repeat(np.arange(N), k)
    cols = indices.ravel()
    # Union of directed edges, matching W.maximum(W.T) in the public helper.
    # Unique edge keys avoid summing reciprocal distances. Explicit zeros must
    # survive: self edges and distinct coincident points have kernel weight 1.
    keys = np.concatenate((rows * N + cols, cols * N + rows))
    values = np.tile(distances.ravel() ** 2, 2)
    keys, first = np.unique(keys, return_index=True)
    return sp.csr_matrix((values[first], (keys // N, keys % N)), shape=(N, N))


def _score_rows(distances_sq, X, rows, epsilon):
    """Compute ||weighted mean - point|| / sqrt(epsilon), with no Laplacian."""
    if sp.issparse(distances_sq):
        weights = distances_sq.copy()
        weights.data *= -1.0 / epsilon
        np.exp(weights.data, out=weights.data)
        row_sums = np.asarray(weights.sum(axis=1)).ravel()
    else:
        weights = np.multiply(distances_sq, -1.0 / epsilon)
        np.exp(weights, out=weights)
        row_sums = weights.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1e-12, row_sums)
    displacement = (weights @ X) / row_sums[:, None] - X[rows]
    return np.linalg.norm(displacement, axis=1) / np.sqrt(epsilon)


def _evaluate_boundary(X, eps_vals=None, use_knn=False, batch_size=256,
                       threshold=None, n_scales=None, **kwargs):
    """Share cached geometry across scales and optionally stop classified rows."""
    _validate_batch_size(batch_size)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a nonempty 2D array")
    N = X.shape[0]
    if use_knn:
        distances_sq = _knn_squared_distances(X, **kwargs)
    else:
        # One persistent N-by-N cache; all other dense graph work is blocked.
        distances_sq = np.empty((N, N), dtype=np.float64)
        for start in range(0, N, batch_size):
            stop = min(start + batch_size, N)
            cdist(X[start:stop], X, 'sqeuclidean', out=distances_sq[start:stop])
    if eps_vals is None:
        eps_vals = _adaptive_epsilons(
            X, batch_size=batch_size,
            distances_sq=None if use_knn else distances_sq)
    eps_vals = np.asarray(eps_vals, dtype=float)
    if (eps_vals.ndim != 1 or not np.all(np.isfinite(eps_vals))
            or np.any(eps_vals <= 0)):
        raise ValueError("eps_vals must be a 1D array of finite positive values")
    if n_scales is not None:
        eps_vals = eps_vals[:n_scales]
    result = (np.zeros((N, len(eps_vals))) if threshold is None
              else np.zeros(N, dtype=int))
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        rows = np.arange(start, stop)
        block = distances_sq[start:stop]
        for j, eps in enumerate(eps_vals):
            scores = _score_rows(block, X, rows, eps)
            if threshold is None:
                result[rows, j] = scores
            else:
                hits = scores > threshold
                result[rows[hits]] = 1
                if np.all(hits):
                    break
                if np.any(hits):
                    # Drop query rows only. All original points remain neighbors.
                    rows = rows[~hits]
                    block = block[~hits]
    return eps_vals, result


def compute_b_vals(X, eps_vals=None, use_knn=False, batch_size=256, **kwargs):
    """Compute boundary scores for every point and epsilon.

    Dense distances or kNN neighbors are cached once per call. batch_size limits
    the number of rows of weights in memory; dense mode retains one N-by-N
    float64 distance cache. kNN mode retains only the sparse neighbor graph and
    uses distance blocks to select adaptive scales. Pass k through kwargs.
    Returns an (N, number of epsilons) array, as before.
    """
    return _evaluate_boundary(X, eps_vals, use_knn, batch_size, **kwargs)[1]


def classify_boundary_concavity(X, margin=1e-4, min_val_ratio=0.1, batch_size=256):
    """
    Classify points as boundary (1) or interior (0) by testing the sign of 
    the second derivative of b with respect to log(eps) using a scale-invariant margin
    smoothed over the smallest scales.

    Args:
        X: (N, d) array of coordinates
        margin: relative threshold multiplier (default: 1e-4) to filter out 
                near-zero numerical fluctuations in the deep interior.
        min_val_ratio: minimum ratio of the local b value to the global maximum of b
                       (default: 0.1) required to prevent interior noise from firing.

    Returns:
        classes: (N,) array of 0s and 1s
    """
    eps_vals, b_vals = _evaluate_boundary(X, batch_size=batch_size)
    log_eps = np.log(eps_vals)
    
    # Compute first and second derivatives along the epsilon axis
    db = np.gradient(b_vals, log_eps, axis=1)
    d2b = np.gradient(db, log_eps, axis=1)
    
    # Average the concavity over the first 3 scales to smooth out discretization noise
    mean_d2b = np.mean(d2b[:, :3], axis=1)
    
    # Scale-invariant noise floor: threshold is proportional to the global maximum of b
    global_max = np.max(b_vals)
    threshold = -margin * global_max
    
    # Must be concave-down, and the value itself must be significant (not near-zero noise)
    is_concave_down = mean_d2b < threshold
    is_significant = np.mean(b_vals[:, :3], axis=1) > (min_val_ratio * global_max)
    
    is_boundary = is_concave_down & is_significant
    return is_boundary.astype(int)

def classify_boundary_threshold(X, fuzz=0.1, use_knn=False, batch_size=256, **kwargs):
    """Label points whose score at any adaptive scale exceeds 1/sqrt(pi)-fuzz.

    Geometry is cached across scales. Once a point exceeds the threshold its
    remaining scores are skipped, but it still contributes to other points'
    neighborhoods. batch_size controls row-block memory; kwargs accepts k.
    Returns an (N,) integer array: boundary=1, interior=0.
    """
    limit = 1.0 / np.sqrt(np.pi) - fuzz
    return _evaluate_boundary(X, use_knn=use_knn, batch_size=batch_size,
                              threshold=limit, **kwargs)[1]


def classify_boundary_slope(
        X, eps_vals=None, n_scales=6, slope_threshold=0.4, b_floor=1e-12,
        use_knn=False, batch_size=256, return_slopes=False,
        return_diagnostics=False, fuzz=0.1, max_log_residual=0.1,
        min_slope=0.0, **kwargs):
    """Retired experiment: unused by the notebook; prefer classify_boundary_threshold.

    Combine a nondecreasing log-log plateau with the existing b-value cutoff.

    Fit log(b_i(epsilon)) = intercept_i + slope_i * log(epsilon) over
    the first n_scales of the shared epsilon grid. Boundary candidates must
    satisfy all three conditions:
        min_slope <= slope_i <= slope_threshold
        max(b_i over fitted scales) > 1/sqrt(pi) - fuzz
        RMS log-fit residual <= max_log_residual

    The default interval [0, 0.25] follows the leading half-space response:
    constant on a boundary, increasing near one. It excludes falling saturation
    curves instead of treating all negative slopes as boundary-like. No boundary
    distance or drift model is fitted. Scores include self weight as in cutoff.
    With automatic scales and the same graph/fuzz, detections are a subset of
    classify_boundary_threshold's detections.

    Args:
        X: finite (N,d) coordinates. No PCA or dimension reduction.
        eps_vals: optional shared positive epsilon grid; sorted and deduplicated.
            Otherwise use the same adaptive grid as compute_b_vals/cutoff.
        n_scales: number of smallest scales to fit (default 6, at least 3).
            None fits all available scales, which may include nonlocal scales.
        slope_threshold: upper accepted slope (default 0.25, the midpoint
            between ideal boundary 0 and interior 0.5). None also uses 0.25.
        min_slope: lower accepted slope (default 0). A small negative value
            permits finite-scale downward drift, but weakens saturation rejection.
        b_floor: scores at or below this numerical floor cannot be log-fitted;
            they are not clamped to create a spurious flat curve.
        use_knn: use a symmetric sparse graph; k defaults to min(N,256).
        batch_size: maximum query rows in the weight arrays.
        return_slopes: return (labels, slopes).
        return_diagnostics: return (labels, details), taking precedence over
            return_slopes. raw_slopes now directly determine classification.
        fuzz: amplitude tolerance, exactly as in classify_boundary_threshold.
            Default 0.1 gives about 0.4642. No negative fuzz is needed by the rule;
            negative values remain supported for comparison with the baseline.
        max_log_residual: maximum RMS residual in log(b), default 0.1.
            This checks approximate power-law behavior, not statistical confidence.
        **kwargs: k when use_knn=True.

    Returns:
        Integer labels: boundary candidate=1, no boundary evidence=0.
        Diagnostic status is 'boundary', 'no_boundary_evidence',
        'insufficient_scales', or 'model_mismatch'. rejection_reason identifies
        the failed criterion. Details also contain raw_slopes, max_b,
        log_fit_residual, eps_vals, h_min, h_max, and amplitude_threshold.

    Limitations:
        This is a conservative amplitude-and-scaling heuristic, not a proof of
        manifold boundary membership. Singularities and density effects can
        also produce a strong plateau. A large negative fuzz selects extreme
        displacements in image space, rather than the ideal half-space limit.
        The nonnegative-slope requirement may miss true boundaries with negative
        finite-scale curvature/density corrections. kNN truncation,
        self-dominated small scales, and nonlocal large scales
        can distort slopes. A narrow window can also hide a curved score profile.
        Parameters from the former vector model (boundary_band, min_snr, and
        effective-neighbor limits) are no longer used or accepted.
    """
    _validate_batch_size(batch_size)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.size or not np.all(np.isfinite(X)):
        raise ValueError("X must be a nonempty, finite 2D array")
    if n_scales is not None and (
            not isinstance(n_scales, (int, np.integer)) or n_scales < 3):
        raise ValueError("n_scales must be None or an integer of at least 3")
    if slope_threshold is None:
        slope_threshold = 0.25
    if not np.isfinite(slope_threshold) or slope_threshold < 0:
        raise ValueError("slope_threshold must be finite and nonnegative")
    if not np.isfinite(min_slope) or min_slope > slope_threshold:
        raise ValueError("min_slope must be finite and <= slope_threshold")
    if not np.isfinite(b_floor) or b_floor < 0:
        raise ValueError("b_floor must be finite and nonnegative")
    if not np.isfinite(fuzz):
        raise ValueError("fuzz must be finite")
    if not np.isfinite(max_log_residual) or max_log_residual < 0:
        raise ValueError("max_log_residual must be finite and nonnegative")
    if eps_vals is not None:
        eps_vals = np.asarray(eps_vals, dtype=float)
        if (eps_vals.ndim != 1 or not np.all(np.isfinite(eps_vals))
                or np.any(eps_vals <= 0)):
            raise ValueError("eps_vals must be finite positive values in a 1D array")
        eps_vals = np.unique(eps_vals)
    k = kwargs.pop("k", min(len(X), 256))
    if kwargs:
        raise TypeError("Unexpected arguments: " + ", ".join(kwargs))
    if use_knn and (not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X)):
        raise ValueError("k must be an integer between 1 and N")

    eps, scores = _evaluate_boundary(
        X, eps_vals, use_knn, batch_size, n_scales=n_scales,
        **({"k": k} if use_knn else {}))
    N = len(X)
    threshold = 1.0 / np.sqrt(np.pi) - fuzz
    slopes = np.full(N, np.nan)
    residual = np.full(N, np.nan)
    max_b = scores.max(axis=1) if scores.shape[1] else np.full(N, np.nan)
    labels = np.zeros(N, dtype=int)
    status = np.full(N, "insufficient_scales", dtype="<U24")
    reason = np.full(N, "too_few_scales", dtype="<U32")
    if len(eps) >= 3:
        valid = np.all(np.isfinite(scores) & (scores > b_floor), axis=1)
        reason[~valid] = "below_score_floor"
        if np.any(valid):
            log_eps = np.log(eps)
            centered = log_eps - log_eps.mean()
            log_b = np.log(scores[valid])
            slopes[valid] = log_b @ centered / (centered @ centered)
            fitted = log_b.mean(axis=1, keepdims=True) + slopes[valid, None] * centered
            residual[valid] = np.sqrt(np.mean((log_b - fitted) ** 2, axis=1))
            status[valid] = "no_boundary_evidence"
            strong = valid & (max_b > threshold)
            # Only a roundoff allowance at the interval endpoints, not an
            # empirical negative-slope tolerance.
            flat = (strong & (slopes >= min_slope - 1e-12)
                    & (slopes <= slope_threshold + 1e-12))
            accepted = flat & (residual <= max_log_residual)
            reason[valid] = "below_amplitude_cutoff"
            reason[strong] = "slope_outside_plateau"
            reason[flat] = "nonlinear_log_curve"
            status[flat & ~accepted] = "model_mismatch"
            labels[accepted] = 1
            status[accepted] = "boundary"
            reason[accepted] = "accepted"

    if return_diagnostics:
        return labels, {
            "raw_slopes": slopes, "log_fit_residual": residual, "max_b": max_b,
            "status": status, "rejection_reason": reason, "eps_vals": eps,
            "h_min": np.full(N, np.sqrt(eps[0]) if len(eps) else np.nan),
            "h_max": np.full(N, np.sqrt(eps[-1]) if len(eps) else np.nan),
            "amplitude_threshold": threshold,
        }
    if return_slopes:
        return labels, slopes
    return labels
