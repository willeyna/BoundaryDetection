import numpy as np
from scipy.spatial.distance import cdist

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
    Sample points uniformly from the surface of a cone (z = sqrt(x^2 + y^2), z in [0, 1]).

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
    z = r
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

def choose_adaptive_epsilons(X, num_eps=30, k=None, gamma=0.2):
    """
    Adaptively chooses a log-spaced range of epsilons based on local density and global diameter.

    Args:
        X: (N, d) array of coordinates
        num_eps: number of epsilons to return
        k: number of nearest neighbors for density estimation (default: max(5, ln(N)))
        gamma: locality fraction for upper bound

    Returns:
        eps_vals: (num_eps,) array of epsilons
    """
    N = X.shape[0]
    if N < 2:
        return np.array([1.0] * num_eps)

    # 1. Compute pairwise distances
    dists = cdist(X, X, 'euclidean')

    # 2. Lower bound: k-th nearest neighbor distance squared
    if k is None:
        k = max(5, int(np.floor(np.log(N))))
    k = min(k, N - 1)
    
    sorted_dists = np.partition(dists, k, axis=1)
    k_th_dists = sorted_dists[:, k]
    eps_min = np.median(k_th_dists) ** 2
    eps_min = max(eps_min, 1e-8)  # prevent exact zero

    # 3. Upper bound: square of (gamma * diameter)
    diameter = np.max(dists)
    eps_max = (gamma * diameter) ** 2
    
    # Ensure eps_max > eps_min
    if eps_max <= eps_min:
        eps_max = eps_min * 10.0

    return np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)

def compute_b_vals(X, eps_vals=None):
    """
    Computes the normalized boundary constant b(eps) for all points over a range of epsilons.

    Args:
        X: (N, d) array of coordinates
        eps_vals: optional array of epsilon values. If None, chosen adaptively.

    Returns:
        b_vals: (N, len(eps_vals)) array where b_vals[i, j] is the boundary constant for point i at eps_vals[j]
    """
    if eps_vals is None:
        eps_vals = choose_adaptive_epsilons(X)
    N = X.shape[0]
    b_vals = np.zeros((N, len(eps_vals)))
    for j, eps in enumerate(eps_vals):
        L = compute_laplacian(X, eps)
        v = L @ X
        b_vals[:, j] = np.linalg.norm(v, axis=1) / np.sqrt(eps)
    return b_vals

def classify_boundary_concavity(X, margin=1e-4, min_val_ratio=0.1):
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
    eps_vals = choose_adaptive_epsilons(X)
    b_vals = compute_b_vals(X, eps_vals)
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

def classify_boundary_threshold(X, fuzz=0.1):
    """
    Classify points as boundary (1) or interior (0) by checking if b(eps) 
    ever goes above the theoretical boundary limit.

    Args:
        X: (N, d) array of coordinates
        fuzz: safety margin subtracted from theoretical limit (default: 0.1)

    Returns:
        classes: (N,) array of 0s and 1s
    """
    b_vals = compute_b_vals(X)
    limit = 1.0 / np.sqrt(np.pi) - fuzz
    
    # Check if the maximum b over the epsilons exceeds the limit
    is_boundary = np.max(b_vals, axis=1) > limit
    return is_boundary.astype(int)
