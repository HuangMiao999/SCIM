#------adapted from dynamo

import numpy as np
import numpy.matlib
from sklearn.neighbors import NearestNeighbors
from scipy.linalg.blas import dgemm
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist


def con_K(x, y, beta, method="cdist", return_d=False):
    """con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).
    Arguments
    ---------
        x: :class:`~numpy.ndarray`
            Original training data points.
        y: :class:`~numpy.ndarray`
            Control points used to build kernel basis functions.
        beta: float (default: 0.1)
            Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),
        return_d: bool
            If True the intermediate 3D matrix x - y will be returned for analytical Jacobian.
    Returns
    -------
    K: :class:`~numpy.ndarray`
    the kernel to represent the vector field function.
    """
    if method == "cdist" and not return_d:
        K = cdist(x, y, "sqeuclidean")
        if len(K) == 1:
            K = K.flatten()
    else:
        n = x.shape[0]
        m = y.shape[0]

        # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
        # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
        D = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
        K = np.squeeze(np.sum(D**2, 1))
    K = -beta * K
    K = np.exp(K)

    if return_d:
        return K, D
    else:
        return K



def bandwidth_selector(X):
    """
    This function computes an empirical bandwidth for a Gaussian kernel.
    """
    n, m = X.shape
    if n > 200000 and m > 2:
        from pynndescent import NNDescent

        nbrs = NNDescent(
            X,
            metric="euclidean",
            n_neighbors=max(2, int(0.2 * n)),
            n_jobs=-1,
            random_state=19491001,
        )
        _, distances = nbrs.query(X, k=max(2, int(0.2 * n)))
    else:
        alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
        nbrs = NearestNeighbors(n_neighbors=max(2, int(0.2 * n)), algorithm=alg, n_jobs=-1).fit(X)
        distances, _ = nbrs.kneighbors(X)

    d = np.mean(distances[:, 1:]) / 1.5
    return np.sqrt(2) * d



def get_P(Y, V, sigma2, gamma, a, div_cur_free_kernels=False):
    """GET_P estimates the posterior probability and part of the energy.
    Arguments
    ---------
        Y: 'np.ndarray'
            Velocities from the data.
        V: 'np.ndarray'
            The estimated velocity: V=f(X), f being the vector field function.
        sigma2: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is a.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the
            vector field.
    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.
    """

    if div_cur_free_kernels:
        Y = Y.reshape((2, int(Y.shape[0] / 2)), order="F").T
        V = V.reshape((2, int(V.shape[0] / 2)), order="F").T

    D = Y.shape[1]
    temp1 = np.exp(-np.sum((Y - V) ** 2, 1) / (2 * sigma2))
    temp2 = (2 * np.pi * sigma2) ** (D / 2) * (1 - gamma) / (gamma * a)
    temp1[temp1 == 0] = np.min(temp1[temp1 != 0])
    P = temp1 / (temp1 + temp2)
    E = P.T.dot(np.sum((Y - V) ** 2, 1)) / (2 * sigma2) + np.sum(P) * np.log(sigma2) * D / 2

    return (P[:, None], E) if P.ndim == 1 else (P, E)



def linear_least_squares(a, b, residuals=False):
    """
    Return the least-squares solution to a linear matrix equation.
    Solves the equation `a x = b` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `a` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `a`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.
    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : (M,) array_like
        Ordinate or "dependent variable" values.
    residuals : bool
        Compute the residuals associated with the least-squares solution
    Returns
    -------
    x : (M,) ndarray
        Least-squares solution. The shape of `x` depends on the shape of
        `b`.
    residuals : int (Optional)
        Sums of residuals; squared Euclidean 2-norm for each column in
        ``b - a*x``.
    """
    if type(a) != np.ndarray or not a.flags["C_CONTIGUOUS"]:
        main_warning(
            "Matrix a is not a C-contiguous numpy array. The solver will create a copy, which will result"
            + " in increased memory usage."
        )

    a = np.asarray(a, order="c")
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b))

    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x



def lstsq_solver(lhs, rhs, method="drouin"):
    if method == "scipy":
        C = lstsq(lhs, rhs)[0]
    elif method == "drouin":
        C = linear_least_squares(lhs, rhs)
    else:
        main_warning("Invalid linear least squares solver. Use Drouin's method instead.")
        C = linear_least_squares(lhs, rhs)
    return C



def SparseVFC(
    X: np.ndarray,
    Y: np.ndarray,
    Grid: np.ndarray,
    M: int = 100,
    a: float = 5,
    beta: float = None,
    ecr: float = 1e-5,
    gamma: float = 0.9,
    lambda_: float = 3,
    minP: float = 1e-5,
    MaxIter: int = 500,
    theta: float = 0.75,
    div_cur_free_kernels: bool = False,
#     velocity_based_sampling: bool = True,
    sigma: float = 0.8,
    eta: float = 0.5,
    seed=0,
    lstsq_method: str = "drouin",
    verbose: int = 1,
) -> dict:
    """Apply sparseVFC (vector field consensus) algorithm to learn a functional form of the vector field from random
    samples with outlier on the entire space robustly and efficiently. (Ma, Jiayi, etc. al, Pattern Recognition, 2013)
    Arguments
    ---------
        X: 'np.ndarray'
            Current state. This corresponds to, for example, the spliced transcriptomic state.
        Y: 'np.ndarray'
            Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic
            velocity or total RNA velocity based on metabolic labeling data estimated calculated by dynamo.
        Grid: 'np.ndarray'
            Current state on a grid which is often used to visualize the vector field. This corresponds to, for example,
            the spliced transcriptomic state or total RNA state.
        M: 'int' (default: 100)
            The number of basis functions to approximate the vector field.
        a: 'float' (default: 10)
            Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is `a`.
        beta: 'float' (default: 0.1)
            Parameter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
            If None, a rule-of-thumb bandwidth will be computed automatically.
        ecr: 'float' (default: 1e-5)
            The minimum limitation of energy change rate in the iteration process.
        gamma: 'float' (default: 0.9)
            Percentage of inliers in the samples. This is an initial value for EM iteration, and it is not important.
        lambda_: 'float' (default: 3)
            Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
            weights on regularization.
        minP: 'float' (default: 1e-5)
            The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as
            minP.
        MaxIter: 'int' (default: 500)
            Maximum iteration times.
        theta: 'float' (default: 0.75)
            Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta,
            then it is regarded as an inlier.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the
            vector field.
        sigma: 'int' (default: `0.8`)
            Bandwidth parameter.
        eta: 'int' (default: `0.5`)
            Combination coefficient for the divergence-free or the curl-free kernels.
        seed : int or 1-d array_like, optional (default: `0`)
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points.
            Default is to be 0 for ensure consistency between different runs.
        lstsq_method: 'str' (default: `drouin`)
           The name of the linear least square solver, can be either 'scipy` or `douin`.
        verbose: `int` (default: `1`)
            The level of printing running information.
    Returns
    -------
    VecFld: 'dict'
        A dictionary which contains:
            X: Current state.
            valid_ind: The indices of cells that have finite velocity values.
            X_ctrl: Sample control points of current state.
            ctrl_idx: Indices for the sampled control points.
            Y: Velocity estimates in delta t.
            beta: Parameter of the Gaussian Kernel for the kernel matrix (Gram matrix).
            V: Prediction of velocity of X.
            C: Finite set of the coefficients for the
            P: Posterior probability Matrix of inliers.
            VFCIndex: Indexes of inliers found by sparseVFC.
            sigma2: Energy change rate.
            grid: Grid of current state.
            grid_V: Prediction of velocity of the grid.
            iteration: Number of the last iteration.
            tecr_vec: Vector of relative energy changes rate comparing to previous step.
            E_traj: Vector of energy at each iteration,
        where V = f(X), P is the posterior probability and VFCIndex is the indexes of inliers found by sparseVFC.
        Note that V = `con_K(Grid, X_ctrl, beta).dot(C)` gives the prediction of velocity on Grid (but can also be any
        point in the gene expression state space).
    """
#     logger = LoggerManager.gen_logger("SparseVFC")
#     temp_logger = LoggerManager.get_temp_timer_logger()
#     logger.info("[SparseVFC] begins...")
#     logger.log_time()

    need_utility_time_measure = verbose > 1
    X_ori, Y_ori = X.copy(), Y.copy()
    valid_ind=np.arange(Y.shape[0])
    N, D = Y.shape
    grid_U = None

    # Construct kernel matrix K
    tmp_X, uid = np.unique(X, axis=0, return_index=True)  # return unique rows
    M = min(M, tmp_X.shape[0])
#     if velocity_based_sampling:
# #         logger.info("Sampling control points based on data velocity magnitude...")
#         idx = sample_by_velocity(Y[uid], M, seed=seed)
#     else:
    idx = np.random.RandomState(seed=seed).permutation(tmp_X.shape[0])  # rand select some initial points
    idx = idx[range(M)]
    ctrl_pts = tmp_X[idx, :]

    if beta is None:
        h = bandwidth_selector(ctrl_pts)
        beta = 1 / h**2

    K = (
        con_K(ctrl_pts, ctrl_pts, beta)#, timeit=need_utility_time_measure)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(ctrl_pts, ctrl_pts, sigma, eta)[0]
    )
    U = (
        con_K(X, ctrl_pts, beta)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(X, ctrl_pts, sigma, eta)[0]
    )
    if Grid is not None:
        grid_U = (
            con_K(Grid, ctrl_pts, beta)
            if div_cur_free_kernels is False
            else con_K_div_cur_free(Grid, ctrl_pts, sigma, eta)[0]
        )
    M = ctrl_pts.shape[0] * D if div_cur_free_kernels else ctrl_pts.shape[0]

    if div_cur_free_kernels:
        X = X.flatten()[:, None]
        Y = Y.flatten()[:, None]

    # Initialization
    V = X.copy() if div_cur_free_kernels else np.zeros((N, D))
    C = np.zeros((M, 1)) if div_cur_free_kernels else np.zeros((M, D))
    i, tecr, E = 0, 1, 1
    # test this
    sigma2 = sum(sum((Y - X) ** 2)) / (N * D) if div_cur_free_kernels else sum(sum((Y - V) ** 2)) / (N * D)
    sigma2 = 1e-7 if sigma2 < 1e-8 else sigma2
    tecr_vec = np.ones(MaxIter) * np.nan
    E_vec = np.ones(MaxIter) * np.nan
    P = None
    while i < MaxIter and tecr > ecr and sigma2 > 1e-8:
        # E_step
        E_old = E
        P, E = get_P(Y, V, sigma2, gamma, a, div_cur_free_kernels)

        E = E + lambda_ / 2 * np.trace(C.T.dot(K).dot(C))
        E_vec[i] = E
        tecr = abs((E - E_old) / E)
        tecr_vec[i] = tecr

        # logger.report_progress(count=i, total=MaxIter, progress_name="E-step iteration")
#         if need_utility_time_measure:
#             logger.info(
#                 "iterate: %d, gamma: %.3f, energy change rate: %s, sigma2=%s"
#                 % (i, gamma, scinot(tecr, 3), scinot(sigma2, 3))
#             )

        # M-step. Solve linear system for C.
#         temp_logger.log_time()
        P = np.maximum(P, minP)
        if div_cur_free_kernels:
            P = np.kron(P, np.ones((int(U.shape[0] / P.shape[0]), 1)))  # np.kron(P, np.ones((D, 1)))
            lhs = (U.T * numpy.matlib.tile(P.T, [M, 1])).dot(U) + lambda_ * sigma2 * K
            rhs = (U.T * numpy.matlib.tile(P.T, [M, 1])).dot(Y)
        else:
            UP = U.T * numpy.matlib.repmat(P.T, M, 1)
            lhs = UP.dot(U) + lambda_ * sigma2 * K
            rhs = UP.dot(Y)
#         if need_utility_time_measure:
#             temp_logger.finish_progress(progress_name="computing lhs and rhs")
#         temp_logger.log_time()

        C = lstsq_solver(lhs, rhs, method=lstsq_method)#, timeit=need_utility_time_measure

        # Update V and sigma**2
        V = U.dot(C)
        Sp = sum(P) / 2 if div_cur_free_kernels else sum(P)
        sigma2 = (sum(P.T.dot(np.sum((Y - V) ** 2, 1))) / np.dot(Sp, D))#[0]

        # Update gamma
        numcorr = len(np.where(P > theta)[0])
        gamma = numcorr / X.shape[0]

        if gamma > 0.95:
            gamma = 0.95
        elif gamma < 0.05:
            gamma = 0.05

        i += 1
    if i == 0 and not (tecr > ecr and sigma2 > 1e-8):
        raise Exception(
            "please check your input parameters, "
            f"tecr: {tecr}, ecr {ecr} and sigma2 {sigma2},"
            f"tecr must larger than ecr and sigma2 must larger than 1e-8"
        )

    grid_V = None
    if Grid is not None:
        grid_V = np.dot(grid_U, C)

    VecFld = {
        "X": X_ori,
        "valid_ind": valid_ind,
        "X_ctrl": ctrl_pts,
        "ctrl_idx": idx,
        "Y": Y_ori,
        "beta": beta,
        "V": V.reshape((N, D)) if div_cur_free_kernels else V,
        "C": C,
        "P": P,
        "VFCIndex": np.where(P > theta)[0],
        "sigma2": sigma2,
        "grid": Grid,
        "grid_V": grid_V,
        "iteration": i - 1,
        "tecr_traj": tecr_vec[:i],
        "E_traj": E_vec[:i],
    }
    if div_cur_free_kernels:
        VecFld["div_cur_free_kernels"], VecFld["sigma"], VecFld["eta"] = (
            True,
            sigma,
            eta,
        )
#         temp_logger.log_time()
#         (
#             _,
#             VecFld["df_kernel"],
#             VecFld["cf_kernel"],
#         ) = con_K_div_cur_free(X, ctrl_pts, sigma, eta, timeit=need_utility_time_measure)
#         temp_logger.finish_progress(progress_name="con_K_div_cur_free")

#     logger.finish_progress(progress_name="SparseVFC")
    return VecFld



def Jacobian_rkhs_gaussian(x, vf_dict,vectorize=False):
    """analytical Jacobian for RKHS vector field functions with Gaussian kernel.
    Arguments
    ---------
    x: :class:`~numpy.ndarray`
        Coordinates where the Jacobian is evaluated.
    vf_dict: dict
        A dictionary containing RKHS vector field control points, Gaussian bandwidth,
        and RKHS coefficients.
        Essential keys: 'X_ctrl', 'beta', 'C'
    Returns
    -------
    J: :class:`~numpy.ndarray`
        Jacobian matrices stored as d-by-d-by-n numpy arrays evaluated at x.
        d is the number of dimensions and n the number of coordinates in x.
    """
    if x.ndim == 1:
        K, D = con_K(x[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        J = (vf_dict["C"].T * K) @ D[0].T
    elif not vectorize:
        n, d = x.shape
        d0 = vf_dict['C'].shape[1]
        J = np.zeros((d0, d, n))
        for i, xi in enumerate(x):
            K, D = con_K(xi[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
            J[:, :, i] = (vf_dict["C"].T * K) @ D[0].T
    else:
        K, D = con_K(x, vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        if K.ndim == 1:
            K = K[None, :]
        J = np.einsum("nm, mi, njm -> ijn", K, vf_dict["C"], D)

    return -2 * vf_dict["beta"] * J