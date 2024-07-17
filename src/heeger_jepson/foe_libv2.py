# See egomotion.ipynb for exploratory introduction to this code
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from .motionfield import *
from functools import partial
from typing import Tuple
from jaxtyping import Array, Float, PyTree, Integer, jaxtyped, ScalarLike
from typeguard import typechecked as typechecker
import matplotlib.pyplot as plt
from typing import Union


@jaxtyped(typechecker=typechecker)
def generate_random_points_on_positive_hemisphere(
    num_points: int,
) -> Float[Array, "3 num_points"]:
    """Generate num_points random points on the positive hemisphere in R^3
    See: https://dornsife.usc.edu/sergey-lototsky/wp-content/uploads/sites/211/2023/06/UniformOnTheSphere.pdf
    Parameters
    ------------
    num_points : int
        Number of points to generate
    Returns
    --------
    points : ndarray (3, num_points)
        Random points on the positive hemisphere
    """
    points = np.zeros((num_points, 3))
    for i in range(num_points):
        # Generate random point on positive hemisphere using spherical coordinates
        phi = np.random.uniform(0, 2 * np.pi)  # azimuthal angle
        u = np.random.uniform()  # random value between 0 and 1
        theta = np.arccos(u)  # polar angle
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points[i] = [x, y, z]
    return jnp.vstack([points[:, 0], points[:, 1], points[:, 2]])


@partial(jax.jit, static_argnames=["num_samples"])
@jaxtyped(typechecker=typechecker)
def sample_points(
    mask_in: Float[Array, "H W"],
    flow: Float[Array, "H W 2"],
    num_samples: Union[int, Integer[ScalarLike, ""]],
    k_size: Union[int, Integer[ScalarLike, ""]] = 21,
) -> Tuple[Integer[Array, "3 num_samples"], Float[Array, "num_samples 2"]]:
    """
    Samples Flow Points based of valid mask
    Samples without replacement based on the density of point in an area
    Parameters
    ------------
    mask_in : ndarray (H, W)
        Mask of valid points
    flow : ndarray (H, W, 2)
        Flow Field
    num_samples : int
        Number of flow vectors to use
    k_size : int
        Kernel Size for density estimation
    """
    mask = jnp.array(mask_in).astype(jnp.float32)
    kernel = jnp.ones((1, 1, k_size, k_size))

    xval = jnp.arange(flow.shape[1])
    yval = jnp.arange(flow.shape[0])
    xx, yy = jnp.meshgrid(xval, yval)

    # Generate number of valid pixels in k_size x k_size region
    cnt_img = lax.conv(
        mask[None, None, :, :],  # lhs = NCHW image tensor
        kernel,  # rhs = OIHW conv kernel tensor
        (1, 1),  # window strides
        "SAME",  # padding mode
    )
    # Use this count to create a probability distribution
    mask_cnt_img = cnt_img * mask
    prob_out = mask_cnt_img.reshape(-1) / jnp.sum(mask_cnt_img)
    # Select points based on the count probability distribution

    rnd_pts = jax.random.choice(
        jax.random.PRNGKey(0),
        jnp.arange(len(prob_out)),
        shape=[num_samples],
        p=prob_out,
        replace=False,
    )
    # Output points selected
    x_sel = xx.reshape(-1)[rnd_pts]
    y_sel = yy.reshape(-1)[rnd_pts]
    pts = jnp.stack([x_sel, y_sel, jnp.ones_like(x_sel)])
    return pts, flow[y_sel, x_sel]


@jaxtyped(typechecker=typechecker)
def coarse_to_fine(
    V: Float[Array, "num_samples 2"],
    cam_pts: Integer[Array, "3 num_samples"],
    K: Float[Array, "3 3"],
    f: Union[float, Float[ScalarLike, ""]],
    res: Tuple[
        Union[int, Integer[ScalarLike, ""]], Union[int, Integer[ScalarLike, ""]]
    ],
    T_search_coarse: Float[Array, "3 num_cand_points"] = jnp.zeros((3, 0)),
    pixel_search: Union[int, Integer[ScalarLike, ""]] = 1000,
    width_search: Union[int, Integer[ScalarLike, ""]] = 50,
    coarse_num_dirs: Union[int, Integer[ScalarLike, ""]] = 2000,
) -> Tuple[
    Tuple[
        Float[Array, "3 num_cand_points_ret"],
        Float[Array, "3"],
        Float[Array, "3"],
        Float[Array, "num_cand_points_ret"],
    ],
    Tuple[
        Float[Array, "3 num_fine_cand_points"],
        Float[Array, "3"],
        Float[Array, "3"],
        Float[Array, "num_fine_cand_points"],
    ],
]:
    """Coarse to Fine Search
    Parameters
    ------------
    V : ndarray (num_samples,2),
        Flow Field
    cam_pts : ndarray (3, num_samples)
        Camera Points
    K : jaxarray (3 ,3)
        Intrinsics Matrix
    f : float
        Focal Length
    res : tuple
        Resolution of the image
    T_search_coarse : ndarray (3, num_cand_points)
        Candidate Translation Directions
    pixel_search : int
        Number of pixels to search
    width_search : int
        Width of search
    coarse_num_dirs : int
        Number of coarse directions to search
    Returns
    --------
    This is repeated for the coarse and find points
    T_search : ndarray (3 num_cand_points)
        Translation Vectors that are searched over
    T_min : ndarray (3)
        Minimum Translation out T_search
    Omega_min : ndarray (3)
        Minimum Rotation based on T_search
    tot_res : ndarray (num_cand_points)
        Total Residuals for each flow vector
    """
    if T_search_coarse.shape[1] == 0:
        T_search_coarse = generate_random_points_on_positive_hemisphere(coarse_num_dirs)
    T_min_coarse, Omega_min_coarse, tot_res_coarse = heeger_jepson_RT(
        V,
        cam_pts,
        K,
        f,
        res,
        T_search_coarse,
    )

    min_pix = K @ T_min_coarse
    min_pix = min_pix / min_pix[2]
    x_pt = jnp.linspace(
        min_pix[0].astype(int) - width_search,
        min_pix[0].astype(int) + width_search,
        pixel_search,
    )
    y_pt = jnp.linspace(
        min_pix[1].astype(int) - width_search,
        min_pix[1].astype(int) + width_search,
        pixel_search,
    )
    nx, ny = jnp.meshgrid(x_pt, y_pt)
    T_dir_un = jnp.linalg.inv(K) @ jnp.vstack(
        [
            nx.reshape(-1),
            ny.reshape(-1),
            jnp.ones(pixel_search * pixel_search),
        ]
    )
    T_search_fine = T_dir_un

    T_min_fine, Omega_min_fine, tot_res_fine = heeger_jepson_RT(
        V,
        cam_pts,
        K,
        f,
        res,
        T_search_fine,
    )

    return (T_search_coarse, T_min_coarse, Omega_min_coarse, tot_res_coarse), (
        T_search_fine,
        T_min_fine,
        Omega_min_fine,
        tot_res_fine,
    )


@jax.jit
@jaxtyped(typechecker=typechecker)
def heeger_jepson_RT(
    V: Float[Array, "num_samples 2"],
    cam_pts: Integer[Array, "3 num_samples"],
    K: Float[Array, "3 3"],
    f: Union[float, Float[ScalarLike, ""]],
    res: Tuple[
        Union[int, Integer[ScalarLike, ""]], Union[int, Integer[ScalarLike, ""]]
    ],
    T_search: Float[Array, "3 num_cand_points"],
) -> Tuple[Float[Array, "3"], Float[Array, "3"], Float[Array, "num_cand_points"]]:
    """Heeger Jepson Algorithm
    https://www.cs.toronto.edu/~jepson/papers/HeegerJepsonJCV1992.pdf
    Parameters
    ------------
    V : jaxarray (num_samples,2),
        Flow Field, normalized coordinates unit, not pixels
    cam_pts : jaxarray (3, num_samples)
        Camera Points
    K : jaxarray (3 ,3)
        Intrinsics Matrix
    f : float
        Focal Length
    T_search : jaxarray (3, num_cand_points)
        Candidate Translation Directions
    res : tuple
        Resolution of the image
    Returns
    --------
    T_min : jaxarray (3)
        Minimum Translation out T_search
    Omega_min : jaxarray (3)
        Minimum Rotation based on T_search
    tot_res : jaxarray (num_cand_points)
        Total Residuals for each flow vector
    """

    # Normalize Camera Points + Flow

    K_inv = jnp.linalg.inv(K)

    norm_cords = K_inv @ cam_pts  # Normalized Camera Points (3,num_samples)

    # Create B Matrix for Norm Camera Points, note f=1 with norm cords
    # vmap over all camera points 1 dimension to create a different B matrix for each
    create_B = jax.vmap(create_B_matrix, in_axes=(1, None), out_axes=0)
    B = create_B(norm_cords, 1.0)  # (res[0] * res[1], 2, 3)

    # Going to calculate terms for:
    # min_T || A_perp(T)B Omega_LSQ -A_perp(T)V)||_2^2

    # Creat A_perp Matrix, map over norm camera points, note f=1 with norm cords
    # vamp over 1st dimension, whihc is all camera points
    create_A_perp = jax.vmap(create_A_perp_matrix, in_axes=(1, None), out_axes=0)
    A_perp = create_A_perp(norm_cords, 1.0)  # (Pts, 2, 3)

    # Calculate A_perp(T)=(JAT/|JAT|)
    A_perp_T = (
        A_perp @ T_search
    )  #  A_perp (# pts, 2, 3), T: (3, # Dirs), Out: (# pts, 2, # Dirs)
    tmp_rearange = A_perp_T.transpose(1, 0, 2)  # A_perp_T(Rearanged) (2, # pts, # Dirs)
    A_perp_T_norm = tmp_rearange / (
        jnp.linalg.norm(tmp_rearange, axis=0)[None, :, :] + 1e-10
    )  # A_perp_T_norm (2, # pts, Dirs)

    # Calculate ther term A_perp(T)V, we do with broadcasted multipication + sum
    A_perp_T_V = jnp.sum(
        A_perp_T_norm.transpose(1, 0, 2) * V[:, :, None], axis=1
    )  # (#pts,2,#Dirs) x (#pts,2,1)= \sum_a1 (#pts,2,#Dirs)= (#pts,#Dirs)

    # Calculate ther term A_perp(T)B, we do with broadcasted multipication + sum
    A_perp_T_B = jnp.sum(
        A_perp_T_norm.transpose(1, 0, 2)[:, :, None, :] * B[:, :, :, None], axis=1
    )  # (#pts,2,1,#Dirs) x (#pts,2,3, 1)= \sum_a1 (#pts,2,3,#Dirs)= (#pts,3, #Dirs)

    # Solve LSQ Problem: A_perp(T)B Omega_LSQ = A_perp(T)V, solve for Omega_LSQ
    # Going to calculat the psudeoinverse, by first mutiplicting both sides (A_perp(T)B )^T
    # First going to compute the inverse of the LHS: ((A_perp(T)B)^T A_perp(T)B)^-1
    # vmap over each of the searched directions 2nd dimension
    inv_func = jax.vmap(calc_inverse, in_axes=(2))
    A_perp_T_B_inv = inv_func(A_perp_T_B)  # (#pts,3, #Dirs) -> (#Dirs,3,3)

    # Additionally, mutiply RHS by (A_perp(T)B)^T
    # A_perp_T_B:(#pts,3,#Dirs),
    # A_perp_T_V:(#pts,#Dirs),
    # A_perp_T_B_A_perp_T_V: (3, #Dirs)
    A_perp_T_B_A_perp_T_V = jax.lax.batch_matmul(
        A_perp_T_B.transpose(2, 1, 0), A_perp_T_V.T[:, :, None]
    )[:, :, 0].T

    # Calculate Omega_LSQ:  Omega_LSQ = ((A_perp(T)B)^T A_perp(T)B)^-1 (A_perp(T)B)^T A_perp(T)V
    # A_perp_T_B_inv: (#Dirs,3,3),
    # A_perp_T_B_A_perp_T_V: (3, #Dirs)
    # Omega: (#Dirs,3)
    Omega = jax.lax.batch_matmul(A_perp_T_B_inv, A_perp_T_B_A_perp_T_V.T[:, :, None])[
        :, :, 0
    ]

    # Calculate the first term: || A_perp(T)B Omega_LSQ -A_perp(T)V)||_2^2
    # A_perp_T_B_inv: (#Dirs,3,3),
    # A_perp_T_B_A_perp_T_V: (3, #Dirs)
    # Omega: (#Dirs,3)
    A_perp_B_omega = jnp.sum(A_perp_T_B * Omega[:, :, None].T, axis=1)

    # Calculate the full residual given computations above
    flow_res = (A_perp_B_omega - A_perp_T_V) ** 2
    tot_res_per_direction = jnp.sum(flow_res, axis=0)  # Sum all this over all points

    # Find the minimum residual, and output min vals
    min_idx = jnp.argmin(tot_res_per_direction)
    T_min = T_search[:, min_idx]
    Omega_min = Omega[min_idx]
    return T_min, Omega_min, tot_res_per_direction


@jax.jit
@jaxtyped(typechecker=typechecker)
def get_inv_depth(
    V: Float[Array, "num_samples 2"],
    cam_pts: Integer[Array, "3 num_samples"],
    K: Float[Array, "3 3"],
    f: Union[float, Float[ScalarLike, ""]],
    res: Tuple[
        Union[int, Integer[ScalarLike, ""]], Union[int, Integer[ScalarLike, ""]]
    ],
    Omega_min: Float[Array, "3"],
    T_min: Float[Array, "3"],
) -> Float[Array, "num_samples"]:
    """Calculate Inverse Depth
    Parameters
    ------------
    V : ndarray (num_samples,2),
        Flow Field
    cam_pts : ndarray (3, num_samples)
        Camera Points
    K : jaxarray (3 ,3)
        Intrinsics Matrix
    f : float
        Focal Length
    Omega_min : ndarray (3)
        Minimum Rotation based on T_search
    res : tuple
        Resolution of the image
    T_min : ndarray (3)
        Minimum Translation out T_search
    Returns
    --------
    inv_depth : ndarray (num_samples)
        Inverse Depth
    """

    # In this function, recompute many of the values as we may pass in new points not in original function
    K_inv = jnp.linalg.inv(K)

    norm_cords = K_inv @ cam_pts  # Normalized Camera Points (3,C)

    # Create A Matrix for Norm Camera Points, note f=1 with norm cords
    # vmap over all camera points 1 dimension to create a different A matrix for each
    create_A = jax.vmap(create_A_matrix, in_axes=(1, None), out_axes=0)
    A = create_A(norm_cords, 1.0)  # A: (res[0] * res[1], 2, 3)

    # Create B Matrix for Norm Camera Points, note f=1 with norm cords
    # vmap over all camera points 1 dimension to create a different B matrix for each
    create_B = jax.vmap(create_B_matrix, in_axes=(1, None), out_axes=0)
    B = create_B(norm_cords, 1.0)  # (res[0] * res[1], 2, 3)

    # ||(1/z)AT+BOmega||_2^2 -> v-BOmega= (1/z)A(T)
    # Solve: A^T(T)(v-BOmega) =A^T(T)A(T) (1/z)
    # Solve: (A^T(T)(v-BOmega))/(A^T(T)A(T)) = (1/z)

    # rot_corr_flow: (C,2)
    # B: (C,2,3)
    # Omega_min: (3)
    rot_corr_flow = V - (B @ Omega_min)

    # A: (C,2,3)
    # T_min: (3)
    AT_min = A @ T_min
    lhs_term = jnp.sum(rot_corr_flow * AT_min, axis=1)  # (C)
    rhs_term = jnp.sum(AT_min**2, axis=1)  # (C)
    inv_depth = lhs_term / (rhs_term + 1e-7)
    return inv_depth


@jax.jit
@jaxtyped(typechecker=typechecker)
def calc_inverse(A_p_T_B: Float[Array, "num_samples 3"]) -> Float[Array, "3 3"]:
    """Calculate the inverse of A_p_T_B
    Parameters
    ------------
    A_p_T_B : ndarray (#pts, 3)
        A_perp_T_B
    Returns
    --------
    inv : ndarray (3,3)
        Inverse of A_p_T_B
    """
    return jnp.linalg.inv(A_p_T_B.T @ A_p_T_B)


@jax.jit
@jaxtyped(typechecker=typechecker)
def create_A_perp_matrix(
    cord: Float[Array, "3"],
    f: Union[float, Float[ScalarLike, ""]],
) -> Float[Array, "2 3"]:
    """Create A_perp Matrix
    [[0, -f, y],
    [f, 0, -x]]
    A_perp=[[0,1],[-1,0]]A

    Parameters
    ------------
    cord : ndarray (3)
        Camera Point
    f : float
        Focal Length
    Returns
    --------
    A_perp : ndarray (2,3)
    """
    return jnp.array([[0, -f, cord[1]], [f, 0, -cord[0]]])


@jax.jit
@jaxtyped(typechecker=typechecker)
def create_A_matrix(
    cord: Float[Array, "3"],
    f: Union[float, Float[ScalarLike, ""]],
) -> Float[Array, "2 3"]:
    """Create the A matrix for each camera point
    [[-f, 0, x],
    [0, -f, y]]
    Parameters
    ------------
    cord : ndarray (3)
        Camera Point
    f : float
        Focal Length
    Returns
    --------
    A : ndarray (2,3)
    """
    return jnp.array(
        [
            [-f, 0, cord[0]],
            [0, -f, cord[1]],
        ]
    )


@jax.jit
@jaxtyped(typechecker=typechecker)
def create_B_matrix(
    cord: Float[Array, "3"],
    f: Union[float, Float[ScalarLike, ""]],
) -> Float[Array, "2 3"]:
    """Create the B matrix for each camera point
    [[xy/f, -(f+x^2/f), y],
    [f+y^2/f, -(xy)/f, -x]]
    Parameters
    ------------
    cord : ndarray (3)
        Camera Point
    f : float
        Focal Length
    Returns
    --------
    B : ndarray (2,3)
    """
    return jnp.array(
        [
            [(cord[0] * cord[1]) / f, -(f + (cord[0] ** 2) / f), cord[1]],
            [f + (cord[1] ** 2) / f, -(cord[0] * cord[1]) / f, -cord[0]],
        ]
    )


@jaxtyped(typechecker=typechecker)
def visualize_T_res(
    T_search: Float[Array, "3 num_cand_points"],
    tot_res: Float[Array, "num_cand_points"],
    T_min: Float[Array, "3"],
    T: Float[Array, "3"] = jnp.zeros(0),
):
    """Helper function to plot residuals over translation directions
    Parameters
    ------------
    T_search : ndarray (3, num_cand_points)
        Candidate Translation Directions
    tot_res : ndarray (num_cand_points)
        Total Residuals for each flow vector
    T_min : ndarray (3)
        Minimum Translation out T_search
    T : ndarray (3)
        Ground Truth Translation
    """
    plt.hexbin(
        T_search[0],
        T_search[1],
        C=np.clip(tot_res, 0, np.quantile(tot_res, 0.85)),
        gridsize=13,
    )

    plt.scatter(T_min[0], T_min[1], c="red")
    if T.size > 0:
        gt_dir = T / jnp.linalg.norm(T)
        plt.scatter(gt_dir[0], gt_dir[1], c="green")

    plt.xlabel("X Direction")
    plt.ylabel("Y Direction")
    plt.title("Residual over Translation Direction")
    plt.colorbar()


def main():
    T = jnp.array([0, 0, 2])
    Ω = jnp.array([0, 0, 0])
    res = (100, 100)
    f = 1
    min_depth = 5
    max_depth = 10
    Z = jax.random.uniform(
        jax.random.PRNGKey(0), (res[0] * res[1], 1), minval=min_depth, maxval=max_depth
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    K = jnp.array(
        [
            [f, 0, res[0] / 2],
            [0, f, res[1] / 2],
            [0, 0, 1],
        ]
    )

    # Create whatever other method you would like to generate points
    # Tradationally, we do a coarse to fine search!
    T_search = generate_random_points_on_positive_hemisphere(1000)
    # Feel free to define your own policy, but this is a good starting point

    # Function to sample the flow field and return cam_pts
    # Computationally expensive to use entire flow field, so we sample
    flow_eval_pts, flow_eval = sample_points(jnp.ones(res), flow, 100 * 100)
    # Feel free to define your own policy(Grid Sampling, Random,etc), but this is a good starting point

    T_min, Omega_min, tot_res = heeger_jepson_RT(
        flow_eval,
        flow_eval_pts,
        K,
        f,
        res,
        T_search,
    )

    inv_depth = get_inv_depth(
        flow_eval,
        flow_eval_pts,
        K,
        f,
        res,
        Omega_min,
        T_min,
    )
    print("T_min, T_GT")
    print(T_min, T / jnp.linalg.norm(T))
    print("Ω_min, Ω_GT")
    print(Omega_min, Ω)

    # Reminder, these will be off my a scale constant
    # Also, given how well you can solve for Omega and T
    Z_samp = Z.reshape(res)[flow_eval_pts[1], flow_eval_pts[0]]
    print("Z_min, Z_GT")
    print(1 / inv_depth[0], Z_samp[0])


if __name__ == "__main__":
    main()
