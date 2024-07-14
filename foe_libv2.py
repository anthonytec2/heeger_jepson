# See egomotion.ipynb for exploratory introduction to this code

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from motionfield import *
from functools import partial


def generate_random_points_on_positive_hemisphere(num_points):
    """Generate num_points random points on the positive hemisphere in R^3
    Parameters
    ------------
    num_points : int
        Number of points to generate
    Returns
    --------
    points : ndarray (num_points, 3)
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
    return np.vstack([points[:, 0], points[:, 1], points[:, 2]])


@partial(jax.jit, static_argnames=["num_samples"])
def sample_points(mask_in, flow, num_samples, k_size=21):
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


class HeegerJepson:
    def __init__(self, V, cam_pts, f, T_search, res):
        """Heeger Jepson Algorithm
        Parameters
        ------------
        V : ndarray (C,2),
            Flow Field
        cam_pts : ndarray (3, C)
            Camera Points
        f : float
            Focal Length
        T_search : ndarray (3, num_cand_points)
            Candidate Translation Directions
        res : tuple
            Resolution of the image
        """
        K = jnp.array(
            [
                [f, 0, res[0] / 2],
                [0, f, res[1] / 2],
                [0, 0, 1],
            ]
        )
        K_inv = jnp.linalg.inv(K)
        norm_cords = K_inv @ cam_pts

        flow_norm = V * (1 / f)

        # Create a function to create a different A matrix for each camera point
        # (Mapping over each camera point, and passing f=1)
        create_A = jax.vmap(self.create_A_matrix, in_axes=(1, None), out_axes=0)
        A = create_A(norm_cords, 1)  # A: (res[0] * res[1], 2, 3)

        # Create a function to create a different B matrix for each camera point
        # (Mapping over each camera point, and passing f=1)
        create_B = jax.vmap(self.create_B_matrix, in_axes=(1, None), out_axes=0)
        B = create_B(norm_cords, 1)  # (res[0] * res[1], 2, 3)

        # Creat A_perp Matrix, map over norm camera points,
        create_A_perp = jax.vmap(
            self.create_A_perp_matrix, in_axes=(1, None), out_axes=0
        )
        A_perp = create_A_perp(norm_cords, 1)  # (Pts, 2, 3)
        A_perp_T = A_perp @ T_search
        tmp_rearange = A_perp_T.transpose(
            1, 0, 2
        )  # A_perp_T(Rearanged) (2, # pts, # Dirs)
        A_perp_T_norm = tmp_rearange / (
            jnp.linalg.norm(tmp_rearange, axis=0)[None, :, :] + 1e-10
        )

        V = flow_norm

        A_perp_T_V = jnp.sum(
            A_perp_T_norm.transpose(1, 0, 2) * V[:, :, None], axis=1
        )  # (#pts,2,#Dirs) x (#pts,2,1)= \sum_a1 (#pts,2,#Dirs)= (#pts,#Dirs)

        A_perp_T_B = jnp.sum(
            A_perp_T_norm.transpose(1, 0, 2)[:, :, None, :] * B[:, :, :, None], axis=1
        )  # (#pts,2,1,#Dirs) x (#pts,2,3, 1)= \sum_a1 (#pts,2,3,#Dirs)= (#pts,3, #Dirs)

        inv_func = jax.vmap(self.calc_inverse, in_axes=(2))
        A_perp_T_B_inv = inv_func(A_perp_T_B)  # (#pts,3, #Dirs) -> (#Dirs,3,3)
        A_perp_T_B_A_perp_T_V = jax.lax.batch_matmul(
            A_perp_T_B.transpose(2, 1, 0), A_perp_T_V.T[:, :, None]
        )[:, :, 0].T

        Omega = jax.lax.batch_matmul(
            A_perp_T_B_inv, A_perp_T_B_A_perp_T_V.T[:, :, None]
        )[:, :, 0]
        A_perp_B_omega = jnp.sum(A_perp_T_B * Omega[:, :, None].T, axis=1)
        flow_res = (A_perp_B_omega - A_perp_T_V) ** 2  # calculate norm error
        tot_res = jnp.sum(flow_res, axis=0)  # sum up over all points
        min_idx = jnp.argmin(tot_res)
        self.T_min = T_search[:, min_idx]
        self.Omega_min = Omega[min_idx]

        rot_corr_flow = V - (B @ self.Omega_min)
        AT_min = A @ self.T_min
        lhs_term = jnp.sum(rot_corr_flow * AT_min, axis=1)
        rhs_term = jnp.sum(AT_min**2, axis=1)
        self.inv_depth = lhs_term / (rhs_term + 1e-7)

    def calc(
        self,
    ):
        return self.T_min, self.Omega_min, self.inv_depth

    def create_A_perp_matrix(self, cord, f):
        """Create A_perp Matrix
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

    def calc_inverse(self, A_p_T_B):
        """Calculate the inverse of A_p_T_B
        Parameters
        ------------
        A_p_T_B : ndarray (3, #Dirs)
            A_perp_T_B
        Returns
        --------
        inv : ndarray (3,3)
            Inverse of A_p_T_B
        """
        return jnp.linalg.inv(A_p_T_B.T @ A_p_T_B)

    def create_A_matrix(self, cord, f):
        """Create the A matrix for each camera point
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

    def create_B_matrix(self, cord, f):
        """Create the B matrix for each camera point
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
                [(cord[0] * cord[1]), -(f + (cord[0] ** 2) / f), cord[1]],
                [f + (cord[1] ** 2) / f, -(cord[0] * cord[1]) / f, -cord[0]],
            ]
        )


if __name__ == "__main__":
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

    # Create whatever other method you would like to generate points
    # Tradationally, we do a coarse to fine search!
    T_search = generate_random_points_on_positive_hemisphere(1000)
    # Feel free to define your own policy, but this is a good starting point

    # Function to sample the flow field and return cam_pts
    # Computationally expensive to use entire flow field, so we sample
    flow_eval_pts, flow_eval = sample_points(jnp.ones(res), flow, 45 * 45)
    # Feel free to define your own policy(Grid Sampling, Random,etc), but this is a good starting point

    T_min, Omega_min, inv_depth = HeegerJepson(
        flow_eval, flow_eval_pts, f, T_search, res
    ).calc()
