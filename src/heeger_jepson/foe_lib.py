import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import lax


def generate_random_points_on_positive_hemisphere(num_points):
    """Generate num_points random points on the positive hemisphere in R^3"""
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
    return points[:, 0], points[:, 1], points[:, 2]


def create_bmat(pt):
    return jnp.stack(
        [
            jnp.array([(pt[0] * pt[1]), -((1 + pt[0] ** 2)), pt[1]]),
            jnp.array([1 + pt[1] ** 2, -(pt[0] * pt[1]), -pt[0]]),
        ]
    ).reshape(2, 3, -1)


def proj_T(A_p, T_ex):
    return A_p.T @ T_ex


def create_big_a(dA):
    return jsp.linalg.block_diag(*jnp.split(dA, dA.shape[1], axis=1))


def create_aperp_b(bA, bB):
    return bA.T @ bB


def create_aperp_v(bA, V):
    return bA.T @ V


def inverse_func(apb):
    # with jax.named_scope("Prod"):
    #     A = apb.T @ apb

    # det_A = (
    #     A[0, 0] * (A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2])
    #     - A[0, 1] * (A[1, 0] * A[2, 2] - A[2, 0] * A[1, 2])
    #     + A[0, 2] * (A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1])
    # )

    # cofactor_A = jnp.array(
    #     [
    #         [
    #             A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2],
    #             -(A[1, 0] * A[2, 2] - A[2, 0] * A[1, 2]),
    #             A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1],
    #         ],
    #         [
    #             -(A[0, 1] * A[2, 2] - A[2, 1] * A[0, 2]),
    #             A[0, 0] * A[2, 2] - A[2, 0] * A[0, 2],
    #             -(A[0, 0] * A[2, 1] - A[2, 0] * A[0, 1]),
    #         ],
    #         [
    #             A[0, 1] * A[1, 2] - A[1, 1] * A[0, 2],
    #             -(A[0, 0] * A[1, 2] - A[1, 0] * A[0, 2]),
    #             A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1],
    #         ],
    #     ]
    # )
    # return cofactor_A / (det_A + 1e-3)

    # with jax.named_scope("Inv"):
    #     inv = jnp.linalg.inv(prod)
    # inv = prod
    return jnp.linalg.inv(apb.T @ apb)


def project_a_b_v(apb, apv):
    return apb.T @ apv


def project_inv(lhs_inv, rhs_proj):
    return lhs_inv @ rhs_proj


def a_omega(apb, omega):
    return apb @ omega


def crt_aprp_v2(a_n, v_n):
    return jnp.sum(a_n * v_n, axis=0)


def crt_aprp_b2(A, b):
    a_mat = jnp.repeat(A.T, 3, axis=0)
    return jnp.sum(b * a_mat, axis=1).reshape(-1, 3)


def create_amat(pt):
    return jnp.stack(
        [
            jnp.array([-1, 0, pt[0]]),
            jnp.array([0, -1, pt[1]]),
        ]
    ).reshape(2, 3, -1)


def calc_nd(a, b, J, t_vec, flow_vec):
    t_vec = t_vec[:, None]
    num = b.T @ J @ a @ t_vec @ t_vec.T @ a.T @ J.T
    den = jnp.linalg.norm(J @ a @ t_vec) ** 2
    base = num / den
    return base @ b, base @ flow_vec


def calc_e1(a, J, t_vec):
    t_vec = t_vec[:, None]
    num = t_vec.T @ a.T @ J.T
    den = jnp.linalg.norm(J @ a @ t_vec)
    return num / den


calc_A = jax.vmap(create_amat, in_axes=(1), out_axes=0)
ndc_wt = jax.vmap(
    jax.vmap(calc_nd, in_axes=(0, 0, None, None, 1), out_axes=0),
    in_axes=(None, None, None, 1, None),
    out_axes=0,
)

e1c_wt = jax.vmap(
    jax.vmap(calc_e1, in_axes=(0, None, None), out_axes=0),
    in_axes=(None, None, 1),
    out_axes=0,
)


@jax.jit
def calc_pts(mask_in, x_raft, y_raft):
    mask = jnp.array(mask_in).astype(jnp.float32)
    kernel = jnp.ones((1, 1, 21, 21))

    xval = jnp.arange(x_raft.shape[1])
    yval = jnp.arange(y_raft.shape[0])
    xx, yy = jnp.meshgrid(xval, yval)

    cnt_img = lax.conv(
        mask,  # lhs = NCHW image tensor
        kernel,  # rhs = OIHW conv kernel tensor
        (1, 1),  # window strides
        "SAME",
    )  # padding mode
    mask_cnt_img = cnt_img * mask
    prob_out = mask_cnt_img.reshape(-1) / jnp.sum(mask_cnt_img)

    samples = 25 * 25
    rnd_pts = jax.random.choice(
        jax.random.PRNGKey(0),
        jnp.arange(len(prob_out)),
        shape=[samples],
        p=prob_out,
        replace=False,
    )
    x_sel = xx.reshape(-1)[rnd_pts]
    y_sel = yy.reshape(-1)[rnd_pts]
    pts = jnp.stack([x_sel, y_sel, jnp.ones_like(x_sel)])
    return pts


big_b = jax.vmap(create_bmat, in_axes=(1), out_axes=0)
proj_T_vm = jax.vmap(proj_T, in_axes=(0, None), out_axes=1)
crt_big_a = jax.vmap(create_big_a, in_axes=(2))
crt_aprp_b = jax.vmap(create_aperp_b, in_axes=(0, None))
crt_aprp_v = jax.vmap(create_aperp_v, in_axes=(0, None))
crt_aprp_v2 = jax.vmap(crt_aprp_v2, in_axes=(2, None))
crt_aprp_b2 = jax.vmap(crt_aprp_b2, in_axes=(2, None))
calc_mat_inv = jax.vmap(inverse_func, in_axes=(0))
project_a_b_v_vm = jax.vmap(project_a_b_v, in_axes=(0, 0))
project_inv_vm = jax.vmap(project_inv, in_axes=(0, 0))
a_omega_vm = jax.vmap(a_omega, in_axes=(0, 0))


@jax.jit
def standard_foe(pts, x_raft, y_raft, T, intr, K_inv):
    with jax.named_scope("K_inv"):
        # K_inv = jnp.linalg.inv(intr)
        calib_pts = (K_inv @ pts)[:2]

    with jax.named_scope("Pts"):
        x_f = x_raft[pts[1], pts[0]] * (K_inv[0, 0])
        y_f = y_raft[pts[1], pts[0]] * (K_inv[1, 1])
        V = jnp.stack([x_f, y_f])

    # Calculate A
    with jax.named_scope("A"):
        J_T = jnp.array([[0, -1], [1, 0]], dtype=jnp.float16).T
        Jxy = J_T @ calib_pts
        J_T_repeat = jnp.tile(J_T[None, :, :], (calib_pts.shape[1], 1, 1))
        A_perp = jnp.hstack([J_T_repeat, Jxy.T[:, None, :]])

    # Calculate B
    with jax.named_scope("B"):
        big_B = big_b(calib_pts.astype(jnp.float16))[:, :, :, 0]
        tiled_B = big_B.reshape(-1, 3)

    # Project T and Normalize
    with jax.named_scope("APerp"):
        A1 = proj_T_vm(A_perp, T)
        A1_norm = A1 / (jnp.linalg.norm(A1, axis=(0))[None, :, :] + 1e-10)

    with jax.named_scope("TileB"):
        b_mat = jnp.stack(
            [
                tiled_B[:, 0].reshape(-1, 2),
                tiled_B[:, 1].reshape(-1, 2),
                tiled_B[:, 2].reshape(-1, 2),
            ],
            axis=1,
        ).reshape(-1, 2)
    with jax.named_scope("AV"):
        A_perp_v = crt_aprp_v2(A1_norm, V)
    with jax.named_scope("AB"):
        A_perp_b = crt_aprp_b2(A1_norm, b_mat)

    # Calculate Omega
    with jax.named_scope("INV"):
        lhs_inv = calc_mat_inv(A_perp_b)
    with jax.named_scope("ABAV"):
        rhs_proj = project_a_b_v_vm(A_perp_b, A_perp_v)
    with jax.named_scope("Omega"):
        omega = project_inv_vm(lhs_inv, rhs_proj)

    # Solve for Flow residual
    with jax.named_scope("Res"):
        A_omega = a_omega_vm(A_perp_b, omega)  # CHECK SCALE OF FLOW
        flow_res = (A_omega - A_perp_v) ** 2

    # Min Direction and Omega
    with jax.named_scope("Min"):
        min_idx = jnp.argmin(jnp.sum(flow_res, axis=1))
        T_min = T[min_idx]
        Omega_min = T[min_idx]

    return omega, flow_res


@jax.jit
def new_foe(pts, x_raft, y_raft, T, intr):
    with jax.named_scope("K_inv"):
        K_inv = jnp.linalg.inv(intr)
        norm_pts = (K_inv @ pts)[:2]
    with jax.named_scope("Pts"):
        x_f = x_raft[pts[1], pts[0]] * (K_inv[0, 0])
        y_f = y_raft[pts[1], pts[0]] * (K_inv[1, 1])
        V = jnp.stack([x_f, y_f])

        J = jnp.array([[0, -1], [1, 0]])
    with jax.named_scope("B"):
        B = big_b(norm_pts)[:, :, :, 0]
    with jax.named_scope("A"):
        A = calc_A(norm_pts)[:, :, :, 0]
    with jax.named_scope("S1"):
        s1 = ndc_wt(A, B, J, T, V)
    with jax.named_scope("Omega"):
        omega_pred = jax.lax.batch_matmul(
            jnp.linalg.inv(jnp.sum(s1[0], axis=1)), jnp.sum(s1[1], axis=1)[:, :, None]
        )[:, :, 0]
    with jax.named_scope("E"):
        e1 = e1c_wt(A, J, T)
    with jax.named_scope("Res"):
        R1 = ((B @ omega_pred.T) - V.T[:, :, None]) * e1[:, :, 0].transpose(1, 2, 0)
        res_mat = jnp.sum(R1, axis=1)

    return omega_pred, res_mat
