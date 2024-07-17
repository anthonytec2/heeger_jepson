from src.heeger_jepson.foe_libv2 import *
from src.heeger_jepson.motionfield import *


def heeger_jepson_runner(flow, res, f, num_search=1000, num_pts=100 * 100):
    T_search = generate_random_points_on_positive_hemisphere(num_search)

    # Function to sample the flow field and return cam_pts
    # Computationally expensive to use entire flow field, so we sample
    flow_eval_pts, flow_eval = sample_points(jnp.ones(res), flow, num_pts)

    K = jnp.array(
        [
            [f, 0, res[0] / 2],
            [0, f, res[1] / 2],
            [0, 0, 1],
        ]
    )

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
    return T_min, Omega_min, inv_depth


def test_no_rotation():
    # Test setting each of they axes x,y,z
    T = jnp.array([0, 0, 2])
    Ω = jnp.array([0, 0, 0])
    res = (100, 100)
    f = 1
    min_depth = 5
    max_depth = 10
    Z = jax.random.uniform(
        jax.random.PRNGKey(0),
        (res[0] * res[1], 1),
        minval=min_depth,
        maxval=max_depth,
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f)

    T_norm = T / np.linalg.norm(T)
    assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(Omega_min, Ω, atol=1e-1)


def test_translation_rotation():
    # Test setting each of they axes x,y,z
    for i in range(3):
        T = jnp.array([0, 0, 2])
        Ω = jnp.array([1 if j == i else 0 for j in range(3)])
        res = (100, 100)
        f = 1
        min_depth = 5
        max_depth = 10
        Z = jax.random.uniform(
            jax.random.PRNGKey(0),
            (res[0] * res[1], 1),
            minval=min_depth,
            maxval=max_depth,
        )
        flow = MotionField(f, res, T, Ω, Z).generate_flow()
        T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f)

        T_norm = T / np.linalg.norm(T)
        assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(
            Omega_min, Ω, atol=1e-1
        )


def test_comb_xz():
    T = jnp.array([1, 0, 2])
    Ω = jnp.array([0, 0, 1])
    res = (100, 100)
    f = 1
    min_depth = 5
    max_depth = 10
    Z = jax.random.uniform(
        jax.random.PRNGKey(0),
        (res[0] * res[1], 1),
        minval=min_depth,
        maxval=max_depth,
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f)

    T_norm = T / np.linalg.norm(T)
    assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(Omega_min, Ω, atol=1e-1)


def test_comb_yz():
    T = jnp.array([0, 1, 2])
    Ω = jnp.array([0, 0, 1])
    res = (100, 100)
    f = 1
    min_depth = 5
    max_depth = 10
    Z = jax.random.uniform(
        jax.random.PRNGKey(0),
        (res[0] * res[1], 1),
        minval=min_depth,
        maxval=max_depth,
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f)

    T_norm = T / np.linalg.norm(T)
    assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(Omega_min, Ω, atol=1e-1)


def test_comb_all():
    T = jnp.array([0.4, 1, 2])
    Ω = jnp.array([0.2, 0.5, 1])
    res = (100, 100)
    f = 1
    min_depth = 5
    max_depth = 10
    Z = jax.random.uniform(
        jax.random.PRNGKey(0),
        (res[0] * res[1], 1),
        minval=min_depth,
        maxval=max_depth,
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f)

    T_norm = T / np.linalg.norm(T)
    assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(Omega_min, Ω, atol=1e-1)


def test_change_res():
    T = jnp.array([0.4, 0.2, 1])
    Ω = jnp.array([0, 0.5, 1])
    res = (50, 50)
    f = 1
    min_depth = 5
    max_depth = 10
    Z = jax.random.uniform(
        jax.random.PRNGKey(0),
        (res[0] * res[1], 1),
        minval=min_depth,
        maxval=max_depth,
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f, 1000, 40 * 40)

    T_norm = T / np.linalg.norm(T)
    assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(Omega_min, Ω, atol=1e-1)


def test_change_number_search_dir():
    T = jnp.array([0.4, 0.2, 1])
    Ω = jnp.array([0, 0.5, 1])
    res = (40, 40)
    f = 1
    min_depth = 5
    max_depth = 10
    Z = jax.random.uniform(
        jax.random.PRNGKey(0),
        (res[0] * res[1], 1),
        minval=min_depth,
        maxval=max_depth,
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f, 800, 30 * 30)

    T_norm = T / np.linalg.norm(T)
    assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(Omega_min, Ω, atol=1e-1)


def test_change_depth_rang():
    T = jnp.array([0.4, 0.2, 1])
    Ω = jnp.array([0, 0.5, 1])
    res = (50, 50)
    f = 1
    min_depth = 1
    max_depth = 35
    Z = jax.random.uniform(
        jax.random.PRNGKey(0),
        (res[0] * res[1], 1),
        minval=min_depth,
        maxval=max_depth,
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f, 1000, 50 * 50)

    T_norm = T / np.linalg.norm(T)
    assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(Omega_min, Ω, atol=1e-1)


def test_change_f():
    T = jnp.array([0, 0, 1])
    Ω = jnp.array([0, 0.2, 0])
    res = (50, 50)
    f = 0.2
    min_depth = 2
    max_depth = 15
    Z = jax.random.uniform(
        jax.random.PRNGKey(0),
        (res[0] * res[1], 1),
        minval=min_depth,
        maxval=max_depth,
    )
    flow = MotionField(f, res, T, Ω, Z).generate_flow()
    T_min, Omega_min, inv_depth = heeger_jepson_runner(flow, res, f, 1000, 50 * 50)

    T_norm = T / np.linalg.norm(T)
    assert np.allclose(T_min, T_norm, atol=1e-1) & np.allclose(Omega_min, Ω, atol=1e-1)
