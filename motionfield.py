# Class that will generate an arbitrary flow field
import jax.numpy as jnp
import jax


# See egomotion.ipynb for exploratory introduction to this code
class MotionField:
    def __init__(self, f, res, T, Ω, Z):
        """Generate an arbitrary Motion Field
        Paramerters
        ------------
        f : float
            Focal Length
        res : tuple: (h, w)
            Resolution of the image
        T : ndarray: (3)
            Translation Vector
        Ω : ndarray: (3,3)
            Rotation Matrix
        Z : ndarray (h,w)
            Depth of the scene
        """
        self.f = f
        self.res = res
        self.T = T
        self.Ω = Ω
        self.Z = Z

    def generate_flow(self):
        """Generate the Flow Field
        V = (1/z)AT + BΩ
        Returns
        --------
        flow : ndarray (h,w,2)
            Flow Field
        """

        # Generate the camera pixel coordinates
        y = jnp.arange(0, self.res[0])
        x = jnp.arange(0, self.res[1])
        xx, yy = jnp.meshgrid(x, y)
        cam_pts = jnp.stack(
            [xx.flatten(), yy.flatten(), jnp.ones(self.res[0] * self.res[1])]
        )  # (3, res[0] * res[1])

        # Create the Intrinsics Matrix
        K = jnp.array(
            [
                [self.f, 0, self.res[0] / 2],
                [0, self.f, self.res[1] / 2],
                [0, 0, 1],
            ]
        )
        K_inv = jnp.linalg.inv(K)  # (3,3)

        # Convert everything to normalized coordinates
        norm_cords = K_inv @ cam_pts

        # Create a function to create a different A matrix for each camera point
        # (Mapping over each camera point, and passing f=1)
        create_A = jax.vmap(self.create_A_matrix, in_axes=(1, None), out_axes=0)
        A = create_A(norm_cords, 1)  # A: (res[0] * res[1], 2, 3)

        # Create a function to create a different B matrix for each camera point
        # (Mapping over each camera point, and passing f=1)
        create_B = jax.vmap(self.create_B_matrix, in_axes=(1, None), out_axes=0)
        B = create_B(norm_cords, 1)  # (res[0] * res[1], 2, 3)

        v = (1 / self.Z) * (A @ self.T) + B @ self.Ω  # (res[0] * res[1], 2)
        flow = v.reshape(self.res[0], self.res[1], 2)  # (res[0], res[1], 2)
        self.flow = flow
        return flow

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
    # Flow Field Parameters for generation
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
    print(flow.shape)
