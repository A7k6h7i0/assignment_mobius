import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=100):
        """
        Initialize the Mobius Strip.

        Parameters:
        R : float
            Radius from center to the middle of the strip
        w : float
            Width of the strip
        n : int
            Resolution (number of points in u and v directions)
        """
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Generate 3D mesh
        self.X, self.Y, self.Z = self._generate_mesh()

    def _generate_mesh(self):
        """
        Generate 3D mesh points using parametric equations.
        """
        u = self.U
        v = self.V
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def compute_surface_area(self):
        """
        Approximate the surface area using the magnitude of the cross product of partial derivatives.
        """
        # Partial derivatives
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

        # Derivatives
        Xu = np.gradient(self.X, du, axis=1)
        Xv = np.gradient(self.X, dv, axis=0)
        Yu = np.gradient(self.Y, du, axis=1)
        Yv = np.gradient(self.Y, dv, axis=0)
        Zu = np.gradient(self.Z, du, axis=1)
        Zv = np.gradient(self.Z, dv, axis=0)

        # Cross product of partial derivatives
        cross_x = Yu * Zv - Zu * Yv
        cross_y = Zu * Xv - Xu * Zv
        cross_z = Xu * Yv - Yu * Xv

        # Area element
        dA = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        surface_area = np.sum(dA) * du * dv
        return surface_area

    def compute_edge_length(self):
        """
        Approximate the edge length along the boundary (e.g., v = w/2)
        """
        u = self.u
        v = self.w / 2
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)

        length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
        return length

    def plot(self):
        """
        Visualize the Mobius strip using matplotlib.
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_title('Möbius Strip')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    mobius = MobiusStrip(R=1, w=0.3, n=200)
    print("Surface Area:", mobius.compute_surface_area())
    print("Edge Length:", mobius.compute_edge_length())
    mobius.plot()
