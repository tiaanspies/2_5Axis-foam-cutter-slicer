import matplotlib.pyplot as plt
import numpy
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import math
# Using an existing stl file:


def do_stuff():
    # plot stl 3d model--------------------------
    figure3 = pyplot.figure(3)
    axes = mplot3d.Axes3D(figure3)

    # Load the STL files and add the vectors to the plot
    # your_mesh = mesh.Mesh.from_file('cube_1x1.stl')
    # your_mesh = mesh.Mesh.from_file('tool_holder_bars.stl')
    your_mesh = mesh.Mesh.from_file('LabradorLowPoly.stl')
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.title("Stl file displaying")
    # define plane to project onto ----------------------

    theta = 45 * numpy.pi/180
    n = numpy.array([math.cos(theta), math.sin(theta), 0])
    n[abs(n) < 1e-15] = 0.0

    # n = [1, 0, 0]  # direction of plane normal
    # theta = np.arctan2(n[1], n[0])  # orientation of plane

    print(theta*180/numpy.pi)

    n_normalized = n/np.linalg.norm(n)

    # project all points onto the plane------

    projected_points = np.zeros([len(your_mesh.vectors), 3, 3])

    for index_tri, triangle in enumerate(your_mesh.vectors):
        for index_vert, vertex in enumerate(triangle):
            q = np.multiply(np.dot(n_normalized, triangle[index_vert]), n_normalized)
            projected_points[index_tri, index_vert] = np.subtract(triangle[index_vert], q)

    plot_values = projected_points.reshape(-1, projected_points.shape[-1])

    # plot projected values-------------------------------

    fig1 = pyplot.figure(1)

    ax = fig1.add_subplot(111, projection='3d')

    ax.scatter(plot_values[:, 0], plot_values[:, 1], plot_values[:, 2])
    pyplot.xlabel('X')

    # rotate plane onto yz plane creating a 2d view-----------

    plot_values_2d_x, plot_values_2d_y = rotate_origin_only((plot_values[:, 0], plot_values[:, 1]), theta)
    plot_values_2d_z = plot_values[:, 2]
    plt.figure(2)

    pyplot.plot(plot_values_2d_y, plot_values_2d_z, 'o', color='blue')

    plt.figure(4)
    plot_values_2d_x_reshaped = numpy.reshape(plot_values_2d_y, (-1, 3))
    plot_values_2d_z_reshaped = numpy.reshape(plot_values_2d_z, (-1, 3))

    for x, y in zip(plot_values_2d_x_reshaped, plot_values_2d_z_reshaped):
        pyplot.plot([x[0], x[1]], [y[0], y[1]], color='red')
        pyplot.plot([x[1], x[2]], [y[1], y[2]], color='red')
        pyplot.plot([x[2], x[0]], [y[2], y[0]], color='red')

    pyplot.show()


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


if __name__ == '__main__':
    do_stuff()
