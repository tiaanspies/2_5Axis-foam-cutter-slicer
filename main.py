import matplotlib.pyplot as plt
import numpy
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import math
import trimesh


def do_stuff():
    your_mesh = mesh.Mesh.from_file('LabradorLowPoly.stl')
    # plot stl 3d model--------------------------
    # figure3 = pyplot.figure(3)
    # plot_axes = mplot3d.Axes3D(figure3)

    # Load the STL files and add the vectors to the plot
    # your_mesh = mesh.Mesh.from_file('cube_1x1.stl')
    # your_mesh = mesh.Mesh.from_file('tool_holder_bars.stl')

    # plot_axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
    #
    # # Auto scale to the mesh size
    # scale = your_mesh.points.flatten()
    # plot_axes.auto_scale_xyz(scale, scale, scale)
    # pyplot.title("Stl file displaying")
    # define plane to project onto ----------------------

    plane_normal_theta = math.radians(90)
    plane_normal_vector = numpy.array([math.cos(plane_normal_theta), math.sin(plane_normal_theta), 0])
    plane_normal_vector[abs(plane_normal_vector) < 1e-15] = 0.0

    # n = [1, 0, 0]  # direction of plane normal
    # theta = np.arctan2(n[1], n[0])  # orientation of plane

    plane_normal_vector_normalized = plane_normal_vector / np.linalg.norm(plane_normal_vector)

    # project all points onto the plane------

    points_on_plane = np.zeros([len(your_mesh.vectors), 3, 3])

    for index_tri, triangle in enumerate(your_mesh.vectors):
        for index_vert, vertex in enumerate(triangle):
            q = np.multiply(np.dot(plane_normal_vector_normalized, vertex)
                            , plane_normal_vector_normalized)
            points_on_plane[index_tri, index_vert] = np.subtract(triangle[index_vert], q)

    # change from nx3 matrix to px1 matrix
    points_on_plane_stretched = points_on_plane.reshape(-1, points_on_plane.shape[-1])

    # ---------------plot projected values-------------------------------

    # fig1 = pyplot.figure(1)
    #
    # ax = fig1.add_subplot(111, projection='3d')
    #
    # ax.scatter(points_on_plane_stretched[:, 0], points_on_plane_stretched[:, 1], points_on_plane_stretched[:, 2])
    # pyplot.xlabel('X')
    # --------------------------------------------------------------

    # ------------------rotate plane onto yz plane creating a 2d view-----------
    # x values should all be equal to zero
    points_2d_x, points_2d_y = rotate_origin_only((points_on_plane_stretched[:, 0]
                                                   , points_on_plane_stretched[:, 1])
                                                  , plane_normal_theta)
    points_2d_z = points_on_plane_stretched[:, 2]

    points_2d_y[abs(points_2d_y) < 1e-6] = 0
    points_2d_z[abs(points_2d_z) < 1e-6] = 0

    # group back into nx3 matrix so that triangles vertices can be joined
    points_2d_y_collapsed = numpy.reshape(points_2d_y, (-1, 3))
    points_2d_z_collapsed = numpy.reshape(points_2d_z, (-1, 3))

    # ---------------plot 2d shape with vectors included--------------
    plt.figure(2)
    pyplot.plot(points_2d_y, points_2d_z, 'o', color='blue')

    for y, z in zip(points_2d_y_collapsed, points_2d_z_collapsed):
        pyplot.plot([y[0], y[1]], [z[0], z[1]], color='red')
        pyplot.plot([y[1], y[2]], [z[1], z[2]], color='red')
        pyplot.plot([y[2], y[0]], [z[2], z[0]], color='red')

    # pyplot.show()
    # ----------------------------------------------------------------

    # -------FIND PATH AROUND 2D IMAGE-------------------------------
    # -------find starting point in bottom right of image------------
    z_min_index = numpy.argmin(points_2d_z)

    y_max_given_z_min = points_2d_y[z_min_index]  # select initial z to compare rest with
    z_min = points_2d_z[z_min_index]
    z_exact = z_min

    for y, z in zip(points_2d_y, points_2d_z):
        if abs(z - z_min) < 0.1 and y > y_max_given_z_min:
            y_max_given_z_min = y
            z_exact = z

    home_z = z_exact
    home_y = y_max_given_z_min
    angle_previous = 420

    counter = 0
    print("hi")
    # end condition for future: (home_z != z_exact and home_y != y_max_given_z_min)
    while counter < 3:

        # ------find all triangles that contain bottom right point---------------------
        z_match_indexes = numpy.where(points_2d_z == home_z)
        y_match_indexes = numpy.where(points_2d_y == home_y)

        match_intersection_indexes = numpy.intersect1d(z_match_indexes, y_match_indexes)

        # -------find first neighbor from starting from 0 degrees--------
        min_angle_to_neighbor = angle_previous
        min_neigh_z = 0
        min_neigh_y = 0
        for index in match_intersection_indexes:

            triangle_id = index % 3
            triangle_index_start = index - triangle_id
            neighbors_id = numpy.where([0, 1, 2] != triangle_id)

            z_diff_1 = points_2d_z[triangle_index_start + neighbors_id[0][0]] - points_2d_z[index]
            y_diff_1 = points_2d_y[triangle_index_start + neighbors_id[0][0]] - points_2d_y[index]
            z_diff_2 = points_2d_z[triangle_index_start + neighbors_id[0][1]] - points_2d_z[index]
            y_diff_2 = points_2d_y[triangle_index_start + neighbors_id[0][1]] - points_2d_y[index]

            if abs(z_diff_1) > 1e-8 or abs(y_diff_1) > 1e-8:
                angle_to_neighbor = numpy.arctan2(z_diff_1, y_diff_1)
                if angle_to_neighbor > angle_previous:
                    difference = angle_to_neighbor-angle_previous
                else:
                    difference = 360-angle_previous+angle_to_neighbor

                if difference < min_angle_to_neighbor:
                    min_angle_to_neighbor = difference
                    min_neigh_z = points_2d_z[triangle_index_start + neighbors_id[0][0]]
                    min_neigh_y = points_2d_y[triangle_index_start + neighbors_id[0][0]]

                pyplot.plot([points_2d_y[index], points_2d_y[triangle_index_start + neighbors_id[0][0]]]
                        , [points_2d_z[index], points_2d_z[triangle_index_start + neighbors_id[0][0]]]
                        , color='green')

            if abs(z_diff_2) > 1e-8 or abs(y_diff_2) > 1e-8:
                angle_to_neighbor = numpy.arctan2(z_diff_2, y_diff_2)
                if angle_to_neighbor > angle_previous:
                    difference = angle_to_neighbor - angle_previous
                else:
                    difference = 360 - angle_previous + angle_to_neighbor

                if difference < min_angle_to_neighbor:
                    min_angle_to_neighbor = difference
                    min_neigh_z = points_2d_z[triangle_index_start + neighbors_id[0][1]]
                    min_neigh_y = points_2d_y[triangle_index_start + neighbors_id[0][1]]

                pyplot.plot([points_2d_y[index], points_2d_y[triangle_index_start + neighbors_id[0][1]]]
                        , [points_2d_z[index], points_2d_z[triangle_index_start + neighbors_id[0][1]]]
                        , color='green')

        pyplot.plot(min_neigh_y, min_neigh_z, 'o', color='orange')

        home_z = min_neigh_z
        home_y = min_neigh_y
        counter += 1
        angle_previous = (min_angle_to_neighbor + numpy.pi) % (2 * numpy.pi)
    pyplot.show()


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


def is_between(x0, x, x1):
    return (x >= x0) and (x <= x1)


def find_intersection(x0, y0, x1, y1, a0, b0, a1, b1):
    #    // four endpoints are x0, y0 & x1,y1 & a0,b0 & a1,b1
    #    // returned values xy and ab are the fractional distance along xy and ab
    #    // and are only defined when the result is true
    partial = False
    denom = (b0 - b1) * (x0 - x1) - (y0 - y1) * (a0 - a1)
    xy = 0.0
    ab = 0.0

    if denom == 0:
        xy = -1
        ab = -1
    else:
        xy = (a0 * (y1 - b1) + a1 * (b0 - y1) + x1 * (b1 - b0)) / denom
        partial = is_between(0, xy, 1)
    if partial:
        ab = (y1 * (x0 - a1) + b1 * (x1 - x0) + y0 * (a1 - x1)) / denom

    if partial and is_between(0, ab, 1):
        ab = 1 - ab
        xy = 1 - xy
        return True, xy, ab
    else:
        return False, 0, 0


# }


if __name__ == '__main__':
    do_stuff()
