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
    # your_mesh = mesh.Mesh.from_file('cube_1x1.stl')
    # your_mesh = mesh.Mesh.from_file('tool_holder_bars.stl')
    # your_mesh = mesh.Mesh.from_file('scad_chess_pawn.stl')
    """plot stl 3d model--------------------------"""
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

    """project all points onto the plane------"""

    points_on_plane = np.zeros([len(your_mesh.vectors), 3, 3])

    for index_tri, triangle in enumerate(your_mesh.vectors):
        for index_vert, vertex in enumerate(triangle):
            q = np.multiply(np.dot(plane_normal_vector_normalized, vertex)
                            , plane_normal_vector_normalized)
            points_on_plane[index_tri, index_vert] = np.subtract(triangle[index_vert], q)

    # change from nx3 matrix to px1 matrix
    points_on_plane_stretched = points_on_plane.reshape(-1, points_on_plane.shape[-1])

    # ---------------plot projected values-------------------------------
    """plot projected values"""

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

    class Vertex:
        def __init__(self, y_p, z_p, neighbors_z, neighbors_y):
            self.y = y_p
            self.z = z_p
            self.neighbors_z = neighbors_z
            self.neighbors_y = neighbors_y
    """Create new data system where vertices are not duplicated and there is a reference to every neighbor"""
    vertexes = []
    for index, (z, y) in enumerate(zip(points_2d_z, points_2d_y)):
        for vertex in vertexes:
            if abs(vertex.y - y) < 0.01 and abs(vertex.z - z) < 0.01:  # it is a duplicate.
                # Check if new neighbors are found
                triangle_id = index % 3
                triangle_index_start = index - triangle_id

                if triangle_id == 0:  # neighbors are two points left out of 3
                    neighbors_id = [1, 2]
                elif triangle_id == 1:
                    neighbors_id = [0, 2]
                else:
                    neighbors_id = [0, 1]

                is_duplicate_z = points_2d_z[triangle_index_start + neighbors_id[0]] in vertex.neighbors_z
                is_duplicate_y = points_2d_y[triangle_index_start + neighbors_id[0]] in vertex.neighbors_y

                if is_duplicate_z is False and is_duplicate_y is False:
                    vertex.neighbors_z.append(points_2d_z[triangle_index_start + neighbors_id[0]])
                    vertex.neighbors_y.append(points_2d_y[triangle_index_start + neighbors_id[0]])

                is_duplicate_z = points_2d_z[triangle_index_start + neighbors_id[1]] in vertex.neighbors_z
                is_duplicate_y = points_2d_y[triangle_index_start + neighbors_id[1]] in vertex.neighbors_y

                if is_duplicate_z is False and is_duplicate_y is False:
                    vertex.neighbors_z.append(points_2d_z[triangle_index_start + neighbors_id[1]])
                    vertex.neighbors_y.append(points_2d_y[triangle_index_start + neighbors_id[1]])

                break
        else:  # Point is not a duplicate, add it to list
            triangle_id = index % 3
            triangle_index_start = index - triangle_id
            if triangle_id == 0:
                neighbors_id = [1, 2]
            elif triangle_id == 1:
                neighbors_id = [0, 2]
            else:
                neighbors_id = [0, 1]

            if (abs(points_2d_z[triangle_index_start + neighbors_id[0]]  # check if neighbors are the same point
                    - points_2d_z[triangle_index_start + neighbors_id[1]]) < 0.01) and (
                    abs(points_2d_y[triangle_index_start + neighbors_id[0]]
                        - points_2d_y[triangle_index_start + neighbors_id[1]]) < 0.01):
                neighbor_z = [points_2d_z[triangle_index_start + neighbors_id[0]]]
                neighbor_y = [points_2d_y[triangle_index_start + neighbors_id[0]]]
            else:
                neighbor_z = [points_2d_z[triangle_index_start + neighbors_id[0]],
                              points_2d_z[triangle_index_start + neighbors_id[1]]]
                neighbor_y = [points_2d_y[triangle_index_start + neighbors_id[0]],
                              points_2d_y[triangle_index_start + neighbors_id[1]]]

            vertexes.append(Vertex(y, z, neighbor_z, neighbor_y))

    plt.figure(2)
    for vertex in vertexes:
        pyplot.plot(vertex.y, vertex.z, 'o', color='blue')
        for neigh_z, neigh_y in zip(vertex.neighbors_z, vertex.neighbors_y):
            pyplot.plot([vertex.y, neigh_y], [vertex.z, neigh_z], color='red', linewidth=1)
    pyplot.show()
    # # group back into nx3 matrix so that triangles vertices can be joined
    # points_2d_y_collapsed = numpy.reshape(points_2d_y, (-1, 3))
    # points_2d_z_collapsed = numpy.reshape(points_2d_z, (-1, 3))
    #
    # # ---------------plot 2d shape with vectors included--------------
    # plt.figure(2)
    # pyplot.plot(points_2d_y, points_2d_z, 'o', color='blue')
    #
    # for y, z in zip(points_2d_y_collapsed, points_2d_z_collapsed):
    #     pyplot.plot([y[0], y[1]], [z[0], z[1]], color='red', linewidth=1)
    #     pyplot.plot([y[1], y[2]], [z[1], z[2]], color='red', linewidth=1)
    #     pyplot.plot([y[2], y[0]], [z[2], z[0]], color='red', linewidth=1)
    #
    # # pyplot.show()
    # # ----------------------------------------------------------------
    #
    # # -------FIND PATH AROUND 2D IMAGE-------------------------------
    # # -------find starting point in bottom right of image------------
    # z_min_index = numpy.argmin(points_2d_z)
    #
    # y_max_given_z_min = points_2d_y[z_min_index]  # select initial z to compare rest with
    # y_left_given_z_min = points_2d_y[z_min_index]
    # z_min = points_2d_z[z_min_index]
    # z_exact = z_min
    #
    # for y, z in zip(points_2d_y, points_2d_z):
    #     if abs(z - z_min) < 0.1 and y > y_max_given_z_min:
    #         y_max_given_z_min = y
    #         z_exact = z
    #
    # for y, z in zip(points_2d_y, points_2d_z):
    #     if abs(z - z_min) < 0.1 and y < y_left_given_z_min:
    #         y_left_given_z_min = y
    #         z_exact = z
    #
    # home_z = z_exact
    # home_y = y_max_given_z_min
    # angle_previous = numpy.deg2rad(180)
    #
    # counter = 0
    # stop_limit = 12
    # # rough_path_z = numpy.array([home_z])
    # # rough_path_y = numpy.array([home_y])]
    #
    # # end condition for future: (home_z != z_exact and home_y != y_max_given_z_min)
    # while (abs(z_min - home_z) > 0.1 or abs(y_left_given_z_min - home_y) > 0.1 or counter == 0) \
    #         and counter < stop_limit:
    #     # ------find all triangles that contain point---------------------
    #     z_match_indexes = numpy.where(abs(points_2d_z - home_z) < 0.05)
    #     y_match_indexes = numpy.where(abs(points_2d_y - home_y) < 0.05)
    #
    #     match_intersection_indexes = numpy.intersect1d(z_match_indexes, y_match_indexes)
    #
    #     # -------find first neighbor from starting from 0 degrees--------
    #     min_angle = 10
    #     min_difference = numpy.pi * 2
    #     min_neigh_z = 0
    #     min_neigh_y = 0
    #     max_distance = 0
    #
    #     for index in match_intersection_indexes:
    #
    #         triangle_id = index % 3
    #         triangle_index_start = index - triangle_id
    #         neighbors_id = numpy.where([0, 1, 2] != triangle_id)
    #
    #         z_diff_1 = points_2d_z[triangle_index_start + neighbors_id[0][0]] - points_2d_z[index]
    #         y_diff_1 = points_2d_y[triangle_index_start + neighbors_id[0][0]] - points_2d_y[index]
    #         z_diff_2 = points_2d_z[triangle_index_start + neighbors_id[0][1]] - points_2d_z[index]
    #         y_diff_2 = points_2d_y[triangle_index_start + neighbors_id[0][1]] - points_2d_y[index]
    #
    #         if abs(z_diff_1) > 1e-8 or abs(y_diff_1) > 1e-8:
    #             angle_to_neighbor = numpy.arctan2(z_diff_1, y_diff_1)
    #             if angle_to_neighbor < 0:
    #                 angle_to_neighbor += numpy.pi * 2
    #
    #             if angle_to_neighbor > angle_previous + 1e-10:
    #                 difference = angle_to_neighbor - angle_previous
    #             else:
    #                 difference = numpy.pi * 2 - angle_previous + angle_to_neighbor
    #
    #             if abs(difference - min_difference) < 1e-2:
    #                 dist = math.sqrt(z_diff_1 ** 2 + y_diff_1 ** 2)
    #                 if dist > max_distance:
    #                     max_distance = dist
    #                     min_angle = angle_to_neighbor
    #                     min_difference = difference
    #                     min_neigh_z = points_2d_z[triangle_index_start + neighbors_id[0][0]]
    #                     min_neigh_y = points_2d_y[triangle_index_start + neighbors_id[0][0]]
    #             elif difference < min_difference:
    #                 max_distance = math.sqrt(z_diff_1 ** 2 + y_diff_1 ** 2)
    #                 min_angle = angle_to_neighbor
    #                 min_difference = difference
    #                 min_neigh_z = points_2d_z[triangle_index_start + neighbors_id[0][0]]
    #                 min_neigh_y = points_2d_y[triangle_index_start + neighbors_id[0][0]]
    #
    #             # pyplot.plot([points_2d_y[index], points_2d_y[triangle_index_start + neighbors_id[0][0]]]
    #             #             , [points_2d_z[index], points_2d_z[triangle_index_start + neighbors_id[0][0]]]
    #             #             , color='green')
    #
    #         if abs(z_diff_2) > 1e-8 or abs(y_diff_2) > 1e-8:
    #             angle_to_neighbor = numpy.arctan2(z_diff_2, y_diff_2)
    #             if angle_to_neighbor < 0:
    #                 angle_to_neighbor += numpy.pi * 2
    #
    #             if angle_to_neighbor > angle_previous + 1e-10:
    #                 difference = angle_to_neighbor - angle_previous
    #             else:
    #                 difference = numpy.pi * 2 - angle_previous + angle_to_neighbor
    #
    #             if abs(difference - min_difference) < 1e-2:
    #                 dist = math.sqrt(z_diff_2 ** 2 + y_diff_2 ** 2)
    #                 if dist > max_distance:
    #                     max_distance = dist
    #                     min_angle = angle_to_neighbor
    #                     min_difference = difference
    #                     min_neigh_z = points_2d_z[triangle_index_start + neighbors_id[0][1]]
    #                     min_neigh_y = points_2d_y[triangle_index_start + neighbors_id[0][1]]
    #             elif difference < min_difference:
    #                 max_distance = math.sqrt(z_diff_2 ** 2 + y_diff_2 ** 2)
    #                 min_angle = angle_to_neighbor
    #                 min_difference = difference
    #                 min_neigh_z = points_2d_z[triangle_index_start + neighbors_id[0][1]]
    #                 min_neigh_y = points_2d_y[triangle_index_start + neighbors_id[0][1]]
    #
    #             # pyplot.plot([points_2d_y[index], points_2d_y[triangle_index_start + neighbors_id[0][1]]]
    #             #             , [points_2d_z[index], points_2d_z[triangle_index_start + neighbors_id[0][1]]]
    #             #             , color='green')
    #
    #     # pyplot.plot(min_neigh_y, min_neigh_z, 'o', color='orange')
    #     pyplot.plot([home_y, min_neigh_y], [home_z, min_neigh_z], color='black', linewidth=5)
    #
    #     home_z = min_neigh_z
    #     home_y = min_neigh_y
    #
    #     rough_path_z = numpy.append(rough_path_z, home_z)
    #     rough_path_y = numpy.append(rough_path_y, home_y)
    #
    #     counter += 1
    #     angle_previous = (min_angle + numpy.pi) % (2 * numpy.pi)
    #
    # print("Iterations: ")
    # print(counter)
    #
    # for i in range(0, len(rough_path_y) - 1):
    #     for j in range(i + 1, i + 10):
    #         if j >= len(rough_path_y) - 1:
    #             break
    #
    #         intersect_state, d1, d2 = find_intersection(rough_path_y[i], rough_path_z[i]
    #                                                     , rough_path_y[i + 1], rough_path_z[i + 1]
    #                                                     , rough_path_y[j], rough_path_z[j]
    #                                                     , rough_path_y[j + 1], rough_path_z[j + 1])
    #         if intersect_state:
    #             new_point_y = rough_path_y[i] + d1 * (rough_path_y[i + 1] - rough_path_y[i])
    #             new_point_z = rough_path_z[i] + d1 * (rough_path_z[i + 1] - rough_path_z[i])
    #             pyplot.plot(new_point_y, new_point_z, 'o', color='Red')
    #
    # pyplot.show()


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


def is_between(x0, x, x1):
    return (x > x0) and (x < x1)


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
