import matplotlib.pyplot as plt
import numpy
import numpy as np
from stl import mesh
from matplotlib import pyplot
from mpl_toolkits import mplot3d
import math
import random
from operator import attrgetter

p_tol = 0.001


class Vertex:
    def __init__(self, y_p, z_p, neighbours_y, neighbours_z):
        self.y = y_p
        self.z = z_p
        self.neighbours_z = neighbours_z

        self.neighbours_y = neighbours_y


def do_stuff(model_angle):
    your_mesh = mesh.Mesh.from_file('LabradorLowPoly.stl')
    # your_mesh = mesh.Mesh.from_file('cube_1x1.stl')
    # your_mesh = mesh.Mesh.from_file('cubev2.stl')
    # your_mesh = mesh.Mesh.from_file('cubev3.stl')
    # your_mesh = mesh.Mesh.from_file('cubev4.stl')
    # your_mesh = mesh.Mesh.from_file('tool_holder_bars.stl')
    # your_mesh = mesh.Mesh.from_file('scad_chess_pawn.stl')

    """plot stl 3d model--------------------------"""
    # figure3 = pyplot.figure(3)
    # plot_axes = mplot3d.Axes3D(figure3)
    #
    # # # Load the STL files and add the vectors to the plot
    # # your_mesh = mesh.Mesh.from_file('cube_1x1.stl')
    # # your_mesh = mesh.Mesh.from_file('tool_holder_bars.stl')
    #
    # plot_axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
    #
    # # Auto scale to the mesh size
    # scale = your_mesh.points.flatten()
    # plot_axes.auto_scale_xyz(scale, scale, scale)
    # pyplot.title("Stl file displaying")
    # # define plane to project onto ----------------------
    """-------------------------------------------"""
    plane_normal_theta = math.radians(model_angle)
    plane_normal_vector = numpy.array([math.cos(plane_normal_theta), math.sin(plane_normal_theta), 0])
    plane_normal_vector[abs(plane_normal_vector) < 1e-15] = 0.0

    # n = [1, 0, 0]  # direction of plane normal
    # theta = np.arctan2(n[1], n[0])  # orientation of plane

    plane_normal_vector_normalized = plane_normal_vector / np.linalg.norm(plane_normal_vector)
    print("\n-----------start-------------------")
    print("Angle:", model_angle)
    """project all points onto the plane------"""

    points_on_plane = np.zeros([len(your_mesh.vectors), 3, 3])

    for index_tri, triangle in enumerate(your_mesh.vectors):
        for index_vert, vertex in enumerate(triangle):
            q = np.multiply(np.dot(plane_normal_vector_normalized, vertex),
                            plane_normal_vector_normalized)
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
    points_2d_x, points_2d_y = rotate_origin_only((points_on_plane_stretched[:, 0],
                                                   points_on_plane_stretched[:, 1]),
                                                  plane_normal_theta)
    points_2d_z = points_on_plane_stretched[:, 2]

    points_2d_y[abs(points_2d_y) < 1e-6] = 0
    points_2d_z[abs(points_2d_z) < 1e-6] = 0

    print("Projected mesh onto plane")

    """Create new data system where vertices are not duplicated and there is a reference to every neighbour"""
    vertexes = []
    for index, (z, y) in enumerate(zip(points_2d_z, points_2d_y)):
        for vertex in vertexes:
            if abs(vertex.y - y) < p_tol and abs(vertex.z - z) < p_tol:  # it is a duplicate.
                # Check if new neighbours are found
                triangle_id = index % 3
                triangle_index_start = index - triangle_id

                if triangle_id == 0:  # neighbours are two points left out of 3
                    neighbours_id = [1, 2]
                elif triangle_id == 1:
                    neighbours_id = [0, 2]
                else:
                    neighbours_id = [0, 1]

                if abs(points_2d_y[triangle_index_start + neighbours_id[0]] - y) > p_tol or abs(
                        points_2d_z[triangle_index_start + neighbours_id[0]] - z) > p_tol:
                    for y_n, z_n in zip(vertex.neighbours_y, vertex.neighbours_z):
                        if abs(points_2d_z[triangle_index_start + neighbours_id[0]] - z_n) < p_tol and (
                                abs(points_2d_y[triangle_index_start + neighbours_id[0]] - y_n) < p_tol):
                            break
                    else:
                        vertex.neighbours_z.append(points_2d_z[triangle_index_start + neighbours_id[0]])
                        vertex.neighbours_y.append(points_2d_y[triangle_index_start + neighbours_id[0]])

                if abs(points_2d_y[triangle_index_start + neighbours_id[1]] - y) > p_tol or abs(
                        points_2d_z[triangle_index_start + neighbours_id[1]] - z) > p_tol:
                    for y_n, z_n in zip(vertex.neighbours_y, vertex.neighbours_z):
                        if abs(points_2d_z[triangle_index_start + neighbours_id[1]] - z_n) < p_tol and (
                                abs(points_2d_y[triangle_index_start + neighbours_id[1]] - y_n) < p_tol):
                            break
                    else:
                        vertex.neighbours_z.append(points_2d_z[triangle_index_start + neighbours_id[1]])
                        vertex.neighbours_y.append(points_2d_y[triangle_index_start + neighbours_id[1]])
                break
        else:  # Point is not a duplicate, add it to list
            triangle_id = index % 3
            triangle_index_start = index - triangle_id
            if triangle_id == 0:
                neighbours_id = [1, 2]
            elif triangle_id == 1:
                neighbours_id = [0, 2]
            else:
                neighbours_id = [0, 1]

            neighbour_z0 = points_2d_z[triangle_index_start + neighbours_id[0]]
            neighbour_y0 = points_2d_y[triangle_index_start + neighbours_id[0]]

            neighbour_z1 = points_2d_z[triangle_index_start + neighbours_id[1]]
            neighbour_y1 = points_2d_y[triangle_index_start + neighbours_id[1]]

            neigh_z = []
            neigh_y = []
            if z != neighbour_z0 or y != neighbour_y0:
                neigh_z = [neighbour_z0]
                neigh_y = [neighbour_y0]
                # check if neighbours are the same point
                if abs(neighbour_z0 - neighbour_z1) > p_tol or abs(
                        neighbour_y0 - neighbour_y1) > p_tol:
                    if z != neighbour_z1 or y != neighbour_y1:
                        neigh_z.append(neighbour_z1)
                        neigh_y.append(neighbour_y1)
            else:
                neigh_z.append(neighbour_z1)
                neigh_y.append(neighbour_y1)

            vertexes.append(Vertex(y, z, neigh_y, neigh_z))

    print("Translated data system into class")

    # plt.figure(2)
    # for vertex in vertexes:
    #     pyplot.plot(vertex.y, vertex.z, 'o', color='blue')
    #     for neigh_z, neigh_y in zip(vertex.neighbours_z, vertex.neighbours_y):
    #         pyplot.plot([vertex.y, neigh_y], [vertex.z, neigh_z], color='red', linewidth=1)
    # pyplot.show()
    # print_matrix(vertexes)
    """delete all interior vertices by checking the following:"""
    # if the two points at end of edge only have one common neighbour then the edge is on the outside
    # therefore both the points are outer vertices
    # If the two points have more than one neighbour in common check which sides of the line the neighbours lie on
    # If all the common neighbours lie on the same side it is also a exterior edge

    outer_edges = []

    for each_vertex in vertexes:
        for neighbour_vert_y, neighbour_vert_z in zip(each_vertex.neighbours_y, each_vertex.neighbours_z):
            vert_2_index = find_vert(vertexes, neighbour_vert_y, neighbour_vert_z)
            common_y, common_z = find_common(each_vertex, vertexes[vert_2_index])

            if len(common_y) == 1:
                if each_vertex not in outer_edges:
                    outer_edges.append([each_vertex, vertexes[vert_2_index]])
            else:
                old_sign = 0
                inner_edge = False
                for y, z in zip(common_y, common_z):
                    a = each_vertex.z - vertexes[vert_2_index].z
                    b = vertexes[vert_2_index].y - each_vertex.y
                    c = (vertexes[vert_2_index].z - each_vertex.z) * each_vertex.y - \
                        (vertexes[vert_2_index].y - each_vertex.y) * each_vertex.z

                    f = a * y + b * z + c
                    if f < 0:
                        sign = -1
                    else:
                        sign = 1

                    if old_sign == 0:
                        old_sign = sign
                    elif sign != old_sign:
                        inner_edge = True

                if not inner_edge:
                    outer_edges.append([each_vertex, vertexes[vert_2_index]])

    print("Interior vertexes are deleted")

    # for edge in outer_edges:
    #     pyplot.plot(edge[0].y, edge[0].z, 'o', color='blue')
    #     pyplot.plot(edge[1].y, edge[1].z, 'o', color='blue')
    #
    #     pyplot.plot([edge[0].y, edge[1].y], [edge[0].z, edge[1].z], color='red', linewidth=1)
    #
    # # pyplot.show()

    """Reconstruct edges"""
    outer_vertexes = [Vertex(outer_edges[0][0].y, outer_edges[0][0].z, [outer_edges[0][1].y], [outer_edges[0][1].z]),
                      Vertex(outer_edges[0][1].y, outer_edges[0][1].z, [outer_edges[0][0].y], [outer_edges[0][0].z])]
    for edge_i in outer_edges:
        for edge_o in outer_edges:
            if edge_o is not edge_i:
                if edge_i[0] is edge_o[1]:
                    reconstruct_vert(outer_vertexes, edge_i[0], edge_i[1], edge_o[0])
                elif edge_i[1] is edge_o[0]:
                    reconstruct_vert(outer_vertexes, edge_i[1], edge_i[0], edge_o[1])
                elif edge_i[0] is edge_o[0]:
                    reconstruct_vert(outer_vertexes, edge_i[0], edge_i[1], edge_o[1])
                elif edge_i[1] is edge_o[1]:
                    reconstruct_vert(outer_vertexes, edge_i[1], edge_i[0], edge_o[0])
    # print_matrix(outer_vertexes)

    print("Reconstructed outer vertexes using outer edges")

    """--------chain neighbours on straight lines instead of them jumping over multiple vertices------"""
    z_min_vtx = min(outer_vertexes, key=attrgetter('z'))
    z_min = z_min_vtx.z
    bottom_right_index = outer_vertexes.index(z_min_vtx)

    y_max_given_z_min = -1000

    for index, vertex in enumerate(outer_vertexes):
        if abs(vertex.z - z_min) < p_tol and vertex.y > y_max_given_z_min:
            y_max_given_z_min = vertex.y
            bottom_right_index = index

    queue = [bottom_right_index]
    completed_list_y = []
    completed_list_z = []
    while len(queue) > 0:
        shorten_algorithm(outer_vertexes, queue[0], completed_list_y, completed_list_z)

        for y, z in zip(outer_vertexes[queue[0]].neighbours_y, outer_vertexes[queue[0]].neighbours_z):
            if not is_coord_in_lists(y, z, completed_list_y, completed_list_z):
                neigh_id = find_vert(outer_vertexes, y, z)
                if neigh_id not in queue:
                    queue.append(neigh_id)

        queue.pop(0)

    print_matrix(outer_vertexes)
    pyplot.show()
    """---------Add intersections as points and neighbours------------------"""

    first = 0
    j = 0
    f_neigh_i = 0
    while True:  # for first, vert in enumerate(outer_vertexes):
        # print(len(outer_vertexes))
        for first_neigh_i, (first_neigh_y, first_neigh_z) in enumerate(
                zip(outer_vertexes[first].neighbours_y, outer_vertexes[first].neighbours_z)):
            i = 2
            while True:  # for i in range(first + 1, len(outer_vertexes) - 1):
                for col_neigh_y, col_neigh_z in zip(outer_vertexes[i].neighbours_y, outer_vertexes[i].neighbours_z):

                    intersect_state, d1, d2 = find_intersection(outer_vertexes[first].y, outer_vertexes[first].z,
                                                                first_neigh_y, first_neigh_z,
                                                                outer_vertexes[i].y, outer_vertexes[i].z,
                                                                col_neigh_y, col_neigh_z)
                    if intersect_state:

                        for index_j, vert in enumerate(outer_vertexes):
                            if vert.y == col_neigh_y and vert.z == col_neigh_z:
                                j = index_j
                                break

                        for index_j, vert in enumerate(outer_vertexes):
                            if vert.y == first_neigh_y and vert.z == first_neigh_z:
                                f_neigh_i = index_j
                                break

                        new_point_y = outer_vertexes[first].y + d1 * (first_neigh_y - outer_vertexes[first].y)
                        new_point_z = outer_vertexes[first].z + d1 * (first_neigh_z - outer_vertexes[first].z)

                        new = False
                        new = new or replace_neighbours(outer_vertexes[first], outer_vertexes[f_neigh_i],
                                                        new_point_y, new_point_z)
                        new = new or replace_neighbours(outer_vertexes[f_neigh_i], outer_vertexes[first],
                                                        new_point_y, new_point_z)
                        new = new or replace_neighbours(outer_vertexes[i], outer_vertexes[j], new_point_y, new_point_z)
                        new = new or replace_neighbours(outer_vertexes[j], outer_vertexes[i], new_point_y, new_point_z)

                        if not new:
                            neighs_y = [outer_vertexes[first].y, outer_vertexes[f_neigh_i].y,
                                        outer_vertexes[i].y, outer_vertexes[j].y]
                            neighs_z = [outer_vertexes[first].z, outer_vertexes[f_neigh_i].z,
                                        outer_vertexes[i].z, outer_vertexes[j].z]
                            # pyplot.plot(new_point_y, new_point_z, 'o', color='Green')
                            outer_vertexes.insert(first, Vertex(new_point_y, new_point_z, neighs_y, neighs_z))

                i += 1
                if i >= len(outer_vertexes):
                    break
        first += 1
        if first >= len(outer_vertexes) - 3:
            break

    # print_matrix(outer_vertexes)
    print("Intersections Added")

    """---------------Find bottom left and right points-------------------------"""
    z_min_vtx = min(outer_vertexes, key=attrgetter('z'))
    z_min = z_min_vtx.z
    bottom_right_index = outer_vertexes.index(z_min_vtx)

    y_max_given_z_min = -1000
    y_min_given_z_min = 1000

    for index, vertex in enumerate(outer_vertexes):
        if abs(vertex.z - z_min) < p_tol and vertex.y < y_min_given_z_min:
            y_min_given_z_min = vertex.y

    for index, vertex in enumerate(outer_vertexes):
        if abs(vertex.z - z_min) < p_tol and vertex.y > y_max_given_z_min:
            y_max_given_z_min = vertex.y
            bottom_right_index = index

    min_angle = 10
    current_vert = outer_vertexes[bottom_right_index]

    min_neigh_z = current_vert.neighbours_z[0]
    min_neigh_y = current_vert.neighbours_y[0]

    counter = 0
    stop_limit = 50
    angle_previous = numpy.deg2rad(180)
    pyplot.figure(6)
    print_matrix(outer_vertexes)
    while (abs(z_min - current_vert.z) > p_tol or abs(y_min_given_z_min - current_vert.y) > p_tol or counter == 0) and (
            counter < stop_limit):
        min_difference = numpy.pi * 3
        min_dist = 100000
        for neigh_z, neigh_y in zip(current_vert.neighbours_z, current_vert.neighbours_y):
            z_diff_1 = neigh_z - current_vert.z
            y_diff_1 = neigh_y - current_vert.y

            angle_to_neighbor = numpy.arctan2(z_diff_1, y_diff_1)
            if angle_to_neighbor < 0:
                angle_to_neighbor += numpy.pi * 2

            if angle_to_neighbor > angle_previous + 1e-10:
                difference = angle_to_neighbor - angle_previous
            else:
                difference = numpy.pi * 2 - angle_previous + angle_to_neighbor

            if abs(difference - min_difference) < 1e-2:  # if it lies in same direction check distance
                dist = z_diff_1 ** 2 + y_diff_1 ** 2
                if dist < min_dist:
                    min_dist = dist
                    min_angle = angle_to_neighbor
                    min_difference = difference
                    min_neigh_z = neigh_z
                    min_neigh_y = neigh_y
            elif difference < min_difference:  # has closer angle in counterclockwise direction
                min_dist = z_diff_1 ** 2 + y_diff_1 ** 2
                min_angle = angle_to_neighbor
                min_difference = difference
                min_neigh_z = neigh_z
                min_neigh_y = neigh_y

        pyplot.plot([current_vert.y, min_neigh_y],
                    [current_vert.z, min_neigh_z],
                    color='black', linewidth=4)
        pyplot.plot(min_neigh_y, min_neigh_z, 'o', color='red')

        counter += 1
        angle_previous = (min_angle + numpy.pi) % (2 * numpy.pi)
        for vertex in outer_vertexes:
            if vertex.y == min_neigh_y and vertex.z == min_neigh_z:
                current_vert = vertex
                break
    print("Path length: ", counter)
    plt.show()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def shorten_algorithm(vert_list, trgt_index, completed_vert_list_y, completed_vert_list_z):
    directions = []
    max_distances = []  # max for each direction
    points_sorted_by_direction_y = []
    points_sorted_by_direction_z = []
    points_sorted_by_direction_ids = []
    max_dist = 0  # max for all directions

    # create list with unique directions
    # create point lists with rows corresponding to unique directions and columns all points in that direction
    for neigh_id, (neigh_y, neigh_z) in enumerate(zip(vert_list[trgt_index].neighbours_y,
                                                      vert_list[trgt_index].neighbours_z)):
        if is_coord_in_lists(neigh_y, neigh_z, completed_vert_list_y, completed_vert_list_z):
            continue

        z_diff = neigh_z - vert_list[trgt_index].z
        y_diff = neigh_y - vert_list[trgt_index].y

        dist_sq = z_diff ** 2 + y_diff ** 2

        current_dir = numpy.arctan2(z_diff, y_diff)
        pos_in_list = where_num_in_list(directions, current_dir)

        neigh_full_id = find_vert(vert_list, neigh_y, neigh_z)
        if pos_in_list == -1:
            points_sorted_by_direction_y.append([neigh_y])
            points_sorted_by_direction_z.append([neigh_z])
            points_sorted_by_direction_ids.append([neigh_full_id])
            directions.append(current_dir)
            max_distances.append(dist_sq)
        else:
            points_sorted_by_direction_y[pos_in_list].append(neigh_y)
            points_sorted_by_direction_z[pos_in_list].append(neigh_z)
            points_sorted_by_direction_ids[pos_in_list].append(neigh_full_id)

            if dist_sq > max_distances[pos_in_list]:
                max_distances[pos_in_list] = dist_sq
    if len(max_distances) > 0:
        max_dist = max(max_distances)

    # add points that arent neighbours
    for pt_id, pt in enumerate(vert_list):

        if is_coord_in_lists(pt.y, pt.z, completed_vert_list_y, completed_vert_list_z):
            continue

        z_diff = vert_list[trgt_index].z - pt.z
        y_diff = vert_list[trgt_index].y - pt.y

        if z_diff == 0 and y_diff == 0:  # current pt matched trgt pt
            continue

        dist_sq = z_diff ** 2 + y_diff ** 2

        if dist_sq < max_dist:  # check that point is within max dist circle
            current_dir = numpy.arctan2(z_diff, y_diff)
            pos_in_list = where_num_in_list(directions, current_dir)
            if pos_in_list > -1 and dist_sq < max_distances[pos_in_list]:
                # point matches a direction and is within max dist
                points_sorted_by_direction_y[pos_in_list].append(pt.y)
                points_sorted_by_direction_z[pos_in_list].append(pt.z)
                points_sorted_by_direction_ids[pos_in_list].append(pt_id)

    if len(points_sorted_by_direction_z) == 0:
        return False

    closest_list_y = []
    closest_list_z = []

    for dir_i, theta in enumerate(directions):
        min_dist = 1000000
        closest_y = 0
        closest_z = 0

        for point_y, point_z in zip(points_sorted_by_direction_y[dir_i], points_sorted_by_direction_z[dir_i]):
            z_diff = vert_list[trgt_index].z - point_z
            y_diff = vert_list[trgt_index].y - point_y

            dist_sq = z_diff ** 2 + y_diff ** 2

            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_y = point_y
                closest_z = point_z
        # delete trgt from all points
        # if len(points_sorted_by_direction_z) > 1:
        delete_neighbours(vert_list, points_sorted_by_direction_ids[dir_i], [trgt_index], 0)

        closest_list_z.append(closest_z)
        closest_list_y.append(closest_y)

        # add all points to closest
        closest_id = find_vert(vert_list, closest_y, closest_z)
        # add all pts to closest except itself
        for point_y, point_z in zip(points_sorted_by_direction_y[dir_i], points_sorted_by_direction_z[dir_i]):
            if point_z != closest_z or point_y != closest_y:  # dont add itself as a neighbor
                add_neighbour(vert_list[closest_id], point_y, point_z)
        # add target to shortest
        add_neighbour(vert_list[closest_id], vert_list[trgt_index].y, vert_list[trgt_index].z)

        delete_neighbours(vert_list, [trgt_index], points_sorted_by_direction_ids[dir_i], 0)
        add_neighbour(vert_list[trgt_index], closest_y, closest_z)

    completed_vert_list_z.append(vert_list[trgt_index].z)
    completed_vert_list_y.append(vert_list[trgt_index].y)

    return True


def is_coord_in_lists(y, z, list_y, list_z):
    for l_y, l_z in zip(list_y, list_z):
        if abs(l_y - y) < p_tol and abs(l_z - z) < p_tol:
            return True
    else:
        return False


def find_pt(vertex_list, pt_y, pt_z):
    for index, vertex in enumerate(vertex_list):
        if abs(vertex.z - pt_z) < p_tol and abs(vertex.y - pt_y) < p_tol:
            return index, vertex
    else:
        return None


def where_num_in_list(num_list, number):
    for i, p in enumerate(num_list):
        if abs(p - number) < 0.0001:
            return i
    else:
        return -1


def print_matrix(matrix):
    for vertex in matrix:
        pyplot.plot(vertex.y, vertex.z, 'o', color='blue')
        for neigh_z, neigh_y in zip(vertex.neighbours_z, vertex.neighbours_y):
            r = random.random()
            b = random.random()
            g = random.random()
            c = (r, g, b)
            pyplot.plot([vertex.y, neigh_y], [vertex.z, neigh_z], color=c, linewidth=3)
    # pyplot.show()


def find_common(vert1, vert2):
    common_y = []
    common_z = []
    for neigh_1_y, neigh_1_z in zip(vert1.neighbours_y, vert1.neighbours_z):
        for neigh_2_y, neigh_2_z in zip(vert2.neighbours_y, vert2.neighbours_z):
            if neigh_1_y == neigh_2_y and neigh_2_z == neigh_1_z:
                common_y.append(neigh_1_y)
                common_z.append(neigh_1_z)

    return common_y, common_z


def find_vert(vertexes_a, p_y, p_z):
    for index, vert in enumerate(vertexes_a):
        if abs(vert.y - p_y) < p_tol and abs(vert.z - p_z) < p_tol:
            return index
    else:
        raise ValueError("NO vertex found")


def delete_neighbours(vertexes, main_ids, to_del_ids, current_pos):
    if len(to_del_ids) == 0:
        return 0
    # cannot use count if multiple main ids
    count = 0
    for main_id in main_ids:
        length = len(vertexes[main_id].neighbours_y)
        for i, (y, z) in enumerate(
                zip(reversed(vertexes[main_id].neighbours_y), reversed(vertexes[main_id].neighbours_z))):
            j = length - i - 1
            for del_id in to_del_ids:
                if abs(y - vertexes[del_id].y) < p_tol and abs(z - vertexes[del_id].z) < p_tol:
                    vertexes[main_id].neighbours_y.pop(j)
                    vertexes[main_id].neighbours_z.pop(j)

                    if j <= current_pos:
                        count += 1

                    break

    return count


def reconstruct_vert(vertexes, pt_com, pt_1, pt_2):
    for index, vert in enumerate(vertexes):
        if pt_com.y == vert.y and pt_com.z == vert.z:
            add_neighbour(vertexes[index], pt_1.y, pt_1.z)
            add_neighbour(vertexes[index], pt_2.y, pt_2.z)
            break
    else:
        vertexes.append(Vertex(pt_com.y, pt_com.z, [pt_1.y, pt_2.y], [pt_1.z, pt_2.z]))

    for index, vert in enumerate(vertexes):
        if pt_1.y == vert.y and pt_1.z == vert.z:
            add_neighbour(vertexes[index], pt_com.y, pt_com.z)
            break

    for index, vert in enumerate(vertexes):
        if pt_2.y == vert.y and pt_2.z == vert.z:
            add_neighbour(vertexes[index], pt_com.y, pt_com.z)
            break


def add_vertex(vertexes, new_vert):
    for vert in vertexes:
        if vert is new_vert:
            break
    else:
        vertexes.append(new_vert)
    return None


def add_neighbour(vertex, new_y, new_z):
    for neigh_y, neigh_z in zip(vertex.neighbours_y, vertex.neighbours_z):
        if abs(neigh_y - new_y) < p_tol and abs(neigh_z - new_z) < p_tol:
            break
    else:
        vertex.neighbours_y.append(new_y)
        vertex.neighbours_z.append(new_z)

    return None


def insert_neighbour(vertex, new_y, new_z, pos):
    for index, (neigh_y, neigh_z) in enumerate(zip(vertex.neighbours_y, vertex.neighbours_z)):
        if abs(neigh_y - new_y) < p_tol and abs(neigh_z - new_z) < p_tol:
            return False

    else:
        vertex.neighbours_y.insert(pos, new_y)
        vertex.neighbours_z.insert(pos, new_z)
    return True


def replace_neighbours(vertex, old, new_y, new_z):
    for index, (neigh_y, neigh_z) in enumerate(zip(vertex.neighbours_y, vertex.neighbours_z)):
        if neigh_y == old.y and neigh_z == old.z:
            pos = index
            vertex.neighbours_z[pos] = new_z
            vertex.neighbours_y[pos] = new_y
            return False
    else:
        return True


def is_c_between(ay, az, by, bz, cy, cz):
    # Check if c is between a and b
    d_ab = (ay - by) ** 2 + (az - bz) ** 2
    d_ac = (ay - cy) ** 2 + (az - cz) ** 2
    d_bc = (by - cy) ** 2 + (bz - cz) ** 2

    if max(d_ac, d_bc) > d_ab:
        return False
    else:
        return True


def sq_shortest_dist_to_point(ax, ay, bx, by, px, py):
    dx = bx - ax
    dy = by - ay

    p1x = px - ax
    p2x = px - bx
    p1y = py - ay
    p2y = py - by

    if (dx == 0 and dy == 0) or (p1x == 0 and p1y == 0) or (p2x == 0 and p2y == 0):
        return 1000

    dr2 = float(dx ** 2 + dy ** 2)

    lerp = ((px - ax) * dx + (py - ay) * dy) / dr2

    x = lerp * dx + ax
    y = lerp * dy + ay

    _dx = x - px
    _dy = y - py
    square_dist = _dx ** 2 + _dy ** 2
    return square_dist


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
    ab = 0.0

    if denom == 0:
        xy = -1
        ab = -1
    else:
        xy = (a0 * (y1 - b1) + a1 * (b0 - y1) + x1 * (b1 - b0)) / denom
        partial = is_between(0, xy, 1)
        if partial:
            ab = (y1 * (x0 - a1) + b1 * (x1 - x0) + y0 * (a1 - x1)) / denom

    if partial and is_between(0, ab, 1) and ab > p_tol and xy > p_tol:
        ab = 1 - ab
        xy = 1 - xy
        if ab > p_tol and xy > p_tol:
            return True, xy, ab
        else:
            return False, 0, 0
    else:
        return False, 0, 0


if __name__ == '__main__':
    # toolbox bug angle: 341.05263157894734
    # dog bug angle 75.78947368421052
    # dog glitch angle 132.6315789473684
    # 151.57894736842104

    # for i in numpy.linspace(0, 360, 20):
    do_stuff(90)
