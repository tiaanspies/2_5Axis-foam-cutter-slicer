import matplotlib.pyplot as plt
import numpy
import numpy as np
from stl import mesh
from matplotlib import pyplot
from mpl_toolkits import mplot3d
import math
import random

p_tol = 0.01


class Vertex:
    def __init__(self, y_p, z_p, neighbours_y, neighbours_z):
        self.y = y_p
        self.z = z_p
        self.neighbours_z = neighbours_z
        self.neighbours_y = neighbours_y


def do_stuff():
    # your_mesh = mesh.Mesh.from_file('LabradorLowPoly.stl')
    # your_mesh = mesh.Mesh.from_file('cube_1x1.stl')
    # your_mesh = mesh.Mesh.from_file('cubev2.stl')
    your_mesh = mesh.Mesh.from_file('cubev3.stl')
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
    plane_normal_theta = math.radians(0)
    plane_normal_vector = numpy.array([math.cos(plane_normal_theta), math.sin(plane_normal_theta), 0])
    plane_normal_vector[abs(plane_normal_vector) < 1e-15] = 0.0

    # n = [1, 0, 0]  # direction of plane normal
    # theta = np.arctan2(n[1], n[0])  # orientation of plane

    plane_normal_vector_normalized = plane_normal_vector / np.linalg.norm(plane_normal_vector)
    print("start")
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

    # plt.figure(2)
    # for vertex in vertexes:
    #     pyplot.plot(vertex.y, vertex.z, 'o', color='blue')
    #     for neigh_z, neigh_y in zip(vertex.neighbours_z, vertex.neighbours_y):
    #         pyplot.plot([vertex.y, neigh_y], [vertex.z, neigh_z], color='red', linewidth=1)
    # # pyplot.show()
    """delete all interior vertices by checking the following:"""
    # if the two points at end of edge only have one common neighbour then the edge is on the outside
    # therefore both the points are outer vertices
    # If the two points have more than one neighbour in common check which sides of the line the neighbours lie on
    # If all the common neighbours lie on the same side it is also a exterior edge

    print("Translated data system into class")

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
    # pyplot.figure(5)
    # for vertex in outer_vertexes:
    #     pyplot.plot(vertex.y, vertex.z, 'o', color='blue')
    #     for neigh_z, neigh_y in zip(vertex.neighbours_z, vertex.neighbours_y):
    #         pyplot.plot([vertex.y, neigh_y], [vertex.z, neigh_z], color='red', linewidth=1)
    #
    # pyplot.show()

    print("Reconstructed outer vertexes using outer edges")

    """chain neighbours on straight lines instead of them jumping over multiple vertices"""

    for close_i, close_vtx in enumerate(outer_vertexes):
        far_i = 0
        while True:
            far_y = close_vtx.neighbours_y[far_i]
            far_z = close_vtx.neighbours_z[far_i]
        # for far_i, (far_y, far_z) in enumerate(zip(close_vtx.neighbours_y, close_vtx.neighbours_z)):
            intersecting_pt_ids = []
            for mid_id, mid_vtx in enumerate(outer_vertexes):
                if mid_id == close_i:
                    continue

                if sq_shortest_dist_to_point(close_vtx.y, close_vtx.z, far_y, far_z, mid_vtx.y, mid_vtx.z) < 0.01:
                    if is_c_between(close_vtx.y, close_vtx.z, far_y, far_z, mid_vtx.y, mid_vtx.z):
                        intersecting_pt_ids.append(mid_id)

            if len(intersecting_pt_ids) > 0:
                min_dist = 1000000
                min_i = intersecting_pt_ids[0]
                closest_id = 0
                for index, pt in enumerate(intersecting_pt_ids):
                    d2 = (outer_vertexes[pt].y-close_vtx.y)**2 + (outer_vertexes[pt].z-close_vtx.z)**2
                    if d2 < min_dist:
                        min_dist = d2
                        min_i = pt
                        closest_id = index

                add_neighbour(outer_vertexes[min_i], close_vtx.y, close_vtx.z)  #
                add_neighbour(outer_vertexes[min_i], far_y, far_z)

                far_i -= len(intersecting_pt_ids)
                intersecting_pt_ids.pop(closest_id)

                add_neighbour(outer_vertexes[close_i], outer_vertexes[min_i].y, outer_vertexes[min_i].z)

                for i, vert in enumerate(outer_vertexes):
                    if abs(far_y - vert.y) < p_tol and abs(far_z - vert.z) < p_tol:
                        add_neighbour(outer_vertexes[i], outer_vertexes[min_i].y, outer_vertexes[min_i].z)
                        delete_neighbours(outer_vertexes, i, [close_i])
                        delete_neighbours(outer_vertexes, close_i, [i])
                        delete_neighbours(outer_vertexes, close_i, intersecting_pt_ids)
                        break

            far_i += 1
            if far_i >= len(close_vtx.neighbours_y):
                break

    print("Shortening complete")
    pyplot.figure(6)
    r = random.random()
    b = random.random()
    g = random.random()
    c = (r, g, b)
    for vertex in outer_vertexes:
        pyplot.plot(vertex.y, vertex.z, 'o', color='blue')
        for neigh_z, neigh_y in zip(vertex.neighbours_z, vertex.neighbours_y):
            pyplot.plot([vertex.y, neigh_y], [vertex.z, neigh_z], color=c, linewidth=3)
            r = random.random()
            b = random.random()
            g = random.random()
            c = (r, g, b)

    pyplot.show()
    """---------------Find bottom left and right points-------------------------"""
    # z_min = min(vertexes, key=attrgetter('z')).z
    #
    # # y_max_given_z_min = points_2d_y[z_min_index]  # select initial z to compare rest with
    # # y_left_given_z_min = points_2d_y[z_min_index]
    # z_exact = z_min
    # y_max_given_z_min = -1000
    # y_min_given_z_min = 1000
    #
    # for vertex in vertexes:
    #     if abs(vertex.z - z_min) < 0.1 and vertex.y < y_min_given_z_min:
    #         y_min_given_z_min = vertex.y
    #         z_exact = vertex.z
    #
    # for vertex in vertexes:
    #     if abs(vertex.z - z_min) < 0.1 and vertex.y > y_max_given_z_min:
    #         y_max_given_z_min = vertex.y
    #         z_exact = vertex.z

    """---------Add intersections as points and neighbours------------------"""
    # p = 0
    #
    # while True:  # for p, vert in enumerate(ro_vertexes):
    #     i = 2
    #     while True:  # for i in range(p + 1, len(ro_vertexes) - 1):
    #         for col_neigh_y, col_neigh_z in zip(ro_vertexes[i].neighbours_y, ro_vertexes[i].neighbours_z):
    #             intersect_state, d1, d2 = find_intersection(ro_vertexes[p].y, ro_vertexes[p].z,
    #                                                         ro_vertexes[p + 1].y, ro_vertexes[p + 1].z,
    #                                                         ro_vertexes[i].y, ro_vertexes[i].z,
    #                                                         col_neigh_y, col_neigh_z)
    #             if intersect_state:
    #                 if i < len(ro_vertexes)-1:
    #                     if ro_vertexes[i + 1].y == col_neigh_y and ro_vertexes[i + 1].z == col_neigh_z:
    #                         j = i+1
    #                 else:
    #                     for index_j, vert in enumerate(ro_vertexes):
    #                         if vert.y == col_neigh_y and vert.z == col_neigh_y:
    #                             j = index_j
    #                             break
    #
    #                 new_point_y = ro_vertexes[p].y + d1 * (ro_vertexes[p + 1].y - ro_vertexes[p].y)
    #                 new_point_z = ro_vertexes[p].z + d1 * (ro_vertexes[p + 1].z - ro_vertexes[p].z)
    #
    #                 new = False
    #                 new = new or replace_neighbours(ro_vertexes[p], ro_vertexes[p + 1], new_point_y, new_point_z)
    #                 new = new or replace_neighbours(ro_vertexes[p + 1], ro_vertexes[p], new_point_y, new_point_z)
    #                 new = new or replace_neighbours(ro_vertexes[i], ro_vertexes[j], new_point_y, new_point_z)
    #                 new = new or replace_neighbours(ro_vertexes[j], ro_vertexes[i], new_point_y, new_point_z)
    #
    #                 if not new:
    #                     neighs_y = [ro_vertexes[p].y, ro_vertexes[p + 1].y, ro_vertexes[i].y, ro_vertexes[j].y]
    #                     neighs_z = [ro_vertexes[p].z, ro_vertexes[p + 1].z, ro_vertexes[i].z, ro_vertexes[j].z]
    #                     pyplot.plot(new_point_y, new_point_z, 'o', color='Green')
    #                     ro_vertexes.insert(p + 1, Vertex(new_point_y, new_point_z, neighs_y, neighs_z))
    #
    #         i += 1
    #         if i >= len(ro_vertexes):
    #             break
    #     p += 1
    #     if p >= len(ro_vertexes) - 3:
    #         break
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


def delete_neighbours(vertexes, main_id, to_del_ids):

    if len(to_del_ids) == 0:
        return None

    for del_id in to_del_ids:
        for i, (y, z) in enumerate(zip(vertexes[main_id].neighbours_y, vertexes[main_id].neighbours_z)):
            if abs(y - vertexes[del_id].y) < p_tol and abs(z - vertexes[del_id].z) < p_tol:
                vertexes[main_id].neighbours_y.pop(i)
                vertexes[main_id].neighbours_z.pop(i)
                break


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
    d_ab = (ay - by)**2 + (az - bz)**2
    d_ac = (ay - cy)**2 + (az - cz)**2
    d_bc = (by - cy)**2 + (bz - cz)**2

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
    do_stuff()
