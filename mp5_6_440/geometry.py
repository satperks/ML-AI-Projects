# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    w = alien.get_width()
    for wall in walls:
        width_to_wall = point_segment_distance(alien.get_centroid(), ((wall[0], wall[1]), (wall[2], wall[3])))
        if width_to_wall <= w:
            return True
        hx, hy = alien.get_head_and_tail()[0]
        tx, ty = alien.get_head_and_tail()[1]
        check_wall = segment_distance(((tx, ty), (hx, hy)), ((wall[0], wall[1]), (wall[2], wall[3]))) <= w
        if (check_wall): return True
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    x, y = alien.get_centroid()
    w, h = window
    r = alien.get_width()
    hx, hy = alien.get_head_and_tail()[0]
    tx, ty = alien.get_head_and_tail()[1]
    # print(hx, hy)
    # print(tx, ty)
    # print (r)
    shape = alien.get_shape()
    if (alien.is_circle()):
        return (r <= x <= w-r and r <= y <= h-r)
    if (shape == 'Horizontal'):
        return (0 < tx-r and hx+r < w) and (r <= y <= h-r)
    if (shape == 'Vertical'):
        return (0 < hy-r and ty+r < h) and (r <= x <= w-r)
    
    return False


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    x, y = point
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    for vertex in polygon:
        xv, yv = vertex
        x_min = min(x_min, xv)
        y_min = min(y_min, yv)
        x_max = max(x_max, xv)
        y_max = max(y_max, yv)

    if x_min <= x <= x_max and y_min <= y <= y_max:
        vectors = []
        for vertex in polygon:
            vertex_x, vertex_y = vertex
            vectors.append((vertex_x - x, vertex_y - y))

        cross_products = []
        for i in range(4):
            j = (i + 1) % 4
            cross_products.append(vectors[i][0] * vectors[j][1] - vectors[i][1] * vectors[j][0])

        all_positive = all(cross_product >= 0 for cross_product in cross_products)
        all_negative = all(cross_product <= 0 for cross_product in cross_products)

        return all_positive or all_negative

    return False


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    hx, hy = alien.get_head_and_tail()[0]
    tx, ty = alien.get_head_and_tail()[1]
    cx, cy = alien.get_centroid()
    path_x = waypoint[0]-cx
    path_y = waypoint[1]-cy
    r = alien.get_width()
    if alien.get_shape() == 'Ball':
        path_seg = (alien.get_centroid(), waypoint)
        for wall in walls:
            wall_seg = ((wall[0], wall[1]), (wall[2], wall[3]))
            if (segment_distance(path_seg, wall_seg) <= r):
                return True
    else:
        for wall in walls:
            wall_seg = ((wall[0], wall[1]), (wall[2], wall[3]))
            if (path_x == 0) and (path_y == 0):
                if (point_segment_distance((hx, hy), wall_seg) <= r or point_segment_distance((tx, ty), wall_seg) <= r):
                    return True
            else:
                if segment_distance(((hx, hy), (hx + path_x, hy + path_y)), wall_seg) <= r or segment_distance(((tx, ty), (tx + path_x, ty + path_y)), wall_seg) <= r:
                    return True
            path_seg = ((hx, hy), (tx, ty), (tx + path_x, ty + path_y), (hx + path_x, ty + path_y))
            if (is_point_in_polygon(wall_seg[0], path_seg)) or (is_point_in_polygon(wall_seg[1], path_seg)):
                return True
    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """

    if (p in s):
        return 0
    
    v_ab = (s[1][0]-s[0][0], s[1][1]-s[0][1]) #vector of start to end of segment
    v_ac = (p[0] - s[0][0], p[1]-s[0][1]) #vector from start of segment to point
    v_bc = (p[0] - s[1][0], p[1]-s[1][1]) #vector from end of segment to point
    

    def dot_product(va, vb): #dot product helper
        return (va[0]*vb[0] + va[1]*vb[1])
    
    def cross_product(va, vb): #cross product helper
        return (va[0]*vb[1] - va[1]*vb[0])
    
    def magnitude(v): #magnitude helper
        return np.sqrt(v[0]*v[0] + v[1]*v[1])

    def straight_line_dist(p1, p2): #straight line distance helper
        y = (p2[1]-p1[1])
        x = (p2[0]-p1[0])
        return np.sqrt(y*y + x*x)
    
    d_ab_bc = (dot_product(v_ab, v_bc))
    d_ab_ac = (dot_product(v_ab, v_ac))


    if (d_ab_bc > 0):
        return straight_line_dist(p, s[1])
    elif (d_ab_ac < 0):
        return straight_line_dist(p, s[0])
    else:
        if (magnitude(v_ab) == 0):
            return straight_line_dist(p, s[0])
        else:
            return abs(float(cross_product(v_ac, v_ab)) / magnitude(v_ab))




def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """ 
    a = s1[0]
    b = s1[1]
    c = s2[0]
    d = s2[1]
    
    def orientation(x, y, z):
        val = (y[1]-x[1]) * (z[0] - y[0]) - (y[0] - x[0]) * (z[1]-y[1])
        if (val > 0):
            return 1
        elif (val < 0):
            return 2
        else: return 0

    def colinear(x, y, z):
        return (y[0] <= max(x[0], z[0]) and y[0] >= min(x[0], z[0]) and
                y[1] <= max(x[1], z[1]) and y[1] >= min(x[1], z[1]))

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if ((o1 != o2) and (o3 != o4)): return True
    if (o1 == 0) and colinear(a, c, b): return True
    if (o2 == 0) and colinear(a, d, b): return True
    if (o3 == 0) and colinear(c, a, d): return True
    if (o4 == 0) and colinear(c, b, d): return True

    return False



def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if (do_segments_intersect(s1, s2)):
        return 0
    
    dist1 = min(point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2))
    dist2 = min(point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1))

    return min(dist1, dist2)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])  
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")