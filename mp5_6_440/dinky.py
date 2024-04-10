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
import math



def is_alien_within_window(alien: Alien, window: Tuple[int]) -> bool:
    """Determine whether the alien stays within the window
        
    Args:
        alien (Alien): Alien instance
        window (tuple): (width, height) of the window
    """
    # Get the centroid position of the alien
    alien_centroid = alien.get_centroid()

    # Check if the alien is in circle form
    if alien.is_circle():
        # If the alien is in circle form, calculate the distance from the centroid to the window boundaries
        x_dist_to_left = alien_centroid[0] - alien.get_width()
        x_dist_to_right = window[0] - alien_centroid[0] - alien.get_width()
        y_dist_to_top = alien_centroid[1] - alien.get_width()
        y_dist_to_bottom = window[1] - alien_centroid[1] - alien.get_width()

        # Check if the alien is entirely within the window
        if x_dist_to_left >= 0 and x_dist_to_right >= 0 and y_dist_to_top >= 0 and y_dist_to_bottom >= 0:
            return True
    else:
        # If the alien is in oblong form, get its head and tail positions
        head, tail = alien.get_head_and_tail()
        
        # Check if both the head and tail are within the window
        if 0 <= head[0] <= window[0] and 0 <= tail[0] <= window[0] and 0 <= head[1] <= window[1] and 0 <= tail[1] <= window[1]:
            return True

    # If none of the above conditions are met, the alien is not entirely within the window
    return False



def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]) -> bool:
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """

    return False


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall.

    Args:
        alien (Alien): the current alien instance
        walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
        waypoint (tuple): the coordinate of the waypoint where the alien wants to move

    Return:
        True if the path touches a wall, False if not
    """





def is_point_in_polygon(point, polygon):
    x, y = point
    vertices = np.array(polygon)

    # Check if the point is inside the bounding box of the parallelogram
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)

    if x < x_min or x > x_max or y < y_min or y > y_max:
        return False

    # Check if the point is on the same side of all edges using the cross product
    for i in range(vertices.shape[0]):
        j = (i + 1) % vertices.shape[0]
        edge = vertices[j] - vertices[i]
        to_point = np.array([x, y]) - vertices[i]

        cross_product = np.cross(edge, to_point)

        if cross_product < 0:
            return False  # The point is on the opposite side of an edge

    return True  # The point is inside the parallelogram

    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    return False



# is_point_in_polygon Tests
point = (3, 3)
parallelogram = [(1, 1), (5, 1), (4, 4), (2, 4)]
result = is_point_in_polygon(point, parallelogram)
print(result)



def point_segment_distance(p, s):
    vector_start = (s[1][0] - s[0][0], s[1][1] - s[0][1])

    vector_second = (p[0] - s[0][0], p[1] - s[0][1])

    mag = vector_start[0] ** 2 + vector_start[1] ** 2

    proj = (vector_start[0] * vector_second[0] + vector_start[1] * vector_second[1]) / mag

    proj_vector = (proj * vector_start[0], proj * vector_start[1])

    final_point = (s[0][0] + proj_vector[0], s[0][1] + proj_vector[1])

    final_dist = math.sqrt((p[0] - final_point[0]) ** 2 + (p[1] - final_point[1]) ** 2)

    return final_dist

    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    return -1


#Tests for point_segment_distance

p_1 = (2, 2)
s_1 = ((2, 5), (5, 2))
point_segment_distance(p_1, s_1)


def do_segments_intersect(s1, s2):
    x1, y1 = s1[0]
    x2, y2 = s1[1]
    x3, y3 = s2[0]
    x4, y4 = s2[1]


    cross_product_1 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    cross_product_2 = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)
    cross_product_3 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    cross_product_4 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3)


    if (cross_product_1 * cross_product_2 < 0) and (cross_product_3 * cross_product_4 < 0):
        return True
    else:
        return False



    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """

    return False


# Tests for do_segments_intersect

""" s_intersect = ((2, 5), (5, 2))
s_intersect_2 = ((3, 0), (3, 8))
do_segments_intersect(s_intersect, s_intersect_2)

print(do_segments_intersect(s_intersect, s_intersect_2)) """



def segment_distance(s1, s2):

    if(do_segments_intersect(s1, s2)):
        return 0
    
    point_distances = [
        point_segment_distance(s1[0], s2),
        point_segment_distance(s1[1], s2),
        point_segment_distance(s2[0], s1),
        point_segment_distance(s2[1], s1)
    ]

    min_distance = min(point_distances)
    return min_distance



    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    return -1





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
