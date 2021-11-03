import numpy as np


def dist_euclid(point1, point2):
    """ Calculates the Euclidean distance of two points in any dimension
    
    Args:
        point1: a numpy array containing the coordinates of the first point
        point2: a numpy array containing the coordinates of the second point

    Returns:
        The Euclidean distance of the points
    """
    dist = np.linalg.norm(point1 - point2)
    return dist


def closest_to_point(points_list, focus_point):
    """ Calculates which point is closest to the focus point from a list of points

    Args:
        points_list: a numpy array containing the coordinates of the points
        focus_point: a numpy array containing the coordinate of the focus point

    Returns:
        a numpy array containing the coordinate of the closest point
    """
    closest_point_distance = float('inf')
    closest_point = None
    for point in points_list:
        current_point_distance = dist_euclid(focus_point, point)
        if current_point_distance < closest_point_distance:
            closest_point_distance = current_point_distance
            closest_point = point
    return closest_point


def sort_string_list_by_lenght(string_list, reverse=False):
    """ Sorts a list of strings by the number of elements in each item

    Args:
        string_list: list of strings
        reverse: true for descending order

    Returns:
        Returns the sorted list
    """
    sorted_list = string_list.copy()
    if reverse:
        sorted_list.sort(key=lambda x: len(x))
    else:
        sorted_list.sort(key=lambda x: len(x), reverse=True)
    return sorted_list


def questao4(string_list, string_key):
    """ Finds the number of elements equal to the string_key in the string_list

    Args:
        string_list: list of strings
        string_key: string

    Returns:
        Number of elements equal to the string_key in the string_list
    """
    qtd = 0
    for item in string_list:
        if item == string_key:
            qtd += 1
    return qtd


def questao5(string_list):
    """ Calculates the count of each string in a list of strings

    Args:
        string_list: list of strings

    Returns:
        A dictionary containing each string with its count
    """
    string_count = {}
    for item in string_list:
        if item not in string_count:
            count = string_list.count(item)
            string_count[item] = count
    return string_count


def questao06(string_list, analyzed_string, threshold):
    """
    
    Args:
        string_list: 
        analyzed_string: 
        threshold: 

    Returns:

    """
    analyzed_string_indexes = []
    for i in range(len(string_list)):
        if string_list[i] == analyzed_string:
            analyzed_string_indexes.append(i)

    threshold_indexes = set()
    for i in range(1, threshold + 1):
        for p in analyzed_string_indexes:
            new_threshold_index = p - i
            if new_threshold_index >= 0:
                threshold_indexes.add(new_threshold_index)
            new_threshold_index = p + i
            if new_threshold_index <= (len(string_list) - 1):
                threshold_indexes.add(new_threshold_index)

    threshold_indexes = list(threshold_indexes)
    threshold_indexes.sort()

    result_list = []
    for p in threshold_indexes:
        if string_list[p] not in result_list:
            result_list.append(string_list[p])

    return result_list


def questao07(input_list):
    """

    Args:
        input_list:

    Returns:

    """
    no_repeat_list = input_list.copy()
    no_repeat_list.reverse()
    i = 0
    while i < (len(no_repeat_list) - 1):
        current_item = no_repeat_list[i]
        while no_repeat_list.count(current_item) > 1:
            no_repeat_list.remove(current_item)
        i += 1
    no_repeat_list.reverse()
    return no_repeat_list


def region_of_interest(matrix, roi_center, roi_shape):
    """

    Args:
        matrix: matriz numpy
        roi_center: (x,y)
        roi_shape: (height, width)

    Returns:

    """
    x, y = roi_center
    height, width = roi_shape
    roi = matrix[x - (width - 1) // 2: 1 + x + (width - 1) // 2, y - (height - 1) // 2: 1 + y + (height - 1) // 2]
    return roi


def region_of_interest_list(matrix, roi_center_list, roi_shape_list):
    """

    Args:
        matrix:
        roi_center_list:
        roi_shape_list:

    Returns:

    """
    roi_list = []
    for i in range(len(roi_center_list)):
        new_roi = region_of_interest(matrix, roi_center_list[i], roi_shape_list[i])
        roi_list.append(new_roi)
    return roi_list


def questao10(matrix, threshold):
    """

    Args:
        matrix:
        threshold:

    Returns:

    """
    def f(x):
        return 1 if x >= threshold else 0

    return np.vectorize(f)(matrix)
