from src.gen_functions import region_of_interest
import argparse
import numpy as np


def main():
    default_matrix = np.array([[1, 2, 3, 4, 5],
                               [6, 7, 8, 9, 10],
                               [11, 12, 13, 14, 15],
                               [16, 17, 18, 19, 20],
                               [21, 22, 23, 24, 25]])

    ap = argparse.ArgumentParser()
    ap.add_argument('-ma', '--matrix',
                    default=default_matrix,
                    help='')
    ap.add_argument('-rc', '--roi_center',
                    default=(1, 1),
                    help='')
    ap.add_argument('-rs', '--roi_shape',
                    default=(3, 3),
                    help='')

    args = vars(ap.parse_args())

    matrix = args['matrix']
    roi_center = args['roi_center']
    roi_shape = args['roi_shape']
    result = region_of_interest(matrix, roi_center, roi_shape)
    print('Original matrix:\n{}'.format(matrix))
    print('ROI matrix:\n{}'.format(result))


if __name__ == '__main__':
    main()
