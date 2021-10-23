from src.gen_functions import dist_euclid
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--point1',
                    default=np.array([0,0]),
                    help='primeiro ponto para calcular a distancia euclidiana')
    ap.add_argument('-p2', '--point2',
                    default=np.array([0,1]),
                    help='segundo ponto para calcular a distancia euclidiana')

    args = vars(ap.parse_args())

    point1 = args['point1']
    point2 = args['point2']
    result = dist_euclid(point1, point2)
    print('a distancia euclidiana dos pontos é: ', result)


if __name__ == '__main__':
    main()
