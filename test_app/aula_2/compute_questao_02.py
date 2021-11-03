from src.gen_functions import closest_to_point
import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--focus_point',
                    default=(0,10),
                    help='primeiro ponto para calcular a distancia euclidiana')
    ap.add_argument('-p2', '--points_list',
                    default=[(0,1),(1,8),(0,3)],
                    help='lista de pontos')

    args = vars(ap.parse_args())

    focus_point = args['focus_point']
    points_list = args['points_list']
    result = closest_to_point(focus_point, points_list)
    print('o ponto mais perto é: ', result)


if __name__ == '__main__':
    main()