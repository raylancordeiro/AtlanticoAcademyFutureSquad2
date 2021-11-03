from src.gen_functions import questao07
import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--input_list',
                    default=['uva','uva','uva','uva', 'uva','uva','uva','feijão','arroz'],
                    help='lista de string')

    args = vars(ap.parse_args())

    input_list = args['input_list']

    result = questao07(input_list)
    print('Os elementos únicos são: ', result)


if __name__ == '__main__':
    main()