from src.gen_functions import soma
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n1', '--num_1',
                    default=5,
                    help='primeiro número para somar')
    ap.add_argument('-n2', '--num_2',
                    default=8,
                    help='segundo número para somar')

    args = vars(ap.parse_args())

    num_1 = args['num_1']
    num_2 = args['num_2']
    result = soma(num_1, num_2)
    print('o resultado da soma é: ', result)


if __name__ == '__main__':
    main()
