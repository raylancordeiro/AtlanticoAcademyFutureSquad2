from src.gen_functions import questao5
import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--string_list',
                    default=['abacate','abacate','uva','banana', 'abacate','repolho','uva','feijão','Pera'],
                    help='lista de string')

    args = vars(ap.parse_args())

    string_list = args['string_list']

    result = questao5(string_list)
    print('As quantidades são: ', result)


if __name__ == '__main__':
    main()