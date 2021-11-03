from src.gen_functions import questao06
import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--string_list',
                    default=['abacate','pera','uva','banana', 'maça','repolho','uva','feijão','arroz'],
                    help='lista de string')
    ap.add_argument('-p2', '--analyzed_string',
                    default='arroz',
                    help='palavra que será analisada')
    ap.add_argument('-p3', '--threshold',
                    default=3,
                    help='limiar de distância de análise')

    args = vars(ap.parse_args())

    string_list = args['string_list']
    analyzed_string = args['analyzed_string']
    threshold = args['threshold']

    result = questao06(string_list, analyzed_string, threshold)
    print('As palavras mais próximas são: ', result)


if __name__ == '__main__':
    main()
