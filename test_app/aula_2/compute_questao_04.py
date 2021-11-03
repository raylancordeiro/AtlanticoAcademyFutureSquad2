from src.gen_functions import questao4
import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--string_list',
                    default=['Maria','vania','vagner','Vagner', 'vagner','maria','dirceu','ana'],
                    help='lista de string')
    ap.add_argument('-p2', '--string_key',
                    default='ana',
                    help='palavra buscada')

    args = vars(ap.parse_args())

    string_list = args['string_list']
    string_key = args['string_key']
    result = questao4(string_list,string_key)
    print('a quantidade da palavra: ',string_key, 'eh:', result)


if __name__ == '__main__':
    main()