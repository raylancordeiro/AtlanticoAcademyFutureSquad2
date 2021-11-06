from src.gen_functions import string_list_occurrences
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--string_list',
                    default=['abacate', 'abacate', 'uva', 'banana', 'pera', 'abacate', 'repolho', 'uva', 'feijão',
                             'pera'],
                    help='list of strings.')

    args = vars(ap.parse_args())

    string_list = args['string_list']

    result = string_list_occurrences(string_list)
    print('string_list: {}'.format(string_list))
    print('Result: {}'.format(result))


if __name__ == '__main__':
    main()
