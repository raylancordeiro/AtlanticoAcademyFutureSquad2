from src.gen_functions import sort_string_list_by_lenght
import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p1', '--string_list',
                    default=['Maria','vania','vagner','Vagner', 'vagner','maria','dirceu','ana'],
                    help='lista de string')
    ap.add_argument('-p2', '--reverse',
                    default='True',
                    help='Condição booleana, True: crescente, False: decrescente')

    args = vars(ap.parse_args())

    string_list = args['string_list']
    reverse = args['reverse']
    result = sort_string_list_by_lenght(string_list,reverse)
    print('Lista ordenada na condição: ',reverse, 'eh:', result)


if __name__ == '__main__':
    main()

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