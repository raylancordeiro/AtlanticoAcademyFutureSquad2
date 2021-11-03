from src.gen_functions import sort_string_list_by_lenght
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-sl', '--string_list',
                    default=['4444', '1', '333', '666666', '55555', '22'],
                    help='first point to calculate the Euclidean distance')
    ap.add_argument('-rv', '--reverse',
                    default=False,
                    help='second point to calculate the Euclidean distance')

    args = vars(ap.parse_args())

    string_list = args['string_list']
    reverse = args['reverse']
    result = sort_string_list_by_lenght(string_list, reverse)

    print('Original list: {}'.format(string_list))
    print('Sorted list by lenght of elements: {}'.format(result))


if __name__ == '__main__':
    main()
