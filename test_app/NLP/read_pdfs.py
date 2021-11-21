import argparse
from src.NLP.vagner import join_and_remove_breaks
from src.NLP.vagner import load_pfds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-pl', '--directory path',
                    default='/home/vagnersv/my_tensorflow/arquivos',
                    help='full path to the directory in which the files are')

    args = vars(ap.parse_args())

    directory_path = args['directory path']
    result = join_and_remove_breaks(load_pfds(directory_path))
    print(type(result))
    print(result)

if __name__ == '__main__':
    main()
