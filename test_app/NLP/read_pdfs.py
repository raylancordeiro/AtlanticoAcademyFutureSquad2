import argparse
from src.NLP.vagner import join_and_remove_breaks
from src.NLP.vagner import load_pfds
from src.NLP.vagner import calculate_idf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-pl', '--directory path',
                    default=r'C:\Users\r211315\Documents\ia_express\arquivos',
                    help='full path to the directory in which the files are')

    args = vars(ap.parse_args())

    directory_path = args['directory path']
    result = join_and_remove_breaks(load_pfds(directory_path))
    print('Quantidade de documentos (corpora): ', len(result))
    print('Primeiro corpus')
    print(result[0])
    print('Quantidade de palavras no 1º Corpus: ', len(result[0]))
    print('Segundo corpus')
    print(result[1])
    print('Quantidade de palavras no 2º Corpus: ', len(result[1]))
    print('Terceiro corpus')
    print(result[2])
    print('Quantidade de palavras no 3º Corpus: ', len(result[2]))

    idf = calculate_idf(result)
    print('Matriz IDF')
    print(idf)

if __name__ == '__main__':
    main()
