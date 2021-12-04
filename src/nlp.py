import re
import nltk
import string
import stanza
import numpy as np
from src import gen_functions
import csv


def download_stanza_portugues():
    """
    Faz o download do stanza em portugues
    """
    stanza.download(lang='pt')


def tokenizer_and_lemmatizer(text):
    """
        Performs tokenization and lemmatization on input text

    Args:
        text: A string with the content of the text

    Returns:
        A stanza Document with the tokens and lemmas

    """
    nlp = stanza.Pipeline('pt', processors='tokenize,mwt,pos,lemma')
    return nlp(text)


def show_nlp_doc(doc):
    """
    Imprime os tokens (somente para debug)
    """
    sentence_id = 0
    for sentence in doc.sentences:
        sentence_id += 1
        print('\nSentença {}:'.format(sentence_id))
        for word in sentence.words:
            print('palavra = {}, lema = {}, id = {}'.format(word.text, word.lemma, word.id))


def count_occurrences(doc, method='lemma'):
    # dicionário de contagem: chave = palavra, valor = ocorrencias
    count_dict = dict()

    # metodo lemma -> contagem dos lemas
    if method == 'lemma':
        for word in doc.iter_words():
            lemma = word.lemma
            if lemma in count_dict.keys():
                count_dict[lemma] += 1
            else:
                count_dict[lemma] = 1

    # metodo token -> contagem dos tokens
    # não agrupa tokens diferentes com lemas iguais
    elif method == 'token':
        for token in doc.iter_tokens():
            text = token.text
            if text in count_dict.keys():
                count_dict[text] += 1
            else:
                count_dict[text] = 1

    return count_dict


def frequency(doc, method='lemma'):
    # dicionário de frequencia: chave = palavra, valor = frequencia
    count_dic = dict()

    # metodo lemma -> contagem dos lemas
    if method == 'lemma':
        count_dic = count_occurrences(doc, method=method)
        for lemma in count_dic.keys():
            count_dic[lemma] = count_dic[lemma] / doc.num_words

    # metodo token -> contagem dos tokens
    # não agrupa tokens diferentes com lemas iguais
    elif method == 'token':
        count_dic = count_occurrences(doc, method=method)
        for token in count_dic.keys():
            count_dic[token] = count_dic[token] / doc.num_tokens

    return count_dic


def neighborhood_by_sentences(doc, tf_dict, lemma, threshold):
    neighbors_list = []
    for sentence in doc.sentences:
        sentence_list = []
        for word in sentence.words:
            sentence_list.append(word.lemma)
        current_neighbors_list = gen_functions.string_proximity(sentence_list, lemma, threshold)

        for current_neighbor in current_neighbors_list:
            if current_neighbor not in neighbors_list:
                neighbors_list.append(current_neighbor)

    neighbors_tf_dict = dict()
    for lemma in neighbors_list:
        neighbors_tf_dict[lemma] = tf_dict[lemma]

    return neighbors_tf_dict


def print_sentences(doc):
    for sentence in doc.sentences:
        sentence_list = []
        for word in sentence.words:
            sentence_list.append(word.lemma)
        print(sentence_list)


def stanza_sentence_to_list_of_lemmas(corpus_lematizado_stanza_doc):
    list_of_list_of_lemmas = []
    for document in corpus_lematizado_stanza_doc:
        document_list = []
        for sentence in document.sentences:
            for word in sentence.words:
                document_list.append(word.lemma)
        list_of_list_of_lemmas.append(document_list)
    return list_of_list_of_lemmas


def clean_text(text):
    # remove numbers
    text_nonum = re.sub(r'\d+', '', text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation])
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace


def removestopwords(texto):
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('portuguese')
    frases = []
    for palavras in texto:
        semstop = [p for p in palavras.split() if p not in stopwords]
        frases.append(semstop)
    return frases


def convert_list_to_string(_corpus, seperator=' '):
    return seperator.join(_corpus)


def calc_tf(ocorrencias_termo, termos_documento):
    result = {}
    for i in range(len(ocorrencias_termo)):
        result[i] = dict(map(lambda kv: (kv, ocorrencias_termo[i][kv] / termos_documento[i]), ocorrencias_termo[i]))
    return result


def document_frequency(corpus_lists):
    todas_palavras = gen_functions.concatenate_lists_without_repetitions(*corpus_lists)

    df = dict.fromkeys(todas_palavras, 0)

    for word in todas_palavras:
        for doc in corpus_lists:
            if word in doc:
                df[word] += 1

    return df


def calc_idf(corpus):
    import math
    df = document_frequency(corpus)
    idf = {k: math.log10((len(corpus) / v)) for k, v in df.items()}
    return idf


def calc_tf_idf(corpus):
    ocorrencias_termo = list(map(gen_functions.map_occurrences, corpus))
    termos_documento = list(map(len, corpus))
    tf = calc_tf(ocorrencias_termo, termos_documento)

    idf = calc_idf(corpus)

    tf_idf_list = []

    for i in range(len(corpus)):
        current_tf_idf = tf[i].copy()
        for word in current_tf_idf.keys():
            current_tf_idf[word] = tf[i][word] * idf[word]
        tf_idf_list.append(current_tf_idf)

    return tf_idf_list


def calc_palavras_mais_signiticativas(corpus, tf_dicts, df_dict, idf_dict, tf_idf_dicts, n_items=5, n_neighbors=2):
    tf_idf_all_list = []
    for i in range(len(corpus)):
        for word in tf_idf_dicts[i].keys():
            tf_idf_all_list.append((word, tf_idf_dicts[i][word]))

    tf_idf_all_list.sort(key=lambda x: x[1], reverse=True)

    n_palavras_mais_significativas = []
    for i in range(n_items):
        n_palavras_mais_significativas.append(tf_idf_all_list[i][0])

    palavras_mais_significativas_dict = {}
    for word in n_palavras_mais_significativas:
        full_list_of_neighbors = []
        for i in range(len(corpus)):
            parcial_list_of_neighbors = []
            parcial_list_of_neighbors_plus_tf = []
            if word in corpus[i]:
                parcial_list_of_neighbors += gen_functions.string_proximity(corpus[i], word, n_neighbors)
            for neighbor in parcial_list_of_neighbors:
                parcial_list_of_neighbors_plus_tf.append([neighbor, tf_dicts[i][neighbor]])
            full_list_of_neighbors += parcial_list_of_neighbors_plus_tf
        palavras_mais_significativas_dict[word] = full_list_of_neighbors

    return palavras_mais_significativas_dict


def write_csv_metricas_gerais(corpus, tf_dicts, df_dict, idf_dict, tf_idf_dicts):
    with open('resultados.csv', mode='w', newline='') as csv_file:
        fieldnames = ["PALAVRA", "TF-doc1", "TF-doc2", "TF-doc3",
                      "DF", "IDF", "TF-IDF-doc1", "TF-IDF-doc2", "TF-IDF-doc3"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        all_words = gen_functions.concatenate_lists_without_repetitions(*corpus)

        tf = ['-'] * len(corpus)
        tf_idf = ['-'] * len(corpus)
        for word in all_words:
            for i in range(len(corpus)):
                if word in corpus[i]:
                    tf[i] = tf_dicts[i][word]
                    tf_idf[i] = tf_idf_dicts[i][word]
                else:
                    tf[i] = '-'
                    tf_idf[i] = '-'
            df = df_dict[word]
            idf = idf_dict[word]
            writer.writerow({"PALAVRA": word,
                             "TF-doc1": str(tf[0]),
                             "TF-doc2": str(tf[1]),
                             "TF-doc3": str(tf[2]),
                             "DF": str(df),
                             "IDF": str(idf),
                             "TF-IDF-doc1": str(tf_idf[0]),
                             "TF-IDF-doc2": str(tf_idf[1]),
                             "TF-IDF-doc3": str(tf_idf[2])
                             })


def write_csv_palavras_mais_importantes(palavras_mais_significativas_dict):
    with open('tokens_de_maior_tf-idf.csv', mode='w', newline='') as csv_file:
        fieldnames = ["tokens de maiores tf-idf", "lista de prox com tf de cada string"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for word in palavras_mais_significativas_dict.keys():
            writer.writerow({"tokens de maiores tf-idf": str(word),
                             "lista de prox com tf de cada string": str(palavras_mais_significativas_dict[word]),
                             })
