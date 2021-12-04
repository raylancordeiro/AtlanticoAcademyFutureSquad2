from src import nlp, pdf_reader, gen_functions
import csv

if __name__ == '__main__':
    # comentar a linha abaixo após a primeira execução
    # nlp.download_stanza_portugues()

    """
    1) Carregar o conjunto de documentos em PDF e armazená-los em alguma estrutura de dados
    """

    directory_path = r'/home/victor/code/AtlanticoAcademyFutureSquad2/test_app/NLP/data'  # acessar online
    corpus = pdf_reader.join_and_remove_breaks(pdf_reader.load_pfds(directory_path))

    """
    2) Realizar o pré-processamento destes ( tokenização e remoção de stop words, deixar todos 
    os caracteres minúsculos...)
    """

    corpus_clean_text = [nlp.clean_text(doc) for doc in corpus]
    corpus_sem_stopwords = nlp.removestopwords(corpus_clean_text)

    corpus_sem_stopwords = list(map(nlp.convert_list_to_string, corpus_sem_stopwords))

    # tokenização e lematização com a biblioteca stanza
    corpus_lematizado_stanza_doc = [nlp.tokenizer_and_lemmatizer(doc) for doc in corpus_sem_stopwords]
    corpus_lematizado_lists = nlp.stanza_sentence_to_list_of_lemmas(corpus_lematizado_stanza_doc)

    # 5.1) Term Frequency (TF):

    ocorrencias_termo = list(map(gen_functions.map_occurrences, corpus_lematizado_lists))
    termos_documento = list(map(len, corpus_lematizado_lists))
    term_frequency = nlp.calc_tf(ocorrencias_termo, termos_documento)

    # 5.2) Document Frequency (DF)

    document_frequency = nlp.document_frequency(corpus_lematizado_lists)

    # 5.3) Inverse Document Frequency (IDF)

    inverse_document_frequency = nlp.calc_idf(corpus_lematizado_lists)

    # 5.4) TF-IDF

    tf_idf_list = nlp.calc_tf_idf(corpus_lematizado_lists)

    # 5.5) Lista de strings com proximidade até 2 dos 5 termos de maior TF-IDF.
    # Essas strings devem ser acompanhadas de seu valor de TF.

    # proximity_lists = []
    # for i in range(len(corpus_lematizado_lists)):
    #     proximity_lists.append(
    #         nlp.questao5_5(
    #             corpus_lematizado_lists[i],
    #             tf_idf_list[i],
    #             term_frequency[i],
    #             n_items=5, n_neighbors=2)
    #     )

    palavras_mais_significativas = nlp.calc_palavras_mais_signiticativas(corpus=corpus_lematizado_lists, tf_dicts=term_frequency,
                                                                         df_dict=document_frequency, idf_dict=inverse_document_frequency,
                                                                         tf_idf_dicts=tf_idf_list, n_items=5, n_neighbors=2)

    # 6) Gerar um arquivo csv que possui todas as palavras de todos os documentos na primeira coluna,
    # em que cada linha é um token. Para cada token, informe nas colunas vizinhas as informações
    # determinadas no objetivo 5.

    nlp.write_csv_metricas_gerais(corpus=corpus_lematizado_lists, tf_dicts=term_frequency,
                                  df_dict=document_frequency, idf_dict=inverse_document_frequency,
                                  tf_idf_dicts=tf_idf_list)

    nlp.write_csv_palavras_mais_importantes(palavras_mais_significativas_dict=palavras_mais_significativas)

    # frequency = extra.frequency(txt3_processed, method='lemma')
    # count_occurrences = extra.count_occurrences(txt3_processed, method='token')

    # impressão da estrutura de dados contendo o texto3 processado
    # nlp.show_nlp_doc(txt_processed)
    # nlp.print_sentences(txt_processed)

    # tf_dict = nlp.frequency(txt_processed, method='lemma')
    # neighborhood = nlp.neighborhood(txt_processed, tf_dict, 'câncer', 2)
    # print(neighborhood)
