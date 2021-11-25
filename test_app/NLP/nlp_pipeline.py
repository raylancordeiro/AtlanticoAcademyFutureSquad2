from src import nlp
from src.pdf_reader import join_and_remove_breaks, load_pfds

if __name__ == '__main__':

    # comentar a linha abaixo após a primeira execução
    nlp.download_stanza_portugues()

    # carregamento dos pdf's
    # é necessário passar o path completo
    # atualizar para o path da sua pasta!
    directory_path = r'/home/victor/code/AtlanticoAcademyFutureSquad2/test_app/NLP/data'
    documents = join_and_remove_breaks(load_pfds(directory_path))

    # processamento do texto3
    txt3_processed = nlp.tokenizer_and_lemmatizer(documents[0])

    # impressão da estrutura de dados contendo o texto3 processado
    nlp.show_nlp_doc(txt3_processed)
