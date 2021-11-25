import stanza
import numpy as np


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
