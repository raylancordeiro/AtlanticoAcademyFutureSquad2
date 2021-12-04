import nlp


class Document:
    def __init__(self, documento_bruto):
        self.documento_bruto = documento_bruto
        self.documento_limpo = nlp.removestopwords(nlp.clean_text(self.documento_bruto))
        self.documento_lematizado = nlp.stanza_sentence_to_list_of_lemmas(
            nlp.tokenizer_and_lemmatizer(self.documento_limpo))
