import numpy as np


def soma(num_1, num_2):
    '''Esta função executa a soma de dois números
    entrada:
    num_1 - primeiro número em tipo int ou float
    num_2 - segundo número em tipo int ou float
    retorno:
    A soma dos números
    '''
    return num_1 + num_2


def dist_euclid(point1, point2):
    dist = np.linalg.norm(point1 - point2)
    return dist


def ponto_mais_proximo(ponto1, lista_de_pontos):
    menor_distancia = dist_euclid(ponto1, lista_de_pontos[0])
    ponto_mais_proximo = lista_de_pontos[0]
    for ponto in lista_de_pontos:
        dist_atual = dist_euclid(ponto1, ponto)
        if dist_atual < menor_distancia:
            menor_distancia = dist_atual
            ponto_mais_proximo = ponto
    return ponto_mais_proximo
