"""
este pipeline utiliza os dados gerados pelo preprocessing_pipeline
treina a rede neural escolhida e exibe o desempenho alcançado
"""

import time
import numpy as np
from tensorflow import keras
from src import computer_vision

# escolher o modelo de rede neural: cnn ou vgg
MODEL_CHOICE = 'cnn'  # 'cnn' ou 'vgg'

# hiperparametros gerais
batch_size = 16
epochs = 5


def main():
    # carregamento do dataset preprocessado
    class_names = np.load('preprocessed_data/class_names.npy')
    test_data = np.load('preprocessed_data/test_data.npy')
    test_labels = np.load('preprocessed_data/test_labels.npy')
    train_data = np.load('preprocessed_data/train_data.npy')
    train_labels = np.load('preprocessed_data/train_labels.npy')
    val_data = np.load('preprocessed_data/val_data.npy')
    val_labels = np.load('preprocessed_data/val_labels.npy')

    # métricas
    metrics = [keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]

    # Criando o modelo e verificando a estrutura
    model = computer_vision.create_model(model_type=MODEL_CHOICE,
                                         input_shape=train_data.shape[1:],
                                         number_of_classes=len(class_names),
                                         metrics=metrics)

    # exibindo resumo do modelo
    model.summary()

    # Marcando o tempo de início
    start = time.time()

    # Treinamento do modelo
    history_model = model.fit(train_data, train_labels, batch_size=batch_size,
                              epochs=epochs, initial_epoch=0,
                              validation_data=(val_data, val_labels))

    # Marcando o tempo final
    end = time.time()
    duration = end - start
    print('\n Modelo CNN - Duração %0.2f segundos (%0.1f minutos) para treinamento de %d epocas' % (
        duration, duration / 60, epochs))

    # exibição dos gráficos de treinamento
    computer_vision.plot_model(history_model.history, epochs)

    # exibição de exemplos de predição
    computer_vision.show_predictions(model=model,
                                     test_data=test_data,
                                     test_labels=test_labels,
                                     class_names=class_names)


if __name__ == '__main__':
    main()
