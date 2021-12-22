# Imports de bibliotecas para treinamento de redes neurais
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Nadam, Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np
import pathlib
import os
import cv2


def load_dataset(data_dir):
    data_dir = pathlib.Path(data_dir)
    class_names = np.array([item.name for item in data_dir.glob('*')])

    fnames = []
    for classes in class_names:
        images_folder = os.path.join(data_dir, classes)
        file_names = os.listdir(images_folder)
        full_path = [os.path.join(images_folder, file_name) for file_name in file_names]
        fnames.append(full_path)

    # Carregando as imagens das fotos usando CV2
    images = []
    for names in fnames:
        one_class_images = [cv2.imread(name) for name in names if (cv2.imread(name)) is not None]
        images.append(one_class_images)

    return images, class_names


def custom_train_test_split(images, random_state=42):
    # Criando as listas vazias
    train_images = []
    val_images = []
    test_images = []
    aux_images = []

    # Loop percorrendo todas as imagens redimensionadas e preenchendo as listas de treino e validação
    for imgs in images:
        train, test = train_test_split(imgs, train_size=0.9, test_size=0.1,
                                       random_state=42)
        aux_images.append(train)
        test_images.append(test)

    for imgs2 in aux_images:
        train, val = train_test_split(imgs2, train_size=0.78, test_size=0.22,
                                      random_state=42)
        train_images.append(train)
        val_images.append(val)

    return train_images, val_images, test_images


def create_labels(train_images, val_images, test_images):
    # Exibindo a quantidade de dados para treinamento e a distribuição de cada classe

    # tamanhos das classes
    len_train_images = [len(imgs) for imgs in train_images]
    len_val_images = [len(imgs) for imgs in val_images]
    len_test_images = [len(imgs) for imgs in test_images]

    # arrays para armazenar os labels
    train_classe = np.zeros((np.sum(len_train_images)), dtype='uint8')
    val_classe = np.zeros((np.sum(len_val_images)), dtype='uint8')
    test_classe = np.zeros((np.sum(len_test_images)), dtype='uint8')

    # atribuição
    for i in range(4):
        if i == 0:
            train_classe[:len_train_images[i]] = i
            val_classe[:len_val_images[i]] = i
            test_classe[:len_test_images[i]] = i
        else:
            train_classe[np.sum(len_train_images[:i]):np.sum(len_train_images[:i + 1])] = i
            val_classe[np.sum(len_val_images[:i]):np.sum(len_val_images[:i + 1])] = i
            test_classe[np.sum(len_test_images[:i]):np.sum(len_test_images[:i + 1])] = i

    return train_classe, val_classe, test_classe


def convert_to_numpy(train_images, val_images, test_images):
    # Criando listas temporarias
    tmp_train_imgs = []
    tmp_val_imgs = []
    tmp_test_imgs = []

    # Percorrendo o dataset de treinamento e adicionando na lista temporaria
    for imgs in train_images:
        tmp_train_imgs += imgs

    # Percorrendo o dataset de validação e adicionando na lista temporaria
    for imgs in val_images:
        tmp_val_imgs += imgs

    # Percorrendo o dataset de testes e adicionando na lista temporaria
    for imgs in test_images:
        tmp_test_imgs += imgs

    # Convertendo em formato array
    train_images_np = np.array(tmp_train_imgs)
    val_images_np = np.array(tmp_val_imgs)
    test_images_np = np.array(tmp_test_imgs)

    # Transformando os dados para o tipo float32
    train_data = train_images_np.astype('float32')
    val_data = val_images_np.astype('float32')
    test_data = test_images_np.astype('float32')

    return train_data, val_data, test_data


def convert_to_rgb(img):
    # Função para converter as imagens para formato RGB
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)


def create_model(model_type, input_shape, number_of_classes, metrics):
    if model_type == 'cnn':
        return create_model_cnn(input_shape, number_of_classes, metrics)
    elif model_type == 'vgg':
        return create_model_vgg19(input_shape, number_of_classes, metrics)


# Função para criar a estrutura de nosso modelo CNN
def create_model_cnn(input_shape, number_of_classes, metrics):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))

    input_shape = input_shape
    model.build((None,) + input_shape)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=metrics)

    return model


# Função para criar a estrutura de nosso modelo usando como base o modelo pre-treinado VGG19
def create_model_vgg19(input_shape, number_of_classes, metrics):
    model = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)

    # Congelando as camadas que não serão treinadas
    for layer in model.layers[:20]:
        layer.trainable = False

    # Adicionando nova camadas ao nosso modelo
    x = model.output
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(number_of_classes, activation="softmax")(x)

    # Criando o modelo final
    final_model = Model(inputs=model.input, outputs=predictions)
    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=metrics)

    return final_model


# Função para exibir o desempenho do modelo em treino e teste
def plot_model(history, epochs):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(history['precision'])), history['precision'], 'r')
    plt.plot(np.arange(1, len(history['val_precision']) + 1), history['val_precision'], 'g')
    plt.xticks(np.arange(0, epochs + 1, epochs / 10))
    plt.title('Training Precision vs. Validation Precision')
    plt.xlabel('Nro de Epochs')
    plt.ylabel('Precision')
    plt.legend(['train', 'validation'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history['recall']) + 1), history['recall'], 'r')
    plt.plot(np.arange(1, len(history['val_recall']) + 1), history['val_recall'], 'g')
    plt.xticks(np.arange(0, epochs + 1, epochs / 10))
    plt.title('Training Recall vs. Validation Recall')
    plt.xlabel('Nro de Epochs')
    plt.ylabel('Recall')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


# Função para realizar previsão da classe das imagens passadas como parâmetro
def predict_val(test_data, model):
    val_input = np.reshape(test_data, (1, 256, 256, 3))
    val_input = val_input / 255.
    pred = model.predict(val_input)
    class_num = np.argmax(pred)
    return class_num, np.max(pred)


# Função para buscar a descrição do label
def desc_label(label):
    idx = np.where(label == 1)
    return idx[0][0]


def show_predictions(model, test_data, test_labels, class_names):
    # Realizando as previsões e exibindo as imagens com os labels verdadeiros e previstos
    plt.figure(figsize=(15, 15))
    for i in range(9):
        idx = np.random.randint(len(test_data))

        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(convert_to_rgb(test_data.astype('uint8')[idx]))
        class_idx = desc_label(test_labels[idx])

        pred, prob = predict_val(test_data[idx], model)
        plt.title('True: %s || Pred: %s %d%%' % (class_names[class_idx], class_names[pred], round(prob, 2) * 100))
        plt.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.show()


def reshape_img_dataset(images, new_shape):
    img_width, img_height = new_shape

    resized_images = []
    for i, imgs in enumerate(images):
        resized_images.append([cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC) for img in imgs])
    return resized_images


def apply_bilateral_filter(images):
    filtered_images = []
    for i, imgs in enumerate(images):
        filtered_images.append([cv2.bilateralFilter(img, 15, 75, 75) for img in imgs])
    return filtered_images
