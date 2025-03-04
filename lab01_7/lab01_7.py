"""
Лабораторная работа №01.7
Капсульные нейронные сети (часть 1)

Цель работы: знакомство с капсульной нейронной сетью.

Задачи работы: реализация капсульной нейронной сети согласно варианту.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.layers import Conv2D, Input, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Установка seed для воспроизводимости результатов
np.random.seed(42)
tf.random.set_seed(42)


# Функция для загрузки и подготовки данных
def load_and_prepare_data(dataset_name="mnist"):
    if dataset_name.lower() == "mnist":
        # Загрузка набора данных MNIST (рукописные цифры)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        class_names = [str(i) for i in range(10)]
    elif dataset_name.lower() == "fashion_mnist":
        # Загрузка набора данных Fashion MNIST (предметы одежды)
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        num_classes = 10
        class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    else:
        raise ValueError(
            "Неподдерживаемый набор данных. Используйте 'mnist' или 'fashion_mnist'"
        )

    # Нормализация данных
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Добавление канала для черно-белых изображений
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Преобразование меток в категориальное представление (one-hot encoding)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat), class_names


# Функция для сжатия (squash)
def squash(vectors, axis=-1):
    """
    Функция сжатия для капсульных сетей.
    Сжимает вектора так, чтобы их длина была в диапазоне [0, 1].
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-8)
    return scale * vectors


# Функция для создания капсульного слоя
def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Первичный капсульный слой.
    """
    output = Conv2D(
        filters=dim_capsule * n_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
    )(inputs)
    output = Reshape((-1, dim_capsule))(output)
    return Lambda(squash)(output)


# Функция для маршрутизации по соглашению (routing by agreement)
def routing(u_hat, b_ij, routing_iterations=3):
    """
    Маршрутизация по соглашению между капсулами.
    """
    # Создаем переменную для хранения выходов маршрутизации
    for i in range(routing_iterations):
        c_ij = tf.nn.softmax(b_ij, axis=1)
        s_j = tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=1, keepdims=True)
        v_j = squash(s_j)

        if i < routing_iterations - 1:
            b_ij += tf.reduce_sum(
                tf.multiply(tf.expand_dims(v_j, 1), u_hat), axis=-1, keepdims=True
            )

    return v_j


# Функция для создания капсульного слоя с маршрутизацией
def DigitCaps(inputs, num_capsules, dim_capsule, routing_iterations=3):
    """
    Капсульный слой для цифр с маршрутизацией.
    """
    input_shape = inputs.shape
    u_i = inputs

    # Инициализация весов
    w_ij = tf.Variable(
        tf.random.normal(
            shape=(1, input_shape[1], num_capsules, dim_capsule, input_shape[-1])
        ),
        dtype=tf.float32,
    )

    # Умножение входов на веса
    u_hat = tf.reduce_sum(
        tf.multiply(
            tf.expand_dims(tf.expand_dims(u_i, 2), 3),
            tf.expand_dims(w_ij, 0),
        ),
        axis=-1,
    )

    # Инициализация логитов маршрутизации
    b_ij = tf.zeros(shape=(input_shape[0], input_shape[1], num_capsules, 1))

    # Маршрутизация по соглашению
    v_j = routing(u_hat, b_ij, routing_iterations)

    return v_j


# Функция для вычисления длины капсул
def length(vectors):
    """
    Вычисляет длину векторов капсул.
    """
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1))


# Функция для создания маски
def mask(inputs, y_true):
    """
    Создает маску для капсул на основе истинных меток.
    """
    mask_with_labels = tf.multiply(tf.expand_dims(y_true, -1), tf.ones_like(inputs))
    return tf.multiply(mask_with_labels, inputs)


# Функция для создания и обучения капсульной нейронной сети
def create_and_train_capsnet(
    x_train,
    y_train_cat,
    x_test,
    y_test_cat,
    primary_dim=8,
    primary_channels=32,
    digit_dim=16,
    routing_iterations=3,
):
    # Получение размеров входных данных
    input_shape = x_train.shape[1:]
    num_classes = y_train_cat.shape[1]

    # Входной слой
    inputs = Input(shape=input_shape)

    # Первый сверточный слой
    x = Conv2D(
        filters=256,
        kernel_size=9,
        strides=1,
        padding="valid",
        activation="relu",
    )(inputs)

    # Первичный капсульный слой
    primary_caps = PrimaryCaps(
        x,
        dim_capsule=primary_dim,
        n_channels=primary_channels,
        kernel_size=9,
        strides=2,
        padding="valid",
    )

    # Капсульный слой для цифр
    digit_caps = DigitCaps(
        primary_caps,
        num_capsules=num_classes,
        dim_capsule=digit_dim,
        routing_iterations=routing_iterations,
    )

    # Вычисление длины капсул
    out_caps = Lambda(length)(digit_caps)

    # Создание модели для классификации
    model = Model(inputs=inputs, outputs=out_caps)

    # Компиляция модели
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Ранняя остановка для предотвращения переобучения
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Замер времени обучения
    start_time = time.time()

    # Обучение модели
    history = model.fit(
        x_train,
        y_train_cat,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1,
    )

    training_time = time.time() - start_time

    # Замер времени предсказания
    start_time = time.time()
    y_pred = model.predict(x_test)
    prediction_time = time.time() - start_time

    # Оценка модели
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"CapsNet - Точность на тестовом наборе: {test_acc:.4f}")
    print(f"CapsNet - Время обучения: {training_time:.2f} секунд")
    print(f"CapsNet - Время предсказания: {prediction_time:.2f} секунд")
    print(f"CapsNet - Количество параметров: {model.count_params()}")

    return model, history, test_acc, training_time, prediction_time


# Функция для визуализации результатов обучения
def plot_training_history(history, filename):
    plt.figure(figsize=(12, 5))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # График функции потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Функция для визуализации матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Функция для визуализации примеров с предсказаниями
def plot_examples(x, y, y_pred, class_names, filename, n_examples=25):
    plt.figure(figsize=(10, 10))

    # Выбираем случайные примеры
    indices = np.random.choice(len(x), n_examples, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x[idx].reshape(28, 28), cmap="gray")

        # Добавляем заголовок с истинным и предсказанным классами
        true_class = class_names[y[idx]]
        pred_class = class_names[y_pred[idx]]
        color = "green" if y[idx] == y_pred[idx] else "red"
        plt.title(f"T: {true_class}\nP: {pred_class}", color=color, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Определение пути к текущему скрипту
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Создание директории для сохранения результатов внутри директории скрипта
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Выбор набора данных
    dataset_name = "mnist"  # Можно изменить на "fashion_mnist"

    print(f"Загрузка и подготовка набора данных {dataset_name}...")
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat), class_names = (
        load_and_prepare_data(dataset_name)
    )

    # Вывод информации о наборе данных
    print(f"Размер обучающего набора: {x_train.shape}")
    print(f"Размер тестового набора: {x_test.shape}")
    print(f"Количество классов: {len(class_names)}")

    # Визуализация примеров из набора данных
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
        plt.title(class_names[y_train[i]], fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{dataset_name}_examples.png"))
    plt.close()

    # Создание и обучение капсульной нейронной сети
    print("\nСоздание и обучение капсульной нейронной сети...")
    model, history, test_acc, training_time, prediction_time = create_and_train_capsnet(
        x_train,
        y_train_cat,
        x_test,
        y_test_cat,
        primary_dim=8,
        primary_channels=32,
        digit_dim=16,
        routing_iterations=3,
    )

    # Визуализация истории обучения
    plot_training_history(
        history, os.path.join(results_dir, f"{dataset_name}_training_history.png")
    )

    # Получение предсказаний модели
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Визуализация матрицы ошибок
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        os.path.join(results_dir, f"{dataset_name}_confusion_matrix.png"),
    )

    # Визуализация примеров с предсказаниями
    plot_examples(
        x_test,
        y_test,
        y_pred,
        class_names,
        os.path.join(results_dir, f"{dataset_name}_predictions.png"),
    )

    # Вывод отчета о классификации
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Сохранение модели
    model.save(os.path.join(results_dir, f"{dataset_name}_capsnet_model.keras"))

    print("\nЛабораторная работа №01.7 выполнена успешно!")
    print(f"Результаты сохранены в директории {results_dir}")


if __name__ == "__main__":
    main()
