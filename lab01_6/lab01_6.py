"""
Лабораторная работа №01.6
Классификация изображений с помощью нейронных сетей (часть 2)

Цель работы: сравнение полносвязного перцептрона и сверточных нейронных сетей.

Задачи работы: расчет сравнительных характеристик для сверточной нейронной сети и перцептрона.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
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
        # Добавление канала для MNIST (черно-белые изображения)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
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
        # Добавление канала для Fashion MNIST (черно-белые изображения)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif dataset_name.lower() == "cifar10":
        # Загрузка набора данных CIFAR-10 (10 классов объектов)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
        class_names = [
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ]
        # Преобразование меток из формы (n, 1) в (n,)
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
    else:
        raise ValueError(
            "Неподдерживаемый набор данных. Используйте 'mnist', 'fashion_mnist' или 'cifar10'"
        )

    # Нормализация данных
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Преобразование меток в категориальное представление (one-hot encoding)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat), class_names


# Функция для создания и обучения полносвязной нейронной сети (MLP)
def create_and_train_mlp(
    x_train, y_train_cat, x_test, y_test_cat, hidden_layers=[128, 64], dropout_rate=0.3
):
    # Получение размеров входных данных
    input_shape = x_train.shape[1:]
    num_classes = y_train_cat.shape[1]

    # Создание модели
    model = Sequential()

    # Преобразование входных данных в одномерный вектор
    model.add(Flatten(input_shape=input_shape))

    # Добавление скрытых слоев
    for units in hidden_layers:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Выходной слой
    model.add(Dense(num_classes, activation="softmax"))

    # Компиляция модели
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
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
    print(f"MLP - Точность на тестовом наборе: {test_acc:.4f}")
    print(f"MLP - Время обучения: {training_time:.2f} секунд")
    print(f"MLP - Время предсказания: {prediction_time:.2f} секунд")
    print(f"MLP - Количество параметров: {model.count_params()}")

    return model, history, test_acc, training_time, prediction_time


# Функция для создания и обучения сверточной нейронной сети (CNN)
def create_and_train_cnn(
    x_train,
    y_train_cat,
    x_test,
    y_test_cat,
    conv_layers=[(32, 3), (64, 3)],
    dense_layers=[128],
    dropout_rate=0.3,
):
    # Получение размеров входных данных
    input_shape = x_train.shape[1:]
    num_classes = y_train_cat.shape[1]

    # Создание модели
    model = Sequential()

    # Добавление сверточных слоев
    for i, (filters, kernel_size) in enumerate(conv_layers):
        if i == 0:
            # Первый слой должен указывать input_shape
            model.add(
                Conv2D(
                    filters,
                    kernel_size=(kernel_size, kernel_size),
                    activation="relu",
                    padding="same",
                    input_shape=input_shape,
                )
            )
        else:
            model.add(
                Conv2D(
                    filters,
                    kernel_size=(kernel_size, kernel_size),
                    activation="relu",
                    padding="same",
                )
            )
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))

    # Преобразование в одномерный вектор
    model.add(Flatten())

    # Добавление полносвязных слоев
    for units in dense_layers:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Выходной слой
    model.add(Dense(num_classes, activation="softmax"))

    # Компиляция модели
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
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
    print(f"CNN - Точность на тестовом наборе: {test_acc:.4f}")
    print(f"CNN - Время обучения: {training_time:.2f} секунд")
    print(f"CNN - Время предсказания: {prediction_time:.2f} секунд")
    print(f"CNN - Количество параметров: {model.count_params()}")

    return model, history, test_acc, training_time, prediction_time


# Функция для визуализации результатов обучения
def plot_training_history(mlp_history, cnn_history, filename):
    plt.figure(figsize=(15, 10))

    # График точности
    plt.subplot(2, 2, 1)
    plt.plot(mlp_history.history["accuracy"], label="MLP Train Accuracy")
    plt.plot(mlp_history.history["val_accuracy"], label="MLP Validation Accuracy")
    plt.plot(cnn_history.history["accuracy"], label="CNN Train Accuracy")
    plt.plot(cnn_history.history["val_accuracy"], label="CNN Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # График функции потерь
    plt.subplot(2, 2, 2)
    plt.plot(mlp_history.history["loss"], label="MLP Train Loss")
    plt.plot(mlp_history.history["val_loss"], label="MLP Validation Loss")
    plt.plot(cnn_history.history["loss"], label="CNN Train Loss")
    plt.plot(cnn_history.history["val_loss"], label="CNN Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # График точности (только валидационная)
    plt.subplot(2, 2, 3)
    plt.plot(mlp_history.history["val_accuracy"], label="MLP Validation Accuracy")
    plt.plot(cnn_history.history["val_accuracy"], label="CNN Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # График функции потерь (только валидационная)
    plt.subplot(2, 2, 4)
    plt.plot(mlp_history.history["val_loss"], label="MLP Validation Loss")
    plt.plot(cnn_history.history["val_loss"], label="CNN Validation Loss")
    plt.title("Validation Loss Comparison")
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
def plot_examples(x, y, y_pred_mlp, y_pred_cnn, class_names, filename, n_examples=25):
    plt.figure(figsize=(15, 15))

    # Выбираем случайные примеры
    indices = np.random.choice(len(x), n_examples, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)

        # Для цветных изображений (CIFAR-10)
        if len(x[idx].shape) == 3 and x[idx].shape[2] == 3:
            plt.imshow(x[idx])
        else:  # Для черно-белых изображений (MNIST, Fashion MNIST)
            plt.imshow(x[idx].reshape(x[idx].shape[0], x[idx].shape[1]), cmap="gray")

        # Добавляем заголовок с истинным и предсказанным классами
        true_class = class_names[y[idx]]
        pred_class_mlp = class_names[y_pred_mlp[idx]]
        pred_class_cnn = class_names[y_pred_cnn[idx]]

        # Определяем цвет для MLP и CNN предсказаний
        mlp_color = "green" if y[idx] == y_pred_mlp[idx] else "red"
        cnn_color = "green" if y[idx] == y_pred_cnn[idx] else "red"

        plt.title(
            f"True: {true_class}\nMLP: {pred_class_mlp} ({mlp_color})\nCNN: {pred_class_cnn} ({cnn_color})",
            fontsize=8,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Функция для визуализации сравнительных характеристик
def plot_comparison_metrics(
    mlp_acc,
    cnn_acc,
    mlp_train_time,
    cnn_train_time,
    mlp_pred_time,
    cnn_pred_time,
    mlp_params,
    cnn_params,
    filename,
):
    metrics = ["Accuracy", "Training Time (s)", "Prediction Time (s)", "Parameters"]
    mlp_values = [mlp_acc, mlp_train_time, mlp_pred_time, mlp_params]
    cnn_values = [cnn_acc, cnn_train_time, cnn_pred_time, cnn_params]

    # Для параметров используем логарифмическую шкалу
    mlp_values[3] = np.log10(mlp_values[3])
    cnn_values[3] = np.log10(cnn_values[3])

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, mlp_values, width, label="MLP")
    rects2 = ax.bar(x + width / 2, cnn_values, width, label="CNN")

    ax.set_ylabel("Value")
    ax.set_title("Comparison of MLP and CNN")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Добавление значений над столбцами
    def autolabel(rects, values):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if i == 3:  # Для параметров показываем реальное значение, а не логарифм
                if values[i] == mlp_values[i]:
                    value_text = f"{10 ** values[i]:.0f}"
                else:
                    value_text = f"{10 ** values[i]:.0f}"
            else:
                value_text = f"{values[i]:.4f}" if i == 0 else f"{values[i]:.2f}"
            ax.annotate(
                value_text,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90 if len(value_text) > 6 else 0,
            )

    autolabel(rects1, mlp_values)
    autolabel(rects2, cnn_values)

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Определение пути к текущему скрипту
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Создание директории для сохранения результатов внутри директории скрипта
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Выбор набора данных
    dataset_name = "fashion_mnist"  # Можно изменить на "mnist" или "cifar10"

    print(f"Загрузка и подготовка набора данных {dataset_name}...")
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat), class_names = (
        load_and_prepare_data(dataset_name)
    )

    # Вывод информации о наборе данных
    print(f"Размер обучающего набора: {x_train.shape}")
    print(f"Размер тестового набора: {x_test.shape}")
    print(f"Количество классов: {len(class_names)}")

    # Создание и обучение полносвязной нейронной сети (MLP)
    print("\nСоздание и обучение полносвязной нейронной сети (MLP)...")
    mlp_model, mlp_history, mlp_acc, mlp_train_time, mlp_pred_time = (
        create_and_train_mlp(
            x_train,
            y_train_cat,
            x_test,
            y_test_cat,
            hidden_layers=[256, 128, 64],
            dropout_rate=0.3,
        )
    )

    # Создание и обучение сверточной нейронной сети (CNN)
    print("\nСоздание и обучение сверточной нейронной сети (CNN)...")
    cnn_model, cnn_history, cnn_acc, cnn_train_time, cnn_pred_time = (
        create_and_train_cnn(
            x_train,
            y_train_cat,
            x_test,
            y_test_cat,
            conv_layers=[(32, 3), (64, 3), (128, 3)],
            dense_layers=[128],
            dropout_rate=0.3,
        )
    )

    # Визуализация истории обучения
    plot_training_history(
        mlp_history,
        cnn_history,
        os.path.join(results_dir, f"{dataset_name}_training_history.png"),
    )

    # Получение предсказаний моделей
    y_pred_mlp_prob = mlp_model.predict(x_test)
    y_pred_mlp = np.argmax(y_pred_mlp_prob, axis=1)

    y_pred_cnn_prob = cnn_model.predict(x_test)
    y_pred_cnn = np.argmax(y_pred_cnn_prob, axis=1)

    # Визуализация матрицы ошибок для MLP
    plot_confusion_matrix(
        y_test,
        y_pred_mlp,
        class_names,
        os.path.join(results_dir, f"{dataset_name}_mlp_confusion_matrix.png"),
    )

    # Визуализация матрицы ошибок для CNN
    plot_confusion_matrix(
        y_test,
        y_pred_cnn,
        class_names,
        os.path.join(results_dir, f"{dataset_name}_cnn_confusion_matrix.png"),
    )

    # Визуализация примеров с предсказаниями
    plot_examples(
        x_test,
        y_test,
        y_pred_mlp,
        y_pred_cnn,
        class_names,
        os.path.join(results_dir, f"{dataset_name}_predictions.png"),
    )

    # Визуализация сравнительных характеристик
    plot_comparison_metrics(
        mlp_acc,
        cnn_acc,
        mlp_train_time,
        cnn_train_time,
        mlp_pred_time,
        cnn_pred_time,
        mlp_model.count_params(),
        cnn_model.count_params(),
        os.path.join(results_dir, f"{dataset_name}_comparison_metrics.png"),
    )

    # Вывод отчета о классификации для MLP
    print("\nОтчет о классификации для MLP:")
    print(classification_report(y_test, y_pred_mlp, target_names=class_names))

    # Вывод отчета о классификации для CNN
    print("\nОтчет о классификации для CNN:")
    print(classification_report(y_test, y_pred_cnn, target_names=class_names))

    # Сохранение моделей
    mlp_model.save(os.path.join(results_dir, f"{dataset_name}_mlp_model.keras"))
    cnn_model.save(os.path.join(results_dir, f"{dataset_name}_cnn_model.keras"))

    print("\nЛабораторная работа №01.6 выполнена успешно!")
    print(f"Результаты сохранены в директории {results_dir}")


if __name__ == "__main__":
    main()
