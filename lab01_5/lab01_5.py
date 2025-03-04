"""
Лабораторная работа №01.5
Классификация изображений с помощью нейронных сетей (часть 1)

Цель работы: знакомство с методами классификации изображений с использованием нейронных сетей.

Задачи работы: реализация нейронной сети для классификации изображения согласно варианту.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
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


# Функция для создания и обучения полносвязной нейронной сети
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

    # Оценка модели
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Точность на тестовом наборе: {test_acc:.4f}")

    return model, history


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


# Функция для визуализации примеров изображений
def plot_examples(x, y, y_pred, class_names, filename, n_examples=25):
    plt.figure(figsize=(10, 10))

    # Выбираем случайные примеры
    indices = np.random.choice(len(x), n_examples, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)

        # Для цветных изображений (CIFAR-10)
        if len(x[idx].shape) == 3 and x[idx].shape[2] == 3:
            plt.imshow(x[idx])
        else:  # Для черно-белых изображений (MNIST, Fashion MNIST)
            plt.imshow(x[idx], cmap="gray")

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
    dataset_name = "fashion_mnist"  # Можно изменить на "mnist" или "cifar10"

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

        # Для цветных изображений (CIFAR-10)
        if len(x_train[i].shape) == 3 and x_train[i].shape[2] == 3:
            plt.imshow(x_train[i])
        else:  # Для черно-белых изображений (MNIST, Fashion MNIST)
            plt.imshow(x_train[i], cmap="gray")

        plt.title(class_names[y_train[i]], fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{dataset_name}_examples.png"))
    plt.close()

    # Создание и обучение модели
    print("Создание и обучение модели...")
    model, history = create_and_train_mlp(
        x_train,
        y_train_cat,
        x_test,
        y_test_cat,
        hidden_layers=[256, 128, 64],
        dropout_rate=0.3,
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
    model.save(os.path.join(results_dir, f"{dataset_name}_mlp_model.keras"))

    print("\nЛабораторная работа №01.5 выполнена успешно!")
    print(f"Результаты сохранены в директории {results_dir}")


if __name__ == "__main__":
    main()
