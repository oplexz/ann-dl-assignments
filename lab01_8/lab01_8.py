"""
Лабораторная работа №01.8
Капсульные нейронные сети (часть 2)

Цель работы: исследование различных характеристик капсульных сетей.

Задачи работы: расчет характеристик капсульной сети при разных значениях гиперпараметров.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
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


# Функция для создания и обучения капсульной нейронной сети
def create_and_train_capsnet(
    x_train,
    y_train_cat,
    x_test,
    y_test_cat,
    conv_filters=256,
    primary_dim=8,
    primary_channels=32,
    digit_dim=16,
    routing_iterations=3,
    learning_rate=0.001,
    batch_size=128,
    epochs=50,
):
    # Получение размеров входных данных
    input_shape = x_train.shape[1:]
    num_classes = y_train_cat.shape[1]

    # Входной слой
    inputs = Input(shape=input_shape)

    # Первый сверточный слой
    x = Conv2D(
        filters=conv_filters,
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
        optimizer=Adam(learning_rate=learning_rate),
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
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,  # Отключаем вывод для экономии места в консоли
    )

    training_time = time.time() - start_time

    # Замер времени предсказания
    start_time = time.time()
    y_pred = model.predict(x_test, verbose=0)
    prediction_time = time.time() - start_time

    # Оценка модели
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

    # Получение предсказаний
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_cat, axis=1)

    # Расчет метрик
    from sklearn.metrics import f1_score, precision_score, recall_score

    precision = precision_score(y_test_classes, y_pred_classes, average="weighted")
    recall = recall_score(y_test_classes, y_pred_classes, average="weighted")
    f1 = f1_score(y_test_classes, y_pred_classes, average="weighted")

    # Вывод информации о модели
    print(
        f"CapsNet - Точность: {test_acc:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )
    print(
        f"Время обучения: {training_time:.2f} с, Время предсказания: {prediction_time:.2f} с, "
        f"Параметры: {model.count_params()}"
    )

    return (
        model,
        history,
        test_acc,
        precision,
        recall,
        f1,
        training_time,
        prediction_time,
        model.count_params(),
    )


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


# Функция для визуализации сравнения результатов
def plot_comparison_results(results_df, metric, x_param, hue_param, filename):
    plt.figure(figsize=(12, 8))

    # Группировка данных по параметрам
    grouped_data = results_df.groupby([x_param, hue_param])[metric].mean().reset_index()

    # Создание графика
    sns.lineplot(
        data=grouped_data,
        x=x_param,
        y=metric,
        hue=hue_param,
        marker="o",
    )

    plt.title(f"{metric} vs {x_param} by {hue_param}")
    plt.xlabel(x_param)
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.7)

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

    # Для ускорения экспериментов, используем подвыборку данных
    # Используем 20% обучающих данных и 20% тестовых данных
    train_subset_size = int(len(x_train) * 0.2)
    test_subset_size = int(len(x_test) * 0.2)

    indices_train = np.random.choice(len(x_train), train_subset_size, replace=False)
    indices_test = np.random.choice(len(x_test), test_subset_size, replace=False)

    x_train_subset = x_train[indices_train]
    y_train_cat_subset = y_train_cat[indices_train]
    x_test_subset = x_test[indices_test]
    y_test_cat_subset = y_test_cat[indices_test]
    y_test_subset = y_test[indices_test]

    # Гиперпараметры для исследования
    conv_filters_list = [128, 256]
    primary_dim_list = [4, 8]
    primary_channels_list = [16, 32]
    digit_dim_list = [8, 16]
    routing_iterations_list = [1, 3]
    learning_rate_list = [0.001, 0.0001]

    # Таблица результатов
    results = []

    # Перебор комбинаций гиперпараметров
    # Для ускорения экспериментов, выберем только некоторые комбинации
    combinations = [
        # Базовая модель
        (256, 8, 32, 16, 3, 0.001),
        # Изменение conv_filters
        (128, 8, 32, 16, 3, 0.001),
        # Изменение primary_dim
        (256, 4, 32, 16, 3, 0.001),
        # Изменение primary_channels
        (256, 8, 16, 16, 3, 0.001),
        # Изменение digit_dim
        (256, 8, 32, 8, 3, 0.001),
        # Изменение routing_iterations
        (256, 8, 32, 16, 1, 0.001),
        # Изменение learning_rate
        (256, 8, 32, 16, 3, 0.0001),
    ]

    for (
        conv_filters,
        primary_dim,
        primary_channels,
        digit_dim,
        routing_iterations,
        learning_rate,
    ) in combinations:
        print(
            f"\nОбучение модели с параметрами: "
            f"conv_filters={conv_filters}, "
            f"primary_dim={primary_dim}, "
            f"primary_channels={primary_channels}, "
            f"digit_dim={digit_dim}, "
            f"routing_iterations={routing_iterations}, "
            f"learning_rate={learning_rate}"
        )

        # Создание и обучение модели
        (
            model,
            history,
            accuracy,
            precision,
            recall,
            f1,
            training_time,
            prediction_time,
            num_params,
        ) = create_and_train_capsnet(
            x_train_subset,
            y_train_cat_subset,
            x_test_subset,
            y_test_cat_subset,
            conv_filters=conv_filters,
            primary_dim=primary_dim,
            primary_channels=primary_channels,
            digit_dim=digit_dim,
            routing_iterations=routing_iterations,
            learning_rate=learning_rate,
            batch_size=128,
            epochs=20,  # Уменьшаем количество эпох для ускорения
        )

        # Сохранение результатов
        results.append(
            {
                "conv_filters": conv_filters,
                "primary_dim": primary_dim,
                "primary_channels": primary_channels,
                "digit_dim": digit_dim,
                "routing_iterations": routing_iterations,
                "learning_rate": learning_rate,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "training_time": training_time,
                "prediction_time": prediction_time,
                "num_params": num_params,
            }
        )

        # Визуализация истории обучения для некоторых моделей
        if (
            conv_filters == 256
            and primary_dim == 8
            and primary_channels == 32
            and digit_dim == 16
            and routing_iterations == 3
            and learning_rate == 0.001
        ):
            # Базовая модель
            plot_training_history(
                history,
                os.path.join(results_dir, f"{dataset_name}_base_model_history.png"),
            )

            # Получение предсказаний
            y_pred = model.predict(x_test_subset, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)

            # Визуализация матрицы ошибок
            cm = confusion_matrix(y_test_subset, y_pred_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.title("Confusion Matrix - Base Model")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    results_dir, f"{dataset_name}_base_model_confusion_matrix.png"
                )
            )
            plt.close()

            # Визуализация примеров с предсказаниями
            plt.figure(figsize=(10, 10))
            indices = np.random.choice(len(x_test_subset), 25, replace=False)
            for i, idx in enumerate(indices):
                plt.subplot(5, 5, i + 1)
                plt.imshow(x_test_subset[idx].reshape(28, 28), cmap="gray")
                true_class = class_names[y_test_subset[idx]]
                pred_class = class_names[y_pred_classes[idx]]
                color = "green" if y_test_subset[idx] == y_pred_classes[idx] else "red"
                plt.title(f"T: {true_class}\nP: {pred_class}", color=color, fontsize=8)
                plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(results_dir, f"{dataset_name}_base_model_predictions.png")
            )
            plt.close()

    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            results_dir, f"{dataset_name}_capsnet_hyperparameters_results.csv"
        ),
        index=False,
    )

    # Визуализация сравнения результатов
    # Точность в зависимости от различных параметров
    plot_comparison_results(
        results_df,
        "accuracy",
        "primary_dim",
        "routing_iterations",
        os.path.join(results_dir, f"{dataset_name}_accuracy_vs_primary_dim.png"),
    )

    plot_comparison_results(
        results_df,
        "accuracy",
        "digit_dim",
        "routing_iterations",
        os.path.join(results_dir, f"{dataset_name}_accuracy_vs_digit_dim.png"),
    )

    plot_comparison_results(
        results_df,
        "accuracy",
        "primary_channels",
        "primary_dim",
        os.path.join(results_dir, f"{dataset_name}_accuracy_vs_primary_channels.png"),
    )

    # Время обучения в зависимости от различных параметров
    plot_comparison_results(
        results_df,
        "training_time",
        "routing_iterations",
        "primary_dim",
        os.path.join(
            results_dir, f"{dataset_name}_training_time_vs_routing_iterations.png"
        ),
    )

    plot_comparison_results(
        results_df,
        "training_time",
        "primary_channels",
        "primary_dim",
        os.path.join(
            results_dir, f"{dataset_name}_training_time_vs_primary_channels.png"
        ),
    )

    # Количество параметров в зависимости от различных параметров
    plot_comparison_results(
        results_df,
        "num_params",
        "primary_dim",
        "digit_dim",
        os.path.join(results_dir, f"{dataset_name}_num_params_vs_primary_dim.png"),
    )

    # Визуализация сравнения метрик для всех моделей
    plt.figure(figsize=(14, 8))

    # Создаем уникальные метки для моделей
    model_labels = [
        f"CF{row['conv_filters']}_PD{row['primary_dim']}_PC{row['primary_channels']}_DD{row['digit_dim']}_RI{row['routing_iterations']}_LR{row['learning_rate']}"
        for _, row in results_df.iterrows()
    ]

    # Метрики для сравнения
    metrics = ["accuracy", "precision", "recall", "f1"]

    # Создаем данные для графика
    metrics_data = []
    for i, row in results_df.iterrows():
        for metric in metrics:
            metrics_data.append(
                {"model": model_labels[i], "metric": metric, "value": row[metric]}
            )

    metrics_df = pd.DataFrame(metrics_data)

    # Создаем график
    sns.barplot(data=metrics_df, x="model", y="value", hue="metric")
    plt.title("Comparison of Metrics Across Models")
    plt.xlabel("Model")
    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{dataset_name}_metrics_comparison.png"))
    plt.close()

    # Вывод лучшей модели по точности
    best_model_idx = results_df["accuracy"].idxmax()
    best_model = results_df.iloc[best_model_idx]

    print("\nЛучшая модель по точности:")
    print(f"Conv Filters: {best_model['conv_filters']}")
    print(f"Primary Dim: {best_model['primary_dim']}")
    print(f"Primary Channels: {best_model['primary_channels']}")
    print(f"Digit Dim: {best_model['digit_dim']}")
    print(f"Routing Iterations: {best_model['routing_iterations']}")
    print(f"Learning Rate: {best_model['learning_rate']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print(f"F1 Score: {best_model['f1']:.4f}")
    print(f"Training Time: {best_model['training_time']:.2f} seconds")
    print(f"Prediction Time: {best_model['prediction_time']:.2f} seconds")
    print(f"Number of Parameters: {best_model['num_params']}")

    # Вывод лучшей модели по времени предсказания
    fastest_model_idx = results_df["prediction_time"].idxmin()
    fastest_model = results_df.iloc[fastest_model_idx]

    print("\nСамая быстрая модель по времени предсказания:")
    print(f"Conv Filters: {fastest_model['conv_filters']}")
    print(f"Primary Dim: {fastest_model['primary_dim']}")
    print(f"Primary Channels: {fastest_model['primary_channels']}")
    print(f"Digit Dim: {fastest_model['digit_dim']}")
    print(f"Routing Iterations: {fastest_model['routing_iterations']}")
    print(f"Learning Rate: {fastest_model['learning_rate']}")
    print(f"Accuracy: {fastest_model['accuracy']:.4f}")
    print(f"Prediction Time: {fastest_model['prediction_time']:.2f} seconds")

    # Вывод модели с наименьшим количеством параметров
    smallest_model_idx = results_df["num_params"].idxmin()
    smallest_model = results_df.iloc[smallest_model_idx]

    print("\nМодель с наименьшим количеством параметров:")
    print(f"Conv Filters: {smallest_model['conv_filters']}")
    print(f"Primary Dim: {smallest_model['primary_dim']}")
    print(f"Primary Channels: {smallest_model['primary_channels']}")
    print(f"Digit Dim: {smallest_model['digit_dim']}")
    print(f"Routing Iterations: {smallest_model['routing_iterations']}")
    print(f"Learning Rate: {smallest_model['learning_rate']}")
    print(f"Accuracy: {smallest_model['accuracy']:.4f}")
    print(f"Number of Parameters: {smallest_model['num_params']}")

    print("\nЛабораторная работа №01.8 выполнена успешно!")
    print(f"Результаты сохранены в директории {results_dir}")


if __name__ == "__main__":
    main()
