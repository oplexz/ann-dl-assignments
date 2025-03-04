"""
Лабораторная работа №01.2
Интервальное прогнозирование временного ряда (часть 2)

Цель работы: исследование различных характеристик рекуррентных нейронных сетей.

Задачи работы: расчет характеристик рекуррентной сети при выборе разных гиперпараметров.
"""

import os
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Определение общей функции предсказания
@tf.function
def predict_with_model(model, x):
    # Преобразуем входные данные в тензор, если они еще не являются тензором
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x)
    return model(x, training=False)


# Кэш для хранения конкретных функций для каждой модели
concrete_functions = {}


# Функция для получения или создания конкретной функции для модели
def get_concrete_function(model, input_shape):
    model_id = id(model)
    if model_id not in concrete_functions:
        # Создаем конкретную функцию для данной модели и формы входных данных
        dummy_input = tf.zeros(input_shape, dtype=tf.float32)
        concrete_functions[model_id] = predict_with_model.get_concrete_function(
            model, dummy_input
        )
    return concrete_functions[model_id]


# Установка seed для воспроизводимости результатов
np.random.seed(42)
tf.random.set_seed(42)


# Функция для создания временного ряда (синусоида с шумом)
def create_time_series(n_samples=1000):
    time = np.arange(0, n_samples)
    # Создаем синусоиду с шумом
    noise = np.random.normal(0, 0.1, n_samples)
    series = np.sin(0.02 * time) + noise
    return series


# Функция для подготовки данных для обучения
def prepare_data(series, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(series) - n_steps_in - n_steps_out + 1):
        X.append(series[i : i + n_steps_in])
        y.append(series[i + n_steps_in : i + n_steps_in + n_steps_out])
    return np.array(X), np.array(y)


# Функция для создания и обучения рекуррентной нейронной сети
def create_rnn_model(
    input_shape,
    output_shape,
    rnn_type="simple",
    units=64,
    learning_rate=0.001,
    activation="tanh",
):
    model = Sequential()

    # Выбор типа рекуррентной сети
    if rnn_type.lower() == "simple":
        model.add(tf.keras.layers.Input(shape=(input_shape, 1)))
        model.add(SimpleRNN(units, activation=activation, return_sequences=False))
    elif rnn_type.lower() == "gru":
        model.add(tf.keras.layers.Input(shape=(input_shape, 1)))
        model.add(GRU(units, activation=activation, return_sequences=False))
    else:
        raise ValueError("Неподдерживаемый тип RNN. Используйте 'simple' или 'gru'")

    model.add(Dense(32, activation="relu"))
    model.add(Dense(output_shape))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


# Функция для оценки модели
def evaluate_model(X_test, y_test, model, scaler):
    # Подготовка данных для RNN
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Получение или создание конкретной функции для данной модели
    concrete_fn = get_concrete_function(model, X_test_rnn.shape)

    # Предсказания с использованием конкретной функции
    predictions = concrete_fn(
        model, tf.convert_to_tensor(X_test_rnn, dtype=tf.float32)
    ).numpy()

    # Инвертирование нормализации
    if scaler is not None:
        y_test = scaler.inverse_transform(y_test)
        predictions = scaler.inverse_transform(predictions)

    # Расчет метрик
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return mse, mae, predictions


# Функция для визуализации результатов
def plot_results(y_test, predictions, title, filename, n_samples=5):
    plt.figure(figsize=(15, 10))

    for i in range(min(n_samples, len(y_test))):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(y_test[i], label="Actual")
        plt.plot(predictions[i], label="Prediction")
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Определение пути к текущему скрипту
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Создание директории для сохранения результатов внутри директории скрипта
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Параметры
    n_samples = 1000
    n_steps_in = 10  # количество временных шагов для входа
    n_steps_out = 5  # количество временных шагов для выхода
    test_size = 0.2

    # Создание временного ряда
    series = create_time_series(n_samples)

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Подготовка данных
    X, y = prepare_data(series_scaled, n_steps_in, n_steps_out)

    # Разделение на обучающую и тестовую выборки
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Подготовка данных для RNN
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Гиперпараметры для исследования
    rnn_types = ["simple", "gru"]
    units_list = [32, 64, 128]
    learning_rates = [0.01, 0.001, 0.0001]
    activations = ["tanh", "relu"]

    # Таблица результатов
    results = []

    # Перебор всех комбинаций гиперпараметров
    for rnn_type, units, lr, activation in product(
        rnn_types, units_list, learning_rates, activations
    ):
        print(
            f"Обучение модели: {rnn_type} RNN, units={units}, lr={lr}, activation={activation}"
        )

        # Создание и обучение модели
        model = create_rnn_model(
            n_steps_in,
            n_steps_out,
            rnn_type=rnn_type,
            units=units,
            learning_rate=lr,
            activation=activation,
        )

        # Замер времени обучения
        start_time = time.time()

        history = model.fit(
            X_train_rnn,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
        )

        training_time = time.time() - start_time

        # Оценка модели
        mse, mae, predictions = evaluate_model(X_test, y_test, model, scaler)

        # Сохранение результатов
        results.append(
            {
                "rnn_type": rnn_type,
                "units": units,
                "learning_rate": lr,
                "activation": activation,
                "mse": mse,
                "mae": mae,
                "training_time": training_time,
                "final_val_loss": history.history["val_loss"][-1],
            }
        )

        # Визуализация результатов для текущей модели
        title = f"{rnn_type} RNN (units={units}, lr={lr}, activation={activation})"
        filename = os.path.join(
            results_dir, f"rnn_{rnn_type}_u{units}_lr{lr}_act{activation}.png"
        )
        plot_results(y_test, predictions, title, filename)

        # Визуализация истории обучения
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"Training History: {title}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                results_dir, f"history_{rnn_type}_u{units}_lr{lr}_act{activation}.png"
            )
        )
        plt.close()

    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(results_dir, "rnn_hyperparameters_results.csv"), index=False
    )

    # Визуализация сравнения результатов
    plt.figure(figsize=(15, 10))

    # MSE для разных типов RNN
    plt.subplot(2, 2, 1)
    for rnn_type in rnn_types:
        type_results = results_df[results_df["rnn_type"] == rnn_type]
        plt.plot(type_results["units"], type_results["mse"], marker="o", label=rnn_type)
    plt.title("MSE vs Units by RNN Type")
    plt.xlabel("Units")
    plt.ylabel("MSE")
    plt.legend()

    # MSE для разных learning rates
    plt.subplot(2, 2, 2)
    for lr in learning_rates:
        lr_results = results_df[results_df["learning_rate"] == lr]
        plt.plot(lr_results["units"], lr_results["mse"], marker="o", label=f"lr={lr}")
    plt.title("MSE vs Units by Learning Rate")
    plt.xlabel("Units")
    plt.ylabel("MSE")
    plt.legend()

    # Время обучения для разных типов RNN
    plt.subplot(2, 2, 3)
    for rnn_type in rnn_types:
        type_results = results_df[results_df["rnn_type"] == rnn_type]
        plt.plot(
            type_results["units"],
            type_results["training_time"],
            marker="o",
            label=rnn_type,
        )
    plt.title("Training Time vs Units by RNN Type")
    plt.xlabel("Units")
    plt.ylabel("Training Time (s)")
    plt.legend()

    # Финальная validation loss для разных типов RNN
    plt.subplot(2, 2, 4)
    for rnn_type in rnn_types:
        type_results = results_df[results_df["rnn_type"] == rnn_type]
        plt.plot(
            type_results["units"],
            type_results["final_val_loss"],
            marker="o",
            label=rnn_type,
        )
    plt.title("Final Validation Loss vs Units by RNN Type")
    plt.xlabel("Units")
    plt.ylabel("Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_results.png"))
    plt.close()

    # Вывод лучшей модели
    best_model_idx = results_df["mse"].idxmin()
    best_model = results_df.iloc[best_model_idx]

    print("\nЛучшая модель:")
    print(f"Тип RNN: {best_model['rnn_type']}")
    print(f"Количество нейронов: {best_model['units']}")
    print(f"Learning rate: {best_model['learning_rate']}")
    print(f"Функция активации: {best_model['activation']}")
    print(f"MSE: {best_model['mse']}")
    print(f"MAE: {best_model['mae']}")
    print(f"Время обучения: {best_model['training_time']} секунд")

    print("\nЛабораторная работа №01.2 выполнена успешно!")
    print(f"Результаты сохранены в директории {results_dir}")


if __name__ == "__main__":
    main()
