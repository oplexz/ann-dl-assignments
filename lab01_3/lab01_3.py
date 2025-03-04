"""
Лабораторная работа №01.3
Нейронные сети с краткосрочной памятью LSTM (часть 1)

Цель работы: знакомство с сетями с краткосрочной памятью.

Задачи работы: реализация LSTM, обучение, проверка на тестовом наборе данных.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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


# Функция для создания более сложного временного ряда (сумма синусоид с шумом)
def create_complex_time_series(n_samples=1000):
    time = np.arange(0, n_samples)
    # Создаем сумму синусоид с разными частотами и шумом
    series = (
        np.sin(0.02 * time)
        + 0.5 * np.sin(0.05 * time)
        + 0.3 * np.sin(0.1 * time)
        + np.random.normal(0, 0.1, n_samples)
    )
    return series


# Функция для подготовки данных для обучения
def prepare_data(series, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(series) - n_steps_in - n_steps_out + 1):
        X.append(series[i : i + n_steps_in])
        y.append(series[i + n_steps_in : i + n_steps_in + n_steps_out])
    return np.array(X), np.array(y)


# Функция для создания и обучения LSTM модели
def create_lstm_model(input_shape, output_shape, units=64, dropout_rate=0.2):
    model = Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape, 1)),
            LSTM(units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units // 2),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dense(output_shape),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


# Функция для оценки модели
def evaluate_model(X_test, y_test, model, scaler):
    # Подготовка данных для LSTM
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Предсказания
    predictions = model.predict(X_test_lstm)

    # Инвертирование нормализации
    if scaler is not None:
        y_test = scaler.inverse_transform(y_test)
        predictions = scaler.inverse_transform(predictions)

    # Расчет метрик
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

    return predictions


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


# Функция для визуализации истории обучения
def plot_training_history(history, filename):
    plt.figure(figsize=(12, 5))

    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

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
    n_steps_in = 20  # количество временных шагов для входа
    n_steps_out = 10  # количество временных шагов для выхода
    test_size = 0.2

    # Создание временных рядов
    simple_series = create_time_series(n_samples)
    complex_series = create_complex_time_series(n_samples)

    # Визуализация временных рядов
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(simple_series[:200])
    plt.title("Simple Time Series (First 200 points)")
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.subplot(2, 1, 2)
    plt.plot(complex_series[:200])
    plt.title("Complex Time Series (First 200 points)")
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "time_series_visualization.png"))
    plt.close()

    # Обработка простого временного ряда
    print("\nОбработка простого временного ряда:")

    # Нормализация данных
    simple_scaler = MinMaxScaler(feature_range=(0, 1))
    simple_series_scaled = simple_scaler.fit_transform(
        simple_series.reshape(-1, 1)
    ).flatten()

    # Подготовка данных
    X_simple, y_simple = prepare_data(simple_series_scaled, n_steps_in, n_steps_out)

    # Разделение на обучающую и тестовую выборки
    train_size = int(len(X_simple) * (1 - test_size))
    X_train_simple, X_test_simple = X_simple[:train_size], X_simple[train_size:]
    y_train_simple, y_test_simple = y_simple[:train_size], y_simple[train_size:]

    # Подготовка данных для LSTM
    X_train_simple_lstm = X_train_simple.reshape(
        X_train_simple.shape[0], X_train_simple.shape[1], 1
    )

    # Создание и обучение LSTM модели
    simple_model = create_lstm_model(n_steps_in, n_steps_out)

    simple_history = simple_model.fit(
        X_train_simple_lstm,
        y_train_simple,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    # Оценка модели
    simple_predictions = evaluate_model(
        X_test_simple, y_test_simple, simple_model, simple_scaler
    )

    # Визуализация результатов
    plot_results(
        y_test_simple,
        simple_predictions,
        "LSTM Predictions for Simple Time Series",
        os.path.join(results_dir, "simple_series_predictions.png"),
    )

    # Визуализация истории обучения
    plot_training_history(
        simple_history, os.path.join(results_dir, "simple_series_training_history.png")
    )

    # Обработка сложного временного ряда
    print("\nОбработка сложного временного ряда:")

    # Нормализация данных
    complex_scaler = MinMaxScaler(feature_range=(0, 1))
    complex_series_scaled = complex_scaler.fit_transform(
        complex_series.reshape(-1, 1)
    ).flatten()

    # Подготовка данных
    X_complex, y_complex = prepare_data(complex_series_scaled, n_steps_in, n_steps_out)

    # Разделение на обучающую и тестовую выборки
    train_size = int(len(X_complex) * (1 - test_size))
    X_train_complex, X_test_complex = X_complex[:train_size], X_complex[train_size:]
    y_train_complex, y_test_complex = y_complex[:train_size], y_complex[train_size:]

    # Подготовка данных для LSTM
    X_train_complex_lstm = X_train_complex.reshape(
        X_train_complex.shape[0], X_train_complex.shape[1], 1
    )

    # Создание и обучение LSTM модели
    complex_model = create_lstm_model(
        n_steps_in, n_steps_out, units=128, dropout_rate=0.3
    )

    complex_history = complex_model.fit(
        X_train_complex_lstm,
        y_train_complex,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    # Оценка модели
    complex_predictions = evaluate_model(
        X_test_complex, y_test_complex, complex_model, complex_scaler
    )

    # Визуализация результатов
    plot_results(
        y_test_complex,
        complex_predictions,
        "LSTM Predictions for Complex Time Series",
        os.path.join(results_dir, "complex_series_predictions.png"),
    )

    # Визуализация истории обучения
    plot_training_history(
        complex_history,
        os.path.join(results_dir, "complex_series_training_history.png"),
    )

    # Сохранение моделей
    simple_model.save(os.path.join(results_dir, "simple_lstm_model.keras"))
    complex_model.save(os.path.join(results_dir, "complex_lstm_model.keras"))

    print("\nЛабораторная работа №01.3 выполнена успешно!")
    print(f"Результаты сохранены в директории {results_dir}")


if __name__ == "__main__":
    main()
