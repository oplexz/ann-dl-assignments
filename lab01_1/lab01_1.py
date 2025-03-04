"""
Лабораторная работа №01.1
Интервальное прогнозирование временного ряда (часть 1)

Цель работы: сравнение полносвязного перцептрона с рекуррентными нейронными сетями.

Задачи работы: реализация полносвязного перцептрона и рекуррентной нейронной сети,
обучение, проверка на тестовом наборе данных, расчет сравнительных характеристик.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, SimpleRNN
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


# Функция для подготовки данных для обучения
def prepare_data(series, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(series) - n_steps_in - n_steps_out + 1):
        X.append(series[i : i + n_steps_in])
        y.append(series[i + n_steps_in : i + n_steps_in + n_steps_out])
    return np.array(X), np.array(y)


# Функция для создания и обучения полносвязного перцептрона
def create_mlp_model(input_shape, output_shape):
    model = Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(output_shape),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


# Функция для создания и обучения рекуррентной нейронной сети
def create_rnn_model(input_shape, output_shape):
    model = Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape, 1)),
            SimpleRNN(64, return_sequences=False),
            Dense(32, activation="relu"),
            Dense(output_shape),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


# Функция для оценки моделей
def evaluate_models(X_test, y_test, mlp_model, rnn_model, scaler):
    # Подготовка данных для MLP
    X_test_mlp = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Подготовка данных для RNN
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Предсказания
    mlp_predictions = mlp_model.predict(X_test_mlp)
    rnn_predictions = rnn_model.predict(X_test_rnn)

    # Инвертирование нормализации
    if scaler is not None:
        y_test = scaler.inverse_transform(y_test)
        mlp_predictions = scaler.inverse_transform(mlp_predictions)
        rnn_predictions = scaler.inverse_transform(rnn_predictions)

    # Расчет метрик
    mlp_mse = mean_squared_error(y_test, mlp_predictions)
    mlp_mae = mean_absolute_error(y_test, mlp_predictions)

    rnn_mse = mean_squared_error(y_test, rnn_predictions)
    rnn_mae = mean_absolute_error(y_test, rnn_predictions)

    print("MLP - MSE:", mlp_mse, "MAE:", mlp_mae)
    print("RNN - MSE:", rnn_mse, "MAE:", rnn_mae)

    return mlp_predictions, rnn_predictions


# Функция для визуализации результатов
def plot_results(y_test, mlp_predictions, rnn_predictions, output_dir, n_samples=5):
    plt.figure(figsize=(15, 10))

    for i in range(min(n_samples, len(y_test))):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(y_test[i], label="Actual")
        plt.plot(mlp_predictions[i], label="MLP Prediction")
        plt.plot(rnn_predictions[i], label="RNN Prediction")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_results.png"))
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

    # Подготовка данных для MLP
    X_train_mlp = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test_mlp = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Подготовка данных для RNN
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Создание и обучение MLP
    mlp_model = create_mlp_model(n_steps_in, n_steps_out)
    mlp_history = mlp_model.fit(
        X_train_mlp, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1
    )

    # Создание и обучение RNN
    rnn_model = create_rnn_model(n_steps_in, n_steps_out)
    rnn_history = rnn_model.fit(
        X_train_rnn, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1
    )

    # Оценка моделей
    mlp_predictions, rnn_predictions = evaluate_models(
        X_test, y_test, mlp_model, rnn_model, scaler
    )

    # Визуализация результатов
    plot_results(y_test, mlp_predictions, rnn_predictions, results_dir)

    # Визуализация истории обучения
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(mlp_history.history["loss"], label="Train Loss")
    plt.plot(mlp_history.history["val_loss"], label="Validation Loss")
    plt.title("MLP Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rnn_history.history["loss"], label="Train Loss")
    plt.plot(rnn_history.history["val_loss"], label="Validation Loss")
    plt.title("RNN Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_history.png"))
    plt.close()

    # Сохранение моделей
    mlp_model.save(os.path.join(results_dir, "mlp_model.keras"))
    rnn_model.save(os.path.join(results_dir, "rnn_model.keras"))

    print("Лабораторная работа №01.1 выполнена успешно!")
    print(f"Результаты сохранены в директории {results_dir}")


if __name__ == "__main__":
    main()
