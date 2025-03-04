"""
Лабораторная работа №01.4
Нейронные сети с краткосрочной памятью LSTM (часть 2)

Цель работы: исследование различных характеристик сетей с краткосрочной памятью.

Задачи работы: расчет характеристик LSTM сети при разных значениях гиперпараметров.
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
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Установка seed для воспроизводимости результатов
np.random.seed(42)
tf.random.set_seed(42)


# Функция для создания временного ряда (сумма синусоид с шумом)
def create_time_series(n_samples=1000):
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
def create_lstm_model(
    input_shape,
    output_shape,
    lstm_type="standard",
    units=64,
    layers=2,
    dropout_rate=0.2,
    learning_rate=0.001,
):
    model = Sequential()

    # Добавление входного слоя
    if lstm_type == "standard":
        # Стандартная LSTM
        model.add(tf.keras.layers.Input(shape=(input_shape, 1)))
        model.add(LSTM(units, return_sequences=(layers > 1)))
    elif lstm_type == "bidirectional":
        # Двунаправленная LSTM
        model.add(tf.keras.layers.Input(shape=(input_shape, 1)))
        model.add(Bidirectional(LSTM(units, return_sequences=(layers > 1))))
    else:
        raise ValueError(
            "Неподдерживаемый тип LSTM. Используйте 'standard' или 'bidirectional'"
        )

    # Добавление dropout после первого слоя
    model.add(Dropout(dropout_rate))

    # Добавление дополнительных слоев LSTM, если требуется
    for i in range(1, layers):
        if lstm_type == "standard":
            model.add(LSTM(units // (2**i), return_sequences=(i < layers - 1)))
        elif lstm_type == "bidirectional":
            model.add(
                Bidirectional(LSTM(units // (2**i), return_sequences=(i < layers - 1)))
            )
        model.add(Dropout(dropout_rate))

    # Добавление выходных слоев
    model.add(Dense(32, activation="relu"))
    model.add(Dense(output_shape))

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    return model


# Функция для оценки модели
def evaluate_model(X_test, y_test, model, scaler):
    # Подготовка данных для LSTM
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Замер времени предсказания
    start_time = time.time()
    predictions = model.predict(X_test_lstm, verbose=0)
    prediction_time = time.time() - start_time

    # Инвертирование нормализации
    if scaler is not None:
        y_test = scaler.inverse_transform(y_test)
        predictions = scaler.inverse_transform(predictions)

    # Расчет метрик
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return mse, mae, predictions, prediction_time


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

    # Создание временного ряда
    series = create_time_series(n_samples)

    # Визуализация временного ряда
    plt.figure(figsize=(12, 6))
    plt.plot(series[:200])
    plt.title("Time Series (First 200 points)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.savefig(os.path.join(results_dir, "time_series.png"))
    plt.close()

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Подготовка данных
    X, y = prepare_data(series_scaled, n_steps_in, n_steps_out)

    # Разделение на обучающую и тестовую выборки
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Подготовка данных для LSTM
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Гиперпараметры для исследования
    lstm_types = ["standard", "bidirectional"]
    units_list = [32, 64, 128]
    layers_list = [1, 2, 3]
    dropout_rates = [0.0, 0.2, 0.5]
    learning_rates = [0.01, 0.001, 0.0001]

    # Таблица результатов
    results = []

    # Перебор комбинаций гиперпараметров
    for lstm_type, units, layers, dropout_rate, lr in product(
        lstm_types, units_list, layers_list, dropout_rates, learning_rates
    ):
        # Ограничим количество комбинаций для ускорения выполнения
        # Пропустим некоторые комбинации
        if (lstm_type == "bidirectional" and units == 128 and layers == 3) or (
            dropout_rate == 0.5 and lr == 0.0001
        ):
            continue

        model_name = f"{lstm_type}_u{units}_l{layers}_d{dropout_rate}_lr{lr}"
        print(f"Обучение модели: {model_name}")

        # Создание модели
        model = create_lstm_model(
            n_steps_in,
            n_steps_out,
            lstm_type=lstm_type,
            units=units,
            layers=layers,
            dropout_rate=dropout_rate,
            learning_rate=lr,
        )

        # Замер времени обучения
        start_time = time.time()

        # Обучение модели
        history = model.fit(
            X_train_lstm,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
        )

        training_time = time.time() - start_time

        # Оценка модели
        mse, mae, predictions, prediction_time = evaluate_model(
            X_test, y_test, model, scaler
        )

        # Сохранение результатов
        results.append(
            {
                "lstm_type": lstm_type,
                "units": units,
                "layers": layers,
                "dropout_rate": dropout_rate,
                "learning_rate": lr,
                "mse": mse,
                "mae": mae,
                "training_time": training_time,
                "prediction_time": prediction_time,
                "final_val_loss": history.history["val_loss"][-1],
                "model_params": model.count_params(),
            }
        )

        # Визуализация результатов для некоторых моделей
        if (
            lstm_type == "standard"
            and units == 64
            and layers == 2
            and dropout_rate == 0.2
            and lr == 0.001
        ) or (
            lstm_type == "bidirectional"
            and units == 64
            and layers == 2
            and dropout_rate == 0.2
            and lr == 0.001
        ):
            # Визуализация предсказаний
            plt.figure(figsize=(15, 10))
            for i in range(min(5, len(y_test))):
                plt.subplot(5, 1, i + 1)
                plt.plot(y_test[i], label="Actual")
                plt.plot(predictions[i], label="Prediction")
                plt.legend()

            plt.suptitle(f"LSTM Predictions: {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"predictions_{model_name}.png"))
            plt.close()

            # Визуализация истории обучения
            plt.figure(figsize=(10, 5))
            plt.plot(history.history["loss"], label="Train Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.title(f"Training History: {model_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(results_dir, f"history_{model_name}.png"))
            plt.close()

    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(results_dir, "lstm_hyperparameters_results.csv"), index=False
    )

    # Визуализация сравнения результатов
    plt.figure(figsize=(20, 15))

    # MSE для разных типов LSTM
    plt.subplot(3, 2, 1)
    for lstm_type in lstm_types:
        type_results = results_df[results_df["lstm_type"] == lstm_type]
        plt.plot(
            type_results["units"], type_results["mse"], marker="o", label=lstm_type
        )
    plt.title("MSE vs Units by LSTM Type")
    plt.xlabel("Units")
    plt.ylabel("MSE")
    plt.legend()

    # MSE для разного количества слоев
    plt.subplot(3, 2, 2)
    for layers in layers_list:
        layers_results = results_df[results_df["layers"] == layers]
        plt.plot(
            layers_results["units"],
            layers_results["mse"],
            marker="o",
            label=f"layers={layers}",
        )
    plt.title("MSE vs Units by Number of Layers")
    plt.xlabel("Units")
    plt.ylabel("MSE")
    plt.legend()

    # MSE для разных dropout rates
    plt.subplot(3, 2, 3)
    for dropout_rate in dropout_rates:
        dropout_results = results_df[results_df["dropout_rate"] == dropout_rate]
        plt.plot(
            dropout_results["units"],
            dropout_results["mse"],
            marker="o",
            label=f"dropout={dropout_rate}",
        )
    plt.title("MSE vs Units by Dropout Rate")
    plt.xlabel("Units")
    plt.ylabel("MSE")
    plt.legend()

    # Время обучения для разных типов LSTM
    plt.subplot(3, 2, 4)
    for lstm_type in lstm_types:
        type_results = results_df[results_df["lstm_type"] == lstm_type]
        plt.plot(
            type_results["units"],
            type_results["training_time"],
            marker="o",
            label=lstm_type,
        )
    plt.title("Training Time vs Units by LSTM Type")
    plt.xlabel("Units")
    plt.ylabel("Training Time (s)")
    plt.legend()

    # Время предсказания для разных типов LSTM
    plt.subplot(3, 2, 5)
    for lstm_type in lstm_types:
        type_results = results_df[results_df["lstm_type"] == lstm_type]
        plt.plot(
            type_results["units"],
            type_results["prediction_time"],
            marker="o",
            label=lstm_type,
        )
    plt.title("Prediction Time vs Units by LSTM Type")
    plt.xlabel("Units")
    plt.ylabel("Prediction Time (s)")
    plt.legend()

    # Количество параметров для разных типов LSTM
    plt.subplot(3, 2, 6)
    for lstm_type in lstm_types:
        type_results = results_df[results_df["lstm_type"] == lstm_type]
        plt.plot(
            type_results["units"],
            type_results["model_params"],
            marker="o",
            label=lstm_type,
        )
    plt.title("Model Parameters vs Units by LSTM Type")
    plt.xlabel("Units")
    plt.ylabel("Number of Parameters")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_results.png"))
    plt.close()

    # Вывод лучшей модели
    best_model_idx = results_df["mse"].idxmin()
    best_model = results_df.iloc[best_model_idx]

    print("\nЛучшая модель:")
    print(f"Тип LSTM: {best_model['lstm_type']}")
    print(f"Количество нейронов: {best_model['units']}")
    print(f"Количество слоев: {best_model['layers']}")
    print(f"Dropout rate: {best_model['dropout_rate']}")
    print(f"Learning rate: {best_model['learning_rate']}")
    print(f"MSE: {best_model['mse']}")
    print(f"MAE: {best_model['mae']}")
    print(f"Время обучения: {best_model['training_time']} секунд")
    print(f"Время предсказания: {best_model['prediction_time']} секунд")
    print(f"Количество параметров: {best_model['model_params']}")

    # Дополнительный анализ: влияние гиперпараметров на MSE
    print("\nВлияние гиперпараметров на MSE:")

    # Влияние типа LSTM
    lstm_type_effect = results_df.groupby("lstm_type")["mse"].mean()
    print(f"Средний MSE по типу LSTM:")
    for lstm_type, mse in lstm_type_effect.items():
        print(f"  {lstm_type}: {mse}")

    # Влияние количества нейронов
    units_effect = results_df.groupby("units")["mse"].mean()
    print(f"Средний MSE по количеству нейронов:")
    for units, mse in units_effect.items():
        print(f"  {units}: {mse}")

    # Влияние количества слоев
    layers_effect = results_df.groupby("layers")["mse"].mean()
    print(f"Средний MSE по количеству слоев:")
    for layers, mse in layers_effect.items():
        print(f"  {layers}: {mse}")

    # Влияние dropout rate
    dropout_effect = results_df.groupby("dropout_rate")["mse"].mean()
    print(f"Средний MSE по dropout rate:")
    for dropout, mse in dropout_effect.items():
        print(f"  {dropout}: {mse}")

    # Влияние learning rate
    lr_effect = results_df.groupby("learning_rate")["mse"].mean()
    print(f"Средний MSE по learning rate:")
    for lr, mse in lr_effect.items():
        print(f"  {lr}: {mse}")

    print("\nЛабораторная работа №01.4 выполнена успешно!")
    print(f"Результаты сохранены в директории {results_dir}")


if __name__ == "__main__":
    main()
