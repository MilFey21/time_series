# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional
from itertools import product
import itertools

# Для моделирования
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

# Для GARCH моделей
try:
    from arch import arch_model
except ImportError:
    print("Модуль arch не установлен. GARCH модели будут недоступны.")

# Для Prophet
try:
    from prophet import Prophet
except ImportError:
    print("Модуль prophet не установлен. Prophet модели будут недоступны.")


class SalesForecastingModel:
    """
    Класс для прогнозирования продаж в магазине.
    
    Функциональность:
    1. Предобработка исходных данных
    2. Обучение моделей прогнозирования
    3. Оценка качества прогнозов
    4. Сохранение и загрузка моделей
    5. Прогнозирование на неделю, месяц и квартал
    """
    
    def __init__(self, store_id: str = 'STORE_1'):
        """
        Инициализация класса.
        
        Args:
            store_id: ID магазина для прогнозирования (по умолчанию STORE_1)
        """
        self.store_id = store_id
        self.models = {}
        self.best_models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'cnt'
        self.date_column = 'date'
        self.item_column = 'item_id'
        self.metrics = {}
        self.processed_data = None
        self.sales_data = None
        self.calendar_data = None
        self.prices_data = None
        self.items = None
        
    def load_data(self, sales_path: str, calendar_path: str, prices_path: str) -> None:
        """
        Загрузка исходных данных.
        
        Args:
            sales_path: Путь к файлу с данными о продажах
            calendar_path: Путь к файлу с календарными данными
            prices_path: Путь к файлу с данными о ценах
        """
        # Загрузка данных
        sales = pd.read_csv('data/shop_sales.csv')
        calendar = pd.read_csv('data/shop_sales_dates.csv')
        prices = pd.read_csv('data/shop_sales_prices.csv')
        
        # Фильтрация данных по выбранному магазину
        sales = sales[sales['store_id'] == self.store_id]
        prices = prices[prices['store_id'] == self.store_id]
        
        # Сохранение данных
        self.sales_data = sales
        self.calendar_data = calendar
        self.prices_data = prices
        
        # Получение списка товаров в магазине
        self.items = sales['item_id'].unique()
        
        print(f"Данные загружены. Магазин {self.store_id} содержит {len(self.items)} товаров.")
        
    def preprocess_data(self) -> pd.DataFrame:
        """
        Предобработка данных в удобный формат для моделирования.
        
        Returns:
            Предобработанный DataFrame
        """
        if self.sales_data is None or self.calendar_data is None or self.prices_data is None:
            raise ValueError("Данные не загружены. Используйте метод load_data() сначала.")
        
        # Объединение данных о продажах с календарем
        sales = self.sales_data.copy()
        calendar = self.calendar_data.copy()
        prices = self.prices_data.copy()
        
        # Преобразование даты в datetime
        calendar['date'] = pd.to_datetime(calendar['date'])
        
        # Объединение данных о продажах с календарем
        data = sales.merge(calendar[['date_id', 'date', 'wm_yr_wk', 'weekday', 'month', 'year', 
                                    f'CASHBACK_{self.store_id}']], 
                          on='date_id', how='left')
        
        # Объединение с данными о ценах
        data = data.merge(prices[['item_id', 'wm_yr_wk', 'sell_price']], 
                         on=['item_id', 'wm_yr_wk'], how='left')
        
        # Создание дополнительных признаков
        data['day_of_week'] = data['date'].dt.dayofweek
        data['day_of_month'] = data['date'].dt.day
        data['week_of_year'] = data['date'].dt.isocalendar().week
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Создание лаговых признаков для продаж
        for lag in [1, 7, 14, 28]:
            data[f'lag_{lag}'] = data.groupby('item_id')['cnt'].shift(lag)
        
        # Создание признаков скользящего среднего
        for window in [7, 14, 28]:
            data[f'rolling_mean_{window}'] = data.groupby('item_id')['cnt'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Создание признаков скользящего стандартного отклонения
        for window in [7, 14, 28]:
            data[f'rolling_std_{window}'] = data.groupby('item_id')['cnt'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
        
        # Создание признаков для праздников
        holiday_cols = [col for col in calendar.columns if 'event' in col]
        if holiday_cols:
            data = data.merge(calendar[['date_id'] + holiday_cols], on='date_id', how='left')
        
        # Заполнение пропущенных значений
        for col in data.columns:
            if data[col].dtype == 'float64' and data[col].isna().sum() > 0:
                data[col] = data[col].fillna(data.groupby('item_id')[col].transform('mean'))
        
        # Сохранение предобработанных данных
        self.processed_data = data
        
        # Определение признаков для моделирования
        self.feature_columns = [
            'sell_price', 'day_of_week', 'day_of_month', 'week_of_year', 'month', 'year',
            'is_weekend', f'CASHBACK_{self.store_id}'
        ]
        
        # Добавление лаговых признаков и скользящих статистик
        for lag in [1, 7, 14, 28]:
            self.feature_columns.append(f'lag_{lag}')
        
        for window in [7, 14, 28]:
            self.feature_columns.append(f'rolling_mean_{window}')
            self.feature_columns.append(f'rolling_std_{window}')
        
        # Добавление праздничных признаков
        if holiday_cols:
            self.feature_columns.extend(holiday_cols)
        
        print(f"Данные предобработаны. Создано {len(self.feature_columns)} признаков.")
        
        return data
    
    def train_models(self, test_size: int = 30) -> Dict:
        """
        Обучение различных моделей прогнозирования.
        
        Args:
            test_size: Размер тестовой выборки в днях
            
        Returns:
            Словарь с обученными моделями
        """
        if self.processed_data is None:
            raise ValueError("Данные не предобработаны. Используйте метод preprocess_data() сначала.")
        
        print("Начало обучения моделей...")
        
        # Обучение ARIMA моделей
        arima_models = self._train_arima_models(test_size)
        
        # Обучение ETS моделей
        ets_models = self._train_ets_models(test_size)
        
        # Обучение моделей линейной регрессии
        linear_models = self._train_linear_models(test_size)
        
        # Обучение моделей градиентного бустинга
        gbm_models = self._train_gbm_models(test_size)
        
        # Объединение всех моделей
        all_models = {
            'arima': arima_models,
            'ets': ets_models,
            'linear': linear_models,
            'gbm': gbm_models
        }
        
        # Сохранение моделей
        self.models = all_models
        
        print("Обучение моделей завершено.")
        
        return all_models
    
    def _train_arima_models(self, test_size: int = 30) -> Dict:
        """
        Обучение ARIMA моделей для каждого товара.
        
        Args:
            test_size: Размер тестовой выборки в днях
            
        Returns:
            Словарь с обученными ARIMA моделями
        """
        print("Обучение ARIMA моделей...")
        
        arima_models = {}
        
        for item in self.items:
            print(f"  Обучение ARIMA модели для товара {item}...")
            
            # Фильтрация данных для текущего товара
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            
            # Разделение на обучающую и тестовую выборки
            train_data = item_data.iloc[:-test_size]
            
            try:
                # Создание временного ряда для ARIMA
                # Важно: используем числовой индекс вместо дат, чтобы избежать ошибки с индексом
                y_train = train_data['cnt'].reset_index(drop=True)
                
                # Подбор параметров ARIMA
                best_aic = float('inf')
                best_params = None
                
                # Перебор параметров
                for p, d, q in itertools.product(range(3), range(2), range(3)):
                    try:
                        model = ARIMA(y_train, order=(p, d, q))
                        results = model.fit()
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_params = (p, d, q)
                    except:
                        continue
                
                if best_params is None:
                    print(f"    Не удалось подобрать параметры ARIMA для товара {item}. Используем (1,1,1).")
                    best_params = (1, 1, 1)
                
                # Обучение модели с лучшими параметрами
                final_model = ARIMA(y_train, order=best_params)
                arima_models[item] = final_model.fit()
                
                print(f"    ARIMA{best_params} модель обучена для товара {item}.")
            except Exception as e:
                print(f"    Ошибка при обучении ARIMA модели для товара {item}: {e}")
        
        return arima_models
    
    def _train_ets_models(self, test_size: int) -> Dict:
        """
        Обучение ETS моделей для каждого товара.
        
        Args:
            test_size: Размер тестовой выборки в днях
            
        Returns:
            Словарь с обученными ETS моделями
        """
        print("Обучение ETS моделей...")
        
        ets_models = {}
        
        for item in self.items:
            # Фильтрация данных для текущего товара
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            
            # Выделение обучающей выборки
            train_data = item_data.iloc[:-test_size]
            
            # Обучение ETS модели
            try:
                # Подбор оптимальных параметров
                best_aic = float('inf')
                best_params = None
                
                # Перебор параметров
                for trend, seasonal in product(['add', 'mul', None], ['add', 'mul', None]):
                    try:
                        # Пропускаем комбинацию, где оба параметра None
                        if trend is None and seasonal is None:
                            continue
                        
                        # Определяем сезонный период
                        seasonal_periods = 7  # Недельная сезонность
                        
                        model = ExponentialSmoothing(
                            train_data['cnt'],
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=seasonal_periods
                        )
                        results = model.fit()
                        
                        # Используем AIC как критерий качества
                        aic = len(train_data) * np.log(np.mean(results.resid ** 2)) + 2 * len(results.params)
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (trend, seasonal)
                    except:
                        continue
                
                # Если не удалось подобрать параметры, используем значения по умолчанию
                if best_params is None:
                    best_params = ('add', 'add')
                
                # Обучение модели с лучшими параметрами
                model = ExponentialSmoothing(
                    train_data['cnt'],
                    trend=best_params[0],
                    seasonal=best_params[1],
                    seasonal_periods=7
                )
                ets_models[item] = model.fit()
                
                print(f"  ETS модель для товара {item} обучена с параметрами {best_params}")
            except Exception as e:
                print(f"  Ошибка при обучении ETS модели для товара {item}: {e}")
                # Используем простую модель в случае ошибки
                model = ExponentialSmoothing(train_data['cnt'], trend='add', seasonal='add', seasonal_periods=7)
                ets_models[item] = model.fit()
        
        return ets_models
    
    def _train_linear_models(self, test_size: int = 30) -> Dict:
        """
        Обучение линейных моделей для каждого товара.
        
        Args:
            test_size: Размер тестовой выборки в днях
            
        Returns:
            Словарь с обученными линейными моделями
        """
        print("Обучение линейных моделей...")
        
        linear_models = {}
        
        for item in self.items:
            print(f"  Обучение линейной модели для товара {item}...")
            
            # Фильтрация данных для текущего товара
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            
            # Разделение на обучающую и тестовую выборки
            train_data = item_data.iloc[:-test_size]
            
            try:
                # Подготовка признаков и целевой переменной
                # Преобразуем категориальные признаки в числовые
                X_train = pd.get_dummies(train_data[self.feature_columns], drop_first=True)
                y_train = train_data[self.target_column]
                
                # Удаляем строки с пропущенными значениями
                mask = ~(X_train.isna().any(axis=1) | y_train.isna())
                X_train = X_train[mask]
                y_train = y_train[mask]
                
                # Обучение линейной модели
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Сохранение модели
                linear_models[item] = model
                
                print(f"    Линейная модель обучена для товара {item}.")
            except Exception as e:
                print(f"    Ошибка при обучении линейной модели для товара {item}: {e}")
        
        return linear_models
    
    def _train_gbm_models(self, test_size: int) -> Dict:
        """
        Обучение моделей градиентного бустинга для каждого товара.
        
        Args:
            test_size: Размер тестовой выборки в днях
            
        Returns:
            Словарь с обученными моделями градиентного бустинга и скейлерами
        """
        print("Обучение моделей градиентного бустинга...")
        
        gbm_models = {}
        self.scalers['gbm'] = {}
        
        for item in self.items:
            # Фильтрация данных для текущего товара
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            
            # Выделение обучающей выборки
            train_data = item_data.iloc[:-test_size]
            
            # Подготовка признаков и целевой переменной
            X = train_data[self.feature_columns].fillna(0)
            y = train_data[self.target_column]
            
            # Масштабирование признаков
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Сохранение скейлера
            self.scalers['gbm'][item] = scaler
            
            # Обучение модели градиентного бустинга
            try:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                model.fit(X_scaled, y)
                gbm_models[item] = model
                
                print(f"  Модель градиентного бустинга для товара {item} обучена")
            except Exception as e:
                print(f"  Ошибка при обучении модели градиентного бустинга для товара {item}: {e}")
        
        return gbm_models
    
    def evaluate_models(self, test_size: int = 30) -> pd.DataFrame:
        """
        Оценка качества моделей на тестовой выборке.
        
        Args:
            test_size: Размер тестовой выборки в днях
            
        Returns:
            DataFrame с метриками качества для каждой модели и товара
        """
        if not self.models:
            raise ValueError("Модели не обучены. Используйте метод train_models() сначала.")
        
        print("Оценка качества моделей...")
        
        # Список для хранения результатов
        results = []
        
        for item in self.items:
            # Фильтрация данных для текущего товара
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            
            # Выделение тестовой выборки
            test_data = item_data.iloc[-test_size:]
            
            # Фактические значения
            y_true = test_data[self.target_column].values
            
            # Оценка ARIMA модели
            if 'arima' in self.models and item in self.models['arima']:
                try:
                    # Прогноз ARIMA модели
                    y_pred = self.models['arima'][item].forecast(steps=test_size)
                    
                    # Расчет метрик
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    
                    # Добавление результатов
                    results.append({
                        'item_id': item,
                        'model': 'arima',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"  Ошибка при оценке ARIMA модели для товара {item}: {e}")
            
            # Оценка ETS модели
            if 'ets' in self.models and item in self.models['ets']:
                try:
                    # Прогноз ETS модели
                    y_pred = self.models['ets'][item].forecast(steps=test_size)
                    
                    # Расчет метрик
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    
                    # Добавление результатов
                    results.append({
                        'item_id': item,
                        'model': 'ets',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"  Ошибка при оценке ETS модели для товара {item}: {e}")
            
            # Оценка линейной модели
            if 'linear' in self.models and item in self.models['linear']:
                try:
                    # Подготовка признаков
                    X_test = test_data[self.feature_columns].fillna(0)
                    
                    # Масштабирование признаков
                    X_test_scaled = self.scalers['linear'][item].transform(X_test)
                    
                    # Прогноз линейной модели
                    y_pred = self.models['linear'][item].predict(X_test_scaled)
                    
                    # Расчет метрик
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    
                    # Добавление результатов
                    results.append({
                        'item_id': item,
                        'model': 'linear',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"  Ошибка при оценке линейной модели для товара {item}: {e}")
            
            # Оценка модели градиентного бустинга
            if 'gbm' in self.models and item in self.models['gbm']:
                try:
                    # Подготовка признаков
                    X_test = test_data[self.feature_columns].fillna(0)
                    
                    # Масштабирование признаков
                    X_test_scaled = self.scalers['gbm'][item].transform(X_test)
                    
                    # Прогноз модели градиентного бустинга
                    y_pred = self.models['gbm'][item].predict(X_test_scaled)
                    
                    # Расчет метрик
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    
                    # Добавление результатов
                    results.append({
                        'item_id': item,
                        'model': 'gbm',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"  Ошибка при оценке модели градиентного бустинга для товара {item}: {e}")
        
        # Создание DataFrame с результатами
        results_df = pd.DataFrame(results)
        
        # Сохранение результатов
        self.metrics = results_df
        
        # Определение лучшей модели для каждого товара
        best_models = results_df.loc[results_df.groupby('item_id')['MAE'].idxmin()]
        self.best_models = {row['item_id']: row['model'] for _, row in best_models.iterrows()}
        
        print("Оценка качества моделей завершена.")
        
        return results_df
    
    def save_models(self, path: str = 'models') -> None:
        """
        Сохранение обученных моделей.
        
        Args:
            path: Путь для сохранения моделей
        """
        if not self.models:
            raise ValueError("Модели не обучены. Используйте метод train_models() сначала.")
        
        # Создание директории, если она не существует
        os.makedirs(path, exist_ok=True)
        
        # Сохранение моделей
        with open(f"{path}/{self.store_id}_models.pkl", 'wb') as f:
            pickle.dump(self.models, f)
        
        # Сохранение скейлеров
        with open(f"{path}/{self.store_id}_scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Сохранение метаданных
        metadata = {
            'store_id': self.store_id,
            'feature_columns': self.feature_columns,
            'best_models': self.best_models,
            'items': self.items.tolist()
        }
        
        with open(f"{path}/{self.store_id}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Модели сохранены в директории {path}")
    
    def load_models(self, path: str = 'models') -> None:
        """
        Загрузка сохраненных моделей.
        
        Args:
            path: Путь к сохраненным моделям
        """
        # Загрузка моделей
        with open(f"{path}/{self.store_id}_models.pkl", 'rb') as f:
            self.models = pickle.load(f)
        
        # Загрузка скейлеров
        with open(f"{path}/{self.store_id}_scalers.pkl", 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Загрузка метаданных
        with open(f"{path}/{self.store_id}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Обновление атрибутов
        self.store_id = metadata['store_id']
        self.feature_columns = metadata['feature_columns']
        self.best_models = metadata['best_models']
        self.items = np.array(metadata['items'])
        
        print(f"Модели загружены из директории {path}")
    
    def forecast(self, horizon: int = 7, use_best_model: bool = True) -> pd.DataFrame:
        """
        Прогнозирование продаж на указанный горизонт.
        
        Args:
            horizon: Горизонт прогнозирования в днях
            use_best_model: Использовать только лучшую модель для каждого товара
            
        Returns:
            DataFrame с прогнозами
        """
        if not self.models:
            raise ValueError("Модели не обучены. Используйте метод train_models() сначала.")
        
        if self.processed_data is None:
            raise ValueError("Данные не предобработаны. Используйте метод preprocess_data() сначала.")
        
        print(f"Прогнозирование на {horizon} дней вперед...")
        
        # Список для хранения прогнозов
        all_forecasts = []
        
        # Последняя дата в данных
        last_date = self.processed_data['date'].max()
        
        for item in self.items:
            print(f"  Прогнозирование для товара {item}...")
            
            # Фильтрация данных для текущего товара
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            
            # Определение модели для прогнозирования
            if use_best_model and item in self.best_models:
                model_type = self.best_models[item]
                print(f"    Используется лучшая модель: {model_type}")
                
                # Прогнозирование с использованием лучшей модели
                item_forecast = self._forecast_item(item, model_type, item_data, last_date, horizon)
                if item_forecast is not None:
                    all_forecasts.append(item_forecast)
            else:
                # Если лучшая модель не определена или use_best_model=False, используем все доступные модели
                for model_type in self.models.keys():
                    if item in self.models[model_type]:
                        print(f"    Используется модель: {model_type}")
                        item_forecast = self._forecast_item(item, model_type, item_data, last_date, horizon)
                        if item_forecast is not None:
                            all_forecasts.append(item_forecast)
        
        # Объединение всех прогнозов
        if all_forecasts:
            forecast_df = pd.concat(all_forecasts, ignore_index=True)
            print(f"Прогнозирование завершено. Создано {len(forecast_df)} прогнозов.")
            return forecast_df
        else:
            print("Не удалось создать прогнозы.")
            return pd.DataFrame()
    
    def _forecast_item(self, item: str, model_type: str, item_data: pd.DataFrame, 
                      last_date: datetime, horizon: int) -> Optional[pd.DataFrame]:
        """
        Прогнозирование продаж для конкретного товара с использованием указанной модели.
        
        Args:
            item: ID товара
            model_type: Тип модели ('arima', 'linear', 'gbm', 'ets')
            item_data: DataFrame с данными для товара
            last_date: Последняя дата в данных
            horizon: Горизонт прогнозирования в днях
            
        Returns:
            DataFrame с прогнозами или None в случае ошибки
        """
        try:
            # Создание дат для прогноза
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
            
            # Создание DataFrame для прогноза
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'item_id': item,
                'store_id': self.store_id
            })
            
            # Добавление календарных признаков
            forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
            forecast_df['day_of_month'] = forecast_df['date'].dt.day
            forecast_df['week_of_year'] = forecast_df['date'].dt.isocalendar().week
            forecast_df['month'] = forecast_df['date'].dt.month
            forecast_df['year'] = forecast_df['date'].dt.year
            forecast_df['is_weekend'] = forecast_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Получение последних значений продаж для создания лагов
            last_sales = item_data['cnt'].tail(28).values
            
            # Прогнозирование в зависимости от типа модели
            if model_type == 'arima':
                # Прогнозирование с использованием ARIMA
                model = self.models['arima'][item]
                forecast_values = model.forecast(steps=horizon)
                forecast_df['cnt'] = forecast_values
                
            elif model_type == 'ets':
                # Прогнозирование с использованием ETS
                model = self.models['ets'][item]
                forecast_values = model.forecast(horizon)
                forecast_df['cnt'] = forecast_values
                
            elif model_type in ['linear', 'gbm']:
                # Прогнозирование с использованием ML моделей
                # Для этого нужно создать все необходимые признаки
                
                # Инициализация прогнозов
                forecasted_sales = np.zeros(horizon)
                
                # Последние значения для создания лагов
                for i in range(horizon):
                    # Создание лаговых признаков
                    if i == 0:
                        forecast_df.loc[i, 'lag_1'] = last_sales[-1]
                        forecast_df.loc[i, 'lag_7'] = last_sales[-7] if len(last_sales) >= 7 else np.mean(last_sales)
                        forecast_df.loc[i, 'lag_14'] = last_sales[-14] if len(last_sales) >= 14 else np.mean(last_sales)
                        forecast_df.loc[i, 'lag_28'] = last_sales[-28] if len(last_sales) >= 28 else np.mean(last_sales)
                    else:
                        forecast_df.loc[i, 'lag_1'] = forecasted_sales[i-1]
                        forecast_df.loc[i, 'lag_7'] = last_sales[-7+i] if i < 7 else forecasted_sales[i-7]
                        forecast_df.loc[i, 'lag_14'] = last_sales[-14+i] if i < 14 else forecasted_sales[i-14]
                        forecast_df.loc[i, 'lag_28'] = last_sales[-28+i] if i < 28 else forecasted_sales[i-28]
                    
                    # Создание скользящих средних
                    if i == 0:
                        forecast_df.loc[i, 'rolling_mean_7'] = np.mean(last_sales[-7:])
                        forecast_df.loc[i, 'rolling_mean_14'] = np.mean(last_sales[-14:])
                        forecast_df.loc[i, 'rolling_mean_28'] = np.mean(last_sales[-28:])
                        forecast_df.loc[i, 'rolling_std_7'] = np.std(last_sales[-7:])
                        forecast_df.loc[i, 'rolling_std_14'] = np.std(last_sales[-14:])
                        forecast_df.loc[i, 'rolling_std_28'] = np.std(last_sales[-28:])
                    else:
                        # Обновляем скользящие средние с учетом прогнозов
                        values_7 = np.concatenate([last_sales[-7+i:], forecasted_sales[:i]]) if i < 7 else forecasted_sales[i-7:i]
                        values_14 = np.concatenate([last_sales[-14+i:], forecasted_sales[:i]]) if i < 14 else forecasted_sales[i-14:i]
                        values_28 = np.concatenate([last_sales[-28+i:], forecasted_sales[:i]]) if i < 28 else forecasted_sales[i-28:i]
                        
                        forecast_df.loc[i, 'rolling_mean_7'] = np.mean(values_7)
                        forecast_df.loc[i, 'rolling_mean_14'] = np.mean(values_14)
                        forecast_df.loc[i, 'rolling_mean_28'] = np.mean(values_28)
                        forecast_df.loc[i, 'rolling_std_7'] = np.std(values_7)
                        forecast_df.loc[i, 'rolling_std_14'] = np.std(values_14)
                        forecast_df.loc[i, 'rolling_std_28'] = np.std(values_28)
                    
                    # Добавляем цену (используем последнюю известную цену)
                    forecast_df.loc[i, 'sell_price'] = item_data['sell_price'].iloc[-1]
                    
                    # Добавляем CASHBACK (используем последнее известное значение)
                    cashback_col = f'CASHBACK_{self.store_id}'
                    if cashback_col in item_data.columns:
                        forecast_df.loc[i, cashback_col] = item_data[cashback_col].iloc[-1]
                    
                    # Заполняем пропущенные значения средними
                    for col in self.feature_columns:
                        if col not in forecast_df.columns and col in item_data.columns:
                            forecast_df[col] = item_data[col].mean()
                    
                    # Прогнозирование
                    X = forecast_df.iloc[i:i+1][self.feature_columns].fillna(0)
                    
                    if model_type == 'linear':
                        # Прогноз линейной модели
                        forecasted_sales[i] = self.models['linear'][item].predict(X)[0]
                    elif model_type == 'gbm':
                        # Масштабирование признаков
                        X_scaled = self.scalers['gbm'][item].transform(X)
                        # Прогноз модели градиентного бустинга
                        forecasted_sales[i] = self.models['gbm'][item].predict(X_scaled)[0]
                
                # Добавляем прогнозы в DataFrame
                forecast_df['cnt'] = forecasted_sales
            
            # Округляем прогнозы до целых чисел и обеспечиваем неотрицательность
            forecast_df['cnt'] = np.maximum(0, np.round(forecast_df['cnt'])).astype(int)
            
            return forecast_df
            
        except Exception as e:
            print(f"    Ошибка при прогнозировании для товара {item} с моделью {model_type}: {e}")
            return None
    
    def forecast_weekly(self, use_best_model: bool = True) -> pd.DataFrame:
        """
        Прогнозирование продаж на неделю вперед.
        
        Args:
            use_best_model: Использовать только лучшую модель для каждого товара
            
        Returns:
            DataFrame с прогнозами на неделю
        """
        return self.forecast(horizon=7, use_best_model=use_best_model)
    
    def forecast_monthly(self, use_best_model: bool = True) -> pd.DataFrame:
        """
        Прогнозирование продаж на месяц вперед.
        
        Args:
            use_best_model: Использовать только лучшую модель для каждого товара
            
        Returns:
            DataFrame с прогнозами на месяц
        """
        return self.forecast(horizon=30, use_best_model=use_best_model)
    
    def forecast_quarterly(self, use_best_model: bool = True) -> pd.DataFrame:
        """
        Прогнозирование продаж на квартал вперед.
        
        Args:
            use_best_model: Использовать только лучшую модель для каждого товара
            
        Returns:
            DataFrame с прогнозами на квартал
        """
        return self.forecast(horizon=90, use_best_model=use_best_model)
    
    def plot_forecast(self, item_id: str, forecast_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None, 
                     days_to_show: int = 30) -> None:
        """
        Визуализация прогноза для выбранного товара.
        
        Args:
            item_id: ID товара
            forecast_data: DataFrame с прогнозами
            test_data: DataFrame с тестовыми данными (если есть)
            days_to_show: Количество дней исторических данных для отображения
        """
        if self.processed_data is None:
            raise ValueError("Данные не предобработаны. Используйте метод preprocess_data() сначала.")
        
        # Фильтрация данных для выбранного товара
        item_data = self.processed_data[self.processed_data['item_id'] == item_id].sort_values('date')
        item_forecast = forecast_data[forecast_data['item_id'] == item_id].sort_values('date')
        
        if item_data.empty:
            print(f"Нет данных для товара {item_id}")
            return
        
        if item_forecast.empty:
            print(f"Нет прогнозов для товара {item_id}")
            return
        
        # Выбор последних days_to_show дней исторических данных
        item_data = item_data.tail(days_to_show)
        
        # Создание графика
        plt.figure(figsize=(12, 6))
        
        # Исторические данные
        plt.plot(item_data['date'], item_data['cnt'], label='Исторические продажи', color='blue')
        
        # Тестовые данные (если есть)
        if test_data is not None:
            item_test = test_data[test_data['item_id'] == item_id].sort_values('date')
            if not item_test.empty:
                plt.plot(item_test['date'], item_test['cnt'], label='Тестовые данные', color='green')
        
        # Прогноз
        plt.plot(item_forecast['date'], item_forecast['cnt'], label='Прогноз', color='red', linestyle='--')
        
        # Добавление вертикальной линии, разделяющей историю и прогноз
        last_date = item_data['date'].max()
        plt.axvline(x=last_date, color='gray', linestyle='--')
        
        # Настройка графика
        plt.title(f'Прогноз продаж для товара {item_id}')
        plt.xlabel('Дата')
        plt.ylabel('Количество продаж')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Отображение графика
        plt.show()


def main():
    """
    Основная функция для запуска прогнозирования продаж.
    """
    # Пути к файлам данных
    sales_path = 'data/sales_data.csv'
    calendar_path = 'data/calendar_data.csv'
    prices_path = 'data/prices_data.csv'
    
    # Создание экземпляра класса
    model = SalesForecastingModel(store_id='STORE_1')
    
    # Проверка наличия сохраненных моделей
    models_path = 'models'
    try_load_models = False
    
    if os.path.exists(f"{models_path}/STORE_1_models.pkl"):
        try_load_models = True
    
    if try_load_models:
        print("Найдены сохраненные модели. Загрузка...")
        try:
            model.load_models(path=models_path)
            print("Модели успешно загружены.")
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
            try_load_models = False
    
    # Если модели не загружены, обучаем новые
    if not try_load_models:
        print("Загрузка и предобработка данных...")
        model.load_data(sales_path, calendar_path, prices_path)
        processed_data = model.preprocess_data()
        
        print("Обучение моделей...")
        model.train_models()
        
        print("Оценка качества моделей...")
        evaluation_results = model.evaluate_models()
        
        print("Сохранение моделей...")
        model.save_models(path=models_path)
    
    # Прогнозирование
    print("\nПрогнозирование продаж...")
    
    # Прогноз на неделю
    print("\nПрогноз на неделю:")
    weekly_forecast = model.forecast_weekly()
    
    # Прогноз на месяц
    print("\nПрогноз на месяц:")
    monthly_forecast = model.forecast_monthly()
    
    # Прогноз на квартал
    print("\nПрогноз на квартал:")
    quarterly_forecast = model.forecast_quarterly()
    
    # Создание директории для сохранения прогнозов
    os.makedirs('forecasts', exist_ok=True)
    
    # Сохранение прогнозов
    weekly_forecast.to_csv('forecasts/weekly_forecast.csv', index=False)
    monthly_forecast.to_csv('forecasts/monthly_forecast.csv', index=False)
    quarterly_forecast.to_csv('forecasts/quarterly_forecast.csv', index=False)
    
    print("\nПрогнозы сохранены в директории 'forecasts'")
    
    # Визуализация прогнозов для первых 3 товаров
    print("\nВизуализация прогнозов:")
    for item in model.items[:3]:
        print(f"\nПрогноз для товара {item}:")
        model.plot_forecast(item, monthly_forecast)


if __name__ == "__main__":
    main()







# # Загрузка данных
# sales = pd.read_csv('data/sales_data.csv')  
# calendar = pd.read_csv('data/calendar_data.csv')  
# prices = pd.read_csv('data/prices_data.csv') 


# class Store1Inference:
#     def __init__(self, sales_data, calendar_data, prices_data):
#         self.sales_data = sales_data
#         self.calendar_data = calendar_data
#         self.prices_data = prices_data

#     # Функция для предобработки данных
#     def preprocess_data(self, sales_data, calendar_data, prices_data, store_id='STORE_1'):
#         # Код функции preprocess_data (как в предыдущем ответе)s
#         # ...

#     # Функции для обучения моделей
#     def train_arma_models(data, items, max_p=3, max_q=3):
#         # Код функции train_arma_models (как в предыдущем ответе)
#         # ...

#     def train_linear_trend_models(data, items):
#         # Код функции train_linear_trend_models (как в предыдущем ответе)
#         # ...

#     def train_dynamic_regression_models(data, items):
#         # Код функции train_dynamic_regression_models (как в предыдущем ответе)
#         # ...

#     def train_ets_models(data, items):
#         # Код функции train_ets_models (как в предыдущем ответе)
#         # ...

#     def train_garch_models(data, items):
#         # Код функции train_garch_models (как в предыдущем ответе)
#         # ...

#     # Функция для оценки моделей
#     def evaluate_models(models, data, items, test_size=30):
#         # Код функции evaluate_models (как в предыдущем ответе)
#         # ...

#     # Функция для визуализации прогнозов
#     def visualize_forecasts(models, data, items, test_size=30, best_model_only=False):
#         # Код функции visualize_forecasts (как в предыдущем ответе)
#         # ...

# # Основной код для выполнения анализа
# # 1. Проводим EDA
# print("Проведение разведочного анализа данных...")
# eda_results, eda_summary = perform_eda(sales, calendar, prices)

# # 2. Предобработка данных
# print("Предобработка данных...")
# processed_data = preprocess_data(sales, calendar, prices)

# # 3. Получаем список товаров для STORE_1
# items = sales[sales['store_id'] == 'STORE_1']['item_id'].unique()
# print(f"Найдено {len(items)} товаров для анализа: {items}")

# # 4. Обучаем модели
# print("Обучение моделей ARMA...")
# arma_models = train_arma_models(processed_data, items)

# print("Обучение моделей линейного тренда...")
# linear_trend_models = train_linear_trend_models(processed_data, items)

# print("Обучение моделей динамической регрессии...")
# dynamic_regression_models = train_dynamic_regression_models(processed_data, items)

# print("Обучение моделей ETS...")
# ets_models = train_ets_models(processed_data, items)

# print("Обучение моделей GARCH...")
# garch_models = train_garch_models(processed_data, items)

# print("Обучение моделей Prophet...")
# prophet_models = train_prophet_models(processed_data, items)

# # 5. Собираем все модели в один словарь
# all_models = {
#     'arma': arma_models,
#     'linear_trend': linear_trend_models,
#     'dynamic_regression': dynamic_regression_models,
#     'ets': ets_models,
#     'garch': garch_models,
#     'prophet': prophet_models
# }

# # 6. Оцениваем модели
# print("Оценка качества моделей...")
# evaluation_results = evaluate_models(all_models, processed_data, items, test_size=30)

# # 7. Выводим сводную таблицу
# print("Сводная таблица результатов по всем моделям:")
# summary_table = evaluation_results.pivot_table(
#     index='model', 
#     values=['MAE', 'RMSE', 'MAPE', 'SMAPE', 'MASE'], 
#     aggfunc='mean'
# ).sort_values('MASE')
# print(summary_table)

# # 8. Находим лучшую модель для каждого товара
# best_models = evaluation_results.loc[evaluation_results.groupby('item_id')['MASE'].idxmin()]
# print("\nЛучшие модели для каждого товара:")
# print(best_models[['item_id', 'model', 'MASE', 'SMAPE']])

# # 9. Подсчитываем, сколько раз каждая модель оказалась лучшей
# model_counts = best_models['model'].value_counts()
# print("\nКоличество товаров, для которых каждая модель оказалась лучшей:")
# print(model_counts)

# # 10. Визуализируем прогнозы лучших моделей
# print("\nВизуализация прогнозов лучших моделей...")
# visualize_forecasts(all_models, processed_data, items, test_size=30, best_model_only=True)

# # 11. Создаем график распределения метрик по моделям
# plt.figure(figsize=(12, 8))
# sns.boxplot(x='model', y='MASE', data=evaluation_results)
# plt.title('Распределение MASE по моделям')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 12. Создаем тепловую карту для сравнения моделей
# pivot_metrics = evaluation_results.pivot_table(
#     index='item_id', 
#     columns='model', 
#     values='MASE'
# )

# plt.figure(figsize=(12, 10))
# sns.heatmap(pivot_metrics, annot=True, cmap='YlGnBu', fmt='.3f')
# plt.title('Тепловая карта MASE по товарам и моделям')
# plt.tight_layout()
# plt.show()
            