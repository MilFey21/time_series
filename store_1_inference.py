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
        
        calendar['date'] = pd.to_datetime(calendar['date'])
        
        data = sales.merge(calendar[['date_id', 'date', 'wm_yr_wk', 'weekday', 'month', 'year', 
                                    f'CASHBACK_{self.store_id}']], 
                          on='date_id', how='left')
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
        print("Обучение ARIMA моделей...")
        
        arima_models = {}
        
        for item in self.items:
            print(f"  Обучение ARIMA модели для товара {item}...")
            
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            train_data = item_data.iloc[:-test_size]
            
            try:
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
    
    def _train_ets_models(self, test_size: int = 30) -> Dict:
        print("Обучение ETS моделей...")
        
        ets_models = {}
        
        for item in self.items:
            print(f"  Обучение ETS модели для товара {item}...")
            
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            train_data = item_data.iloc[:-test_size]
            
            try:
                y_train = train_data['cnt'].reset_index(drop=True)
                
                # Обучение ETS модели
                model = ExponentialSmoothing(
                    y_train,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=7  # Недельная сезонность
                )
                
                ets_models[item] = model.fit()
                
                print(f"    ETS модель обучена для товара {item}.")
            except Exception as e:
                print(f"    Ошибка при обучении ETS модели для товара {item}: {e}")
        
        return ets_models
    
    def _train_linear_models(self, test_size: int = 30) -> Dict:
        print("Обучение линейных моделей...")
        
        linear_models = {}
        
        for item in self.items:
            print(f"  Обучение линейной модели для товара {item}...")
            
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            
            train_data = item_data.iloc[:-test_size]
            
            try:
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
    
    def _train_gbm_models(self, test_size: int = 30) -> Dict:
        print("Обучение моделей градиентного бустинга...")
        
        gbm_models = {}
        self.scalers['gbm'] = {}
        
        for item in self.items:
            print(f"  Обучение модели градиентного бустинга для товара {item}...")
            
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            
            train_data = item_data.iloc[:-test_size]
            
            try:
                X_train = pd.get_dummies(train_data[self.feature_columns], drop_first=True)
                y_train = train_data[self.target_column]
                
                # Удаляем строки с пропущенными значениями
                mask = ~(X_train.isna().any(axis=1) | y_train.isna())
                X_train = X_train[mask]
                y_train = y_train[mask]
                
                # Масштабирование признаков
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Сохранение скейлера
                self.scalers['gbm'][item] = scaler
                
                # Обучение модели градиентного бустинга
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                gbm_models[item] = model
                
                print(f"    Модель градиентного бустинга обучена для товара {item}.")
            except Exception as e:
                print(f"    Ошибка при обучении модели градиентного бустинга для товара {item}: {e}")
        
        return gbm_models
    
    def evaluate_models(self, test_size: int = 30) -> pd.DataFrame:
        if not self.models:
            raise ValueError("Модели не обучены. Используйте метод train_models() сначала.")
        
        if self.processed_data is None:
            raise ValueError("Данные не предобработаны. Используйте метод preprocess_data() сначала.")
        
        print("Оценка качества моделей...")
        
        results = []
        
        for item in self.items:
            print(f"  Оценка моделей для товара {item}...")
            item_data = self.processed_data[self.processed_data['item_id'] == item].sort_values('date')
            test_data = item_data.iloc[-test_size:]
            y_true = test_data[self.target_column].values
            
            # Оценка ARIMA модели
            if 'arima' in self.models and item in self.models['arima']:
                try:
                    y_pred = self.models['arima'][item].forecast(steps=test_size)
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    
                    results.append({
                        'item_id': item,
                        'model': 'arima',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"    Ошибка при оценке ARIMA модели для товара {item}: {e}")
            
            # Оценка ETS модели
            if 'ets' in self.models and item in self.models['ets']:
                try:
                    y_pred = self.models['ets'][item].forecast(test_size)
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    
                    results.append({
                        'item_id': item,
                        'model': 'ets',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"    Ошибка при оценке ETS модели для товара {item}: {e}")
            
            # Оценка линейной модели
            if 'linear' in self.models and item in self.models['linear']:
                try:
                    X_test = pd.get_dummies(test_data[self.feature_columns], drop_first=True)
                    missing_cols = set(self.models['linear'][item].feature_names_in_) - set(X_test.columns)
                    for col in missing_cols:
                        X_test[col] = 0
                    X_test = X_test[self.models['linear'][item].feature_names_in_]
                    
                    y_pred = self.models['linear'][item].predict(X_test)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    
                    results.append({
                        'item_id': item,
                        'model': 'linear',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"    Ошибка при оценке линейной модели для товара {item}: {e}")
            
            # Оценка модели градиентного бустинга
            if 'gbm' in self.models and item in self.models['gbm']:
                try:
                    X_test = pd.get_dummies(test_data[self.feature_columns], drop_first=True)
                    
                    missing_cols = set(self.models['gbm'][item].feature_names_in_) - set(X_test.columns)
                    for col in missing_cols:
                        X_test[col] = 0
                    X_test = X_test[self.models['gbm'][item].feature_names_in_]
                    X_test_scaled = self.scalers['gbm'][item].transform(X_test)
                    
                    y_pred = self.models['gbm'][item].predict(X_test_scaled)
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    
                    results.append({
                        'item_id': item,
                        'model': 'gbm',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"    Ошибка при оценке модели градиентного бустинга для товара {item}: {e}")
        
        # Создание DataFrame с результатами
        results_df = pd.DataFrame(results)
        self.metrics = results_df
        
        # Определение лучшей модели для каждого товара
        best_models = results_df.loc[results_df.groupby('item_id')['MAE'].idxmin()]
        self.best_models = {row['item_id']: row['model'] for _, row in best_models.iterrows()}
        
        print("Оценка качества моделей завершена.")
        
        return results_df
    
    def save_models(self, path: str = 'models') -> None:
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
        if not self.models:
            raise ValueError("Модели не обучены. Используйте метод train_models() сначала.")
        
        if self.processed_data is None:
            raise ValueError("Данные не предобработаны. Используйте метод preprocess_data() сначала.")
        
        print(f"Прогнозирование на {horizon} дней вперед...")
        
        all_forecasts = []
        
        last_date = self.processed_data['date'].max()
        
        for item in self.items:
            print(f"  Прогнозирование для товара {item}...")
            
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
        try:
            # Создание дат для прогноза
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
            
            # Прогнозирование в зависимости от типа модели
            if model_type == 'arima':
                # Прогноз ARIMA модели
                forecast_values = self.models[model_type][item].forecast(steps=horizon)
                
                # Проверка на наличие NA или inf значений
                if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)):
                    print(f"    Предупреждение: ARIMA модель для товара {item} вернула NA или inf значения. Заменяем их на 0.")
                    forecast_values = np.nan_to_num(forecast_values, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Округление и преобразование в целые числа (продажи не могут быть дробными)
                forecast_values = np.round(forecast_values).astype(int)
                
                # Отрицательные значения заменяем на 0
                forecast_values = np.maximum(forecast_values, 0)
                
            elif model_type == 'ets':
                # Прогноз ETS модели
                forecast_values = self.models[model_type][item].forecast(horizon)
                
                # Проверка на наличие NA или inf значений
                if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)):
                    print(f"    Предупреждение: ETS модель для товара {item} вернула NA или inf значения. Заменяем их на 0.")
                    forecast_values = np.nan_to_num(forecast_values, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Округление и преобразование в целые числа
                forecast_values = np.round(forecast_values).astype(int)
                
                # Отрицательные значения заменяем на 0
                forecast_values = np.maximum(forecast_values, 0)
                
            elif model_type in ['linear', 'gbm']:
                # Создание признаков для прогноза
                forecast_features = self._create_forecast_features(item, item_data, forecast_dates)
                
                if forecast_features is None:
                    print(f"    Ошибка при создании признаков для прогноза товара {item}")
                    return None
                
                # Преобразование категориальных признаков
                X_forecast = pd.get_dummies(forecast_features, drop_first=True)
                
                # Убедимся, что у нас те же столбцы, что и при обучении
                if hasattr(self.models[model_type][item], 'feature_names_in_'):
                    missing_cols = set(self.models[model_type][item].feature_names_in_) - set(X_forecast.columns)
                    for col in missing_cols:
                        X_forecast[col] = 0
                    X_forecast = X_forecast[self.models[model_type][item].feature_names_in_]
                
                # Масштабирование признаков для GBM
                if model_type == 'gbm' and item in self.scalers.get('gbm', {}):
                    X_forecast_scaled = self.scalers['gbm'][item].transform(X_forecast)
                    forecast_values = self.models[model_type][item].predict(X_forecast_scaled)
                else:
                    forecast_values = self.models[model_type][item].predict(X_forecast)
                
                # Проверка на наличие NA или inf значений
                if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)):
                    print(f"    Предупреждение: Модель {model_type} для товара {item} вернула NA или inf значения. Заменяем их на 0.")
                    forecast_values = np.nan_to_num(forecast_values, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Округление и преобразование в целые числа
                forecast_values = np.round(forecast_values).astype(int)
                
                # Отрицательные значения заменяем на 0
                forecast_values = np.maximum(forecast_values, 0)
            else:
                print(f"    Неизвестный тип модели: {model_type}")
                return None
            
            # Создание DataFrame с прогнозами
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'cnt': forecast_values,
                'item_id': item,
                'model': model_type
            })
            
            return forecast_df
        
        except Exception as e:
            print(f"    Ошибка при прогнозировании для товара {item} с моделью {model_type}: {e}")
            return None
    
    def _create_forecast_features(self, item: str, historical_data: pd.DataFrame, 
                                 forecast_dates: List[datetime]) -> Optional[pd.DataFrame]:
        try:
            # Создание DataFrame с датами для прогноза
            forecast_df = pd.DataFrame({'date': forecast_dates})
            
            # Добавление признаков на основе даты
            forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
            forecast_df['day_of_month'] = forecast_df['date'].dt.day
            forecast_df['week_of_year'] = forecast_df['date'].dt.isocalendar().week
            forecast_df['month'] = forecast_df['date'].dt.month
            forecast_df['year'] = forecast_df['date'].dt.year
            forecast_df['is_weekend'] = forecast_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Добавление последней известной цены
            last_price = historical_data['sell_price'].iloc[-1]
            forecast_df['sell_price'] = last_price
            
            # Добавление CASHBACK (если есть)
            if f'CASHBACK_{self.store_id}' in historical_data.columns:
                # Используем среднее значение CASHBACK
                cashback_mean = historical_data[f'CASHBACK_{self.store_id}'].mean()
                forecast_df[f'CASHBACK_{self.store_id}'] = cashback_mean
            
            # Добавление лаговых признаков
            for lag in [1, 7, 14, 28]:
                lag_values = []
                for date in forecast_dates:
                    # Находим дату, от которой нужно взять лаг
                    lag_date = date - timedelta(days=lag)
                    # Находим ближайшую дату в исторических данных
                    closest_data = historical_data[historical_data['date'] <= lag_date].sort_values('date').tail(1)
                    if not closest_data.empty:
                        lag_values.append(closest_data['cnt'].values[0])
                    else:
                        # Если нет данных, используем 0
                        lag_values.append(0)
                forecast_df[f'lag_{lag}'] = lag_values
            
            # Добавление скользящих средних
            for window in [7, 14, 28]:
                # Используем последние значения скользящих средних
                rolling_mean = historical_data[f'rolling_mean_{window}'].iloc[-1]
                forecast_df[f'rolling_mean_{window}'] = rolling_mean
                
                # Используем последние значения скользящих стандартных отклонений
                rolling_std = historical_data[f'rolling_std_{window}'].iloc[-1]
                forecast_df[f'rolling_std_{window}'] = rolling_std
            
            # Добавление праздничных признаков (если есть)
            holiday_cols = [col for col in historical_data.columns if 'event' in col]
            for col in holiday_cols:
                # Для простоты используем 0 (нет праздника)
                forecast_df[col] = 0
            
            return forecast_df
        
        except Exception as e:
            print(f"    Ошибка при создании признаков для прогноза товара {item}: {e}")
            return None
    
    def forecast_weekly(self, use_best_model: bool = True) -> pd.DataFrame:
        return self.forecast(horizon=7, use_best_model=use_best_model)
    
    def forecast_monthly(self, use_best_model: bool = True) -> pd.DataFrame:
        return self.forecast(horizon=30, use_best_model=use_best_model)
    
    def forecast_quarterly(self, use_best_model: bool = True) -> pd.DataFrame:
        return self.forecast(horizon=90, use_best_model=use_best_model)
    
    def plot_forecast(self, item_id: str, forecast_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None, 
                 days_to_show: int = 30) -> None:
        if self.processed_data is None:
            raise ValueError("Данные не предобработаны. Используйте метод preprocess_data() сначала.")
        
        item_data = self.processed_data[self.processed_data['item_id'] == item_id].sort_values('date')
        
        # Проверяем, есть ли столбец 'item_id' в прогнозных данных
        if 'item_id' in forecast_data.columns:
            item_forecast = forecast_data[forecast_data['item_id'] == item_id].sort_values('date')
        else:
            item_forecast = forecast_data.sort_values('date')
            print(f"Предупреждение: в прогнозных данных отсутствует столбец 'item_id'. Используются все данные.")
        
        if item_data.empty:
            print(f"Нет данных для товара {item_id}")
            return
        
        if item_forecast.empty:
            print(f"Нет прогнозов для товара {item_id}")
            return
        
        item_data = item_data.tail(days_to_show)
        
        # Создание графика
        plt.figure(figsize=(12, 6))
        
        # Исторические данные
        plt.plot(item_data['date'], item_data['cnt'], label='Исторические продажи', color='blue')
        
        # Тестовые данные (если есть)
        if test_data is not None:
            if 'item_id' in test_data.columns:
                item_test = test_data[test_data['item_id'] == item_id].sort_values('date')
            else:
                item_test = test_data.sort_values('date')
            
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
    # Пути к файлам данных
    sales_path = 'data/shop_sales.csv'
    calendar_path = 'data/shop_sales_dates.csv'
    prices_path = 'data/shop_sales_prices.csv'
    
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
            
            # Загружаем и предобрабатываем данные даже при загрузке моделей
            print("Загрузка и предобработка данных...")
            model.load_data(sales_path, calendar_path, prices_path)
            model.preprocess_data()
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
            try_load_models = False
    
    # Если модели не загружены, обучаем новые
    if not try_load_models:
        print("Загрузка и предобработка данных...")
        model.load_data(sales_path, calendar_path, prices_path)
        model.preprocess_data()  
        
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

