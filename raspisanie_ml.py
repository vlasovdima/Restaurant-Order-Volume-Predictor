import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import logging

# Настраиваем логирование, чтобы вывод в терминале выглядел аккуратно и профессионально
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class TrafficDataPipeline:
    """Класс для подготовки данных (наш воображаемый датасет из БД ресторана)."""
    
    @staticmethod
    def generate_data(days: int = 180) -> pd.DataFrame:
        """Собираем фейковые, но реалистичные данные по количеству чеков."""
        logger.info(f" Выгружаем историю заказов за {days} дней...")
        np.random.seed(42) # Фиксируем рандом, чтобы данные не менялись при перезапусках
        
        # Создаем временной ряд (кафе работает с 8:00 до 23:00)
        dates = pd.date_range(start='2025-01-01', periods=days * 24, freq='h')
        df = pd.DataFrame({'datetime': dates})
        df = df[(df['datetime'].dt.hour >= 8) & (df['datetime'].dt.hour <= 23)].copy()
        
        # Вытаскиваем полезные признаки (фичи) из даты
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Погода: 0 - ливень/метель, 1 - нормальная, 2 - отличная
        df['weather'] = np.random.choice([0, 1, 2], size=len(df), p=[0.2, 0.6, 0.2])
        
        # Симулируем реальную нагрузку: базовые 50 чеков в час + пики в обед и вечером
        base_orders = 50
        
        def calc_multiplier(row):
            mult = 1.0
            # Час-пик (обед и ужин)
            if row['hour'] in [12, 13, 18, 19]: mult *= 2.5
            # Выходные дни (наплыв людей)
            if row['is_weekend']: mult *= 1.5
            # Плохая погода отпугивает гостей
            if row['weather'] == 0: mult *= 0.8
            return mult

        df['multiplier'] = df.apply(calc_multiplier, axis=1)
        
        # Добавляем немного случайного шума (жизнь не идеальна, трафик всегда прыгает)
        noise = np.random.normal(0, 10, size=len(df))
        df['target_orders'] = (base_orders * df['multiplier'] + noise).astype(int)
        
        # Меньше 10 чеков в час почти не бывает даже в самый тухлый день - ставим лимит
        df['target_orders'] = df['target_orders'].clip(lower=10)
        
        logger.info(f" Готово! Собрано {len(df)} строк данных.")
        return df

class TrafficPredictor:
    """Модель, которая учится угадывать количество чеков в час."""
    
    def __init__(self):
        # Случайный лес из 100 деревьев - классика для табличных данных
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['hour', 'day_of_week', 'is_weekend', 'weather']
        self.is_trained = False

    def train_and_evaluate(self, df: pd.DataFrame):
        """Учим модель и сразу проверяем, насколько она адекватно предсказывает."""
        logger.info(" Начинаем обучение модели...")
        
        # ВАЖНО: Для данных со временем (таймсерии) нельзя мешать строки случайно.
        # Мы обучаемся на прошлом (первые 80% данных) и тестируем на будущем (последние 20%).
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train, y_train = train_df[self.features], train_df['target_orders']
        X_test, y_test = test_df[self.features], test_df['target_orders']
        
        # Запускаем обучение
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Проверяем на тестовой выборке (тех самых 20% из "будущего")
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        logger.info(" Обучение завершено!")
        logger.info(f" Средняя ошибка модели (MAE): промахиваемся примерно на {mae:.1f} чеков/час.")

    def predict(self, hour: int, day_of_week: int, is_weekend: int, weather: int) -> int:
        """Делаем точечный прогноз для конкретных условий."""
        if not self.is_trained:
            raise RuntimeError("Сначала нужно обучить модель (вызвать train_and_evaluate)!")
            
        input_data = pd.DataFrame([[hour, day_of_week, is_weekend, weather]], columns=self.features)
        
        # Предсказываем и возвращаем целое число чеков
        return int(self.model.predict(input_data)[0])

def main():
    print("\n=== Запуск ML-системы прогнозирования загрузки ===\n")
    
    # 1. Достаем данные 
    pipeline = TrafficDataPipeline()
    data = pipeline.generate_data(days=180)
    
    # 2. Учим алгоритм
    predictor = TrafficPredictor()
    predictor.train_and_evaluate(data)
    
   # 3. Пробуем в деле (Инференс со случайными условиями)
    print("\n--- Прогноз со случайным временем и погодой для каждого дня ---")
    
    days_of_week = [
        "Понедельник", "Вторник", "Среда", "Четверг", 
        "Пятница", "Суббота", "Воскресенье"
    ]
    
    # Словарик для красивого отображения погоды в терминале
    weather_labels = {
        0: "🌧️ Ливень",
        1: "☁️ Норм",
        2: "☀️ Супер"
    }
    
    for day_idx, day_name in enumerate(days_of_week):
        is_weekend = 1 if day_idx in [5, 6] else 0
        
        # Рандомизируем час (от 8 до 23 включительно) и погоду (0, 1 или 2)
        rand_hour = np.random.randint(8, 24)
        rand_weather = np.random.choice([0, 1, 2])
        
        # Делаем предсказание с нашими случайными вводными
        pred = predictor.predict(
            hour=rand_hour, 
            day_of_week=day_idx, 
            is_weekend=is_weekend, 
            weather=rand_weather
        )
        
        # Оформляем вывод
        day_type = " Выходной" if is_weekend else " Будний  "
        weather_str = weather_labels[rand_weather]
        
        # f"{rand_hour:02d}:00" сделает так, чтобы 8 часов отображалось как 08:00
        print(f"{day_type} | {day_name.ljust(11)} | 🕒 {rand_hour:02d}:00 | {weather_str.ljust(10)} -> Ждем ~{pred} заказов")

if __name__ == "__main__":
    main()
