import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import sys


def calculate_variance_contributions(X):
    """
    Рассчитывает вклад каждого признака в общую дисперсию
    
    Входные данные:
        X - матрица признаков
    
    Выходные данные:
        contributions - доли вклада каждого признака в общую дисперсию
        feature_variances - дисперсии каждого признака
    """
    try:
        # Проверка входных данных
        if X is None or len(X) == 0:
            raise ValueError("Матрица признаков пуста или None")
        
        X = np.array(X)
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Матрица признаков содержит NaN или бесконечные значения")
        
        # Посчитать дисперсию каждого признака
        feature_variances = np.var(X, axis=0)
        
        # Проверка на нулевую дисперсию
        if np.any(feature_variances < 0):
            raise ValueError("Обнаружена отрицательная дисперсия")
        
        total_variance = np.sum(feature_variances)
        
        if total_variance == 0:
            raise ValueError("Общая дисперсия равна нулю - все признаки постоянны")
        
        # Разделить на общую сумму
        contributions = feature_variances / total_variance
        return contributions, feature_variances
        
    except Exception as e:
        raise ValueError(f"Ошибка при расчете дисперсии: {str(e)}")


def plot_contributions(contributions, feature_names):
    """
    Строит bar plot долей вклада признаков
    
    Входные данные:
        contributions - доли вклада каждого признака
        feature_names - названия признаков
    """
    try:
        # Проверка входных данных
        if len(contributions) != len(feature_names):
            raise ValueError("Количество признаков не совпадает с количеством названий")
        
        if len(contributions) == 0:
            raise ValueError("Нет данных для построения графика")
        
        if not all(0 <= x <= 1 for x in contributions):
            raise ValueError("Доли вклада должны быть в диапазоне [0, 1]")
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(feature_names, contributions, color='skyblue', alpha=0.7)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, contributions):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title('Вклад признаков в общую дисперсию', fontsize=14)
        plt.xlabel('Признаки', fontsize=12)
        plt.ylabel('Доля в общей дисперсии', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
    except Exception as e:
        raise ValueError(f"Ошибка при построении графика: {str(e)}")


def show_plot_on_the_screen(contributions, feature_names):
    """
    Выводит bar plot долей вклада признаков на экран
    """
    try:
        plot_contributions(contributions, feature_names)
        plt.show()
    except Exception as e:
        raise ValueError(f"Ошибка при отображении графика: {str(e)}")


def save_plot_to_file(contributions, feature_names, filename='bar_plot.png'):
    """
    Сохраняет график в файл
    
    Входные данные:
        contributions - доли вклада каждого признака
        feature_names - названия признаков
        filename - имя файла для сохранения
    """
    try:
        # Проверка имени файла
        if not filename or not isinstance(filename, str):
            raise ValueError("Имя файла должно быть непустой строкой")
        
        # Проверка расширения файла
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            print(f"Предупреждение: нестандартное расширение файла: {filename}")
        
        plot_contributions(contributions, feature_names)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранен")
    
    except Exception as e:
        raise IOError(f"Ошибка при сохранении графика: {str(e)}")


def get_top_features(contributions, feature_names, top_k=3):
    """
    Возвращает топ-k признаков по вкладу в дисперсию
    
    Входные данные:
        contributions - доли вклада каждого признака
        feature_names - названия признаков
        top_k - количество топовых признаков для вывода
    
    Выходные данные:
        top_features - список кортежей (название_признака, доля_вклада)
    """
    try:
        # Проверка входных данных
        if len(contributions) != len(feature_names):
            raise ValueError("Количество признаков не совпадает с количеством названий")
        
        if top_k <= 0:
            raise ValueError("top_k должен быть положительным числом")
        
        if top_k > len(contributions):
            raise ValueError(f"top_k={top_k} больше количества признаков ({len(contributions)})")
        
        # Сортируем признаки по убыванию вклада
        sorted_indices = np.argsort(contributions)[::-1]
        
        top_features = []
        for i in range(min(top_k, len(contributions))):
            idx = sorted_indices[i]
            top_features.append((feature_names[idx], contributions[idx]))
        
        return top_features
        
    except Exception as e:
        raise ValueError(f"Ошибка при получении топ-признаков: {str(e)}")


def run_analysis(top_k=3):
    """
    Основная функция для анализа вклада признаков в дисперсию
    
    Входные данные:
        top_k - количество топовых признаков для вывода
    """
    try:
        # Загрузка датасета
        iris = load_iris()
        X = iris.data
        feature_names = iris.feature_names
        
        print("Анализ вклада признаков в дисперсию")
        print("Датасет: Iris")
        
        # Расчет вкладов
        contributions, variances = calculate_variance_contributions(X)
        
        # Вывод информации о дисперсиях
        print("\nДисперсии признаков:")
        for i, (name, var) in enumerate(zip(feature_names, variances)):
            print(f"{i+1}. {name}: {var:.4f}")
        
        print(f"\nОбщая дисперсия: {np.sum(variances):.4f}")
        
        # Вывод долей вклада
        print("\nДоли вклада в общую дисперсию:")
        for i, (name, contrib) in enumerate(zip(feature_names, contributions)):
            print(f"{i+1}. {name}: {contrib:.4f} ({contrib*100:.2f}%)")
        
        # Получение топ-k признаков
        top_features = get_top_features(contributions, feature_names, top_k=top_k)
        
        print(f"\nТОП-{top_k} признака по вкладу в дисперсию:\n")
        
        for i, (name, contrib) in enumerate(top_features):
            print(f"{i+1}. {name}: {contrib:.4f} ({contrib*100:.2f}%)")
            
        return contributions, feature_names
        
    except Exception as e:
        raise ValueError(f"Ошибка при выполнении анализа: {str(e)}")


def parse_arguments():
    """
    Парсинг аргументов командной строки
    """
    parser = argparse.ArgumentParser(
        description='Анализ вклада признаков в дисперсию для датасета Iris'
    )

    parser.add_argument(
        '--no-plot',
        '-np',
        action='store_true',
        help='Не показывать график'
    )
    
    parser.add_argument(
        '--save-plot',
        '-sp',
        action='store_true',
        help='Сохранить график в файл'
    )
    
    parser.add_argument(
        '--output-file',
        '-o',
        type=str,
        default='bar_plot.png',
        help='Имя файла для сохранения графика (по умолчанию: bar_plot.png)'
    )
    
    parser.add_argument(
        '--top-k',
        '-k',
        type=int,
        default=3,
        help='Количество топовых признаков для вывода (по умолчанию: 3)'
    )
    
    return parser.parse_args()


def run_variance_analysis_program():
    """
    Основная функция запуска программы из командной строки
    
    Возвращает:
        int: код возврата (0 - успех, 1 - ошибка)
    """
    try:
        # Парсинг аргументов командной строки
        args = parse_arguments()
        
        # Вывод параметров
        print("Параметры:")
        print(f"  no_plot = {args.no_plot}")
        print(f"  save_plot = {args.save_plot}")
        print(f"  top_k = {args.top_k}")
        print(f"  output_file = {args.output_file}\n")
        
        # Запуск анализа
        contributions, feature_names = run_analysis(top_k=args.top_k)
        
        # Обработка аргументов для графиков
        if not args.no_plot:
            show_plot_on_the_screen(contributions, feature_names)
        
        if args.save_plot:
            try:
                save_plot_to_file(contributions, feature_names, args.output_file)
            except Exception as e:
                print(f"Ошибка при сохранении графика: {e}")
                return 1
        return 0
    
    except Exception as e:
        print(f"\nОшибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_variance_analysis_program())