import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys


def validate_parameters(min_line_length, max_line_gap, canny_low, canny_high):
    """
    Проверка корректности параметров.
    """
    if min_line_length <= 0:
        raise ValueError("min_line_length должен быть положительным числом")
    if max_line_gap < 0:
        raise ValueError("max_line_gap не может быть отрицательным")
    if canny_low <= 0 or canny_high <= 0:
        raise ValueError("Пороги Кэнни должны быть положительными")
    if canny_low >= canny_high:
        raise ValueError("Нижний порог Кэнни должен быть меньше верхнего")


def load_image(image_path):
    """
    Загрузка изображения с проверкой существования файла.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не существует")
    
    if not os.path.isfile(image_path):
        raise ValueError(f"{image_path} не является файлом")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение из {image_path}")
    
    return image


def preprocess_image(image, canny_low=50, canny_high=150):
    """
    Предобработка изображения: преобразование в серый и применение Canny.
    """
    # Преобразование в серый
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применение детектора Canny
    edges = cv2.Canny(gray, canny_low, canny_high)
    
    return edges


def detect_lines(edges, min_line_length=50, max_line_gap=10):
    """
    Обнаружение линий с помощью преобразования Хафа.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    return lines


def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    """
    Рисование обнаруженных линий на изображении.
    """
    result_image = image.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), color, thickness)
    
    return result_image


def visualize_results(original_image, edges, result_image, num_lines):
    """
    Визуализация результатов обработки.
    """
    plt.figure(figsize=(15, 5))
    
    # Исходное изображение
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Исходное изображение')
    plt.axis('off')
    
    # Границы после Кэнни
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Границы (детектор Кэнни)')
    plt.axis('off')
    
    # Результат с линиями
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Обнаружено линий: {num_lines}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def detect_lines_complete(image_path, min_line_length=50, max_line_gap=10, 
                         canny_low=50, canny_high=150, output_path=None):
    """
    Полный процесс обработки изображения: загрузка, обнаружение линий, визуализация.
    """
    # Валидация параметров
    validate_parameters(min_line_length, max_line_gap, canny_low, canny_high)
    
    # Загрузка изображения
    image = load_image(image_path)
    
    # Предобработка
    edges = preprocess_image(image, canny_low, canny_high)
    
    # Обнаружение линий
    lines = detect_lines(edges, min_line_length, max_line_gap)
    
    # Подсчет количества линий
    num_lines = 0 if lines is None else len(lines)
    
    # Рисование линий
    if lines is not None:
        result_image = draw_lines(image, lines)
    else:
        result_image = image.copy()
    
    # Сохранение результата
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Результат сохранен в: {output_path}")
    
    # Визуализация результатов
    visualize_results(image, edges, result_image, num_lines)
    
    return result_image, num_lines, image, edges


def parse_arguments():
    """
    Парсинг аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description='Обнаружение линий на изображении с помощью преобразования Хафа'
    )
    
    # Обязательный аргумент - путь к изображению
    parser.add_argument(
        'image_path',
        help='Путь к входному изображению'
    )

    # Опциональные аргументы с значениями по умолчанию
    parser.add_argument(
        '--min-line-length',
        '-minl',
        type=int, 
        default=50,
        help='Минимальная длина линии для обнаружения (по умолчанию: 50)'
    )
    
    parser.add_argument(
        '--max-line-gap',
        '-maxl',
        type=int, 
        default=10,
        help='Максимальный разрыв между линиями для их соединения (по умолчанию: 10)'
    )
    
    parser.add_argument(
        '--canny-low',
        '-cl',
        type=int, 
        default=50,
        help='Нижний порог для детектора Кэнни (по умолчанию: 50)'
    )
    
    parser.add_argument(
        '--canny-high',
        '-ch',
        type=int, 
        default=150,
        help='Верхний порог для детектора Кэнни (по умолчанию: 150)'
    )
    
    parser.add_argument(
        '--output-path',
        '-out',
        help='Путь для сохранения результата'
    )
    
    return parser


def detect_lines_from_args():
    """
    Основная функция для демонстрации работы детектора линий.
    """
    parser = parse_arguments()

    try:
        # Парсинг аргументов
        args = parser.parse_args()

        # Проверяем обязательный аргумент
        if not args.image_path:
            print("Ошибка: не указан обязательный аргумент image_path")
            print("Используйте --help для просмотра справки")
            return 1
        
        print(f"Параметры:")
        print(f"  min_line_length = {args.min_line_length}")
        print(f"  max_line_gap = {args.max_line_gap}")
        print(f"  canny_low = {args.canny_low}")
        print(f"  canny_high = {args.canny_high}")
        print(f"  output_path = {args.output_path}")
        
        # Обработка изображения
        result_image, num_lines, original_image, edges = detect_lines_complete(
            image_path=args.image_path,
            min_line_length=args.min_line_length,
            max_line_gap=args.max_line_gap,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            output_path=args.output_path
        )
        
        print(f"Обнаружено линий: {num_lines}")
        
        # Проверка метрики
        if num_lines > 2:
            print("Метрика выполнена: найдено более 2 линий")
        else:
            print("Метрика не выполнена: найдено 2 или менее линий")

    except SystemExit:
        # Перехватываем SystemExit от argparse
        print("Ошибка в аргументах командной строки")
        print("Используйте --help для просмотра справки")
        return 1
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(detect_lines_from_args())