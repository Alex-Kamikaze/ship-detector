from typing import Annotated

import torch
import typer
from rich import print as prettyprint
from ultralytics import YOLO


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    model = YOLO(
        "yolo11n.pt"
    )  # можно использовать yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

    model.train(
        data="dataset.yaml",  # путь к файлу с конфигурацией данных
        epochs=50,  # количество эпох
        imgsz=150,  # размер изображения
        batch=16,  # размер батча
        name="yolov8_custom",  # имя эксперимента
        device=device,  # устройство для обучения
        workers=8,  # количество рабочих процессов
        lr0=0.01,  # начальная скорость обучения
        patience=50,  # ранняя остановка
        save=True,  # сохранение лучшей модели
        save_period=10,  # сохранение каждые N эпох
        pretrained=True,  # использовать предобученные веса
        optimizer="AdamW",  # оптимизатор
        seed=42,  # seed для воспроизводимости
        verbose=True,  # подробный вывод
    )


def recognize(
    image: Annotated[
        str | None,
        typer.Argument(
            help="Путь до изображения, на котором необходимо распознать корабли"
        ),
    ] = None,
    save_dir: Annotated[
        str | None, typer.Argument(help="Путь для сохранения готового изображения")
    ] = None,
    train: Annotated[
        bool,
        typer.Option(
            help="Флаг для запуска обучения нейронки вместо распознавания", is_flag=True
        ),
    ] = False,
):
    """
    Распознавание кораблей на спутниковых изображениях
    """
    if train:
        train_model()
        return
    model = YOLO("yolov8n.pt")
    try:
        result = model(source=image, show=True, conf=0.1, verbose=False)  # ty:ignore[invalid-argument-type]
        result[0].save(save_dir)
        prettyprint(
            f"[green] Изображение успешно сохранено! Путь: {save_dir} [/green] ✅"
        )
    except Exception:
        prettyprint(
            "[bold red] Произошла ошибка при распознавании объектов! Проверьте корректность пути до изображения, и попробуйте еще раз! [/bold red] ❌"
        )


if __name__ == "__main__":
    typer.run(recognize)
