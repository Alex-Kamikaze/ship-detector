from typing import Annotated
from ultralytics import YOLO
import typer
from rich import print as prettyprint


def recognize(
    image: Annotated[
        str,
        typer.Argument(
            help="Путь до изображения, на котором необходимо распознать корабли"
        ),
    ],
    save_dir: Annotated[
        str, typer.Argument(help="Путь для сохранения готового изображения")
    ],
):
    """
    Распознавание кораблей на спутниковых изображениях
    """
    model = YOLO("yolo11n.pt")
    try:
        result = model(source=image, show=True, conf=0.1, verbose=False)
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
