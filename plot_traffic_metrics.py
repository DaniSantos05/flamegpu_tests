"""
Generación de gráficas para el modelo de tráfico.

Por defecto lee traffic_metrics.csv y guarda cinco gráficas en graficas/.
Permite elegir otro CSV y otra carpeta mediante argumentos.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


CSV_FILE = Path(__file__).with_name("traffic_metrics.csv")
OUTPUT_DIR = Path(__file__).with_name("graficas")


def read_metrics(csv_file):
    data = {
        "steps": [],
        "stopped_cars": [],
        "stopped_by_light": [],
        "stopped_by_car": [],
        "queue_lengths": [],
        "crossings_this_step": [],
        "total_crossings": [],
        "total_waiting_time": [],
        "average_waiting_time": [],
    }

    with csv_file.open(mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            data["steps"].append(int(row["paso"]))
            data["stopped_cars"].append(int(row["coches_parados"]))
            data["stopped_by_light"].append(int(row["parados_por_semaforo"]))
            data["stopped_by_car"].append(int(row["parados_por_coche_delante"]))
            data["queue_lengths"].append(int(row["cola_aproximada"]))
            data["crossings_this_step"].append(int(row["coches_cruzan_este_paso"]))
            data["total_crossings"].append(int(row["cruces_totales"]))
            data["total_waiting_time"].append(int(row["espera_total"]))
            data["average_waiting_time"].append(
                float(row["espera_media"].replace(",", "."))
            )

    if not data["steps"]:
        raise ValueError(
            "El CSV solo contiene la cabecera o no tiene datos. "
            "Ejecuta correctamente traffic_model.py antes de generar las gráficas."
        )

    return data


def save_line_plot(x, y, title, xlabel, ylabel, output_file):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def save_multiple_line_plot(x, series, title, xlabel, ylabel, output_file):
    plt.figure()

    for label, y in series.items():
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Genera las gráficas del modelo de tráfico.")
    parser.add_argument("--csv", type=Path, default=CSV_FILE, help="CSV que se quiere representar.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Carpeta de las gráficas.")
    args = parser.parse_args()
    csv_file = args.csv
    output_dir = args.output_dir

    if not csv_file.exists():
        raise FileNotFoundError(
            f"No se ha encontrado {csv_file}. "
            "Ejecuta primero traffic_model.py o selecciona un CSV con --csv."
        )

    data = read_metrics(csv_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = data["steps"]

    save_line_plot(
        steps,
        data["average_waiting_time"],
        "Evolución de la espera media",
        "Paso de simulación",
        "Espera media por coche",
        output_dir / "espera_media.png",
    )

    save_line_plot(
        steps,
        data["queue_lengths"],
        "Evolución de la cola aproximada",
        "Paso de simulación",
        "Coches en cola",
        output_dir / "cola_aproximada.png",
    )

    save_line_plot(
        steps,
        data["total_crossings"],
        "Cruces totales del semáforo",
        "Paso de simulación",
        "Número acumulado de cruces",
        output_dir / "cruces_totales.png",
    )

    save_multiple_line_plot(
        steps,
        {
            "Parados por semáforo": data["stopped_by_light"],
            "Parados por coche delante": data["stopped_by_car"],
        },
        "Motivo de parada de los coches",
        "Paso de simulación",
        "Número de coches",
        output_dir / "motivos_parada.png",
    )

    save_multiple_line_plot(
        steps,
        {
            "Coches parados": data["stopped_cars"],
            "Cola aproximada": data["queue_lengths"],
        },
        "Coches parados y cola aproximada",
        "Paso de simulación",
        "Número de coches",
        output_dir / "parados_y_cola.png",
    )

    print("Gráficas generadas correctamente en la carpeta:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
