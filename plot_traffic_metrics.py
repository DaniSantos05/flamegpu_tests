"""
Generación de gráficas para el modelo de tráfico V0.5.

Lee traffic_metrics_v05.csv y genera varias gráficas PNG.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


CSV_FILE = Path("traffic_metrics_v05.csv")
OUTPUT_DIR = Path("graficas_v05")


def read_metrics(csv_file):
    steps = []
    stopped_cars = []
    stopped_by_light = []
    stopped_by_car = []
    queue_lengths = []
    crossings_this_step = []
    total_crossings = []
    total_waiting_time = []
    average_waiting_time = []

    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            steps.append(int(row["paso"]))
            stopped_cars.append(int(row["coches_parados"]))
            stopped_by_light.append(int(row["parados_por_semaforo"]))
            stopped_by_car.append(int(row["parados_por_coche_delante"]))
            queue_lengths.append(int(row["cola_aproximada"]))
            crossings_this_step.append(int(row["coches_cruzan_este_paso"]))
            total_crossings.append(int(row["cruces_totales"]))
            total_waiting_time.append(int(row["espera_total"]))
            average_waiting_time.append(float(row["espera_media"].replace(",", ".")))

    return {
        "steps": steps,
        "stopped_cars": stopped_cars,
        "stopped_by_light": stopped_by_light,
        "stopped_by_car": stopped_by_car,
        "queue_lengths": queue_lengths,
        "crossings_this_step": crossings_this_step,
        "total_crossings": total_crossings,
        "total_waiting_time": total_waiting_time,
        "average_waiting_time": average_waiting_time,
    }


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
    if not CSV_FILE.exists():
        print(f"No se ha encontrado el archivo {CSV_FILE}")
        print("Ejecuta primero traffic_model_v05.py para generar el CSV.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    data = read_metrics(CSV_FILE)
    steps = data["steps"]

    save_line_plot(
        steps,
        data["average_waiting_time"],
        "Evolución de la espera media",
        "Paso de simulación",
        "Espera media por coche",
        OUTPUT_DIR / "espera_media.png",
    )

    save_line_plot(
        steps,
        data["queue_lengths"],
        "Evolución de la cola aproximada",
        "Paso de simulación",
        "Coches en cola",
        OUTPUT_DIR / "cola_aproximada.png",
    )

    save_line_plot(
        steps,
        data["total_crossings"],
        "Cruces totales del semáforo",
        "Paso de simulación",
        "Número acumulado de cruces",
        OUTPUT_DIR / "cruces_totales.png",
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
        OUTPUT_DIR / "motivos_parada.png",
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
        OUTPUT_DIR / "parados_y_cola.png",
    )

    print("Gráficas generadas correctamente en la carpeta:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()