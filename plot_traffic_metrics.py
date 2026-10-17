"""Genera gráficas de tráfico y mantiene compatibilidad con los CSV históricos.

Lee traffic_metrics.csv y guarda las cinco gráficas habituales en graficas/.
Si el CSV incluye medidas físicas y entradas/salidas, añade sus gráficas.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


CSV_FILE = Path(__file__).with_name("traffic_metrics.csv")
OUTPUT_DIR = Path(__file__).with_name("graficas")

REQUIRED_COLUMNS = {
    "steps": ("paso", int),
    "stopped_cars": ("coches_parados", int),
    "stopped_by_light": ("parados_por_semaforo", int),
    "stopped_by_car": ("parados_por_coche_delante", int),
    "queue_lengths": ("cola_aproximada", int),
    "crossings_this_step": ("coches_cruzan_este_paso", int),
    "total_crossings": ("cruces_totales", int),
    "total_waiting_time": ("espera_total", float),
    "average_waiting_time": ("espera_media", float),
}

OPTIONAL_COLUMNS = {
    "minimum_distance": ("distancia_minima", float),
    "unsafe_pairs": ("pares_consecutivos_inseguros", int),
    "red_crossings": ("cruces_en_rojo", int),
    "first_car_position": ("posicion_primero_antes_linea", float),
    "time_seconds": ("tiempo_s", float),
    "spatial_queue_cars": ("coches_en_cola", int),
    "queue_length_metres": ("longitud_cola_m", float),
    "present_cars": ("coches_presentes", int),
    "minimum_free_gap_metres": ("hueco_libre_minimo_m", float),
    "requested_arrivals": ("llegadas_solicitadas", int),
    "admitted_arrivals": ("llegadas_admitidas", int),
    "pending_arrivals": ("pendientes_entrada", int),
    "exits_this_step": ("salidas_este_paso", int),
    "total_exits": ("salidas_totales", int),
    "total_introduced_cars": ("coches_introducidos_totales", int),
    "present_average_wait_seconds": ("espera_media_presentes_s", float),
    "exited_average_wait_seconds": ("espera_media_salidos_s", float),
}


def read_metrics(csv_file):
    """Lee los CSV con separador ';' y coma o punto decimal.

    Las claves históricas se conservan. Las medidas nuevas solo se incluyen
    cuando su columna está presente; así no se atribuyen segundos ni una cola
    espacial a resultados antiguos que únicamente registraban pasos y parados.
    Una celda opcional vacía conserva como None una medida no definida.
    """
    with Path(csv_file).open(mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        headers = reader.fieldnames or []
        columns = dict(REQUIRED_COLUMNS)
        columns.update({
            key: spec for key, spec in OPTIONAL_COLUMNS.items() if spec[0] in headers
        })
        data = {key: [] for key in columns}

        for row in reader:
            for key, (column, convert) in columns.items():
                try:
                    value = row[column]
                    if key in OPTIONAL_COLUMNS and value.strip() == "":
                        data[key].append(None)
                    else:
                        data[key].append(convert(value.replace(",", ".")))
                except (KeyError, AttributeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Valor ausente o inválido en la columna '{column}', "
                        f"línea {reader.line_num} de {csv_file}."
                    ) from error

    if not data["steps"]:
        raise ValueError(
            "El CSV solo contiene la cabecera o no tiene datos. "
            "Ejecuta correctamente traffic_model.py antes de generar las gráficas."
        )

    return data


def save_line_plot(x, y, title, xlabel, ylabel, output_file):
    figure, axes = plt.subplots()
    axes.plot(x, y)
    axes.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axes.grid(True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def save_multiple_line_plot(x, series, title, xlabel, ylabel, output_file):
    figure, axes = plt.subplots()
    for label, y in series.items():
        axes.plot(x, y, label=label)
    axes.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axes.legend()
    axes.grid(True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def save_entries_exits_plot(x, data, xlabel, output_file):
    """Separa llegadas/salidas acumuladas de vehículos presentes y pendientes."""
    figure, (flows, stocks) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for key, label in (
        ("requested_arrivals", "Llegadas solicitadas"),
        ("admitted_arrivals", "Llegadas admitidas"),
        ("total_exits", "Salidas"),
    ):
        flows.plot(x, data[key], label=label)
    flows.set(title="Entradas y salidas acumuladas", ylabel="Vehículos acumulados")
    for key, label in (
        ("present_cars", "Presentes en la carretera"),
        ("pending_arrivals", "Pendientes de entrada"),
    ):
        stocks.plot(x, data[key], label=label)
    stocks.set(xlabel=xlabel, ylabel="Número de vehículos")
    for axes in (flows, stocks):
        axes.legend()
        axes.grid(True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def generate_plots(data, output_dir):
    """Guarda las gráficas disponibles y devuelve sus rutas."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_time = "time_seconds" in data
    x = data["time_seconds"] if physical_time else data["steps"]
    xlabel = "Tiempo de simulación (s)" if physical_time else "Paso de simulación"
    wait_unit = "s" if physical_time else "pasos"
    spatial_queue = "spatial_queue_cars" in data
    queue = data["spatial_queue_cars"] if spatial_queue else data["queue_lengths"]
    queue_label = "Cola espacial" if spatial_queue else "Cola aproximada"
    output_files = []

    def output(name):
        path = output_dir / name
        output_files.append(path)
        return path

    save_line_plot(
        x, data["average_waiting_time"], "Evolución de la espera media", xlabel,
        f"Espera media por coche ({wait_unit})", output("espera_media.png"),
    )
    save_line_plot(
        x, queue, f"Evolución de la {queue_label.lower()}", xlabel,
        "Coches en cola", output("cola_aproximada.png"),
    )
    save_line_plot(
        x, data["total_crossings"], "Cruces totales del semáforo", xlabel,
        "Número acumulado de cruces", output("cruces_totales.png"),
    )
    save_multiple_line_plot(
        x,
        {
            "Parados por semáforo": data["stopped_by_light"],
            "Parados por coche delante": data["stopped_by_car"],
        },
        "Motivo de parada de los coches", xlabel, "Número de coches",
        output("motivos_parada.png"),
    )
    save_multiple_line_plot(
        x, {"Coches parados": data["stopped_cars"], queue_label: queue},
        f"Coches parados y {queue_label.lower()}", xlabel, "Número de coches",
        output("parados_y_cola.png"),
    )

    if "queue_length_metres" in data:
        save_line_plot(
            x, data["queue_length_metres"], "Longitud espacial de la cola", xlabel,
            "Longitud de cola (m)", output("longitud_cola.png"),
        )

    flow_columns = (
        "requested_arrivals", "admitted_arrivals", "total_exits",
        "present_cars", "pending_arrivals",
    )
    if all(key in data for key in flow_columns):
        save_entries_exits_plot(x, data, xlabel, output("entradas_salidas.png"))

    return output_files


def main():
    parser = argparse.ArgumentParser(description="Genera las gráficas del modelo de tráfico.")
    parser.add_argument("--csv", type=Path, default=CSV_FILE, help="CSV que se quiere representar.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Carpeta de las gráficas.")
    args = parser.parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(
            f"No se ha encontrado {args.csv}. "
            "Ejecuta primero traffic_model.py o selecciona un CSV con --csv."
        )

    generate_plots(read_metrics(args.csv), args.output_dir)
    print("Gráficas generadas correctamente en la carpeta:")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
