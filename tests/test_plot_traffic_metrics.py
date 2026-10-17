import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

import plot_traffic_metrics as plots


LEGACY_HEADER = (
    "paso;estado_semaforo;coches_parados;parados_por_semaforo;"
    "parados_por_coche_delante;cola_aproximada;coches_cruzan_este_paso;"
    "cruces_totales;espera_total;espera_media"
)
LEGACY_ROWS = "0;ROJO;2;1;1;2;0;0;2;0,10\n1;VERDE;1;0;1;1;1;1;3;0,15\n"
NEW_HEADER = (
    LEGACY_HEADER
    + ";distancia_minima;pares_consecutivos_inseguros;cruces_en_rojo;"
    "posicion_primero_antes_linea;tiempo_s;coches_en_cola;longitud_cola_m;"
    "coches_presentes;hueco_libre_minimo_m;llegadas_solicitadas;"
    "llegadas_admitidas;pendientes_entrada;salidas_este_paso;salidas_totales;"
    "coches_introducidos_totales;espera_media_presentes_s;espera_media_salidos_s"
)
NEW_ROWS = (
    "0;ROJO;2;1;1;2;0;0;0,5;0,025;7,0;0;0;50,0;0,25;3;19,5;20;"
    "2,5;1;0;1;0;0;20;0,025;0,0\n"
    "1;VERDE;1;0;1;1;1;1;0,75;0,035714;7.0;0;0;49,0;0,50;2;12,5;20;"
    "2,5;2;1;1;1;1;21;0,025;0,25\n"
)
LEGACY_IMAGES = {
    "espera_media.png", "cola_aproximada.png", "cruces_totales.png",
    "motivos_parada.png", "parados_y_cola.png",
}


class TrafficPlotsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.addCleanup(plt.close, "all")

    def write_csv(self, text):
        path = self.root / "metrics.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def assert_pngs(self, paths, expected_names):
        self.assertEqual({path.name for path in paths}, expected_names)
        for path in paths:
            with self.subTest(image=path.name):
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                pixels = plt.imread(path)
                self.assertGreater(pixels.shape[0], 100)
                self.assertGreater(pixels.shape[1], 100)

    def test_historical_csv_keeps_steps_and_five_graphs(self):
        data = plots.read_metrics(self.write_csv(LEGACY_HEADER + "\n" + LEGACY_ROWS))
        self.assertEqual(data["steps"], [0, 1])
        self.assertEqual(data["queue_lengths"], [2, 1])
        self.assertEqual(data["total_waiting_time"], [2, 3])
        self.assertEqual(data["average_waiting_time"], [0.1, 0.15])
        self.assertNotIn("time_seconds", data)
        self.assertNotIn("spatial_queue_cars", data)
        with mock.patch.object(plots, "save_line_plot", wraps=plots.save_line_plot) as saved:
            paths = plots.generate_plots(data, self.root / "legacy")
        wait_plot = saved.call_args_list[0].args
        self.assertEqual(wait_plot[0], [0, 1])
        self.assertEqual(wait_plot[3], "Paso de simulación")
        self.assertIn("pasos", wait_plot[4])
        self.assert_pngs(paths, LEGACY_IMAGES)

    def test_new_csv_uses_seconds_spatial_queue_and_flow_graphs(self):
        data = plots.read_metrics(self.write_csv(NEW_HEADER + "\n" + NEW_ROWS))
        self.assertEqual(data["time_seconds"], [0.25, 0.5])
        self.assertEqual(data["total_waiting_time"], [0.5, 0.75])
        self.assertEqual(data["queue_lengths"], [2, 1])
        self.assertEqual(data["spatial_queue_cars"], [3, 2])
        self.assertEqual(data["minimum_distance"], [7, 7])
        self.assertEqual(data["pending_arrivals"], [1, 1])
        with mock.patch.object(plots, "save_line_plot", wraps=plots.save_line_plot) as saved:
            paths = plots.generate_plots(data, self.root / "physical")
        wait_plot, queue_plot = (call.args for call in saved.call_args_list[:2])
        self.assertEqual(wait_plot[0], [0.25, 0.5])
        self.assertIn("(s)", wait_plot[3])
        self.assertIn("(s)", wait_plot[4])
        self.assertEqual(queue_plot[1], [3, 2])
        self.assertIn("espacial", queue_plot[2])
        self.assert_pngs(paths, LEGACY_IMAGES | {"longitud_cola.png", "entradas_salidas.png"})
        self.assertEqual(plt.get_fignums(), [])

    def test_historical_safety_columns_are_optional(self):
        header = LEGACY_HEADER + ";distancia_minima;pares_consecutivos_inseguros"
        row = "0;ROJO;2;1;1;2;0;0;2;0,10;2,000000;0\n"
        data = plots.read_metrics(self.write_csv(header + "\n" + row))
        self.assertEqual(data["minimum_distance"], [2])
        self.assertEqual(data["unsafe_pairs"], [0])
        self.assertNotIn("time_seconds", data)

    def test_empty_road_accepts_undefined_optional_metrics_and_generates_graphs(self):
        row = ";".join((
            "0", "ROJO", "0", "0", "0", "0", "0", "0", "0,0", "0,0",
            "", "0", "0", "", "0,25", "0", "0,0", "0", "", "0", "0",
            "0", "0", "0", "0", "", "",
        ))
        data = plots.read_metrics(self.write_csv(NEW_HEADER + "\n" + row + "\n"))
        self.assertEqual(data["average_waiting_time"], [0])
        self.assertEqual(data["present_cars"], [0])
        for key in (
            "minimum_distance", "first_car_position", "minimum_free_gap_metres",
            "present_average_wait_seconds", "exited_average_wait_seconds",
        ):
            self.assertEqual(data[key], [None], key)
        paths = plots.generate_plots(data, self.root / "empty_road")
        self.assert_pngs(paths, LEGACY_IMAGES | {"longitud_cola.png", "entradas_salidas.png"})

    def test_empty_and_header_only_csv_report_no_data(self):
        for content in ("", LEGACY_HEADER + "\n", NEW_HEADER + "\n"):
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, "no tiene datos"):
                    plots.read_metrics(self.write_csv(content))

    def test_bad_metric_reports_column_and_line(self):
        invalid = LEGACY_ROWS.replace("2;0,10", "desconocido;0,10", 1)
        with self.assertRaisesRegex(ValueError, "espera_total.*línea 2"):
            plots.read_metrics(self.write_csv(LEGACY_HEADER + "\n" + invalid))


if __name__ == "__main__":
    unittest.main()
