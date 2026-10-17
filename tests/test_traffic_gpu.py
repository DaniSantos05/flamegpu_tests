"""Opt-in integration checks for the installed FLAME GPU CUDA backend.

Enable with TRAFFIC_GPU_TESTS=1 before running unittest discovery. The default
CPU suite does not import pyflamegpu or traffic_model through this module.
Every run writes into a temporary directory, preserving reference results.
"""

import csv
import importlib
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from traffic_core import TrafficConfig


GPU_ENABLED = os.environ.get("TRAFFIC_GPU_TESTS") == "1"
REFERENCE_CSV = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "traffic_metrics_point_agents.csv"
)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter=";"))


def number(row, column):
    value = row[column]
    return None if value == "" else float(value.replace(",", "."))


@unittest.skipUnless(GPU_ENABLED, "Set TRAFFIC_GPU_TESTS=1 to run CUDA integration checks.")
class TrafficGPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An explicitly requested GPU run must fail if its backend is missing;
        # ordinary CPU discovery skips this class before importing the model.
        cls.model = importlib.import_module("traffic_model")

    def setUp(self):
        self.output_dir = tempfile.TemporaryDirectory(prefix="traffic_gpu_test_")
        self.addCleanup(self.output_dir.cleanup)

    def run_case(self, config, *, steps, initial_positions=None):
        csv_path = Path(self.output_dir.name) / "metrics.csv"
        state = self.model.run_simulation(
            config,
            steps=steps,
            seed=12,
            output_csv=csv_path,
            no_visualisation=True,
            quiet=True,
            initial_positions=initial_positions,
        )
        rows = read_rows(csv_path)
        self.assertEqual(len(rows), steps)
        self.assertEqual(state.step, steps)
        return state, rows

    def assert_balanced(self, rows, initial_cars):
        for row in rows:
            with self.subTest(step=row["paso"]):
                requested = number(row, "llegadas_solicitadas")
                admitted = number(row, "llegadas_admitidas")
                introduced = number(row, "coches_introducidos_totales")
                exited = number(row, "salidas_totales")
                self.assertEqual(introduced, initial_cars + admitted)
                self.assertEqual(number(row, "coches_presentes"), introduced - exited)
                self.assertEqual(number(row, "pendientes_entrada"), requested - admitted)

    def test_twenty_step_reference_preserves_events_and_scales_positions(self):
        _, rows = self.run_case(TrafficConfig(arrival_interval_s=0), steps=20)
        reference = read_rows(REFERENCE_CSV)
        self.assertEqual(len(reference), 20)
        positional_columns = {"distancia_minima", "posicion_primero_antes_linea"}
        for actual, expected in zip(rows, reference):
            for column in expected:
                with self.subTest(step=expected["paso"], column=column):
                    if column == "estado_semaforo":
                        self.assertEqual(actual[column], expected[column])
                    else:
                        scale = 5 if column in positional_columns else 1
                        self.assertAlmostEqual(
                            number(actual, column), number(expected, column) * scale,
                            delta=5e-5,
                        )
        self.assertEqual(number(rows[-1], "cruces_totales"), 4)
        self.assertEqual(number(rows[-1], "espera_total"), 145)
        self.assert_balanced(rows, initial_cars=20)

    def test_host_birth_moves_in_the_same_step_from_an_empty_road(self):
        config = TrafficConfig(initial_cars=0, arrival_interval_s=1)
        state, rows = self.run_case(config, steps=1)
        row = rows[0]
        self.assertEqual(number(row, "llegadas_solicitadas"), 1)
        self.assertEqual(number(row, "llegadas_admitidas"), 1)
        self.assertEqual(number(row, "coches_presentes"), 1)
        # Seeing its moved position proves birth was committed before both
        # message publication and movement, rather than at the end of the step.
        self.assertAlmostEqual(
            number(row, "posicion_primero_antes_linea"),
            config.vehicle_length_m + config.max_move_m,
            delta=5e-5,
        )
        self.assertEqual(state.pending, 0)
        self.assert_balanced(rows, initial_cars=0)

    def test_empty_road_without_demand_remains_empty(self):
        config = TrafficConfig(initial_cars=0, arrival_interval_s=0)
        state, rows = self.run_case(config, steps=3)
        for row in rows:
            with self.subTest(step=row["paso"]):
                for column in (
                    "coches_presentes", "llegadas_solicitadas", "llegadas_admitidas",
                    "pendientes_entrada", "salidas_totales", "cruces_totales",
                    "espera_total", "coches_en_cola", "longitud_cola_m",
                ):
                    self.assertEqual(number(row, column), 0)
                for column in (
                    "distancia_minima", "hueco_libre_minimo_m",
                    "posicion_primero_antes_linea", "espera_media_presentes_s",
                    "espera_media_salidos_s",
                ):
                    self.assertIsNone(number(row, column))
        self.assertEqual((state.introduced, state.exited), (0, 0))
        self.assert_balanced(rows, initial_cars=0)

    def test_failed_run_preserves_previous_csv_and_keeps_partial_rows(self):
        output = Path(self.output_dir.name) / "protected.csv"
        sentinel = b"previous valid result\n"
        output.write_bytes(sentinel)
        original_run = self.model.TrafficMetrics.run

        def fail_after_second_row(metrics, flamegpu):
            step = metrics.state.step
            original_run(metrics, flamegpu)
            if step == 1:
                raise RuntimeError("forced metrics failure after row 1")

        with mock.patch.object(self.model.TrafficMetrics, "run", fail_after_second_row):
            with self.assertRaisesRegex(RuntimeError, "forced metrics failure"):
                self.model.run_simulation(
                    TrafficConfig(initial_cars=1, arrival_interval_s=0),
                    steps=3,
                    seed=12,
                    output_csv=output,
                    no_visualisation=True,
                    quiet=True,
                )
        self.assertEqual(output.read_bytes(), sentinel)
        partial_rows = read_rows(output.with_name("protected_partial.csv"))
        self.assertEqual([row["paso"] for row in partial_rows], ["0", "1"])
        self.assertFalse(output.with_suffix(".metadata.json").exists())
        self.assertFalse(output.with_suffix(".metadata.tmp.json").exists())

    def test_exit_is_counted_before_death_and_the_next_step_stays_empty(self):
        config = TrafficConfig(initial_cars=1, arrival_interval_s=0)
        state, rows = self.run_case(
            config, steps=2, initial_positions=[config.road_length_m]
        )
        self.assertEqual(number(rows[0], "salidas_este_paso"), 1)
        self.assertEqual(number(rows[1], "salidas_este_paso"), 0)
        for row in rows:
            self.assertEqual(number(row, "salidas_totales"), 1)
            self.assertEqual(number(row, "coches_presentes"), 0)
            self.assertIsNone(number(row, "distancia_minima"))
            self.assertIsNone(number(row, "espera_media_presentes_s"))
        # run_simulation also downloads the final GPU population after death
        # and raises if that population disagrees with this ledger.
        self.assertEqual((state.introduced, state.exited), (1, 1))
        self.assert_balanced(rows, initial_cars=1)

    def test_crossing_and_waiting_survive_the_vehicle_departure(self):
        config = TrafficConfig(initial_cars=1, arrival_interval_s=0)
        state, rows = self.run_case(
            config, steps=120, initial_positions=[config.light_x_m]
        )
        self.assertEqual(number(rows[19], "cruces_totales"), 1)
        self.assertEqual(number(rows[19], "espera_total"), config.red_duration_s)
        departures = [index for index, row in enumerate(rows)
                      if number(row, "salidas_este_paso")]
        self.assertEqual(len(departures), 1)
        self.assertGreater(departures[0], 19)
        self.assertLess(departures[0], len(rows) - 1)
        for row in rows[departures[0]:]:
            with self.subTest(step=row["paso"]):
                self.assertEqual(number(row, "coches_presentes"), 0)
                self.assertEqual(number(row, "cruces_totales"), 1)
                self.assertEqual(number(row, "espera_total"), config.red_duration_s)
                self.assertEqual(number(row, "espera_media_salidos_s"), config.red_duration_s)
        self.assertEqual((state.total_crossings, state.exited), (1, 1))
        self.assertEqual(state.departed_waiting_s, config.red_duration_s)
        self.assert_balanced(rows, initial_cars=1)

    def test_dense_arrivals_preserve_backlog_population_and_safety(self):
        config = TrafficConfig(arrival_interval_s=1)
        state, rows = self.run_case(config, steps=300)
        self.assertEqual(state.requested, 300)
        self.assertGreater(state.admitted, 0)
        self.assertGreater(state.exited, 0)
        self.assertGreater(state.pending, 0)
        self.assertTrue(any(number(row, "pendientes_entrada") > 0 for row in rows))
        previous_admitted = previous_exited = previous_crossed = 0
        previous_waiting = 0.0
        for row in rows:
            with self.subTest(step=row["paso"]):
                admitted = number(row, "llegadas_admitidas")
                exited = number(row, "salidas_totales")
                crossed = number(row, "cruces_totales")
                waiting = number(row, "espera_total")
                self.assertIn(admitted - previous_admitted, (0, 1))
                self.assertEqual(exited - previous_exited, number(row, "salidas_este_paso"))
                self.assertEqual(crossed - previous_crossed, number(row, "coches_cruzan_este_paso"))
                self.assertGreaterEqual(waiting, previous_waiting)
                self.assertEqual(number(row, "cruces_en_rojo"), 0)
                self.assertEqual(number(row, "pares_consecutivos_inseguros"), 0)
                free_gap = number(row, "hueco_libre_minimo_m")
                if free_gap is not None:
                    self.assertGreaterEqual(
                        free_gap, config.min_gap_m - config.spacing_tolerance_m
                    )
                previous_admitted, previous_exited = admitted, exited
                previous_crossed, previous_waiting = crossed, waiting
        self.assertEqual(state.introduced - state.exited, number(rows[-1], "coches_presentes"))
        self.assert_balanced(rows, initial_cars=config.initial_cars)

    def test_half_second_steps_measure_actual_speed_and_wait_in_seconds(self):
        config = TrafficConfig(
            initial_cars=1, time_step_s=0.5,
            red_duration_s=1, green_duration_s=1, arrival_interval_s=0,
        )
        _, rows = self.run_case(config, steps=3, initial_positions=[249.5])
        self.assertEqual(number(rows[0], "posicion_primero_antes_linea"), 250)
        self.assertEqual(number(rows[0], "espera_total"), 0)
        self.assertEqual(number(rows[0], "coches_parados"), 0)
        self.assertEqual(number(rows[1], "espera_total"), 0.5)
        self.assertEqual(number(rows[1], "coches_parados"), 1)
        self.assertEqual(number(rows[2], "coches_cruzan_este_paso"), 1)
        self.assertEqual(number(rows[2], "espera_total"), 0.5)
        self.assertEqual([number(row, "tiempo_s") for row in rows], [0.5, 1.0, 1.5])

    def test_nonbinary_stop_line_keeps_the_waiting_car_in_the_queue(self):
        config = TrafficConfig(initial_cars=1, light_x_m=250.1, arrival_interval_s=0)
        _, rows = self.run_case(config, steps=1, initial_positions=[250.1])
        row = rows[0]
        effective_line = struct.unpack("f", struct.pack("f", config.light_x_m))[0]
        self.assertEqual(number(row, "coches_parados"), 1)
        self.assertEqual(number(row, "coches_en_cola"), 1)
        self.assertIsNotNone(number(row, "posicion_primero_antes_linea"))
        self.assertAlmostEqual(
            number(row, "posicion_primero_antes_linea"), effective_line, delta=1e-6
        )

    def test_sub_float_resolution_movement_is_counted_as_waiting(self):
        config = TrafficConfig(
            initial_cars=2, min_gap_m=5.8, queue_join_gap_m=5.8,
            arrival_interval_s=0,
        )
        _, rows = self.run_case(
            config, steps=1, initial_positions=[239.6999969482422, 250.0]
        )
        row = rows[0]
        self.assertEqual(number(row, "coches_parados"), 2)
        self.assertEqual(number(row, "parados_por_semaforo"), 1)
        self.assertEqual(number(row, "parados_por_coche_delante"), 1)
        self.assertEqual(number(row, "espera_total"), 2)
        self.assertEqual(number(row, "coches_en_cola"), 2)


if __name__ == "__main__":
    unittest.main()
