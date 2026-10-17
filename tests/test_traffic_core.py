"""Physical invariants and metric counterexamples; no GPU is required."""

from dataclasses import FrozenInstanceError, replace
import unittest

from traffic_core import (
    TrafficConfig,
    VehicleSnapshot,
    calculate_queue,
    calculate_spacing,
    phase_for_step,
)


class ConfigTests(unittest.TestCase):
    def test_default_units_and_vehicle_geometry(self):
        config = TrafficConfig()
        self.assertEqual(config.max_move_m, 5.0)
        self.assertEqual(config.safe_distance_m, 10.0)
        self.assertEqual(config.message_radius_m, 15.0)
        self.assertEqual((config.red_steps, config.green_steps, config.arrival_steps), (10, 10, 6))
        with self.assertRaises(FrozenInstanceError):
            config.vehicle_length_m = 5.0

    def test_time_resolution_converts_seconds_to_steps(self):
        config = TrafficConfig(time_step_s=0.5)
        self.assertEqual(config.max_move_m, 2.5)
        self.assertEqual((config.red_steps, config.green_steps, config.arrival_steps), (20, 20, 12))
        self.assertEqual(TrafficConfig(arrival_interval_s=0).arrival_steps, 0)
        self.assertEqual(TrafficConfig(time_step_s=0.1, red_duration_s=0.3).red_steps, 3)

    def test_empty_population_and_first_rear_at_entrance_are_valid(self):
        self.assertEqual(TrafficConfig(initial_cars=0).initial_cars, 0)
        self.assertEqual(TrafficConfig(initial_x_m=4.5).initial_x_m, 4.5)

    def test_invalid_parameters_are_rejected(self):
        cases = (
            {"initial_cars": -1}, {"initial_cars": 2.5}, {"initial_cars": True},
            {"time_step_s": 0}, {"time_step_s": float("nan")},
            {"max_speed_mps": float("inf")}, {"vehicle_length_m": -1},
            {"vehicle_width_m": 4}, {"lane_y_m": 1}, {"lane_y_m": 100},
            {"initial_x_m": 4.0}, {"initial_x_m": 501},
            {"initial_spacing_m": 9.9}, {"initial_cars": 100},
            {"light_x_m": 0}, {"light_x_m": 2}, {"light_x_m": 500}, {"road_length_m": -1},
            {"min_gap_m": -0.1}, {"queue_join_gap_m": -1},
            {"spacing_tolerance_m": -1}, {"spacing_tolerance_m": 10},
            {"red_duration_s": 0}, {"green_duration_s": 2.5},
            {"arrival_interval_s": -1}, {"arrival_interval_s": 0.5},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                TrafficConfig(**parameters)

    def test_new_vehicles_cannot_be_born_past_the_stop_line(self):
        with self.assertRaises(ValueError):
            TrafficConfig(initial_cars=0, light_x_m=4.49, arrival_interval_s=1)
        config = TrafficConfig(initial_cars=0, light_x_m=4.5, arrival_interval_s=1)
        self.assertEqual(config.light_x_m, config.vehicle_length_m)

    def test_signal_phase_and_cycle_fit_the_gpu_unsigned_integer(self):
        unsigned_max = 2**32 - 1
        cases = (
            {"red_duration_s": unsigned_max + 1, "green_duration_s": 1},
            {"red_duration_s": 1, "green_duration_s": unsigned_max + 1},
            {"red_duration_s": unsigned_max, "green_duration_s": 1},
            {"red_duration_s": 2_000_000_000.25, "green_duration_s": 1},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                TrafficConfig(**parameters)
        config = TrafficConfig(red_duration_s=unsigned_max - 1, green_duration_s=1)
        self.assertEqual(config.red_steps + config.green_steps, unsigned_max)

    def test_signal_boundaries_with_asymmetric_cycle(self):
        config = TrafficConfig(red_duration_s=3, green_duration_s=2)
        self.assertEqual(
            [phase_for_step(step, config) for step in range(7)],
            ["ROJO", "ROJO", "ROJO", "VERDE", "VERDE", "ROJO", "ROJO"],
        )
        for step in (-1, 0.5, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                phase_for_step(step, config)


class SpacingTests(unittest.TestCase):
    def setUp(self):
        self.config = TrafficConfig()

    def test_no_pairs_have_no_minimum(self):
        self.assertEqual(calculate_spacing([], self.config), (None, None, 0))
        self.assertEqual(calculate_spacing([25], self.config), (None, None, 0))

    def test_unsorted_positions_use_free_gap_and_adjacent_pairs(self):
        self.assertEqual(calculate_spacing([35, 10, 20], self.config), (10.0, 5.5, 0))
        self.assertEqual(calculate_spacing([10, 0, 19], self.config), (9.0, 4.5, 1))

    def test_physical_overlap_is_detected_even_with_distinct_positions(self):
        self.assertEqual(calculate_spacing([25, 28], self.config), (3.0, -1.5, 1))
        self.assertEqual(calculate_spacing([25, 25], self.config), (0.0, -4.5, 1))

    def test_tolerance_does_not_hide_material_spacing_errors(self):
        tolerance = self.config.spacing_tolerance_m
        self.assertEqual(calculate_spacing([0, 10 - tolerance / 2], self.config)[2], 0)
        self.assertEqual(calculate_spacing([0, 10 - tolerance * 2], self.config)[2], 1)

    def test_nonfinite_positions_are_rejected(self):
        for position in (float("nan"), float("inf")):
            with self.subTest(position=position), self.assertRaises(ValueError):
                calculate_spacing([position], self.config)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.config = TrafficConfig()

    def test_empty_moving_and_downstream_stopped_vehicles_are_not_a_queue(self):
        self.assertEqual(calculate_queue([], self.config), (0, 0.0))
        vehicles = [VehicleSnapshot(250, 1), VehicleSnapshot(251, 0), VehicleSnapshot(300, 0)]
        self.assertEqual(calculate_queue(vehicles, self.config), (0, 0.0))
        self.assertEqual(calculate_queue([VehicleSnapshot(250.000001, 0)], self.config), (0, 0.0))

    def test_queue_length_includes_both_end_vehicles(self):
        self.assertEqual(calculate_queue([VehicleSnapshot(250, 0)], self.config), (1, 4.5))
        vehicles = [VehicleSnapshot(230, 0), VehicleSnapshot(250, 0), VehicleSnapshot(240, 0)]
        self.assertEqual(calculate_queue(vehicles, self.config), (3, 24.5))

    def test_gap_separates_stopped_groups_instead_of_summing_them(self):
        vehicles = [VehicleSnapshot(250, 0), VehicleSnapshot(240, 0), VehicleSnapshot(220, 0)]
        self.assertEqual(calculate_queue(vehicles, self.config), (2, 14.5))

    def test_moving_vehicle_separates_two_stopped_groups(self):
        vehicles = [VehicleSnapshot(250, 0), VehicleSnapshot(240, 1), VehicleSnapshot(230, 0)]
        self.assertEqual(calculate_queue(vehicles, self.config), (1, 4.5))

    def test_green_discharge_preserves_remaining_stopped_group(self):
        vehicles = [VehicleSnapshot(251, 5), VehicleSnapshot(240, 5),
                    VehicleSnapshot(230, 0), VehicleSnapshot(220, 0)]
        self.assertEqual(calculate_queue(vehicles, self.config), (2, 14.5))

    def test_nearest_stopped_group_does_not_need_to_touch_the_stop_line(self):
        vehicles = [VehicleSnapshot(100, 0), VehicleSnapshot(90, 0)]
        self.assertEqual(calculate_queue(vehicles, self.config), (2, 14.5))

    def test_join_gap_is_a_separate_free_space_parameter(self):
        vehicles = [VehicleSnapshot(250, 0), VehicleSnapshot(240, 0)]
        config = replace(self.config, queue_join_gap_m=4)
        self.assertEqual(calculate_queue(vehicles, config), (1, 4.5))

    def test_join_tolerance_and_strictly_positive_speed(self):
        tolerance = self.config.spacing_tolerance_m
        vehicles = [VehicleSnapshot(250, 0), VehicleSnapshot(240 - tolerance / 2, 0)]
        self.assertEqual(calculate_queue(vehicles, self.config)[0], 2)
        vehicles[1] = VehicleSnapshot(240 - tolerance * 2, 0)
        self.assertEqual(calculate_queue(vehicles, self.config)[0], 1)
        self.assertEqual(calculate_queue([VehicleSnapshot(250, 1e-8)], self.config), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
