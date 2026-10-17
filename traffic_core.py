"""Physical parameters and metrics shared by the traffic model.

All longitudinal positions refer to the FRONT of a vehicle, in metres.
This module has no dependency on FLAME GPU and does not simulate movement.
"""

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from typing import Iterable


UINT_MAX = 2**32 - 1


def _finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _whole_steps(duration_s: float, time_step_s: float, name: str) -> int:
    ratio = duration_s / time_step_s
    if not math.isfinite(ratio):
        raise ValueError(f"{name} produces an invalid number of steps")
    steps = round(ratio)
    # Scale by floating-point resolution, not by duration: a relative tolerance
    # would accept fractional steps in very long signal phases.
    tolerance = max(1e-9, 4 * math.ulp(ratio))
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{name} must be a positive multiple of time_step_s")
    return steps


@dataclass(frozen=True)
class TrafficConfig:
    """Provisional physical assumptions, validated when constructed.

    The default longitudinal positions, road length, following distance and
    travel per step scale the earlier abstract model by five. Lateral geometry
    and vehicle dimensions are separate prototype assumptions, not calibrated
    data.
    """

    initial_cars: int = 20
    time_step_s: float = 1.0
    road_length_m: float = 500.0
    domain_width_m: float = 100.0
    lane_y_m: float = 50.0
    lane_width_m: float = 3.5
    vehicle_length_m: float = 4.5
    vehicle_width_m: float = 1.8
    vehicle_height_m: float = 1.5
    min_gap_m: float = 5.5
    max_speed_mps: float = 5.0
    initial_x_m: float = 25.0
    initial_spacing_m: float = 15.0
    light_x_m: float = 250.0
    red_duration_s: float = 10.0
    green_duration_s: float = 10.0
    arrival_interval_s: float = 6.0
    queue_join_gap_m: float = 5.5
    spacing_tolerance_m: float = 5e-5

    def __post_init__(self) -> None:
        self.validate()

    @property
    def max_move_m(self) -> float:
        return self.max_speed_mps * self.time_step_s

    @property
    def safe_distance_m(self) -> float:
        return self.vehicle_length_m + self.min_gap_m

    @property
    def message_radius_m(self) -> float:
        return self.safe_distance_m + self.max_move_m

    @property
    def red_steps(self) -> int:
        return _whole_steps(self.red_duration_s, self.time_step_s, "red_duration_s")

    @property
    def green_steps(self) -> int:
        return _whole_steps(self.green_duration_s, self.time_step_s, "green_duration_s")

    @property
    def arrival_steps(self) -> int:
        if self.arrival_interval_s == 0:
            return 0
        return _whole_steps(self.arrival_interval_s, self.time_step_s, "arrival_interval_s")

    def validate(self) -> None:
        if isinstance(self.initial_cars, bool) or not isinstance(self.initial_cars, Integral):
            raise ValueError("initial_cars must be a nonnegative integer")
        if self.initial_cars < 0:
            raise ValueError("initial_cars must be a nonnegative integer")

        for field in fields(self):
            if field.name != "initial_cars":
                _finite_number(getattr(self, field.name), field.name)

        positive = (
            "time_step_s", "road_length_m", "domain_width_m", "lane_width_m",
            "vehicle_length_m", "vehicle_width_m", "vehicle_height_m",
            "max_speed_mps", "initial_spacing_m", "red_duration_s", "green_duration_s",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("min_gap_m", "arrival_interval_s", "queue_join_gap_m", "spacing_tolerance_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

        for name in ("max_move_m", "safe_distance_m", "message_radius_m"):
            _finite_number(getattr(self, name), name)
        if self.spacing_tolerance_m >= self.safe_distance_m:
            raise ValueError("spacing_tolerance_m must be smaller than safe_distance_m")
        if self.vehicle_width_m > self.lane_width_m:
            raise ValueError("vehicle_width_m must fit inside lane_width_m")
        if not self.lane_width_m / 2 <= self.lane_y_m <= self.domain_width_m - self.lane_width_m / 2:
            raise ValueError("the full lane must fit inside the lateral domain")
        if not self.vehicle_length_m <= self.light_x_m < self.road_length_m:
            raise ValueError("light_x_m must be inside the road and not before the entering vehicle's front")
        if not self.vehicle_length_m <= self.initial_x_m <= self.road_length_m:
            raise ValueError("the first vehicle must fit completely inside the road")
        if self.initial_spacing_m < self.safe_distance_m:
            raise ValueError("initial_spacing_m must include vehicle length and minimum free gap")
        if self.initial_cars:
            last_front_m = self.initial_x_m + (self.initial_cars - 1) * self.initial_spacing_m
            if not math.isfinite(last_front_m) or last_front_m > self.road_length_m:
                raise ValueError("the initial population must fit inside the road")

        red_steps = self.red_steps
        green_steps = self.green_steps
        if red_steps > UINT_MAX or green_steps > UINT_MAX or red_steps + green_steps > UINT_MAX:
            raise ValueError("signal phases and their sum must fit in an unsigned 32-bit integer")
        self.arrival_steps


def phase_for_step(step: int, config: TrafficConfig) -> str:
    """Phase used during a zero-based movement step; the first phase is red."""
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    cycle_step = step % (config.red_steps + config.green_steps)
    return "ROJO" if cycle_step < config.red_steps else "VERDE"


@dataclass(frozen=True)
class VehicleSnapshot:
    front_m: float
    speed_mps: float
    waiting_s: float = 0.0
    stopped_by_light: bool = False
    stopped_by_car: bool = False
    crossed_this_step: bool = False

    def __post_init__(self) -> None:
        for name in ("front_m", "speed_mps", "waiting_s"):
            _finite_number(getattr(self, name), name)
        if self.speed_mps < 0 or self.waiting_s < 0:
            raise ValueError("speed_mps and waiting_s must be nonnegative")


def calculate_spacing(
    positions: Iterable[float], config: TrafficConfig
) -> tuple[float | None, float | None, int]:
    """Return minimum front spacing, minimum free gap, and unsafe adjacent pairs.

    A free gap subtracts the leading vehicle's length from front spacing.
    Negative free gaps therefore expose physical overlap. The tolerance only
    absorbs small floating-point errors in the unsafe-pair diagnostic.
    """
    fronts = []
    for position in positions:
        _finite_number(position, "position")
        fronts.append(float(position))
    fronts.sort()
    if len(fronts) < 2:
        return None, None, 0
    gaps = [ahead - behind for behind, ahead in zip(fronts, fronts[1:])]
    minimum_front_gap = min(gaps)
    unsafe_count = sum(
        gap < config.safe_distance_m - config.spacing_tolerance_m for gap in gaps
    )
    return minimum_front_gap, minimum_front_gap - config.vehicle_length_m, unsafe_count


def calculate_queue(
    vehicles: Iterable[VehicleSnapshot], config: TrafficConfig
) -> tuple[int, float]:
    """Measure the nearest contiguous stopped group upstream of the signal.

    Vehicles are stopped only when their actual speed is zero. Vehicles past
    the line are excluded. Starting with the stopped vehicle nearest the line,
    walk upstream until a moving vehicle or an excessive free gap separates
    the group. Other disconnected stopped groups are not added to this queue.

    This also measures a remaining stopped group while its former head moves
    away on green; no signal-colour condition or line-distance anchor is used.
    Length runs from the head's front to the last vehicle's rear, including
    a full vehicle length for a queue consisting of just one vehicle.
    """
    upstream = sorted(
        (vehicle for vehicle in vehicles if vehicle.front_m <= config.light_x_m),
        key=lambda vehicle: vehicle.front_m,
        reverse=True,
    )
    head_index = next(
        (index for index, vehicle in enumerate(upstream) if vehicle.speed_mps == 0),
        None,
    )
    if head_index is None:
        return 0, 0.0

    head = upstream[head_index]
    last = head
    count = 1
    for vehicle in upstream[head_index + 1:]:
        free_gap_m = last.front_m - vehicle.front_m - config.vehicle_length_m
        if vehicle.speed_mps != 0 or free_gap_m > config.queue_join_gap_m + config.spacing_tolerance_m:
            break
        count += 1
        last = vehicle
    return count, head.front_m - last.front_m + config.vehicle_length_m
