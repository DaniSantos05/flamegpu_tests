"""
Carretera de un carril con FLAME GPU 2, unidades físicas y población dinámica.

x es el frontal del vehículo (m), speed su velocidad real (m/s) y
waiting_time la espera observada acumulada (s). Los supuestos iniciales
no están calibrados con tráfico real. El modelo sigue siendo un prototipo
en Python con comportamiento RTC en GPU; la migración final sigue aplazada.
"""

import argparse
import csv
import json
import struct
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pyflamegpu

from traffic_core import (
    TrafficConfig,
    VehicleSnapshot,
    calculate_queue,
    calculate_spacing,
    phase_for_step,
)


OUTPUT_CSV = Path(__file__).with_name("traffic_metrics.csv")
CSV_COLUMNS = [
    "paso", "estado_semaforo", "coches_parados", "parados_por_semaforo",
    "parados_por_coche_delante", "cola_aproximada", "coches_cruzan_este_paso",
    "cruces_totales", "espera_total", "espera_media", "distancia_minima",
    "pares_consecutivos_inseguros", "cruces_en_rojo", "posicion_primero_antes_linea",
    "tiempo_s", "coches_presentes", "coches_en_cola", "longitud_cola_m",
    "hueco_libre_minimo_m", "llegadas_solicitadas", "llegadas_admitidas",
    "pendientes_entrada", "salidas_este_paso", "salidas_totales",
    "coches_introducidos_totales", "espera_media_presentes_s", "espera_media_salidos_s",
]


car_output_position = r"""
FLAMEGPU_AGENT_FUNCTION(car_output_position, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));
    return flamegpu::ALIVE;
}
"""

car_move = r"""
FLAMEGPU_AGENT_FUNCTION(car_move, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const flamegpu::id_t ID = FLAMEGPU->getID();
    const float DT = FLAMEGPU->environment.getProperty<float>("time_step_s");
    const float MAX_SPEED = FLAMEGPU->environment.getProperty<float>("max_speed_mps");
    const float SAFE_DISTANCE = FLAMEGPU->environment.getProperty<float>("safe_distance_m");
    const float LENGTH = FLAMEGPU->environment.getProperty<float>("vehicle_length_m");
    const float ROAD_LENGTH = FLAMEGPU->environment.getProperty<float>("road_length_m");
    const float LIGHT_X = FLAMEGPU->environment.getProperty<float>("light_x_m");
    const unsigned int RED_STEPS = FLAMEGPU->environment.getProperty<unsigned int>("red_steps");
    const unsigned int GREEN_STEPS = FLAMEGPU->environment.getProperty<unsigned int>("green_steps");

    float x = FLAMEGPU->getVariable<float>("x");
    const float y = FLAMEGPU->getVariable<float>("y");
    const unsigned int step = FLAMEGPU->getStepCounter();
    const bool red = step % (RED_STEPS + GREEN_STEPS) < RED_STEPS;
    float allowed_move = MAX_SPEED * DT;
    bool red_light_stop = false;
    FLAMEGPU->setVariable<float>("crossed_this_step", 0.0f);

    // Se usan posiciones anteriores al movimiento de todos los coches.
    for (const auto &message : FLAMEGPU->message_in(x, y)) {
        if (message.getVariable<flamegpu::id_t>("id") == ID) {
            continue;
        }
        const float other_x = message.getVariable<float>("x");
        const float other_y = message.getVariable<float>("y");
        const float distance = other_x - x;
        if (fabsf(other_y - y) < 0.1f && distance > 0.0f) {
            // Distancia entre frontales = longitud + hueco libre.
            allowed_move = fminf(allowed_move, fmaxf(0.0f, distance - SAFE_DISTANCE));
        }
    }

    if (red && x <= LIGHT_X) {
        const float distance_to_line = LIGHT_X - x;
        allowed_move = fminf(allowed_move, distance_to_line);
        red_light_stop = distance_to_line <= 0.0f;
    }

    const float old_x = x;
    const float new_x = x + allowed_move;
    const float actual_move = new_x - old_x;
    // Un avance menor que la resolución float puede dejar x sin cambios.
    if (actual_move <= 0.0f) {
        FLAMEGPU->setVariable<float>("speed", 0.0f);
        FLAMEGPU->setVariable<float>("stopped", 1.0f);
        FLAMEGPU->setVariable<float>("stopped_by_light", red_light_stop ? 1.0f : 0.0f);
        FLAMEGPU->setVariable<float>("stopped_by_car", red_light_stop ? 0.0f : 1.0f);
        FLAMEGPU->setVariable<float>("waiting_time",
            FLAMEGPU->getVariable<float>("waiting_time") + DT);
    } else {
        x = new_x;
        FLAMEGPU->setVariable<float>("speed", actual_move / DT);
        FLAMEGPU->setVariable<float>("stopped", 0.0f);
        FLAMEGPU->setVariable<float>("stopped_by_light", 0.0f);
        FLAMEGPU->setVariable<float>("stopped_by_car", 0.0f);
        if (old_x <= LIGHT_X && x > LIGHT_X) {
            FLAMEGPU->setVariable<float>("crossed_this_step", 1.0f);
            FLAMEGPU->setVariable<float>("total_crossings",
                FLAMEGPU->getVariable<float>("total_crossings") + 1.0f);
        }
    }

    FLAMEGPU->setVariable<float>("x", x);
    FLAMEGPU->setVariable<float>("x_centre", x - LENGTH / 2.0f);
    // Se contabiliza primero; otra capa lo elimina cuando sale la trasera.
    FLAMEGPU->setVariable<float>("exit_pending", x - LENGTH >= ROAD_LENGTH ? 1.0f : 0.0f);
    return flamegpu::ALIVE;
}
"""

traffic_light_update = r"""
FLAMEGPU_AGENT_FUNCTION(traffic_light_update, flamegpu::MessageNone, flamegpu::MessageNone) {
    const unsigned int RED_STEPS = FLAMEGPU->environment.getProperty<unsigned int>("red_steps");
    const unsigned int GREEN_STEPS = FLAMEGPU->environment.getProperty<unsigned int>("green_steps");
    const unsigned int step = FLAMEGPU->getStepCounter();
    FLAMEGPU->setVariable<float>("light_state",
        step % (RED_STEPS + GREEN_STEPS) < RED_STEPS ? 0.0f : 1.0f);
    return flamegpu::ALIVE;
}
"""

car_remove = r"""
FLAMEGPU_AGENT_FUNCTION(car_remove, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<float>("exit_pending") > 0.0f) {
        return flamegpu::DEAD;
    }
    return flamegpu::ALIVE;
}
"""


@dataclass
class RunState:
    """Contadores que sobreviven a la eliminación de los vehículos."""
    config: TrafficConfig
    step: int = 0
    requested: int = 0
    admitted: int = 0
    pending: int = 0
    exited: int = 0
    total_crossings: int = 0
    departed_waiting_s: float = 0.0
    last_row: dict | None = None

    @property
    def introduced(self):
        return self.config.initial_cars + self.admitted


def initialise_car(instance, front_m, config):
    """Misma inicialización para AgentVector y nacimientos desde el host."""
    values = {
        "x": front_m, "x_centre": front_m - config.vehicle_length_m / 2,
        "y": config.lane_y_m, "z": config.vehicle_height_m / 2,
        "speed": config.max_speed_mps, "waiting_time": 0.0, "stopped": 0.0,
        "stopped_by_light": 0.0, "stopped_by_car": 0.0,
        "crossed_this_step": 0.0, "total_crossings": 0.0, "exit_pending": 0.0,
    }
    for name, value in values.items():
        instance.setVariableFloat(name, float(value))


class TrafficArrivals(pyflamegpu.HostFunction):
    """Demanda periódica; las solicitudes sin hueco esperan fuera del tramo."""

    def __init__(self, state):
        super().__init__()
        self.state = state

    def run(self, flamegpu):
        state = self.state
        config = state.config
        if config.arrival_steps and state.step % config.arrival_steps == 0:
            state.requested += 1
            state.pending += 1
        if not state.pending:
            return

        # Nace con la trasera en x=0 y el frontal en la longitud del coche.
        entry_front = config.vehicle_length_m
        car_api = flamegpu.agent("Car")
        # rc4 no admite descargar una población vacía aún sin buffers CUDA.
        population = car_api.getPopulationData() if car_api.count() else []
        nearest = min(
            (instance.getVariableFloat("x") for instance in population),
            default=None,
        )
        if nearest is not None and nearest - entry_front < config.safe_distance_m:
            return

        new_car = flamegpu.agent("Car").newAgent()
        initialise_car(new_car, entry_front, config)
        state.admitted += 1
        state.pending -= 1
        # El nacimiento se hace efectivo al terminar esta capa host.


def read_vehicle(instance):
    return VehicleSnapshot(
        front_m=instance.getVariableFloat("x"),
        speed_mps=instance.getVariableFloat("speed"),
        waiting_s=instance.getVariableFloat("waiting_time"),
        stopped_by_light=bool(instance.getVariableFloat("stopped_by_light")),
        stopped_by_car=bool(instance.getVariableFloat("stopped_by_car")),
        crossed_this_step=bool(instance.getVariableFloat("crossed_this_step")),
    )


def decimal_text(value):
    return "" if value is None else f"{value:.6f}".replace(".", ",")


def gpu_float(value):
    """Valor efectivo de una propiedad float del modelo CUDA."""
    return struct.unpack("f", struct.pack("f", value))[0]


class TrafficMetrics(pyflamegpu.HostFunction):
    """Muestra posterior al movimiento y contabilidad anterior a la eliminación."""

    def __init__(self, state, csv_path, quiet=False):
        super().__init__()
        self.state = state
        self.csv_path = Path(csv_path)
        self.quiet = quiet
        # La clasificación host debe usar exactamente la misma línea que GPU.
        self.metrics_config = replace(state.config, light_x_m=gpu_float(state.config.light_x_m))

    def run(self, flamegpu):
        state = self.state
        config = self.metrics_config
        # Diagnóstico de un prototipo pequeño, con transferencias GPU-host.
        car_api = flamegpu.agent("Car")
        population = car_api.getPopulationData() if car_api.count() else []
        active, departing = [], []
        for instance in population:
            vehicle = read_vehicle(instance)
            if instance.getVariableFloat("exit_pending") > 0:
                departing.append(vehicle)
            else:
                active.append(vehicle)

        all_vehicles = active + departing
        crossed = sum(vehicle.crossed_this_step for vehicle in all_vehicles)
        state.total_crossings += crossed
        state.exited += len(departing)
        state.departed_waiting_s += sum(vehicle.waiting_s for vehicle in departing)
        active_waiting = sum(vehicle.waiting_s for vehicle in active)
        total_waiting = state.departed_waiting_s + active_waiting

        # Comprobar también los que abandonan durante este paso.
        min_front, min_free, unsafe_pairs = calculate_spacing(
            (vehicle.front_m for vehicle in all_vehicles), config
        )
        queue_cars, queue_m = calculate_queue(active, config)
        stopped = sum(vehicle.speed_mps == 0 for vehicle in active)
        by_light = sum(vehicle.stopped_by_light for vehicle in active)
        by_car = sum(vehicle.stopped_by_car for vehicle in active)
        red_crossings = crossed if phase_for_step(state.step, config) == "ROJO" else 0
        first = max(
            (vehicle.front_m for vehicle in active if vehicle.front_m <= config.light_x_m),
            default=None,
        )
        row = dict(zip(CSV_COLUMNS, [
            state.step, phase_for_step(state.step, config), stopped, by_light, by_car,
            stopped, crossed, state.total_crossings, total_waiting,
            total_waiting / state.introduced if state.introduced else 0.0,
            min_front, unsafe_pairs, red_crossings, first,
            (state.step + 1) * config.time_step_s, len(active), queue_cars, queue_m,
            min_free, state.requested, state.admitted, state.pending, len(departing),
            state.exited, state.introduced,
            active_waiting / len(active) if active else None,
            state.departed_waiting_s / state.exited if state.exited else None,
        ]))
        state.last_row = row
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, delimiter=";")
            writer.writerow({
                name: decimal_text(value) if isinstance(value, float) or value is None else value
                for name, value in row.items()
            })

        if not self.quiet and state.step % 10 == 0:
            print(
                f"Paso {state.step} (t={row['tiempo_s']:g} s): "
                f"{row['estado_semaforo']}, presentes={len(active)}, "
                f"cola={queue_cars} coches/{queue_m:g} m, "
                f"cruces={state.total_crossings}, salidas={state.exited}, "
                f"pendientes={state.pending}"
            )

        errors = []
        if red_crossings:
            errors.append(f"{red_crossings} cruces en rojo")
        if unsafe_pairs:
            errors.append(f"{unsafe_pairs} pares inseguros; distancia frontal mínima {min_front} m")
        if len(active) != state.introduced - state.exited:
            errors.append("no se conserva el balance de población")
        if state.pending != state.requested - state.admitted:
            errors.append("no se conserva el balance de solicitudes de entrada")
        if stopped != by_light + by_car:
            errors.append("los motivos de parada no coinciden con los detenidos")
        if errors:
            raise RuntimeError(f"Paso {state.step}: " + "; ".join(errors))
        state.step += 1


def create_csv_file(csv_path):
    with Path(csv_path).open("w", newline="", encoding="utf-8") as file:
        csv.writer(file, delimiter=";").writerow(CSV_COLUMNS)


def build_model(config, state, csv_path, quiet=False):
    model = pyflamegpu.ModelDescription("traffic_model")
    env = model.Environment()
    for name, value in {
        "time_step_s": config.time_step_s, "road_length_m": config.road_length_m,
        "max_speed_mps": config.max_speed_mps, "safe_distance_m": config.safe_distance_m,
        "vehicle_length_m": config.vehicle_length_m, "light_x_m": config.light_x_m,
    }.items():
        env.newPropertyFloat(name, float(value))
    env.newPropertyUInt("red_steps", config.red_steps)
    env.newPropertyUInt("green_steps", config.green_steps)

    message = model.newMessageSpatial2D("car_location")
    message.newVariableID("id")
    message.setRadius(config.message_radius_m)
    message.setMin(0.0, 0.0)
    # La trasera debe abandonar el tramo antes de eliminar el agente.
    message.setMax(
        config.road_length_m + config.vehicle_length_m + config.max_move_m + 1.0,
        config.domain_width_m,
    )

    car = model.newAgent("Car")
    for name in (
        "x", "x_centre", "y", "z", "speed", "waiting_time", "stopped",
        "stopped_by_light", "stopped_by_car", "crossed_this_step",
        "total_crossings", "exit_pending",
    ):
        car.newVariableFloat(name)

    traffic_light = model.newAgent("TrafficLight")
    for name in ("x", "y", "z", "light_state"):
        traffic_light.newVariableFloat(name)

    output_fn = car.newRTCFunction("car_output_position", car_output_position)
    output_fn.setMessageOutput("car_location")
    move_fn = car.newRTCFunction("car_move", car_move)
    move_fn.setMessageInput("car_location")
    light_fn = traffic_light.newRTCFunction("traffic_light_update", traffic_light_update)
    remove_fn = car.newRTCFunction("car_remove", car_remove)
    remove_fn.setAllowAgentDeath(True)

    # Los nacimientos host solo se incorporan al terminar su capa.
    model.newLayer().addHostFunction(TrafficArrivals(state))
    model.newLayer().addAgentFunction(output_fn)
    model.newLayer().addAgentFunction(move_fn)
    model.newLayer().addAgentFunction(light_fn)
    # Se guardan eventos y espera antes de que desaparezcan los agentes.
    model.newLayer().addHostFunction(TrafficMetrics(state, csv_path, quiet))
    model.newLayer().addAgentFunction(remove_fn)
    return model, car, traffic_light


def create_initial_population(car, config, positions=None):
    if positions is None:
        positions = [
            config.initial_x_m + index * config.initial_spacing_m
            for index in range(config.initial_cars)
        ]
    positions = list(positions)
    if len(positions) != config.initial_cars:
        raise ValueError("La población proporcionada no coincide con initial_cars.")
    if any(
        position < config.vehicle_length_m or position > config.road_length_m
        for position in positions
    ):
        raise ValueError("Los coches iniciales deben caber completamente en la carretera.")
    if calculate_spacing(positions, config)[2]:
        raise ValueError("La población inicial no respeta el hueco libre mínimo.")
    population = pyflamegpu.AgentVector(car, len(positions))
    for instance, front in zip(population, positions):
        initialise_car(instance, front, config)
    return population


def create_traffic_light_population(traffic_light, config):
    population = pyflamegpu.AgentVector(traffic_light, 1)
    light = population[0]
    for name, value in {
        "x": config.light_x_m,
        "y": config.lane_y_m + config.lane_width_m / 2 + 3.0,
        "z": 1.0, "light_state": 0.0,
    }.items():
        light.setVariableFloat(name, float(value))
    return population


def configure_visualisation(simulation, config, view_mode="detail"):
    visualisation = simulation.getVisualisation()
    visualisation.setWindowTitle("TFG - carretera con unidades, cola y entradas/salidas")
    window_width = 1280
    visualisation.setWindowDimensions(window_width, 720)
    visualisation.setClearColor(0.035, 0.04, 0.05)
    if view_mode == "overview":
        visible_width_m = config.road_length_m * 1.1
        camera_x = config.road_length_m / 2
    else:
        visible_width_m = min(180.0, config.road_length_m * 1.1)
        if visible_width_m >= config.road_length_m:
            camera_x = config.road_length_m / 2
        else:
            # Más espacio aguas arriba de la línea para observar la cola.
            camera_x = max(visible_width_m / 2, min(
                config.road_length_m - visible_width_m / 2,
                config.light_x_m - 0.2 * visible_width_m,
            ))
    # En ortográfica el zoom expresa metros visibles por píxel.
    visualisation.setOrthographic(True)
    visualisation.setOrthographicZoomModifier(visible_width_m / window_width)
    visualisation.setInitialCameraTarget(camera_x, config.lane_y_m, 0.0)
    visualisation.setInitialCameraLocation(camera_x, config.lane_y_m, 150.0)
    visualisation.setInitialCameraRoll(0.0)
    visualisation.setCameraSpeed(0.05)
    visualisation.setFPSVisible(False)
    visualisation.setSimulationSpeed(10)  # Solo velocidad de demostración.

    car_vis = visualisation.addAgent("Car")
    car_vis.setModel(pyflamegpu.CUBE)
    car_vis.setXVariable("x_centre")
    car_vis.setModelScale(
        config.vehicle_length_m, config.vehicle_width_m, config.vehicle_height_m
    )
    car_vis.setColor(pyflamegpu.HSVInterpolation.REDGREEN(
        "speed", 0.0, config.max_speed_mps
    ))
    light_vis = visualisation.addAgent("TrafficLight")
    light_vis.setModel(pyflamegpu.SPHERE)
    light_vis.setModelScale(2.0)
    light_vis.setColor(pyflamegpu.HSVInterpolation.REDGREEN("light_state", 0.0, 1.0))

    # Banda de asfalto dibujada con segmentos densos, sin cambiar la geometría
    # física ni requerir texturas o modelos externos.
    asphalt = visualisation.newLineSketch(0.20, 0.22, 0.26, 1.0)
    stripes = max(1, int(config.lane_width_m / 0.08) + 1)
    for stripe in range(stripes + 1):
        y = config.lane_y_m - config.lane_width_m / 2 + stripe * config.lane_width_m / stripes
        asphalt.addVertex(0.0, y, -0.02)
        asphalt.addVertex(config.road_length_m, y, -0.02)
    for offset in (-config.lane_width_m / 2, config.lane_width_m / 2):
        edge = visualisation.newPolylineSketch(1.0, 1.0, 1.0, 0.8)
        edge.addVertex(0.0, config.lane_y_m + offset, 0.0)
        edge.addVertex(config.road_length_m, config.lane_y_m + offset, 0.0)
    stop_line = visualisation.newPolylineSketch(1.0, 1.0, 0.0, 1.0)
    for offset in (-config.lane_width_m / 2, config.lane_width_m / 2):
        stop_line.addVertex(config.light_x_m, config.lane_y_m + offset, 0.02)
    visualisation.activate()
    return visualisation


def run_simulation(
    config=None, *, steps=200, seed=12, output_csv=OUTPUT_CSV,
    no_visualisation=False, quiet=False, initial_positions=None,
    view_mode="detail",
):
    """Ejecuta GPU; solo publica el CSV y su configuración tras completarse."""
    config = config or TrafficConfig()
    config.validate()
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps debe ser un entero positivo.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed debe ser un entero no negativo.")
    if view_mode not in ("detail", "overview"):
        raise ValueError("view_mode debe ser detail u overview.")
    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(output_csv.stem + ".tmp.csv")
    partial = output_csv.with_name(output_csv.stem + "_partial.csv")
    metadata_path = output_csv.with_suffix(".metadata.json")
    metadata_temp = output_csv.with_suffix(".metadata.tmp.json")
    state = RunState(config)
    simulation = None
    visualisation = None
    temporary.unlink(missing_ok=True)
    metadata_temp.unlink(missing_ok=True)
    try:
        model, car, traffic_light = build_model(config, state, temporary, quiet)
        simulation = pyflamegpu.CUDASimulation(model)
        simulation.initialise(["traffic_model.py", "-s", str(steps), "-r", str(seed)])
        simulation.setPopulationData(create_initial_population(car, config, initial_positions))
        simulation.setPopulationData(create_traffic_light_population(traffic_light, config))
        if pyflamegpu.VISUALISATION and not no_visualisation:
            visualisation = configure_visualisation(simulation, config, view_mode)

        create_csv_file(temporary)
        simulation.simulate()
        # Se confirma el balance también después de aplicar la capa de muerte.
        final_population = pyflamegpu.AgentVector(car)
        simulation.getPopulationData(final_population)
        if len(final_population) != state.introduced - state.exited:
            raise RuntimeError("La eliminación GPU no coincide con las salidas contabilizadas.")
        metadata = {
            "model": "carretera_fisica_poblacion_dinamica",
            "config": asdict(config), "steps": steps, "seed": seed,
            "python_version": sys.version.split()[0],
            "pyflamegpu_version": getattr(pyflamegpu, "__version__", "consultar requirements.txt"),
            "backend": "FLAME GPU 2 CUDA", "front_coordinate": "x",
            "effective_light_x_m": gpu_float(config.light_x_m),
            "visualisation_view": view_mode,
            "units": {"position": "m", "speed": "m/s", "waiting_time": "s"},
            "assumptions": "Parámetros provisionales, sin calibración con datos reales.",
        }
        metadata_temp.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(output_csv)
        metadata_temp.replace(metadata_path)
        if not quiet:
            print(f"Simulación terminada correctamente. CSV: {output_csv}")
        if visualisation is not None:
            visualisation.join()
        return state
    except Exception:
        if temporary.exists():
            temporary.replace(partial)
            if not quiet:
                print(f"Datos diagnósticos conservados en: {partial}")
        raise
    finally:
        metadata_temp.unlink(missing_ok=True)
        if simulation is not None:
            pyflamegpu.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("-r", "--seed", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--initial-cars", type=int, default=20)
    parser.add_argument("--arrival-interval", type=float, default=6.0,
                        help="Segundos entre solicitudes; 0 desactiva las llegadas.")
    parser.add_argument("--time-step", type=float, default=1.0, help="Segundos por paso.")
    parser.add_argument("--red-seconds", type=float, default=10.0)
    parser.add_argument("--green-seconds", type=float, default=10.0)
    parser.add_argument("--no-visualisation", action="store_true")
    parser.add_argument("--view", choices=("detail", "overview"), default="detail",
                        help="Encuadre del visor: detalle del semáforo o carretera completa.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        config = TrafficConfig(
            initial_cars=args.initial_cars, arrival_interval_s=args.arrival_interval,
            time_step_s=args.time_step, red_duration_s=args.red_seconds,
            green_duration_s=args.green_seconds,
        )
        run_simulation(
            config, steps=args.steps, seed=args.seed, output_csv=args.output,
            no_visualisation=args.no_visualisation, quiet=args.quiet,
            view_mode=args.view,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
