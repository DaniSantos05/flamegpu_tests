"""
Modelo V0.6 de tráfico con FLAME GPU 2.

Mejoras respecto a V0.5:
- Mantiene la carretera recta con semáforo.
- Mantiene la detección de coche delantero mediante MessageSpatial2D.
- Mantiene las métricas de cola, espera y cruces.
- Mantiene la exportación robusta de métricas a CSV.
- Incorpora el visor oficial de FLAME GPU.
- Representa visualmente los coches sobre la carretera.
- Colorea los coches según su estado:
    rojo = coche detenido
    verde = coche en movimiento
- Incorpora un agente TrafficLight para representar visualmente el semáforo.
- El semáforo cambia visualmente entre rojo y verde de forma sincronizada con la lógica de la simulación.
- Mantiene la línea de parada y los límites de la carretera como elementos visuales.
"""

import csv
import sys
from pathlib import Path

import pyflamegpu


# Parámetros básicos
CAR_COUNT = 100

ROAD_LENGTH = 100.0
ROAD_WIDTH = 100.0
LANE_Y = 50.0

MAX_SPEED = 1.0
SAFE_DISTANCE = 2.0

# Semáforo
LIGHT_X = 50.0
STOP_DISTANCE = 20.0
SIGNAL_PERIOD = 10  # 10 pasos rojo, 10 pasos verde

# Archivos de salida
OUTPUT_CSV = Path(__file__).with_name("traffic_metrics_v06.csv")
TEMP_CSV = Path(__file__).with_name("traffic_metrics_v06.tmp.csv")
PARTIAL_CSV = Path(__file__).with_name("traffic_metrics_v06_partial.csv")


# Función de agente: cada coche publica su posición.
car_output_position = r"""
FLAMEGPU_AGENT_FUNCTION(car_output_position, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());

    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));

    return flamegpu::ALIVE;
}
"""


# Función de agente: cada coche decide si se mueve o se detiene.
car_move = r"""
FLAMEGPU_AGENT_FUNCTION(car_move, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const flamegpu::id_t ID = FLAMEGPU->getID();

    const float ROAD_LENGTH = FLAMEGPU->environment.getProperty<float>("road_length");
    const float MAX_SPEED = FLAMEGPU->environment.getProperty<float>("max_speed");
    const float SAFE_DISTANCE = FLAMEGPU->environment.getProperty<float>("safe_distance");

    const float LIGHT_X = FLAMEGPU->environment.getProperty<float>("light_x");
    const float STOP_DISTANCE = FLAMEGPU->environment.getProperty<float>("stop_distance");
    const int SIGNAL_PERIOD = FLAMEGPU->environment.getProperty<int>("signal_period");

    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");

    bool car_ahead = false;
    bool red_light_stop = false;

    FLAMEGPU->setVariable<float>("crossed_this_step", 0.0f);

    // Fase 0: rojo. Fase 1: verde.
    const unsigned int step = FLAMEGPU->getStepCounter();
    const int signal_phase = (step / SIGNAL_PERIOD) % 2;
    const bool light_is_green = signal_phase == 1;

    // Detectar coches delante usando mensajes de posición.
    for (const auto &message : FLAMEGPU->message_in(x, y)) {
        if (message.getVariable<flamegpu::id_t>("id") != ID) {
            const float other_x = message.getVariable<float>("x");
            const float other_y = message.getVariable<float>("y");

            const bool same_lane = fabsf(other_y - y) < 0.1f;
            const float distance = other_x - x;
            const bool ahead = distance > 0.0f;
            const bool too_close = distance < SAFE_DISTANCE;

            if (same_lane && ahead && too_close) {
                car_ahead = true;
            }
        }
    }

    // Detectar semáforo rojo.
    if (!light_is_green) {
        const bool before_light = x < LIGHT_X;
        const bool close_to_light = (LIGHT_X - x) <= STOP_DISTANCE;

        if (before_light && close_to_light) {
            red_light_stop = true;
        }
    }

    if (red_light_stop || car_ahead) {
        FLAMEGPU->setVariable<float>("speed", 0.0f);
        FLAMEGPU->setVariable<float>("stopped", 1.0f);

        // El semáforo tiene prioridad como motivo si el coche está en su zona de parada.
        if (red_light_stop) {
            FLAMEGPU->setVariable<float>("stopped_by_light", 1.0f);
            FLAMEGPU->setVariable<float>("stopped_by_car", 0.0f);
        } else {
            FLAMEGPU->setVariable<float>("stopped_by_light", 0.0f);
            FLAMEGPU->setVariable<float>("stopped_by_car", 1.0f);
        }

        FLAMEGPU->setVariable<float>(
            "waiting_time",
            FLAMEGPU->getVariable<float>("waiting_time") + 1.0f
        );
    } else {
        FLAMEGPU->setVariable<float>("speed", MAX_SPEED);
        FLAMEGPU->setVariable<float>("stopped", 0.0f);
        FLAMEGPU->setVariable<float>("stopped_by_light", 0.0f);
        FLAMEGPU->setVariable<float>("stopped_by_car", 0.0f);

        const float old_x = x;
        x += MAX_SPEED;

        if (old_x < LIGHT_X && x >= LIGHT_X) {
            FLAMEGPU->setVariable<float>("crossed_this_step", 1.0f);
            FLAMEGPU->setVariable<float>(
                "total_crossings",
                FLAMEGPU->getVariable<float>("total_crossings") + 1.0f
            );
        }

        // Regla provisional para mantener constante la población.
        if (x > ROAD_LENGTH) {
            x = 0.0f;
        }

        FLAMEGPU->setVariable<float>("x", x);
    }

    return flamegpu::ALIVE;
}
"""

# Función de agente: actualiza el estado visual del semáforo.
traffic_light_update = r"""
FLAMEGPU_AGENT_FUNCTION(traffic_light_update, flamegpu::MessageNone, flamegpu::MessageNone) {

    const int SIGNAL_PERIOD =
        FLAMEGPU->environment.getProperty<int>("signal_period");

    const unsigned int step = FLAMEGPU->getStepCounter();

    const int signal_phase = (step / SIGNAL_PERIOD) % 2;

    // 0.0 = rojo
    // 1.0 = verde
    if (signal_phase == 0) {
        FLAMEGPU->setVariable<float>("light_state", 0.0f);
    } else {
        FLAMEGPU->setVariable<float>("light_state", 1.0f);
    }

    return flamegpu::ALIVE;
}
"""


class TrafficMetrics(pyflamegpu.HostFunction):
    """Calcula, muestra y guarda las métricas después de actualizar los coches."""

    def __init__(self, csv_path: Path):
        super().__init__()
        self.step = 0
        self.csv_path = csv_path

    def run(self, flamegpu):
        stopped_cars = flamegpu.agent("Car").sumFloat("stopped")
        stopped_by_light = flamegpu.agent("Car").sumFloat("stopped_by_light")
        stopped_by_car = flamegpu.agent("Car").sumFloat("stopped_by_car")
        total_waiting_time = flamegpu.agent("Car").sumFloat("waiting_time")
        crossed_this_step = flamegpu.agent("Car").sumFloat("crossed_this_step")
        total_crossings = flamegpu.agent("Car").sumFloat("total_crossings")

        average_waiting_time = total_waiting_time / CAR_COUNT
        queue_length = stopped_cars

        phase = (self.step // SIGNAL_PERIOD) % 2
        light_state = "ROJO" if phase == 0 else "VERDE"

        with self.csv_path.open(mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow([
                self.step,
                light_state,
                int(stopped_cars),
                int(stopped_by_light),
                int(stopped_by_car),
                int(queue_length),
                int(crossed_this_step),
                int(total_crossings),
                int(total_waiting_time),
                f"{average_waiting_time:.2f}".replace(".", ","),
            ])

        if self.step % 5 == 0:
            print(
                f"Paso {self.step}: "
                f"semáforo = {light_state}, "
                f"parados = {stopped_cars:.0f}, "
                f"por semáforo = {stopped_by_light:.0f}, "
                f"por coche delante = {stopped_by_car:.0f}, "
                f"cola aprox = {queue_length:.0f}, "
                f"cruzan ahora = {crossed_this_step:.0f}, "
                f"cruces totales = {total_crossings:.0f}, "
                f"espera media = {average_waiting_time:.2f}"
            )

        self.step += 1


def create_csv_file(csv_path: Path) -> None:
    """Crea un CSV nuevo con la cabecera, sin tocar el último resultado válido."""
    with csv_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "paso",
            "estado_semaforo",
            "coches_parados",
            "parados_por_semaforo",
            "parados_por_coche_delante",
            "cola_aproximada",
            "coches_cruzan_este_paso",
            "cruces_totales",
            "espera_total",
            "espera_media",
        ])


def build_model():
    """Construye la descripción del modelo y devuelve el modelo y el agente Car."""
    model = pyflamegpu.ModelDescription("traffic_model_v06")

    env = model.Environment()
    env.newPropertyFloat("road_length", ROAD_LENGTH)
    env.newPropertyFloat("max_speed", MAX_SPEED)
    env.newPropertyFloat("safe_distance", SAFE_DISTANCE)
    env.newPropertyFloat("light_x", LIGHT_X)
    env.newPropertyFloat("stop_distance", STOP_DISTANCE)
    env.newPropertyInt("signal_period", SIGNAL_PERIOD)

    message = model.newMessageSpatial2D("car_location")
    message.newVariableID("id")
    message.setRadius(SAFE_DISTANCE)
    message.setMin(0.0, 0.0)
    message.setMax(ROAD_LENGTH, ROAD_WIDTH)

    car = model.newAgent("Car")
    car.newVariableFloat("x")
    car.newVariableFloat("y")
    car.newVariableFloat("speed")
    car.newVariableFloat("waiting_time")
    car.newVariableFloat("stopped")
    car.newVariableFloat("stopped_by_light")
    car.newVariableFloat("stopped_by_car")
    car.newVariableFloat("crossed_this_step")
    car.newVariableFloat("total_crossings")


        # ---------------------------------------------------------
    # AGENTE VISUAL DEL SEMÁFORO
    # ---------------------------------------------------------

    traffic_light = model.newAgent("TrafficLight")

    # Posición del semáforo en el visor.
    traffic_light.newVariableFloat("x")
    traffic_light.newVariableFloat("y")
    traffic_light.newVariableFloat("z")

    # 0.0 = rojo
    # 1.0 = verde
    traffic_light.newVariableFloat("light_state")

    # Función que actualiza su estado en cada paso.
    light_fn = traffic_light.newRTCFunction(
        "traffic_light_update",
        traffic_light_update
    )

    output_fn = car.newRTCFunction("car_output_position", car_output_position)
    output_fn.setMessageOutput("car_location")

    move_fn = car.newRTCFunction("car_move", car_move)
    move_fn.setMessageInput("car_location")

    layer_output = model.newLayer()
    layer_output.addAgentFunction(output_fn)

    layer_move = model.newLayer()
    layer_move.addAgentFunction(move_fn)

    # Actualización del semáforo visual.
    layer_light = model.newLayer()
    layer_light.addAgentFunction(light_fn)

    layer_metrics = model.newLayer()
    layer_metrics.addHostFunction(TrafficMetrics(TEMP_CSV))

    return model, car, traffic_light


def create_initial_population(car):
    """Crea la población inicial utilizada por el prototipo."""
    population = pyflamegpu.AgentVector(car, CAR_COUNT)

    for i, instance in enumerate(population):
        x = 5.0 + (i % 80) * 0.55

        instance.setVariableFloat("x", x)
        instance.setVariableFloat("y", LANE_Y)
        instance.setVariableFloat("speed", MAX_SPEED)
        instance.setVariableFloat("waiting_time", 0.0)
        instance.setVariableFloat("stopped", 0.0)
        instance.setVariableFloat("stopped_by_light", 0.0)
        instance.setVariableFloat("stopped_by_car", 0.0)
        instance.setVariableFloat("crossed_this_step", 0.0)
        instance.setVariableFloat("total_crossings", 0.0)

    return population


def create_traffic_light_population(traffic_light):
    """Crea un único semáforo para la representación visual."""

    population = pyflamegpu.AgentVector(traffic_light, 1)

    for light in population:

        # Lo situamos justo en la posición x del semáforo,
        # ligeramente por encima de la carretera.
        light.setVariableFloat("x", LIGHT_X)
        light.setVariableFloat("y", LANE_Y + 6.0)
        light.setVariableFloat("z", 0.0)

        # La simulación comienza en rojo.
        light.setVariableFloat("light_state", 0.0)

    return population


def main() -> None:
    simulation = None
    visualisation = None

    # El CSV definitivo no se toca hasta que la simulación haya terminado bien.
    TEMP_CSV.unlink(missing_ok=True)
    PARTIAL_CSV.unlink(missing_ok=True)

    try:
        model, car, traffic_light = build_model()

        simulation = pyflamegpu.CUDASimulation(model)
        simulation.initialise(sys.argv)

        population = create_initial_population(car)
        simulation.setPopulationData(population)

        # Creo e incorporo el semáforo visual.
        light_population = create_traffic_light_population(traffic_light)
        simulation.setPopulationData(light_population)

        # ---------------------------------------------------------
        # VISUALIZACIÓN DEL MODELO DE TRÁFICO V0.6
        # ---------------------------------------------------------
        if pyflamegpu.VISUALISATION:

            # Obtengo el visor asociado a esta simulación.
            visualisation = simulation.getVisualisation()

            # Título y dimensiones de la ventana.
            visualisation.setWindowTitle("Traffic model V0.6")
            visualisation.setWindowDimensions(1280, 720)

            # Fondo oscuro.
            visualisation.setClearColor(0.05, 0.05, 0.05)

            # -----------------------------------------------------
            # CÁMARA
            # -----------------------------------------------------

            # La cámara mira al centro de la carretera.
            visualisation.setInitialCameraTarget(
                ROAD_LENGTH / 2.0,
                LANE_Y,
                0.0
            )

            # Situamos la cámara encima de la carretera.
            visualisation.setInitialCameraLocation(
                ROAD_LENGTH / 2.0,
                LANE_Y,
                120.0
            )

            visualisation.setCameraSpeed(0.05)

            # Limito la simulación para poder observar el movimiento.
            visualisation.setSimulationSpeed(10)

            # -----------------------------------------------------
            # COCHES
            # -----------------------------------------------------

            # Incorporo los agentes Car al visor.
            car_vis = visualisation.addAgent("Car")

            # Cada coche se representa mediante un cubo.
            car_vis.setModel(pyflamegpu.CUBE)

            # Lo deformo para que tenga una forma rectangular.
            car_vis.setModelScale(1.2, 0.6, 0.3)

            # Por ahora todos los coches serán azules.
            car_vis.setColor(
                pyflamegpu.HSVInterpolation.REDGREEN(
                    "speed",
                    0.0,
                    MAX_SPEED
                )
            )
            # -----------------------------------------------------
            # SEMÁFORO VISUAL
            # -----------------------------------------------------

            light_vis = visualisation.addAgent("TrafficLight")

            # Representamos el semáforo mediante una esfera.
            light_vis.setModel(pyflamegpu.SPHERE)

            # Lo hacemos suficientemente grande para verlo.
            light_vis.setModelScale(2.0)

            # state = 0.0 -> rojo
            # state = 1.0 -> verde
            light_vis.setColor(
                pyflamegpu.HSVInterpolation.REDGREEN(
                    "light_state",
                    0.0,
                    1.0
                )
            )




            # -----------------------------------------------------
            # CARRETERA
            # -----------------------------------------------------

            # Línea superior de la carretera.
            road_upper = visualisation.newPolylineSketch(
                1.0, 1.0, 1.0, 0.8
            )

            road_upper.addVertex(
                0.0,
                LANE_Y + 3.0,
                0.0
            )

            road_upper.addVertex(
                ROAD_LENGTH,
                LANE_Y + 3.0,
                0.0
            )

            # Línea inferior de la carretera.
            road_lower = visualisation.newPolylineSketch(
                1.0, 1.0, 1.0, 0.8
            )

            road_lower.addVertex(
                0.0,
                LANE_Y - 3.0,
                0.0
            )

            road_lower.addVertex(
                ROAD_LENGTH,
                LANE_Y - 3.0,
                0.0
            )

            # -----------------------------------------------------
            # LÍNEA DEL SEMÁFORO
            # -----------------------------------------------------

            # Por ahora dibujo una línea amarilla que indica
            # la posición del semáforo.
            stop_line = visualisation.newPolylineSketch(
                1.0, 1.0, 0.0, 1.0
            )

            stop_line.addVertex(
                LIGHT_X,
                LANE_Y - 3.0,
                0.0
            )

            stop_line.addVertex(
                LIGHT_X,
                LANE_Y + 3.0,
                0.0
            )

            # Abro la ventana del visor.
            visualisation.activate()

        # ---------------------------------------------------------
        # EJECUCIÓN NORMAL DE LA SIMULACIÓN
        # ---------------------------------------------------------

        create_csv_file(TEMP_CSV)

        print("Iniciando simulación V0.6 con visualización...")

        simulation.simulate()

        # Solo sustituyo el CSV definitivo si la ejecución
        # ha terminado correctamente.
        TEMP_CSV.replace(OUTPUT_CSV)

        print("Simulación terminada correctamente.")
        print(f"Archivo CSV creado: {OUTPUT_CSV}")

        # Mantengo el visor abierto una vez finalizada la simulación.
        if visualisation is not None:
            visualisation.join()

    except Exception:

        # Si había datos parciales, los conservo.
        if TEMP_CSV.exists():
            TEMP_CSV.replace(PARTIAL_CSV)
            print(f"Se han conservado datos parciales en: {PARTIAL_CSV}")

        raise

    finally:

        if simulation is not None:
            pyflamegpu.cleanup()


if __name__ == "__main__":
    main()
