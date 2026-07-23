"""
Modelo V0.5 de tráfico con FLAME GPU 2.

Mejoras respecto a V0.4:
- Mantiene la carretera recta con semáforo.
- Mantiene la detección de coche delante.
- Mantiene las métricas de cola, espera y cruces.
- Guarda las métricas en un archivo CSV para poder analizarlas después.
- Evita borrar el último CSV válido si CUDA o la simulación fallan.
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
OUTPUT_CSV = Path(__file__).with_name("traffic_metrics_v05.csv")
TEMP_CSV = Path(__file__).with_name("traffic_metrics_v05.tmp.csv")
PARTIAL_CSV = Path(__file__).with_name("traffic_metrics_v05_partial.csv")


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
    model = pyflamegpu.ModelDescription("traffic_model_v05")

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

    output_fn = car.newRTCFunction("car_output_position", car_output_position)
    output_fn.setMessageOutput("car_location")

    move_fn = car.newRTCFunction("car_move", car_move)
    move_fn.setMessageInput("car_location")

    layer_output = model.newLayer()
    layer_output.addAgentFunction(output_fn)

    layer_move = model.newLayer()
    layer_move.addAgentFunction(move_fn)

    layer_metrics = model.newLayer()
    layer_metrics.addHostFunction(TrafficMetrics(TEMP_CSV))

    return model, car


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


def main() -> None:
    simulation = None

    # El CSV definitivo no se toca hasta que la simulación haya terminado bien.
    TEMP_CSV.unlink(missing_ok=True)
    PARTIAL_CSV.unlink(missing_ok=True)

    try:
        model, car = build_model()

        simulation = pyflamegpu.CUDASimulation(model)
        simulation.initialise(sys.argv)

        population = create_initial_population(car)
        simulation.setPopulationData(population)

        # Se crea el temporal solo después de que CUDA y la simulación se inicialicen.
        create_csv_file(TEMP_CSV)

        print("Iniciando simulación V0.5: métricas guardadas en CSV...")
        simulation.simulate()

        # Sustitución atómica: el resultado anterior se conserva hasta este punto.
        TEMP_CSV.replace(OUTPUT_CSV)

        print("Simulación terminada correctamente.")
        print(f"Archivo CSV creado: {OUTPUT_CSV}")

    except Exception:
        # Si había datos parciales, se guardan aparte para facilitar el diagnóstico.
        if TEMP_CSV.exists():
            TEMP_CSV.replace(PARTIAL_CSV)
            print(f"Se han conservado datos parciales en: {PARTIAL_CSV}")
        raise

    finally:
        if simulation is not None:
            pyflamegpu.cleanup()


if __name__ == "__main__":
    main()
