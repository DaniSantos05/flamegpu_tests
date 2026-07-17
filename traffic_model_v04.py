"""
Modelo V0.4 de tráfico con FLAME GPU 2.

Mejoras respecto a V0.3:
- Carretera recta.
- Coches como agentes.
- Semáforo en x = 50.
- Detección de coche delante.
- Parada por semáforo rojo.
- Parada por coche delante.
- Métricas nuevas:
    1. coches parados;
    2. coches parados por semáforo;
    3. coches parados por coche delante;
    4. espera total;
    5. espera media;
    6. coches que cruzan el semáforo en cada paso;
    7. coches que han cruzado el semáforo en total;
    8. longitud aproximada de cola.
"""

import pyflamegpu
import sys


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

    // Reiniciamos esta variable en cada paso.
    FLAMEGPU->setVariable<float>("crossed_this_step", 0.0f);

    // Estado del semáforo.
    // Fase 0: rojo.
    // Fase 1: verde.
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
        // El coche se detiene.
        FLAMEGPU->setVariable<float>("speed", 0.0f);
        FLAMEGPU->setVariable<float>("stopped", 1.0f);

        // Damos prioridad al semáforo como motivo principal si está en zona de parada.
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
        // El coche avanza.
        FLAMEGPU->setVariable<float>("speed", MAX_SPEED);
        FLAMEGPU->setVariable<float>("stopped", 0.0f);
        FLAMEGPU->setVariable<float>("stopped_by_light", 0.0f);
        FLAMEGPU->setVariable<float>("stopped_by_car", 0.0f);

        const float old_x = x;
        x += MAX_SPEED;

        // Detectamos si el coche ha cruzado el semáforo en este paso.
        if (old_x < LIGHT_X && x >= LIGHT_X) {
            FLAMEGPU->setVariable<float>("crossed_this_step", 1.0f);

            FLAMEGPU->setVariable<float>(
                "total_crossings",
                FLAMEGPU->getVariable<float>("total_crossings") + 1.0f
            );
        }

        // Regla provisional: si llega al final, vuelve al inicio.
        if (x > ROAD_LENGTH) {
            x = 0.0f;
        }

        FLAMEGPU->setVariable<float>("x", x);
    }

    return flamegpu::ALIVE;
}
"""


# Función host: imprime métricas después de mover los coches.
class traffic_metrics(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()
        self.step = 0

    def run(self, FLAMEGPU):
        stopped_cars = FLAMEGPU.agent("Car").sumFloat("stopped")
        stopped_by_light = FLAMEGPU.agent("Car").sumFloat("stopped_by_light")
        stopped_by_car = FLAMEGPU.agent("Car").sumFloat("stopped_by_car")
        total_waiting_time = FLAMEGPU.agent("Car").sumFloat("waiting_time")

        crossed_this_step = FLAMEGPU.agent("Car").sumFloat("crossed_this_step")
        total_crossings = FLAMEGPU.agent("Car").sumFloat("total_crossings")

        average_waiting_time = total_waiting_time / CAR_COUNT
        queue_length = stopped_cars

        phase = (self.step // SIGNAL_PERIOD) % 2
        light_state = "ROJO" if phase == 0 else "VERDE"

        # Imprimimos cada 5 pasos para que sea legible.
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


def main():
    # Crear modelo
    model = pyflamegpu.ModelDescription("traffic_model_v04")

    # Propiedades globales
    env = model.Environment()
    env.newPropertyFloat("road_length", ROAD_LENGTH)
    env.newPropertyFloat("max_speed", MAX_SPEED)
    env.newPropertyFloat("safe_distance", SAFE_DISTANCE)
    env.newPropertyFloat("light_x", LIGHT_X)
    env.newPropertyFloat("stop_distance", STOP_DISTANCE)
    env.newPropertyInt("signal_period", SIGNAL_PERIOD)

    # Mensaje espacial para comunicar posiciones de coches
    message = model.newMessageSpatial2D("car_location")
    message.newVariableID("id")
    message.setRadius(SAFE_DISTANCE)
    message.setMin(0.0, 0.0)
    message.setMax(ROAD_LENGTH, ROAD_WIDTH)

    # Agente Car
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

    # Función 1: publicar posición
    output_fn = car.newRTCFunction("car_output_position", car_output_position)
    output_fn.setMessageOutput("car_location")

    # Función 2: moverse o detenerse
    move_fn = car.newRTCFunction("car_move", car_move)
    move_fn.setMessageInput("car_location")

    # Capas explícitas de ejecución.
    # Capa 1: los coches publican su posición.
    layer_output = model.newLayer()
    layer_output.addAgentFunction(output_fn)

    # Capa 2: los coches leen posiciones y deciden si avanzan o paran.
    layer_move = model.newLayer()
    layer_move.addAgentFunction(move_fn)

    # Capa 3: se calculan métricas.
    layer_metrics = model.newLayer()
    layer_metrics.addHostFunction(traffic_metrics())

    # Crear simulación CUDA
    simulation = pyflamegpu.CUDASimulation(model)
    simulation.initialise(sys.argv)

    # Crear población inicial
    population = pyflamegpu.AgentVector(car, CAR_COUNT)

    for i, instance in enumerate(population):
        # Coches antes del semáforo y bastante juntos.
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

    simulation.setPopulationData(population)

    print("Iniciando simulación V0.4: métricas de flujo, cola y espera...")

    simulation.simulate()

    print("Simulación terminada correctamente.")

    pyflamegpu.cleanup()


if __name__ == "__main__":
    main()