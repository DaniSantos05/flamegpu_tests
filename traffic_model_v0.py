"""
Modelo V0.1 de tráfico con FLAME GPU 2.

Primera versión propia:
- Coches como agentes.
- Movimiento en una carretera recta.
- Cada coche publica su posición.
- Cada coche detecta si tiene otro coche cerca delante.
- Se calculan métricas básicas: coches parados y tiempo total de espera.
"""

import pyflamegpu
import sys
import random


# Parámetros básicos del modelo
CAR_COUNT = 100
ROAD_LENGTH = 100.0
ROAD_WIDTH = 100.0
LANE_Y = 50.0
MAX_SPEED = 1.0
SAFE_DISTANCE = 3.0


# Función de agente: cada coche publica su ID y su posición en un mensaje espacial 2D.
car_output_position = r"""
FLAMEGPU_AGENT_FUNCTION(car_output_position, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());

    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));

    return flamegpu::ALIVE;
}
"""


# Función de agente: cada coche lee coches cercanos y decide si avanza o se detiene.
car_move = r"""
FLAMEGPU_AGENT_FUNCTION(car_move, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const flamegpu::id_t ID = FLAMEGPU->getID();

    const float ROAD_LENGTH = FLAMEGPU->environment.getProperty<float>("road_length");
    const float MAX_SPEED = FLAMEGPU->environment.getProperty<float>("max_speed");
    const float SAFE_DISTANCE = FLAMEGPU->environment.getProperty<float>("safe_distance");

    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");

    bool car_ahead = false;

    // Recorre coches cercanos a la posición actual.
    for (const auto &message : FLAMEGPU->message_in(x, y)) {
        if (message.getVariable<flamegpu::id_t>("id") != ID) {
            const float other_x = message.getVariable<float>("x");
            const float other_y = message.getVariable<float>("y");

            // En esta primera versión solo hay un carril horizontal.
            const bool same_lane = fabsf(other_y - y) < 0.1f;
            const bool ahead = other_x > x;
            const bool too_close = (other_x - x) < SAFE_DISTANCE;

            if (same_lane && ahead && too_close) {
                car_ahead = true;
            }
        }
    }

    if (car_ahead) {
        // Si hay un coche muy cerca delante, el coche se detiene.
        FLAMEGPU->setVariable<float>("speed", 0.0f);
        FLAMEGPU->setVariable<float>("stopped", 1.0f);

        // Aumenta el tiempo de espera del coche.
        FLAMEGPU->setVariable<float>(
            "waiting_time",
            FLAMEGPU->getVariable<float>("waiting_time") + 1.0f
        );
    } else {
        // Si no hay coche delante, avanza.
        FLAMEGPU->setVariable<float>("speed", MAX_SPEED);
        FLAMEGPU->setVariable<float>("stopped", 0.0f);

        x += MAX_SPEED;

        // Si el coche llega al final de la carretera, vuelve al inicio.
        // Esto es provisional para mantener siempre el mismo número de coches.
        if (x > ROAD_LENGTH) {
            x = 0.0f;
        }

        FLAMEGPU->setVariable<float>("x", x);
    }

    return flamegpu::ALIVE;
}
"""


# Función host: calcula métricas globales en cada paso de simulación.
class traffic_metrics(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()
        self.step = 0

    def run(self, FLAMEGPU):
        stopped_cars = FLAMEGPU.agent("Car").sumFloat("stopped")
        total_waiting_time = FLAMEGPU.agent("Car").sumFloat("waiting_time")

        # Para no imprimir demasiado, mostramos métricas cada 10 pasos.
        if self.step % 10 == 0:
            print(
                f"Paso {self.step}: "
                f"coches parados = {stopped_cars:.0f}, "
                f"espera total = {total_waiting_time:.0f}"
            )

        self.step += 1


def main():
    # Crear el modelo
    model = pyflamegpu.ModelDescription("traffic_model_v0")

    # Propiedades globales del entorno
    env = model.Environment()
    env.newPropertyFloat("road_length", ROAD_LENGTH)
    env.newPropertyFloat("max_speed", MAX_SPEED)
    env.newPropertyFloat("safe_distance", SAFE_DISTANCE)

    # Mensaje espacial para comunicar posiciones de coches
    message = model.newMessageSpatial2D("car_location")
    message.newVariableID("id")
    message.setRadius(SAFE_DISTANCE)
    message.setMin(0.0, 0.0)
    message.setMax(ROAD_LENGTH, ROAD_WIDTH)

    # Definir agente Car
    car = model.newAgent("Car")
    car.newVariableFloat("x")
    car.newVariableFloat("y")
    car.newVariableFloat("speed")
    car.newVariableFloat("waiting_time")
    car.newVariableFloat("stopped")

    # Asociar funciones al agente
    output_fn = car.newRTCFunction("car_output_position", car_output_position)
    output_fn.setMessageOutput("car_location")

    move_fn = car.newRTCFunction("car_move", car_move)
    move_fn.setMessageInput("car_location")

    # Primero los coches publican su posición, después se mueven.
    move_fn.dependsOn(output_fn)
    model.addExecutionRoot(output_fn)

    # Añadir métricas de tráfico en cada paso
    metrics_fn = traffic_metrics()
    model.addStepFunction(metrics_fn)

    # Crear simulación CUDA
    simulation = pyflamegpu.CUDASimulation(model)
    simulation.initialise(sys.argv)

    # Crear población inicial de coches
    random.seed(simulation.SimulationConfig().random_seed)

    population = pyflamegpu.AgentVector(car, CAR_COUNT)

    for i, instance in enumerate(population):
        instance.setVariableFloat("x", random.uniform(0.0, ROAD_LENGTH))
        instance.setVariableFloat("y", LANE_Y)
        instance.setVariableFloat("speed", MAX_SPEED)
        instance.setVariableFloat("waiting_time", 0.0)
        instance.setVariableFloat("stopped", 0.0)

    simulation.setPopulationData(population)

    print("Iniciando simulación del modelo de tráfico V0...")

    # Ejecutar simulación
    simulation.simulate()

    print("Simulación terminada correctamente.")

    pyflamegpu.cleanup()


if __name__ == "__main__":
    main()