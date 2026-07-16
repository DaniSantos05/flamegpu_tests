"""
Modelo V0.2 de tráfico con FLAME GPU 2.

Versión simplificada y corregida:
- Carretera recta.
- Coches como agentes.
- Semáforo en x = 50.
- El semáforo empieza en rojo.
- Si un coche está antes del semáforo y cerca de él, se para.
- Se usan capas explícitas para asegurar que car_move se ejecuta.
"""

import pyflamegpu
import sys


CAR_COUNT = 100

ROAD_LENGTH = 100.0
LANE_Y = 50.0

MAX_SPEED = 1.0

LIGHT_X = 50.0
STOP_DISTANCE = 20.0
SIGNAL_PERIOD = 10  # 10 pasos rojo, 10 pasos verde


car_move = r"""
FLAMEGPU_AGENT_FUNCTION(car_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float ROAD_LENGTH = FLAMEGPU->environment.getProperty<float>("road_length");
    const float MAX_SPEED = FLAMEGPU->environment.getProperty<float>("max_speed");

    const float LIGHT_X = FLAMEGPU->environment.getProperty<float>("light_x");
    const float STOP_DISTANCE = FLAMEGPU->environment.getProperty<float>("stop_distance");
    const int SIGNAL_PERIOD = FLAMEGPU->environment.getProperty<int>("signal_period");

    float x = FLAMEGPU->getVariable<float>("x");

    // Fase 0: rojo.
    // Fase 1: verde.
    const unsigned int step = FLAMEGPU->getStepCounter();
    const int signal_phase = (step / SIGNAL_PERIOD) % 2;
    const bool light_is_green = signal_phase == 1;

    bool red_light_stop = false;

    if (!light_is_green) {
        const bool before_light = x < LIGHT_X;
        const bool close_to_light = (LIGHT_X - x) <= STOP_DISTANCE;

        if (before_light && close_to_light) {
            red_light_stop = true;
        }
    }

    if (red_light_stop) {
        FLAMEGPU->setVariable<float>("speed", 0.0f);
        FLAMEGPU->setVariable<float>("stopped", 1.0f);
        FLAMEGPU->setVariable<float>("stopped_by_light", 1.0f);

        FLAMEGPU->setVariable<float>(
            "waiting_time",
            FLAMEGPU->getVariable<float>("waiting_time") + 1.0f
        );
    } else {
        FLAMEGPU->setVariable<float>("speed", MAX_SPEED);
        FLAMEGPU->setVariable<float>("stopped", 0.0f);
        FLAMEGPU->setVariable<float>("stopped_by_light", 0.0f);

        x += MAX_SPEED;

        if (x > ROAD_LENGTH) {
            x = 0.0f;
        }

        FLAMEGPU->setVariable<float>("x", x);
    }

    return flamegpu::ALIVE;
}
"""


class traffic_metrics(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()
        self.step = 0

    def run(self, FLAMEGPU):
        stopped_cars = FLAMEGPU.agent("Car").sumFloat("stopped")
        stopped_by_light = FLAMEGPU.agent("Car").sumFloat("stopped_by_light")
        total_waiting_time = FLAMEGPU.agent("Car").sumFloat("waiting_time")

        phase = (self.step // SIGNAL_PERIOD) % 2
        light_state = "ROJO" if phase == 0 else "VERDE"

        print(
            f"Paso {self.step}: "
            f"semáforo = {light_state}, "
            f"coches parados = {stopped_cars:.0f}, "
            f"por semáforo = {stopped_by_light:.0f}, "
            f"espera total = {total_waiting_time:.0f}"
        )

        self.step += 1


def main():
    model = pyflamegpu.ModelDescription("traffic_model_v02_simple")

    env = model.Environment()
    env.newPropertyFloat("road_length", ROAD_LENGTH)
    env.newPropertyFloat("max_speed", MAX_SPEED)
    env.newPropertyFloat("light_x", LIGHT_X)
    env.newPropertyFloat("stop_distance", STOP_DISTANCE)
    env.newPropertyInt("signal_period", SIGNAL_PERIOD)

    car = model.newAgent("Car")
    car.newVariableFloat("x")
    car.newVariableFloat("y")
    car.newVariableFloat("speed")
    car.newVariableFloat("waiting_time")
    car.newVariableFloat("stopped")
    car.newVariableFloat("stopped_by_light")

    move_fn = car.newRTCFunction("car_move", car_move)

    # Capa 1: mover coches.
    layer_move = model.newLayer()
    layer_move.addAgentFunction(move_fn)

    # Capa 2: calcular métricas después de mover coches.
    layer_metrics = model.newLayer()
    layer_metrics.addHostFunction(traffic_metrics())

    simulation = pyflamegpu.CUDASimulation(model)
    simulation.initialise(sys.argv)

    population = pyflamegpu.AgentVector(car, CAR_COUNT)

    for i, instance in enumerate(population):
        # Colocamos todos los coches cerca del semáforo.
        # Semáforo en x=50, coches entre x=30 y x=48 aprox.
        x = 30.0 + (i % 40) * 0.45

        instance.setVariableFloat("x", x)
        instance.setVariableFloat("y", LANE_Y)
        instance.setVariableFloat("speed", MAX_SPEED)
        instance.setVariableFloat("waiting_time", 0.0)
        instance.setVariableFloat("stopped", 0.0)
        instance.setVariableFloat("stopped_by_light", 0.0)

    simulation.setPopulationData(population)

    print("Iniciando simulación V0.2 simplificada con semáforo...")

    simulation.simulate()

    print("Simulación terminada correctamente.")

    pyflamegpu.cleanup()


if __name__ == "__main__":
    main()