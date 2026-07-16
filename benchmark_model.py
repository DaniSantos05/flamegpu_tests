"""
Modelo Circles de FLAME GPU 2 en Python.

Este ejemplo sirve para aprender la estructura básica de una simulación:
modelo, agentes, mensajes, funciones de agente, inicialización y ejecución.
"""

import pyflamegpu
import sys, random, math


# Función de agente: cada agente publica su ID y su posición en un mensaje espacial 3D.
# En el TFG del cruce, esto sería parecido a que cada coche publique dónde está.
output_message = r"""
FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // Guarda el ID del agente en el mensaje.
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());

    // Publica la posición actual del agente.
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));

    return flamegpu::ALIVE;
}
"""


# Función de agente: cada agente lee los mensajes de vecinos cercanos y actualiza su posición.
# Es la función principal del comportamiento del modelo.
move = r"""
FLAMEGPU_AGENT_FUNCTION(move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    // ID del agente actual, usado para no procesarse a sí mismo.
    const flamegpu::id_t ID = FLAMEGPU->getID();

    // Parámetros globales del modelo.
    const float REPULSE_FACTOR = FLAMEGPU->environment.getProperty<float>("repulse");
    const float RADIUS = FLAMEGPU->message_in.radius();

    // Movimiento acumulado que se calculará a partir de los vecinos.
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;

    // Posición actual del agente.
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");

    int count = 0;

    // Recorre los mensajes de agentes cercanos.
    for (const auto &message : FLAMEGPU->message_in(x1, y1, z1)) {
        if (message.getVariable<flamegpu::id_t>("id") != ID) {
            // Posición del vecino.
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");
            const float z2 = message.getVariable<float>("z");

            // Vector desde el agente actual hasta el vecino.
            float x21 = x2 - x1;
            float y21 = y2 - y1;
            float z21 = z2 - z1;

            const float separation = cbrt(x21*x21 + y21*y21 + z21*z21);

            if (separation < RADIUS && separation > 0.0f) {
                // Calcula la influencia del vecino sobre el movimiento.
                float k = sinf((separation / RADIUS) * 3.141f * -2.0f) * REPULSE_FACTOR;

                // Normaliza el vector de dirección.
                x21 /= separation;
                y21 /= separation;
                z21 /= separation;

                // Acumula el desplazamiento producido por este vecino.
                fx += k * x21;
                fy += k * y21;
                fz += k * z21;

                count++;
            }
        }
    }

    // Calcula el movimiento medio.
    fx /= count > 0 ? count : 1;
    fy /= count > 0 ? count : 1;
    fz /= count > 0 ? count : 1;

    // Actualiza la posición del agente.
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("z", z1 + fz);

    // Guarda cuánto se ha movido el agente para validar el modelo.
    FLAMEGPU->setVariable<float>("drift", cbrt(fx*fx + fy*fy + fz*fz));

    return flamegpu::ALIVE;
}
"""


# Función host que se ejecuta en cada paso para comprobar la evolución del modelo.
class step_validation(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()

        # Valores usados para comparar el movimiento total entre pasos.
        self.prevTotalDrift = 3.402823e+38
        self.driftDropped = 0
        self.driftIncreased = 0

    def run(self, FLAMEGPU):
        # Suma el drift de todos los agentes Circle.
        totalDrift = FLAMEGPU.agent("Circle").sumFloat("drift")

        # Comprueba si el movimiento total baja respecto al paso anterior.
        if totalDrift <= self.prevTotalDrift:
            self.driftDropped += 1
        else:
            self.driftIncreased += 1

        self.prevTotalDrift = totalDrift

        # Imprime el porcentaje de pasos en los que el drift ha evolucionado correctamente.
        print("{:.2f} Drift correct".format(
            100 * self.driftDropped / float(self.driftDropped + self.driftIncreased)
        ))


# Raíz cúbica sin usar NumPy.
def cbrt(x):
    root = abs(x) ** (1 / 3)
    return root if x >= 0 else -root


# Número de agentes del modelo.
AGENT_COUNT = 1048576

# Tamaño del entorno 3D.
ENV_MAX = math.floor(cbrt(AGENT_COUNT))

# Radio dentro del cual un agente detecta vecinos.
RADIUS = 2.0

# Activa o desactiva la visualización de la cuadrícula espacial.
VISUALISE_COMMUNICATION_GRID = False


def main():
    # Crea el modelo FLAME GPU.
    model = pyflamegpu.ModelDescription("example_model")

    # Define propiedades globales del entorno.
    env = model.Environment()
    env.newPropertyFloat("repulse", 0.05)

    # Define el mensaje espacial que usaremos para comunicar posiciones.
    message = model.newMessageSpatial3D("location")
    message.newVariableID("id")

    # Configura el espacio donde se moverán los agentes.
    message.setRadius(RADIUS)
    message.setMin(0, 0, 0)
    message.setMax(ENV_MAX, ENV_MAX, ENV_MAX)

    # Define el tipo de agente del modelo.
    agent = model.newAgent("Circle")
    agent.newVariableInt("id")
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
    agent.newVariableFloat("z")
    agent.newVariableFloat("drift")

    # Asocia las funciones de comportamiento al agente.
    output_message_description = agent.newRTCFunction("output_message", output_message)
    output_message_description.setMessageOutput("location")

    move_description = agent.newRTCFunction("move", move)
    move_description.setMessageInput("location")

    # Primero se publican posiciones y después se calcula el movimiento.
    move_description.dependsOn(output_message_description)
    model.addExecutionRoot(output_message_description)

    # Añade la función de validación en cada paso.
    # step_validation_fn = step_validation()
    # model.addStepFunction(step_validation_fn)

    # Crea la simulación CUDA.
    simulation = pyflamegpu.CUDASimulation(model)

    # Inicializa la simulación con los argumentos de terminal.
    simulation.initialise(sys.argv)

    # Si no se proporciona archivo inicial, crea agentes aleatoriamente.
    if not simulation.SimulationConfig().input_file:
        random.seed(simulation.SimulationConfig().random_seed)

        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)

        for i, instance in enumerate(population):
            instance.setVariableFloat("x", random.uniform(0, ENV_MAX))
            instance.setVariableFloat("y", random.uniform(0, ENV_MAX))
            instance.setVariableFloat("z", random.uniform(0, ENV_MAX))

        simulation.setPopulationData(population)

    # Ejecuta la simulación.
    simulation.simulate()

    # Libera recursos internos de FLAME GPU.
    pyflamegpu.cleanup()


if __name__ == "__main__":
    main()