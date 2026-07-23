# Prototipo de tráfico con FLAME GPU 2

Repositorio de trabajo para el TFG sobre simulación basada en agentes de tráfico urbano mediante FLAME GPU 2.

El proyecto se encuentra en fase de **prototipado en Python**. El objetivo actual es comprender y validar la arquitectura del modelo antes de ampliar el escenario a un cruce completo y estudiar su migración final a C++.

## Estado actual

La versión más avanzada es `traffic_model_v05.py`. Representa una carretera recta con:

- 100 coches modelados como agentes.
- Un semáforo situado en una posición fija.
- Detección de vehículos próximos mediante mensajes espaciales.
- Parada por semáforo rojo.
- Parada por vehículo delantero.
- Formación y liberación progresiva de cola.
- Métricas de espera, cola y cruces.
- Exportación de resultados a CSV.
- Generación automática de gráficas PNG.

## Arquitectura

### Parte gestionada desde Python / host

Python se encarga de:

- Crear el `ModelDescription`.
- Definir las propiedades globales del entorno.
- Declarar el agente `Car` y sus variables.
- Configurar el mensaje espacial `car_location`.
- Construir las capas y su orden de ejecución.
- Inicializar la población de vehículos.
- Lanzar la simulación CUDA.
- Calcular métricas globales mediante una función host.
- Guardar métricas en CSV.
- Generar las gráficas de resultados.

### Parte ejecutada en GPU

Las funciones RTC se compilan en tiempo de ejecución y se ejecutan en paralelo para cada agente:

- `car_output_position`: publica el identificador y la posición del coche.
- `car_move`: consulta vehículos cercanos, comprueba el semáforo, decide si el coche avanza o se detiene y actualiza sus variables.

### Orden de ejecución

La versión V0.5 utiliza tres capas:

1. Publicación de las posiciones.
2. Lectura de mensajes y actualización del movimiento.
3. Cálculo y almacenamiento de métricas.

## Estructura principal

```text
benchmark_model.py          Pruebas iniciales de rendimiento con Circles
model.py                    Ejemplo oficial Circles comentado
model_original.py           Copia del ejemplo oficial
traffic_model_v0.py         Carretera recta inicial
traffic_model_v02.py        Incorporación del semáforo
traffic_model_v03.py        Semáforo y detección de coche delantero
traffic_model_v04.py        Incorporación de métricas
traffic_model_v05.py        Exportación robusta a CSV
plot_traffic_metrics.py     Generación de gráficas
traffic_metrics_v05.csv     Métricas generadas por la simulación
graficas_v05/               Gráficas generadas desde el CSV
```

## Requisitos utilizados

- Windows.
- GPU NVIDIA compatible con CUDA.
- CUDA Toolkit 13.x.
- Python 3.12.
- `pyflamegpu` para CUDA 13.
- `matplotlib`.

## Instalación

Desde PowerShell, en la raíz del repositorio:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El archivo `requirements.txt` utiliza el repositorio de paquetes de FLAME GPU para CUDA 13.

## Ejecución

Ejemplo reproducible con 200 pasos y semilla 12:

```powershell
.\venv\Scripts\activate
python traffic_model_v05.py -s 200 -r 12
python plot_traffic_metrics.py
```

También puede ejecutarse directamente con el intérprete del entorno, sin activarlo:

```powershell
.\venv\Scripts\python.exe traffic_model_v05.py -s 200 -r 12
.\venv\Scripts\python.exe plot_traffic_metrics.py
```

## Resultados

`traffic_model_v05.py` genera `traffic_metrics_v05.csv` con estas columnas:

- `paso`
- `estado_semaforo`
- `coches_parados`
- `parados_por_semaforo`
- `parados_por_coche_delante`
- `cola_aproximada`
- `coches_cruzan_este_paso`
- `cruces_totales`
- `espera_total`
- `espera_media`

`plot_traffic_metrics.py` genera en `graficas_v05/`:

- `espera_media.png`
- `cola_aproximada.png`
- `cruces_totales.png`
- `motivos_parada.png`
- `parados_y_cola.png`

La versión corregida escribe primero en un CSV temporal. El CSV definitivo solo se sustituye cuando la simulación termina correctamente. Si la ejecución falla, se conserva el CSV anterior y, cuando existen datos parciales, se guardan en `traffic_metrics_v05_partial.csv`.

## Visualización

La instalación actual devuelve:

```python
pyflamegpu.VISUALISATION == False
```

Por tanto, actualmente se utilizan métricas, CSV y gráficas. Queda pendiente decidir entre instalar una distribución de FLAME GPU con visualización o crear un visor propio a partir de las posiciones de los vehículos.

## Limitaciones actuales

- El escenario todavía es una carretera recta, no un cruce completo.
- Solo existe un carril y una dirección de circulación.
- Los vehículos iniciales se crean desde el propio código.
- Los vehículos vuelven al inicio al final de la carretera.
- El semáforo se representa mediante propiedades y reglas del entorno.
- La cola aproximada equivale, por ahora, al número de coches parados.
- El prototipo está implementado en Python; la versión final se plantea en C++.

## Referencias

- Documentación: https://docs.flamegpu.com/
- Tutorial oficial: https://docs.flamegpu.com/tutorial/index.html
- Plantilla Python: https://github.com/FLAMEGPU/FLAMEGPU2-model-template-python
- Plantilla C++: https://github.com/FLAMEGPU/FLAMEGPU2-model-template-cpp
