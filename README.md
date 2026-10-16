# Prototipo de tráfico con FLAME GPU 2

Proyecto de trabajo del TFG sobre simulación de tráfico mediante agentes en GPU.
El desarrollo actual utiliza Python para describir y ejecutar el modelo, y funciones RTC en C++ para actualizar los agentes en la GPU.

## Archivos de trabajo

| Archivo | Uso |
| --- | --- |
| `traffic_model.py` | Modelo principal que se modifica a partir de ahora. |
| `plot_traffic_metrics.py` | Genera las gráficas del CSV seleccionado. |
| `requirements.txt` | Dependencias del entorno sin visualización. |
| `traffic_metrics.csv` | Resultado de la última ejecución correcta del modelo principal. |
| `traffic_metrics_v08.csv` | Resultado de referencia que se conserva al reorganizar el proyecto. |

Las etapas anteriores del modelo se conservan mediante commits. Antes de retirar los archivos `traffic_model_v*.py` de la carpeta de trabajo, se guardan en Git el código de V0.8 y su CSV de referencia. Los ejemplos con Circles y los resultados anteriores se mantienen como material de trabajo independiente.

## Estado del modelo

La base de `traffic_model.py` es la lógica de V0.8, comprobada en el portátil Windows con GPU NVIDIA el 15 de octubre de 2026. La reorganización cambia los nombres del modelo y de sus archivos de salida; conserva las funciones RTC, las reglas de tráfico y los parámetros.

- Carretera de un carril con longitud de 100 unidades y un semáforo en la posición 50.
- 20 coches en posiciones iniciales distintas: de 5 a 62, separados por 3 unidades.
- Avance máximo de 1 unidad por paso y separación de seguridad de 2 unidades entre posiciones.
- Consulta de posiciones con `MessageSpatial2D` y radio de 3 unidades.
- Fase inicial roja de 10 pasos, seguida de 10 pasos en verde.
- Visor oficial con colores según el avance de los coches y la fase del semáforo.

Cada paso utiliza cuatro capas: publicar posiciones, actualizar movimiento, actualizar el semáforo visual y calcular las métricas.

## Ejecución en Windows

Desde PowerShell, en la carpeta del proyecto, utilizando el entorno con visualización ya instalado:

```powershell
.\venv_vis\Scripts\python.exe traffic_model.py -s 20 -r 12
```

La prueba de referencia utiliza 20 pasos y semilla 12. Al terminar, el visor mantiene el último fotograma; hay que cerrar su ventana para finalizar el programa.

Para generar las gráficas del nuevo CSV:

```powershell
.\venv\Scripts\python.exe plot_traffic_metrics.py
```

También se puede representar un resultado anterior sin cambiar el script:

```powershell
.\venv\Scripts\python.exe plot_traffic_metrics.py --csv traffic_metrics_v08.csv --output-dir graficas_v08
```

Los entornos utilizados son `venv` (sin visor) y `venv_vis` (con `pyflamegpu==2.0.0rc4+cuda130.vis`). `requirements.txt` corresponde al primero. Para reproducir el segundo en otro equipo:

```powershell
py -3.12 -m venv venv_vis
.\venv_vis\Scripts\python.exe -m pip install --upgrade pip
.\venv_vis\Scripts\python.exe -m pip install --extra-index-url https://whl.flamegpu.com/whl/cuda130-vis/ pyflamegpu==2.0.0rc4+cuda130.vis
```

## Resultados y comprobaciones

El CSV usa punto y coma como separador y coma decimal. Registra paso, fase, coches parados, motivos de parada, cola aproximada, cruces por paso, cruces acumulados, espera total y media, distancia mínima y parejas consecutivas con separación insuficiente.

La distancia mínima se calcula ordenando las posiciones del único carril después del movimiento. Si detecta una distancia inferior a 2 unidades, admitiendo una tolerancia de `1e-5` por redondeo, la prueba termina con error. Esta lectura desde el host sirve como diagnóstico para 20 coches y no como medida del rendimiento de la GPU.

El resultado se escribe primero en `traffic_metrics.tmp.csv`. Tras una ejecución correcta se sustituye `traffic_metrics.csv`; en caso de error se conserva el CSV anterior y los datos disponibles quedan en `traffic_metrics_partial.csv`.

Resultados observados en GPU con 20 pasos y semilla 12:

| Etapa | Cruces finales | Espera media acumulada por coche | Comprobación de separación |
| --- | ---: | ---: | --- |
| V0.7 | 3 | 5,25 pasos | No incluía esta métrica. |
| V0.8 | 3 | 7,50 pasos | Mínimo global de 2 unidades y cero parejas inseguras en los 20 pasos. |

Las gráficas se guardan en `graficas/`: espera media, cola aproximada, cruces totales, motivos de parada y coches parados frente a cola aproximada. Estas dos últimas series siguen siendo equivalentes mientras no se defina una cola espacial real.

## Próximos cambios y límites

La siguiente mejora es sustituir la zona de parada de 20 unidades por una parada ante la línea del semáforo, conservando la separación entre coches.

El modelo todavía usa posiciones puntuales y unidades abstractas. La espera cuenta pasos detenidos y no segundos ni el tiempo medio de un trayecto completo. Al terminar la carretera, los coches vuelven provisionalmente a cero; esa regla puede introducir conflictos en pruebas largas. La prueba de 20 pasos evita ese retorno. Quedan pendientes la longitud de los vehículos, las unidades físicas, la cola espacial, las entradas y salidas y el escenario de cruce.

## Referencias

- [Documentación de FLAME GPU 2](https://docs.flamegpu.com/)
- [Mensajes espaciales](https://docs.flamegpu.com/guide/defining-messages-communication/index.html)
- [Operaciones de agentes desde el host](https://docs.flamegpu.com/guide/host-functions/agent-operations.html)
- [Repositorio del proyecto](https://github.com/DaniSantos05/flamegpu_tests)
