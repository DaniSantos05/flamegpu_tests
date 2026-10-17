# Prototipo de tráfico con FLAME GPU 2

TFG de Daniel Santos Méndez. El modelo actual es una carretera recta de un carril y un sentido, con semáforo, vehículos con dimensiones, cola espacial y población dinámica. Python describe y coordina el modelo; las funciones RTC de los agentes se ejecutan en GPU.

El alcance previsto del TFG final es un único cruce de dos vías, con los movimientos mínimos que se acuerden con David, llegadas, salidas, fases seguras, métricas y experimentos. La implementación final en C++ es un requisito confirmado; su migración sigue aplazada hasta nuevas indicaciones. No se ha implementado todavía el cruce ni una conexión con datos reales.

## Archivos

| Archivo | Uso |
| --- | --- |
| traffic_model.py | Modelo principal, funciones RTC, llegadas, salidas, métricas y visor. |
| traffic_core.py | Parámetros físicos validados y cálculo de separación y cola, sin dependencia de CUDA. |
| plot_traffic_metrics.py | Gráficas de CSV actuales e históricos. |
| tests/ | Pruebas de CPU y pruebas de integración CUDA opcionales. |
| traffic_metrics.csv | Última ejecución correcta del modelo actual. |
| traffic_metrics.metadata.json | Configuración y versiones de esa ejecución. |
| graficas/ | Siete gráficas del resultado actual. |
| references/traffic_metrics_point_agents.csv | CSV anterior conservado: 20 agentes puntuales y unidades abstractas. |
| [docs/limpieza_2026-10-17.md](docs/limpieza_2026-10-17.md) | Registro de la limpieza y de la retirada de antecedentes; detalle por archivo en [JSON](docs/limpieza_2026-10-17.json). |

Se mantiene un único archivo principal de tráfico, sin crear versiones numeradas. Se retiraron los ejemplos Circles y los resultados antiguos que no utilizaba el modelo actual. La referencia necesaria para las pruebas permanece en references/.

## Unidades y supuestos iniciales

Estos valores son decisiones provisionales para un prototipo; no proceden de una calibración con tráfico real.

| Magnitud | Valor |
| --- | ---: |
| Duración del paso | 1 s |
| Longitud de carretera | 500 m |
| Línea del semáforo | 250 m |
| Anchura del carril | 3,5 m |
| Longitud, anchura y altura del coche | 4,5 × 1,8 × 1,5 m |
| Velocidad máxima | 5 m/s = 18 km/h |
| Hueco libre mínimo | 5,5 m |
| Separación mínima entre frontales | 10 m |
| Avance máximo por paso | 5 m |
| Radio de mensajes | 15 m |
| Población inicial | 20 coches |
| Frontales iniciales | 25, 40, 55, …, 310 m |
| Rojo / verde | 10 s / 10 s |
| Intervalo entre solicitudes de entrada | 6 s |

La coordenada x representa el **frontal**. La trasera está en x - 4,5 y el centro dibujado en x - 2,25. El cubo del visor utiliza las dimensiones del vehículo; ya no define el punto de parada mediante su centro.

Las posiciones longitudinales, el avance y la separación se han reescalado por cinco respecto al caso anterior. La geometría lateral se define por separado: ROAD_WIDTH del modelo anterior era el tamaño del dominio espacial, no la anchura física de un carril. El dominio lateral actual mide 100 m y el carril se centra en y = 50 m.

Se conserva un coche inicialmente sobre la línea y cuatro después de ella. Por eso esta población sirve para comprobar equivalencia y casos límite, pero no representa veinte llegadas nuevas pendientes de cruzar. Para estudiar una demanda que nace íntegramente en la entrada, se puede usar --initial-cars 0.

La duración del paso, las fases y el intervalo de llegada se configuran en segundos. Las duraciones deben ser múltiplos del paso; cambiar la resolución temporal puede cambiar la dinámica del seguimiento. No hay aceleración ni frenada continua.

## Orden y reglas de cada paso

1. Una función host genera solicitudes e incorpora un vehículo si hay espacio.
2. car_output_position publica identificadores y posiciones en MessageSpatial2D.
3. car_move calcula el movimiento con las posiciones publicadas al comienzo del paso.
4. traffic_light_update actualiza la fase visual.
5. TrafficMetrics registra eventos, espera, geometría y salidas.
6. car_remove elimina en GPU los vehículos cuya trasera ya ha abandonado la carretera.

Los nacimientos host se hacen efectivos al finalizar su capa, antes de publicar posiciones. La eliminación ocurre después de guardar las métricas: así no se pierden los cruces ni la espera de los coches que salen.

El avance se limita por el líder y por la línea durante el rojo. En seguimiento, la distancia permitida es la separación frontal menos la longitud del coche y el hueco mínimo. Puede haber avances parciales; speed contiene el desplazamiento realmente representado dividido entre la duración del paso. Si el desplazamiento queda en cero, se acumula espera aunque una operación float hubiera calculado un avance positivo demasiado pequeño para modificar x.

Llegar a la línea no cuenta como cruzar. Un cruce requiere old_x <= LIGHT_X y new_x > LIGHT_X. Los coches que ya están después pueden continuar en rojo. La clasificación host utiliza la misma línea float32 efectiva que CUDA.

### Entradas y salidas

La demanda es periódica y determinista. Con el valor por defecto se solicita un coche en t = 0, 6, 12, … segundos: 600 vehículos/hora de demanda nominal. --arrival-interval 0 desactiva las llegadas. La semilla se conserva para reproducibilidad, aunque actualmente no hay componentes aleatorios.

Un vehículo nuevo nace con la trasera en 0 y el frontal en 4,5 m. Solo se admite si el coche más próximo deja la separación necesaria. Cada paso admite como máximo un vehículo. Las solicitudes bloqueadas se mantienen en una espera exterior y se reintenta su entrada: no se descartan ni se insertan encima de otros vehículos.

Un coche sale definitivamente cuando su trasera alcanza el extremo de 500 m. Su frontal puede superar brevemente ese límite; el dominio de mensajes incluye ese margen. No vuelve a cero.

Se comprueban en cada fila estos balances:

- coches_presentes = coches_introducidos_totales - salidas_totales.
- coches_introducidos_totales = población inicial + llegadas_admitidas.
- pendientes_entrada = llegadas_solicitadas - llegadas_admitidas.

### Cola espacial

Se selecciona el detenido más próximo al semáforo, aguas arriba o sobre la línea, y se recorre hacia atrás el grupo contiguo de detenidos. Un coche en movimiento o un hueco libre mayor que queue_join_gap_m corta la cola. Los detenidos después del semáforo y otros grupos desconectados no se suman.

coches_en_cola cuenta ese grupo. longitud_cola_m mide desde el frontal de la cabeza hasta la trasera del último vehículo; un solo coche ocupa 4,5 m. La cola puede permanecer durante el verde mientras sus seguidores aún no han arrancado.

Es una definición espacial explícita de cola detenida. No incluye vehículos que avanzan lentamente ni todas las posibles agrupaciones de la carretera. cola_aproximada se conserva como alias histórico de coches_parados; las gráficas actuales utilizan coches_en_cola cuando está disponible. Ambas series pueden coincidir en un escenario compacto.

## CSV, tiempos y denominadores

El CSV conserva las catorce primeras columnas anteriores y añade:

tiempo_s, coches_presentes, coches_en_cola, longitud_cola_m,
hueco_libre_minimo_m, llegadas_solicitadas, llegadas_admitidas,
pendientes_entrada, salidas_este_paso, salidas_totales,
coches_introducidos_totales, espera_media_presentes_s y espera_media_salidos_s.

El separador sigue siendo punto y coma, con coma decimal. Las posiciones y distancias del CSV actual están en metros. El lector también admite el formato antiguo de unidades abstractas y pasos, utilizado en las pruebas de compatibilidad.

Cada fila describe el estado posterior al movimiento. El paso 0 usa el rojo del intervalo [0, 1) y se guarda en tiempo_s = 1. La distancia mínima comprueba todos los vehículos antes de eliminar las salidas de ese paso; con menos de dos coches queda vacía. La posición del primero antes de la línea incluye al que está sobre ella y queda vacía si no hay ninguno.

waiting_time suma segundos únicamente cuando el vehículo no se mueve. espera_total es la suma de espera observada de todos los coches introducidos, incluidos los que ya salieron: su unidad agregada es vehículo·s. espera_media divide esa suma entre todos los coches introducidos, no entre la población presente. Si todavía no se introdujo ninguno, se registra 0 como convención.

Las otras dos medias separan vehículos presentes y salidos; quedan vacías si no hay denominador. No son tiempos medios de viaje. La espera exterior antes de entrar todavía no se incorpora a estas medias; se informa su número de solicitudes pendientes. Los coches iniciales solo aportan la espera observada desde el inicio de la simulación.

La espera total no disminuye al salir un coche. La media global sí puede bajar cuando entran vehículos nuevos, porque aumenta su denominador; los pequeños descensos de esa curva son esperables.

Las comprobaciones host de geometría implican transferencias y no sirven para medir el rendimiento de GPU.

### Protección de resultados

Cada ejecución escribe primero un CSV .tmp.csv. Solo después de completar la simulación y comprobar la población final se sustituye el CSV definitivo y se publica su configuración .metadata.json. Un fallo conserva el CSV válido anterior y deja las filas disponibles en _partial.csv, incluida la fila diagnóstica cuando el error se detecta en métricas. Los recursos se liberan en finally.

Se aborta ante separación insegura (tolerancia de redondeo 0,00005 m), cruces en rojo o balances inconsistentes.

## Ejecución y gráficas

En un entorno Python 3.12 con requirements.txt instalado:

~~~powershell
.\venv\Scripts\python.exe traffic_model.py -s 200 -r 12 --no-visualisation
.\venv\Scripts\python.exe plot_traffic_metrics.py
~~~

Para el visor, se utiliza el entorno con pyflamegpu 2.0.0rc4+cuda130.vis:

~~~powershell
.\venv_vis\Scripts\python.exe traffic_model.py -s 200 -r 12
~~~

Las gráficas se generan con `venv`, donde está instalado Matplotlib. `venv_vis` se conserva como entorno del modelo con visor.

El visor abre por defecto una vista ortográfica de detalle: muestra hasta 180 m alrededor del semáforo, con más espacio para observar la cola. Para ver la carretera completa:

~~~powershell
.\venv_vis\Scripts\python.exe traffic_model.py -s 200 -r 12 --view overview
~~~

Ambas vistas conservan las dimensiones físicas de la carretera y de los vehículos.

join() conserva la ventana al terminar; cerrarla devuelve el control a la terminal. setSimulationSpeed(10) solo fija la velocidad de demostración.

venv y venv_vis utilizan el runtime oficial Python 3.12.10 de NuGet, ubicado en .tools/python312-nuget/tools. Esta distribución local está documentada por [CPython para Windows](https://docs.python.org/3.12/using/windows.html#the-nuget-org-packages). Hay que conservarlo completo: ambos entornos dependen de él.

Se actualizaron home, executable y command de sus archivos pyvenv.cfg, conservando los paquetes instalados y la separación entre entornos. Las configuraciones anteriores están en .tools/environment-backups/. venv vuelve a cargar su distribución sin visor y venv_vis carga pyflamegpu 2.0.0rc4+cuda130.vis con VISUALISATION = True. Los comandos habituales anteriores ya funcionan.

El runtime y las copias de configuración son locales, están excluidos mediante .gitignore y no cambian el PATH del sistema. Si se mueve el proyecto, hay que recrear los entornos o actualizar sus rutas base; no se deben copiar como entornos portables a otro equipo.

Para reproducir el caso antiguo con distancias físicas, sin llegadas nuevas y sin sobrescribir el resultado principal:

~~~powershell
.\venv\Scripts\python.exe traffic_model.py -s 20 -r 12 --arrival-interval 0 --output results\reference_physical.csv --no-visualisation
~~~

Ejemplos de controles adicionales: --initial-cars 0, --arrival-interval 1, --time-step 0.5, --red-seconds 15 y --green-seconds 10.

El script de gráficas acepta --csv y --output-dir. Conserva los cinco nombres habituales y añade longitud_cola.png y entradas_salidas.png para el CSV actual. Los CSV antiguos siguen produciendo cinco gráficas, con ejes en pasos.

## Comprobaciones realizadas el 17 de octubre de 2026

Se han ejecutado correctamente **36 pruebas: 26 de CPU y 10 de integración real con FLAME GPU CUDA**.

Las pruebas CPU verifican configuración, unidades, geometría, separación, grupos de cola, lectura histórica y actual y generación de PNG. Las de GPU cubren:

- Comparación de las veinte filas de referencia: mismos estados, paradas, cruces y espera; posiciones multiplicadas por cinco.
- Carretera vacía, nacimientos que se mueven en su primer paso y salidas definitivas.
- Conservación de cruces y espera después de eliminar vehículos.
- Trescientos pasos con demanda alta, entrada bloqueada, conservación de solicitudes y población y comprobaciones de seguridad.
- Protección del CSV anterior ante un fallo host provocado.
- Avances parciales, pasos de medio segundo, línea no representable exactamente en float32 y avances que redondean a cero.

La referencia sin nuevas llegadas termina con 4 cruces, espera total 145 vehículo·s, media 7,25 s y separación mínima entre frontales de 10 m (5,5 m libres).

La ejecución principal guardada dura 200 s, diez ciclos semafóricos, con veinte coches iniciales y una solicitud cada seis segundos:

| Resultado final | Valor |
| --- | ---: |
| Solicitudes / admitidos / pendientes | 34 / 34 / 0 |
| Vehículos introducidos en total | 54 |
| Salidas / presentes | 32 / 22 |
| Cruces acumulados | 40 |
| Espera total observada | 979 vehículo·s |
| Espera media sobre todos los introducidos | 18,129630 s |
| Separación libre mínima global | 5,5 m |
| Pares inseguros / cruces en rojo | 0 / 0 |

Las siete gráficas de graficas/ corresponden a esa ejecución. El visor se comprobó en rojo y verde, sin nuevas llegadas, y se inspeccionaron las vistas de detalle de 180 m y panorámica de 500 m, con carretera gris, vehículos y semáforo visibles. Las ejecuciones terminaron sin error y conservaron el resultado principal. Los registros transitorios de estas comprobaciones se retiraron durante la limpieza; no se guardaron capturas como archivos del proyecto.

Para repetir las pruebas CPU:

~~~powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

Para incluir las pruebas CUDA, con GPU disponible:

~~~powershell
$env:TRAFFIC_GPU_TESTS = '1'
.\venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:\TRAFFIC_GPU_TESTS
~~~

Las 36 pruebas pasaron con venv y los 10 casos CUDA también pasaron con venv_vis. Ambos entornos superaron pip check sin dependencias rotas. Las pruebas guardan sus salidas en directorios temporales.

## Límites y siguiente etapa

El modelo sigue siendo determinista, con una velocidad deseada fija, sin aceleración, frenada física ni calibración. La demanda, las dimensiones, las separaciones y las fases son hipótesis iniciales, no recomendaciones de ingeniería vial. La validación es funcional para los casos descritos; no demuestra realismo, tiempo real ni aceleración frente a CPU.

Quedan pendientes el cruce de dos vías y sus conflictos, la definición final de movimientos y fases, los experimentos y su análisis, la memoria y la defensa. La migración a C++ permanece aplazada. No se incluyen por ahora redes de múltiples cruces, peatones, aprendizaje automático ni una escena 3D realista.

## Fuentes técnicas

Se contrastaron las operaciones utilizadas con los wrappers de la versión instalada y la documentación oficial:

- [Creación y acceso a agentes desde el host](https://docs.flamegpu.com/guide/host-functions/agent-operations.html): los nacimientos se incorporan al terminar su capa.
- [Nacimiento y eliminación de agentes](https://docs.flamegpu.com/guide/agent-functions/agent-birth-death.html): eliminación mediante DEAD y habilitación en el descriptor de función.
- [Posición y dimensiones de agentes en el visor](https://docs.flamegpu.com/guide/visualisation/visualising-agents.html).
- [Repositorio del proyecto](https://github.com/DaniSantos05/flamegpu_tests).

La elección de parámetros y los resultados numéricos proceden de este prototipo y sus ejecuciones, no de esas fuentes.
