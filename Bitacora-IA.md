# Bitácora de IA

Herramienta utilizada: Claude Code (modelo Opus 5).

## Task 1

### Prompt

```text
Implementa el modelo ABM en un Jupyter Notebook nuevo usando Mesa. Fija mesa==3.0.3 en
requirements.txt, es la última serie que conserva mesa.time.RandomActivation y schedule.remove(),
que son la clase y el método que se necesitan. NumPy para el muestreo y matplotlib para graficar.

Constantes del modelo: GRID = 20 (cuadrícula de 20x20 = 400 celdas), N = 150 conmutadores,
T = 10 pasos, UMBRAL_BICI = 3.0 km, MU = [8500 Q, 6.5 km] y
SIGMA = [[4000000, -800], [-800, 4]].

Estructura del código:

1. `muestrear_atributos(n, rng)` que genere n pares (ingreso, distancia) desde la normal
   multivariada por descomposición de Cholesky: factoriza SIGMA = L @ L.T, sortea Z de normales
   estándar independientes de forma (n, 2) y devuelve MU + Z @ L.T. El generador se recibe como
   argumento, no se crea adentro, para que la corrida completa dependa de una sola semilla.
   Grafica la dispersión ingreso vs distancia con una línea horizontal punteada en el umbral de
   3 km e imprime la correlación muestral junto a la teórica SIGMA[0,1]/sqrt(SIGMA[0,0]*SIGMA[1,1]).
2. `VehicleAgent(mesa.Agent)` sin lógica propia y sin entrar al schedule (existe solo para ser
   contado). Lo mueve el conmutador dueño.
3. `CommutterAgent(mesa.Agent)` con atributos ingreso (Q), distancia (km), destino (celda), modo y
   vehiculo. Método `choose_transport(self)` que devuelva "bicicleta" si
   `distancia <= UMBRAL_BICI and self.model.policy_active`, y "automovil" en otro caso.
4. `CommutterAgent.step(self)`: registra su unique_id en `self.model.orden_activacion`, calcula la
   celda objetivo con un paso del camino de Manhattan más corto corrigiendo primero el eje x y
   luego el eje y, y se mueve solo si esa celda no contiene otro CommutterAgent; si la contiene,
   espera en su posición. Cuando se mueve y tiene vehículo, el vehículo se mueve a la misma celda.
   Al coincidir su posición con el destino, saca su vehículo de la cuadrícula y de la lista del
   modelo y llama a `self.model.schedule.remove(self)` y `self.model.grid.remove_agent(self)`.
5. `CityModel(mesa.Model)` con firma `__init__(self, policy_active=False, seed=None)` que llame a
   `super().__init__(seed=seed)`, cree un `MultiGrid(GRID, GRID, torus=False)`, un
   `RandomActivation`, un `congestion_map` de NumPy de 20x20 tipo int, la lista `vehiculos` y la
   lista `orden_activacion`. Para cada par de atributos sortea origen y destino uniformes en la
   cuadrícula, coloca al conmutador en su origen, lo agrega al schedule y, si eligió automóvil,
   crea su VehicleAgent en esa misma celda y lo agrega a `vehiculos`.
6. `CityModel.actualizar_congestion(self)`: pone `congestion_map` en cero y suma uno en la celda de
   cada vehículo activo. `CityModel.step(self)`: vacía `orden_activacion`, llama a
   `self.schedule.step()` y recalcula la congestión al final del paso.

Verificación, tres celdas independientes que impriman PASA o FALLA con su evidencia:

- Scheduler: corre tres pasos guardando el conjunto de ids activos antes de cada uno y la lista
  `orden_activacion` después. Imprime los primeros ocho ids de cada paso y verifica que no haya
  repetidos, que el conjunto activado coincida con el esperado y que el orden cambie entre pasos.
- Inicialización: media y desviación de ingreso y distancia sobre los 150 agentes contra MU y
  sqrt(diag(SIGMA)), con margen de tres errores estándar de la media, es decir 3*sigma/sqrt(N).
- Conservación: corre T pasos guardando `schedule.get_agent_count()` después de cada uno y verifica
  que la secuencia nunca aumente.
```

### ¿Por qué funcionó?

Funcionó porque el prompt resuelve por adelantado el único punto donde la implementación podía romperse: la versión de la librería. Mesa eliminó `mesa.time` a partir de la versión 3.1, así que pedir `RandomActivation` y `schedule.remove()` sin fijar `mesa==3.0.3` habría producido código contra una API inexistente.

También distribuye las responsabilidades entre las dos clases antes de escribir una línea. El vehículo queda como entidad pasiva que solo existe para ser contada, y el conmutador queda como el único que decide, se mueve y elimina. Con eso, la congestión se calcula recorriendo una lista de vehículos vivos en lugar de barrer las 400 celdas, y el conteo del schedule mide solo conmutadores, que es exactamente lo que la prueba de conservación necesita observar. La instrucción de recibir el generador aleatorio como argumento de `muestrear_atributos` en lugar de crearlo adentro es la que hace que toda la corrida dependa de una sola semilla.

Finalmente, las tres pruebas se piden con su criterio numérico ya definido y no como una idea general de verificar. El margen de tres errores estándar es un número que se calcula, no una apreciación, y lo mismo ocurre con comparar el conjunto de ids activados contra el conjunto de agentes activos antes del paso. Eso evita el resultado habitual de estas verificaciones, que es imprimir valores y dejar el veredicto a la vista del lector.

## Task 2

### Prompt

```text
Sobre el mismo notebook y reutilizando `CityModel`, agrega el análisis estadístico de corridas
múltiples. Solo NumPy y matplotlib.

Estructura del código:

1. `run_simulation(policy_active, seed)` que construya el modelo con esa semilla, lo corra T pasos
   y devuelva `float(modelo.congestion_map.mean())`, es decir el promedio de vehículos por celda
   sobre las 400 celdas al final del paso T.
2. M = 100 corridas por escenario usando las mismas semillas 0..99 en ambos, guardadas en dos
   arreglos de NumPy `Y_sin` y `Y_con`.
3. Histograma de las dos distribuciones superpuestas en una sola figura. Calcula los bordes de los
   bins una sola vez sobre la concatenación de ambos arreglos y pásalos a los dos `hist`, para que
   las barras sean comparables. Colores distintos, alpha 0.6, leyenda y ejes rotulados con unidades.
4. Tabla por escenario con media, desviación estándar con ddof=1, coeficiente de variación
   CV = desviación/media y el número mínimo de corridas M* = ceil((1.96 * CV / 0.05)^2), para un
   error relativo del 5%. Imprímela con anchos de columna fijos.
5. `bootstrap_medias(datos, B=2000, semilla=12345)` que devuelva las B medias remuestreadas.
   Vectorízala: genera una matriz de índices de forma (B, len(datos)) con `rng.integers` y calcula
   `datos[indices].mean(axis=1)` en una sola operación, sin bucle sobre B. Para cada escenario
   reporta media bootstrap, error estándar con ddof=1 e intervalo de confianza del 95% por
   percentiles 2.5 y 97.5.
6. Curva de convergencia: aplica `bootstrap_medias` a los primeros M valores para M de 10 a 100 en
   pasos de 10 y grafica el error estándar resultante para ambos escenarios en la misma figura.
   Imprime también la tabla de valores.
7. Diferencia entre escenarios: remuestrea cada arreglo de forma independiente con un generador
   propio, calcula las B diferencias de medias y reporta ΔY = media(Y_sin) - media(Y_con), su
   intervalo de confianza del 95% por percentiles, si el intervalo incluye el cero, el equivalente
   en vehículos multiplicando por las 400 celdas y la reducción relativa como porcentaje de
   media(Y_sin).
```

### ¿Por qué funcionó?

Funcionó porque `run_simulation(policy_active, seed)` reduce toda la simulación a una función de dos argumentos que devuelve un número. Con esa firma, las 100 corridas de cada escenario son una comprensión de lista y no un bloque con estado acumulado, y la semilla explícita hace que cualquiera pueda repetir una corrida individual para inspeccionarla. Pedir además las mismas semillas 0..99 en ambos escenarios elimina una fuente de diferencia entre ellos: la comparación aísla el efecto de la política en lugar de mezclarlo con el azar de la inicialización.

También se especifica las tres decisiones de graficación y cálculo que suelen salir mal. Los bordes de los bins se calculan una vez sobre la concatenación, así que las dos distribuciones se dibujan sobre la misma malla y el desplazamiento entre ellas es legible; sin eso, cada histograma elige sus propios bins y la figura sugiere diferencias que no existen. La desviación con `ddof=1` queda fijada en todos lados, y la fórmula de M* se entrega escrita con sus valores de z y de error relativo, no como una referencia a "la fórmula vista en clase".

Finalmente, el bootstrap se pide vectorizado y con semilla propia. Generar una matriz de índices de forma (B, M) y promediar sobre el eje 1 reemplaza el bucle de 2000 iteraciones por una sola operación de NumPy, lo que importa porque la curva de convergencia vuelve a llamar a la misma función diez veces por escenario. Fijar la semilla dentro de `bootstrap_medias` hace que la curva sea reproducible entre ejecuciones y que sus escalones se puedan atribuir al tamaño de muestra y no al remuestreo.
