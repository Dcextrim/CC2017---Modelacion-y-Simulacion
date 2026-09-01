# Documentación ODD

Protocolo ODD (Overview, Design concepts, Details) según Grimm et al. (2020).  
Implementación de referencia: `Task-1-2.ipynb`.

## Overview

### Propósito

El modelo responde si un sistema de bicicletas compartidas reduce la congestión vehicular en el centro urbano y en qué magnitud, con incertidumbre cuantificada.

### Entidades, variables de estado y escalas

**CommutterAgent** — persona que se desplaza diariamente.

| Variable    | Unidad           | Descripción                                               |
| ----------- | ---------------- | --------------------------------------------------------- |
| `unique_id` | —                | Identificador asignado por Mesa                           |
| `pos`       | celda `(x, y)`   | Posición actual en la cuadrícula                          |
| `ingreso`   | quetzales al mes | Ingreso mensual del agente                                |
| `distancia` | kilómetros       | Distancia de su vivienda al trabajo                       |
| `destino`   | celda `(x, y)`   | Celda de llegada                                          |
| `modo`      | categórica       | `bicicleta` o `automovil`                                 |
| `vehiculo`  | referencia       | `VehicleAgent` emparejado, o `None` si viaja en bicicleta |

**VehicleAgent** — vehículo en circulación. Solo tiene `unique_id` y `pos`. No decide nada, lo mueve su conmutador. Existe uno por cada conmutador que elige automóvil.

**Entorno.** Cuadrícula `MultiGrid` de 20×20 = 400 celdas sin frontera periódica, que representa el centro urbano. La variable de entorno `congestion_map` es un arreglo NumPy de 20×20 con el número de `VehicleAgent` en cada celda.

**Escalas.** Una celda es una unidad de vía del centro; el desplazamiento se mide en pasos de Manhattan sobre la cuadrícula. Un paso de tiempo es un instante de decisión y movimiento del pico matutino. Cada corrida dura T = 10 pasos y arranca con N = 150 conmutadores.

### Descripción general del proceso

En cada paso de tiempo el scheduler activa a los conmutadores activos en orden aleatorio. Al ser activado, un conmutador intenta avanzar una celda hacia su destino por el camino de Manhattan más corto: primero corrige el eje horizontal y luego el vertical. Si la celda objetivo ya está ocupada por otro conmutador, espera en su posición actual. Si viaja en automóvil, su vehículo se mueve con él. Cuando llega a su destino, el conmutador y su vehículo salen del schedule y de la cuadrícula. Al terminar el paso, el modelo recalcula `congestion_map` contando los vehículos presentes en cada celda.

## Design concepts

### Emergencia

De reglas puramente individuales (elegir modo, avanzar, esperar si hay bloqueo, salir al llegar) emerge la distribución espacial y el nivel agregado de congestión al final de la ventana observada. Ese nivel no está programado en ninguna regla, resulta de cuántos agentes eligen automóvil, cuánto tardan en llegar y cuántas veces se bloquean entre sí. La política de bicicletas desplaza toda la distribución del índice de congestión, no solo su promedio.

### Heterogeneidad

Los agentes difieren en ingreso y distancia al trabajo. Ambos atributos se muestrean juntos de una normal multivariada con

- media `μ = (8500 Q, 6.5 km)`
- covarianza `Σ = [[4 000 000, −800], [−800, 4]]`

lo que implica desviaciones de Q2000 y 2 km y una correlación teórica negativa (quien gana más tiende a vivir más cerca del trabajo). Los orígenes y destinos también son distintos entre agentes. Esa heterogeneidad es la que hace que solo una fracción de la población cumpla el criterio de los 3 km y pueda cambiarse a la bicicleta.

### Estocasticidad

1. Muestreo de los atributos correlacionados de los 150 agentes (normales estándar transformadas por Cholesky).
2. Sorteo de la celda de origen de cada agente.
3. Sorteo de la celda de destino de cada agente.
4. Orden de activación de los agentes en cada paso (`RandomActivation`).

No hay aleatoriedad en la elección de modo ni en el movimiento, ambos son deterministas dada la inicialización. Toda la aleatoriedad se controla con una sola semilla por corrida, lo que hace el modelo exactamente reproducible.

### Observación

Al final de cada paso se actualiza `congestion_map`. La variable de salida es el índice de congestión promedio al final de la corrida,

`Y = promedio de congestion_map en el paso T = 10`,

es decir, el número de vehículos por celda sobre las 400 celdas. Se recolecta una vez por corrida. Para el análisis estadístico se ejecutan M = 100 corridas por escenario con las mismas semillas en ambos, y se comparan las dos distribuciones de Y mediante bootstrap con B = 2000 remuestras. Durante la verificación (Task 1.3) se observan además el orden de activación por paso, la media y desviación de los atributos, y el conteo de agentes activos por paso.

## Details

### Inicialización

Con la semilla de la corrida se crea un generador NumPy y un modelo `CityModel` con la política activa o inactiva. Después:

1. Se factoriza la covarianza por Cholesky, `Σ = L Lᵀ`.
2. Se sortea una matriz `Z` de 150×2 normales estándar independientes.
3. Se obtienen los atributos como `X = μ + Z Lᵀ`, de modo que cada fila es un par (ingreso, distancia) con la media y la covarianza especificadas.
4. Para cada agente se sortean una celda de origen y una celda de destino uniformes en la cuadrícula de 20×20.
5. El agente elige su modo con la regla determinista y se coloca en su origen; si eligió automóvil, se crea un `VehicleAgent` en esa misma celda y queda emparejado con él.
6. Se calcula `congestion_map` para el estado inicial.

Todas las corridas empiezan con 150 conmutadores activos. El número de vehículos depende del escenario: 150 sin política y alrededor de 144 con política, porque cerca del 4% de los agentes vive a 3 km o menos.

### Submodelos

**Elección de modo (`choose_transport`)**

Regla determinista:

```
modo = bicicleta   si distancia ≤ 3 km y política activa
modo = automovil   en otro caso
```

Dos agentes con la misma distancia siempre eligen lo mismo y una corrida repetida con la misma semilla produce las mismas elecciones. Para hacerla estocástica bastaría sortear el modo con una probabilidad decreciente en la distancia, por ejemplo `p = 1 / (1 + exp(β·(distancia − 3)))`.

**Movilidad (`CommutterAgent.step`)**

En cada activación:

1. Calcula la celda objetivo. Si `x ≠ x_destino`, avanza una celda en el eje x hacia el destino; si no, y `y ≠ y_destino`, avanza una celda en el eje y. Es un paso del camino de Manhattan más corto.
2. Si la celda objetivo contiene otro `CommutterAgent`, el agente no se mueve y espera en su celda.
3. Si se mueve y viaja en automóvil, su `VehicleAgent` se mueve a la misma celda.
4. Si su posición coincide con el destino, se elimina, quita su vehículo de la cuadrícula y llama a `self.model.schedule.remove(self)` y `self.model.grid.remove_agent(self)`.

**Vehículos (`VehicleAgent`)**

No tienen regla propia ni entran al schedule. Su única función es ser contados por `congestion_map`; su posición la determina el conmutador al que pertenecen.

**Actualización del entorno (`CityModel.actualizar_congestion`)**

Al final de cada paso, pone `congestion_map` en cero y suma uno en la celda de cada vehículo activo. Actúa como comunicación por entorno (cualquier agente puede leer la congestión sin comunicarse directamente con otro).

**Scheduling**

Se usa `RandomActivation` (asíncrono aleatorio). Se justifica en la naturaleza del proceso de modelado (el desplazamiento matutino es comportamiento social, no un proceso físico simultáneo). Cada persona sale de su casa a una hora distinta y avanza sin esperar a las demás, y cuando dos personas compiten por el mismo espacio de calle una pasa primero y la otra espera; ese orden de llegada forma parte del fenómeno. Un esquema síncrono supondría un reloj que sincroniza a toda la ciudad, lo que no corresponde a este proceso. Reordenar la activación en cada paso evita además que un mismo agente conserve siempre la prioridad de paso.
