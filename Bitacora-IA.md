# Bitácora de IA

Herramienta utilizada: Claude Code (modelo Opus 5).

## Task 2

### Prompt

```text
Implementa el modelo SIR en un Jupyter Notebook nuevo. Usa únicamente NumPy para el modelo y el
integrador: no uses scipy.integrate ni ninguna librería de simulación, RK4 debe ser una
implementación propia. pandas se permite solo para mostrar tablas y matplotlib solo para graficar.

    dS/dt = -βSI/N
    dI/dt =  βSI/N - γI
    dR/dt =  γI

Parámetros: N=500000, S0=499000, I0=800, R(0)=200, γ=1/7 día⁻¹, β=R0·γ con R0=2.8. El tiempo se
mide en días.

Datos observados de infectados activos al final de cada semana: 1850, 4200, 8900 y 16400.

Estructura del código:

1. El campo vectorial en una función pura `derivadas(y, beta, gamma, N)` que reciba el estado
   [S, I, R] y devuelva un arreglo de NumPy con las tres derivadas.
2. Un integrador `rk4(f, y0, dt, T, *args) -> (t, Y)` donde `Y` tenga forma (n+1, 3) para guardar
   la trayectoria completa y no solo el estado final. Usa dt = 0.1 días, que divide exactamente a
   7 días: lee los valores semanales por índice entero, sin interpolar.
3. Una función `sce(beta)` que corra el modelo 28 días y devuelva la suma de cuadrados del error
   contra los datos observados. Reporta una tabla con semana, I observado, I del modelo y el
   residuo, más el valor del SCE.
4. Calibración de β por mínimos cuadrados con un barrido `np.linspace` que minimice `sce`. Reporta
   β*, R0* = β*/γ, el SCE mínimo y la reducción porcentual frente al caso base. Grafica el SCE
   contra β en escala logarítmica y el ajuste calibrado sobre los datos observados.
5. Análisis de sensibilidad con β ∈ {0.8β*, β*, 1.2β*} y γ fijo, en un horizonte de 365 días. Para
   cada valor tabula el pico de I obtenido con `np.argmax` (en habitantes y como porcentaje de N),
   el tiempo al pico en días, el tamaño final de la epidemia R(T)/N y la variación porcentual del
   pico respecto a β*.
6. Dos gráficas finales: S, I y R del modelo calibrado, e I(t) para los tres valores de β. Todos
   los ejes rotulados con unidades.

Reutiliza el mismo `rk4` para las corridas cortas y para la de horizonte largo.
```

### ¿Por qué funcionó?

Funcionó porque el prompt entrega como dato duro todo lo que el modelo necesita: las tres ecuaciones, los parámetros con sus unidades y los cuatro valores observados. No queda margen para equivocar un signo, invertir el término de recuperación o suponer un valor faltante. Las restricciones se expresan como prohibiciones concretas, de modo que basta revisar los imports para confirmar que no se usó una librería de simulación.

También fija la arquitectura antes de escribir código. El campo vectorial queda como función pura y el integrador con una firma que devuelve la trayectoria completa, lo que permite reutilizarlo tanto para los 28 días de la calibración como para los 365 días de la sensibilidad. La decisión clave es pedir `sce(beta)` como función de un solo argumento, con eso la calibración se reduce a evaluar la misma función sobre una malla de `np.linspace`, sin duplicar el código de integración ni arrastrar estado entre corridas.

Finalmente, dos elecciones numéricas se justifican dentro del propio prompt. El paso dt = 0.1 días divide exactamente a 7, así que los valores semanales se leen por índice entero y se elimina la interpolación como fuente de error. Y extender la sensibilidad a 365 días evita el error de buscar el pico dentro de la ventana de cuatro semanas de la calibración, cuando en realidad ocurre mucho después.

## Task 4.1

### Prompt

```text
Construye un simulador interactivo del modelo SIR en un único archivo HTML autocontenido. Sin CDN,
sin librerías de gráficas y sin recursos externos. JavaScript puro y la gráfica dibujada a mano
sobre un <canvas>.

Modelo: integra dS/dt = -βSI/N, dI/dt = βSI/N - γI, dR/dt = γI con implementación de RK4 propia.
Trabaja en fracciones de la población (N = 1), de modo que el umbral del 10% sea simplemente 0.10.
Deriva γ = 1/(período infeccioso) y β = R0·γ. Usa dt = 0.25 días y un horizonte de 365 días.

Estructura del código:

1. El campo vectorial en `campo(s, i, beta, gamma)`, que devuelva [dS, dI, dR].
2. Una función `simular(R0, dur, frac)` que corra RK4 sobre `Float64Array` y devuelva las tres
   trayectorias junto con beta, gamma, el pico de I y el tiempo al pico. El pico se obtiene
   recorriendo el arreglo de I; cuando R0 < 1 el máximo cae en el índice 0 y el tiempo al pico debe
   salir 0.
3. Tres `input type="range"` con etiqueta y valor en vivo: R0 de 0.5 a 5.0 paso 0.1, período
   infeccioso 1/γ de 3 a 21 días paso 1, y fracción inicial infectada I0/N de 0.001 a 0.05 paso
   0.001. Muestra además el β y el γ derivados junto a su control, para que se vea qué cambia al
   mover R0.
4. Una función `dibujar(sim, tMax)` que pinte el canvas escalando por `devicePixelRatio`. Eje Y
   fijo de 0 a 100% de N y eje X en días, ambos rotulados con unidades. Las tres curvas en colores
   distinguibles con leyenda en HTML, y una línea horizontal punteada en 0.10 con la etiqueta
   "Capacidad hospitalaria (10% de N)" visible sobre la gráfica.
5. Un panel de métricas que se actualice solo: R0 actual, p* = 1 - 1/R0 (mostrado como 0 cuando
   R0 ≤ 1), el pico de infectados como porcentaje de N, el tiempo al pico en días y un indicador
   verde o rojo según R0 sea menor o mayor que 1.
6. Una función `actualizar()` enlazada al evento `input` de los tres controles y al `resize` de la
   ventana, que lea los sliders, simule, refresque las métricas y redibuje. Mide el ciclo completo
   con `performance.now()` y muestra los milisegundos en pantalla.

Estilización: `body` a 100vh con `overflow: hidden` y layout en grid, de modo que los controles y
el panel de métricas queden visibles sin hacer scroll y el canvas ocupe el espacio restante.
```

### ¿Por qué funcionó?

Funcionó porque la restricción central, un solo archivo sin CDN ni recursos externos, es verificable de forma sencilla con `grep` de `http` o `<script src` basta para comprobarla. Esa misma restricción garantiza que el simulador corra sin conexión, lo cual importa porque el artifact se usa para grabar un video y no puede depender de que una librería remota cargue a tiempo.

También fija dos decisiones de implementación que simplifican todo lo demás. Trabajar con N = 1 hace que las fracciones sean el estado natural del sistema, el umbral hospitalario es literalmente 0.10 y las métricas en porcentaje salen de multiplicar por cien, sin arrastrar la población como parámetro por todas las funciones. Y separar `simular`, `dibujar` y `actualizar` mantiene el bucle de interacción acotado, con el cálculo numérico aislado del dibujo y del manejo de eventos.

Finalmente, los detalles que suelen fallar se entregan como dato y no como descripción. Los rangos y pasos exactos de los tres controles son explícitos. Pedir la medición con `performance.now()` en pantalla convierte el criterio de los 500 milisegundos en evidencia observable. Y el caso R0 < 1 se menciona de forma explícita porque es justamente el que rompe una búsqueda innecesaria del máximo (el pico está en el índice inicial y el tiempo al pico debe reportarse como cero).
