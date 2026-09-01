# Laboratorio 4

Modelo basado en agentes (ABM) en Mesa que simula el desplazamiento diario en un centro urbano de 20×20 celdas, para estimar si un sistema de bicicletas compartidas reduce la congestión vehicular. El modelo compara dos escenarios (con y sin política) mediante 100 corridas de cada uno, y cuantifica la diferencia con intervalos de confianza bootstrap.

## Cómo ejecutar

Requiere Python 3.14 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab Task-1-2.ipynb
```

En Jupyter, seleccionar el kernel del entorno virtual y ejecutar todas las celdas en orden (*Run → Restart Kernel and Run All Cells*).

Para ejecutarlo sin abrir Jupyter:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Task-1-2.ipynb
```
