# Laboratorio 3

Análisis de la dinámica de contagio de un brote con el modelo SIR y una herramienta interactiva para comunicar el efecto de las intervenciones a tomadores de decisión no técnicos.

## Cómo ejecutar

### Notebook (Task 2)

Requiere Python 3.14 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab Task-2.ipynb
```

En Jupyter, seleccionar el kernel del entorno virtual y ejecutar todas las celdas en orden
(*Run → Restart Kernel and Run All Cells*).

Para registrar el kernel con un nombre propio:

```bash
python -m ipykernel install --user --name lab3-modelacion --display-name "Python (Lab-3)"
```

### Artifact interactivo (Task 4)

No requiere instalación ni servidor. Abrir el archivo directamente en el navegador:

```bash
xdg-open Task-4/sir-interactivo.html     # Linux
# macOS: open Task-4/sir-interactivo.html
# Windows: start Task-4\sir-interactivo.html
```

## Video Demostrativo

[![Video Demostración](https://i.imgur.com/tzWPj2e.png)](https://canva.link/cc2017-laboratorio-3)

> Haz clic en la imagen para ver el video completo mostrando el artefacto corriendo y las explicaciones de sus componentes.
