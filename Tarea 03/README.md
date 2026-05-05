<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td width="38%" valign="middle" style="padding-right: 18px;">
      <img src="imgs/fciencias.webp" alt="UNAM y Facultad de Ciencias" width="360" />
    </td>
    <td valign="middle">
      <p style="margin: 0; font-size: 28px; font-weight: 700;">
        Universidad Nacional Autónoma de México
      </p>
      <p style="margin: 6px 0 0 0; font-size: 20px; font-weight: 600;">
        Facultad de Ciencias
      </p>
      <p style="margin: 12px 0 0 0; font-size: 18px; font-weight: 600;">
        Almacenes y Minería de Datos
      </p>
      <p style="margin: 6px 0 0 0; font-size: 16px;">
        Profesora: MSc Data Analytics Jessica Santizo Galicia
      </p>
      <p style="margin: 4px 0 0 0; font-size: 15px;">
        Ayudantes: Diego Antonio Villalba González, Ares Gael Castro Romero
      </p>
    </td>
  </tr>
</table>

---

# Tarea 03 — Reglas de Asociación y Clasificación

**Integrantes del equipo:**
- Brenda Jiménez Ruiz
- Irany Solano Marcial

**Semestre:** 2026-2  
**Fecha de entrega:** 4 de mayo de 2026

---

# Introducción

Este repositorio contiene la implementación y análisis correspondientes a la **Tarea 03** de Almacenes y Minería de Datos. Se trabaja con el dataset del Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP), que contiene registros de personas desaparecidas en México.

El análisis se divide en tres partes:

- **Implementación manual de Apriori** — algoritmo de reglas de asociación construido desde cero en Python puro.
- **Reglas de asociación** — comparación de tres algoritmos: Apriori propio, Apriori de mlxtend y FP-Growth, evaluados con soporte, confianza y lift.
- **Clasificación** — modelo de predicción de `ESTATUS_VICTIMA` con árboles de decisión y Random Forest, con análisis de métricas adecuadas para datos desbalanceados.

---

# Dataset

El dataset limpio fue generado en la Tarea 02 a partir del dataset original del SESNSP (133,887 registros totales).

| Grupo | Registros | Porcentaje | Decisión |
|---|---|---|---|
| Registros CONFIDENCIAL | 49,149 | 36.7% | Excluidos — información protegida |
| Registros con datos públicos | 84,738 | 63.3% | Base del análisis |
| **TOTAL** | **133,887** | **100%** | |

> **Limitación importante:** los patrones encontrados aplican únicamente al 63.3% de los casos con información pública. Los patrones en los datos protegidos podrían ser completamente diferentes.

---

# Instalación y ejecución (entorno virtual)

## Requisitos previos
1. **Python 3.11 o superior** instalado.
2. **pip** disponible (normalmente viene con Python).
3. Una terminal (por ejemplo, la terminal integrada de VS Code o Terminal en Linux/macOS).

Verifica que Python y pip funcionan:

```bash
python --version
pip --version
```

En macOS/Linux, si `python` no funciona, prueba:

```bash
python3 --version
pip3 --version
```

---

## Crear y activar el entorno virtual

Asegúrate de estar en la carpeta del proyecto (donde está `requirements.txt`).

### Windows

1) Crear el entorno virtual

```bash
python -m venv .venv
```

2) Activar el entorno

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

CMD:

```bat
.\.venv\Scripts\activate.bat
```

3) Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### macOS / Linux / Unix

1) Crear el entorno virtual

```bash
python3 -m venv .venv
```

2) Activar el entorno

```bash
source .venv/bin/activate
```

3) Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Uso diario

Activa el entorno cada vez que vayas a trabajar en el proyecto:

```bash
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate    # Windows
```

Para salir del entorno:

```bash
deactivate
```

---

## Ejecución

### Reglas de asociación

```bash
python3 src/tarea03.ipynb
```

### Clasificación

```bash
python3 src/clasificacion_tarea03_v2.py
```

Las gráficas se guardan automáticamente en la carpeta `src/` como archivos PNG.

---

## Problemas comunes

### Actualizar pip
Dentro del entorno virtual:

```bash
python -m pip install --upgrade pip
```

### VS Code no detecta el entorno
- Abre la paleta de comandos: `Ctrl + Shift + P`
- Busca: **Python: Select Interpreter**
- Elige el intérprete dentro de `.venv`

---
