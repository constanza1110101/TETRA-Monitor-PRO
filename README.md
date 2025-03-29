# Herramienta Avanzada para Análisis de Señales TETRA

**SOLO PARA USO EDUCATIVO E INVESTIGACIÓN LEGAL**

## Descripción

TETRA Monitor Pro es una herramienta de análisis de radiofrecuencia especializada en la monitorización y análisis de señales TETRA (Terrestrial Trunked Radio). Diseñada con fines educativos y de investigación en ciberseguridad, permite detectar, analizar y clasificar señales en las bandas utilizadas por sistemas TETRA y tecnologías similares.

## Características Principales

- **Análisis Espectral**: Visualización en tiempo real del espectro de frecuencias.
- **Escaneo de Bandas**: Detección automática de actividad en rangos de frecuencia configurables.
- **Clasificación de Señales**: Identificación de tipos de señales mediante machine learning.
- **Grabación de Audio**: Captura de audio demodulado para análisis posterior.
- **Base de Datos**: Almacenamiento estructurado de señales detectadas y metadatos asociados.
- **Geolocalización**: Registro de posición GPS para mapeo de señales.
- **Interfaz Web**: Visualización remota de datos y análisis.
- **Generación de Informes**: Creación de informes detallados en formato HTML y gráficos.
- **Exportación de Datos**: Exportación a formatos CSV y JSON para análisis externo.

## Requisitos

- **Python 3.7+**
- **GNU Radio con gr-osmosdr**
- **Dispositivo SDR compatible** (Airspy, HackRF One, RTL-SDR)

### Dependencias de Python:
- numpy
- matplotlib
- scipy
- tensorflow
- sqlite3
- pyaudio
- folium
- gpsd-py3 (opcional, para GPS)

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/usuario/tetra-monitor-pro.git
cd tetra-monitor-pro
Instalar dependencias:

bash
Copiar
Editar
pip install -r requirements.txt
Instalar GNU Radio y drivers SDR (específico para cada sistema operativo).

Uso Básico
Escaneo de Banda TETRA:
bash
Copiar
Editar
python tetra_monitor.py -s --start 380 --end 400 --step 0.025
Monitoreo de Frecuencia Específica:
bash
Copiar
Editar
python tetra_monitor.py -f 395.0 -t 120 --audio
Análisis con Funciones Avanzadas:
bash
Copiar
Editar
python tetra_monitor.py -f 395.0 --advanced -t 300
Iniciar Interfaz Web:
bash
Copiar
Editar
python tetra_monitor.py --web --port 8080
Analizar Base de Datos:
bash
Copiar
Editar
python tetra_monitor.py --analyze
Exportar Datos:
bash
Copiar
Editar
python tetra_monitor.py --export csv
Estructura de Directorios
plaintext
Copiar
Editar
tetra_data/
├── audio/               # Grabaciones de audio demodulado
├── exports/             # Exportaciones CSV/JSON
├── keys/                # Información de operadores (no incluye claves)
├── logs/                # Registros de actividad
├── maps/                # Mapas de señales generados
├── ml_models/           # Modelos de machine learning
├── protocols/           # Información de protocolos
├── reports/             # Informes generados
└── spectrum/            # Datos de espectro y espectrogramas
Modos de Operación
Modo Básico: Análisis espectral y detección de señales.

Modo Avanzado: Añade clasificación ML, análisis de protocolos y detección de anomalías.

Modo Web: Permite visualización remota a través de navegador web.

Limitaciones Legales
Esta herramienta:

Solo realiza monitoreo pasivo del espectro radioeléctrico.

No implementa capacidades de desencriptación de comunicaciones protegidas.

No está diseñada para interferir con sistemas de comunicaciones operativos.

El uso de esta herramienta debe cumplir con las regulaciones locales sobre radiocomunicaciones. En muchos países, la interceptación de comunicaciones no autorizadas es ilegal, incluso con fines de investigación.

Consideraciones de Seguridad
Utilizar solo en entornos controlados o con autorización explícita.

No utilizar para monitorear comunicaciones privadas.

Considerar las implicaciones éticas de cualquier investigación.

Contribuciones
Las contribuciones son bienvenidas a través de pull requests. Para cambios importantes, abra primero un issue para discutir su propuesta.

Licencia
Este proyecto está licenciado bajo la Licencia MIT - vea el archivo LICENSE para más detalles.

Descargo de Responsabilidad
Este software se proporciona "tal cual", sin garantía de ningún tipo. Los autores no son responsables del uso indebido o ilegal de esta herramienta. Este proyecto tiene fines exclusivamente educativos y de investigación en ciberseguridad.
