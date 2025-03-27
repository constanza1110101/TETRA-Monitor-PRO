# Contribuyendo a TETRA Monitor Pro

¡Gracias por tu interés en contribuir a TETRA Monitor Pro! Este documento proporciona directrices para contribuir al proyecto de manera efectiva.

## Consideraciones Éticas y Legales

Antes de contribuir, ten en cuenta que este proyecto está destinado **exclusivamente para uso educativo e investigación legal**. Todas las contribuciones deben:

- No facilitar la interceptación ilegal de comunicaciones privadas
- No implementar capacidades de desencriptación de sistemas protegidos
- Mantener el enfoque educativo y de investigación en ciberseguridad

## Cómo Contribuir

### Reportando Problemas

Si encuentras un bug o tienes una sugerencia:

1. Verifica primero que el problema no haya sido reportado previamente
2. Utiliza la plantilla de issues para proporcionar:
   - Descripción clara del problema
   - Pasos para reproducirlo
   - Comportamiento esperado vs. observado
   - Capturas de pantalla si es posible
   - Información del entorno (OS, versiones de Python/dependencias)

### Solicitando Funcionalidades

Para solicitar nuevas características:

1. Describe claramente la funcionalidad deseada
2. Explica cómo beneficiaría al proyecto
3. Proporciona ejemplos de casos de uso
4. Indica si estás dispuesto a implementarla

### Pull Requests

Para enviar código:

1. Crea un fork del repositorio
2. Crea una rama específica para tu contribución (`git checkout -b feature/nueva-funcionalidad`)
3. Implementa tus cambios siguiendo el estilo de código del proyecto
4. Escribe o actualiza las pruebas según sea necesario
5. Asegúrate de que todas las pruebas pasen
6. Actualiza la documentación relevante
7. Envía un pull request con una descripción clara de los cambios

## Estilo de Código

- Sigue PEP 8 para el estilo de código Python
- Utiliza nombres descriptivos para variables y funciones
- Escribe docstrings para todas las funciones, clases y módulos
- Comenta el código cuando la lógica no sea obvia
- Mantén las funciones pequeñas y con un propósito único

## Estructura del Proyecto

Familiarízate con la estructura del proyecto antes de contribuir:

tetra_monitor_pro/
├── tetra_monitor.py       # Script principal
├── requirements.txt       # Dependencias
├── README.md              # Documentación principal
└── tetra_data/            # Directorio de datos
├── audio/             # Grabaciones de audio
├── logs/              # Archivos de registro
├── ml_models/         # Modelos de machine learning
└── ...                # Otros directorios de datos

plaintext

Hide

## Áreas de Contribución

Estas son algunas áreas específicas donde las contribuciones son especialmente bienvenidas:

1. **Mejoras de Rendimiento**: Optimizaciones para el procesamiento de señales
2. **Modelos de ML**: Mejora de los algoritmos de clasificación de señales
3. **Interfaz Web**: Mejoras en la visualización y experiencia de usuario
4. **Documentación**: Tutoriales, ejemplos y mejoras en la documentación existente
5. **Pruebas**: Ampliación de la cobertura de pruebas
6. **Compatibilidad**: Soporte para más dispositivos SDR

## Proceso de Revisión

Después de enviar un pull request:

1. Los mantenedores revisarán tu código
2. Es posible que se soliciten cambios o aclaraciones
3. Una vez aprobado, tu código será fusionado en la rama principal

## Comunicación

Para discusiones sobre el desarrollo:

- Utiliza los issues de GitHub para preguntas específicas
- Para discusiones más amplias, utiliza [canal de comunicación]

## Reconocimiento

Los contribuyentes serán reconocidos en el archivo README.md y en las notas de la versión.

---

Al contribuir a este proyecto, confirmas que tus contribuciones se ajustan a los propósitos educativos y legales del proyecto, y que no tienes la intención de facilitar actividades no autorizadas o ilegales.
