% Reporte
% Dr. Horacio Tapia-McClung ^1^
## Estancia Postdoctoral, LANIA 2014
### Libretas
====================================

% ^1^ htapia@lania.mx

\vfill  

**Address for correspondence: LANIA esta en Xalapa, IIM en la UNAM; eMail:
**htapia@lania.mx


# Abstract


# Introducción

## CUDA
## Python
### IPython Notebooks

# Aplicaciones
Del inmenso rango de posibles aplicaciones relacionadas al cómputo científico de alto rendimiento, hemos elegido algunas representativas de las áreas de interés para el Laboratorio y que son ejemplos que pueden encontrarse en diversas áreas de investigación. El principal objetivo de los ejemplos expuestos es presentar las ventajas que ofrece 1) el uso de procesadores paralelos (GPU) frente a procesadores seriales (CPU), 2) el uso de librerías especializadas para acceder a estos procesadores (PyCUDA, Numba) y 3) nuevas tecnologías que permiten reusar y distribuir el código en diversos formatos de manera sencilla (IPython Notebooks), en el desarrollo de aplicaciones computacionales para la solución de problemas científicos que requieren de un alto poder de cómputo.

El contenido de este reporte puede accederse en un repositorio en linea en el formato de __libretas__ (IPython Notebooks) que contienen texto y código mixto (por ejemplo de C y/o de Python) para resolver el problema en cuestión. Las libretas se han descrito en la introducción, y su uso se verá justificado. El contenido de las libretas también se anexa como documento en este reporte, mostrando así la flexibilidad que tienen en la generación de información (electrónica y tradicional).

# Organización de las libretas
Como podrá apreciarse al revisar las libretas, el éstas son _autocontenidas_ en el sentido que la descripción del problema, el método de solución, la implementación del método, los resultados, etc., están dentro de un mismo documento. Por este motivo las descripciones que se ofrecen a continuación de cada uno de los problemas seleccionados es muy breve, omitiendo detalles que pueden consultarse en las libretas.

Las libretas que se han desarrollado son:

1. [Introducción Básica a Python y a las libretas de IPython Notebooks] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/00_Libreta00_IntroduccionBasicaPython.ipynb)
2. [Ecuación de Difusión] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/01_Libreta01_.ipynb)
3. [Ejemplos de Algebra Lineal] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/02_Libreta02_.ipynb)
4. [Generación de Imágenes] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/03_Libreta03_.ipynb)
5. [Análisis de Imágenes] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/04_Libreta04_.ipynb)
6. [Aplicaciones masivas] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/05_Libreta05_.ipynb)

En ellas se incluyen problemas elementales relacionados al cómputo de alto rendimiento y que aparecen en diversas aplicaciones científicas. A continuación una breve descripción de cada uno de ellos.

# Ejemplos de Matemáticas Básicas.
Las matemáticas forman la columna principal sobre la que se desarrollan todas las aplicaciones científicas. Aqui hemos elegido dos ejemplos de matemáticas básicas tradicionales que tienen un impacto en diversas áreas del conocimiento, con el fin de mostrar las bondades que proponcionan las metodologías presentadas para acelerar las soluciones de cómputo científico.

# Ecuación de Difusión
[Enlace a Libreta] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/01_Libreta01_.ipynb)

La ecuación de difusión es una ecuación diferencial parcial que se usa para describir procesos dinámicos de difusión, por ejemplo describe la variación de la temperatura en una región ddada del espacio a través del tiempo. Esta ecuación y sus variantes, es de importancia en diversas disciplinas científicas, como en la teoría de la  probabilidad, donde ésta relacionada al estudio del movimiento Browniano; en matemáticas financieras se puede usar para resolver la ecuación diferencial parcial de Black–Scholes. Incluso puede relacionarse con generadores de filtros (kernels) aplicables al análisis de imágenes.

## Objetivos:
1. Implementar una solución numérica a la discretización de la ecuación de difusión en 2D para comparar tiempos de desempeño entre el CPU y el GPU.
2. Comparar la implementación y la obtención de la solución entre lenguajes de programación y marcos de referencia.

## Metodologia
La implementación serial en el CPU se desarrolla 1) utilizando el lenguaje de programación C++ y 2) utilizando Python. La implementación paralela en el GPU se desarrolla 1) en el lenguaje de programación CUDA-C y 2) empleando la librería de Python PyCUDA. En todos los casos de miden los tiempos de desempeño utilizando la función interna de Python `timeit` y, cuando es posible, utilizando funciones nativas del lenguaje de programación (C++ y CUDA-C).

La flexibilidad de las libretas de IPython permite realizar todas las actividades desde el mismo ambiente (veáse 00_Libreta00 para detalles), así que la presentación de la solución a este problema es extensa, de modo que se dividirá esta en 4 libretas auxiliares, una para cada implementación.

# Algebra Lineal
[Enlace a Libreta] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/01_Libreta01_.ipynb)

## Objetivos:
1. Implementar una solución numérica a la discretización de la ecuación de difusión en 2D para comparar tiempos de desempeño entre el CPU y el GPU.
2. Comparar la implementación y la obtención de la solución entre lenguajes de programación y marcos de referencia.

## Metodologia
La implementación serial en el CPU se desarrolla 1) utilizando el lenguaje de programación C++ y 2) utilizando Python. La implementación paralela en el GPU se desarrolla 1) en el lenguaje de programación CUDA-C y 2) empleando la librería de Python PyCUDA. En todos los casos de miden los tiempos de desempeño utilizando la función interna de Python `timeit` y, cuando es posible, utilizando funciones nativas del lenguaje de programación (C++ y CUDA-C).

La flexibilidad de las libretas de IPython permite realizar todas las actividades desde el mismo ambiente (veáse 00_Libreta00 para detalles), así que la presentación de la solución a este problema es extensa, de modo que se dividirá esta en 4 libretas auxiliares, una para cada implementación.

# Ejemplos de Imágenes

# Generación de imágenes
[Enlace a Libreta] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/01_Libreta01_.ipynb)

# Análisis de imágenes
[Enlace a Libreta] (http://nbviewer.ipython.org/github/htapia/lania-pd/blob/master/01_Libreta01_.ipynb)

# Conclusiones y trabajo futuro

# Referencias