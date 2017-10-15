# ProyectoHPC-KMeans

Este es un proyecto basado en el algoritmo K-Means para clustering de documentos desarrollado en Python de manera serial y paralela usando como herramienta principal de paralelismo la libreria MPI4PY.

Este proyecto se divide en dos subproblemas:
- Diseño e implementación de una función de similaridad entre 2 documentos: Para esto nos basamos en el algoritmo de Jaccard con una pequeña modificación para hacerlo más preciso.
- Una vez se tiene definida e implementada la función de similaridad, se puede ejecutar el algoritmo de clustering K-Means.
  
## Empezando

### Prerequisitos

- pip | [Anaconda] (https://anaconda.org/anaconda/python)
- Python -> 2.7.13
- openMPI -> 1.6.3


### Instalación

Instala las dependencias para el proyecto.

```
$ conda create -n mpi python=2.7
$ conda install -n mpi numpy
$ conda install -n mpi mpi4pi
```
luego

```
$ source activate mpi
(mpi) $ mpiexec --version
> mpiexec (OpenRTE) 1.6.3
```

luego al final de la definicion de los metodos de cualquiera de las implementaciones se deben cambiar las variables:

```
k = <Numero de clusters (Entero)>
ruta = <ruta de la carpeta que contiene todos los archivos a clusterizar>
```

## Corriendo pruebas

Para el paralelo:

```
(mpi) $ mpiexec -n <number of cores> python ./Paralel.py

```
Para el serial:

```
 $ python Serial.py

```
la salida de ambas implementaciones es:

```
> centroides: <Array de centroides en la primera iteración>
sin relacion: <Set de documentos que no tienen relación a ningún cluster>
clusters: <Map de clusters en donde la clave es el numero de cluster y la key un Array de documentos de ese cluster>
centroides: <Array de nuevos centroides despúes del recentrado>
sin relacion: <Set de documentos que no tienen relación a ningún cluster>
clusters: <Map de clusters despúes de recentrar y recalcular>
time: <Tiempo que tómo la ejecución del algoritmo en segundos>

```


## Authors

* **Daniela Serna Escobar**
* **Daniel Hoyos Ospina**


## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

