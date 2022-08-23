---
jupyter:
  kernelspec:
    display_name: Python 3.9.12 (\'base\')
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.12
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
  vscode:
    interpreter:
      hash: 3a95b72ad26caed2f125c20bb8f543b875a0d1a71c369d3d6c01c0c3af4e63b9
---

::: {.cell .code execution_count="150"}
``` {.python}
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import pandas as pn
```
:::

::: {.cell .markdown}
# K-Means
:::

::: {.cell .markdown}
## Generador de muestras
:::

::: {.cell .markdown}
### Condiciones:

Definimos el número de clusters que se generarán, el número máximo de
iteraciones y el número de muestras o datos que se generarán
:::

::: {.cell .code execution_count="151"}
``` {.python}
n_clusters = 3
max_iterations = 50
n_samples = 400
```
:::

::: {.cell .markdown}
Se definen aleatoriamente los centroides reales de los clusters
:::

::: {.cell .code execution_count="152"}
``` {.python}
true_centroid = [[ rn.uniform(-10,10)  for i in range(0, n_clusters)], [rn.uniform(-10,10)  for i in range(0, n_clusters)]]
print(true_centroid)
```

::: {.output .stream .stdout}
    [[-8.96303374168955, -0.13386471558721524, -3.1141202962638603], [1.153616487573494, 3.889960539462882, -6.78776637626364]]
:::
:::

::: {.cell .markdown}
Generamos los n datos de acuerdo a la condición \'n_samples\'. Estos
datos se generan aleatoriamente alrededor de cada centroide.
:::

::: {.cell .code execution_count="153"}
``` {.python}
ndata = n_samples//n_clusters
datax = []
datay = []
for j in range(n_clusters):  
    for i in range(0, ndata):  
        datax.append(true_centroid[0][j]+rn.random()*rn.uniform(-10,10) )
        datay.append(true_centroid[1][j]+rn.random()*rn.uniform(-10,10) )
data = [datax, datay]
plt.scatter(data[0][:], data[1][:])
plt.scatter(true_centroid[0][:], true_centroid[1][:], c='black', s=200, alpha=0.5)
```

::: {.output .execute_result execution_count="153"}
    <matplotlib.collections.PathCollection at 0x1d4af40ebb0>
:::

::: {.output .display_data}
![](vertopal_dfbe5d48b4c04db08839a5f7c687f1a6/3b1568b2da6feece90523e82b7a67add2935c81c.png)
:::
:::

::: {.cell .markdown}
## Algoritmo de K-Means
:::

::: {.cell .markdown}
El objetivo es clasificar todos los datos en clusters. Para ello es
necesario crear n centroides ubicados aleatoriamente. Posteriormente se
debe de obtener un vector para cada dato y centroide, con ello obtenemos
la distancia mínima de cada dato y centroide, para guardar ese punto en
el cluster con el centroide más cercano. Posteriormente se calcula el
centro de cada cluster y se redefine el centroide y el proceso anterior
se repite hasta alcanzar las n interaciones máximas.
:::

::: {.cell .markdown}
En la siguiente función se crea cada cluster vacio y los n centroides
ubicados aleatoriamente.
:::

::: {.cell .code execution_count="154"}
``` {.python}
def create_clusters():
    clustersDic = {'cluster'+str(i): [] for i in range(n_clusters)}
    centroides = [[rn.uniform(np.min(data[0]),np.max(data[0])), rn.uniform(np.min(data[1]),np.max(data[1]))] for i in range(n_clusters)]
    return clustersDic, centroides
```
:::

::: {.cell .markdown}
Definimos una función para calcular la distancia entre dos puntos.
:::

::: {.cell .code execution_count="155"}
``` {.python}
def distance(point1, point2):
    distace = np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return distace
```
:::

::: {.cell .markdown}
En la siguiente función se ingresa un cluster no vacío para encontrar el
nuevo centroide de su conjunto de datos.
:::

::: {.cell .code execution_count="156"}
``` {.python}
def findCentroid(clusters):
    centroids = []
    for points in clusters:
        points = np.array(points).T
        centroids.append([np.mean(points[0]), np.mean(points[1])])
    return centroids
```
:::

::: {.cell .markdown}
Para cada iteración, los clusters se redefinen clasificando cada dato de
acuerdo a un nuevo centroide.
:::

::: {.cell .code execution_count="157"}
``` {.python}
def update_cluster(centroids):
    clustersDis = [[] for i in range(n_clusters)]
    for i in range(len(data[0])):
        dis = [distance([data[0][i],data[1][i]], centroids[j]) for j in range(n_clusters)]
        minDist = dis.index(np.min(dis))
        clustersDis[minDist].append([data[0][i],data[1][i]])
        clustersDis = np.array(clustersDis)
    centroids = findCentroid(clustersDis)
    return clustersDis, centroids
```
:::

::: {.cell .code execution_count="158"}
``` {.python}
color = []
for i in range(n_clusters):
    color.append('#%06X' % rn.randint(0, 0xFFFFFF))
```
:::

::: {.cell .markdown}
Finalmente se definen los clusters vacios y los centroides aleatorios y
se comienza con los procesos iterativos de clasificación de datos y para
encontrar el centroide más cercano al real.
:::

::: {.cell .code execution_count="159"}
``` {.python}
clusters, centroids = create_clusters()
for i in range(max_iterations):
     clusters, centroids = update_cluster(centroids)
for j in range(n_clusters):
    for point in clusters[j]:
        plt.plot(point[0],point[1],'o', color = color[j])
    plt.scatter(centroids[j][0], centroids[j][1], c='black', s=200, alpha=0.8)
```

::: {.output .stream .stderr}
    C:\Users\baruc\AppData\Local\Temp\ipykernel_2328\4039889107.py:7: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      clustersDis = np.array(clustersDis)
:::

::: {.output .display_data}
![](vertopal_dfbe5d48b4c04db08839a5f7c687f1a6/cfea12e2c7db114bf929cf52e15d77411120193c.png)
:::
:::
