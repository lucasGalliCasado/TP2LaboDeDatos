# Autores: Ivan Moyano, Lucas Galli, Sebastian Zubieta Hernandez

# TP2 Laboratorio de Datos - Verano 2022

# En el archivo de encuentran bloques de código correspondientes a la resolución de la consigna del trabajo
# Para asegurar el correcto funcionamiento deben ejeuctarse en orden
#%%
#Imports
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
#%%
    #Carga de datos
datos = pd.read_csv("sign_mnist_train.csv")

#%%
#Funciones

# Para cada pixel de la matrix de greyscale, calculamos la varianza del mismo a lo largo de todas las imagenes
def matrixVarianza(data):
    map_dirty = [[[] for _ in range(28)] for _ in range(28)]
    var_map = [[0 for _ in range(28)] for _ in range(28)]

    for row in data.values:
        mat = row[1:].reshape(28, 28)
        for i in range(28):
            for j in range(28):
                map_dirty[i][j].append(mat[i][j])
    for i in range(28):
        for j in range(28):
            var_map[i][j] = np.var(map_dirty[i][j])
    return var_map

# Reliza un plot de n imagenes aleatorias
def randomLetras(n):
    listaTablaLetrasC = datos[datos['label'] == 2]

    # Numero de filas y columnas
    num_rows = int(np.ceil(n / 3))
    num_cols = min(3, n)

    # subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5), gridspec_kw={'hspace': 0.3})

    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < n:
                # Seleccion de imagen random
                index = random.randint(0, len(listaTablaLetrasC) - 1)
                image_data = listaTablaLetrasC.iloc[index, 1:].values.reshape(28, 28)

                # Plot the image
                axs[i, j].matshow(image_data, cmap="gray")
                axs[i, j].axis('off')  
    plt.show()

#%%##############################################################################################################################
#1a.

# Usamos PCA
pca = PCA(n_components=3)

# Assuming datos is your DataFrame
letras = datos.iloc[:, 0].values
X = datos.iloc[:, 1:].values

# Normalizamos los datos 
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# PCA con 3 componentes
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_normalized)

# Creamos el DataFrame con los datos en tres dimensiones y los labels de las letras
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pc_df = pd.concat([pd.DataFrame(letras, columns=['Letra']), pc_df], axis=1)

# Graficamos las componentes principales

# # El lambda aplicado toma una muestra de 3 instancias de cada letra
sample = pc_df.groupby('Letra', group_keys=False).apply(lambda group: group.sample(n=min(3, len(group)), replace=False))

# Ploteamos
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sample['PC1'], sample['PC2'], sample['PC3'], c='blue', marker='o')

#Nombres
for i, txt in enumerate(sample['Letra']):
    ax.text(sample['PC1'].iloc[i], sample['PC2'].iloc[i], sample['PC3'].iloc[i],txt, size=8, color='red')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('Reduccion a 3-D usando PCA')
plt.show()

# Identificamos los atributos mas importantes (pixeles)

# Accedemos a las constantes phi
loadings = pca.components_
# loadings[i, j] nos retorna el valor de phi para el j-esimo atributo del dataset y la i-esima componente principal (PC_i)

# Buscamos el atributo con el valor de phi mas alto para cada uno de nuestros componentes principales
for i in range(3):
    phi_i = loadings[i]
    max = np.argmax(phi_i)
    #Le sumamos uno a max porque la primera columna indica la letra, luego viene el X que usamos para reducir dimensiones
    print(f"El atributo mas significativo en PC{i+1} es {datos.columns[max+1]}")

#%%##############################################################################################################################
#1b.

#Vamos a formar dos datasets, uno formado por las instancias de 'E' y 'M', otro formado por las instancias de 'L' y 'M'
    
#Visualizemos un primer ejemplo 
L = 11
M = 12
E = 4
#Vamos a formar dos datasets, uno formado por las instancias de 'E' y 'M', otro formado
#por las instancias de 'L' y 'M'
datosLyE =  datos[(datos['label'] == L) | (datos['label'] == E)]
datosLyM =  datos[(datos['label'] == L) | (datos['label'] == M)]

# Plot de un heatmap usando la matriz de varianza

# Matriz de varianza de los datosLyE y datosLyM
varLyE = matrixVarianza(datosLyE)
varLyM = matrixVarianza(datosLyM)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot para datosLyE
axs[0].imshow(varLyE, cmap='hot', interpolation='nearest')
axs[0].set_title('Heatmap de Varianza - L y E')

# Plot for datosLyM
axs[1].imshow(varLyM, cmap='hot', interpolation='nearest')
axs[1].set_title('Heatmap de Varianza - L y M')

# Show the combined plot
plt.show()

# Verificacion de las conclusiones del heatmap con PCA

#PCA para L y M
pcaLyM = PCA(n_components=3)

letras = datosLyM.iloc[:, 0].values
XLyM = datosLyM.iloc[:, 1:].values

# Normalizamos los datos 
scaler = StandardScaler()
XLyM_normalized = scaler.fit_transform(XLyM)

# PCA con 3 componentes
pcaLyM = PCA(n_components=3)
PCLyM = pcaLyM.fit_transform(XLyM_normalized)

# Creamos el DataFrame con los datos en tres dimensiones y los labels de las letras
pcLyM_df = pd.DataFrame(data=PCLyM, columns=['PC1', 'PC2', 'PC3'])
pcLyM_df = pd.concat([pd.DataFrame(letras, columns=['Letra']), pcLyM_df], axis=1)

#PCA para L y E
pcaLyE = PCA(n_components=3)

letras = datosLyE.iloc[:, 0].values
XLyE = datosLyE.iloc[:, 1:].values

# Normalizamos los datos 
scaler = StandardScaler()
XLyE_normalized = scaler.fit_transform(XLyE)

# PCA con 3 componentes
pcaLyE = PCA(n_components=3)
PCLyE = pcaLyE.fit_transform(XLyE_normalized)

# Creamos el DataFrame con los datos en tres dimensiones y los labels de las letras
pcLyE_df = pd.DataFrame(data=PCLyE, columns=['PC1', 'PC2', 'PC3'])
pcLyE_df = pd.concat([pd.DataFrame(letras, columns=['Letra']), pcLyE_df], axis=1)

# El lambda aplicado toma una muestra de 3 instancias de cada letra
sampleLyM = pcLyM_df.groupby('Letra', group_keys=False).apply(lambda group: group.sample(n=min(25, len(group)), replace=False))

sampleM = sampleLyM[sampleLyM['Letra'] == 12]

sampleL = sampleLyM[sampleLyM['Letra'] == 11]

# El lambda aplicado toma una muestra de 3 instancias de cada letra
sampleLyE = pcLyE_df.groupby('Letra', group_keys=False).apply(lambda group: group.sample(n=min(25, len(group)), replace=False))

sampleE = sampleLyE[sampleLyE['Letra'] == 4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': '3d'})

# Ploteos de sampleE y sampleL
ax1.scatter(sampleE['PC1'], sampleE['PC2'], sampleE['PC3'], c='blue', marker='o', label='Sample E')
ax1.scatter(sampleL['PC1'], sampleL['PC2'], sampleL['PC3'], c='red', marker='o', label='Sample L')

for (x, y, z, label) in zip(sampleE['PC1'], sampleE['PC2'], sampleE['PC3'], sampleE['Letra']):
    ax1.text(x, y, z, label, color='blue')

for (x, y, z, label) in zip(sampleL['PC1'], sampleL['PC2'], sampleL['PC3'], sampleL['Letra']):
    ax1.text(x, y, z, label, color='red')

ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.set_title('Reduccion 3-D de L and E')
ax1.legend()

ax2.scatter(sampleM['PC1'], sampleM['PC2'], sampleM['PC3'], c='blue', marker='o', label='Sample M')
ax2.scatter(sampleL['PC1'], sampleL['PC2'], sampleL['PC3'], c='red', marker='o', label='Sample L')

for (x, y, z, label) in zip(sampleM['PC1'], sampleM['PC2'], sampleM['PC3'], sampleM['Letra']):
    ax2.text(x, y, z, label, color='blue')

for (x, y, z, label) in zip(sampleL['PC1'], sampleL['PC2'], sampleL['PC3'], sampleL['Letra']):
    ax2.text(x, y, z, label, color='red')

ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.set_title('Reduccion 3-D de L and M')
ax2.legend()
plt.show()

#%%##############################################################################################################################
#1c.

# Se toman 9 imagenes random correspondiente a la letra c
randomLetras(9)

# Clustering jerarquico

datosC = datos[datos['label'] == 2]
# Sacamos los labels
datosC = datosC.iloc[:, 1:].values

# Normalizamos los datos 
scaler = StandardScaler()
datosC_normalized = scaler.fit_transform(datosC)

# PCA con 3 componentes
pcaC= PCA(n_components=3)
C3D = pcaC.fit_transform(datosC_normalized)

num_samples = 50
random_indices = np.random.choice(C3D.shape[0], size=num_samples, replace=False)
sampleC = C3D[random_indices, :]

# Clustering Jerarquico con single Linkage
Z = linkage(sampleC, method='single')  # You can choose a different linkage method

# Visualizamos el dendograma 
dendrogram(Z)
plt.title('Dendograma de Clustering Jerarquico')
plt.show()

# Menor cantidad posible de clusters
threshold = 1  

clusters = fcluster(Z, t=threshold, criterion='distance')

#Para poner en contexto el dendograma vamos a ver el rango de distincia entre los puntos que tomamos
distances = pdist(sampleC)

#Rango de la distancia de los puntos utilizados
print(f'La maxima distancia entre dos datos es {np.ptp(distances)}') 

#%%##############################################################################################################################

#2.a
# Armamos un DF con datos de L y A
L = 11
A = 0
LyA =  datos[(datos['label'] == L) | (datos['label'] == A)]

#%%##############################################################################################################################

#2.b
#Para ver el balance de datos 
count = LyA['label'].value_counts()
print(count)

#%%##############################################################################################################################

#2.c

# Holdout
LyA_temp, LyA_hold = train_test_split(LyA, test_size=0.1, random_state=457)

#Variables dependientes
X = LyA_temp.iloc[:,1:]
#Variables a predecir
y = LyA_temp['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%##############################################################################################################################

#2.d y e

#Primero vamos a generar una particion de atributos, i.e pixel1,pixel2,...pixel784

#Vamos a usar una random seed para generar agrupaciones de a 3 atributos random(pero recreables) y asi comparar  
# los modelos generado por atributos distintos 

#Tambien, por separado, vamos a tomar los pixeles mas significativos segun el criterio utilizado en 1.a

random.seed(457)

modeloLyA = KNeighborsClassifier(n_neighbors = 3) 

#Para cada cantidad de atributos vamos a realizar 3 modelos distintos seleccionando atributos al azar 
Nrep = 10
cantAtributos = [3,15,50]

#Vamos a guardar los score de los modelos aleatorios para luego comparar 
resultados_test  = np.zeros(( len(cantAtributos), Nrep ))
resultados_train = np.zeros((len(cantAtributos), Nrep ))
resultados_holdout = np.zeros((len(cantAtributos), Nrep ))
# Las filas corresponden a una cantidad de atributos(preservando el oroden del vector cantAtributos) y las columnas corresponden 
# una intancia de un modelo con atributos randomizados  

#Almacenamos los atributos que utilizamos en cada instancia 
atributosXmodelo = [[] for _ in range(len(cantAtributos))]

for n in range(len(cantAtributos)):
    for i in range(Nrep):
        pixeles = random.sample(range(0, 783), cantAtributos[n])
        atributosXmodelo.append(pixeles) 
        X_trainRand = X_train.iloc[:, pixeles]
        X_testRand =  X_test.iloc[:, pixeles]
        neighRand = KNeighborsClassifier(n_neighbors = cantAtributos[n])
        # Entrenamos el modelo (con datos de train)
        neighRand.fit(X_trainRand, y_train) 
        # Evaluamos el modelo con datos de train y luego de test
        resultados_train[n,i] = neighRand.score(X_trainRand, y_train)
        resultados_test[n,i]  = neighRand.score(X_testRand , y_test )
        

#%%# Modelo no Random-------------------------------------------------------------------------------------------------------------------------------------


#Hacemos un modelo con los tres pixeles mas significativo segun el citerio empleado en el 1.a
pixelsSig = [784-1, 263-1, 505-1]
X_trainSig = X_train.iloc[:, pixelsSig]
X_testSig =  X_test.iloc[:, pixelsSig]
neighSig = KNeighborsClassifier(n_neighbors = 3)
# Entrenamos el modelo (con datos de train)
neighSig.fit(X_trainSig, y_train) 
# Evaluamos el modelo con datos de train y luego de test
resultadosSigTrain = neighSig.score(X_trainSig, y_train)
resultadosSigTest  = neighSig.score(X_testSig , y_test)

#%%#Plots----------------------------------------------------------------------------------------------------------------------------------------

# creamos una figura y ejes para cada cantidad de atributos
fig, axs = plt.subplots(len(cantAtributos), 1, figsize=(9, 7))

# iteramos sobre cada cantidad de atributos
for idx, n in enumerate(cantAtributos):
    # Iterar sobre las 5 instancias de un modelo con ésa cantidad de atributos
    
    if (idx==0):
        # ploteamos los resultados de entrenamiento y prueba para cada instancia
        axs[idx].plot(0, resultados_train[idx, i], 'bo', label='train')
        axs[idx].plot(0, resultados_test[idx, i], 'ro', label='test')
        for i in range(1,Nrep):
            # ploteamos los resultados de entrenamiento y prueba para cada instancia
            axs[idx].plot(i, resultados_train[idx, i], 'bo')
            axs[idx].plot(i, resultados_test[idx, i], 'ro')
        
            # configuramos etiquetas y título para el subplot actual
            axs[idx].set_xlabel('Instancia del modelo')
            axs[idx].set_ylabel('Score')
            axs[idx].set_title(f'Resultados para {n} atributos')
            axs[idx].legend()
    
        # ploteamos la parte del ejercicio 1.a
        axs[idx].plot(Nrep, resultadosSigTrain, 'go', label='train (1.a)')
        axs[idx].plot(Nrep, resultadosSigTest, 'yo', label='test (1.a)')
        # configuramos etiquetas y título para el subplot actual
        axs[idx].set_xlabel('Instancia del modelo')
        axs[idx].set_ylabel('Score')
        axs[idx].set_title(f'Resultados para {n} atributos')
        axs[idx].legend()
    

    else:
        # ploteamos los resultados de entrenamiento y prueba para cada instancia
        axs[idx].plot(0, resultados_train[idx, i], 'bo', label='train')
        axs[idx].plot(0, resultados_test[idx, i], 'ro', label='test')
        for i in range(1,Nrep):
            # ploteamos los resultados de entrenamiento y prueba para cada instancia
            axs[idx].plot(i, resultados_train[idx, i], 'bo')
            axs[idx].plot(i, resultados_test[idx, i], 'ro')
        
            # configuramos etiquetas y título para el subplot actual
            axs[idx].set_xlabel('Instancia del modelo')
            axs[idx].set_ylabel('Score')
            axs[idx].set_title(f'Resultados para {n} atributos')
            axs[idx].legend()

# ajusto el espacio entre subplots
plt.tight_layout()

# gráfico
plt.show()

#%%#Hold-out---------------------------------------------------------------------------------------------------------------------------------

#Vamos a tomar un modelo por cada cantidad de atributos y medir su error usando el metodo hold-out.

#IMPORTANTE: esto depende de la random seed que fue seleccionada, la cual es 457 en esta instancia

#Vamos a tomar los 3 modelos Random de mejor rendimiento y ademas el modelo basado en los 3 px dados por el 1.a como instancias definitivas.

#- 1.a de 3px
 
#- Modelo#6 de 3px (modelo numero 6 en atributosXmodelo)

#- Modelo#2 de 15px (modelo numero 12 en atributosXmodelo)

#- Modelo#5 de 50px (modelo numero 25 en atributosXmodelo)

#Emparejamos los indices para re-crear los modelos 
atributosXmodelo5 = atributosXmodelo[5].copy()

for i in range(len(atributosXmodelo5)):
    atributosXmodelo5[i] = atributosXmodelo5[i] -1    

atributosXmodelo11 = atributosXmodelo[11].copy()

for i in range(len(atributosXmodelo11)):
    atributosXmodelo11[i] = atributosXmodelo11[i] -1    

atributosXmodelo24 = atributosXmodelo[24].copy()

for i in range(len(atributosXmodelo24)):
    atributosXmodelo24[i] = atributosXmodelo24[i] -1
        
#%%#Re-creamos los modelos y evaluamos sus errores usando Hold-out ---------------------------------------------------------------

#Modelo Definitivo 1:

pixelsSig = [784-1, 263-1, 505-1]
X_trainSig = X_train.iloc[:, pixelsSig]
neighSig = KNeighborsClassifier(n_neighbors = 3)
# Entrenamos el modelo (con datos de train)
neighSig.fit(X_trainSig, y_train) 

# Error con Hold-out de modelo de 3 Pixeles del item 1.a.)
error3Pix1a = neighSig.score(LyA_hold.iloc[:,[784, 263, 505]],LyA_hold['label'])
print(f"Error de modelo final de item 1.a con 3px: {error3Pix1a}")

#---------------------------------------------------------------------------------------------------------------------------------

#Modelo Definitivo 2:
X_trainTemp = X_train.iloc[:, atributosXmodelo5]
neighTemp = KNeighborsClassifier(n_neighbors = 3)
neighTemp.fit(X_trainTemp, y_train) 

# Error con Hold-out de modelo de 15 Pixeles
error3Pix = neighTemp.score(LyA_hold.iloc[:,atributosXmodelo[5]],LyA_hold['label'])

print(f"Error de modelo final con 3px seleccionados con random: {error3Pix}")

#---------------------------------------------------------------------------------------------------------------------------------

#Modelo Definitivo 3:

#Re-creamos el modelo
X_trainTemp = X_train.iloc[:, atributosXmodelo11]
neighTemp = KNeighborsClassifier(n_neighbors = 15)
neighTemp.fit(X_trainTemp, y_train) 

# Error con Hold-out de modelo de 15 Pixeles
error15Pix = neighTemp.score(LyA_hold.iloc[:,atributosXmodelo[11]],LyA_hold['label'])

print(f"Error de modelo final con 15px: {error15Pix}")
#---------------------------------------------------------------------------------------------------------------------------------

#Modelo Definitivo 4:

X_trainTemp = X_train.iloc[:, atributosXmodelo24]
neighTemp = KNeighborsClassifier(n_neighbors = 50)
neighTemp.fit(X_trainTemp, y_train) 

# Error con Hold-out de modelo de 50 Pixeles
error50Pix = neighTemp.score(LyA_hold.iloc[:,atributosXmodelo[24]],LyA_hold['label'])

print(f"Error de modelo final con 50px: {error50Pix}")







#%%#############################################################################################################################
#3a.

# importacion de los datos de vocales
vowels = datos[(datos['label'] == 0) | (datos['label'] == 4) | (datos['label'] == 8) |
                (datos['label'] == 14) | (datos['label'] == 20)]
# separacion de los datos train, test y eval
y = vowels.label

X_train, X_temp, y_train, y_temp = train_test_split(vowels, y, train_size=0.7,
                                                    random_state=7, stratify=y, shuffle=True)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                    random_state=7, stratify=y_temp, shuffle=True)

#%%##############################################################################################################################
#3b.

# lista que contiene 4 modelos de arbol de profundidad: indice + 1
trees = [DecisionTreeClassifier(criterion="entropy", max_depth=i+1) for i in range(5)]
# se entrenan los modelos
for tree in trees:
    tree.fit(X_train, y_train)
# score individual de cada modelo
scores = [accuracy_score(y_test, tree.predict(X_test)) for tree in trees]

#Graficamos el score individual (profundidad -> score)
depths = list(range(1,6))
plt.plot(depths, scores, marker='o')
plt.title('Profundidad del Árbol vs Score')
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Score')
plt.xticks(depths)
plt.grid(True)
plt.show()

#%%##############################################################################################################################
#3c.

# lista que contiene el score de cada modelo segun cross validation
k_fold_scores = [cross_val_score(tree, X_train, y_train, cv=4) for tree in trees]
# se queda con la media del score de los folds
for i in range(len(k_fold_scores)):
    k_fold_scores[i] = sum(k_fold_scores[i])/len(k_fold_scores[i])

# graficamos el cross validation score (profundidad -> score)
plt.plot(depths, k_fold_scores, marker='o')
plt.title('Profundidad del Árbol vs Cross Validation Score')
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Cross Validation Score')
plt.xticks(depths)
plt.grid(True)
plt.show()

#%%##############################################################################################################################
#3d.

conf_mat = []
for tree in trees:
    #labels 0, 4, 8, 14, 20 corresponden a, e, i, o, u respectivamente
    conf_mat.append(confusion_matrix(y_test, tree.predict(X_test), labels=[0, 4, 8, 14, 20]))

#gráfico matrices de confusión
fig, axs = plt.subplots(1, 5, figsize=(12, 6))
fig.suptitle('Matriz de confusión por profundidad', fontsize=20)
for i in range(len(conf_mat)):
    axs[i].imshow(conf_mat[i])
    axs[i].set_title('Profundidad ' + str(i+1))
    axs[i].set_xticks([i for i in range(0, 5)], labels=['a', 'e', 'i', 'o', 'u'])
    axs[i].set_yticks([i for i in range(0, 5)], labels=['a', 'e', 'i', 'o', 'u'])
    for x in range(5):
        for y in range(5):
            axs[i].text(y, x, conf_mat[i][x, y], ha="center", va="center", color="w")
plt.tight_layout()
plt.show()

#score final del modelo de profundidad 3 usando los datos eval
print("#########", "score final del tree con profundidad 3:", accuracy_score(y_eval, trees[2].predict(X_eval)), "#########")