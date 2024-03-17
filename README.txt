Utilizando los datos de lenguaje de señas de MNITS, ver https://www.kaggle.com/datasets/datamunge/sign-language-mnist hacemos un analisis de datos utilizando herramientas de reduccion de dimension y luego implementamos 
algunos modelos basicos de aprendizaje automatico supervisado.

La ejecucion del codigo debe respetar el orden de los bloques definidos en el .py (o el jupyter notebook)

Importaciones Requeridas;
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
