#Diplomado python 
#Autor: Jonnier Andres Teran Morales 
#ID No:502195
#Id:1003064599
#ID:correo:Jonnier.teran@upb.edu.co
#Cel:3255644212


#Importar  librerias 
# Las librerías utilizadas para el tratamiento de datos
# ================================================= =============================
import pandas as pd
import numpy as np

#Graficos 
# Las librerías utilizadas  para Gráficos
# ================================================= =============================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Las librerias usadas para el preprocesado y modelado
# ================================================= =============================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
 

# Las librerías utilizadas  para la configuración matplotlib
# ================================================= =============================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')


# Advertencias de configuración warnings
# ================================================= =============================
import warnings
warnings.filterwarnings('ignore')



#DATOS 
#Aqui almacenamos el conjunto de datos

# Datos 
# ================================================= =============================
matrícula  =  np . matriz ([ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 1 , 0 , 0 , 0 , 0 , 1 ,
                     0 , 0 , 1 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 1 , 0 , 1 ,
                     0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 0 ,
                     0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 ,
                     1 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 1 , 0 , 0 , 0 , 0 , 1 , 0 , 1 , 0 , 0 ,
                     1 , 1 , 1 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 ,
                     1 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 ,
                     0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 ,
                     0 , 0 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 0 , 0 ,
                     0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 ,
                     0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 ])

matematicas  =  np . matriz ([
                  41 , 53 , 54 , 47 , 57 , 51 , 42 , 45 , 54 , 52 , 51 , 51 , 71 , 57 , 50 , 43 ,
                  51 , 60 , 62 , 57 , 35 , 75 , 45 , 57 , 45 , 46 , 66 , 57 , 49 , 49 , 57 , 64 ,
                  63 , 57 , 50 , 58 , 75 , 68 , 44 , 40 , 41 , 62 , 57 , 43 , 48 , 63 , 39 , 70 ,
                  63 , 59 , 61 , 38 , 61 , 49 , 73 , 44 , 42 , 39 , 55 , 52 , 45 , 61 , 39 , 41 ,
                  50 , 40 , 60 , 47 , 59 , 49 , 46 , 58 , 71 , 58 , 46 , 43 , 54 , 56 , 46 , 54 ,
                  57 , 54 , 71 , 48 , 40 , 64 , 51 , 39 , 40 , 61 , 66 , 49 , 65 , 52 , 46 , 61 ,
                  72 , 71 , 40 , 69 , 64 , 56 , 49 , 54 , 53 , 66 , 67 , 40 , 46 , 69 , 40 , 41 ,
                  57 , 58 , 57 , 37 , 55 , 62 , 64 , 40 , 50 , 46 , 53 , 52 , 45 , 56 , 45 , 54 ,
                  56 , 41 , 54 , 72 , 56 , 47 , 49 , 60 , 54 , 55 , 33 , 49 , 43 , 50 , 52 , 48 ,
                  58 , 43 , 41 , 43 , 46 , 44 , 43 , 61 , 40 , 49 , 56 , 61 , 50 , 51 , 42 , 67 ,
                  53 , 50 , 51 , 72 , 48 , 40 , 53 , 39 , 63 , 51 , 45 , 39 , 42 , 62 , 44 , 65 ,
                  63 , 54 , 45 , 60 , 49 , 48 , 57 , 55 , 66 , 64 , 55 , 42 , 56 , 53 , 41 , 42 ,
                  53 , 42 , 60 , 52 , 38 , 57 , 58 , 65 ])

datos = pd.DataFrame({'matricula': matricula, 'matematicas': matematicas})
datos.head(3)


#Exploracion grafica
#Aqui representamos los datos para intuir si existe relacion entre las variables 

# Número de observaciones por clase
# ================================================= =============================
datos.matricula.value_counts().sort_index()
#============================================== =============================

#Gráfico

fig, ax = plt.subplots(figsize=(6, 3.84))

sns.violinplot(
        x     = 'matricula',
        y     = 'matematicas',
        data  = datos,
        #color = "white",
        ax    = ax
    )

ax.set_title('Distribución notas de matemáticas por clase');


# T-test entre clases
# Esta información es útil para considerar la nota de matemáticas como un buen predictor para el modelo.
#============================= =============================

res_ttest = ttest_ind(
                x1 = matematicas[matricula == 0],
                x2 = matematicas[matricula == 1],
                alternative='two-sided'
            )
print(f"t={res_ttest[0]}, p-value={res_ttest[1]}")




#Ajuste del modelo 
# Como en todo estudio predictivo, no solo es importante ajustar el modelo, sino también cuantificar su capacidad para predecir nuevas observaciones.

#Scikit-learn

# División de los datos en train y test 
# ================================================ =============================

X = datos[['matematicas']]
y = datos['matricula']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
# Creación del modelo 
# ================================================= =============================
# Para no incluir ningún tipo de regularización en el modelo se indica
# penalización='ninguna'

modelo = LogisticRegression(penalty='none')
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)


# Información del modelo
# ================================================= =============================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Accuracy de entrenamiento:", modelo.score(X, y))




# Predicciones probabilísticas
# ================================================= =============================
# Con .predict_proba() se obtiene, para cada observación, la probabilidad predicha
# de pertenecer a cada una de las dos clases.

predicciones = modelo.predict_proba(X = X_test)
predicciones = pd.DataFrame(predicciones, columns = modelo.classes_)
predicciones.head(3)



# Predicciones con clasificacion final
# ================================================= =============================
# Con .predict() se obtiene, para cada observación, la clasificación predicha por
# el modelo. Esta clasificación se corresponde con la clase con mayor probabilidad.

predicciones = modelo.predict(X = X_test)
predicciones


#Stantmodels 

# División de los datos en train y test
# ================================================= =============================

X = datos[['matematicas']]
y = datos['matricula']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True

# Creación del modelo utilizando el modo fórmula (similar a R)
# ==============================================================================
# datos_train = pd.DataFrame(np.hstack((X_train, y_train)),
#                            columns=['matematicas', 'matricula'])
# modelo = smf.logit(formula = 'matricula ~matematicas', data = datos_train)
# modelo = modelo.fit()
# print(modelo.summary())

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.Logit(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())

#Intervalos de confianza de los coeficientes 

# Intervalos de confianza para los coeficientes del modelo
# ==============================================================================
intervalos_ci = modelo.conf_int(alpha=0.05)
intervalos_ci = pd.DataFrame(intervalos_ci)
intervalos_ci.columns = ['2.5%', '97.5%']
intervalos_ci

#Predicciones
#Los modelos de regresion de statsmodels devuelven la probabilidad de pertenecer a la clase de referencia 

#Predicción de probabilidades
# ================================================= =============================
predicciones = modelo.predict(exog = X_train)
predicciones[:4]



# clasificación predicha
# ================================================= =============================
clasificacion = np.where(predicciones<0.5, 0, 1)
clasificacion

#Representacion grafica del modelo 

#Permite encontrar el valor promedio de la variable respuesta 

# Predicciones en todo el rango de X
# ==============================================================================
# Se crea un vector con nuevos valores interpolados en el rango de observaciones.
grid_X = np.linspace(
            start = min(datos.matematicas),
            stop  = max(datos.matematicas),
            num   = 200
         ).reshape(-1,1)

grid_X = sm.add_constant(grid_X, prepend=True)
predicciones = modelo.predict(exog = grid_X)


#Gráfico del modelo
# ================================================= =============================
fig, ax = plt.subplots(figsize=(6, 3.84))

ax.scatter(
    X_train[(y_train == 1).flatten(), 1],
    y_train[(y_train == 1).flatten()].flatten()
)
ax.scatter(
    X_train[(y_train == 0).flatten(), 1],
    y_train[(y_train == 0).flatten()].flatten()
)
ax.plot(grid_X[:, 1], predicciones, color = "gray")
ax.set_title("Modelo regresión logística")
ax.set_ylabel("P(matrícula = 1 | matemáticas)")
ax.set_xlabel("Nota matemáticas");


#Accuracy test 
#calcula el porcentaje de aciertos que tiene el modelo al predecir las observaciones 


# Accyracy de test del modelo  
# ================================================= =============================

X_test = sm.add_constant(X_test, prepend=True)
predicciones = modelo.predict(exog = X_test)
clasificacion = np.where(predicciones<0.5, 0, 1)
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = clasificacion,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")


#Los resultados obtenidos con el conjunto de test indican que el modelo es capaz de clasificar el 87.5% de las observaciones 


