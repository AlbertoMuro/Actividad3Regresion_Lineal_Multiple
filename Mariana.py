#Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from turtle import color
import statsmodels.api as sm

#Carga de Archivos
credicel = pd.read_csv("df_limpio.csv")
credicel.info()

# Funciones
def correlacion_determinacion(variableX, variableY, df):
    model = LinearRegression()
  
    # Asegurar que variableY sea un arreglo de numpy unidimensional
    if isinstance(variableY, pd.Series):
        variableY = variableY.values
    model.fit(variableX, variableY)
    # Calcular coeficiente de determinación
    coef_determinacion = model.score(variableX, variableY)
    # Calcular coeficiente de correlación
    coef_correlacion = np.sqrt(coef_determinacion)
    # Mostrar coeficientes de determinación y correlación
    print("Coeficiente de determinación:", coef_determinacion)
    print("Coeficiente de correlación:", coef_correlacion)
    model = LinearRegression()
    print(type(model))
    print(model._dict_)

print("ENGANCHE")

#Enganche 1
    # semana
    # monto_financiado
    # Costo_total
    # monto_accesorios
    # Puntos
    # score_buro

#Declaración de variables
x_para_enganche1 = credicel[["semana", "monto_financiado", "costo_total", "monto_accesorios", "puntos", "score_buro"]]
y_enganche = credicel["enganche"]

#Modelo
enganche_modelo1 = correlacion_determinacion(x_para_enganche1, y_enganche, credicel)

#enganche 2. ESTE ES EL MEJOR
x_para_enganche2 = credicel[["precio", "semana", "descuento", "monto_financiado", "costo_total", "monto_accesorios", "status", "fraude", "inversion", "pagos_realizados", "reautorizacion", "puntos", "porc_eng", "score_buro", "semana_actual", "riesgo", "limite_credito"]]

#modelo
enganche_modelo2 = correlacion_determinacion(x_para_enganche2, y_enganche, credicel)

#enganche 3
x_para_enganche3 = credicel[["precio", "descuento", "monto_financiado", "costo_total", "limite_credito", "riesgo", "monto_accesorios", "inversion", "reautorizacion", "puntos", "score_buro"]]

#modelo
enganche_modelo3 = correlacion_determinacion(x_para_enganche3, y_enganche, credicel)

print("PAGOS REALIZADOS")

#Pagos realizados 1
    # monto_financiado
    # Costo_total
    # enganche
x_para_pagos_realizados1 = credicel[["monto_financiado", "costo_total", "enganche"]]
y_pagos_realizados = credicel["pagos_realizados"]

#Modelo
pagos_realizados_modelo1 = correlacion_determinacion(x_para_pagos_realizados1, y_pagos_realizados, credicel)

#pagos realizados 2 Caso raro ESTE ES EL MEJOR (CORRELACION Y DETERMINACION 1)
x_para_pagos_realizados2 = credicel[["precio", "semana", "descuento", "monto_financiado", "costo_total", "monto_accesorios", "status", "fraude", "inversion", "pagos_realizados", "enganche", "reautorizacion", "puntos", "porc_eng", "score_buro", "semana_actual", "riesgo", "limite_credito"]]

#modelo
pagos_realizados_modelo2 = correlacion_determinacion(x_para_pagos_realizados2, y_pagos_realizados, credicel)

#Pagos realizados 3
x_para_pagos_realizados3 = credicel[["precio", "descuento", "monto_financiado", "costo_total", "riesgo", "monto_accesorios", "inversion", "puntos", "enganche", "semana", "semana_actual"]]

#modelo
pagos_realizados_modelo3 = correlacion_determinacion(x_para_pagos_realizados3, y_pagos_realizados, credicel)

print("PORCENTAJE DE ENGANCHE")

#Porcentaje de enganche 1
    # monto_financiado
    # Costo_total
    # Enganche
    # Precio
    # descuento
    # limite_credito
x_para_porc_eng1 = credicel[["monto_financiado", "costo_total", "enganche", "precio", "descuento", "limite_credito"]]
y_porc_eng = credicel["porc_eng"]

#Modelo
porc_eng_modelo_1 = correlacion_determinacion(x_para_porc_eng1, y_porc_eng, credicel)

#Porcentaje de enganche 2 ESTE ES EL MEJOR
x_para_porc_eng2 = credicel[["precio", "semana", "descuento", "monto_financiado", "costo_total", "monto_accesorios", "status", "fraude", "inversion", "pagos_realizados", "enganche", "reautorizacion", "puntos", "pagos_realizados", "score_buro", "semana_actual", "riesgo", "limite_credito"]]

#modelo
porc_eng_modelo_2 = correlacion_determinacion(x_para_porc_eng2, y_porc_eng, credicel)

#Porcentaje de enganche 3
x_para_porc_eng3 = credicel[["precio", "enganche", "descuento", "monto_financiado", "costo_total", "riesgo", "monto_accesorios", "inversion", "puntos", "enganche", "semana", "semana_actual", "inversion", "reautorizacion", "score_buro"]]

#modelo
porc_eng_modelo_3 = correlacion_determinacion(x_para_porc_eng3, y_porc_eng, credicel)

print("__ _ _")
print("MODELOS FINALES")
print("ENGANCHE")

model = LinearRegression()
print(type(model))
model.fit(X = x_para_enganche2, y = y_enganche)
print(model._dict_)

print("PAGOS REALIZADOS")

model = LinearRegression()
print(type(model))
model.fit(X = x_para_pagos_realizados2, y = y_pagos_realizados)
print(model._dict_)

print("PORCENTAJE DE ENGANCHE")

model = LinearRegression()
print(type(model))
model.fit(X = x_para_porc_eng2, y = y_porc_eng)
print(model._dict_)