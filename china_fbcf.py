import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

caminho = r"C:\Users\victo\Downloads\china_fbcf.csv"

dados = pd.read_csv(caminho)

x = sm.add_constant(dados['Taxa de Juros (%)'])
y = dados['FBCF (CNY centenas de milhões)']

modelo = sm.OLS(y,x).fit()

print(modelo.summary())

# dados["FBCF_Prev"] = modelo.predict(x)

# Gráfico
# plt.scatter(dados["Taxa de Juros (%)"], dados["FBCF (CNY centenas de milhões)"], label="Dados Reais")
# plt.plot(dados["Taxa de Juros (%)"], dados["FBCF_Prev"], color="red", label="Regressão Linear")
# plt.xlabel("Taxa de Juros (%)")
# plt.ylabel("FBCF (CNY centenas de milhões)")
# plt.legend()
# plt.show()

b0 = modelo.params[0]
b1 = modelo.params[1]

print(f"Fórmula da regressão: FBCF = {b0:.2f} + {b1:.2f} * Taxa de Juros")