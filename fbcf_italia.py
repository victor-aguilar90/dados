import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

caminho = r"C:\Users\23025173\Downloads\FBCF_Italy.xlsx"

dados = pd.read_excel(caminho, engine="openpyxl")

correlacao = dados['FBCF'].corr(dados['Taxa Real'])

x = sm.add_constant(dados['Taxa Real'])
y = dados['FBCF']

modelo = sm.OLS(y,x).fit()

dados['FBCF_PREV'] = modelo.predict(x)

plt.scatter(dados['Taxa Real'], dados['FBCF'], label="Dados Reais")
plt.plot(dados["Taxa Real"], dados["FBCF_PREV"], color="red", label="Regressão Linear")
plt.xlabel("Taxa Real")
plt.ylabel("FBCF")
plt.legend()
plt.show()


print(modelo.summary())