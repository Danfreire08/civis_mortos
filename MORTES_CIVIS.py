import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

# Geração de dados de exemplo (substitua com seus dados reais)
X, y = datasets.make_regression(n_samples=149, n_features=50, noise=10, random_state=42)

# Imprimir a forma da matriz de dados e seu primeiro elemento
print(X.shape)
print(X[0])

# Definição de variáveis
tempo_evacuacao_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
municao_utilizada_blindado_range = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
alcance_municao_blindado_range = [600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200]
municao_utilizada_de_morteiro_range = [64]
alcance_municao_morteiro_range = [5600]
civis_na_localidade_range = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
num_soldados_range = [30, 60, 90, 180, 360]
tempo_planejamento_range = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 72]
densidade_demografica_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
disparos_inimigo_range = [10000, 20000, 30000, 40000, 50000, 100000, 15000, 20000, 250000, 30000, 35000, 40000, 45000, 50000]
duração_dias_combate_range = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

# Encontrar o comprimento máximo das variáveis
max_length = max(len(tempo_evacuacao_range), len(municao_utilizada_blindado_range),
                 len(alcance_municao_blindado_range), len(municao_utilizada_de_morteiro_range),
                 len(alcance_municao_morteiro_range), len(civis_na_localidade_range),
                 len(num_soldados_range), len(tempo_planejamento_range),
                 len(densidade_demografica_range), len(disparos_inimigo_range))

# Ajuste da matriz X
X = np.column_stack((
    np.repeat(tempo_evacuacao_range, max_length)[:max_length],
    np.repeat(municao_utilizada_blindado_range, max_length)[:max_length],
    np.repeat(alcance_municao_blindado_range, max_length)[:max_length],
    np.repeat(municao_utilizada_de_morteiro_range, max_length)[:max_length],
    np.repeat(alcance_municao_morteiro_range, max_length)[:max_length],
    np.repeat(civis_na_localidade_range, max_length)[:max_length],
    np.repeat(num_soldados_range, max_length)[:max_length],
    np.repeat(tempo_planejamento_range, max_length)[:max_length],
    np.repeat(densidade_demografica_range, max_length)[:max_length],
    np.repeat(disparos_inimigo_range, max_length)[:max_length]
))

# Redefinindo a variável alvo 'y'
y = np.random.rand(X.shape[0])

# Redefinindo X para a coluna desejada
X = X[:, 2]
X = X.reshape((-1, 1))

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

# Criação e treinamento do modelo de regressão linear
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Previsão com dados de teste
y_pred = model.predict(X_test)

# Plotagem do gráfico
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Variável X')
plt.ylabel('Variável alvo y')
plt.title('Um gráfico mostrando a relação entre X e y')
plt.show()

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

# Criação e treinamento do modelo de regressão linear
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Previsão com dados de teste
y_pred = model.predict(X_test)

# Plotagem do gráfico
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Variável X')
plt.ylabel('Variável alvo y')
plt.title('Um gráfico mostrando a relação entre X e y')
plt.show()

