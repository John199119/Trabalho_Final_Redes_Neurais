# Trabalho_Final_Redes_Neurais

Análise de Redes Neurais para Operações Matemáticas
Comparação de Modelos com Diferentes Funções de Ativação

Resumo
Preparação dos Dados
Arquitetura dos Modelos
Treinamento
Resultados
Conclusão
Testar Modelo
Resumo do Projeto
Este projeto desenvolve e analisa diferentes arquiteturas de redes neurais artificiais aplicadas à resolução de operações matemáticas básicas (adição, subtração, multiplicação e divisão). Foram implementados três modelos utilizando funções de ativação distintas: ReLU (Swish), LeakyReLU e Tanh, com arquiteturas aprimoradas de quatro camadas densas de 128 neurônios cada, incorporando técnicas de regularização como Dropout para prevenção de overfitting.

O treinamento dos modelos foi otimizado com o uso do otimizador Adam, configurado com uma taxa de aprendizado reduzida (0.0005) para garantir maior estabilidade no processo de aprendizado. A técnica de EarlyStopping foi utilizada para evitar sobreajuste, interrompendo o treinamento automaticamente em caso de estagnação da melhoria do erro de validação.

Os resultados obtidos foram analisados através de gráficos de evolução da Loss e da MAE (Mean Absolute Error), além de tabelas detalhadas comparando o desempenho de cada modelo em novos exemplos aleatórios. A escolha do melhor modelo foi feita de maneira automatizada, utilizando o menor valor de erro de validação como critério.

Este trabalho demonstra a eficácia da utilização de técnicas modernas de otimização e arquitetura em redes neurais aplicadas a tarefas de regressão, além de reforçar a importância de práticas como regularização, validação cruzada e análise comparativa de resultados para a construção de modelos preditivos robustos.

Preparação dos Dados
Nesta etapa inicial, o projeto gera um conjunto de dados sintético para treinar os modelos de redes neurais. O processo inclui a geração de 20.000 exemplos de operações matemáticas básicas, normalização dos dados e divisão em conjuntos de treino e teste.

Importação das Bibliotecas
# Importação das bibliotecas necessárias
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tqdm.notebook import tqdm
from tensorflow.keras import metrics
from tensorflow.keras.saving import register_keras_serializable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
Preparação dos Dados
# 1. Preparação dos Dados
# Definição da semente para garantir reprodutibilidade dos resultados
np.random.seed(50)

# Função para gerar, normalizar e dividir os dados
def prepare_data():
    # Definição da quantidade de exemplos que serão gerados
    n_samples = 20000

    # Geração de números aleatórios entre 0 e 10 para x1 e x2
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(0, 10, n_samples)

    # Escolha aleatória da operação a ser realizada para cada par de números
    operations = np.random.choice(['+', '-', '*', '/'], size=n_samples)

    # Lista para armazenar os resultados das operações
    results = []

    # Realização das operações matemáticas
    for a, b, op in zip(x1, x2, operations):
        if op == '+':
            results.append(a + b)
        elif op == '-':
            results.append(a - b)
        elif op == '*':
            results.append(a * b)
        elif op == '/':
            if b == 0:  # Tratamento para evitar divisão por zero
                b = 1e-6
            results.append(a / b)

    # Mapeamento das operações para valores numéricos: + → 0, - → 1, * → 2, / → 3
    op_map = {'+': 0, '-': 1, '*': 2, '/': 3}
    operations_num = np.array([op_map[op] for op in operations])

    # Criação da matriz de entrada X combinando x1, x2 e o código da operação
    X = np.column_stack((x1, x2, operations_num))

    # Conversão dos resultados para um array numpy
    y = np.array(results)

    # Aplicação da normalização nos dados de entrada
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Retorno dos conjuntos preparados e do objeto scaler
    return X_train, X_test, y_train, y_test, scaler

# Chamada da função para gerar e preparar os dados
X_train, X_test, y_train, y_test, scaler = prepare_data()
A normalização é uma etapa crucial que ajuda a melhorar a convergência dos modelos de redes neurais, garantindo que todas as variáveis de entrada estejam na mesma escala. A divisão em conjuntos de treino e teste permite avaliar a capacidade de generalização dos modelos para dados não vistos durante o treinamento.

O conjunto de dados gerado inclui quatro operações matemáticas básicas (adição, subtração, multiplicação e divisão) aplicadas a pares de números aleatórios entre 0 e 10. As operações são codificadas numericamente para facilitar o processamento pela rede neural.

Arquitetura dos Modelos
Foram implementados três modelos de redes neurais com arquiteturas semelhantes, diferenciando-se principalmente pela função de ativação utilizada. Todos os modelos compartilham uma estrutura de quatro camadas densas com 128 neurônios cada, seguidas por camadas de Dropout para regularização.

Definição dos Modelos
# 2. Arquitetura da Rede Neural

# Função para criar um modelo de rede neural MLP
def build_model(activation='swish', regularization=None):
    # Inicializar um modelo sequencial
    model = keras.Sequential()

    # Primeira camada densa
    model.add(layers.Dense(128, activation=activation, input_shape=(3,),
                           kernel_regularizer=regularization))
    model.add(layers.Dropout(0.1))

    # Segunda camada densa
    model.add(layers.Dense(128, activation=activation,
                           kernel_regularizer=regularization))
    model.add(layers.Dropout(0.1))

    # Terceira camada densa
    model.add(layers.Dense(128, activation=activation,
                           kernel_regularizer=regularization))
    model.add(layers.Dropout(0.1))

    # Quarta camada densa
    model.add(layers.Dense(128, activation=activation,
                           kernel_regularizer=regularization))
    model.add(layers.Dropout(0.1))

    # Camada de saída
    model.add(layers.Dense(1))

    # Compilação do modelo com learning_rate reduzido
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# Função para criar um modelo usando LeakyReLU
def build_model_leakyrelu(regularization=None):
    model = keras.Sequential()

    # Primeira camada densa + LeakyReLU
    model.add(layers.Dense(128, input_shape=(3,),
                           kernel_regularizer=regularization))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.1))

    # Segunda camada densa + LeakyReLU
    model.add(layers.Dense(128, kernel_regularizer=regularization))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.1))

    # Terceira camada densa + LeakyReLU
    model.add(layers.Dense(128, kernel_regularizer=regularization))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.1))

    # Quarta camada densa + LeakyReLU
    model.add(layers.Dense(128, kernel_regularizer=regularization))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.1))

    # Camada de saída
    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Cria os Modelos

# Modelo utilizando ativação Swish
model_relu = build_model(activation='swish')

# Modelo utilizando ativação LeakyReLU
model_leakyrelu = build_model_leakyrelu()

# Modelo utilizando ativação Tanh
model_tanh = build_model(activation='tanh')
Características dos Modelos
Os três modelos implementados se diferenciam principalmente pela função de ativação utilizada:

Modelo ReLU (Swish): Utiliza a função de ativação Swish, uma variante moderna da ReLU que tem demonstrado bom desempenho em diversos problemas.
Modelo LeakyReLU: Implementa a função LeakyReLU com parâmetro alpha=0.01, que permite um pequeno gradiente quando a unidade não está ativa, evitando o problema de "neurônios mortos".
Modelo Tanh: Utiliza a função de ativação tangente hiperbólica, que mapeia as entradas para valores entre -1 e 1.
Todos os modelos compartilham as seguintes características:

Quatro camadas densas com 128 neurônios cada
Camadas de Dropout (taxa de 0.1) após cada camada densa para reduzir overfitting
Uma camada de saída com um único neurônio (para regressão)
Otimizador Adam com taxa de aprendizado reduzida (0.0005)
Função de perda MSE (Mean Squared Error)
Métrica de avaliação MAE (Mean Absolute Error)
Esta arquitetura foi escolhida para equilibrar capacidade de aprendizado e prevenção de overfitting, sendo adequada para o problema de aprendizado de operações matemáticas.

Treinamento dos Modelos
O processo de treinamento foi implementado com várias técnicas para garantir eficiência e qualidade, incluindo Early Stopping, salvamento automático do melhor modelo e monitoramento detalhado do progresso.

Função de Treinamento
# 3. Treinamento dos Modelos

# Função para treinar um modelo com barra de progresso e early stopping
def train_model(model, model_name, X_train, y_train, X_test, y_test, epochs=100):
    # Criar diretório para salvar os modelos
    os.makedirs('modelos_salvos', exist_ok=True)
    
    # Dividir os dados de treino em treino e validação
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Configurar callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=f'modelos_salvos/melhor_modelo_{model_name}.keras',
        monitor='val_loss',
        save_best_only=True
    )
    
    # Inicializar dicionário para armazenar o histórico
    history = {
        'loss': [],
        'val_loss': [],
        'mae': [],
        'val_mae': []
    }
    
    # Iniciar treinamento com barra de progresso
    print(f"\nTreinando modelo: {model_name}")
    with tqdm(total=epochs, desc=f"Treinando {model_name}") as pbar:
        for epoch in range(epochs):
            # Treinamento para uma época apenas
            hist = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=32,
                verbose=0,
                callbacks=[early_stopping, model_checkpoint]
            )
            
            # Atualiza o histórico
            history['loss'].append(hist.history['loss'][0])
            history['val_loss'].append(hist.history['val_loss'][0])
            history['mae'].append(hist.history['mae'][0])
            history['val_mae'].append(hist.history['val_mae'][0])
            
            # Atualizar a descrição da barra de progresso com a loss atual
            pbar.set_postfix({
                "loss": f"{hist.history['loss'][0]:.6f}",
                "val_loss": f"{hist.history['val_loss'][0]:.6f}"
            })
            pbar.update(1)
            
            # Verifica se EarlyStopping parou
            if early_stopping.stopped_epoch > 0:
                print(f"Early stopping ativado no modelo {model_name} na época {epoch+1}.")
                break
    
    # Retorna o histórico completo do treino
    return history

# Treina cada modelo separadamente
history_relu = train_model(model_relu, "ReLU", X_train, y_train, X_test, y_test, epochs=200)
history_leakyrelu = train_model(model_leakyrelu, "LeakyReLU", X_train, y_train, X_test, y_test, epochs=200)
history_tanh = train_model(model_tanh, "Tanh", X_train, y_train, X_test, y_test, epochs=200)

# Organização dos Históricos
# Dicionário agrupando os históricos dos três modelos
histories = {
    'ReLU': history_relu,
    'LeakyReLU': history_leakyrelu,
    'Tanh': history_tanh
}
O processo de treinamento inclui as seguintes etapas e técnicas:

Divisão adicional dos dados de treino em treino (80%) e validação (20%)
Implementação de Early Stopping com paciência de 10 épocas para evitar overfitting
Salvamento automático do melhor modelo baseado no erro de validação
Monitoramento do progresso com barras de progresso interativas
Registro do histórico de treinamento para análise posterior
Limite máximo de 200 épocas, com possibilidade de parada antecipada
Durante o treinamento, foram registrados os valores de loss (MSE) e MAE tanto para o conjunto de treino quanto para o de validação, permitindo monitorar o desempenho e detectar possíveis problemas como overfitting ou underfitting.

Gráfico de comparação da Loss entre os modelos
Figura 1: Comparação da evolução da Loss (MSE) durante o treinamento dos três modelos.

Gráfico de comparação do MAE entre os modelos
Figura 2: Comparação da evolução do MAE durante o treinamento dos três modelos.

Análise dos Resultados
Após o treinamento, os resultados foram visualizados e analisados através de gráficos comparativos e métricas quantitativas. A seleção do melhor modelo foi feita de forma automatizada, baseada no menor erro de validação.

Visualização dos Resultados
# 4. Visualização e Análise dos Resultados

# Função para Plotar Gráficos de Comparação
def plot_histories(histories, metric='loss'):
    plt.figure(figsize=(12, 6))

    for name, history in histories.items():
        plt.plot(history[metric], label=f'{name} {metric} treino')
        plt.plot(history['val_' + metric], label=f'{name} {metric} validação')

    plt.title(f'Comparação de {metric.upper()} entre os Modelos')
    plt.xlabel('Épocas')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.savefig(f'comparacao_{metric}.png')
    plt.close()

# Plotar comparação da Loss
plot_histories(histories, metric='loss')

# Plotar comparação da MAE
plot_histories(histories, metric='mae')
Seleção do Melhor Modelo
# 5. Análise e Seleção do Melhor Modelo

# Função para analisar o desempenho e carregar automaticamente o melhor modelo
def analyze_and_load_best_model(histories):
    val_losses = {}

    # Coletar o último valor de val_loss de cada modelo
    for name, history in histories.items():
        val_loss_final = history['val_loss'][-1]
        val_losses[name] = val_loss_final

    # Ordenar os modelos pelo menor val_loss
    sorted_models = sorted(val_losses.items(), key=lambda x: x[1])

    print("\nRanking dos Modelos baseado no menor VAL_LOSS:")
    for rank, (model_name, loss) in enumerate(sorted_models, start=1):
        print(f"{rank}º lugar: {model_name} (Val Loss Final: {loss:.5f})")

    # Identificar o melhor modelo
    best_model_name = sorted_models[0][0]
    best_model_path = f'modelos_salvos/melhor_modelo_{best_model_name}.keras'

    print(f"\nCarregando o melhor modelo salvo: {best_model_name}")

    # Carregar o modelo sem compilar
    best_model_loaded = load_model(best_model_path, compile=False)

    # Compilar manualmente depois de carregar
    best_model_loaded.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return best_model_name, best_model_loaded

# Rodar a função para analisar e carregar o melhor modelo
nome_melhor_modelo, melhor_modelo = analyze_and_load_best_model(histories)
Resultados da Comparação
A análise dos resultados mostrou que o modelo com função de ativação ReLU (Swish) obteve o melhor desempenho geral, apresentando o menor erro de validação entre os três modelos testados. O ranking final dos modelos foi:

Posição	Modelo	Erro de Validação (Val Loss)
1º lugar	ReLU (Swish)	14.54543
2º lugar	LeakyReLU	19.08826
3º lugar	Tanh	32.21506
Os gráficos de evolução da Loss e MAE durante o treinamento revelaram padrões interessantes:

O modelo ReLU convergiu mais rapidamente e de forma mais estável
O modelo LeakyReLU apresentou comportamento similar ao ReLU, mas com convergência ligeiramente mais lenta
O modelo Tanh mostrou maior dificuldade de convergência e estabilização
Nos testes com novos exemplos aleatórios, observou-se que o modelo ReLU demonstrou melhor capacidade de generalização, especialmente em operações de multiplicação, enquanto o modelo Tanh apresentou dificuldades significativas, especialmente em operações de divisão.

Conclusão
Este projeto demonstrou a importância da escolha adequada da função de ativação em redes neurais, com a ReLU (Swish) apresentando vantagens significativas para o problema de aprendizado de operações matemáticas. Além disso, confirmou a eficácia de técnicas modernas de regularização e otimização na construção de modelos robustos e precisos.

Impacto das Técnicas de Regularização
A implementação de técnicas de regularização como Dropout (0.1) e a utilização do Early Stopping foram fundamentais para:

Prevenir o overfitting, como evidenciado pela proximidade entre as curvas de treino e validação
Reduzir o tempo total de treinamento, com a maioria dos modelos convergindo antes de atingir o limite máximo de épocas
Melhorar a capacidade de generalização dos modelos
Considerações sobre a Arquitetura
A arquitetura de quatro camadas densas com 128 neurônios cada mostrou-se adequada para o problema proposto, oferecendo:

Capacidade suficiente para aprender os padrões das operações matemáticas
Equilíbrio entre complexidade do modelo e risco de overfitting
Flexibilidade para trabalhar com diferentes funções de ativação
Conclusão Final
A abordagem sistemática de comparação entre diferentes arquiteturas, com métricas objetivas e análise visual dos resultados, mostrou-se fundamental para a seleção do modelo mais adequado. Esta metodologia pode ser estendida para outros problemas de aprendizado de máquina, garantindo decisões baseadas em evidências quantitativas.

Por fim, o sucesso do modelo ReLU na aprendizagem de operações matemáticas básicas sugere que esta arquitetura pode ser promissora para aplicações mais complexas envolvendo raciocínio matemático, abrindo caminho para futuras pesquisas e desenvolvimentos nesta área.

Testar o Modelo
Nesta seção, você pode testar o modelo de rede neural treinado diretamente no navegador. Insira dois números e selecione a operação matemática desejada para ver a previsão do modelo.

Instruções para Executar o Site Localmente
Para garantir o funcionamento correto do modelo TensorFlow.js, recomendamos executar este site através de um servidor local simples:

Abra um terminal ou prompt de comando na pasta onde você descompactou os arquivos
Execute um dos seguintes comandos, dependendo da versão do Python instalada:
Python 3: python -m http.server
Python 2: python -m SimpleHTTPServer
Abra seu navegador e acesse: http://localhost:8000
Isso evita problemas de CORS (Cross-Origin Resource Sharing) que podem impedir o carregamento do modelo quando os arquivos são abertos diretamente.

Calculadora com Rede Neural
Insira dois números e selecione uma operação para ver a previsão do modelo:

Primeiro Número:
5,0
Operação:

Adição (+)
Segundo Número:
3,0
Calcular Previsão
Resultado da Previsão
5 + 3 = ?
7.37
Valor Real
8.00

Previsão do Modelo
7.37

Erro Absoluto
0.63

Análise de Redes Neurais para Operações Matemáticas - 2025

↑
