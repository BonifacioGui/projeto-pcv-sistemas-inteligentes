# Dentro de visualizacao.py

import matplotlib.pyplot as plt
import numpy as np

def plotar_convergencia(historico_resultados, nome_arquivo="convergencia.png"):
    """
    Gera e salva um gráfico de convergência.
    
    :param historico_resultados: Uma lista de listas. Cada lista interna
                                 contém o histórico de 'melhor distância'
                                 de uma única execução (ex: 30 listas).
    :param nome_arquivo: Nome do arquivo para salvar o gráfico.
    """
    
    print(f"\nGerando gráfico de convergência... salvando em {nome_arquivo}")
    
    plt.figure(figsize=(12, 7))
    
    # 1. Calcula a média da convergência das 30 execuções
    #    'np.array(historico_resultados)' transforma [[10,8,6], [12,9,7]]
    #    em uma matriz numpy.
    #    '.mean(axis=0)' calcula a média vertical (média da Geração 0, média da Geração 1, etc.)
    media_convergencia = np.array(historico_resultados).mean(axis=0)
    
    # 2. Plota a linha de média
    plt.plot(media_convergencia, label="Média da Melhor Distância", color='blue', linewidth=2)
    
    # 3. (Opcional) Adiciona uma sombra mostrando o Desvio Padrão
    std_convergencia = np.array(historico_resultados).std(axis=0)
    plt.fill_between(
        range(len(media_convergencia)),
        media_convergencia - std_convergencia,
        media_convergencia + std_convergencia,
        color='lightblue',
        alpha=0.4,
        label="Desvio Padrão (Robustez)"
    )

    # Configurações do gráfico
    plt.title("Gráfico de Convergência (30 Execuções)")
    plt.xlabel("Geração / Iteração")
    plt.ylabel("Melhor Distância (Custo)")
    plt.legend()
    plt.grid(True)
    
    # Salva o gráfico em um arquivo
    plt.savefig(nome_arquivo)
    plt.close() # Fecha a figura para economizar memória
    
    print("Gráfico salvo com sucesso.")
    
def plotar_boxplot_comparativo(dados_experimentos, nome_arquivo="boxplot.png"):
    """
    Gera e salva um boxplot comparando os resultados de diferentes
    configurações de algoritmos.

    :param dados_experimentos: Um dicionário onde a chave é o nome
                               do experimento (ex: "AG - Torneio") e o
                               valor é a lista de resultados (as 30
                               distâncias finais).
    :param nome_arquivo: Nome do arquivo para salvar o gráfico.
    """
    
    print(f"\nGerando boxplot comparativo... salvando em {nome_arquivo}")

    # Prepara os dados e os rótulos para o boxplot
    lista_de_dados = list(dados_experimentos.values())
    rotulos = list(dados_experimentos.keys())

    plt.figure(figsize=(10, 6))
    
    # 

    # [Image of a side-by-side Boxplot diagram]

    
    # Cria o boxplot
    boxplot = plt.boxplot(lista_de_dados, patch_artist=True, labels=rotulos)

    # Adiciona cores para ficar mais fácil de ler
    colors = plt.cm.get_cmap('Pastel1', len(lista_de_dados))
    for patch, color in zip(boxplot['boxes'], colors.colors):
        patch.set_facecolor(color)

    # Configurações do gráfico
    plt.title("Comparação de Desempenho (30 Execuções)")
    plt.ylabel("Melhor Distância Final (Custo)")
    plt.xlabel("Configuração do Algoritmo")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adiciona os valores de Média (Mean) como um ponto no gráfico
    for i, data in enumerate(lista_de_dados):
        media = np.mean(data)
        plt.plot(i + 1, media, 'ro', markersize=8, label='Média' if i == 0 else "")

    plt.legend()
    
    # Salva o gráfico em um arquivo
    plt.savefig(nome_arquivo)
    plt.close() # Fecha a figura para economizar memória
    
    print("Boxplot salvo com sucesso.")
    
def plotar_rota(melhor_rota, cidades, nome_arquivo="rota_final.png"):
    """
    Gera e salva um gráfico de dispersão (scatter plot) mostrando as
    cidades e a melhor rota encontrada.

    :param melhor_rota: Uma lista de índices (a permutação) da melhor rota.
    :param cidades: A lista de tuplas de coordenadas (x, y).
    :param nome_arquivo: Nome do arquivo para salvar o gráfico.
    """
    
    print(f"\nGerando gráfico da melhor rota... salvando em {nome_arquivo}")

    plt.figure(figsize=(12, 8))

    # [Image of a TSP route visualization for 70 cities]
    
    # 1. Cria uma lista de coordenadas na ordem da melhor rota
    rota_ordenada = [cidades[i] for i in melhor_rota]
    
    # Adiciona a cidade inicial ao final para fechar o loop
    rota_ordenada.append(rota_ordenada[0])
    
    # 2. Pega as coordenadas X e Y separadamente
    #    zip(*rota_ordenada) é um truque do Python para "descompactar"
    #    a lista de tuplas [(x1,y1), (x2,y2)] em duas listas: [x1,x2] e [y1,y2]
    coordenadas_x, coordenadas_y = zip(*rota_ordenada)

    # 3. Plota as linhas da rota
    plt.plot(coordenadas_x, coordenadas_y, marker='o', linestyle='-', markersize=5, label="Rota")
    
    # 4. Plota a cidade inicial (ponto vermelho)
    plt.plot(coordenadas_x[0], coordenadas_y[0], 'ro', markersize=10, label="Cidade Inicial/Final")

    # Configurações do gráfico
    plt.title("Visualização da Melhor Rota Encontrada")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.legend()
    plt.grid(True)
    
    # Salva o gráfico em um arquivo
    plt.savefig(nome_arquivo)
    plt.close() # Fecha a figura para economizar memória
    
    print("Gráfico da rota salvo com sucesso.")