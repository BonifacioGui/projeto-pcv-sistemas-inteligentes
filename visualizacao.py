# --- visualizacao.py (Versão PRO Otimizada + Visual Moderno) ---

import os
import numpy as np
import matplotlib.pyplot as plt

# Tema visual moderno
plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------
# 🔹 Função auxiliar: downsampling para históricos muito longos
# ---------------------------------------------------------------------
def _downsample(data, max_points=800):
    """
    Reduz o número de pontos de um array longo sem distorcer a forma.
    """
    if len(data) <= max_points:
        return data
    
    idx = np.linspace(0, len(data) - 1, max_points).astype(int)
    return data[idx]


# ---------------------------------------------------------------------
# 🔹 Convergência
# ---------------------------------------------------------------------
def plotar_convergencia(historico_melhores, historico_medias=None, nome_arquivo="convergencia.png"):
    """
    Plota a evolução da melhor solução E da média da população.
    """
    print(f"\nGerando gráfico de convergência... {nome_arquivo}")
    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)
    
    plt.figure(figsize=(12, 7), dpi=140)
    
    # Plota o MELHOR (que você já tinha)
    # Fazemos uma média das 30 execuções para suavizar a linha
    # Nota: aqui assumimos que historico_melhores é uma lista de listas (uma por execução)
    if historico_melhores:
        # Normalização de tamanho (caso alguma execução tenha parado antes)
        max_len = max(len(h) for h in historico_melhores)
        matriz_melhor = []
        for h in historico_melhores:
            # estica o array para ficar do mesmo tamanho
            h_full = list(h) + [h[-1]] * (max_len - len(h))
            matriz_melhor.append(h_full)
        
        arr_melhor = np.array(matriz_melhor)
        media_melhor = np.mean(arr_melhor, axis=0)
        x = np.arange(len(media_melhor))
        plt.plot(x, media_melhor, label="Melhor Solução (Média das 30 execuções)", color="blue", linewidth=2)

    # Plota a MÉDIA DA POPULAÇÃO (Novo Requisito)
    if historico_medias:
        max_len = max(len(h) for h in historico_medias)
        matriz_media = []
        for h in historico_medias:
            h_full = list(h) + [h[-1]] * (max_len - len(h))
            matriz_media.append(h_full)
            
        arr_media = np.array(matriz_media)
        media_pop = np.mean(arr_media, axis=0)
        
        # Usa o mesmo X
        if len(media_pop) != len(x):
             x = np.arange(len(media_pop))
             
        plt.plot(x, media_pop, label="Média da População", color="orange", linestyle="--", alpha=0.7)

    plt.title("Convergência: Melhor vs Média da População", fontsize=14)
    plt.xlabel("Gerações")
    plt.ylabel("Distância")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()


# ---------------------------------------------------------------------
# 🔹 Boxplot Comparativo
# ---------------------------------------------------------------------
def plotar_boxplot_comparativo(dados_experimentos, nome_arquivo="boxplot.png"):
    """
    Boxplot ordenado pelos melhores resultados médios.
    """
    print(f"\nGerando boxplot comparativo... salvando em {nome_arquivo}")

    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)

    # Ordena experimentos pela média (melhor → pior)
    dados_ordenados = {
        k: v for k, v in sorted(
            dados_experimentos.items(),
            key=lambda item: np.mean(item[1])
        )
    }

    lista_de_dados = list(dados_ordenados.values())
    rotulos = list(dados_ordenados.keys())

    plt.figure(figsize=(12, 7), dpi=140)

    boxplot = plt.boxplot(lista_de_dados, patch_artist=True, labels=rotulos)

    # Colormap moderno
    cmap = plt.cm.get_cmap("Set3")
    for patch, color in zip(boxplot["boxes"], cmap.colors):
        patch.set_facecolor(color)

    # Médias como pontos vermelhos
    for i, data in enumerate(lista_de_dados):
        media = np.mean(data)
        plt.plot(i + 1, media, "ro", markersize=7)

    plt.xticks(rotation=25, ha="right")
    plt.title("Comparação de Desempenho — 30 Execuções", fontsize=16)
    plt.ylabel("Melhor Distância Final", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=180)
    plt.close()

    print("Boxplot salvo com sucesso.")


# ---------------------------------------------------------------------
# 🔹 Visualização de Rota
# ---------------------------------------------------------------------
def plotar_rota(melhor_rota, cidades, nome_arquivo="rota_final.png"):
    """
    Visualização moderna e mais legível do TSP, com destaque para
    cidade inicial e título exibindo a distância total.
    """

    print(f"\nGerando gráfico da melhor rota... salvando em {nome_arquivo}")

    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)

    # Coordenadas ordenadas pela rota
    rota_ordenada = [cidades[i] for i in melhor_rota]
    rota_ordenada.append(rota_ordenada[0])  # fecha o ciclo
    xs, ys = zip(*rota_ordenada)

    total = sum(
        np.hypot(xs[i+1] - xs[i], ys[i+1] - ys[i])
        for i in range(len(xs) - 1)
    )

    plt.figure(figsize=(12, 7), dpi=140)

    # Linhas da rota
    plt.plot(xs, ys, "-o", linewidth=1.5, markersize=4,
             color="#007acc", label="Rota")

    # Destaque da cidade inicial
    plt.plot(xs[0], ys[0], "ro", markersize=9, label="Cidade Inicial")

    plt.title(f"Melhor Rota Encontrada — Distância Total: {total:.2f}", fontsize=16)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=180)
    plt.close()

    print("Gráfico da rota salvo com sucesso.")
