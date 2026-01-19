# --- visualizacao.py (Versão Final: CSS Integrado + Legendas Didáticas) ---

import os
import numpy as np
import matplotlib.pyplot as plt

# Tema visual moderno e limpo para os gráficos
plt.style.use("seaborn-v0_8-whitegrid")

# ==============================================================================
# CSS PROFISSIONAL (Integrado aqui para não sujar a main.py)
# ==============================================================================
CSS_CONTENT = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

:root {
    --bg-body: #0f172a;
    --bg-card: #1e293b;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --success: #4ade80;
    --border: #334155;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background-color: var(--bg-body);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
    padding: 40px;
    line-height: 1.6;
}

.container { max-width: 1200px; margin: 0 auto; }

/* Header */
header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 40px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
}
header h1 { font-size: 1.8rem; font-weight: 600; color: var(--accent); }
.timestamp { color: var(--text-muted); font-size: 0.9rem; }

/* KPIs */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
}
.kpi-card {
    background: var(--bg-card);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    text-align: center;
}
.kpi-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { font-size: 1.5rem; font-weight: 700; color: var(--success); margin-top: 5px; }

/* Seções e Cards */
.section-title { font-size: 1.4rem; margin-bottom: 20px; border-left: 4px solid var(--accent); padding-left: 15px; margin-top: 40px; }

.card {
    background: var(--bg-card);
    border-radius: 16px;
    border: 1px solid var(--border);
    padding: 25px;
    margin-bottom: 30px;
    box-shadow: var(--shadow);
}

img { width: 80%; height: auto; border-radius: 8px; border: 1px solid var(--border); margin-top: 15px; display: block; margin-left: auto; margin-right: auto; }

/* Tabela */
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th { text-align: left; padding: 15px; color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); }
td { padding: 15px; border-bottom: 1px solid var(--border); }
tr:last-child td { border-bottom: none; }
tr:hover { background-color: #263345; }

/* Botões */
.btn {
    display: inline-block;
    padding: 8px 16px;
    background-color: var(--accent);
    color: #0f172a;
    text-decoration: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: transform 0.2s;
}
.btn:hover { transform: translateY(-2px); filter: brightness(1.1); }

/* Legenda Didática */
.legenda-didatica {
    background-color: #263345;
    padding: 15px;
    border-radius: 8px;
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 15px;
    border-left: 3px solid var(--accent);
}
"""

# ---------------------------------------------------------------------
# 🔹 Função auxiliar
# ---------------------------------------------------------------------
def _downsample(data, max_points=800):
    if len(data) <= max_points: return data
    idx = np.linspace(0, len(data) - 1, max_points).astype(int)
    return data[idx]

# ---------------------------------------------------------------------
# 🔹 Gráficos
# ---------------------------------------------------------------------
def plotar_convergencia(historico_melhores, historico_medias=None, nome_arquivo="convergencia.png"):
    print(f"Gerando gráfico: {nome_arquivo}")
    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)
    
    plt.figure(figsize=(10, 6), dpi=120)
    
    if historico_melhores:
        max_len = max(len(h) for h in historico_melhores)
        matriz_melhor = []
        for h in historico_melhores:
            h_full = list(h) + [h[-1]] * (max_len - len(h))
            matriz_melhor.append(h_full)
        
        arr_melhor = np.array(matriz_melhor)
        media_melhor = np.mean(arr_melhor, axis=0)
        x = np.arange(len(media_melhor))
        plt.plot(x, media_melhor, label="Melhor Solução (Média)", color="#38bdf8", linewidth=2)

    if historico_medias:
        max_len = max(len(h) for h in historico_medias)
        matriz_media = []
        for h in historico_medias:
            h_full = list(h) + [h[-1]] * (max_len - len(h))
            matriz_media.append(h_full)
        arr_media = np.array(matriz_media)
        media_pop = np.mean(arr_media, axis=0)
        if 'x' not in locals() or len(media_pop) != len(x): x = np.arange(len(media_pop))
        plt.plot(x, media_pop, label="Média da População", color="#fbbf24", linestyle="--", alpha=0.8)

    plt.title("Convergência: Evolução ao longo das Gerações")
    plt.xlabel("Gerações")
    plt.ylabel("Distância (Custo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()

def plotar_boxplot_comparativo(dados_experimentos, nome_arquivo="boxplot.png"):
    print(f"Gerando boxplot: {nome_arquivo}")
    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)

    dados_ordenados = {k: v for k, v in sorted(dados_experimentos.items(), key=lambda item: np.mean(item[1]))}
    lista_dados = list(dados_ordenados.values())
    rotulos = list(dados_ordenados.keys())

    plt.figure(figsize=(10, 6), dpi=120)
    boxplot = plt.boxplot(lista_dados, patch_artist=True, labels=rotulos)

    cmap = plt.cm.get_cmap("Set3")
    for patch, color in zip(boxplot["boxes"], cmap.colors):
        patch.set_facecolor(color)
    
    for i, data in enumerate(lista_dados):
        plt.plot(i + 1, np.mean(data), "ro", markersize=6, label="Média" if i==0 else "")

    plt.title("Comparativo de Estabilidade (30 Execuções)")
    plt.ylabel("Distância Final")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()

def plotar_rota(melhor_rota, cidades, nome_arquivo="rota_final.png"):
    if not melhor_rota: return
    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)

    rota_ordenada = [cidades[i] for i in melhor_rota]
    rota_ordenada.append(rota_ordenada[0])
    xs, ys = zip(*rota_ordenada)
    total = sum(np.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i]) for i in range(len(xs)-1))

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(xs, ys, "-o", color="#38bdf8", markersize=4, linewidth=1, label="Rota")
    plt.plot(xs[0], ys[0], "ro", markersize=8, label="Início")
    plt.title(f"Melhor Rota: {total:.2f}")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()

# ---------------------------------------------------------------------
# 🔹 RELATÓRIO FINAL (Onde a mágica acontece)
# ---------------------------------------------------------------------
def gerar_relatorio_final(caminho_arquivo, sumario_global, kpis, explicacao_texto=""):
    """
    Gera o HTML com CSS embutido e explicações didáticas nos gráficos.
    """
    
    # 1. Gera o CSS
    pasta = os.path.dirname(caminho_arquivo)
    if pasta: os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, "style.css"), "w", encoding="utf-8") as f:
        f.write(CSS_CONTENT)

    # 2. Gera o HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>TSP Dashboard</title>
        <link rel="stylesheet" href="style.css">
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Dashboard de Resultados</h1>
                <div class="timestamp">Relatório Automático</div>
            </header>

            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Melhor Algoritmo</div><div class="kpi-value">{kpis['melhor_alg']}</div></div>
                <div class="kpi-card"><div class="kpi-label">Melhor Distância</div><div class="kpi-value">{kpis['melhor_dist']:.2f}</div></div>
                <div class="kpi-card"><div class="kpi-label">Total Execuções</div><div class="kpi-value">{kpis['total_exec']}</div></div>
            </div>

            <h2 class="section-title">1. Análise e Conclusões</h2>
            <div class="card" style="border-left: 4px solid var(--success);">
                <h3>Interpretação dos Resultados</h3>
                <p style="font-size: 1.1rem; color: var(--text-main); margin-top: 10px;">
                    {explicacao_texto if explicacao_texto else "Nenhuma explicação fornecida."}
                </p>
            </div>

            <h2 class="section-title">2. Comparativo Visual</h2>
            <div class="card">
                <img src="boxplot_global.png">
                
                <div class="legenda-didatica">
                    <strong>Como ler este gráfico (Boxplot):</strong><br>
                    Este gráfico resume a estabilidade de cada algoritmo em 30 execuções.<br>
                    <ul>
                        <li><strong>A Caixa Colorida:</strong> Representa onde caíram 50% dos resultados mais comuns. Caixas "achatadas" indicam alta estabilidade.</li>
                        <li><strong>O Ponto Vermelho:</strong> É a média de todas as execuções.</li>
                        <li><strong>Posição Vertical:</strong> Quanto mais baixo no gráfico, melhor foi a solução (menor distância percorrida).</li>
                    </ul>
                </div>
            </div>

            <h2 class="section-title">3. Detalhamento por Instância</h2>
            <div class="card">
                <table>
                    <thead><tr><th>Instância</th><th>Melhor Distância</th><th>Vencedor</th><th>Detalhes</th></tr></thead>
                    <tbody>
    """
    
    for inst, info in sumario_global.items():
        html += f"""
        <tr>
            <td>{inst.upper()}</td>
            <td>{info['melhor_dist']:.2f}</td>
            <td style="color:var(--accent)">{info['melhor_alg']}</td>
            <td><a href="{inst}/relatorio_{inst}.html" class="btn" target="_blank">Abrir Relatório</a></td>
        </tr>
        """

    html += """</tbody></table>
    <div class="legenda-didatica" style="margin-top:20px;">
        Clique em <strong>'Abrir Relatório'</strong> para ver as curvas de convergência (velocidade de aprendizado) e a melhor rota desenhada para cada mapa.
    </div>
    </div></div></body></html>"""

    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        f.write(html)