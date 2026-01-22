# --- visualizacao.py (Versão DARK MODE TOTAL) ---

import os
import numpy as np
import matplotlib.pyplot as plt

# Tema escuro para os gráficos combinarem com o site
plt.style.use("dark_background")

# Ajustes globais para gráficos (Tamanho Slide)
plt.rcParams.update({
    "figure.figsize": (8, 4.5),
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "lines.linewidth": 2,
    "axes.facecolor": "#1e293b", # Fundo do gráfico igual ao card
    "figure.facecolor": "#1e293b",
    "text.color": "#e2e8f0",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155"
})

# ==============================================================================
# CSS DARK MODE (O mesmo para tudo)
# ==============================================================================
CSS_CONTENT = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

:root {
    --bg-body: #0f172a;       /* Azul muito escuro */
    --bg-card: #1e293b;       /* Azul acinzentado */
    --text-main: #f8fafc;     /* Branco gelo */
    --text-muted: #94a3b8;    /* Cinza azulado */
    --accent: #38bdf8;        /* Azul neon claro */
    --success: #4ade80;       /* Verde matrix */
    --border: #334155;        /* Borda sutil */
    --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background-color: var(--bg-body);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
    padding: 40px;
    line-height: 1.6;
}

.container { max-width: 1100px; margin: 0 auto; }

/* Header estilizado */
header {
    text-align: center;
    margin-bottom: 50px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
}
header h1 { font-size: 2.2rem; color: var(--accent); margin-bottom: 5px; letter-spacing: -1px; }
.subtitle { color: var(--text-muted); font-size: 1rem; }

/* Navegação (Botão Voltar) */
.nav-bar { margin-bottom: 20px; }
.btn-back { color: var(--text-muted); text-decoration: none; font-size: 0.9rem; display: inline-flex; align-items: center; }
.btn-back:hover { color: var(--accent); }

/* KPIs */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
.kpi-card { background: var(--bg-card); padding: 25px; border-radius: 16px; border: 1px solid var(--border); text-align: center; box-shadow: var(--shadow); }
.kpi-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: var(--success); margin-top: 10px; }

/* Seções e Cards */
.section-title { 
    font-size: 1.4rem; 
    margin-top: 60px; 
    margin-bottom: 20px; 
    border-left: 4px solid var(--accent); 
    padding-left: 15px; 
    color: var(--text-main); 
}
.card { 
    background: var(--bg-card); 
    border-radius: 16px; 
    border: 1px solid var(--border); 
    padding: 30px; 
    margin-bottom: 30px; 
    text-align: center; 
    box-shadow: var(--shadow); 
}

/* Imagens controladas */
img { 
    max-width: 100%; 
    height: auto; 
    border-radius: 8px; 
    border: 1px solid var(--border); 
    margin-top: 10px; 
    max-height: 450px; /* Evita ficar gigante */
}

/* Tabelas bonitas */
table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 10px; }
th { text-align: left; padding: 15px; color: var(--text-muted); border-bottom: 1px solid var(--border); background: #182333; }
td { padding: 15px; border-bottom: 1px solid var(--border); font-size: 0.95rem; }
tr:last-child td { border-bottom: none; }
tr:hover { background-color: #253142; transition: background 0.2s; }

/* Botões */
.btn { 
    display: inline-block; 
    padding: 8px 16px; 
    background-color: var(--accent); 
    color: #0f172a; 
    text-decoration: none; 
    border-radius: 6px; 
    font-weight: 600; 
    font-size: 0.85rem; 
    transition: all 0.2s; 
}
.btn:hover { background-color: #7dd3fc; transform: translateY(-2px); }

/* Explicação Didática */
.didatico {
    background-color: rgba(56, 189, 248, 0.1); /* Accent com transparência */
    border-left: 4px solid var(--accent);
    padding: 20px;
    margin-bottom: 30px;
    text-align: left;
    border-radius: 0 8px 8px 0;
    font-size: 0.95rem;
    color: #e2e8f0;
}
.didatico strong { color: var(--accent); }
"""

# ---------------------------------------------------------------------
# GRÁFICOS (Ajustados para Dark Mode)
# ---------------------------------------------------------------------
def plotar_convergencia(historico_melhores, historico_medias=None, nome_arquivo="convergencia.png"):
    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)
    plt.figure()
    
    if historico_melhores:
        max_len = max(len(h) for h in historico_melhores)
        matriz_melhor = []
        for h in historico_melhores:
            h_full = list(h) + [h[-1]] * (max_len - len(h))
            matriz_melhor.append(h_full)
        
        arr_melhor = np.array(matriz_melhor)
        media_melhor = np.mean(arr_melhor, axis=0)
        x = np.arange(len(media_melhor))
        plt.plot(x, media_melhor, label="Melhor Solução (Média)", color="#38bdf8", linewidth=2.5)

    if historico_medias:
        max_len = max(len(h) for h in historico_medias)
        matriz_media = []
        for h in historico_medias:
            h_full = list(h) + [h[-1]] * (max_len - len(h))
            matriz_media.append(h_full)
        arr_media = np.array(matriz_media)
        media_pop = np.mean(arr_media, axis=0)
        if 'x' not in locals() or len(media_pop) != len(x): x = np.arange(len(media_pop))
        plt.plot(x, media_pop, label="Média da População", color="#fbbf24", linestyle="--", linewidth=1.5, alpha=0.8)

    plt.title("Curva de Aprendizado")
    plt.xlabel("Gerações")
    plt.ylabel("Distância")
    plt.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    plt.grid(True, linestyle='--', alpha=0.1)
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()

def plotar_boxplot_comparativo(dados_experimentos, nome_arquivo="boxplot.png"):
    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)
    
    # 1. Ordenar os dados do Vencedor para o Perdedor
    # Isso ajuda MUITO na leitura: o melhor fica sempre na esquerda
    dados_ordenados = {k: v for k, v in sorted(dados_experimentos.items(), key=lambda item: np.mean(item[1]))}
    lista_dados = list(dados_ordenados.values())
    rotulos = list(dados_ordenados.keys())
    
    # Identifica o índice do vencedor (menor média)
    medias = [np.mean(d) for d in lista_dados]
    idx_vencedor = np.argmin(medias)

    fig, ax = plt.subplots() # Usando subplot para ter mais controle
    
    # 2. Desenhar o Boxplot
    boxplot = ax.boxplot(lista_dados, patch_artist=True, labels=rotulos,
                         widths=0.6, # Caixas um pouco mais largas
                         flierprops=dict(marker='o', markerfacecolor='#f87171', markersize=3, alpha=0.5))

    # 3. Colorir Inteligente: Vencedor Dourado, Outros Azuis
    colors = []
    for i in range(len(lista_dados)):
        if i == idx_vencedor:
            colors.append('#fbbf24') # Amarelo Dourado (Vencedor)
        else:
            colors.append('#38bdf8') # Azul Padrão
            
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color)

    # 4. A GRANDE MELHORIA: Escrever os números no gráfico (Anotações)
    # Isso resolve o problema de "não dá para ver quem ganhou"
    ymax = max([max(sublist) for sublist in lista_dados])
    ymin = min([min(sublist) for sublist in lista_dados])
    amplitude = ymax - ymin if (ymax - ymin) > 0 else 1.0 # Evita erro se amplitude for 0
    
    for i, dados in enumerate(lista_dados):
        media = np.mean(dados)
        melhor = np.min(dados)
        
        # Posição X da caixa
        pos_x = i + 1 
        
        # Escreve a Média (Ponto Vermelho)
        ax.plot(pos_x, media, "o", color="#ef4444", markersize=6, zorder=10)
        
        # Escreve os valores numéricos em cima da caixa
        # Se for o vencedor, coloca em negrito
        peso_fonte = 'bold' if i == idx_vencedor else 'normal'
        cor_texto = '#fbbf24' if i == idx_vencedor else '#e2e8f0'
        
        # Texto: Média
        ax.text(pos_x, max(dados) + (amplitude * 0.02), 
                f"Med: {media:.1f}", 
                ha='center', va='bottom', fontsize=11, color=cor_texto, fontweight=peso_fonte)
        
        # Texto: Melhor (opcional, fica embaixo)
        ax.text(pos_x, min(dados) - (amplitude * 0.08), 
                f"Min: {melhor:.2f}", 
                ha='center', va='top', 
                fontsize=9,       # <--- AUMENTADO (Era 7)
                color='#94a3b8')

    # Cosmética Final
    ax.set_title(f"Comparativo de Estabilidade (Vencedor: {rotulos[idx_vencedor]})")
    ax.set_ylabel("Distância Final")
    ax.grid(True, linestyle='--', alpha=0.15)
    
    # Ajusta o limite Y para caber os textos
    ax.set_ylim(ymin - (amplitude*0.35), ymax + (amplitude*0.3))
    
    plt.xticks(rotation=10, color="#cbd5e1")
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

    plt.figure()
    plt.plot(xs, ys, "-o", color="#38bdf8", markersize=4, linewidth=1.5, label="Rota")
    plt.plot(xs[0], ys[0], "o", color="#4ade80", markersize=8, label="Início") # Verde para início
    
    plt.title(f"Melhor Rota: {total:.2f}")
    plt.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    plt.axis('off') 
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()

def plotar_comparativo_tempo(tempos_dict, nome_arquivo="tempos.png"):
    os.makedirs(os.path.dirname(nome_arquivo) or ".", exist_ok=True)

    algoritmos = list(tempos_dict.keys())
    tempos = list(tempos_dict.values())

    plt.figure(figsize=(8, 4))
    barras = plt.bar(algoritmos, tempos, color="#4ade80", alpha=0.7, edgecolor="#4ade80")

    plt.title("Tempo Médio (s)")
    plt.ylabel("Segundos")
    plt.grid(True, axis='y', linestyle='--', alpha=0.1)
    plt.xticks(rotation=15, color="#cbd5e1")
    
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura,
                 f'{altura:.2f}s',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#f8fafc')

    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()

# ---------------------------------------------------------------------
# GERADORES DE HTML (Agora são 2 funções: Index e Detalhado)
# ---------------------------------------------------------------------

# 1. Gera o Dashboard Principal (Index da Fase)
def gerar_relatorio_final(caminho_arquivo, sumario_global, kpis, titulo_fase):
    pasta = os.path.dirname(caminho_arquivo)
    if pasta: os.makedirs(pasta, exist_ok=True)
    
    # CSS Embutido (Garante que funcione sempre)
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head><meta charset="UTF-8"><title>{titulo_fase}</title>
    <style>{CSS_CONTENT}</style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>📊 {titulo_fase}</h1>
                <div class="subtitle">Visão Geral da Bateria de Testes</div>
            </header>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Campeão da Fase</div>
                    <div class="kpi-value" style="color:var(--accent)">{kpis['melhor_alg']}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Recorde Absoluto</div>
                    <div class="kpi-value">{kpis['melhor_dist']:.2f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total de Execuções</div>
                    <div class="kpi-value" style="color:#fff">{kpis['total_exec']}</div>
                </div>
            </div>

            <h2 class="section-title">1. Comparativo Geral (Estabilidade)</h2>
            <div class="card">
                <img src="boxplot_global.png" alt="Boxplot Global">
                <div style="margin-top:15px; color:#94a3b8; font-size:0.9rem;">
                    * Algoritmos mais baixos e com caixas menores são melhores.
                </div>
            </div>

            <h2 class="section-title">2. Resultados por Instância</h2>
            <div class="card">
                <table>
                    <thead><tr><th>Instância</th><th>Melhor Custo</th><th>Vencedor</th><th>Detalhes</th></tr></thead>
                    <tbody>
    """
    for inst, info in sumario_global.items():
        html += f"""
        <tr>
            <td><strong>{inst.upper()}</strong></td>
            <td>{info['melhor_dist']:.2f}</td>
            <td style="color:var(--accent); font-weight:bold;">{info['melhor_alg']}</td>
            <td><a href="{inst}/relatorio_{inst}.html" class="btn">Abrir Relatório ➔</a></td>
        </tr>
        """
    html += "</tbody></table></div></div></body></html>"
    
    with open(caminho_arquivo, "w", encoding="utf-8") as f: f.write(html)

# 2. Gera o Relatório Detalhado (Instância) - NOVO!
def gerar_relatorio_instancia(caminho_arquivo, nome_instancia, titulo_fase, vencedor, dist_vencedor, ttest_html, imagens):
    pasta = os.path.dirname(caminho_arquivo)
    if pasta: os.makedirs(pasta, exist_ok=True)

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head><meta charset="UTF-8"><title>{nome_instancia.upper()}</title>
    <style>{CSS_CONTENT}</style>
    </head>
    <body>
        <div class="container">
            <div class="nav-bar"><a href="../index.html" class="btn-back">⬅ Voltar para Visão Geral</a></div>
            
            <header>
                <h1>{nome_instancia.upper()}</h1>
                <div class="subtitle">{titulo_fase}</div>
            </header>

            <div class="didatico">
                <strong>Resultado da Instância:</strong><br>
                O algoritmo vencedor foi <span style="color:var(--accent); font-weight:bold">{vencedor}</span> 
                com uma distância de <strong>{dist_vencedor:.2f}</strong>.
            </div>

            <h2 class="section-title">1. Melhor Rota Encontrada</h2>
            <div class="card"><img src="{imagens['rota']}"></div>

            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h2 class="section-title" style="margin-top:0">2. Tempos (Velocidade)</h2>
                    <div class="card"><img src="{imagens['tempos']}"></div>
                </div>
                <div>
                    <h2 class="section-title" style="margin-top:0">3. Estabilidade (Boxplot)</h2>
                    <div class="card"><img src="{imagens['boxplot']}"></div>
                </div>
            </div>

            <h2 class="section-title">4. Testes Estatísticos (T-Student)</h2>
            <div class="card" style="text-align:left; font-size:0.9rem;">{ttest_html}</div>

            <h2 class="section-title">5. Curvas de Convergência</h2>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
    """
    
    for img_conv in imagens['convergencias']:
        html += f"<div class='card'><h3 style='color:#94a3b8; font-size:1rem; margin-bottom:10px;'>{img_conv['nome']}</h3><img src='{img_conv['arquivo']}'></div>"

    html += "</div></div></body></html>"
    
    with open(caminho_arquivo, "w", encoding="utf-8") as f: f.write(html)