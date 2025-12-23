"""
main.py - Versão Final para Entrega (Com Dashboard Profissional)
Contém todos os cenários de teste exigidos no projeto e gera relatório executivo.
"""

import os
import time
import json
import csv
import numpy as np
from datetime import datetime
from scipy import stats

# Módulos internos
import parser_tsplib
import utils
import visualizacao
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO
from recozimento_simulado import RecozimentoSimulado


# ================================================================
# 1. CONFIGURAÇÕES GERAIS
# ================================================================

NUM_EXECUCOES     = 2     # Robustez estatística (Padrão: 30)
NUM_GERACOES      = 10    # Iterações por execução (500)
TAMANHO_POPULACAO = 10    # Tamanho da população/formigueiro (50)

INSTANCIAS = [
    "data/st70.tsp",
    "data/eil101.tsp",
    "data/ch130.tsp"
]

EXECUTION_TIMESTAMP = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


# ================================================================
# 2. BANCO DE EXPERIMENTOS (TODOS OS CENÁRIOS DA TABELA)
# ================================================================

TODOS_EXPERIMENTOS = {
    # --- ANÁLISE 9: COMPARAÇÃO GERAL (AG vs ACO vs SA) ---
    # Use esta opção para gerar o comparativo final nas 3 bases.
    "geral": [
        {"nome": "AG_Padrao", "algoritmo": "AG",
         "params": {"taxa_mutacao": 0.01, "metodo_crossover": "ox", "metodo_mutacao": "swap"}},
        
        {"nome": "AG_AltaMutacao", "algoritmo": "AG",
         "params": {"taxa_mutacao": 0.10, "metodo_crossover": "ox", "metodo_mutacao": "swap"}},
        
        {"nome": "ACO_Padrao", "algoritmo": "ACO",
         "params": {"num_formigas": TAMANHO_POPULACAO, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},
        
        {"nome": "SA_Padrao", "algoritmo": "SA",
         "params": {"temp_inicial": 1000, "cooling_rate": 0.995}}
    ],

    # --- ANÁLISE 2: SELEÇÃO (Roleta vs Torneio) ---
    "selecao": [
        {"nome": "AG_Torneio", "algoritmo": "AG",
         "params": {"metodo_selecao": "torneio", "taxa_mutacao": 0.01, "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},

        {"nome": "AG_Roleta", "algoritmo": "AG",
         "params": {"metodo_selecao": "roleta", "taxa_mutacao": 0.01, "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},
    ],
    
    # --- ANÁLISE 3: MUTAÇÃO (Troca vs Inversão) ---
    "mutacao": [
        {"nome": "AG_MutacaoTroca", "algoritmo": "AG",
         "params": {"metodo_mutacao": "swap", "metodo_selecao": "torneio", "metodo_crossover": "ox", "taxa_mutacao": 0.01}},

        {"nome": "AG_MutacaoInversao", "algoritmo": "AG",
         "params": {"metodo_mutacao": "inversion", "metodo_selecao": "torneio", "metodo_crossover": "ox", "taxa_mutacao": 0.01}}
    ],

    # --- ANÁLISE 4: ELITISMO (0% vs 5% vs 10%) ---
    "elitismo": [
        {"nome": "AG_Elitismo_0", "algoritmo": "AG",
         "params": {"taxa_elitismo": 0.00, "metodo_selecao": "torneio", "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},

        {"nome": "AG_Elitismo_5", "algoritmo": "AG",
         "params": {"taxa_elitismo": 0.05, "metodo_selecao": "torneio", "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},

        {"nome": "AG_Elitismo_10", "algoritmo": "AG",
         "params": {"taxa_elitismo": 0.10, "metodo_selecao": "torneio", "metodo_crossover": "ox", "metodo_mutacao": "inversion"}}
    ],

    # --- ANÁLISE EXTRA: CROSSOVER (OX vs PMX) ---
    "crossover": [
        {"nome": "AG_Crossover_OX", "algoritmo": "AG",
         "params": {"metodo_crossover": "ox", "metodo_selecao": "torneio", "metodo_mutacao": "inversion"}},

        {"nome": "AG_Crossover_PMX", "algoritmo": "AG",
         "params": {"metodo_crossover": "pmx", "metodo_selecao": "torneio", "metodo_mutacao": "inversion"}}
    ]
}


# ============================================================
# 3. SELETOR: QUAL EXPERIMENTO RODAR AGORA?
# ============================================================
# Opções: "geral", "selecao", "mutacao", "elitismo", "crossover"
CHAVE_ESCOLHIDA = "geral"

EXPERIMENTO_ATUAL = TODOS_EXPERIMENTOS[CHAVE_ESCOLHIDA]


# ================================================================
# 4. PREPARAÇÃO DE ESTILO (CSS PROFISSIONAL)
# ================================================================
def garantir_pasta(p):
    if not os.path.exists(p):
        os.makedirs(p)

garantir_pasta("resultados")

# CSS Moderno (Dark Mode / Scientific Style)
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
.section-title { font-size: 1.4rem; margin-bottom: 20px; border-left: 4px solid var(--accent); padding-left: 15px; }

.card {
    background: var(--bg-card);
    border-radius: 16px;
    border: 1px solid var(--border);
    padding: 25px;
    margin-bottom: 30px;
    box-shadow: var(--shadow);
}

img { width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); margin-top: 15px; }

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
"""

with open("resultados/style.css", "w", encoding="utf-8") as f:
    f.write(CSS_CONTENT)


# ================================================================
# 5. EXECUÇÃO PRINCIPAL
# ================================================================
resultados_globais = {}
sumario_global = {}

print(f"\n>>> INICIANDO BATERIA: {CHAVE_ESCOLHIDA.upper()} <<<\n")

for caminho in INSTANCIAS:
    nome_instancia = os.path.basename(caminho).split(".")[0]
    pasta_saida = f"resultados/{nome_instancia}"
    garantir_pasta(pasta_saida)

    print(f"\n=== Rodando Instância: {nome_instancia} ===")

    try:
        cidades = parser_tsplib.carregar_cidades(caminho)
        dist_matrix = utils.calcular_matriz_distancias(cidades)
    except Exception as e:
        print(f"Erro ao carregar {caminho}: {e}")
        continue

    # Dados para o relatório desta instância
    resultados_boxplot_instancia = {}
    melhor_global_dist = float("inf")
    melhor_global_rota = None

    # --- Loop pelos Algoritmos do Experimento Escolhido ---
    for exp in EXPERIMENTO_ATUAL:
        nome_exp = exp["nome"]
        alg = exp["algoritmo"]
        params = exp["params"]

        print(f" -> Executando {nome_exp} ({alg}) ... ", end="")

        resultados_dist = []
        resultados_tempo = []
        
        # Históricos para gráficos de convergência
        historico_30_melhores = []
        historico_30_medias = []

        # --- Loop de Robustez (30 execuções) ---
        for k in range(NUM_EXECUCOES):
            inicio = time.perf_counter()

            if alg == "AG":
                modelo = AlgoritmoGenetico(
                    cidades=cidades,
                    tamanho_populacao=TAMANHO_POPULACAO,
                    num_geracoes=NUM_GERACOES,
                    taxa_mutacao=params.get("taxa_mutacao", 0.01),
                    taxa_elitismo=params.get("taxa_elitismo", 0.05),
                    metodo_selecao=params.get("metodo_selecao", "torneio"),
                    metodo_crossover=params.get("metodo_crossover", "ox"),
                    metodo_mutacao=params.get("metodo_mutacao", "swap"),
                    dist_matrix=dist_matrix
                )
                melhor = modelo.executar()
                dist = melhor.distancia
                rota = melhor.rota
                
                historico_30_melhores.append(getattr(modelo, 'historico_melhores', []))
                historico_30_medias.append(getattr(modelo, 'historico_medias', []))

            elif alg == "ACO":
                modelo = ACO(
                    cidades=cidades,
                    num_formigas=params.get("num_formigas", TAMANHO_POPULACAO), 
                    num_iteracoes=NUM_GERACOES,
                    alfa=params.get("alfa", 1.0),
                    beta=params.get("beta", 2.5),
                    rho=params.get("rho", 0.1)
                )
                rota, dist = modelo.executar()
                historico_30_melhores.append(getattr(modelo, 'historico_melhores', []))

            elif alg == "SA":
                modelo = RecozimentoSimulado(
                    cidades=cidades,
                    temp_inicial=params.get("temp_inicial", 1000),
                    cooling_rate=params.get("cooling_rate", 0.995),
                    max_iterations=NUM_GERACOES * TAMANHO_POPULACAO,
                    dist_matrix=dist_matrix
                )
                rota, dist = modelo.executar()
                historico_30_melhores.append(getattr(modelo, 'historico_melhores', []))

            tempo = time.perf_counter() - inicio
            resultados_dist.append(dist)
            resultados_tempo.append(tempo)

            if dist < melhor_global_dist:
                melhor_global_dist = dist
                melhor_global_rota = rota
            
            # Feedback visual simples
            if k % 10 == 0: print(".", end="", flush=True)

        print(f" OK (Média: {np.mean(resultados_dist):.2f})")

        # Salva JSON bruto
        with open(f"{pasta_saida}/{nome_exp}.json", "w") as jf:
            json.dump({"distancias": resultados_dist, "tempos": resultados_tempo}, jf, indent=2)

        # Gráfico de Convergência Individual
        media_pop = historico_30_medias if alg == "AG" else None
        visualizacao.plotar_convergencia(
            historico_30_melhores, media_pop,
            f"{pasta_saida}/convergencia_{nome_exp}.png"
        )
        
        resultados_boxplot_instancia[nome_exp] = resultados_dist

    # --- Fim do Loop de Algoritmos para esta Instância ---

    # 1. Gerar Boxplot da Instância
    visualizacao.plotar_boxplot_comparativo(
        resultados_boxplot_instancia, 
        f"{pasta_saida}/boxplot_{nome_instancia}.png"
    )

    # 2. Gerar Melhor Rota
    visualizacao.plotar_rota(
        melhor_global_rota, cidades, 
        f"{pasta_saida}/melhor_rota_{nome_instancia}.png"
    )

    # 3. Testes Estatísticos (HTML para o relatório individual)
    ttest_html = "<div class='card'><h3>Testes Estatísticos (T-Student)</h3>"
    chaves = list(resultados_boxplot_instancia.keys())
    
    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            alg_a = chaves[i]
            alg_b = chaves[j]
            dados_a = resultados_boxplot_instancia[alg_a]
            dados_b = resultados_boxplot_instancia[alg_b]

            t_stat, p_val = stats.ttest_ind(dados_a, dados_b, equal_var=False)
            cor = "var(--success)" if p_val < 0.05 else "var(--text-muted)"
            sig = "SIM" if p_val < 0.05 else "NÃO"
            
            ttest_html += f"<p style='margin-bottom:10px; border-bottom:1px solid var(--border); padding-bottom:5px;'><b>{alg_a} vs {alg_b}</b><br>"
            ttest_html += f"p-value: {p_val:.4e} <strong style='color:{cor}'>({sig})</strong></p>"
    ttest_html += "</div>"

    # 4. Relatório HTML da Instância (Agora usando o estilo novo)
    with open(f"{pasta_saida}/relatorio_{nome_instancia}.html", "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><link rel='stylesheet' href='../style.css'></head><body><div class='container'>")
        fh.write(f"<header><h1>Relatório: {nome_instancia}</h1><div class='timestamp'>{CHAVE_ESCOLHIDA.upper()}</div></header>")
        
        fh.write("<h2 class='section-title'>Comparativo de Desempenho</h2>")
        fh.write(f"<div class='card'><img src='boxplot_{nome_instancia}.png'></div>")
        
        fh.write("<h2 class='section-title'>Melhor Solução Encontrada</h2>")
        fh.write(f"<div class='card'><p>Distância Total: <b style='color:var(--accent)'>{melhor_global_dist:.2f}</b></p>")
        fh.write(f"<img src='melhor_rota_{nome_instancia}.png'></div>")
        
        fh.write(ttest_html)
        
        fh.write("<h2 class='section-title'>Curvas de Convergência</h2><div class='card'>")
        for exp in EXPERIMENTO_ATUAL:
            nome_img = exp["nome"]
            fh.write(f"<h3>{nome_img}</h3><img src='convergencia_{nome_img}.png'>")
        fh.write("</div></div></body></html>")

    # Salva melhor para o sumário global
    melhor_alg_instancia = min(resultados_boxplot_instancia, key=lambda k: np.mean(resultados_boxplot_instancia[k]))
    sumario_global[nome_instancia] = {
        "melhor_dist": melhor_global_dist,
        "melhor_alg": melhor_alg_instancia
    }
    
    # Guarda resultados para o boxplot global
    for k, v in resultados_boxplot_instancia.items():
        resultados_globais[f"{nome_instancia}_{k}"] = v


# ================================================================
# 6. DASHBOARD GLOBAL (VISUAL PROFISSIONAL)
# ================================================================

# 1. Gera o gráfico global
visualizacao.plotar_boxplot_comparativo(
    resultados_globais, "resultados/boxplot_global.png"
)

# 2. Calcula KPIs
total_instancias = len(INSTANCIAS)
if sumario_global:
    melhor_alg_geral = max(
        [info['melhor_alg'] for info in sumario_global.values()],
        key=[info['melhor_alg'] for info in sumario_global.values()].count
    )
    melhor_dist_abs = min(info['melhor_dist'] for info in sumario_global.values())
else:
    melhor_alg_geral = "N/A"
    melhor_dist_abs = 0.0

# 3. Gera HTML Premium
html_content = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TSP Dashboard</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        
        <header>
            <div>
                <h1>Dashboard de Otimização</h1>
                <div class="timestamp">Execução: {EXECUTION_TIMESTAMP} | Bateria: {CHAVE_ESCOLHIDA.upper()}</div>
            </div>
        </header>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Instâncias</div>
                <div class="kpi-value">{total_instancias}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Algoritmo Vencedor</div>
                <div class="kpi-value">{melhor_alg_geral}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Melhor Distância (Abs)</div>
                <div class="kpi-value">{melhor_dist_abs:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Execuções / Cenário</div>
                <div class="kpi-value">{NUM_EXECUCOES}</div>
            </div>
        </div>

        <h2 class="section-title">Análise Comparativa Global</h2>
        <div class="card">
            <p style="color:var(--text-muted); margin-bottom:15px;">Dispersão de resultados entre instâncias e algoritmos.</p>
            <img src="boxplot_global.png" alt="Boxplot Global">
        </div>

        <h2 class="section-title">Resultados por Instância</h2>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Instância</th>
                        <th>Melhor Distância</th>
                        <th>Algoritmo Vencedor</th>
                        <th>Ação</th>
                    </tr>
                </thead>
                <tbody>
"""

for inst, info in sumario_global.items():
    html_content += f"""
                    <tr>
                        <td style="font-weight:600; color:var(--text-main);">{inst.upper()}</td>
                        <td>{info['melhor_dist']:.2f}</td>
                        <td><span style="color:var(--accent)">{info['melhor_alg']}</span></td>
                        <td><a href="{inst}/relatorio_{inst}.html" class="btn" target="_blank">Ver Detalhes &rarr;</a></td>
                    </tr>
    """

html_content += """
                </tbody>
            </table>
        </div>

        <footer style="text-align:center; color:var(--text-muted); margin-top:50px; font-size:0.8rem;">
            Sistema de Comparação de Meta-heurísticas v2.0
        </footer>

    </div>
</body>
</html>
"""

with open("resultados/index.html", "w", encoding="utf-8") as fh:
    fh.write(html_content)

print("\n=== Dashboard Profissional Gerado em 'resultados/index.html' ===")