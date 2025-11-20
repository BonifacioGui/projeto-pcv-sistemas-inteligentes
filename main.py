# ======================================================================
# main.py - Versão Automática Completa (st70, eil101, ch130)
# Inclui:
# - index.html global
# - Relatório HTML por instância
# - T-test dentro do HTML
# - CSV de convergência média (AG)
# - Sumário global no console
# - CSS externo dark mode (resultados/style.css)
# ======================================================================

import os
import time
import json
import numpy as np
from scipy import stats
import csv
import html
from datetime import datetime

import parser_tsplib
import utils
import visualizacao
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO
from recozimento_simulado import RecozimentoSimulado


# ======================================================================
# CONFIGURAÇÕES
# ======================================================================

NUM_EXECUCOES = 2 # 30
NUM_GERACOES =  2 # 500
TAMANHO_POPULACAO = 5 # 50

EXECUTION_TIMESTAMP = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

INSTANCIAS = [
    "data/st70.tsp",
    "data/eil101.tsp",
    "data/ch130.tsp"
]

EXPERIMENTOS = [
    {"nome": "AG_Padrao", "algoritmo": "AG",
     "params": {"taxa_mutacao": 0.01, "metodo_crossover": "ox", "metodo_mutacao": "swap"}},
    {"nome": "AG_AltaMutacao", "algoritmo": "AG",
     "params": {"taxa_mutacao": 0.10, "metodo_crossover": "ox", "metodo_mutacao": "swap"}},
    {"nome": "ACO_Padrao", "algoritmo": "ACO",
     "params": {"num_formigas": 20, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},
    {"nome": "SA_Padrao", "algoritmo": "SA",
     "params": {"temp_inicial": 1000, "taxa_resfriamento": 0.995}},
]


def garantir_pasta(caminho):
    if not os.path.exists(caminho):
        os.makedirs(caminho)


# ======================================================================
# CSS DARK MODE (gerado automaticamente)
# ======================================================================

garantir_pasta("resultados")

CSS_CONTENT = """
/* ============================================================
   DARK MODE — Estilo profissional para relatórios e dashboards
   ============================================================ */

:root {
    --bg-main: #111418;
    --bg-card: #1a1e24;
    --bg-table-header: #20252c;
    --bg-table-row: #181c22;
    --text-primary: #e5e7eb;
    --text-secondary: #9ca3af;
    --border-color: #2a2f37;
    --link-color: #4ea1ff;
    --link-hover-color: #7bc0ff;
    --font-main: "Segoe UI", Roboto, Arial, sans-serif;
}

body {
    background: var(--bg-main);
    color: var(--text-primary);
    font-family: var(--font-main);
    margin: 0;
    padding: 32px;
    line-height: 1.6;
}

h1, h2, h3 {
    color: var(--text-primary);
    font-weight: 600;
    margin-top: 32px;
}

h1 { font-size: 32px; }
h2 { font-size: 24px; margin-top: 28px; }
h3 { font-size: 20px; margin-top: 20px; }

p {
    color: var(--text-secondary);
}

/* Links */
a {
    color: var(--link-color);
    text-decoration: none;
    font-weight: 500;
}
a:hover {
    color: var(--link-hover-color);
    text-decoration: underline;
}

/* Imagens (gráficos) */
img {
    max-width: 90%;
    display: block;
    margin: 24px auto;
    border-radius: 10px;
    border: 1px solid var(--border-color);
}

/* Tabelas */
table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 24px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
}

table th {
    background: var(--bg-table-header);
    padding: 12px;
    text-align: left;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-color);
}

table td {
    padding: 12px;
    border-bottom: 1px solid var(--border-color);
}

table tr:hover td {
    background: var(--bg-table-row);
}

/* Código, CSV, JSON */
pre {
    background: var(--bg-card);
    padding: 16px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    overflow-x: auto;
    font-size: 14px;
    color: var(--text-secondary);
}

/* Divisores */
hr {
    border: 0;
    border-top: 1px solid var(--border-color);
    margin: 40px 0;
}

/* Container centralizado (para relatórios longos) */
.container {
    max-width: 1100px;
    margin: auto;
}

"""

with open("resultados/style.css", "w", encoding="utf-8") as fcss:
    fcss.write(CSS_CONTENT)


# ======================================================================
# LOOP PRINCIPAL (TODAS AS INSTÂNCIAS)
# ======================================================================

resultados_globais_boxplot = {}
sumario_global = {}

for arquivo in INSTANCIAS:

    nome_instancia = os.path.basename(arquivo).split(".")[0]
    pasta_saida = f"resultados/{nome_instancia}"
    garantir_pasta(pasta_saida)

    print(f"\n=== Rodando instância: {nome_instancia} ===")

    try:
        cidades = parser_tsplib.carregar_cidades(arquivo)
        dist_matrix = utils.calcular_matriz_distancias(cidades)
    except Exception as e:
        print(f"Erro ao carregar {arquivo}: {e}")
        continue

    resultados_boxplot_instancia = {}
    resultados_tempo_instancia = {}

    melhor_global_rota = None
    melhor_global_dist = float("inf")

    # ==============================================================
    # Rodar todos os experimentos
    # ==============================================================
    for exp in EXPERIMENTOS:

        nome_exp = exp["nome"]
        alg = exp["algoritmo"]
        params = exp["params"]

        print(f"\n--- Executando {nome_exp} ({alg}) ---")

        resultados_dist = []
        resultados_tempo = []

        historico_30_melhores = []
        historico_30_medias = []

        for exe in range(NUM_EXECUCOES):
            print(f"  Exec {exe+1}/{NUM_EXECUCOES}...", end="", flush=True)

            start = time.perf_counter()

            # ========================= AG =========================
            if alg == "AG":
                modelo = AlgoritmoGenetico(
                    cidades=cidades,
                    tamanho_populacao=TAMANHO_POPULACAO,
                    num_geracoes=NUM_GERACOES,
                    taxa_mutacao=params["taxa_mutacao"],
                    taxa_elitismo=0.05,
                    metodo_selecao="torneio",
                    metodo_crossover=params["metodo_crossover"],
                    metodo_mutacao=params["metodo_mutacao"],
                    dist_matrix=dist_matrix
                )
                melhor = modelo.executar()
                dist = melhor.distancia
                rota = melhor.rota

                historico_30_melhores.append(modelo.historico_melhores)
                historico_30_medias.append(modelo.historico_medias)

            # ========================= ACO =========================
            elif alg == "ACO":
                modelo = ACO(
                    cidades=cidades,
                    num_formigas=params["num_formigas"],
                    num_iteracoes=NUM_GERACOES,
                    alfa=params["alfa"],
                    beta=params["beta"],
                    rho=params["rho"]
                )
                rota, dist = modelo.executar()
                historico_30_melhores.append(modelo.historico_melhores)

            # ========================= SA =========================
            elif alg == "SA":
                modelo = RecozimentoSimulado(
                    cidades=cidades,
                    temp_inicial=params["temp_inicial"],
                    max_iterations=NUM_GERACOES * TAMANHO_POPULACAO,
                    dist_matrix=dist_matrix
                )
                rota, dist = modelo.executar()
                historico_30_melhores.append(modelo.historico_melhores)

            tempo = time.perf_counter() - start

            resultados_dist.append(dist)
            resultados_tempo.append(tempo)

            if dist < melhor_global_dist:
                melhor_global_rota = rota
                melhor_global_dist = dist

            print(f" OK (dist={dist:.2f} | tempo={tempo:.2f}s)")

        # ========================= JSON =========================
        json.dump(
            {"distancias": resultados_dist, "tempos": resultados_tempo},
            open(f"{pasta_saida}/{nome_exp}.json", "w"), indent=2
        )

        # CSV de convergência média (AG)
        if alg == "AG":
            mean_curve = np.mean(historico_30_melhores, axis=0)
            with open(f"{pasta_saida}/convergencia_{nome_exp}_mean.csv", "w", newline="") as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["Geracao", "Melhor"])
                for i, val in enumerate(mean_curve):
                    w.writerow([i, val])

        # Coleta para boxplot
        resultados_boxplot_instancia[nome_exp] = resultados_dist
        resultados_globais_boxplot[f"{nome_instancia}_{nome_exp}"] = resultados_dist

        # Convergência
        if alg == "AG":
            visualizacao.plotar_convergencia(
                historico_30_melhores, historico_30_medias,
                f"{pasta_saida}/convergencia_{nome_exp}.png"
            )
        else:
            visualizacao.plotar_convergencia(
                historico_30_melhores, None,
                f"{pasta_saida}/convergencia_{nome_exp}.png"
            )

    # ==========================
    # Boxplot por instância
    # ==========================
    visualizacao.plotar_boxplot_comparativo(
        resultados_boxplot_instancia,
        f"{pasta_saida}/boxplot_{nome_instancia}.png"
    )

    # ==========================
    # Melhor rota
    # ==========================
    visualizacao.plotar_rota(
        melhor_global_rota,
        cidades,
        f"{pasta_saida}/melhor_rota_{nome_instancia}.png"
    )

    with open(f"{pasta_saida}/melhor_rota_{nome_instancia}.txt", "w") as fh:
        fh.write("Melhor rota encontrada:\n")
        fh.write(str(melhor_global_rota) + "\n")
        fh.write(f"Distância total: {melhor_global_dist}\n")

    # ==================================================================
    # T-TEST
    # ==================================================================
    ttest_results_html = "<h2>Testes T</h2>\n"

    pares = [
        ("AG_Padrao", "AG_AltaMutacao"),
        ("ACO_Padrao", "SA_Padrao"),
        ("AG_Padrao", "ACO_Padrao")
    ]

    for A, B in pares:
        if A in resultados_boxplot_instancia and B in resultados_boxplot_instancia:

            valsA = np.array(resultados_boxplot_instancia[A])
            valsB = np.array(resultados_boxplot_instancia[B])

            t, p = stats.ttest_ind(valsA, valsB)

            ttest_results_html += f"<p><b>{A} vs {B}</b><br>"
            ttest_results_html += f"Média A: {np.mean(valsA):.2f}<br>"
            ttest_results_html += f"Média B: {np.mean(valsB):.2f}<br>"
            ttest_results_html += f"t = {t:.4f} | p = {p:.4e}<br>"

            if p < 0.05:
                ttest_results_html += "<b>Conclusão:</b> diferença SIGNIFICATIVA (p < 0.05)</p>"
            else:
                ttest_results_html += "Conclusão: não significativa.</p>"

    # ==================================================================
    # CSV RESUMO
    # ==================================================================
    csv_path = f"{pasta_saida}/resumo_{nome_instancia}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Algoritmo", "Media", "Mediana", "Desvio Padrao", "Melhor", "Pior"])

        for alg_nome, valores in resultados_boxplot_instancia.items():
            arr = np.array(valores)
            writer.writerow([
                alg_nome,
                np.mean(arr),
                np.median(arr),
                np.std(arr),
                np.min(arr),
                np.max(arr)
            ])

    # ==================================================================
    # HTML POR INSTÂNCIA (dark + imagens grandes + links nova aba)
    # ==================================================================
    html_path = f"{pasta_saida}/relatorio_{nome_instancia}.html"

    with open(html_path, "w", encoding="utf-8") as fh:

        fh.write(f"""<!DOCTYPE html>
<html>
<head>
<meta charset='UTF-8'>
<link rel='stylesheet' href='../style.css'>
<title>Relatório {nome_instancia}</title>
</head>

<body>
<h1>Relatório da Instância {nome_instancia}</h1>
<p><b>Data de Execução:</b> {EXECUTION_TIMESTAMP}</p>
<p><b>Número de cidades:</b> {len(cidades)}</p>
<p><b>Execuções por experimento:</b> {NUM_EXECUCOES}</p>
<p><b>Gerações:</b> {NUM_GERACOES}</p>

<hr>

<h2>Boxplot Geral</h2>
<img src='boxplot_{nome_instancia}.png'>

<h2>Melhor Rota</h2>
<img src='melhor_rota_{nome_instancia}.png'>

<h2>Convergência por Algoritmo</h2>
""")

        # Imagens de convergência
        for exp in EXPERIMENTOS:
            nome_exp = exp["nome"]
            conv_path = f"convergencia_{nome_exp}.png"

            fh.write(f"<h3>{nome_exp}</h3>")
            if os.path.exists(f"{pasta_saida}/{conv_path}"):
                fh.write(f"<img src='{conv_path}'><br>")

        fh.write("<hr>")
        fh.write(ttest_results_html)
        fh.write("<hr>")

        fh.write("<h2>Resumo Estatístico</h2>")
        fh.write("<pre>")
        with open(csv_path, "r") as csvf:
            fh.write(html.escape(csvf.read()))
        fh.write("</pre>")

        fh.write("</body></html>")

    # ==================================================================
    # Sumário global para esta instância
    # ==================================================================
    sumario_global[nome_instancia] = {
        "melhor_distancia": melhor_global_dist,
        "melhor_algoritmo": min(
            resultados_boxplot_instancia,
            key=lambda x: np.mean(resultados_boxplot_instancia[x])
        ),
        "media_algoritmos": {
            alg: float(np.mean(vals))
            for alg, vals in resultados_boxplot_instancia.items()
        }
    }

# ======================================================================
# BOXPLOT GLOBAL
# ======================================================================

visualizacao.plotar_boxplot_comparativo(
    resultados_globais_boxplot,
    "resultados/boxplot_global.png"
)

# ======================================================================
# INDEX.HTML GLOBAL (DARK MODE + LINKS EM NOVA ABA)
# ======================================================================

with open("resultados/index.html", "w", encoding="utf-8") as fh:

    fh.write(f"""<!DOCTYPE html>
<html>
<head>
<meta charset='UTF-8'>
<link rel='stylesheet' href='style.css'>
<title>Dashboard Global</title>
</head>

<body>
<h1>Dashboard Global — Resultados</h1>

<p><b>Execução:</b> {EXECUTION_TIMESTAMP}</p>

<hr>

<h2>Boxplot Global</h2>
<img src='boxplot_global.png'>

<h2>Relatórios por Instância</h2>

<table>
<tr>
<th>Instância</th>
<th>Relatório</th>
<th>Boxplot</th>
<th>Rota</th>
<th>CSV Resumo</th>
</tr>
""")

    for arquivo in INSTANCIAS:
        nome_instancia = os.path.basename(arquivo).split(".")[0]

        fh.write(f"""
<tr>
<td>{nome_instancia}</td>
<td><a href='{nome_instancia}/relatorio_{nome_instancia}.html' target='_blank'>Abrir Relatório</a></td>
<td><a href='{nome_instancia}/boxplot_{nome_instancia}.png' target='_blank'>Boxplot</a></td>
<td><a href='{nome_instancia}/melhor_rota_{nome_instancia}.png' target='_blank'>Rota</a></td>
<td><a href='{nome_instancia}/resumo_{nome_instancia}.csv' target='_blank'>Resumo CSV</a></td>
</tr>
""")

    fh.write("</table><hr>")

    fh.write("<h2>Sumário Final</h2>")

    for inst, dados in sumario_global.items():
        fh.write(f"<h3>{inst}</h3>")
        fh.write(f"<p>Melhor distância obtida: {dados['melhor_distancia']}</p>")
        fh.write(f"<p>Melhor algoritmo (pela média): {dados['melhor_algoritmo']}</p>")
        fh.write("<pre>" + html.escape(json.dumps(dados["media_algoritmos"], indent=2)) + "</pre>")

    fh.write("</body></html>")

# ======================================================================
# SUMÁRIO NO CONSOLE
# ======================================================================

print("\n===== SUMÁRIO GLOBAL =====")
for inst, dados in sumario_global.items():
    print(f"\nInstância {inst}:")
    print(f"  Melhor Distância: {dados['melhor_distancia']}")
    print(f"  Melhor Algoritmo: {dados['melhor_algoritmo']}")

print("\nExecução finalizada com sucesso.")
