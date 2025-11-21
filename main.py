"""
main.py - Versão Final Corrigida (Test Mode)
"""

import os
import time
import json
import csv
import html
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
# 1. CONFIGURAÇÕES (MODO DE TESTE)
# ================================================================

NUM_EXECUCOES     = 2     # TESTE (final: 30)
NUM_GERACOES      = 50    # TESTE (final: 500)
TAMANHO_POPULACAO = 20    # TESTE (final: 50)

INSTANCIAS = [
    "data/st70.tsp",
    "data/eil101.tsp",
    "data/ch130.tsp"
]

EXECUTION_TIMESTAMP = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


# ================================================================
# 2. EXPERIMENTOS
# ================================================================

TODOS_EXPERIMENTOS = {
    "geral": [
        {"nome": "AG_Padrao", "algoritmo": "AG",
         "params": {"taxa_mutacao": 0.01, "metodo_crossover": "ox", "metodo_mutacao": "swap"}},

        {"nome": "AG_AltaMutacao", "algoritmo": "AG",
         "params": {"taxa_mutacao": 0.10, "metodo_crossover": "ox", "metodo_mutacao": "swap"}},

        {"nome": "ACO_Padrao", "algoritmo": "ACO",
         "params": {"num_formigas": 20, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},

        {"nome": "SA_Padrao", "algoritmo": "SA",
         "params": {"temp_inicial": 1000, "taxa_resfriamento": 0.995}},
    ]
}


# ================================================================
# QUAL EXPERIMENTO RODAR?
# ================================================================

CHAVE_ESCOLHIDA = "geral"
EXPERIMENTO_ATUAL = TODOS_EXPERIMENTOS[CHAVE_ESCOLHIDA]


# ================================================================
# 3. PASTAS + CSS
# ================================================================

def garantir_pasta(p):
    if not os.path.exists(p):
        os.makedirs(p)

garantir_pasta("resultados")

CSS_CONTENT = """
img {
    max-width: 75%;
    display: block;
    margin: 20px auto;
    border: 1px solid #333;
    border-radius: 10px;
}
body { background: #111418; color: #e5e7eb; font-family: Arial; padding: 30px; }
a { color: #4ea1ff; }
table { width: 100%; border-collapse: collapse; }
td, th { padding: 12px; border-bottom: 1px solid #333; }
"""

with open("resultados/style.css", "w", encoding="utf-8") as f:
    f.write(CSS_CONTENT)


# ================================================================
# 4. EXECUÇÃO PRINCIPAL
# ================================================================

resultados_globais = {}
sumario_global = {}

print(f"\n>>> INICIANDO TESTE: {CHAVE_ESCOLHIDA.upper()} <<<\n")

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

    resultados_boxplot_instancia = {}
    melhor_global_dist = float("inf")
    melhor_global_rota = None

    # ------------------------------------------------------------
    # Executa cada algoritmo selecionado
    # ------------------------------------------------------------
    for exp in EXPERIMENTO_ATUAL:

        nome_exp = exp["nome"]
        alg = exp["algoritmo"]
        params = exp["params"]

        print(f" -> Executando {nome_exp} ({alg}) ... ", end="")

        resultados_dist = []
        resultados_tempo = []
        historico_30_melhores = []
        historico_30_medias = []

        for k in range(NUM_EXECUCOES):
            inicio = time.perf_counter()

            # ------------------------ AG ------------------------
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
                historico_30_melhores.append(modelo.historico_melhores)
                historico_30_medias.append(modelo.historico_medias)

            # ------------------------ ACO ------------------------
            elif alg == "ACO":
                modelo = ACO(
                    cidades=cidades,
                    num_formigas=params.get("num_formigas", TAMANHO_POPULACAO), # Usa a variável global se não houver específico
                    num_iteracoes=NUM_GERACOES,
                    alfa=params.get("alfa", 1.0),
                    beta=params.get("beta", 2.5),
                    rho=params.get("rho", 0.1)
                )
                rota, dist = modelo.executar()
                historico_30_melhores.append(modelo.historico_melhores)

            # ------------------------ SA ------------------------
            elif alg == "SA":
                modelo = RecozimentoSimulado(
                    cidades=cidades,
                    temp_inicial=params.get("temp_inicial", 1000),
                    max_iterations=NUM_GERACOES * TAMANHO_POPULACAO,
                    dist_matrix=dist_matrix
                )
                rota, dist = modelo.executar()
                historico_30_melhores.append(modelo.historico_melhores)

            tempo = time.perf_counter() - inicio
            resultados_dist.append(dist)
            resultados_tempo.append(tempo)

            if dist < melhor_global_dist:
                melhor_global_dist = dist
                melhor_global_rota = rota

            print(".", end="", flush=True)

        print(f" OK (Média: {np.mean(resultados_dist):.2f})")

        # Salva JSON
        with open(f"{pasta_saida}/{nome_exp}.json", "w") as jf:
            json.dump(
                {"distancias": resultados_dist, "tempos": resultados_tempo},
                jf,
                indent=2
            )

        # Convergência
        media_pop = historico_30_medias if alg == "AG" else None
        visualizacao.plotar_convergencia(
            historico_30_melhores, media_pop,
            f"{pasta_saida}/convergencia_{nome_exp}.png"
        )

        resultados_boxplot_instancia[nome_exp] = resultados_dist

    # ------------------------------------------------------------
    # Boxplot da instância
    # ------------------------------------------------------------
    visualizacao.plotar_boxplot_comparativo(
        resultados_boxplot_instancia,
        f"{pasta_saida}/boxplot_{nome_instancia}.png"
    )

    # Melhor rota
    visualizacao.plotar_rota(
        melhor_global_rota, cidades,
        f"{pasta_saida}/melhor_rota_{nome_instancia}.png"
    )

    # ------------------------------------------------------------
    # Relatório HTML (instância)
    # ------------------------------------------------------------
    ttest_html = "<h2>Testes Estatísticos</h2>"
    chaves = list(resultados_boxplot_instancia.keys())

    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            a, b = chaves[i], chaves[j]
            da, db = resultados_boxplot_instancia[a], resultados_boxplot_instancia[b]

            t, p = stats.ttest_ind(da, db, equal_var=False)
            signif = "<b>SIM</b>" if p < 0.05 else "NÃO"

            ttest_html += f"<p><b>{a} vs {b}</b><br>Média: {np.mean(da):.2f} vs {np.mean(db):.2f}<br>p={p:.4e} ({signif})</p>"

    with open(f"{pasta_saida}/relatorio_{nome_instancia}.html", "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><link rel='stylesheet' href='../style.css'></head><body>")
        fh.write(f"<h1>Relatório: {nome_instancia}</h1>")
        fh.write(f"<img src='boxplot_{nome_instancia}.png'>")
        fh.write(f"<img src='melhor_rota_{nome_instancia}.png'>")
        fh.write(ttest_html)

        fh.write("<h2>Convergência</h2>")
        for exp in EXPERIMENTO_ATUAL:
            fh.write(f"<h3>{exp['nome']}</h3><img src='convergencia_{exp['nome']}.png'>")

        fh.write("</body></html>")

    melhor_alg = min(resultados_boxplot_instancia, key=lambda k: np.mean(resultados_boxplot_instancia[k]))
    sumario_global[nome_instancia] = {
        "melhor": melhor_global_dist,
        "algoritmo": melhor_alg
    }

    for k, v in resultados_boxplot_instancia.items():
        resultados_globais[f"{nome_instancia}_{k}"] = v


# ================================================================
# 5. DASHBOARD GLOBAL
# ================================================================

visualizacao.plotar_boxplot_comparativo(
    resultados_globais, "resultados/boxplot_global.png"
)

with open("resultados/index.html", "w", encoding="utf-8") as fh:
    fh.write(f"<html><head><link rel='stylesheet' href='style.css'></head><body>")
    fh.write(f"<h1>Dashboard Global — {EXECUTION_TIMESTAMP}</h1>")
    fh.write("<img src='boxplot_global.png'>")

    fh.write("<h2>Sumário</h2><table>")
    fh.write("<tr><th>Instância</th><th>Melhor</th><th>Algoritmo</th><th>Relatório</th></tr>")

    for inst, info in sumario_global.items():
        fh.write(f"<tr><td>{inst}</td>")
        fh.write(f"<td>{info['melhor']:.2f}</td>")
        fh.write(f"<td>{info['algoritmo']}</td>")
        fh.write(f"<td><a href='{inst}/relatorio_{inst}.html' target='_blank'>Abrir</a></td></tr>")

    fh.write("</table></body></html>")

print("\n=== Execução Finalizada ===")
