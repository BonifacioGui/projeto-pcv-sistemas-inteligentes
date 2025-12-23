"""
main.py - Versão Final para Entrega
Contém todos os cenários de teste exigidos no projeto.
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

NUM_EXECUCOES     = 30     # TESTE (final: 30)
NUM_GERACOES      = 500    # TESTE (final: 500)
TAMANHO_POPULACAO = 50    # TESTE (final: 50)

INSTANCIAS = [
    "data/st70.tsp",
    "data/eil101.tsp",
    "data/ch130.tsp"
]

EXECUTION_TIMESTAMP = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


# ================================================================
# 2. BANCO DE EXPERIMENTOS (TODOS OS CENÁRIOS)
# ================================================================

TODOS_EXPERIMENTOS = {
    # --- ANÁLISE 9: COMPARAÇÃO GERAL (AG vs ACO vs SA) ---
    # Use para comparar o desempenho final dos 3 algoritmos nas 3 bases.
    "geral": [
        {"nome": "AG_Padrao", "algoritmo": "AG",
         "params": {"taxa_mutacao": 0.01, "metodo_crossover": "ox", "metodo_mutacao": "swap"}},
        
        {"nome": "AG_AltaMutacao", "algoritmo": "AG",
         "params": {"taxa_mutacao": 0.10, "metodo_crossover": "ox", "metodo_mutacao": "swap"}},
        
        {"nome": "ACO_Padrao", "algoritmo": "ACO",
         "params": {"num_formigas": TAMANHO_POPULACAO, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},
        
        {"nome": "SA_Padrao", "algoritmo": "SA",
         "params": {"temp_inicial": 1000, "taxa_resfriamento": 0.995}}
    ],

    # --- ANÁLISE 2: SELEÇÃO (Roleta vs Torneio) ---
    "selecao": [
        {"nome": "AG_Torneio", "algoritmo": "AG",
         "params": {"metodo_selecao": "torneio", "taxa_mutacao": 0.01,
                    "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},

        {"nome": "AG_Roleta", "algoritmo": "AG",
         "params": {"metodo_selecao": "roleta", "taxa_mutacao": 0.01,
                    "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},
    ],
    
    # --- ANÁLISE 3: MUTAÇÃO (Troca vs Inversão) ---
    "mutacao": [
        {"nome": "AG_MutacaoTroca", "algoritmo": "AG",
         "params": {"metodo_mutacao": "swap", "metodo_selecao": "torneio",
                    "metodo_crossover": "ox", "taxa_mutacao": 0.01}},

        {"nome": "AG_MutacaoInversao", "algoritmo": "AG",
         "params": {"metodo_mutacao": "inversion", "metodo_selecao": "torneio",
                    "metodo_crossover": "ox", "taxa_mutacao": 0.01}}
    ],

    # --- ANÁLISE 4: ELITISMO (0% vs 5% vs 10%) ---
    "elitismo": [
        {"nome": "AG_Elitismo_0", "algoritmo": "AG",
         "params": {"taxa_elitismo": 0.00, "taxa_mutacao": 0.01, 
                    "metodo_selecao": "torneio", "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},

        {"nome": "AG_Elitismo_5", "algoritmo": "AG",
         "params": {"taxa_elitismo": 0.05, "taxa_mutacao": 0.01,
                    "metodo_selecao": "torneio", "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},

        {"nome": "AG_Elitismo_10", "algoritmo": "AG",
         "params": {"taxa_elitismo": 0.10, "taxa_mutacao": 0.01,
                    "metodo_selecao": "torneio", "metodo_crossover": "ox", "metodo_mutacao": "inversion"}}
    ],

    # --- ANÁLISE EXTRA: CROSSOVER (OX vs PMX) ---
    # Requisito: "Implementar pelo menos dois operadores diferentes"
    "crossover": [
        {"nome": "AG_Crossover_OX", "algoritmo": "AG",
         "params": {"metodo_crossover": "ox", "metodo_selecao": "torneio", 
                    "metodo_mutacao": "inversao", "taxa_mutacao": 0.01, "taxa_elitismo": 0.05}},

        {"nome": "AG_Crossover_PMX", "algoritmo": "AG",
         "params": {"metodo_crossover": "pmx", "metodo_selecao": "torneio", 
                    "metodo_mutacao": "inversao", "taxa_mutacao": 0.01, "taxa_elitismo": 0.05}}
    ]
}


# ============================================================
# 3. SELETOR: QUAL EXPERIMENTO RODAR AGORA?
# ============================================================
# Mude esta string para: "geral", "selecao", "mutacao", "elitismo" ou "crossover"
CHAVE_ESCOLHIDA = "geral"

EXPERIMENTO_ATUAL = TODOS_EXPERIMENTOS[CHAVE_ESCOLHIDA]


# ================================================================
# 4. PREPARAÇÃO DE PASTAS E ESTILO
# ================================================================
def garantir_pasta(p):
    if not os.path.exists(p):
        os.makedirs(p)

garantir_pasta("resultados")

CSS_CONTENT = """
:root { --bg: #111418; --card: #1a1e24; --text: #e5e7eb; --border: #2a2f37; --link: #4ea1ff; }
body { background: var(--bg); color: var(--text); font-family: sans-serif; padding: 30px; }
h1, h2 { border-bottom: 1px solid var(--border); padding-bottom: 10px; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; background: var(--card); }
th { background: #20252c; text-align: left; padding: 12px; border-bottom: 1px solid var(--border); }
td { padding: 12px; border-bottom: 1px solid var(--border); }
a { color: var(--link); text-decoration: none; }
img { max-width: 800px; display: block; margin: 20px 0; border: 1px solid var(--border); border-radius: 8px; }
pre { background: var(--card); padding: 15px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); }
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

        # --- 30 Execuções ---
        for k in range(NUM_EXECUCOES):
            inicio = time.perf_counter()

            # Instanciação dinâmica
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
            
            # Feedback visual
            print(".", end="", flush=True)

        print(f" OK (Média: {np.mean(resultados_dist):.2f})")

        # Salva JSON bruto
        with open(f"{pasta_saida}/{nome_exp}.json", "w") as jf:
            json.dump(
                {"distancias": resultados_dist, "tempos": resultados_tempo},
                jf,
                indent=2
            )

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
    
    # Salva resumo da melhor rota em TXT
    with open(f"{pasta_saida}/melhor_rota_{nome_instancia}.txt", "w") as f:
        f.write(f"Distancia: {melhor_global_dist}\nRota: {melhor_global_rota}")

    # 3. Testes Estatísticos (T-Student) para o Relatório HTML
    ttest_html = "<h2>Testes Estatísticos (T-Student)</h2>"
    chaves = list(resultados_boxplot_instancia.keys())
    
    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            alg_a = chaves[i]
            alg_b = chaves[j]
            dados_a = resultados_boxplot_instancia[alg_a]
            dados_b = resultados_boxplot_instancia[alg_b]

            # ttest_ind assume variâncias iguais por padrão; equal_var=False é mais seguro
            t_stat, p_val = stats.ttest_ind(dados_a, dados_b, equal_var=False)
            
            significativo = "<b>SIM</b>" if p_val < 0.05 else "NÃO"
            ttest_html += f"<p><b>{alg_a} vs {alg_b}</b><br>"
            ttest_html += f"Médias: {np.mean(dados_a):.2f} vs {np.mean(dados_b):.2f}<br>"
            ttest_html += f"p-value: {p_val:.4e} (Significativo? {significativo})</p>"

    # 4. CSV Resumo
    with open(f"{pasta_saida}/resumo_{nome_instancia}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Algoritmo", "Media", "Desvio", "Melhor", "Pior", "Tempo Medio"])
        for k, v in resultados_boxplot_instancia.items():
            # Recupera tempo médio do JSON
            try:
                dados_json = json.load(open(f"{pasta_saida}/{k}.json"))
                tempo_medio = np.mean(dados_json["tempos"])
            except:
                tempo_medio = 0.0
            w.writerow([k, np.mean(v), np.std(v), np.min(v), np.max(v), tempo_medio])

    # 5. Relatório HTML da Instância
    with open(f"{pasta_saida}/relatorio_{nome_instancia}.html", "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><link rel='stylesheet' href='../style.css'></head><body>")
        fh.write(f"<h1>Relatório: {nome_instancia}</h1>")
        fh.write(f"<p>Experimento: {CHAVE_ESCOLHIDA.upper()}</p>")
        
        fh.write("<h2>Comparativo de Desempenho</h2>")
        fh.write(f"<img src='boxplot_{nome_instancia}.png'>")
        
        fh.write("<h2>Melhor Solução Encontrada</h2>")
        fh.write(f"<p>Distância: {melhor_global_dist:.4f}</p>")
        fh.write(f"<img src='melhor_rota_{nome_instancia}.png'>")
        
        fh.write(ttest_html)
        
        fh.write("<h2>Convergência Individual</h2>")
        for exp in EXPERIMENTO_ATUAL:
            nome_img = exp["nome"]
            fh.write(f"<h3>{nome_img}</h3><img src='convergencia_{nome_img}.png'>")
            
        fh.write("</body></html>")

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
# 6. DASHBOARD GLOBAL
# ================================================================

visualizacao.plotar_boxplot_comparativo(
    resultados_globais, "resultados/boxplot_global.png"
)

with open("resultados/index.html", "w", encoding="utf-8") as fh:
    fh.write(f"<html><head><link rel='stylesheet' href='style.css'></head><body>")
    fh.write(f"<h1>Dashboard Global - {EXECUTION_TIMESTAMP}</h1>")
    fh.write(f"<h3>Bateria de Testes: {CHAVE_ESCOLHIDA.upper()}</h3>")
    
    fh.write("<h2>Visão Geral (Boxplot)</h2>")
    fh.write("<img src='boxplot_global.png'>")
    
    fh.write("<h2>Resumo por Instância</h2>")
    fh.write("<table><tr><th>Instância</th><th>Melhor</th><th>Algoritmo (Média)</th><th>Relatório</th></tr>")
    
    for inst, info in sumario_global.items():
        fh.write(f"<tr><td>{inst}</td>")
        fh.write(f"<td>{info['melhor_dist']:.2f}</td>")
        fh.write(f"<td>{info['melhor_alg']}</td>")
        fh.write(f"<td><a href='{inst}/relatorio_{inst}.html' target='_blank'>Abrir Relatório</a></td></tr>")
    
    fh.write("</table></body></html>")

print("\n=== Execução Finalizada com Sucesso! ===")