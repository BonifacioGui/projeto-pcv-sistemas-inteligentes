"""
main.py - VERSÃO FINAL (Dark Mode + Correções)
"""
import os
import time
import json
import csv
import numpy as np
from datetime import datetime
from scipy import stats

import parser_tsplib
import utils
import visualizacao
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO
from recozimento_simulado import RecozimentoSimulado

# ================= CONFIGURAÇÕES =================
MODO_TESTE = True 

if MODO_TESTE:
    print("\n⚠️ [MODO TESTE] Rápido.")
    NUM_EXECUCOES = 2; NUM_GERACOES = 10; TAMANHO_POPULACAO = 10
else:
    print("\n🚀 [MODO ENTREGA] Robusto (30 execuções).")
    NUM_EXECUCOES = 30; NUM_GERACOES = 500; TAMANHO_POPULACAO = 50

INSTANCIAS = ["data/st70.tsp", "data/eil101.tsp", "data/ch130.tsp"]

# ================= ROTEIRO =================
ROTEIRO_EXECUCAO = [
    {"chave": "crossover", "pasta": "1_analise_crossover", "titulo": "Fase 1: Crossover"},
    {"chave": "mutacao",   "pasta": "2_analise_mutacao",   "titulo": "Fase 2: Mutação"},
    {"chave": "geral",     "pasta": "3_comparativo_final", "titulo": "Fase 3: Final"}
]

TODOS_EXPERIMENTOS = {
    "geral": [
        {"nome": "AG_Final", "algoritmo": "AG", "params": {"taxa_mutacao": 0.05, "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},
        {"nome": "ACO_Final", "algoritmo": "ACO", "params": {"num_formigas": TAMANHO_POPULACAO, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},
        {"nome": "SA_Final", "algoritmo": "SA", "params": {"temp_inicial": 1000, "cooling_rate": 0.995}}
    ],
    "mutacao": [
        {"nome": "AG_Swap", "algoritmo": "AG", "params": {"metodo_mutacao": "swap", "taxa_mutacao": 0.05, "metodo_crossover": "ox"}},
        {"nome": "AG_Inversion", "algoritmo": "AG", "params": {"metodo_mutacao": "inversion", "taxa_mutacao": 0.05, "metodo_crossover": "ox"}},
    ],
    "crossover": [
        {"nome": "AG_OX", "algoritmo": "AG", "params": {"metodo_crossover": "ox", "taxa_mutacao": 0.05, "metodo_mutacao": "inversion"}},
        {"nome": "AG_PMX", "algoritmo": "AG", "params": {"metodo_crossover": "pmx", "taxa_mutacao": 0.05, "metodo_mutacao": "inversion"}},
    ]
}

def garantir_pasta(p):
    if not os.path.exists(p): os.makedirs(p)

def mostrar_progresso(atual, total, texto):
    percent = (atual / total) * 100
    bar = "█" * int(percent / 5) + "-" * (20 - int(percent / 5))
    print(f"\r|{bar}| {percent:.1f}% - {texto}", end="", flush=True)

# ================= EXECUÇÃO =================
print("\n>>> INICIANDO BATERIA <<<")

for fase in ROTEIRO_EXECUCAO:
    CHAVE_ATUAL = fase["chave"]
    PASTA_FASE = f"resultados/{fase['pasta']}"
    TITULO_FASE = fase["titulo"]
    EXPERIMENTOS = TODOS_EXPERIMENTOS.get(CHAVE_ATUAL)

    print(f"\n\n{'='*40}\n📂 {TITULO_FASE}\n{'='*40}")
    garantir_pasta(PASTA_FASE)

    sumario_global_fase = {}
    resultados_globais_fase = {}
    total_steps = len(INSTANCIAS) * len(EXPERIMENTOS) * NUM_EXECUCOES
    current_step = 0

    for caminho in INSTANCIAS:
        nome_instancia = os.path.basename(caminho).split(".")[0]
        pasta_instancia = f"{PASTA_FASE}/{nome_instancia}"
        garantir_pasta(pasta_instancia)

        try:
            cidades = parser_tsplib.carregar_cidades(caminho)
            dist_matrix = utils.calcular_matriz_distancias(cidades)
        except: continue

        res_boxplot = {}
        res_tempos = {}
        melhor_dist_inst = float("inf")
        melhor_alg_inst = "N/A"
        melhor_rota_obj_inst = None
        
        # Lista para guardar imagens de convergência para o relatório detalhado
        imgs_convergencia = []

        for exp in EXPERIMENTOS:
            nome_exp = exp["nome"]
            alg = exp["algoritmo"]
            params = exp["params"]
            
            dists, tempos, hist_melhores, hist_medias = [], [], [], []

            for _ in range(NUM_EXECUCOES):
                current_step += 1
                mostrar_progresso(current_step, total_steps, f"{nome_instancia} > {nome_exp}")
                
                inicio = time.perf_counter()
                if alg == "AG":
                    m = AlgoritmoGenetico(cidades, dist_matrix=dist_matrix, **params, num_geracoes=NUM_GERACOES, tamanho_populacao=TAMANHO_POPULACAO)
                    res = m.executar()
                    hist_medias.append(m.historico_medias)
                    d, r = res.distancia, res.rota
                    hist_melhores.append(m.historico_melhores)
                elif alg == "ACO":
                    m = ACO(cidades, **params, num_iteracoes=NUM_GERACOES)
                    r, d = m.executar()
                    hist_melhores.append(m.historico_melhores)
                elif alg == "SA":
                    m = RecozimentoSimulado(cidades, dist_matrix=dist_matrix, **params, max_iterations=NUM_GERACOES*TAMANHO_POPULACAO)
                    r, d = m.executar()
                    hist_melhores.append(m.historico_melhores)
                
                tempos.append(time.perf_counter() - inicio)
                dists.append(d)

                if d < melhor_dist_inst:
                    melhor_dist_inst = d
                    melhor_alg_inst = nome_exp
                    melhor_rota_obj_inst = r

            res_boxplot[nome_exp] = dists
            res_tempos[nome_exp] = np.mean(tempos)
            
            nome_img_conv = f"convergencia_{nome_exp}.png"
            visualizacao.plotar_convergencia(hist_melhores, hist_medias if alg=="AG" else None, f"{pasta_instancia}/{nome_img_conv}")
            imgs_convergencia.append({"nome": nome_exp, "arquivo": nome_img_conv})

        # Gráficos da Instância
        visualizacao.plotar_boxplot_comparativo(res_boxplot, f"{pasta_instancia}/boxplot_{nome_instancia}.png")
        visualizacao.plotar_comparativo_tempo(res_tempos, f"{pasta_instancia}/tempos_{nome_instancia}.png")
        if melhor_rota_obj_inst:
            visualizacao.plotar_rota(melhor_rota_obj_inst, cidades, f"{pasta_instancia}/melhor_rota_{nome_instancia}.png")

        # T-Test
        ttest_html = ""
        ks = list(res_boxplot.keys())
        for i in range(len(ks)):
            for j in range(i+1, len(ks)):
                try:
                    s, p = stats.ttest_ind(res_boxplot[ks[i]], res_boxplot[ks[j]], equal_var=False)
                    cor = "var(--success)" if p < 0.05 else "var(--text-muted)"
                    sig = "Diferença Real" if p < 0.05 else "Empate"
                    ttest_html += f"<p><b>{ks[i]} vs {ks[j]}</b>: p={p:.4f} <span style='color:{cor}'>({sig})</span></p>"
                except: ttest_html += f"<p>{ks[i]} vs {ks[j]}: Dados idênticos</p>"

        # GERA RELATÓRIO DETALHADO (Agora chama a função bonita do visualizacao.py)
        imagens_dict = {
            "rota": f"melhor_rota_{nome_instancia}.png",
            "tempos": f"tempos_{nome_instancia}.png",
            "boxplot": f"boxplot_{nome_instancia}.png",
            "convergencias": imgs_convergencia
        }
        visualizacao.gerar_relatorio_instancia(
            f"{pasta_instancia}/relatorio_{nome_instancia}.html",
            nome_instancia, TITULO_FASE, melhor_alg_inst, melhor_dist_inst, ttest_html, imagens_dict
        )

        sumario_global_fase[nome_instancia] = {"melhor_dist": melhor_dist_inst, "melhor_alg": melhor_alg_inst}
        for k, v in res_boxplot.items(): resultados_globais_fase[f"{nome_instancia}_{k}"] = v

    # Dashboard Global da Fase
    visualizacao.plotar_boxplot_comparativo(resultados_globais_fase, f"{PASTA_FASE}/boxplot_global.png")
    
    if sumario_global_fase:
        lst_algs = [v['melhor_alg'] for v in sumario_global_fase.values()]
        best_of_phase = max(set(lst_algs), key=lst_algs.count) if lst_algs else "N/A"
        vals_dist = [v['melhor_dist'] for v in sumario_global_fase.values()]
        best_dist_phase = min(vals_dist) if vals_dist else 0.0
    else: best_of_phase = "N/A"; best_dist_phase = 0.0

    kpis = {"melhor_alg": best_of_phase, "melhor_dist": best_dist_phase, "total_exec": NUM_EXECUCOES * len(INSTANCIAS) * len(EXPERIMENTOS)}
    visualizacao.gerar_relatorio_final(f"{PASTA_FASE}/index.html", sumario_global_fase, kpis, TITULO_FASE)

print("\n🎉 RELATÓRIOS GERADOS COM SUCESSO!")