"""
main.py - Versão CORRIGIDA (Automação Total)
Correção: Adição de parâmetros obrigatórios (taxa_mutacao) nos experimentos de Crossover e Mutação.
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
# 1. CONFIGURAÇÕES - MODO ENTREGA ATIVADO
# ================================================================
MODO_TESTE = False 

if MODO_TESTE:
    print("\n⚠️  [MODO TESTE] Execução rápida (2 execuções, 10 gerações).")
    NUM_EXECUCOES     = 2
    NUM_GERACOES      = 10
    TAMANHO_POPULACAO = 10
else:
    print("\n🚀 [MODO ENTREGA] Execução robusta (30 execuções, 500 gerações).")
    print("Isso pode levar de 20 a 40 minutos. Vá tomar um café! ☕")
    NUM_EXECUCOES     = 30      # Exigência do PDF
    NUM_GERACOES      = 500     # Para garantir convergência
    TAMANHO_POPULACAO = 50      # Padrão de literatura

INSTANCIAS = ["data/st70.tsp", "data/eil101.tsp", "data/ch130.tsp"]
EXECUTION_TIMESTAMP = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# ================================================================
# 2. ROTEIRO DE EXECUÇÃO (Fases do Projeto)
# ================================================================
ROTEIRO_EXECUCAO = [
    # Fase 1: Análise de Operadores de Crossover
    {"chave": "crossover", "pasta": "1_analise_crossover", "titulo": "Comparativo de Crossover (OX vs PMX)"},
    
    # Fase 2: Análise de Operadores de Mutação
    {"chave": "mutacao",   "pasta": "2_analise_mutacao",   "titulo": "Comparativo de Mutação (Swap vs Inversion)"},
    
    # Fase 3: Comparativo Final (AG vs ACO vs SA)
    {"chave": "geral",     "pasta": "3_comparativo_final", "titulo": "Comparativo Final (Meta-heurísticas)"}
]

# CORREÇÃO AQUI: Adicionamos os parâmetros faltantes em todos os dicionários
TODOS_EXPERIMENTOS = {
    "geral": [
        {"nome": "AG_Final", "algoritmo": "AG", "params": {"taxa_mutacao": 0.05, "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},
        {"nome": "ACO_Final", "algoritmo": "ACO", "params": {"num_formigas": TAMANHO_POPULACAO, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},
        {"nome": "SA_Final", "algoritmo": "SA", "params": {"temp_inicial": 1000, "cooling_rate": 0.995}}
    ],
    "mutacao": [
        # Variando Mutação, mas fixando Crossover (OX) e Taxa (5%)
        {"nome": "AG_Swap", "algoritmo": "AG", "params": {"metodo_mutacao": "swap", "taxa_mutacao": 0.05, "metodo_crossover": "ox"}},
        {"nome": "AG_Inversion", "algoritmo": "AG", "params": {"metodo_mutacao": "inversion", "taxa_mutacao": 0.05, "metodo_crossover": "ox"}},
    ],
    "crossover": [
        # Variando Crossover, mas fixando Mutação (Inversion) e Taxa (5%)
        {"nome": "AG_OX", "algoritmo": "AG", "params": {"metodo_crossover": "ox", "taxa_mutacao": 0.05, "metodo_mutacao": "inversion"}},
        {"nome": "AG_PMX", "algoritmo": "AG", "params": {"metodo_crossover": "pmx", "taxa_mutacao": 0.05, "metodo_mutacao": "inversion"}},
    ]
}

def garantir_pasta(p):
    if not os.path.exists(p): os.makedirs(p)

# ================================================================
# 3. LOOP PRINCIPAL (Automático)
# ================================================================

print("\n>>> INICIANDO AUTOMAÇÃO TOTAL DO PROJETO (MODO REAL) <<<")
print(f"Baterias agendadas: {[b['chave'] for b in ROTEIRO_EXECUCAO]}")

for fase in ROTEIRO_EXECUCAO:
    CHAVE_ATUAL = fase["chave"]
    PASTA_FASE = f"resultados/{fase['pasta']}"
    TITULO_FASE = fase["titulo"]
    EXPERIMENTOS_DA_VEZ = TODOS_EXPERIMENTOS.get(CHAVE_ATUAL)

    print(f"\n" + "="*60)
    print(f"▶️  INICIANDO FASE: {TITULO_FASE}")
    print(f"📂 Salvando em: {PASTA_FASE}")
    print("="*60)

    garantir_pasta(PASTA_FASE)
    
    # Variáveis para o Dashboard desta Fase
    sumario_global_fase = {}
    resultados_globais_fase = {}

    for caminho in INSTANCIAS:
        nome_instancia = os.path.basename(caminho).split(".")[0]
        pasta_saida_instancia = f"{PASTA_FASE}/{nome_instancia}"
        garantir_pasta(pasta_saida_instancia)

        print(f"\n  Processando Instância: {nome_instancia} ...")
        
        try:
            cidades = parser_tsplib.carregar_cidades(caminho)
            dist_matrix = utils.calcular_matriz_distancias(cidades)
        except Exception as e:
            print(f"  Erro ao carregar {caminho}: {e}")
            continue

        resultados_boxplot = {}
        resultados_tempos = {}
        melhor_dist_inst = float("inf")
        melhor_alg_inst = "N/A"
        melhor_rota_obj_inst = None
        ttest_html = ""

        # --- LOOP DOS ALGORITMOS ---
        for exp in EXPERIMENTOS_DA_VEZ:
            nome_exp = exp["nome"]
            alg = exp["algoritmo"]
            params = exp["params"]
            
            print(f"    -> {nome_exp} ({alg})... ", end="")
            
            dists = []
            tempos = []
            hist_melhores = []
            hist_medias = []

            for _ in range(NUM_EXECUCOES):
                inicio = time.perf_counter()
                if alg == "AG":
                    modelo = AlgoritmoGenetico(cidades, dist_matrix=dist_matrix, **params, num_geracoes=NUM_GERACOES, tamanho_populacao=TAMANHO_POPULACAO)
                    res = modelo.executar()
                    hist_medias.append(modelo.historico_medias)
                    d_atual, r_atual = res.distancia, res.rota
                    hist_melhores.append(modelo.historico_melhores)
                elif alg == "ACO":
                    modelo = ACO(cidades, **params, num_iteracoes=NUM_GERACOES)
                    r_atual, d_atual = modelo.executar()
                    hist_melhores.append(modelo.historico_melhores)
                elif alg == "SA":
                    modelo = RecozimentoSimulado(cidades, dist_matrix=dist_matrix, **params, max_iterations=NUM_GERACOES*TAMANHO_POPULACAO)
                    r_atual, d_atual = modelo.executar()
                    hist_melhores.append(modelo.historico_melhores)
                
                tempos.append(time.perf_counter() - inicio)
                dists.append(d_atual)

                if d_atual < melhor_dist_inst:
                    melhor_dist_inst = d_atual
                    melhor_alg_inst = nome_exp
                    melhor_rota_obj_inst = r_atual

            # Estatísticas do Algoritmo
            t_medio = np.mean(tempos)
            resultados_tempos[nome_exp] = t_medio
            resultados_boxplot[nome_exp] = dists
            print(f"OK ({t_medio:.2f}s)")

            # Gráfico de Convergência Individual
            visualizacao.plotar_convergencia(
                hist_melhores, 
                hist_medias if alg == "AG" else None, 
                f"{pasta_saida_instancia}/convergencia_{nome_exp}.png"
            )

        # --- FIM ALGORITMOS: GERAÇÃO DE RELATÓRIOS DA INSTÂNCIA ---
        
        # 1. Boxplot e Tempos
        visualizacao.plotar_boxplot_comparativo(resultados_boxplot, f"{pasta_saida_instancia}/boxplot_{nome_instancia}.png")
        visualizacao.plotar_comparativo_tempo(resultados_tempos, f"{pasta_saida_instancia}/tempos_{nome_instancia}.png")
        
        # 2. Melhor Rota
        if melhor_rota_obj_inst:
            visualizacao.plotar_rota(melhor_rota_obj_inst, cidades, f"{pasta_saida_instancia}/melhor_rota_{nome_instancia}.png")

        # 3. CSV
        with open(f"{pasta_saida_instancia}/resumo_{nome_instancia}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Algoritmo", "Media", "Desvio", "Min", "Max", "Tempo(s)"])
            for nm, dd in resultados_boxplot.items():
                writer.writerow([nm, f"{np.mean(dd):.2f}", f"{np.std(dd):.2f}", f"{np.min(dd):.2f}", f"{np.max(dd):.2f}", f"{resultados_tempos[nm]:.4f}"])

        # 4. T-Tests
        chaves = list(resultados_boxplot.keys())
        for i in range(len(chaves)):
            for j in range(i + 1, len(chaves)):
                a, b = chaves[i], chaves[j]
                try:
                    s, p = stats.ttest_ind(resultados_boxplot[a], resultados_boxplot[b], equal_var=False)
                    color = "var(--success)" if p < 0.05 else "var(--text-muted)"
                    ttest_html += f"<p><b>{a} vs {b}</b>: p={p:.4e} <span style='color:{color}'>({'SIG' if p<0.05 else 'NÃO'})</span></p>"
                except: ttest_html += f"<p>{a} vs {b}: Sem variância</p>"

        # 5. HTML Individual da Instância
        with open(f"{pasta_saida_instancia}/relatorio_{nome_instancia}.html", "w", encoding="utf-8") as f:
            f.write(f"""
            <html><head><link rel='stylesheet' href='../../style.css'></head><body>
            <div class='container'><header><h1>{nome_instancia.upper()} - {TITULO_FASE}</h1></header>
            <h2 class='section-title'>1. Vencedor: {melhor_alg_inst} ({melhor_dist_inst:.2f})</h2>
            <div class='card'><img src='melhor_rota_{nome_instancia}.png'></div>
            <h2 class='section-title'>2. Tempos</h2><div class='card'><img src='tempos_{nome_instancia}.png'></div>
            <h2 class='section-title'>3. Estabilidade</h2><div class='card'><img src='boxplot_{nome_instancia}.png'></div>
            <h2 class='section-title'>4. Estatística</h2><div class='card'>{ttest_html}</div>
            <h2 class='section-title'>5. Convergências</h2><div style='display:grid;grid-template-columns:1fr 1fr;gap:20px'>
            """)
            for exp in EXPERIMENTOS_DA_VEZ:
                f.write(f"<div><h3>{exp['nome']}</h3><img src='convergencia_{exp['nome']}.png'></div>")
            f.write("</div></div></body></html>")

        # Atualiza globais da fase
        sumario_global_fase[nome_instancia] = {"melhor_dist": melhor_dist_inst, "melhor_alg": melhor_alg_inst}
        for k, v in resultados_boxplot.items(): resultados_globais_fase[f"{nome_instancia}_{k}"] = v

    # --- FIM DA FASE: GERA DASHBOARD DA BATERIA ---
    visualizacao.plotar_boxplot_comparativo(resultados_globais_fase, f"{PASTA_FASE}/boxplot_geral_fase.png")
    
    if sumario_global_fase:
        lst = [v['melhor_alg'] for v in sumario_global_fase.values()]
        best_of_phase = max(set(lst), key=lst.count) if lst else "N/A"
    else: best_of_phase = "N/A"

    kpis = {"melhor_alg": best_of_phase, "melhor_dist": 0, "total_exec": NUM_EXECUCOES*len(INSTANCIAS)*len(EXPERIMENTOS_DA_VEZ)}
    visualizacao.gerar_relatorio_final(f"{PASTA_FASE}/index.html", sumario_global_fase, kpis, f"Fase concluída: {TITULO_FASE}")
    print(f"✅ FASE CONCLUÍDA: Relatório em {PASTA_FASE}/index.html")

print("\n🎉 AUTOMAÇÃO TOTAL FINALIZADA COM SUCESSO! Pode abrir os relatórios nas pastas numeradas.")