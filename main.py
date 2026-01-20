"""
main.py - Versão Final para Entrega (Com Análise de Tempo/Escalabilidade)
Orquestrador de Experimentos com Geração de Dashboard Profissional e Relatórios Detalhados.
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

NUM_EXECUCOES     = 2      # Estatística robusta exige 30
NUM_GERACOES      = 10     # Para dar tempo de convergir(500)
TAMANHO_POPULACAO = 10      # Tamanho padrão da literatura (50)

INSTANCIAS = [
    "data/st70.tsp",
    "data/eil101.tsp",
    "data/ch130.tsp"
]

EXECUTION_TIMESTAMP = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# ================================================================
# 2. BANCO DE EXPERIMENTOS
# ================================================================
TODOS_EXPERIMENTOS = {
    "geral": [
        {"nome": "AG_Padrao", "algoritmo": "AG", "params": {"taxa_mutacao": 0.01, "metodo_crossover": "ox"}},
        {"nome": "AG_AltaMutacao", "algoritmo": "AG", "params": {"taxa_mutacao": 0.10, "metodo_crossover": "ox"}},
        {"nome": "ACO_Padrao", "algoritmo": "ACO", "params": {"num_formigas": TAMANHO_POPULACAO, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},
        {"nome": "SA_Padrao", "algoritmo": "SA", "params": {"temp_inicial": 1000, "cooling_rate": 0.995}}
    ],
    "selecao": [
        {"nome": "AG_Torneio", "algoritmo": "AG", "params": {"metodo_selecao": "torneio"}},
        {"nome": "AG_Roleta", "algoritmo": "AG", "params": {"metodo_selecao": "roleta"}},
    ],
    "mutacao": [
        {"nome": "AG_Troca", "algoritmo": "AG", "params": {"metodo_mutacao": "swap"}},
        {"nome": "AG_Inversao", "algoritmo": "AG", "params": {"metodo_mutacao": "inversion"}},
    ],
    "elitismo": [
        {"nome": "AG_Elitismo_0", "algoritmo": "AG", "params": {"taxa_elitismo": 0.0}},
        {"nome": "AG_Elitismo_10", "algoritmo": "AG", "params": {"taxa_elitismo": 0.1}},
    ],
    "crossover": [
        {"nome": "AG_OX", "algoritmo": "AG", "params": {"metodo_crossover": "ox"}},
        {"nome": "AG_PMX", "algoritmo": "AG", "params": {"metodo_crossover": "pmx"}},
    ]
}

# SELETOR
CHAVE_ESCOLHIDA = "geral"
EXPERIMENTO_ATUAL = TODOS_EXPERIMENTOS.get(CHAVE_ESCOLHIDA, TODOS_EXPERIMENTOS["geral"])

# ================================================================
# 3. PREPARAÇÃO
# ================================================================
def garantir_pasta(p):
    if not os.path.exists(p): os.makedirs(p)
garantir_pasta("resultados")

# ================================================================
# 4. EXECUÇÃO PRINCIPAL
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

    resultados_boxplot_instancia = {}
    resultados_tempos_instancia = {}  # <--- [NOVO] Dicionário para guardar tempos
    
    melhor_global_dist = float("inf")
    melhor_global_alg = "N/A"
    melhor_global_rota_obj = None 
    ttest_html_fragments = "" 

    for exp in EXPERIMENTO_ATUAL:
        nome_exp = exp["nome"]
        alg = exp["algoritmo"]
        params = exp["params"]
        
        print(f" -> Executando {nome_exp} ({alg}) ... ", end="")
        
        dists = []
        tempos = []
        hist_melhores = []
        hist_medias = []

        for _ in range(NUM_EXECUCOES):
            inicio = time.perf_counter()
            
            if alg == "AG":
                modelo = AlgoritmoGenetico(cidades, dist_matrix=dist_matrix, **params, num_geracoes=NUM_GERACOES, tamanho_populacao=TAMANHO_POPULACAO)
                res = modelo.executar()
                dist_atual = res.distancia
                rota_atual = res.rota
                hist_melhores.append(modelo.historico_melhores)
                hist_medias.append(modelo.historico_medias)
            elif alg == "ACO":
                modelo = ACO(cidades, **params, num_iteracoes=NUM_GERACOES)
                rota_atual, dist_atual = modelo.executar()
                hist_melhores.append(modelo.historico_melhores)
            elif alg == "SA":
                modelo = RecozimentoSimulado(cidades, dist_matrix=dist_matrix, **params, max_iterations=NUM_GERACOES*TAMANHO_POPULACAO)
                rota_atual, dist_atual = modelo.executar()
                hist_melhores.append(modelo.historico_melhores)

            fim = time.perf_counter()
            dists.append(dist_atual)
            tempos.append(fim - inicio)

            # Verifica se essa execução específica foi a melhor de todas da instância
            if dist_atual < melhor_global_dist:
                melhor_global_dist = dist_atual
                melhor_global_alg = nome_exp
                melhor_global_rota_obj = rota_atual
        
        # --- CÁLCULO DE TEMPO MÉDIO (PARA ANÁLISE DE ESCALABILIDADE) ---
        tempo_medio = np.mean(tempos)
        resultados_tempos_instancia[nome_exp] = tempo_medio
        
        print(f"OK (Média Dist: {np.mean(dists):.2f} | Tempo: {tempo_medio:.4f}s)")
        
        # Salva dados JSON
        with open(f"{pasta_saida}/{nome_exp}.json", "w") as jf:
            json.dump({"distancias": dists, "tempos": tempos}, jf, indent=2)

        resultados_boxplot_instancia[nome_exp] = dists
        media_pop = hist_medias if alg == "AG" else None
        
        # Gera Gráfico de Convergência
        visualizacao.plotar_convergencia(
            hist_melhores, 
            media_pop, 
            f"{pasta_saida}/convergencia_{nome_exp}.png"
        )

    # --- FIM DO LOOP DE ALGORITMOS PARA ESTA INSTÂNCIA ---

    # 1. Gera Boxplot da Instância
    visualizacao.plotar_boxplot_comparativo(
        resultados_boxplot_instancia, 
        f"{pasta_saida}/boxplot_{nome_instancia}.png"
    )
    
    # 2. Gera Gráfico da Melhor Rota
    if melhor_global_rota_obj:
        visualizacao.plotar_rota(
            melhor_global_rota_obj, 
            cidades, 
            f"{pasta_saida}/melhor_rota_{nome_instancia}.png"
        )

    # 3. Gera CSV de Resumo (AGORA COM TEMPO)
    with open(f"{pasta_saida}/resumo_{nome_instancia}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Cabeçalho atualizado
        writer.writerow(["Algoritmo", "Media Dist", "Desvio", "Min", "Max", "Tempo Medio (s)"])
        
        for alg_nome, dados in resultados_boxplot_instancia.items():
            t_med = resultados_tempos_instancia[alg_nome] # Recupera o tempo salvo
            writer.writerow([
                alg_nome, 
                f"{np.mean(dados):.2f}", 
                f"{np.std(dados):.2f}", 
                f"{np.min(dados):.2f}", 
                f"{np.max(dados):.2f}",
                f"{t_med:.4f}" # Escreve o tempo
            ])

    # 4. Calcula Testes T-Student
    chaves = list(resultados_boxplot_instancia.keys())
    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            alg_a = chaves[i]
            alg_b = chaves[j]
            try:
                s, p = stats.ttest_ind(resultados_boxplot_instancia[alg_a], resultados_boxplot_instancia[alg_b], equal_var=False)
                cor = "var(--success)" if p < 0.05 else "var(--text-muted)"
                sig = "SIM" if p < 0.05 else "NÃO"
                ttest_html_fragments += f"<p><b>{alg_a} vs {alg_b}</b>: p-value={p:.4e} <span style='color:{cor}; font-weight:bold'>({sig})</span></p>"
            except:
                ttest_html_fragments += f"<p><b>{alg_a} vs {alg_b}</b>: Dados idênticos (sem variância).</p>"
    # 4.1 Gera Gráfico de Tempo (Isso já estava certo, mantenha)
    visualizacao.plotar_comparativo_tempo(
        resultados_tempos_instancia, 
        f"{pasta_saida}/tempos_{nome_instancia}.png"
    )

    # 5. Gera Relatório HTML Individual (CORRIGIDO: Agora mostra o gráfico de tempo)
    with open(f"{pasta_saida}/relatorio_{nome_instancia}.html", "w", encoding="utf-8") as f:
        f.write(f"""
        <html><head><link rel='stylesheet' href='../style.css'></head><body>
        <div class='container'>
            <header><h1>Detalhes: {nome_instancia.upper()}</h1><div class='timestamp'>Bateria: {CHAVE_ESCOLHIDA}</div></header>
            
            <h2 class='section-title'>1. Melhor Resultado</h2>
            <div class='card'>
                <p>Melhor Algoritmo: <strong>{melhor_global_alg}</strong></p>
                <p>Distância: <strong>{melhor_global_dist:.2f}</strong></p>
                <a href='melhor_rota_{nome_instancia}.png' target='_blank'>
                    <img src='melhor_rota_{nome_instancia}.png' title='Clique para ampliar'>
                </a>
            </div>

            <h2 class='section-title'>2. Desempenho Computacional (Tempo)</h2>
            <div class='card'>
                <p>Comparativo de velocidade média para convergir:</p>
                <a href='tempos_{nome_instancia}.png' target='_blank'>
                    <img src='tempos_{nome_instancia}.png' style='max-width: 600px;' title='Clique para ampliar'>
                </a>
            </div>

            <h2 class='section-title'>3. Comparativo de Qualidade (Boxplot)</h2>
            <div class='card'>
                <a href='boxplot_{nome_instancia}.png' target='_blank'>
                    <img src='boxplot_{nome_instancia}.png' title='Clique para ampliar'>
                </a>
            </div>

            <h2 class='section-title'>4. Testes Estatísticos (Significância)</h2>
            <div class='card'>{ttest_html_fragments if ttest_html_fragments else "Nenhum teste realizado."}</div>

            <h2 class='section-title'>5. Convergência por Algoritmo</h2>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;'>
        """)
        
        for exp in EXPERIMENTO_ATUAL:
            f.write(f"""
            <div class='card'>
                <h3>{exp['nome']}</h3>
                <a href='convergencia_{exp['nome']}.png' target='_blank'>
                    <img src='convergencia_{exp['nome']}.png' style='width:100%; border:1px solid #334155;' title='Clique para ampliar'>
                </a>
            </div>
            """)
        
        f.write("</div></div></body></html>")

    # Atualiza Globais para o Dashboard Final
    sumario_global[nome_instancia] = {"melhor_dist": melhor_global_dist, "melhor_alg": melhor_global_alg}
    for k, v in resultados_boxplot_instancia.items():
        resultados_globais[f"{nome_instancia}_{k}"] = v

# ================================================================
# 5. DASHBOARD GLOBAL
# ================================================================

# 1. Gera gráfico global
visualizacao.plotar_boxplot_comparativo(resultados_globais, "resultados/boxplot_global.png")

# 2. Prepara os dados para os KPIs
if sumario_global:
    lista_vencedores = [i['melhor_alg'] for i in sumario_global.values()]
    melhor_alg = max(set(lista_vencedores), key=lista_vencedores.count)
    melhor_dist = min([i['melhor_dist'] for i in sumario_global.values()])
else:
    melhor_alg, melhor_dist = "N/A", 0.0

kpis = {
    "melhor_alg": melhor_alg,
    "melhor_dist": melhor_dist,
    "total_exec": NUM_EXECUCOES * len(EXPERIMENTO_ATUAL) * len(INSTANCIAS)
}

# 3. LÓGICA DINÂMICA DE CONCLUSÃO
if "SA" in melhor_alg:
    analise_vencedor = """
    O <strong>Recozimento Simulado (SA)</strong> demonstrou ser a estratégia mais robusta nesta bateria. 
    Sua capacidade de escapar de mínimos locais (aceitando pioras temporárias) foi decisiva nas instâncias maiores.
    """
elif "ACO" in melhor_alg:
    analise_vencedor = """
    A <strong>Colônia de Formigas (ACO)</strong> obteve o melhor desempenho geral. 
    A abordagem construtiva guiada por feromônios convergiu rapidamente para soluções de alta qualidade.
    """
elif "AG" in melhor_alg:
    analise_vencedor = """
    O <strong>Algoritmo Genético (AG)</strong> superou as outras abordagens. 
    Isso valida a eficácia dos operadores de crossover e a manutenção da diversidade genética.
    """
else:
    analise_vencedor = """
    Os resultados mostram um <strong>equilíbrio competitivo</strong>. 
    Nenhum algoritmo dominou completamente todas as instâncias.
    """

minha_conclusao = f"""
<strong>Resultado da Bateria:</strong><br>
O algoritmo com maior número de vitórias foi o <span style="color:var(--accent); font-weight:bold;">{melhor_alg}</span>, 
atingindo a melhor distância absoluta de <b>{melhor_dist:.2f}</b>.<br><br>
{analise_vencedor}<br>
<small style="color:var(--text-muted);">Relatório gerado automaticamente em {EXECUTION_TIMESTAMP}</small>
"""

# 4. Chama a função geradora no visualizacao.py
visualizacao.gerar_relatorio_final("resultados/index.html", sumario_global, kpis, minha_conclusao)

print("\n=== Dashboard Gerado com Sucesso! ===")