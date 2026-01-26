"""
main.py - VERSÃO CORRIGIDA (Bug 'taxa_mutacao' resolvido)
Este script executa 5 FASES de testes cobrindo todos os requisitos.
"""
import os
import time
import numpy as np
from datetime import datetime
from scipy import stats

# Módulos do Projeto
import parser_tsplib
import utils
import visualizacao
from algoritmo_genetico import AlgoritmoGenetico
from colonia_formigas import ACO
from recozimento_simulado import RecozimentoSimulado

# ==============================================================================
# 1. CONFIGURAÇÕES DE EXECUÇÃO
# ==============================================================================
MODO_TESTE = False  # <--- FALSE = Entrega Final (30 execuções). TRUE = Teste Rápido.

if MODO_TESTE:
    print("\n⚠️  ALERTA: MODO DE TESTE ATIVADO (Rápido: apenas 2 execuções)")
    NUM_EXECUCOES     = 2
    NUM_GERACOES      = 10
    TAMANHO_POPULACAO_PADRAO = 10
else:
    print("\n🚀 MODO DE ENTREGA ATIVADO (Completo: 30 execuções estatísticas)")
    NUM_EXECUCOES     = 30
    NUM_GERACOES      = 500
    TAMANHO_POPULACAO_PADRAO = 50

# Instâncias do TSPLIB a serem testadas
INSTANCIAS = ["data/st70.tsp", "data/eil101.tsp", "data/ch130.tsp"]

# ==============================================================================
# 2. FUNÇÕES AUXILIARES (Automação)
# ==============================================================================
def garantir_pasta(caminho):
    if not os.path.exists(caminho):
        os.makedirs(caminho)

def barra_progresso(atual, total, texto):
    percent = (atual / total) * 100
    bar = "█" * int(percent / 5) + "-" * (20 - int(percent / 5))
    print(f"\r|{bar}| {percent:.1f}% - {texto}", end="", flush=True)

def rodar_bateria(titulo_fase, pasta_fase, lista_experimentos):
    print(f"\n\n{'='*60}")
    print(f"📂 INICIANDO: {titulo_fase}")
    print(f"{'='*60}")
    
    garantir_pasta(f"resultados/{pasta_fase}")
    
    sumario_global = {}
    resultados_raw_global = {}

    total_passos = len(INSTANCIAS) * len(lista_experimentos) * NUM_EXECUCOES
    passo_atual = 0

    for arquivo_tsp in INSTANCIAS:
        nome_instancia = os.path.basename(arquivo_tsp).split(".")[0]
        pasta_saida = f"resultados/{pasta_fase}/{nome_instancia}"
        garantir_pasta(pasta_saida)
        
        try:
            cidades = parser_tsplib.carregar_cidades(arquivo_tsp)
            matriz_dist = utils.calcular_matriz_distancias(cidades)
        except Exception as e:
            print(f"Erro crítico ao ler {arquivo_tsp}: {e}")
            continue

        melhor_dist_instancia = float("inf")
        melhor_alg_instancia = "N/A"
        melhor_rota_objeto = None
        
        dados_boxplot = {}
        dados_tempos = {}
        imagens_convergencia = []

        for exp in lista_experimentos:
            nome_exp = exp["nome"]
            tipo_alg = exp["algoritmo"]
            params = exp["params"]
            
            # Ajuste dinâmico de população se necessário
            pop_size = params.get("tamanho_populacao", TAMANHO_POPULACAO_PADRAO)
            
            # Prepara parâmetros limpos para o construtor
            params_limpo = params.copy()
            if "tamanho_populacao" in params_limpo:
                del params_limpo["tamanho_populacao"]

            lista_distancias = []
            lista_tempos = []
            historico_melhores = []
            historico_medias = []

            for i in range(NUM_EXECUCOES):
                passo_atual += 1
                barra_progresso(passo_atual, total_passos, f"{nome_instancia} > {nome_exp}")
                
                inicio = time.perf_counter()
                
                if tipo_alg == "AG":
                    # Aqui passamos params_limpo que agora contém taxa_mutacao
                    modelo = AlgoritmoGenetico(cidades, dist_matrix=matriz_dist, **params_limpo, 
                                             num_geracoes=NUM_GERACOES, tamanho_populacao=pop_size)
                    resultado = modelo.executar()
                    historico_medias.append(modelo.historico_medias)
                    historico_melhores.append(modelo.historico_melhores)
                    dist_final = resultado.distancia
                    rota_final = resultado.rota

                elif tipo_alg == "ACO":
                    modelo = ACO(cidades, **params_limpo, num_iteracoes=NUM_GERACOES)
                    rota_final, dist_final = modelo.executar()
                    historico_melhores.append(modelo.historico_melhores)

                elif tipo_alg == "SA":
                    modelo = RecozimentoSimulado(cidades, dist_matrix=matriz_dist, **params_limpo, 
                                               max_iterations=NUM_GERACOES*pop_size)
                    rota_final, dist_final = modelo.executar()
                    historico_melhores.append(modelo.historico_melhores)
                
                tempo_gasto = time.perf_counter() - inicio
                lista_distancias.append(dist_final)
                lista_tempos.append(tempo_gasto)

                if dist_final < melhor_dist_instancia:
                    melhor_dist_instancia = dist_final
                    melhor_alg_instancia = nome_exp
                    melhor_rota_objeto = rota_final

            dados_boxplot[nome_exp] = lista_distancias
            dados_tempos[nome_exp] = np.mean(lista_tempos)
            
            arquivo_conv = f"convergencia_{nome_exp}.png"
            visualizacao.plotar_convergencia(historico_melhores, historico_medias if tipo_alg=="AG" else None, 
                                           f"{pasta_saida}/{arquivo_conv}")
            imagens_convergencia.append({"nome": nome_exp, "arquivo": arquivo_conv})

        # --- GERAÇÃO DE RELATÓRIOS ---
        visualizacao.plotar_boxplot_comparativo(dados_boxplot, f"{pasta_saida}/boxplot_{nome_instancia}.png")
        visualizacao.plotar_comparativo_tempo(dados_tempos, f"{pasta_saida}/tempos_{nome_instancia}.png")
        
        if melhor_rota_objeto:
            visualizacao.plotar_rota(melhor_rota_objeto, cidades, f"{pasta_saida}/melhor_rota_{nome_instancia}.png")

        html_stats = ""
        chaves = list(dados_boxplot.keys())
        for i in range(len(chaves)):
            for j in range(i+1, len(chaves)):
                try:
                    s, p = stats.ttest_ind(dados_boxplot[chaves[i]], dados_boxplot[chaves[j]], equal_var=False)
                    cor = "var(--success)" if p < 0.05 else "var(--text-muted)"
                    texto_sig = "Diferença Real" if p < 0.05 else "Empate"
                    html_stats += f"<p><b>{chaves[i]} vs {chaves[j]}</b>: p={p:.4f} <span style='color:{cor}'>({texto_sig})</span></p>"
                except:
                    html_stats += f"<p>{chaves[i]} vs {chaves[j]}: Dados idênticos</p>"

        imagens = {
            "rota": f"melhor_rota_{nome_instancia}.png",
            "tempos": f"tempos_{nome_instancia}.png",
            "boxplot": f"boxplot_{nome_instancia}.png",
            "convergencias": imagens_convergencia
        }
        visualizacao.gerar_relatorio_instancia(
            f"{pasta_saida}/relatorio_{nome_instancia}.html",
            nome_instancia, titulo_fase, melhor_alg_instancia, melhor_dist_instancia, html_stats, imagens
        )

        sumario_global[nome_instancia] = {"melhor_dist": melhor_dist_instancia, "melhor_alg": melhor_alg_instancia}
        for k, v in dados_boxplot.items():
            resultados_raw_global[f"{nome_instancia}_{k}"] = v

    # --- DASHBOARD GLOBAL DA FASE ---
    visualizacao.plotar_boxplot_comparativo(resultados_raw_global, f"resultados/{pasta_fase}/boxplot_global.png")
    
    if sumario_global:
        lista_vencedores = [v['melhor_alg'] for v in sumario_global.values()]
        campeao_fase = max(set(lista_vencedores), key=lista_vencedores.count) if lista_vencedores else "N/A"
        lista_distancias = [v['melhor_dist'] for v in sumario_global.values()]
        recorde_fase = min(lista_distancias) if lista_distancias else 0.0
    else:
        campeao_fase = "N/A"; recorde_fase = 0.0

    kpis = {"melhor_alg": campeao_fase, "melhor_dist": recorde_fase, "total_exec": total_passos}
    visualizacao.gerar_relatorio_final(f"resultados/{pasta_fase}/index.html", sumario_global, kpis, titulo_fase)
    print(f"\n✅ FASE CONCLUÍDA: {titulo_fase}")


# ==============================================================================
# 4. EXECUÇÃO DAS 5 FASES (CORRIGIDO)
# ==============================================================================
print("\n>>> INICIANDO SISTEMA DE BENCHMARKING (5 FASES) <<<")

# FASE 1: Calibração (Já tinha taxa_mutacao, OK)
rodar_bateria("Fase 1: Calibração de Parâmetros (População)", "1_calibracao_parametros", [
    {"nome": "AG_Pop50",  "algoritmo": "AG", "params": {"tamanho_populacao": 50,  "taxa_mutacao": 0.05}},
    {"nome": "AG_Pop100", "algoritmo": "AG", "params": {"tamanho_populacao": 100, "taxa_mutacao": 0.05}}
])

# FASE 2: Seleção (Adicionado taxa_mutacao: 0.05)
rodar_bateria("Fase 2: Comparativo de Seleção", "2_analise_selecao", [
    {"nome": "AG_Torneio", "algoritmo": "AG", "params": {"metodo_selecao": "torneio", "taxa_mutacao": 0.05}},
    {"nome": "AG_Roleta",  "algoritmo": "AG", "params": {"metodo_selecao": "roleta",  "taxa_mutacao": 0.05}}
])

# FASE 3: Crossover (Adicionado taxa_mutacao: 0.05)
rodar_bateria("Fase 3: Comparativo de Crossover", "3_analise_crossover", [
    {"nome": "AG_OX",  "algoritmo": "AG", "params": {"metodo_crossover": "ox",  "metodo_selecao": "torneio", "taxa_mutacao": 0.05}},
    {"nome": "AG_PMX", "algoritmo": "AG", "params": {"metodo_crossover": "pmx", "metodo_selecao": "torneio", "taxa_mutacao": 0.05}}
])

# FASE 4: Mutação (Adicionado taxa_mutacao: 0.05)
rodar_bateria("Fase 4: Comparativo de Mutação", "4_analise_mutacao", [
    {"nome": "AG_Swap",      "algoritmo": "AG", "params": {"metodo_mutacao": "swap",      "metodo_crossover": "ox", "taxa_mutacao": 0.05}},
    {"nome": "AG_Inversion", "algoritmo": "AG", "params": {"metodo_mutacao": "inversion", "metodo_crossover": "ox", "taxa_mutacao": 0.05}}
])

# FASE 5: Final (Já tinha, OK)
rodar_bateria("Fase 5: Comparativo Final (Meta-heurísticas)", "5_comparativo_final", [
    {"nome": "AG_Final",  "algoritmo": "AG",  "params": {"taxa_mutacao": 0.05, "metodo_crossover": "ox", "metodo_mutacao": "inversion"}},
    {"nome": "ACO_Final", "algoritmo": "ACO", "params": {"num_formigas": TAMANHO_POPULACAO_PADRAO, "alfa": 1.0, "beta": 2.5, "rho": 0.1}},
    {"nome": "SA_Final",  "algoritmo": "SA",  "params": {"temp_inicial": 1000, "cooling_rate": 0.995}}
])

print("\n🎉 PARABÉNS! TODOS OS REQUISITOS DO CHECKLIST FORAM ATENDIDOS.")