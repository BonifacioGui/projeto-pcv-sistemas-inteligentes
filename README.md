# Comparação de Algoritmos Metaheurísticos para o Problema do Caixeiro Viajante (TSP)

**Autores:** Guilherme Bonifácio Feitosa e Jonathan Julião  
**Disciplina:** Sistemas Inteligentes  
**Professora:** Alessandra Maranhão  

## 📝 Descrição Geral do Projeto

Este projeto implementa e compara três metaheurísticas aplicadas ao Problema do Caixeiro Viajante (TSP) utilizando instâncias da TSPLIB. O objetivo é analisar o desempenho de cada técnica em termos de qualidade da solução, estabilidade, tempo de execução e convergência.

O grande diferencial deste trabalho é a **metodologia rigorosa de calibração**. O sistema executa automaticamente **9 baterias de testes (Fases 1 a 7 + sub-análises)** para ajustar cientificamente os parâmetros do Algoritmo Genético e da Colônia de Formigas antes do comparativo final.

### As metaheurísticas implementadas são:
1.  **Algoritmo Genético (AG):** Evolutivo, com calibração de população, seleção, crossover, mutação e elitismo.
2.  **Otimização por Colônia de Formigas (ACO):** Baseado em enxame, com calibração de evaporação e número de agentes.
3.  **Recozimento Simulado (SA):** Baseado em termodinâmica, utilizado como *baseline* comparativo com inicialização aleatória justa.

---

## ⚙️ Funcionalidades e Fases de Execução

O sistema (`main.py`) executa sequencialmente as seguintes análises:

* **Fase 1:** Calibração de População (50 vs 100).
* **Fase 2:** Comparativo de Seleção (Torneio vs Roleta).
* **Fase 3:** Comparativo de Crossover (OX vs PMX).
* **Fase 3b:** Ajuste Fino de Taxa de Mutação.
* **Fase 4:** Comparativo de Mutação (Swap vs Inversion).
* **Fase 4b:** Impacto do Elitismo na convergência.
* **Fase 6:** Calibração do ACO - Taxa de Evaporação (Rho).
* **Fase 7:** Calibração do ACO - Tamanho do Enxame.
* **Fase 5 (Final):** O grande comparativo entre os campeões (AG Otimizado vs ACO Otimizado vs SA).

### 📊 Saídas Geradas
* **Estatística:** Teste T-Student (p-value) para validar diferenças significativas.
* **Visualização:** Boxplots globais, Curvas de Convergência e Mapas das melhores rotas.
* **Relatórios:** Páginas HTML automáticas com *Dark Mode* para cada instância e fase.

---

## 📂 Estrutura do Projeto

```text
/
│   main.py                  # Orquestrador das 9 baterias de testes
│   algoritmo_genetico.py    # Implementação da classe AG
│   colonia_formigas.py      # Implementação da classe ACO
│   recozimento_simulado.py  # Implementação da classe SA
│   visualizacao.py          # Gerador de gráficos e relatórios HTML
│   utils.py                 # Funções auxiliares e métricas
│
├── data/                    # Instâncias do TSPLIB (st70, eil101, ch130)
│
└── resultados/              # Gerado automaticamente com todas as fases
    ├── 1_calibracao_parametros/
    ├── 2_analise_selecao/
    ├── 3_analise_crossover/
    ├── 3b_analise_taxa_mutacao/
    ├── 4_analise_mutacao/
    ├── 4b_analise_elitismo/
    ├── 6_calibracao_aco_rho/       # Calibração específica do ACO
    ├── 7_calibracao_aco_formigas/  # Calibração específica do ACO
    └── 5_comparativo_final/        # RESULTADOS FINAIS
        ├── index.html              # Dashboard Global
        ├── boxplot_global.png
        ├── st70/
        │   ├── relatorio_st70.html
        │   ├── melhor_rota_st70.png
        │   └── ...
        ├── eil101/
        └── ch130/
```
#🚀 Como Executar
**Pré-requisitos**
Python 3.8 ou superior

**Bibliotecas:** numpy, scipy, matplotlib

**Para instalar as dependências:**

Bash
pip install numpy scipy matplotlib
Execução
**Basta rodar o script principal. Ele executará as 9 fases sequencialmente (pode levar algumas horas devido às 30 execuções):**

Bash
python main.py
Ao final, abra o arquivo resultados/5_comparativo_final/index.html para ver o dashboard completo.

#📚 Referências Acadêmicas**
Dorigo, M. & Gambardella, L. M. Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem.

Kirkpatrick, S. et al. Optimization by Simulated Annealing.

Holland, J. Adaptation in Natural and Artificial Systems.

TSPLIB. A Library of Sample Instances for the TSP.

#📄 Licença
Este projeto foi desenvolvido para fins acadêmicos na disciplina de Sistemas Inteligentes.
