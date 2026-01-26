# Comparação de Algoritmos Metaheurísticos para o Problema do Caixeiro Viajante (TSP)

**Autores:** Guilherme Bonifácio Feitosa e Jonathan Julião  
**Disciplina:** Sistemas Inteligentes  
**Professora:** Alessandra Maranhão  

## Descrição Geral do Projeto

Este projeto implementa e compara três metaheurísticas aplicadas ao Problema do Caixeiro Viajante (TSP) utilizando instâncias da TSPLIB. O objetivo é analisar o desempenho de cada técnica em termos de qualidade da solução, estabilidade, tempo de execução e convergência.

O diferencial deste projeto é a execução em **5 Fases de Teste**, garantindo que os parâmetros (população, seleção, crossover, mutação) sejam escolhidos cientificamente antes do comparativo final.

### As metaheurísticas implementadas são:
1.  **Algoritmo Genético (AG):** Com operadores calibrados (OX, PMX, Swap, Inversion).
2.  **Otimização por Colônia de Formigas (ACO):** Baseado em feromônio e visibilidade.
3.  **Recozimento Simulado (SA):** Utilizado como *baseline* comparativo.

A execução é totalmente automatizada e produz relatórios estatísticos (HTML) e gráficos detalhados.

---

## Funcionalidades Implementadas

O sistema realiza automaticamente:
* **Calibração Automática:** Executa testes preliminares para definir os melhores operadores do AG.
* **Execução das instâncias:** `st70` (Pequena), `eil101` (Média) e `ch130` (Grande).
* **Robustez Estatística:** 30 execuções independentes por algoritmo/cenário.
* **Análise Completa:**
    * Melhor Solução, Média e Desvio Padrão.
    * Teste de Hipótese T-Student (para validar vitórias estatísticas).
    * Tempo de execução computacional.

###  Saídas Geradas
* **Gráficos:** Boxplot (Global e por Instância), Curvas de Convergência e Mapa da Melhor Rota.
* **Relatórios:** Dashboard HTML completo com *Dark Mode* e tabelas de resultados.
* **Arquivos:** CSVs de resumo e logs detalhados.

---

##  Estrutura do Projeto

```text
/
│   main.py                  # Script principal (Orquestrador das 5 Fases)
│   algoritmo_genetico.py    # Implementação da classe AG
│   colonia_formigas.py      # Implementação da classe ACO
│   recozimento_simulado.py  # Implementação da classe SA
│   parser_tsplib.py         # Leitor de instâncias .tsp
│   utils.py                 # Funções auxiliares (distância, rotas)
│   visualizacao.py          # Gerador de gráficos e HTML
│
├── data/                    # Instâncias do TSPLIB
│   ├── st70.tsp
│   ├── eil101.tsp
│   └── ch130.tsp
│
└── resultados/              # Gerado automaticamente
    ├── 1_calibracao_parametros/
    ├── 2_analise_selecao/
    ├── 3_analise_crossover/
    ├── 4_analise_mutacao/
    └── 5_comparativo_final/     # <--- Onde estão os resultados principais
        ├── index.html           # Dashboard Global
        ├── boxplot_global.png
        ├── st70/
        │   ├── relatorio_st70.html
        │   ├── melhor_rota_st70.png
        │   └── ...
        ├── eil101/
        └── ch130/
Como Executar
Pré-requisitos
Python 3.8 ou superior

Bibliotecas: numpy, scipy, matplotlib

Para instalar as dependências:

Bash
pip install numpy scipy matplotlib
Execução
Basta rodar o script principal. Ele executará as 5 fases sequencialmente (pode levar algumas horas devido às 30 execuções):

Bash
python main.py
Ao final, abra o arquivo resultados/5_comparativo_final/index.html para ver o dashboard completo.

Referências Acadêmicas
Dorigo, M. & Gambardella, L. M. Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem.

Kirkpatrick, S. et al. Optimization by Simulated Annealing.

Holland, J. Adaptation in Natural and Artificial Systems.

TSPLIB. A Library of Sample Instances for the TSP.

📄 Licença
Este projeto foi desenvolvido para fins acadêmicos na disciplina de Sistemas Inteligentes.
