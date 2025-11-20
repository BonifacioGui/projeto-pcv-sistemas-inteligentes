Comparação de Algoritmos Metaheurísticos para o Problema do Caixeiro Viajante (TSP)

Autores: Guilherme Bonifácio Feitosa e Jonanthan Julião
Data da última execução: gerada automaticamente pelo script
Disciplina: Projeto Algorítmico / Pesquisa Operacional
Instituição: (inserir, se desejado)

1. Descrição Geral do Projeto

Este projeto implementa e compara três metaheurísticas aplicadas ao Problema do Caixeiro Viajante (TSP) utilizando instâncias da TSPLIB. O objetivo é analisar o desempenho de cada técnica em termos de qualidade da solução, estabilidade, tempo de execução e convergência.

As metaheurísticas implementadas são:

Algoritmo Genético (AG)

Otimização por Colônia de Formigas (ACO)

Recozimento Simulado (SA)

A execução é totalmente automatizada e produz diversas saídas estatísticas e visuais para análise detalhada.

2. Funcionalidades Implementadas

O sistema realiza automaticamente:

Execução das instâncias st70, eil101 e ch130

Trinta execuções independentes por algoritmo

Medição de:

melhores soluções

médias

desvios padrão

tempos de execução

Geração de gráficos:

Boxplot por instância

Boxplot global

Convergência (melhor e média, no caso dos AGs)

Visualização da melhor rota

Testes estatísticos T-Student para comparação entre métodos

Exportação de arquivos:

CSV de resumo por instância

JSON contendo distâncias e tempos

CSV da curva média de convergência (AG)

TXT da melhor rota encontrada

Construção automática de:

Relatório HTML por instância (dark mode)

Dashboard global (resultados/index.html)

Arquivo CSS centralizado (style.css)

Impressão de um sumário final no console

3. Estrutura do Projeto
/
│ main.py
│ algoritmo_genetico.py
│ colonia_formigas.py
│ recozimento_simulado.py
│ parser_tsplib.py
│ utils.py
│ visualizacao.py
│
├── data/
│   ├── st70.tsp
│   ├── eil101.tsp
│   └── ch130.tsp
│
└── resultados/
    ├── style.css
    ├── index.html
    ├── st70/
    │   ├── relatorio_st70.html
    │   ├── boxplot_st70.png
    │   ├── melhor_rota_st70.png
    │   ├── resumo_st70.csv
    │   └── ...
    ├── eil101/
    └── ch130/


A pasta resultados/ será criada automaticamente caso não exista.

4. Requisitos

Python 3.8 ou superior

NumPy

SciPy

Matplotlib

Para instalar todas as dependências:

pip install numpy scipy matplotlib

5. Como Executar

Basta executar o script principal:

python main.py


A execução completa gera todos os resultados, gráficos e relatórios automaticamente.

6. Referências Acadêmicas

Dorigo, M. & Gambardella, L. M. Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem.

Kirkpatrick, S. et al. Optimization by Simulated Annealing.

Holland, J. Adaptation in Natural and Artificial Systems.

TSPLIB. A Library of Sample Instances for the TSP.

7. Licença

Este projeto pode ser utilizado para fins acadêmicos e educacionais.