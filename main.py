# Dentro de main.py

# 1. Importando nosso outro arquivo
import parser_tsplib

# 2. Definindo o "alvo"
caminho_st70 = "data/st70.tsp"

# 3. Executando o parser
minhas_cidades = parser_tsplib.carregar_cidades(caminho_st70)

# 4. Validando os resultados
print(f"Arquivo: {caminho_st70}")
print(f"Total de cidades carregadas: {len(minhas_cidades)}")
print("Coordenadas das 5 primeiras cidades:")
print(minhas_cidades[:5])