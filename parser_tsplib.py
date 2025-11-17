# Dentro de parser_tsplib.py

def carregar_cidades(filepath):
    """
    Lê um arquivo .tsp e retorna uma lista de tuplas (x, y) 
    representando as coordenadas das cidades.
    """
    # 1. "A Sacola Vazia"
    cidades = []
    
    # 2. "O Interruptor"
    lendo_coordenadas = False
    
    # 3. Abrindo o arquivo
    with open(filepath, 'r') as f:
        
        # 4. Lendo linha por linha
        for linha in f:
            
            # 5. Limpeza de dados
            linha = linha.strip()
            
            # 6. O "Interruptor" está ligado?
            if lendo_coordenadas:
                
                # 7. Checando se a seção de dados acabou
                if linha == "EOF":
                    lendo_coordenadas = False
                    break # Para o loop
                
                # 8. Quebrando a linha em partes
                partes = linha.split()
                
                # 9. Verificação de segurança
                if len(partes) == 3:
                    
                    # 10. Extraindo e convertendo os dados
                    x = float(partes[1])
                    y = float(partes[2])
                    
                    # 11. Guardando na "sacola"
                    cidades.append((x, y))
            
            # 12. Checando se a seção de dados COMEÇOU
            if linha == "NODE_COORD_SECTION":
                lendo_coordenadas = True
                
    # 13. Devolvendo o resultado
    return cidades