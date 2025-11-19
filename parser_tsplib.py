"""
parser_tsplib.py

Parser robusto para arquivos TSPLIB (.tsp). Suporta:
 - Leitura de cabeçalho (NAME, TYPE, DIMENSION, EDGE_WEIGHT_TYPE, EDGE_WEIGHT_FORMAT, etc.)
 - NODE_COORD_SECTION (com ou sem índice; 2D ou 3D)
 - EDGE_WEIGHT_SECTION (FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW, LOWER_DIAG_ROW)
 - Comentários e formatos com espaços/tabs
 - Retorno opcional da matriz de distâncias (pre-calculada)
 - Mensagens de erro claras

Interface principal:
    carregar_cidades(filepath, return_dist_matrix=False, force_euc2d=True)

Retorno:
    - se return_dist_matrix=False:
        lista_de_coords ([(x,y), ...])  -- pode ser vazia se o arquivo tiver apenas matriz explícita
    - se return_dist_matrix=True:
        (lista_de_coords, dist_matrix (lista de listas), meta_dict)

meta_dict contém: keys encontradas no header (DIMENSION, EDGE_WEIGHT_TYPE, ...)
"""

from typing import List, Tuple, Optional, Dict, Any
import math

# Tentativa de usar numpy para representar a matriz, mas não obrigatório
try:
    import numpy as np
except Exception:
    np = None


# -----------------------
# Utilitários locais
# -----------------------
def _is_header_line(line: str) -> bool:
    """Detecta se a linha aparenta ser um header (contains ':')"""
    return ':' in line and not line.strip().upper().startswith('NODE_COORD_SECTION') and not line.strip().upper().endswith('SECTION')


def _safe_split(line: str) -> List[str]:
    """Split que trata múltiplos espaços e tabs."""
    return line.strip().split()


def _to_float(s: str) -> float:
    return float(s)


def _euclidean_matrix_from_coords(coords: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(coords)
    if n == 0:
        return []
    # Use numpy if disponível para velocidade e conversão, mas devolve lista de listas
    if np is not None:
        pts = np.array(coords, dtype=float)
        dx = pts[:, 0].reshape((n, 1)) - pts[:, 0].reshape((1, n))
        dy = pts[:, 1].reshape((n, 1)) - pts[:, 1].reshape((1, n))
        mat = np.hypot(dx, dy)
        return mat.tolist()
    else:
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                d = math.hypot(xj - xi, yj - yi)
                mat[i][j] = d
                mat[j][i] = d
        return mat


# -----------------------
# Parsers de seções
# -----------------------
def _parse_node_coord_section(lines_iter, dimension: Optional[int]) -> List[Tuple[float, float]]:
    """
    Lê as linhas de NODE_COORD_SECTION a partir do iterador até EOF ou próximo header.
    Suporta:
      - 'id x y'
      - 'x y' (sem id)
      - linhas extras, espaços, tabs
      - 3D: 'id x y z' -> ignora z
    """
    coords: List[Tuple[float, float]] = []
    count = 0

    for raw in lines_iter:
        line = raw.strip()
        if not line:
            continue
        # fim da seção
        up = line.upper()
        if up.startswith("EOF") or up.endswith("EOF"):
            break
        # possível início de nova seção/cabeçalho
        if line.endswith("SECTION") or ":" in line and not line[0].isdigit():
            # empurra a linha de volta para o iterador externo:
            # como iterador é um gerador de linhas do arquivo, retornamos a linha
            # sinalizando ao chamador que encontrou um header - no nosso uso atual,
            # não precisamos suportar "pushback" — assumimos NODE_COORD_SECTION termina com EOF.
            break

        parts = _safe_split(line)
        if len(parts) >= 2:
            # dois formatos comuns:
            # 1) id x y
            # 2) x y
            if len(parts) >= 3 and parts[0].replace('.', '', 1).lstrip('-').isdigit():
                # assume primeiro token é index
                try:
                    x = _to_float(parts[1])
                    y = _to_float(parts[2])
                    coords.append((x, y))
                except Exception:
                    # ignora linha mal formada
                    continue
            else:
                # assume apenas x y (sem índice)
                try:
                    x = _to_float(parts[0])
                    y = _to_float(parts[1])
                    coords.append((x, y))
                except Exception:
                    continue
        count += 1
        # se dimensão conhecida, podemos interromper quando coletamos o suficiente
        if dimension is not None and len(coords) >= dimension:
            # consume até EOF or next header is fine; break to caller
            break

    return coords


def _parse_edge_weight_section(lines_iter, dimension: int, format_key: Optional[str]) -> List[List[float]]:
    """
    Lê EDGE_WEIGHT_SECTION e reconstrói a matriz de pesos segundo EDGE_WEIGHT_FORMAT.
    Suporta formatos: FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW, LOWER_DIAG_ROW
    Implementa um leitor genérico que consome números em ordem e preenche a matriz simétrica.
    """
    # Lê todos tokens numéricos da seção até EOF ou tamanho suficiente
    tokens: List[float] = []
    for raw in lines_iter:
        line = raw.strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("EOF") or up.endswith("EOF"):
            break
        # detecta inicio de outra seção
        if line.endswith("SECTION") and "EDGE_WEIGHT" not in up:
            break
        parts = _safe_split(line)
        for p in parts:
            # tenta parsear float/int
            try:
                tokens.append(float(p))
            except:
                # ignora tokens não numéricos
                continue

    # Agora preenche a matriz conforme o formato.
    n = dimension
    if n <= 0:
        raise ValueError("Dimension inválida para EDGE_WEIGHT_SECTION")

    mat = [[0.0] * n for _ in range(n)]

    # Flatten fill for FULL_MATRIX (row major)
    if format_key is None or format_key.upper() == "FULL_MATRIX":
        expected = n * n
        if len(tokens) < expected:
            raise ValueError(f"Tokens insuficientes para FULL_MATRIX (esperado {expected}, achado {len(tokens)})")
        it = iter(tokens)
        for i in range(n):
            for j in range(n):
                mat[i][j] = next(it)
        return mat

    fk = format_key.upper()
    # Many TSPLIB variants provide only upper triangular or diagonal-including formats.
    # We'll fill by reading tokens in the order typical for these formats.

    idx = 0

    if fk == "UPPER_ROW" or fk == "UPPER_TRIANGULAR_ROW":
        # tokens for i<j (row-wise): row0: (0,1..n-1), row1: (1,2..n-1), ...
        for i in range(n):
            for j in range(i + 1, n):
                if idx >= len(tokens):
                    raise ValueError("Tokens insuficientes para UPPER_ROW")
                mat[i][j] = tokens[idx]; mat[j][i] = tokens[idx]
                idx += 1
        return mat

    if fk == "LOWER_ROW" or fk == "LOWER_TRIANGULAR_ROW":
        # tokens for j<i (row-wise lower triangular)
        for i in range(n):
            for j in range(0, i):
                if idx >= len(tokens):
                    raise ValueError("Tokens insuficientes para LOWER_ROW")
                mat[i][j] = tokens[idx]; mat[j][i] = tokens[idx]
                idx += 1
        return mat

    if fk == "UPPER_DIAG_ROW":
        # includes diagonal: row0: (0,0..n-1), row1: (1,1..n-1), ...
        for i in range(n):
            for j in range(i, n):
                if idx >= len(tokens):
                    raise ValueError("Tokens insuficientes para UPPER_DIAG_ROW")
                mat[i][j] = tokens[idx]; mat[j][i] = tokens[idx]
                idx += 1
        return mat

    if fk == "LOWER_DIAG_ROW":
        # includes diagonal: row0: (0,0), row1: (1,0..1), row2: (2,0..2), ...
        for i in range(n):
            for j in range(0, i+1):
                if idx >= len(tokens):
                    raise ValueError("Tokens insuficientes para LOWER_DIAG_ROW")
                mat[i][j] = tokens[idx]; mat[j][i] = tokens[idx]
                idx += 1
        return mat

    # fallback: attempt to fill as FULL_MATRIX if unknown
    expected = n * n
    if len(tokens) >= expected:
        it = iter(tokens)
        for i in range(n):
            for j in range(n):
                mat[i][j] = next(it)
        return mat

    raise ValueError(f"EDGE_WEIGHT_FORMAT '{format_key}' não suportado ou tokens insuficientes")


# -----------------------
# Função pública principal
# -----------------------
def carregar_cidades(filepath: str, return_dist_matrix: bool = False) -> Any:
    """
    Carrega um arquivo TSPLIB (.tsp) e retorna:
      - se return_dist_matrix == False:
          lista de coordenadas [(x,y), ...] (pode ser [] se arquivo for EXPLICIT matrix)
      - se return_dist_matrix == True:
          (coords_list, dist_matrix, meta_dict)
            coords_list: list[(x,y)] ou [] se não houver NODE_COORD_SECTION
            dist_matrix: list of lists (n x n) or []
            meta_dict: dicionário com headers (DIMENSION, EDGE_WEIGHT_TYPE, ...)
    """

    meta: Dict[str, Any] = {}
    coords: List[Tuple[float, float]] = []
    dist_matrix: List[List[float]] = []

    # Carregamos todo o arquivo em memória (linhas) para facilitar lookahead
    with open(filepath, 'r') as fh:
        raw_lines = fh.readlines()

    # Normalize lines: strip trailing spaces but keep original for numeric parsing
    lines = [ln.rstrip('\n') for ln in raw_lines]

    # Primeiro passe: ler header até encontrar alguma SECTION
    idx = 0
    n_lines = len(lines)
    while idx < n_lines:
        line = lines[idx].strip()
        up = line.upper()
        if not line:
            idx += 1
            continue

        # Detecta inicio de seções
        if up.startswith("NODE_COORD_SECTION"):
            idx += 1
            # parse node coords from the subsequent lines
            coords = _parse_node_coord_section(lines[idx:], meta.get("DIMENSION"))
            # avançar idx até encontrar EOF or end - simplificação: break
            break

        if up.startswith("EDGE_WEIGHT_SECTION"):
            # temos uma matriz explícita - precisamos conhecer DIMENSION e FORMAT
            format_key = meta.get("EDGE_WEIGHT_FORMAT")
            dimension = int(meta.get("DIMENSION", 0))
            # parse from next lines
            # slice from next line and pass an iterator
            sub_lines = lines[idx + 1:]
            dist_matrix = _parse_edge_weight_section(sub_lines, dimension, format_key)
            break

        # Header key: value
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip().upper()
            val = parts[1].strip()
            meta[key] = val
            # ajustes básicos: DIMENSION como int
            if key == "DIMENSION":
                try:
                    meta["DIMENSION"] = int(val)
                except:
                    meta["DIMENSION"] = None
            # normaliza EDGE_WEIGHT_TYPE and EDGE_WEIGHT_FORMAT
            if key == "EDGE_WEIGHT_TYPE":
                meta["EDGE_WEIGHT_TYPE"] = val.upper()
            if key == "EDGE_WEIGHT_FORMAT":
                meta["EDGE_WEIGHT_FORMAT"] = val.upper()
        else:
            # também pode ser um header sem ':' (algumas variações)
            up = line.upper()
            if up.startswith("NAME") or up.startswith("TYPE") or up.startswith("DIMENSION") \
               or up.startswith("EDGE_WEIGHT_TYPE") or up.startswith("EDGE_WEIGHT_FORMAT"):
                parts = line.split()
                if len(parts) >= 2:
                    k = parts[0].strip().upper()
                    v = " ".join(parts[1:]).strip()
                    meta[k] = v
                    if k == "DIMENSION":
                        try:
                            meta["DIMENSION"] = int(v)
                        except:
                            meta["DIMENSION"] = None

        idx += 1

    # Pós-processamento:
    # Se não encontramos coords mas temos DIMENSION e o arquivo é do tipo EUC_2D,
    # tentamos ler NODE_COORD_SECTION mesmo que header tenha sido lido depois.
    # Se ainda não houver coords e não houver dist_matrix, procuramos manualmente a seção
    if not coords and not dist_matrix:
        # scan completo para NODE_COORD_SECTION (caso seja em maiúsculas/minúsculas variadas)
        for j, raw in enumerate(lines):
            if raw.strip().upper().startswith("NODE_COORD_SECTION"):
                coords = _parse_node_coord_section(lines[j + 1:], meta.get("DIMENSION"))
                break

    # Se não temos dist_matrix mas temos coords, construímos a matriz euclidiana
    if not dist_matrix and coords:
        dist_matrix = _euclidean_matrix_from_coords(coords)

    # Se o arquivo definia EDGE_WEIGHT_TYPE e é EUC_2D mas não havia NODE_COORD_SECTION,
    # isso é incoerente; apenas tenta prosseguir
    meta_out = meta.copy()

    if return_dist_matrix:
        return coords, dist_matrix, meta_out
    else:
        return coords
