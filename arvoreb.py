from dataclasses import dataclass, field
from typing import Any, List
from math import ceil


@dataclass
class BTreeNode:
    """
    Página (nó) de uma Árvore B.

    rrn      -> identifica a página (como se fosse o RRN no arquivo)
    keys     -> chaves armazenadas na página (ordenadas)
    children -> ponteiros para as páginas-filhas
    leaf     -> indica se é folha
    """
    rrn: int
    leaf: bool
    keys: List[Any] = field(default_factory=list)
    children: List["BTreeNode"] = field(default_factory=list)

    @property
    def keycount(self) -> int:
        """Número de chaves armazenadas na página (KEYCOUNT)."""
        return len(self.keys)


class BTree:
    """
    Árvore B de ORDEM m (m pode ser par ou ímpar).

    Definições (no estilo da apostila):

      - m = ordem da árvore = número MÁXIMO de filhos de uma página.
      - Máximo de chaves por página:          Kmax = m - 1
      - Mínimo de filhos por página (≠ raiz): Fmin = ceil(m / 2)
      - Mínimo de chaves por página (≠ raiz): Kmin = Fmin - 1

    Observação:
      - O algoritmo clássico usa o "grau mínimo" t.
      - Aqui usamos t = Fmin = ceil(m/2) internamente para fazer o split.
      - Se m for par, isso bate PERFEITO com a teoria padrão (m = 2t).
      - Se m for ímpar, a árvore continua funcionando para busca/inserção,
        mas alguns nós podem ficar com um pouco menos que Kmin chaves
        após splits (didático, não crítico pro seu estudo).
    """

    def __init__(self, order: int):
        if order < 3:
            raise ValueError("A ordem m deve ser >= 3")
        self.order = order           # m
        self.max_children = order    # máximo de filhos
        self.max_keys = order - 1    # máximo de chaves

        # mínimo teórico (usando ceil)
        self.min_children = ceil(order / 2)
        self.min_keys = self.min_children - 1

        # grau mínimo interno para o algoritmo de split
        self.t = self.min_children

        self._next_rrn = 0           # para numerar as páginas
        self.root = self._new_node(leaf=True)

    # -------- criação de páginas (RRN) --------
    def _new_node(self, leaf: bool) -> BTreeNode:
        rrn = self._next_rrn
        self._next_rrn += 1
        return BTreeNode(rrn=rrn, leaf=leaf)

    # ----------------------------------------------------
    # Métodos utilitários para estudo
    # ----------------------------------------------------
    def describe(self):
        print(f"Árvore B de ORDEM m = {self.order}")
        print(f"  Máx. filhos por página        = {self.max_children}")
        print(f"  Máx. chaves por página        = {self.max_keys}")
        print(f"  Mín. filhos (exceto raiz)     = {self.min_children}")
        print(f"  Mín. chaves (exceto raiz)     = {self.min_keys}")
        print(f"  Grau mínimo interno (t)       = {self.t}")
        print()

    def print_tree(self, node: BTreeNode | None = None, level: int = 0):
        """Imprime a árvore, página por página."""
        if node is None:
            node = self.root

        indent = "  " * level
        print(
            f"{indent}Nível {level} | RRN={node.rrn} | "
            f"chaves={node.keys} | KEYCOUNT={node.keycount} | "
            f"filhos={len(node.children)} | folha={node.leaf}"
        )
        for child in node.children:
            self.print_tree(child, level + 1)

    # ================== PESQUISA ==================

    def _search_node(self, node: BTreeNode, key):
        """
        Pesquisa recursiva a partir de 'node'.

        Retorna:
            (node_encontrado, índice_da_chave)
        ou:
            (None, None) se não achou.
        """
        i = 0
        # avança enquanto a chave procurada for maior
        while i < node.keycount and key > node.keys[i]:
            i += 1

        # se a chave está nesta página
        if i < node.keycount and key == node.keys[i]:
            return node, i

        # se é folha, não tem para onde descer
        if node.leaf:
            return None, None

        # senão, desce para o filho apropriado
        return self._search_node(node.children[i], key)

    def search(self, key):
        """
        Função de pesquisa no estilo da apostila.

        Dados:  raiz da árvore + chave procurada.
        Saída:  imprime 'Achou' ou 'Não-Achou' e retorna:

            - (True,  rrn, pos) se achou
            - (False, None, None) se não achou
        """
        if self.root.keycount == 0 and self.root.leaf:
            print(f"Não-Achou: árvore vazia (chave {key})")
            return False, None, None

        node, idx = self._search_node(self.root, key)

        if node is None:
            print(f"Não-Achou: chave {key} não está na árvore.")
            return False, None, None
        else:
            print(
                f"Achou: chave {key} está na página RRN={node.rrn} "
                f"na posição {idx} da página."
            )
            return True, node.rrn, idx

    # ================== INSERÇÃO ==================

    def insert(self, key):
        root = self.root

        # se a raiz está cheia (max_keys), precisamos dividir primeiro
        if root.keycount == self.max_keys:
            new_root = self._new_node(leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
            self._insert_non_full(new_root, key)
        else:
            self._insert_non_full(root, key)

    def _split_child(self, parent: BTreeNode, index: int):
        """
        Divide o filho 'index' de 'parent':
          - y: nó cheio (max_keys chaves)
          - z: novo nó (direita)
          - chave do meio sobe para o pai
        """
        t = self.t
        y = parent.children[index]
        z = self._new_node(leaf=y.leaf)

        # z recebe as últimas t-1 chaves de y
        z.keys = y.keys[t:]
        middle_key = y.keys[t - 1]

        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]

        # y fica com as t-1 primeiras chaves
        y.keys = y.keys[:t - 1]

        parent.children.insert(index + 1, z)
        parent.keys.insert(index, middle_key)

    def _insert_non_full(self, node: BTreeNode, key):
        """Insere uma chave em um nó que não está cheio."""
        i = node.keycount - 1

        if node.leaf:
            node.keys.append(None)  # espaço extra
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            if node.children[i].keycount == self.max_keys:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key)


# ================== EXEMPLO DE USO ==================

if __name__ == "__main__":
    # Ex.: ordem m = 8 -> máx. 7 chaves por página, mín. 3 (exceto raiz)
    arvore = BTree(order=8)
    arvore.describe()

    valores = [99, 10, 20, 20, 5, 10, 5, 6, 12, 30, 7, 17, 3, 4, 2, 8, 9, 1, 0, 0, 10, 71]
    for v in valores:
        print(f"\nInserindo {v}...")
        arvore.insert(v)
        arvore.print_tree()

    print("\n===== Testando pesquisa =====")
    arvore.search(12)   # deve achar
    arvore.search(15)   # não deve achar
