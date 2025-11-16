from dataclasses import dataclass, field
from typing import Any, List
from math import ceil, log


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

      m = ordem = número MÁXIMO de filhos de uma página
      Máx. filhos por página:        Fmax = m
      Máx. chaves por página:        Kmax = m - 1
      Mín. filhos (≠ raiz):          Fmin = ceil(m / 2)
      Mín. chaves (≠ raiz):          Kmin = Fmin - 1

    Internamente usamos t = Fmin (grau mínimo) para os algoritmos
    de inserção e deleção.
    """

    def __init__(self, order: int):
        if order < 3:
            raise ValueError("A ordem m deve ser >= 3")
        self.order = order
        self.max_children = order
        self.max_keys = order - 1

        self.min_children = ceil(order / 2)
        self.min_keys = self.min_children - 1

        # grau mínimo usado nos algoritmos (CLRS)
        self.t = self.min_children

        self._next_rrn = 0
        self.root = self._new_node(leaf=True)

    # ----------------------------------------------------
    # Estatísticas da árvore (para estudos)
    # ----------------------------------------------------
    def _count_keys(self, node: BTreeNode | None = None) -> int:
        """Conta o número total de chaves na árvore (N)."""
        if node is None:
            node = self.root
        total = node.keycount
        for child in node.children:
            total += self._count_keys(child)
        return total

    def total_keys(self) -> int:
        """Interface pública para contar chaves."""
        # árvore vazia: 0
        if self.root.keycount == 0 and self.root.leaf:
            return 0
        return self._count_keys(self.root)

    def _height(self, node: BTreeNode | None = None) -> int:
        """
        Altura (profundidade) da árvore em número de páginas.

        Por definição de Árvore B, todos os caminhos da raiz até as folhas
        têm o mesmo comprimento; então basta seguir sempre o filho 0.
        """
        if node is None:
            node = self.root

        if node.leaf:
            return 1
        return 1 + self._height(node.children[0])

    def actual_depth(self) -> int:
        """
        Profundidade REAL da árvore (número de páginas acessadas
        no pior caso).
        """
        if self.root.keycount == 0 and self.root.leaf:
            return 0
        return self._height(self.root)

    def theoretical_depth_upper_bound(self) -> float:
        """
        Limite superior TEÓRICO da profundidade (slide):

            d ≤ 1 + log_{ceil(m/2)} ((N + 1) / 2)

        onde:
          - N = número total de chaves
          - m = ordem (self.order)
        """
        N = self.total_keys()
        if N == 0:
            return 0.0

        base = self.min_children   # = ceil(m/2)
        # Fórmula do slide
        value = (N + 1) / 2
        d = 1 + log(value, base)
        return d

    
    # -------- criação de páginas (RRN) --------
    def _new_node(self, leaf: bool) -> BTreeNode:
        rrn = self._next_rrn
        self._next_rrn += 1
        return BTreeNode(rrn=rrn, leaf=leaf)

    # ----------------------------------------------------
    # Métodos utilitários para estudo
    # ----------------------------------------------------
    def describe(self):
        N = self.total_keys()
        d_theoretical = self.theoretical_depth_upper_bound()
        max_acessos = int(d_theoretical) 
        d_real = self.actual_depth()

        print(f"Árvore B de ORDEM m = {self.order}")
        print(f"  Máx. filhos por página        = {self.max_children}")
        print(f"  Máx. chaves por página        = {self.max_keys}")
        print(f"  Mín. filhos (exceto raiz)     = {self.min_children}")
        print(f"  Mín. chaves (exceto raiz)     = {self.min_keys}")
        print(f"  Grau mínimo interno (t)       = {self.t}")
        print()
        print(f"  Nº total de chaves (N)        = {N}")
        print(f"  Profundidade REAL (d_real)    = {d_real}")
        print(
            f"  Limite superior teórico d ≤ 1 + log_{self.min_children}"
            f"((N+1)/2) = {d_theoretical:.4f}"
        )
        print(
            f"  Máx. inteiro de acessos a disco "
            f"(ceil(d_teórico))        = {max_acessos}"
        )
        print()

    def _build_tree_text(self, node: BTreeNode | None = None, level: int = 0) -> str:
        """Gera o texto da árvore (mesmo formato do print_tree)."""
        if node is None:
            node = self.root

        indent = "  " * level
        text = (
            f"{indent}Nível {level} | RRN={node.rrn} | "
            f"chaves={node.keys} | KEYCOUNT={node.keycount} | "
            f"filhos={len(node.children)} | folha={node.leaf}\n"
        )

        for child in node.children:
            text += self._build_tree_text(child, level + 1)

        return text
    
    def save_tree_to_file(self, filename: str = "arvore.txt"):
        """Grava a árvore inteira em um arquivo de texto."""
        text = self._build_tree_text(self.root)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Árvore gravada em: {filename}")

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
        while i < node.keycount and key > node.keys[i]:
            i += 1

        if i < node.keycount and key == node.keys[i]:
            return node, i

        if node.leaf:
            return None, None

        return self._search_node(node.children[i], key)

    def search(self, key):
        """
        Função de pesquisa no estilo da apostila.
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

    # ================== DELEÇÃO ==================

    def delete(self, key):
        """
        Remove uma chave da árvore B, reorganizando as páginas.
        """
        self._delete(self.root, key)

        # Se a raiz ficar sem chaves e tiver filho único, encolhe a árvore
        if self.root.keycount == 0 and not self.root.leaf:
            self.root = self.root.children[0]

    def _delete(self, node: BTreeNode, key):
        t = self.t
        idx = 0

        # Encontra a primeira chave >= key no nó
        while idx < node.keycount and key > node.keys[idx]:
            idx += 1

        # ---- Caso 1: a chave está neste nó ----
        if idx < node.keycount and node.keys[idx] == key:
            if node.leaf:
                # 1a) nó folha -> remove direto
                node.keys.pop(idx)
            else:
                # nó interno: filhos à esquerda (idx) e direita (idx+1)
                if node.children[idx].keycount >= t:
                    # 1b) filho esquerdo tem >= t chaves -> usa predecessor
                    pred = self._get_predecessor(node, idx)
                    node.keys[idx] = pred
                    self._delete(node.children[idx], pred)
                elif node.children[idx + 1].keycount >= t:
                    # 1c) filho direito tem >= t chaves -> usa sucessor
                    succ = self._get_successor(node, idx)
                    node.keys[idx] = succ
                    self._delete(node.children[idx + 1], succ)
                else:
                    # 1d) ambos têm t-1 chaves -> merge dos dois filhos + chave
                    self._merge_children(node, idx)
                    self._delete(node.children[idx], key)

        # ---- Caso 2: a chave NÃO está neste nó ----
        else:
            if node.leaf:
                # não achou (chegamos numa folha)
                return

            # Vamos descer para o filho 'idx',
            # garantindo antes que ele terá pelo menos t chaves.
            if idx == node.keycount:
                child = node.children[idx]
            else:
                child = node.children[idx]

            if child.keycount < t:
                # tenta emprestar do irmão à esquerda
                if idx > 0 and node.children[idx - 1].keycount >= t:
                    self._borrow_from_prev(node, idx)
                # ou do irmão à direita
                elif idx < len(node.children) - 1 and node.children[idx + 1].keycount >= t:
                    self._borrow_from_next(node, idx)
                else:
                    # senão, faz merge com um irmão
                    if idx < len(node.children) - 1:
                        self._merge_children(node, idx)
                    else:
                        self._merge_children(node, idx - 1)
                        idx -= 1

            self._delete(node.children[idx], key)

    # ---- auxiliares da deleção ----

    def _get_predecessor(self, node: BTreeNode, idx: int):
        cur = node.children[idx]
        while not cur.leaf:
            cur = cur.children[cur.keycount]
        return cur.keys[-1]

    def _get_successor(self, node: BTreeNode, idx: int):
        cur = node.children[idx + 1]
        while not cur.leaf:
            cur = cur.children[0]
        return cur.keys[0]

    def _merge_children(self, parent: BTreeNode, idx: int):
        """
        Merge:
          child = filho idx
          sibling = filho idx+1
          traz a chave parent.keys[idx] para o child
        """
        child = parent.children[idx]
        sibling = parent.children[idx + 1]

        # chave separadora desce
        child.keys.append(parent.keys[idx])
        # concatena as chaves do irmão
        child.keys.extend(sibling.keys)

        # concatena filhos se não for folha
        if not child.leaf:
            child.children.extend(sibling.children)

        # remove chave e ponteiro do pai
        parent.keys.pop(idx)
        parent.children.pop(idx + 1)

    def _borrow_from_prev(self, parent: BTreeNode, idx: int):
        """Empresta uma chave do irmão à esquerda."""
        child = parent.children[idx]
        left = parent.children[idx - 1]

        # puxa a chave do pai para o começo do filho
        child.keys.insert(0, parent.keys[idx - 1])

        if not child.leaf:
            # puxa o último filho do irmão para o começo
            child.children.insert(0, left.children.pop())

        # sobe a última chave do irmão para o pai
        parent.keys[idx - 1] = left.keys.pop()

    def _borrow_from_next(self, parent: BTreeNode, idx: int):
        """Empresta uma chave do irmão à direita."""
        child = parent.children[idx]
        right = parent.children[idx + 1]

        # puxa a chave do pai para o final do filho
        child.keys.append(parent.keys[idx])

        if not child.leaf:
            # puxa o primeiro filho do irmão
            child.children.append(right.children.pop(0))

        # sobe a primeira chave do irmão para o pai
        parent.keys[idx] = right.keys.pop(0)


# ================== EXEMPLO DE USO ==================

if __name__ == "__main__":
    # Exemplo: ordem 8 (m=8) -> máx. 7 chaves por página, mín. 3 (exceto raiz)
    arvore = BTree(order=3)
    arvore.describe()

    valores = [99, 10, 20, 5, 6, 12, 30, 7, 17, 3, 4, 2, 8, 9]
    for v in valores:
        print(f"\nInserindo {v}...")
        arvore.insert(v)
        arvore.print_tree()
        arvore.save_tree_to_file("arvore.txt")

    print("\n===== Testando pesquisa =====")
    arvore.search(12)   # deve achar
    arvore.search(15)   # não deve achar

    # print("\n===== Deleções para observar reorganização =====")
    # for d in [6, 7, 8, 10, 20, 5, 12, 30]:
    #     print(f"\n>>> Deletando {d}")
    #     arvore.delete(d)
    #     arvore.print_tree()
    
    
    # Ex.: ordem 512 (m=512) como no slide
    arvore = BTree(order=515)

    # só pra testar, insere 1.000.000 chaves sequenciais
    for v in range(10_000_000):
        arvore.insert(v)

    arvore.describe()
