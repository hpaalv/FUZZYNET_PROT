# FUZZYNET_PROT
Interpretable protein–protein interaction predictor combining ProtT5 embeddings with a neuro-fuzzy ANFIS. End-to-end: download FASTA, build balanced pairs, GPU/CPU train, export accuracy &amp; confusion matrix, visualise fuzzy rules. MIT-licensed.
```markdown
# FUZZYNET-PROT 👾🔬

Predição de interações proteína-proteína (PPI) **explicável** usando **ANFIS** (Adaptive Neuro-Fuzzy Inference System) ou **FRNN** (Fuzzy Recurrent Neuro-Network) sobre *embeddings* ProT5.

```

├── fuzzynet\_pipeline.py   # núcleo: embeddings → PCA → z-score → modelo → predição
├── app\_gui.py             # interface gráfica (CustomTkinter)
├── models/
│   ├── anfis.pkl          # modelo ANFIS treinado
│   ├── frnn.pkl           # modelo FRNN (placeholder)
│   ├── pca\_model.pkl      # PCA (≥ 90 % variância)
│   └── scaler.pkl         # StandardScaler
└── tmp/                   # gerado p/ salvar figuras

````

Todos os arquivos podem ficar no mesmo diretório; a GUI cria `tmp/` automaticamente.

---

## 1 · Instalação

Requer **Python ≥ 3.9**.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

**`requirements.txt`**

```
torch>=2.2
transformers>=4.39
sentencepiece
scikit-learn
numpy
pandas
matplotlib
pillow
customtkinter
```

*Caso possua GPU NVIDIA, instale a versão de PyTorch com CUDA conforme guia oficial.*
O download dos pesos ProT5 (\~3,7 GB) ocorre na primeira execução.

---

## 2 · Uso rápido (CLI)

```bash
python fuzzynet_pipeline.py \
  --seq1 "MDAQTRRRSSG..." \
  --seq2 "MQIFVKTLTGK..." \
  --model ANFIS          # ou FRNN
```

Saída JSON:

* `class` – rótulo (0, 1, 2)
* `probs` – probabilidades normalizadas

| Classe | combined\_score (STRING) | Significado                                           |
| ------ | ------------------------ | ----------------------------------------------------- |
| **2**  | > 0.75                   | Forte evidência experimental/computacional            |
| **1**  | 0.40–0.75                | Evidência intermediária                               |
| **0**  | < 0.40                   | Ausência de evidência (tratado como “não interativo”) |

---

## 3 · Interface gráfica

```bash
python app_gui.py
```

| Componente                       | Função                                                       |
| -------------------------------- | ------------------------------------------------------------ |
| **Escolha da Rede**              | Seleciona ANFIS ou FRNN                                      |
| **Sequência 1 / Sequência 2**    | Colar sequências aminoacídicas                               |
| **Variáveis significativas (k)** | Slider 1–10 para número de variáveis nos gráficos            |
| **Executar Predição**            | Executa o pipeline, mostra classe, probabilidades e gráficos |

Saídas:

* Texto com classe prevista e vetor de probabilidades
* Gráfico das funções de pertinência (`tmp/mf.png`) exibido na tela

---

## 4 · Erros comuns

| Mensagem                  | Causa                           | Solução                         |
| ------------------------- | ------------------------------- | ------------------------------- |
| `Model file … not found.` | `.pkl` ausente em `models/`     | Copiar modelos ou treinar novos |
| `CUDA out of memory`      | GPU insuficiente nos embeddings | Rodar no modo CPU               |
| `No sequence provided`    | Sequências em branco na GUI     | Preencher ambas as caixas       |

---

## 5 · Referências

* Jang, J.-S. R. “ANFIS: Adaptive-Network-Based Fuzzy Inference System.” *IEEE TSMC* 23 (3): 665–685, 1993.
* Elnaggar A. *et al.* “ProtT5: A Transformer Model for Protein Sequence Processing.” *bioRxiv* 2022.
* Szklarczyk D. *et al.* “STRING v12: Protein–Protein Association Networks.” *NAR* 2023.

---

## 6 · Contribuições

Pull requests são bem-vindos! Para integrar um FRNN real, coloque o novo `.pkl` em `models/` e ajuste `fuzzynet_pipeline.py`.

---

## 7 · Licença

MIT.

```
```
