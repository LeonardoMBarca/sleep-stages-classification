
# Linha Base: Classificação do Sono com Regressão Logística

## 📄 Introdução
Este projeto estabelece uma linha de base (baseline) para a classificação dos estágios do sono (stage). O fluxo de trabalho demonstra o carregamento dos dados, o pré-processamento, a aplicação do modelo de Regressão Logística, e a avaliação detalhada da performance nos conjuntos de validação e teste.

## 🛠️ Tecnologias e Dependências
O projeto utiliza bibliotecas padrão de ciência de dados e Machine Learning, incluindo:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib.pyplot`
- Componentes do `scikit-learn`:
  - `LabelEncoder`
  - `StandardScaler`
  - `SimpleImputer`
  - `LogisticRegression`
  - `classification_report`
  - `confusion_matrix`

## 📂 Dados
Os dados de treino, validação e teste são carregados a partir de arquivos Parquet localizados no caminho `base_dir / "datalake" / "data-for-model"`. O script verifica:
- Dimensões de cada conjunto de dados
- Distribuição da variável alvo (`stage`)
- Tipos de dados
- Contagem de valores nulos

---

## ⚙️ Pipeline de Modelagem

### 1. Pré-processamento
- **Separação de Features**: As colunas alvo (`stage`) e irrelevantes (`subject_id`, `night_id`, `sex`, `age`) são removidas dos conjuntos de features (`X`).
- **Codificação da Variável Alvo**: A variável alvo categórica (`stage`) é codificada em valores numéricos (`y_train_enc`, etc.) usando `LabelEncoder`. O script imprime as classes codificadas (e.g., N1, N2, N3, REM, W).
- **Normalização das Features**: As features são escalonadas usando `StandardScaler`, resultando em `X_train_scaled`, `X_val_scaled` e `X_test_scaled`. É realizada uma verificação do shape e um resumo estatístico das features normalizadas.

### 2. Tratamento de NaNs
- **Imputação**: Após a normalização, o `SimpleImputer` é utilizado com a estratégia `"mean"` para substituir quaisquer valores nulos (NaNs) remanescentes nas features escalonadas.

### 3. Treinamento
- **Modelo**: `LogisticRegression`
- **Configuração**:
  - `multi_class="multinomial"`
  - `class_weight="balanced"`
  - `random_state=42`
  - `max_iter=1000`
- **Treinamento**: O modelo é ajustado utilizando os dados de treino imputados e escalonados (`X_train_scaled`) e a variável alvo codificada (`y_train_enc`).

---

## 📊 Avaliação do Modelo

A performance é medida através do `classification_report` e da `confusion_matrix`, utilizando os labels originais das classes.

### 🔹 Conjunto de Validação
- **Acurácia**: 72%
- **Macro F1-Score**: 0.66

| Classe | Precision | Recall | F1-Score | Suporte |
|--------|-----------|--------|----------|---------|
| N1     | 0.32      | 0.53   | 0.40     | 4215    |
| N2     | 0.89      | 0.60   | 0.72     | 15066   |
| N3     | 0.50      | 0.85   | 0.63     | 2524    |
| REM    | 0.60      | 0.81   | 0.69     | 5456    |
| W      | 0.93      | 0.84   | 0.88     | 15120   |

### 🔹 Conjunto de Teste (Baseline)
- **Acurácia**: 72%
- **Macro F1-Score**: 0.66

---

## 📈 Visualizações Geradas

O script gera diversos gráficos para visualizar o desempenho do modelo, utilizando as métricas extraídas dos relatórios de classificação:

- **Métricas por Classe**: Gráfico de barras (`classification_metrics.png`) comparando Precision, Recall e F1-Score para cada classe de sono.
- **Matriz de Confusão (Validação)**: Mapa de calor (`sns.heatmap`) da matriz de confusão do conjunto de validação.
- **Comparação Validação x Teste**: Gráfico de barras comparando Precision, Recall e F1-Score entre os conjuntos de validação e teste.
- **Resumo de Performance**: Gráfico de barras comparando Acurácia e Macro F1-Score para os conjuntos de Validação e Teste.

---

## 📌 Conclusões
- O modelo baseline demonstra bom desempenho nas classes de sono **W** (vigília), **REM** e **N3** (sono de ondas lentas), conforme indicado pelos altos valores de F1-Score e Recall.
- As classes **N1** e **N2** apresentam maior dificuldade de distinção para o modelo, o que pode ser esperado devido à sua semelhança fisiológica.
- O desempenho do modelo é consistente entre os conjuntos de Validação e Teste, com Acurácia e Macro F1-Score de 72% e 0.66, respectivamente.
