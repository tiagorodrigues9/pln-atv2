# Estudo de Caso: Classificador de Avaliações com Perceptron (PLN)

O objetivo é classificar avaliações de clientes de uma loja de quadrinhos em **Positivas (1)** ou **Negativas (0)** utilizando uma rede neural **Perceptron de camada única**, implementada em **Python puro** (sem uso de bibliotecas de Machine Learning como Scikit-learn, TensorFlow ou PyTorch).

---

## Etapas do Projeto

O fluxo da aplicação segue o pipeline exigido:

$$\text{Avaliações} \longrightarrow \text{Pré-processamento} \longrightarrow \text{Vetorização (BoW)} \longrightarrow \text{Perceptron} \longrightarrow \text{Classificação}$$

1. **Base de Dados:** 20 avaliações de quadrinhos (10 positivas e 10 negativas).
2. **Pré-Processamento:**
   - Conversão para letras minúsculas;
   - Remoção de pontuações;
   - Separação em palavras (tokenização);
   - Remoção de *stopwords* (palavras comuns como artigos e preposições).
3. **Bag of Words (BoW) Manual:**
   - Criação manual do vocabulário único;
   - Representação binária das avaliações (1 se a palavra aparece, 0 caso contrário).
4. **Rede Neural Perceptron:**
   - Pesos e *bias* com inicialização aleatória;
   - Taxa de aprendizagem: `0.1`;
   - Função de ativação degrau: $f(\text{soma}) = 1$ se $\text{soma} > 0.5$, senão $0$.
5. **Treinamento:**
   - Atualização iterativa dos pesos e bias conforme o erro;
   - Critério de parada antecipada com zero erros.
6. **Avaliação e Métricas:**
   - Teste na base original (acurácia de 100%);
   - Teste com 5 novas avaliações (acurácia de 80%);
   - Análise de limitações do modelo Bag of Words e contexto.

---

## Arquivos do Repositório

- **`perceptron_tbbt.py`**: Código-fonte principal em Python contendo todo o pipeline (base de dados, pré-processamento, BoW, Perceptron e testes).
- **`RELATORIO_ESTUDO_DE_CASO.md`**: Relatório detalhado com tabelas, métricas e respostas completas às questões analíticas.

---

## Como Executar

### Pré-requisitos
Apenas o **Python 3.8+** instalado (não requer instalação de nenhuma biblioteca externa via `pip`).

### Executando o script
No terminal, na pasta do projeto, execute:

```bash
python perceptron_tbbt.py
```

---

## Resultados Resumidos

- **Base de Treino (20 avaliações):** 20 acertos / 0 erros (**100% de acurácia**).
- **Novas Avaliações (5 frases inéditas):** 4 acertos / 1 erro (**80% de acurácia**).
- **Conclusão:** O Perceptron separa linearmente os dados com sucesso, mas o modelo Bag of Words possui limitações para frases com conjunções adversativas (ex.: *"esperava uma história incrível, mas foi decepcionante"*), pois avalia apenas a presença das palavras e não a ordem/contexto sintático.

