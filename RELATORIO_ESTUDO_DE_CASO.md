# Estudo de Caso: Loja de Quadrinhos The Comic Center of Pasadena
## Classificação de Avaliações com Processamento de Linguagem Natural e Perceptron

**Instituição:** Fatec Registro - Centro Paula Souza  
**Disciplina:** Processamento de Linguagem Natural (PLN)  
**Arquivo de Código:** [`perceptron_tbbt.py`](./perceptron_tbbt.py)

---

## 1. Introdução e Objetivo

A loja de quadrinhos **The Comic Center of Pasadena** necessita de um sistema automático para classificar as avaliações deixadas por seus clientes em **Positivas (1)** ou **Negativas (0)**. 

Para solucionar esse problema de forma simples e direta, foi implementado um fluxo completo em **Python puro** (sem uso de frameworks pesados de Machine Learning como Scikit-learn, TensorFlow ou PyTorch):

$$\text{Avaliações} \longrightarrow \text{Pré-processamento} \longrightarrow \text{Vetorização (BoW)} \longrightarrow \text{Perceptron} \longrightarrow \text{Classificação}$$

---

## 2. Base de Dados Fictícia (20 Avaliações)

Foram criadas 20 avaliações variadas sobre quadrinhos, com 10 exemplos da classe positiva ($1$) e 10 exemplos da classe negativa ($0$).

| ID | Avaliação | Classe Real |
|:--:|:---|:---:|
| **01** | História incrível e os desenhos são excelentes, adorei a leitura. | 1 (Positiva) |
| **02** | Os quadrinhos chegaram em perfeito estado e o roteiro é espetacular. | 1 (Positiva) |
| **03** | Edição maravilhosa com ilustrações fantásticas e trama envolvente. | 1 (Positiva) |
| **04** | Muito bom, arte impecável e personagens bem desenvolvidos. | 1 (Positiva) |
| **05** | HQ sensacional, entrega rápida e acabamento de alta qualidade. | 1 (Positiva) |
| **06** | Recomendo demais, o desenvolvimento da história me surpreendeu positivamente. | 1 (Positiva) |
| **07** | Excelente material, capa dura linda e diálogos fascinantes. | 1 (Positiva) |
| **08** | Uma das melhores histórias que já li, arte visual magnífica. | 1 (Positiva) |
| **09** | Experiência de leitura incrível, roteirista brilhante e final emocionante. | 1 (Positiva) |
| **10** | Quadrinho ótimo, arte espetacular e edição impecável. | 1 (Positiva) |
| **11** | A história é muito fraca e a qualidade do quadrinho deixou a desejar. | 0 (Negativa) |
| **12** | Roteiro confuso e personagens sem graça, não gostei nada. | 0 (Negativa) |
| **13** | Desenhos horríveis e impressão de péssima qualidade, muito decepcionante. | 0 (Negativa) |
| **14** | HQ chata e arrastada, história fraca que não prende a atenção. | 0 (Negativa) |
| **15** | Material fraco, páginas soltando e roteiro decepcionante. | 0 (Negativa) |
| **16** | Pior quadrinho do ano, enredo cansativo e trama sem sentido. | 0 (Negativa) |
| **17** | Não recomendo, narrativa tediosa e ilustrações fracas. | 0 (Negativa) |
| **18** | A leitura foi cansativa e o final foi péssimo e previsível. | 0 (Negativa) |
| **19** | Edição veio com defeito e a história é horrível. | 0 (Negativa) |
| **20** | Decepcionado com a compra, história boba e arte medíocre. | 0 (Negativa) |

---

## 3. Pré-Processamento dos Textos

Para cada avaliação, foram aplicadas as 4 etapas solicitadas:
1. **Conversão para minúsculas:** padroniza o texto (`texto.lower()`).
2. **Remoção de pontuação:** retira vírgulas, pontos e sinais (`string.punctuation`).
3. **Tokenização:** divide o texto em palavras individuais (`texto.split()`).
4. **Remoção de Stopwords:** desconsidera artigos, preposições e conectivos irrelevantes para o sentimento (ex.: *de*, *do*, *da*, *e*, *é*, *um*, *uma*, *em*, *por*, *para*, etc.).

---

## 4. Bag of Words (BoW) Manual

- O vocabulário único foi construído coletando todas as palavras resultantes do pré-processamento (94 termos distintos).
- Cada avaliação foi convertida manualmente em um vetor binário $X = [x_1, x_2, \dots, x_n]$, onde:
  - $x_i = 1$, se a palavra $i$ do vocabulário está presente no texto;
  - $x_i = 0$, caso contrário.

---

## 5. Implementação da Rede Neural Perceptron

A rede de camada única contém:
- Um peso $w_i$ para cada palavra do vocabulário, inicializado aleatoriamente em $[-0.1, 0.1]$;
- Um termo de **bias** $b$, também inicializado aleatoriamente;
- Taxa de aprendizagem $\eta = 0.1$;
- **Função de ativação degrau:**
  $$f(\text{soma}) = \begin{cases} 1, & \text{se } \text{soma} > 0.5 \\ 0, & \text{caso contrário} \end{cases}$$
- **Soma Ponderada:**
  $$\text{Soma} = \left(\sum_{i=1}^n x_i \cdot w_i\right) + b$$
- **Regra de Atualização dos Pesos:**
  $$\text{erro} = y_{\text{real}} - y_{\text{previsto}}$$
  $$w_i \leftarrow w_i + \eta \cdot \text{erro} \cdot x_i$$
  $$b \leftarrow b + \eta \cdot \text{erro}$$

O algoritmo executa no máximo 100 épocas e interrompe antecipadamente quando o número de erros na época é zero.

---

## 6. Resultados dos Testes na Base de Treino

O modelo atingiu **convergência completa com 0 erros** já na 4ª época:

- **Total de avaliações:** 20
- **Classificações corretas:** 20
- **Classificações incorretas:** 0
- **Acurácia na base:** **100.00%**

---

## 7. Teste com 5 Novas Avaliações

Foram testadas 5 avaliações inéditas que não faziam parte do treino:

| ID | Nova Avaliação | Classe Esperada | Classe Prevista | Resultado |
|:---:|:---|:---:|:---:|:---:|
| **N1** | "História fantástica, amei os desenhos e recomendo a leitura." | 1 (Positiva) | 1 (Positiva) | **Correto** |
| **N2** | "Quadrinho com roteiro fraco e final decepcionante." | 0 (Negativa) | 0 (Negativa) | **Correto** |
| **N3** | "Excelente edição, arte maravilhosa e acabamento impecável." | 1 (Positiva) | 1 (Positiva) | **Correto** |
| **N4** | "História cansativa e muito tediosa, não gostei nada." | 0 (Negativa) | 0 (Negativa) | **Correto** |
| **N5** | "Esperava uma história incrível e desenhos excelentes, mas foi bem decepcionante." | 0 (Negativa) | 1 (Positiva) | **Incorreto** |

- **Acurácia nas novas avaliações:** **80.0%** (4/5 acertos).

---

## 8. Análise dos Resultados (Questão 9 do Estudo de Caso)

### a) A rede aprendeu a classificar as avaliações?
> **Sim.** Na base de dados original, o Perceptron convergiu rapidamente (época 4) e obteve **100% de acurácia**, demonstrando capacidade de separar linearmente avaliações positivas de negativas quando termos característicos estão presentes. No conjunto de validação com novas frases, alcançou **80% de acurácia**.

### b) Quantas avaliações foram classificadas incorretamente?
> - Na base de dados original: **0 erros** (0 / 20).  
> - No teste com 5 novas frases: **1 erro** (1 / 5).  
>   - **Frase com erro [N5]:** *"Esperava uma história incrível e desenhos excelentes, mas foi bem decepcionante."*  
>   - Classe esperada: **0 (Negativa)**.  
>   - Classe prevista: **1 (Positiva)**.

### c) Por que ocorreram os erros?
O erro observado na avaliação N5 evidencia diretamente os seguintes fatores clássicos de PLN:
1. **Limitações do Modelo Bag of Words:** O modelo ignora a ordem sintática das palavras e a estrutura gramatical. Ele atua como uma simples "contagem" ou "presença" de termos.
2. **Ausência de Análise de Contexto e Conjunções Adversativas:** A frase N5 continha termos que acumulavam fortes pesos positivos durante o treinamento (*"incrível"*, *"desenhos"*, *"excelentes"*). O modelo não compreende a quebra de expectativa (*"Esperava..."*) nem a mudança de sentido imposta pela adversativa (*"mas foi bem decepcionante"*). A soma ponderada das palavras positivas superou o peso negativo isolado da palavra *"decepcionante"*.
3. **Tamanho Reduzido da Base:** Com apenas 20 avaliações, o vocabulário não inclui n-gramas ou construções semânticas mais ricas capazes de ponderar negações e nuances.
4. **Palavras Neutras ou Compartilhadas:** Substantivos genéricos como *"história"*, *"quadrinho"* ou *"edição"* aparecem tanto em contextos positivos quanto negativos, demandando adjetivos qualificadores para a diferenciação.

---

## 9. Como Executar o Código

Para rodar o programa, basta executar no terminal:

```bash
python perceptron_tbbt.py
```
*(ou caso use o gerenciador `uv`: `uv run python perceptron_tbbt.py`)*

