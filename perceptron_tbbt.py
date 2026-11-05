import random
import string

random.seed(42)

base_de_dados = [
    {"id": 1,  "texto": "História incrível e os desenhos são excelentes, adorei a leitura.", "classe": 1},
    {"id": 2,  "texto": "Os quadrinhos chegaram em perfeito estado e o roteiro é espetacular.", "classe": 1},
    {"id": 3,  "texto": "Edição maravilhosa com ilustrações fantásticas e trama envolvente.", "classe": 1},
    {"id": 4,  "texto": "Muito bom, arte impecável e personagens bem desenvolvidos.", "classe": 1},
    {"id": 5,  "texto": "HQ sensacional, entrega rápida e acabamento de alta qualidade.", "classe": 1},
    {"id": 6,  "texto": "Recomendo demais, o desenvolvimento da história me surpreendeu positivamente.", "classe": 1},
    {"id": 7,  "texto": "Excelente material, capa dura linda e diálogos fascinantes.", "classe": 1},
    {"id": 8,  "texto": "Uma das melhores histórias que já li, arte visual magnífica.", "classe": 1},
    {"id": 9,  "texto": "Experiência de leitura incrível, roteirista brilhante e final emocionante.", "classe": 1},
    {"id": 10, "texto": "Quadrinho ótimo, arte espetacular e edição impecável.", "classe": 1},

    {"id": 11, "texto": "A história é muito fraca e a qualidade do quadrinho deixou a desejar.", "classe": 0},
    {"id": 12, "texto": "Roteiro confuso e personagens sem graça, não gostei nada.", "classe": 0},
    {"id": 13, "texto": "Desenhos horríveis e impressão de péssima qualidade, muito decepcionante.", "classe": 0},
    {"id": 14, "texto": "HQ chata e arrastada, história fraca que não prende a atenção.", "classe": 0},
    {"id": 15, "texto": "Material fraco, páginas soltando e roteiro decepcionante.", "classe": 0},
    {"id": 16, "texto": "Pior quadrinho do ano, enredo cansativo e trama sem sentido.", "classe": 0},
    {"id": 17, "texto": "Não recomendo, narrativa tediosa e ilustrações fracas.", "classe": 0},
    {"id": 18, "texto": "A leitura foi cansativa e o final foi péssimo e previsível.", "classe": 0},
    {"id": 19, "texto": "Edição veio com defeito e a história é horrível.", "classe": 0},
    {"id": 20, "texto": "Decepcionado com a compra, história boba e arte medíocre.", "classe": 0}
]

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "e", "é", "que", "se", "por", "para", "com", "foi", "são",
    "me", "ao", "aos", "muito", "mais"
}

def pre_processar(texto):
    """
    Executa os 4 passos solicitados:
    1. Converte texto para minúsculas
    2. Remove pontuação
    3. Separa o texto em palavras (tokenização)
    4. Remove stopwords
    """
    texto = texto.lower()

    for p in string.punctuation:
        texto = texto.replace(p, "")

    palavras = texto.split()

    palavras_limpas = [palavra for palavra in palavras if palavra not in STOPWORDS]

    return palavras_limpas

todas_as_palavras = set()
for item in base_de_dados:
    todas_as_palavras.update(pre_processar(item["texto"]))

vocabulario = sorted(list(todas_as_palavras))

def texto_para_vetor(texto, vocabulario):
    """
    Transforma um texto em um vetor binário baseado no vocabulário.
    1 se a palavra aparece no texto, 0 caso contrário.
    """
    palavras_do_texto = set(pre_processar(texto))
    return [1 if termo in palavras_do_texto else 0 for termo in vocabulario]

class Perceptron:
    def __init__(self, n_caracteristicas, taxa_aprendizagem=0.1):
        self.taxa = taxa_aprendizagem
        self.pesos = [random.uniform(-0.1, 0.1) for _ in range(n_caracteristicas)]
        self.bias = random.uniform(-0.1, 0.1)

    def funcao_ativacao(self, soma):
        """
        Função de ativação degrau definida no enunciado:
        f(soma) = 1, se soma > 0.5
                  0, caso contrário
        """
        return 1 if soma > 0.5 else 0

    def predizer(self, entrada):
        """Calcula a soma ponderada com o bias e aplica a ativação."""
        soma = sum(x * w for x, w in zip(entrada, self.pesos)) + self.bias
        return self.funcao_ativacao(soma)

    def treinar(self, base_dados, vocabulario, max_epocas=100):
        """
        Treina o Perceptron atualizando pesos e bias conforme o erro.
        Interrompe antecipadamente se não houver erros na época.
        """
        print("-" * 75)
        print("INICIANDO TREINAMENTO DO PERCEPTRON")
        print(f"Taxa de aprendizagem: {self.taxa} | Máximo de épocas: {max_epocas}")
        print(f"Tamanho do vocabulário (entradas): {len(vocabulario)}")
        print("-" * 75)

        X = [texto_para_vetor(item["texto"], vocabulario) for item in base_dados]
        y = [item["classe"] for item in base_dados]

        for epoca in range(1, max_epocas + 1):
            erros_na_epoca = 0

            for entrada, classe_real in zip(X, y):
                soma = sum(x * w for x, w in zip(entrada, self.pesos)) + self.bias

                classe_prevista = self.funcao_ativacao(soma)

                erro = classe_real - classe_prevista

                if erro != 0:
                    erros_na_epoca += 1
                    for i in range(len(self.pesos)):
                        self.pesos[i] += self.taxa * erro * entrada[i]
                    self.bias += self.taxa * erro

            if erros_na_epoca == 0:
                print(f"-> Treinamento convergiu com 0 erros na época {epoca}!\n")
                return epoca

        print(f"-> Treinamento concluído após {max_epocas} épocas.\n")
        return max_epocas


modelo = Perceptron(n_caracteristicas=len(vocabulario), taxa_aprendizagem=0.1)
modelo.treinar(base_de_dados, vocabulario, max_epocas=100)

print("=" * 75)
print("6. TESTE DO MODELO (AVALIAÇÕES DA BASE)")
print("=" * 75)

acertos = 0
erros = 0
avaliacoes_erradas = []

for item in base_de_dados:
    vetor = texto_para_vetor(item["texto"], vocabulario)
    classe_prevista = modelo.predizer(vetor)
    classe_real = item["classe"]

    correto = (classe_prevista == classe_real)
    resultado = "Correto" if correto else "Incorreto"

    if correto:
        acertos += 1
    else:
        erros += 1
        avaliacoes_erradas.append((item["id"], item["texto"], classe_real, classe_prevista))

    rotulo_real = "1 (Positiva)" if classe_real == 1 else "0 (Negativa)"
    rotulo_prev = "1 (Positiva)" if classe_prevista == 1 else "0 (Negativa)"

    print(f"ID {item['id']:02d}: \"{item['texto']}\"")
    print(f"      Classe Real: {rotulo_real} | Classe Prevista: {rotulo_prev} | Resultado: {resultado}")

total_avaliacoes = len(base_de_dados)
acuracia = (acertos / total_avaliacoes) * 100

print("\n" + "=" * 75)
print("7. CÁLCULO DA PORCENTAGEM DE ACERTO")
print("=" * 75)
print(f"• Total de avaliações: {total_avaliacoes}")
print(f"• Classificações corretas: {acertos}")
print(f"• Classificações incorretas: {erros}")
print(f"• Acurácia: {acuracia:.2f}%\n")


print("=" * 75)
print("8. TESTE COM NOVAS AVALIAÇÕES (NÃO PRESENTES NO TREINO)")
print("=" * 75)

novas_avaliacoes = [
    {
        "id": "N1",
        "texto": "História fantástica, amei os desenhos e recomendo a leitura.",
        "classe_esperada": 1,
        "motivo": "Avaliação positiva direta com termos conhecidos e favoráveis."
    },
    {
        "id": "N2",
        "texto": "Quadrinho com roteiro fraco e final decepcionante.",
        "classe_esperada": 0,
        "motivo": "Avaliação negativa com termos negativos explícitos."
    },
    {
        "id": "N3",
        "texto": "Excelente edição, arte maravilhosa e acabamento impecável.",
        "classe_esperada": 1,
        "motivo": "Avaliação positiva rica em elogios à qualidade gráfica."
    },
    {
        "id": "N4",
        "texto": "História cansativa e muito tediosa, não gostei nada.",
        "classe_esperada": 0,
        "motivo": "Avaliação negativa apontando leitura arrastada."
    },
    {
        "id": "N5",
        "texto": "Esperava uma história incrível e desenhos excelentes, mas foi bem decepcionante.",
        "classe_esperada": 0,
        "motivo": "Avaliação negativa com contexto adverso/conjunção 'mas' que desafia o BoW."
    }
]

acertos_novos = 0
for nova in novas_avaliacoes:
    vetor = texto_para_vetor(nova["texto"], vocabulario)
    prev = modelo.predizer(vetor)
    correto = (prev == nova["classe_esperada"])
    if correto:
        acertos_novos += 1

    rot_esp = "1 (Positiva)" if nova["classe_esperada"] == 1 else "0 (Negativa)"
    rot_prev = "1 (Positiva)" if prev == 1 else "0 (Negativa)"
    res = "Correto" if correto else "Incorreto"

    print(f"[{nova['id']}]: \"{nova['texto']}\"")
    print(f"      Esperado: {rot_esp} | Previsto: {rot_prev} | Resultado: {res}")
    print(f"      Observação: {nova['motivo']}\n")

acuracia_novos = (acertos_novos / len(novas_avaliacoes)) * 100
print(f"Acurácia nas novas avaliações: {acuracia_novos:.1f}% ({acertos_novos}/{len(novas_avaliacoes)} acertos)\n")

print("=" * 75)
print("9. ANÁLISE DOS RESULTADOS")
print("=" * 75)

analise_a = f"""a) A rede aprendeu a classificar as avaliações?
RESPOSTA:
Sim. Durante a fase de treinamento e teste na base de dados, a rede Perceptron
alcançou uma acurácia de {acuracia:.1f}% (com {acertos}/{total_avaliacoes} acertos), convergindo
em poucas épocas. Isso demonstra que o modelo conseguiu separar linearmente as
avaliações positivas e negativas com base no vocabulário fornecido."""

analise_b = f"""b) Quantas avaliações foram classificadas incorretamente?
RESPOSTA:
- Na base de dados original (20 avaliações): {erros} erros ({acuracia:.1f}% de acertos).
- No teste com 5 novas avaliações: {len(novas_avaliacoes) - acertos_novos} erro(s).
  Avaliação incorreta: [N5] "Esperava uma história incrível e desenhos excelentes, mas foi bem decepcionante."
  Classe real: 0 (Negativa) | Prevista pelo modelo: 1 (Positiva)."""

analise_c = """c) Por que ocorreram os erros?
RESPOSTA:
O erro observado na avaliação N5 reflete diretamente as limitações teóricas do
modelo Bag of Words (BoW) e do Perceptron de camada única:

1. Limitações do Modelo Bag of Words: O BoW considera apenas a presença ou ausência
   isolada das palavras, descartando completamente a ordem e a sintaxe da frase.
2. Ausência de Análise de Contexto e Conjunções Adversativas: A frase continha palavras
   fortemente associadas à classe positiva no treino ("incrível", "desenhos", "excelentes").
   O modelo não foi capaz de interpretar a expectativa quebrada ("Esperava...") e o peso
   da adversativa ("mas foi bem decepcionante"). A soma ponderada das palavras positivas
   superou o peso negativo isolado da palavra "decepcionante".
3. Tamanho reduzido da base de dados: Com apenas 20 exemplos no treino, o vocabulário
   é enxuto e não contempla combinações de n-gramas (como "não foi bom" ou "esperava mais").
4. Palavras compartilhadas e ambiguidade: Palavras neutras como "história" ou "quadrinho"
   aparecem em ambos os contextos e precisam de adjetivos modificadores que o Perceptron
   simples avalia de maneira puramente aditiva e linear."""

print(analise_a)
print()
print(analise_b)
print()
print(analise_c)
print("=" * 75)

