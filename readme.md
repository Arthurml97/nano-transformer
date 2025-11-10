# Nano-GPT: Construindo um Transformer do Zero

## 🤖 O que é este projeto? (Um breve resumo)

Este projeto é um "cérebro" de Inteligência Artificial que eu construí do zero para aprender a escrever.

Pense nele como um "Estudante Robô":
1.  **A Matéria:** Eu dei a ele um único livro para estudar (*A Story of the Golden Age*).
2.  **O Estudo:** O robô leu o livro letra por letra, milhares de vezes, até aprender a *prever* qual é a próxima letra mais provável em qualquer frase.
3.  **A Prova:** No final, ele ficou tão bom em adivinhar os padrões que agora consegue "escrever" seus próprios parágrafos que, embora não façam sentido completo, se *parecem* muito com o estilo do livro original.

A "Jornada" (Bebê, Adolescente, Adulto) foi o meu experimento científico para descobrir o "tamanho" de cérebro ideal para esse robô aprender a matéria sem só "decorar" o livro.

---

## 🚀 Arquitetura e Features

Este projeto é uma implementação de um modelo de linguagem Transformer (estilo GPT) em PyTorch, construído do zero para fins de estudo. O modelo é treinado em nível de caractere para gerar texto baseado em um corpus de entrada.

O foco principal deste repositório não é apenas o código final, mas a **jornada iterativa de engenharia** para construir um modelo que aprende de forma eficaz, mesmo em um ambiente de CPU limitado.

Este modelo é um "Transformer Decoder-Only" (a mesma arquitetura do GPT) e inclui:

* **Tokenização em Nível de Caractere**: O vocabulário é composto por todos os caracteres únicos do texto de entrada.
* **Embeddings de Token e Posição**: Para dar ao modelo o significado dos caracteres e seu senso de ordem.
* **Blocos Transformer**: O coração do modelo, empilhados `n_layer` vezes.
* **Multi-Head Self-Attention**: O mecanismo que permite ao modelo "prestar atenção" a diferentes partes do contexto para prever o próximo caractere.
* **Rede Feed-Forward**: Uma camada de "reflexão" para cada token processar a informação da atenção.
* **Conexões Residuais e Layer Normalization**: Essencial para estabilizar o treinamento em redes profundas.
* **Geração de Texto Autoregressiva**: O modelo usa sua própria saída como entrada para gerar texto novo.

---

## 🛠️ Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Arthurml97/nano-transformer.git](https://github.com/Arthurml97/nano-transformer.git)
    cd nano-transformer
    ```

2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Prepare os dados:**
    Coloque o seu arquivo de texto de treinamento na raiz do projeto com o nome `input.txt`.

4.  **Treine o modelo:**
    O script `nano.py` está configurado com hiperparâmetros otimizados para CPU.
    ```bash
    python nano.py
    ```

---

## 🔬 A Jornada: Do Overfitting ao Aprendizado

Este projeto foi uma experiência prática sobre o equilíbrio entre o tamanho do modelo, o tamanho dos dados e as limitações de hardware.

### Ponto de Partida: O Dataset Mínimo

Inicialmente, o projeto começou com um dataset muito pequeno (um breve resumo da lore de World of Warcraft). Nos estágios iniciais (quando o modelo era mais simples), ele funcionou.

No entanto, à medida que a arquitetura evoluiu para um Transformer completo, o modelo imediatamente "decorou" (overfit) esse dataset minúsculo em poucas iterações. Ele se tornou incapaz de aprender qualquer regra generalizável do idioma. Isso provou que, para um modelo mais complexo, um dataset maior não era opcional — era obrigatório. Foi feita então a troca para um corpus muito maior: o livro *A Story of the Golden Age*.

### O Desafio da CPU: O Teste de 10.7M de Parâmetros

Antes de otimizar o modelo para a CPU, foi realizado um teste crucial: "O que acontece se o modelo otimizado para GPU (10.7M de parâmetros) for treinado na minha CPU (Ryzen 5 5600)?"

O resultado foi, como esperado, **improdutivo**. O treinamento levou **mais de 12 horas** para gerar uma resposta básica e com overfitting severo.

Esse teste provou que uma abordagem "força bruta" não era viável. A solução seria começar do zero, com um modelo pequeno o suficiente para a CPU, e otimizá-lo iterativamente.

### A Estratégia: Forçando a Generalização (Bottom-Up)

A estratégia mudou para: "Qual é o modelo *mais inteligente* que eu consigo treinar na minha CPU *em um tempo razoável*?"

O processo foi feito em três estágios, aumentando o "cérebro" do modelo a cada passo:

#### 1. "Bebê Transformer" (0.2M de parâmetros)

* **Config:** `n_embd=64`, `n_head=4`, `n_layer=4`
* **Resultado:** `val loss ~2.04`. O modelo gerou um "Inglês-Fantasma"—texto que tinha a *forma* do inglês (espaços, pontuação, finais como "ing"), mas sem palavras reais. **Sucesso!** A generalização estava acontecendo.

#### 2. "Adolescente Transformer" (0.8M de parâmetros)

* **Config:** `n_embd=128`, `n_head=4`, `n_layer=4`
* **Resultado:** `val loss ~1.90`. O modelo, com mais capacidade, começou a gerar palavras reais do livro, como "Hellas", "Neleus" e "Iphig's".

#### 3. "Adulto Transformer" (1.2M de parâmetros)

* **Config:** `n_embd=128`, `n_head=6`, `n_layer=6`, `dropout=0.2`
* **Resultado:** `val loss` mínimo de **1.88**. Este foi o modelo mais inteligente. Ele atingiu seu pico de aprendizado por volta de `step 3500` e depois começou a overfitar.

| Modelo | Parâmetros | Melhor Val Loss | Texto Gerado (Exemplo) |
| :--- | :--- | :--- | :--- |
| Babê | 0.2M | 2.0423 | `...intoring Ithaca. he made Gram unthis...` |
| Adolescente | 0.8M | 1.9059 | `...were Neleus to the olders of Mount Iphig’s...` |
| Adulto | 1.2M | **1.8842** | `...said Phemius," "and away bless wrookly upon the dutyings...` |

### 💡 Conclusão da Jornada

Este projeto foi uma demonstração prática de que:
1.  **Hardware Limita o Design**: A falha no teste de 12 horas na CPU forçou uma abordagem de design de modelo "de baixo para cima" (bottom-up), focada em eficiência.
2.  **O Overfitting é Visível**: Ao monitorar o `val loss`, foi possível identificar *exatamente* quando o modelo parou de aprender e começou a decorar (por volta de `step 3500-4000` nos modelos maiores).
3.  **O Nível de Caractere Aprende Estrutura**: Mesmo sem saber o que é uma "palavra", o Transformer aprendeu regras de sintaxe, pontuação e formação de palavras do texto de entrada.

---
## 📊 Google Colab e 3 Million Dataset

Os experimentos anteriores provaram que o hardware (CPU) e o conjunto de dados (um único livro) eram os gargalos.

Esta branch leva o projeto à sua conclusão lógica:
1.  **Hardware:** O treinamento será movido para o Google Colab para usar uma GPU T4.
2.  **Modelo:** Vou usar o "super-cérebro" de 14.5M de parâmetros (ativando os hiperparâmetros de CUDA).
3.  **Dados:** O `input.txt` será expandido para uma Obra de Tolkien.

O objetivo é, finalmente, treinar um modelo onde o `val loss` *diminua* de forma estável, provando que a arquitetura BPE é viável quando recebe os recursos adequados.

## 📜 Créditos

Este código foi desenvolvido como parte de um estudo aprofundado do repositório [nanoGPT](https://github.com/karpathy/nanoGPT) de Andrej Karpathy, adaptado para um ambiente de CPU e focado na análise iterativa de hiperparâmetros.
