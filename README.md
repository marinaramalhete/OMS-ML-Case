# Case Técnico - Product DS

Repositório criado para resolução do seguinte case:

__Descrição do problema__
Você é um funcionário da OMS que deve avaliar os níveis de contaminação de um vírus em um determinado país. As pessoas dentro de uma sociedade podem estar conectadas de alguma maneira (familia, amizade ou trabalho) e cada pessoa possui um conjunto de atributos.
Este vírus afeta esta sociedade como descrito a seguir:
- a taxa de contaminação varia de pessoa para pessoa;
- a taxa de contaminação de uma pessoa A para B é diferente de B para A e depende
das características de ambas as pessoas (A e B);
- a contaminação só passa através de indivíduos conectados;
- não existe cura para essa doença;


__O desafio__
Foram coletados os dados de contaminação (ou seja, as taxas de contaminação) para metade desta sociedade. Neste problema, você deverá estimar a taxa para o restante dessa sociedade e decidir políticas de saúde com base nos resultados obtidos.

--------

# Comece por aqui

## Pré requisitos
Os pacotes necessários estão no arquivo `requirements.txt`.

## Instalação e execução
1. Clone este reporsitório.

2. Crie um ambiente virtual conda (ou o que preferir):

```
conda create --name ml-case python=3.8.12
conda activate ml-case
````

3. Instale os pacotes:
```
pip install -r requirements.txt
```

4. Inicie os notebooks:
```
jupyter notebook
```
--------

# Análise dos Dados

Para visualizar a análise feita, em `.pdf`, acesse `docs > Análise Exploratória dos Dados - Apresentação`