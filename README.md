# Controle de Banca (Python + Streamlit)

Sistema web local para controle de banca, com registro de ganhos/perdas diárias, histórico de operações e gráfico de evolução.

> Esta versão está identificada como **Vtest**.

## Funcionalidades

- Informar banca inicial
- Informar número de dias do mês
- Cálculo automático de valor disponível por dia
- Dois modos de valor disponível por dia:
  - banca inicial por dia no início do dia
- Lucro do dia em % com indicador visual (verde positivo / vermelho negativo) e valor financeiro do dia
- Painel visual de saldo disponível do dia com mudança de cor:
  - saldo disponível (dia) = base inicial do dia ± lucro/prejuízo do dia
  - verde/glow quando o dia está positivo
  - vermelho/glow quando o dia está negativo
  - dourado/glow quando atinge meta diária
  - vermelho/preto quando atinge stop loss diário
- Registro diário de lucros e perdas
- Saldo atualizado da banca
- Histórico completo de entradas
- Total ganho, total perdido e saldo líquido
- Total ganho do dia e prejuízo do dia
- % de lucro do dia (positivo ou negativo)
- % de lucro da semana, mês e ano
- Painéis de % lucro (dia/semana/mês/ano) com verde/glow para positivo e vermelho/glow para negativo
- Limite de perda diária (stop loss) em %
- Meta de lucro diária em %
- Alertas ao atingir stop loss ou meta
- Salvamento local automático em banco SQLite (`banca.db`)
- Tabela de operações com remoção por ID
- Campo de data e descrição da operação
- Gráfico simples de evolução da banca
- Integração com API não oficial da IQ Option para importar operações da conta
- Gráfico detalhado de desempenho por hora/dia/semana/mês/ano
- Botão para ocultar/exibir valores sensíveis (máscara `***`)

## Requisitos

- Python 3.10+
- Bibliotecas:
  - `streamlit`
  - `pandas`
  - `plotly`
  - `iqoptionapi` (instalada via GitHub)

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Como rodar

```bash
streamlit run app.py
```

Depois, abra no navegador o endereço mostrado no terminal (normalmente `http://localhost:8501`).

## Persistência dos dados

Todos os dados ficam salvos localmente no arquivo `banca.db`, criado automaticamente na primeira execução.

## Estrutura do projeto

```text
.
├── app.py
├── requirements.txt
└── README.md
```

## Sincronização com IQ Option

1. Na barra lateral, preencha **Email IQ Option**, **Senha IQ Option** e o limite de operações para busca.
2. Escolha o tipo de conta em **Conta para sincronização** (`REAL` ou `PRACTICE`).
3. Clique em **Sincronizar operações da IQ Option**.
4. O sistema importa operações fechadas disponíveis e adiciona no histórico com origem `iqoption`.
5. Operações já importadas não são duplicadas (controle por `source + external_id`).

> Observação: esta integração usa a API não oficial do projeto no GitHub informado e depende dos métodos disponibilizados pela biblioteca instalada.
