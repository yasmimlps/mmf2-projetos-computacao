import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Funções Utilitárias
# ==============================

def carregar_dados(caminho_csv):
    """Carrega o CSV e retorna um DataFrame pandas."""
    return pd.read_csv(caminho_csv, sep=';', encoding='utf-8', engine='python')


def filtrar_projetos_computacao(df):
    """Filtra projetos relacionados à computação."""
    keywords = ['computação', 'informática', 'ciência da computação', 'engenharia de computação',
                'sistemas embarcados', 'algoritmo', 'inteligência artificial', 'programação', 'imd', 'ccet']

    # Padroniza texto
    for col in ['unidade', 'area_conhecimento_cnpq', 'palavras_chave', 'linha_pesquisa']:
        df[col] = df[col].astype(str).str.lower()

    mask = (
        df['unidade'].str.contains('|'.join(keywords)) |
        df['area_conhecimento_cnpq'].str.contains('|'.join(keywords)) |
        df['palavras_chave'].str.contains('|'.join(keywords)) |
        df['linha_pesquisa'].str.contains('|'.join(keywords))
    )
    return df[mask].copy()


def contar_projetos_por_ano(df):
    """Conta a quantidade de projetos por ano."""
    contagem = df['ano'].value_counts().sort_index()
    df_ano = pd.DataFrame({
        'ano': contagem.index.astype(int),
        'quantidade_projetos': contagem.values
    })
    return df_ano


def aplicar_regressao_linear(df_ano):
    """Aplica regressão linear e retorna modelo, intervalo de confiança e R²."""
    df_ano['ano'] = df_ano['ano'].astype(int)
    X = sm.add_constant(df_ano['ano'])
    y = df_ano['quantidade_projetos']
    modelo = sm.OLS(y, X).fit()
    media = y.mean()
    intervalo = stats.t.interval(confidence=0.95, df=len(y)-1, loc=media, scale=stats.sem(y))
    return modelo, intervalo, media


def gerar_grafico_evolucao(df_ano, media, output_path):
    """Gera gráfico de dispersão + regressão + média."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='ano', y='quantidade_projetos', data=df_ano)
    sns.regplot(x='ano', y='quantidade_projetos', data=df_ano, scatter=False, color='red', label='Regressão linear')
    plt.axhline(media, color='green', linestyle='--', label='Média dos projetos')
    plt.title('Evolução Temporal dos Projetos de Computação na UFRN')
    plt.xlabel('Ano')
    plt.ylabel('Quantidade de Projetos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ==============================
# Execução do processo completo
# ==============================

caminho_csv = "../data/projetos-de-pesquisa.csv"
saida_csv_filtrado = "../data/projetos_computacao_ufrn.csv"
saida_grafico = "../data/evolucao_projetos_computacao.png"

# Processamento
df_raw = carregar_dados(caminho_csv)
df_filtrado = filtrar_projetos_computacao(df_raw)
df_filtrado.to_csv(saida_csv_filtrado, index=False)

df_ano = contar_projetos_por_ano(df_filtrado)
modelo, intervalo, media = aplicar_regressao_linear(df_ano)
gerar_grafico_evolucao(df_ano, media, saida_grafico)

# Exportar resultados
{
    "modelo_coeficientes": modelo.params.to_dict(),
    "r2": modelo.rsquared,
    "p_valores": modelo.pvalues.to_dict(),
    "intervalo_confianca_media": intervalo,
    "csv_filtrado": saida_csv_filtrado,
    "grafico_gerado": saida_grafico
}
