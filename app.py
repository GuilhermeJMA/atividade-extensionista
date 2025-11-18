import streamlit as st
import pandas as pd
import unicodedata
import os
import plotly.express as px
from prophet import Prophet
import glob
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Dashboard de Despesas", layout="wide")

# caminho do arquivo
ARQUIVO = "dados/Relatorio.txt" 

# -------------------------
# FUNÇÕES AUXILIARES
# -------------------------
def normalize(col: str) -> str:
    if not isinstance(col, str):
        return col
    s = unicodedata.normalize("NFKD", col)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = s.replace(" - ", " ").replace("-", " ").replace("/", " ").replace(".", "")
    s = " ".join(s.split())
    return s

def find_col(df_cols, target_norm):
    for c in df_cols:
        if normalize(c) == target_norm:
            return c
    return None

def fmt_real(v):
    try:
        return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "R$ 0,00"

# -------------------------
# LEITURA DO ARQUIVO
# -------------------------
if not os.path.exists(ARQUIVO):
    st.error(f"Arquivo não encontrado: {ARQUIVO}")
    st.stop()

df = pd.read_csv(ARQUIVO, sep=";", encoding="latin1", quotechar='"')
df.columns = [c.strip() for c in df.columns]

# -------------------------
# MAPEAMENTO DE COLUNAS
# -------------------------
expected = {
    "orcado inicial": None,
    "orcado atualizado": None,
    "empenhado no mes": None,
    "empenhado ate o mes": None,
    "liquidado no mes": None,
    "liquidado ate o mes": None,
    "pago no mes": None,
    "pago ate o mes": None,
    "funcao descricao": None,
    "subfuncao descricao": None,
    "descricao categoria economica": None,
    "orgao": None,
}

for k in list(expected.keys()):
    expected[k] = find_col(df.columns, k)

# -------------------------
# CONVERSÃO NUMÉRICA
# -------------------------
numeric_cols_norm = [
    "orcado inicial", "orcado atualizado",
    "empenhado no mes", "empenhado ate o mes",
    "liquidado no mes", "liquidado ate o mes",
    "pago no mes", "pago ate o mes"
]

for ncol in numeric_cols_norm:
    orig = expected.get(ncol)
    if orig is None or orig not in df.columns:
        continue

    series = df[orig].astype(str).str.strip()
    sample = series.dropna().astype(str).head(20).tolist()
    uses_comma_decimal = any(
        ((',' in s) and (s.count(',') >= 1) and (s[-3] == ',' or s[-2] == ','))
        for s in sample if s and s not in ['nan', 'None']
    )
    if uses_comma_decimal:
        series = series.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    df[orig] = pd.to_numeric(series, errors="coerce").fillna(0.0)


# --- LOGO E TÍTULO ---
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    st.image("logo.png", width=150) 
with col_titulo:
    st.markdown(
    """
    <h1 style='text-align: center; margin-top: 10px; color: #2c3e50;
               text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>
        Dashboard Contas Públicas Guaramirim
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
aba_dashboard, aba_previsoes = st.tabs(["📊 Dashboard", "📈 Previsões"])

with aba_dashboard:
    # -------------------------
    # FILTROS
    # -------------------------
    st.markdown("### 🔎 Filtros")
    col1, col2, col3 = st.columns(3)

    # Mapeia colunas esperadas
    func_col = expected.get("funcao descricao")
    subfunc_col = expected.get("subfuncao descricao")
    cat_col = expected.get("descricao categoria economica")

    # Garante que as colunas existem
    df_filtros = df.copy()

    # === FILTRO 1: FUNÇÃO ===
    func_options = sorted(df_filtros[func_col].dropna().unique()) if func_col in df_filtros.columns else []
    with col1:
        func_sel = st.selectbox("Função", ["Todas"] + func_options) if func_options else "Todas"

    if func_sel != "Todas":
        df_filtros = df_filtros[df_filtros[func_col] == func_sel]

    # === FILTRO 2: SUBFUNÇÃO (dependente da função) ===
    subfunc_options = (
        sorted(df_filtros[subfunc_col].dropna().unique())
        if subfunc_col in df_filtros.columns else []
    )
    with col2:
        subfunc_sel = st.selectbox("Subfunção", ["Todas"] + subfunc_options) if subfunc_options else "Todas"

    if subfunc_sel != "Todas":
        df_filtros = df_filtros[df_filtros[subfunc_col] == subfunc_sel]

    # === FILTRO 3: CATEGORIA ECONÔMICA (dependente dos anteriores) ===
    cat_options = (
        sorted(df_filtros[cat_col].dropna().unique())
        if cat_col in df_filtros.columns else []
    )
    with col3:
        cat_sel = st.selectbox("Categoria Econômica", ["Todas"] + cat_options) if cat_options else "Todas"

    if cat_sel != "Todas":
        df_filtros = df_filtros[df_filtros[cat_col] == cat_sel]

    df_filtrado = df.copy()
    if func_col in df.columns and func_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado[func_col] == func_sel]
    if subfunc_col in df.columns and subfunc_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado[subfunc_col] == subfunc_sel]
    if cat_col in df.columns and cat_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado[cat_col] == cat_sel]

    # -------------------------
    # CÁLCULOS
    # -------------------------
    def get_sum(norm_name):
        orig = expected.get(norm_name)
        if orig and orig in df_filtrado.columns:
            return df_filtrado[orig].sum()
        return 0.0

    totais = {
        "Orçado Inicial": get_sum("orcado inicial"),
        "Empenhado no Mês": get_sum("empenhado no mes"),
        "Liquidado no Mês": get_sum("liquidado no mes"),
        "Pago no Mês": get_sum("pago no mes"),
        "Orçado Atualizado": get_sum("orcado atualizado"),
        "Empenhado até o Mês": get_sum("empenhado ate o mes"),
        "Liquidado até o Mês": get_sum("liquidado ate o mes"),
        "Pago até o Mês": get_sum("pago ate o mes"),
    }

    # -------------------------
    # CARDS
    # -------------------------
    st.markdown("### 💰 Totais")
    labels = list(totais.keys())

    # Função para criar cada card com sombra e ícone
    def render_card(icon, label, value, color):
        return f"""
        <div style="
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            padding: 15px 10px;
            text-align: center;
            transition: all 0.2s ease-in-out;
            height: 120px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;  /* 👈 espaçamento vertical entre os cards */
        "
        onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.25)';"
        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.15)';"
        >
            <div style="font-size: 26px; color: {color}; margin-bottom: 5px;">{icon}</div>
            <div style="font-size: 15px; font-weight: 600; color: #333;">{label}</div>
            <div style="font-size: 18px; font-weight: bold; color: {color}; margin-top: 6px;">{fmt_real(value)}</div>
        </div>
        """

    # Ícones e cores
    icons = [
        ("📘", "#1565C0"),  # Orçado Inicial
        ("🧾", "#00897B"),  # Empenhado no Mês
        ("📦", "#6A1B9A"),  # Liquidado no Mês
        ("💰", "#2E7D32"),  # Pago no Mês
        ("📊", "#0277BD"),  # Orçado Atualizado
        ("🧮", "#F57C00"),  # Empenhado até o Mês
        ("💳", "#8E24AA"),  # Liquidado até o Mês
        ("🏦", "#43A047"),  # Pago até o Mês
    ]

    # Primeira linha
    cols1 = st.columns(4, gap="large")
    for i, col in enumerate(cols1):
        with col:
            st.markdown(render_card(icons[i][0], labels[i], totais[labels[i]], icons[i][1]), unsafe_allow_html=True)

    # Segunda linha
    cols2 = st.columns(4, gap="large")
    for i, col in enumerate(cols2):
        with col:
            st.markdown(render_card(icons[i+4][0], labels[i+4], totais[labels[i+4]], icons[i+4][1]), unsafe_allow_html=True)



    # -------------------------
    # EXECUÇÃO ORÇAMENTÁRIA
    # -------------------------
    orc_atual = totais["Orçado Atualizado"]
    pago_ate = totais["Pago até o Mês"]
    if orc_atual and orc_atual > 0:
        perc = (pago_ate / orc_atual) * 100
        execucao_info = f"💡 **Execução orçamentária:** {perc:.2f}%"
        st.info(f"💡 Percentual de execução orçamentária: **{perc:.2f}%**")
    else:
        execucao_info = "💡 Execução orçamentária: não disponível"

    # -------------------------
    # GRÁFICO PRINCIPAL (ordenado)
    # -------------------------
    st.divider()
    

    st.markdown("##### Top 3 Por Função")

    group_col = func_col
    

    if group_col and group_col in df_filtrado.columns:
        y_cols = [expected.get(n) for n in ["empenhado ate o mes", "liquidado ate o mes", "pago ate o mes"] if expected.get(n)]
        graf_df = df_filtrado.groupby(group_col)[y_cols].sum().reset_index()

        graf_df["Total"] = graf_df[y_cols].sum(axis=1)
        graf_df = graf_df.sort_values("Total", ascending=False)

        icons_rank = ["🥇", "🥈", "🥉"]
        colors_rank = ["#DAA520", "#C0C0C0", "#CD7F32"]  # ouro, prata, bronze
        top3 = graf_df.head(3)

        cols = st.columns(3)

        for i in range(3):
            with cols[i]:
                st.markdown(
                    render_card(
                        icon=icons_rank[i],
                        label=str(top3.iloc[i][group_col]),
                        value=top3.iloc[i]["Total"],
                        color=colors_rank[i]
                    ),
                    unsafe_allow_html=True
                )
        st.markdown("### 📈 Execução Financeira por Função")
        fig = px.bar(
        graf_df,
        x=group_col,
        y=y_cols,
        barmode="group",
        title="Execução Financeira por Função",
        labels={"variable": "Indicador"} 
    )

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Coluna de função não encontrada — gráfico não exibido.")

    # -------------------------
    # GRÁFICO DE ROSCA + EXECUÇÃO
    # -------------------------
    st.divider()
    st.markdown("### 📊 Análises Complementares")
    col_g1, col_sep, col_g2 = st.columns([1, 0.05, 1])

    with col_g1:
        metrica_sel = st.radio(
            "Selecione a métrica para análise:",
            ["Pago até o Mês", "Liquidado até o Mês", "Empenhado até o Mês"],
            horizontal=True,
            key="radio_subfuncao"
        )

        sel_col = expected.get(normalize(metrica_sel))
        subfunc_col = expected.get("subfuncao descricao")

        if sel_col and subfunc_col and sel_col in df_filtrado.columns:
            pie_df = df_filtrado.groupby(subfunc_col)[sel_col].sum().reset_index()
            pie_df = pie_df.sort_values(sel_col, ascending=False)

            # Pega top 5 e agrupa o resto como "Outros"
            top5 = pie_df.head(5)
            outros_valor = pie_df[sel_col].iloc[5:].sum()
            if outros_valor > 0:
                outros_df = pd.DataFrame({subfunc_col: ["Outros"], sel_col: [outros_valor]})
                pie_df = pd.concat([top5, outros_df], ignore_index=True)

            fig_pie = px.pie(
                pie_df,
                names=subfunc_col,
                values=sel_col,
                hole=0.5,
                title=f"{metrica_sel} por Subfunção (Top 5 + Outros)"
            )
            fig_pie.update_traces(
            textinfo="label+percent+value",
            textposition="outside",
            textfont=dict(size=12),
            pull=[0.02] * len(pie_df),  # dá um leve destaque aos segmentos
            showlegend=False
        )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Não foi possível gerar o gráfico de rosca por Subfunção.")

    with col_sep:
        st.markdown(
            """
            <div style="border-left: 2px solid #DDD; height: 400px; margin: auto;"></div>
            """,
            unsafe_allow_html=True
        )

    with col_g2:

        metrica_sel_cat = st.radio(
            "Selecione a métrica para análise:",
            ["Pago até o Mês", "Liquidado até o Mês", "Empenhado até o Mês"],
            horizontal=True,
            key="radio_categoria"
        )

        sel_cat = expected.get(normalize(metrica_sel_cat))
        categoria_col = expected.get("descricao categoria economica")

        if sel_cat and categoria_col and sel_cat in df_filtrado.columns:
            cat_df = df_filtrado.groupby(categoria_col)[sel_cat].sum().reset_index()
            cat_df = cat_df.sort_values(sel_cat, ascending=False)

            # Pega top 5 e agrupa o resto como "Outros"
            top5 = cat_df.head(5)
            outros_valor = cat_df[sel_cat].iloc[5:].sum()
            if outros_valor > 0:
                outros_df = pd.DataFrame({categoria_col: ["Outros"], sel_cat: [outros_valor]})
                cat_df = pd.concat([top5, outros_df], ignore_index=True)

            fig_cat = px.pie(
                cat_df,
                names=categoria_col,
                values=sel_cat,
                hole=0.5,
                title=f"{metrica_sel} por Categoria Economica (Top 5 + Outros)"
            )
            fig_cat.update_traces(
            textinfo="label",
            textposition="outside",
            textfont=dict(size=12),
        )

            fig_cat.update_traces(
            textinfo="label",
            textposition="outside",
            textfont=dict(size=12)
        )
            fig_cat.update_traces(
                textinfo="percent+value",
                textposition="inside"
            )

            fig_cat.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Não foi possível gerar o gráfico de rosca por Categoria Economica.")

    # -------------------------
    # TABELA DETALHADA
    # -------------------------
    st.divider()
    st.markdown("### 📋 Dados Detalhados")
    st.dataframe(df_filtrado, use_container_width=True)

with aba_previsoes:
    st.markdown("## 📈 Previsão de Gastos")

    pasta_historico = "dados/historico" 

    arquivos = sorted(glob.glob(os.path.join(pasta_historico, "*.txt")))

    if not arquivos:
        st.warning("Nenhum arquivo encontrado na pasta de histórico.")
    else:
        # Função auxiliar para mapear mês/ano do nome do arquivo
        def extrair_mes_ano(nome_arquivo):
            nome = os.path.basename(nome_arquivo).split(".")[0]
            meses = {
                "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
                "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12
            }
            mes = nome[:3].capitalize()
            ano = int("20" + nome[-2:])
            return datetime(ano, meses.get(mes, 1), 1)

        dados_mensais = []

        for arq in arquivos:
            try:
                df_mes = pd.read_csv(arq, sep=";", encoding="latin1", quotechar='"')
                df_mes.columns = [col.strip().lower() for col in df_mes.columns]
                col_pago = expected.get("pago ate o mes")

                if col_pago and col_pago.lower() in df_mes.columns:
                    total_pago = df_mes[col_pago.lower()].astype(str).str.replace(",", ".").astype(float).sum()
                    dados_mensais.append({
                        "mes": extrair_mes_ano(arq),
                        "valor_pago": total_pago
                    })

            except Exception as e:
                st.error(f"Erro ao ler {arq}: {e}")

        # -------------------
        # Geração da previsão
        # -------------------
        if len(dados_mensais) >= 3:
            df_hist = pd.DataFrame(dados_mensais)
            df_hist["mes"] = pd.to_datetime(df_hist["mes"], errors="coerce")
            df_hist = (
                df_hist.groupby(df_hist["mes"].dt.to_period("M"), as_index=False)
                .last()
                .rename(columns={"mes": "periodo"})
            )

            # Agora converte de Period → Timestamp corretamente
            df_hist["mes"] = df_hist["periodo"].astype("period[M]").dt.to_timestamp()
            df_hist.drop(columns=["periodo"], inplace=True)

            # Dados para Prophet
            df_prophet = df_hist.rename(columns={"mes": "ds", "valor_pago": "y"})

            modelo = Prophet()
            modelo.fit(df_prophet)

            futuro = modelo.make_future_dataframe(periods=2, freq="M")
            previsao = modelo.predict(futuro)

            # 🔹 Mantém apenas o último registro de cada mês
            previsao["mes"] = previsao["ds"].dt.to_period("M").dt.to_timestamp()
            previsao = previsao.groupby("mes", as_index=False).last()

            # -------------------
            # Gráfico
            fig = px.line(
            previsao,
            x="mes",
            y="yhat",
            title="📈 Projeção de Gastos Futuros",
            labels={"mes": "Período", "yhat": "Valor (R$)"},
            )

            # Linha de previsão
            fig.update_traces(
                line=dict(color="orange", width=3),
                mode="markers+lines",
                marker=dict(size=6),
                name="Previsão",
            )

            # 🔹 Adiciona a linha do histórico (azul)
            fig.add_scatter(
                x=df_hist["mes"],
                y=df_hist["valor_pago"],
                mode="markers+lines",
                name="Histórico Real",
                line=dict(color="royalblue", width=3),
                marker=dict(size=6),
            )

            fig.update_layout(
                legend_title="Legenda",
                template="plotly_white",
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Previsão do próximo mês formatada
            proximo_mes = previsao.tail(1)[["mes", "yhat"]].iloc[0]
            valor_formatado = f"{proximo_mes['yhat']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            st.success(f"💡 Previsão para {proximo_mes['mes'].strftime('%b/%Y')}: R$ {valor_formatado}")

            df_prev = previsao[previsao["mes"] > df_hist["mes"].max()][["mes", "yhat"]].copy()

            # ------------------------------------------------------------
            # 🔵 PREVISÃO POR FUNÇÃO
            # ------------------------------------------------------------
            st.markdown("### 🔍 Previsão por Função")

            col_funcao = expected.get("funcao descricao")

            if col_funcao is None:
                st.error("A coluna 'funcao descricao' não foi encontrada nos arquivos.")
            else:
                # ====== 1) LER OS MESMOS ARQUIVOS E AGRUPAR POR FUNÇÃO ======
                funcoes_mensais = []

                for arq in arquivos:
                    try:
                        df_mes = pd.read_csv(arq, sep=";", encoding="latin1", quotechar='"')
                        df_mes.columns = [c.strip().lower() for c in df_mes.columns]

                        if col_funcao.lower() in df_mes.columns and col_pago.lower() in df_mes.columns:
                            df_mes[col_pago.lower()] = (
                                df_mes[col_pago.lower()]
                                .astype(str)
                                .str.replace(",", ".")
                            ).astype(float)

                            df_mes["mes"] = extrair_mes_ano(arq)

                            # somar POR FUNÇÃO
                            df_group = (
                                df_mes.groupby([ "mes", col_funcao.lower() ])[col_pago.lower()]
                                .sum()
                                .reset_index()
                                .rename(columns={col_funcao.lower(): "funcao", col_pago.lower(): "valor_pago"})
                            )

                            funcoes_mensais.append(df_group)

                    except Exception as erro:
                        st.error(f"Erro ao processar {arq}: {erro}")

                if funcoes_mensais:
                    df_funcoes_full = pd.concat(funcoes_mensais, ignore_index=True)

                    lista_funcoes = sorted(df_funcoes_full["funcao"].unique())

                    escolha = st.selectbox("Selecione uma função:", lista_funcoes)

                    df_func = df_funcoes_full[df_funcoes_full["funcao"] == escolha].copy()

                    # ====== 2) Preparar histórico ====== 
                    # Primeiro, converte mes para Period antes de agrupar
                    df_func["mes_period"] = df_func["mes"].dt.to_period("M")

                    # Agrupa e soma apenas valor_pago
                    df_func_grouped = (
                        df_func.groupby("mes_period", as_index=False)["valor_pago"]
                        .sum()
                    )

                    # Converte Period de volta para Timestamp
                    df_func_grouped["mes"] = df_func_grouped["mes_period"].dt.to_timestamp()
                    df_func_grouped = df_func_grouped.drop(columns=["mes_period"])

                    if len(df_func_grouped) >= 3:
                        # ====== 3) Rodar prophet específico ====== 
                        df_prophet_f = df_func_grouped.rename(columns={"mes": "ds", "valor_pago": "y"})
                        modelo_f = Prophet()
                        modelo_f.fit(df_prophet_f)
                        
                        futuro_f = modelo_f.make_future_dataframe(periods=2, freq="M")
                        prev_f = modelo_f.predict(futuro_f)
                        prev_f["mes"] = prev_f["ds"].dt.to_period("M").dt.to_timestamp()
                        prev_f = prev_f.groupby("mes", as_index=False).last()
                        
                        # ====== 4) GRÁFICO PLOTLY ====== 
                        fig_f = px.line(
                            prev_f,
                            x="mes", 
                            y="yhat",
                            title=f"📈 Previsão para a Função: {escolha}",
                            labels={"mes": "Mês", "yhat": "Valor (R$)"}
                        )
                        
                        # linha previsão
                        fig_f.update_traces(
                            line=dict(color="orange", width=3),
                            mode="markers+lines",
                            marker=dict(size=6),
                            name="Previsão"
                        )
                        
                        # linha histórico
                        fig_f.add_scatter(
                            x=df_func_grouped["mes"],
                            y=df_func_grouped["valor_pago"],
                            mode="markers+lines",
                            name="Histórico Real",
                            line=dict(color="royalblue", width=3),
                            marker=dict(size=6),
                        )
                        
                        st.plotly_chart(fig_f, use_container_width=True)
                        
                        # ====== 5) Exibir previsão final ====== 
                        ultimo = prev_f.tail(1)[["mes", "yhat"]].iloc[0]
                        val = f"{ultimo['yhat']:,.2f}".replace(".", ",")
                        st.success(f"📌 Previsão para **{escolha}** em {ultimo['mes'].strftime('%b/%Y')}: **R$ {val}**")
                    else:
                        st.info("São necessários pelo menos 3 meses dessa função para gerar previsão.")


        else:
            st.info("São necessários pelo menos 3 meses de histórico para gerar previsões.")

    st.warning("⚠️ Lembre-se: previsões são estimativas baseadas em dados históricos e podem não refletir com precisão os resultados futuros reais.")
    st.info("""
    ---
    ## 🧠 CURIOSIDADE: 

    ### Neste projeto, foi utilizado o algoritmo Prophet.

    Ele é um algoritmo de previsão de séries temporais desenvolvido pelo Facebook (Meta), criado especialmente para lidar com dados reais que possuem sazonalidade, tendência e variações ao longo do tempo.

    ### 🔍 O que o Prophet faz com os dados?

    O Prophet decompõe a série temporal em três componentes principais:

    **1. Tendência (trend)**  
    Representa o comportamento geral dos valores ao longo do tempo — crescimento, queda ou estabilidade.  
    O Prophet detecta automaticamente mudanças na tendência sem precisar de configuração extra.

    **2. Sazonalidade (seasonality)**  
    O modelo identifica padrões que se repetem em ciclos, como:  
    - comportamento mensal  
    - comportamento anual  
    - padrões semanais  
    - se existe algum padrão que se repete todo mês ou todo ano  
    - se houve algum mês atípico (outlier)

    **3. Feriados e eventos (holidays)**  
    (Quando configurado) o Prophet considera feriados e eventos que impactam os valores.

    ---

    ### 📈 Como o Prophet gera a previsão?

    1. Lê os valores históricos (datas + indicadores).  
    2. Ajusta um modelo com tendência, sazonalidade e mudanças estruturais.  
    3. Projeta o comportamento futuro com base nos padrões aprendidos.  

    ---

    ### 📌 Fórmula simplificada que o Prophet usa: 
    \[ y(t) = g(t) + s(t) + h(t) + \epsilon_t \] 
            
    Onde: 
    - **g(t)** = tendência 
    - **s(t)** = sazonalidade 
    - **h(t)** = efeitos extras (como feriados) 
    - **ε** = ruído/erro

    ---

    ### 🎯 Por que usei Prophet neste projeto?

    - Lida bem com dados financeiros e orçamentários.  
    - Suporta dados faltantes e séries curtas.  
    - É excelente para prever valores mensais de pagamentos e gastos.
    """)

