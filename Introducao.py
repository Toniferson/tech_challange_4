import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(layout='wide')

#Importacao e tratamento da base
dados = pd.read_csv('https://raw.githubusercontent.com/Toniferson/tech_challange_4/main/dados/ipeadata%5B19-04-2024-09-47%5D.csv',sep = ';')
dados['Data'] = pd.to_datetime(dados['Data'],format='%d/%m/%Y')
dados = dados.set_index('Data')
dados = dados.rename(columns={'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366':'Preço - Petróleo Bruto'})
dados = dados.drop(columns='Unnamed: 2')
dados['Preço - Petróleo Bruto'] = dados['Preço - Petróleo Bruto'].str.replace(',','.').astype(float)
dados = dados.fillna(0)

#Tabelas
dados_anuais = dados.groupby(pd.Grouper(freq = 'YE'))['Preço - Petróleo Bruto'].mean()

#Graficos
fig_evolutivo_brent = px.line(dados_anuais)

#Aplicacao
st.image('dados/petrobras.jpg_2090796243-scaled.jpg')
st.markdown("<h1 style='text-align: center; color: orange;'>Tech Challange - Fase 4</h1>", unsafe_allow_html=True)
st.write('Para um time multidisciplinar qualquer desafio é uma oportunidade de unir forças alcançar novas oportunidades :rocket:')
st.divider()

tab1, tab2= st.tabs(['Introdução', 'Base de Dados'])
with tab1:  
    st.markdown("<h2 style='text-align: left; color: lightblue;'>O Desafio<h2>", unsafe_allow_html=True)
    st.write('''Como entrega final para esta fase, foi proposto um trabalho de analisar os dados de preço do pretróleo Brent. Para esta entrega sera feito um dashboard interativo para a geração de insights além de um modelo de Machine Learning com a previsão do preço do Brent
                ''')
    st.markdown("<h2 style='text-align: left; color: lightblue;'>O que é 'Brent'<h2>", unsafe_allow_html=True)
    st.write('''
             Antes de iniciarmos nossas analises é muito importante que se explique o que é o Brent, que será analisado neste trabalho. O nome “Brent” para o petróleo foi criado por uma política interna da Shell. Originalmente, a Shell denominava seus campos de produção com nomes de aves, e no caso do campo de petróleo Brent, o nome foi inspirado no ganso de Brent. Esse campo de petróleo está localizado no Mar do Norte, próximo à costa da Escócia, e foi descoberto em 1971. A produção de petróleo no campo de Brent começou em 1976. Desde então, o Brent se tornou uma referência internacional para o preço do petróleo bruto e é negociado nos mercados globais
                ''')
    st.markdown("<h2 style='text-align: left; color: lightblue;'>Como os preços são determinados?'<h2>", unsafe_allow_html=True)
    st.write('''Os preços do petróleo no mercado internacional são influenciados por uma combinação complexa de fatores. Primeiramente, a lei da oferta e demanda desempenha um papel crucial. O petróleo é uma commodity, caracterizada por baixa industrialização, e seu preço é ajustado com base na demanda global. Quando a demanda por petróleo supera a oferta, os preços tendem a subir, e vice-versa.
Além disso, os países produtores têm um impacto significativo nos preços. Os três maiores produtores de petróleo (Arábia Saudita, Rússia e Estados Unidos) representam cerca de 40% do mercado mundial. Suas decisões sobre produção e exportação afetam diretamente os preços do petróleo.
Eventos geopolíticos e tensões regionais também influenciam o mercado. Conflitos no Oriente Médio, sanções e questões diplomáticas podem afetar a oferta e a estabilidade do mercado petrolífero. Além do preço do petróleo bruto, outros fatores como custos de refino, impostos e margens de lucro das empresas também determinam os preços dos combustíveis.
Por fim, os preços do petróleo são frequentemente associados a marcadores globais, como o Brent (extraído no Mar do Norte) e o WTI (preferido nos EUA). Esses marcadores influenciam desde o custo da gasolina até o preço de produtos derivados do petróleo.
''')
    st.image('dados/preco do brent.jpg',width = 700)

with tab2:
    st.markdown("<h2 style='text-align: left; color: lightblue;'>Time Series<h2>", unsafe_allow_html=True)
   
    col1, col2 = st.columns(2)
    with col1:
        st.write('A base na qual estamos trabalhando pode ser encontrada no site do IPEA - Instituto de Pesquisa Econômica Aplicada e tem como fonte das informações a "Energy Information Administration (EIA)" ')
        st.write('A tabela que é disponibilizada possui duas colunas, uma de data e uma outra com o valor do preço do pretóleo bruto (Brent), podemos já tirar alguns insights só olhando a base brevemente, como por exemplo que tal estrutura pode ser tratada como uma Série Temporal, encaixando com alguns modelos previamente estudado')
    with col2:
        st.dataframe(dados.head(15))
        st.markdown(f'A tabela possui :blue[{dados.shape[0]}] linhas e :blue[{dados.shape[1]}] colunas')
    st.divider()
    st.markdown("<h2 style='text-align: left; color: lightblue;'>Evolutivo das médias dos valores anual<h2>", unsafe_allow_html=True)
    st.plotly_chart(fig_evolutivo_brent,use_container_width = True)

    
