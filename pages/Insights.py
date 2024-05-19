import streamlit as st


st.image('dados/petrobras.jpg_2090796243-scaled.jpg')
st.markdown("<h1 style='text-align: center; color: orange;'>Insights</h1>", unsafe_allow_html=True)
st.write('''O petróleo é uma commodity, caracterizada por baixa industrialização, e seu preço é ajustado com base na demanda global. Quando a demanda por petróleo supera a oferta, os preços tendem a subir, e vice-versa.
Além disso, os países produtores têm um impacto significativo nos preços.''')
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Variações do Preço','2008-2009','2016','2020','Geopolitica e Conflitos'])

with tab1:
    st.markdown("<h2 style='text-align: left; color: lightblue;'>Quais são os principais eventos que podem afetar o preço do petróleo?<h2>", unsafe_allow_html=True)
    st.write('''**Interrupção da oferta:** Guerras podem interromper a produção e o transporte de petróleo, especialmente se ocorrerem em regiões ricas em petróleo. Isso pode levar a uma diminuição da oferta global, o que, por sua vez, pode aumentar os preços.

**Incerteza e especulação do mercado:** A incerteza em torno do resultado de uma guerra e suas implicações para a produção de petróleo pode levar à especulação no mercado de petróleo. Isso pode resultar em flutuações de preços.

**Sanções econômicas:** As sanções econômicas impostas a países em guerra podem afetar a capacidade desses países de exportar petróleo. Isso pode reduzir a oferta global e aumentar os preços.

Além disso, três pontos importantes entram na conta para saber o real impacto sobre os preços: a extensão do conflito, a duração do conflito e como as empresas petrolíferas se comportam frente às oscilações de preços.
''')
    
with tab2:
    st.markdown("<h2 style='text-align: left; color: lightblue;'>2008 - 2009<h2>", unsafe_allow_html=True)
    st.write('''O mercado do petróleo em 2008 foi marcado por dois atos dramáticos. No primeiro semestre, os preços dispararam para recordes quase diários, atingindo o pico de 147,50 dólares o barril em julho. Esse aumento foi impulsionado por uma combinação de fatores, incluindo tensões geopolíticas, aumento da demanda de países emergentes e especulação de fundos de investimento. No entanto, após a falência do Lehman Brothers em setembro, os preços despencaram rapidamente devido ao medo da deflação e à queda no consumo nos países industrializados. O ano terminou com preços muito menores, e as previsões para 2009 variavam, mas muitos acreditavam que o petróleo se tornaria mais caro devido à falta de investimento na indústria petrolífera e ao custo crescente de produção. Em suma, 2008 foi um ano de extremos para o mercado do petróleo, com uma rápida mudança de altos recordes para preços minguados, levando a incertezas sobre o futuro do setor.
 
A crise imobiliária de 2008-2009 foi desencadeada principalmente pelo colapso do mercado de hipotecas subprime nos Estados Unidos. Hipotecas subprime eram empréstimos concedidos a mutuários com histórico de crédito ruim ou sem condições financeiras sólidas para pagar as hipotecas. Muitos desses empréstimos foram agrupados em títulos financeiros complexos conhecidos como "CDOs" (Collateralized Debt Obligations), que foram vendidos para investidores em todo o mundo.
Quando os mutuários começaram a enfrentar dificuldades para pagar suas hipotecas, especialmente quando as taxas de juros começaram a subir, houve um aumento nas execuções hipotecárias e uma queda nos preços dos imóveis. Isso resultou em grandes perdas para os bancos e outras instituições financeiras que possuíam esses ativos. O colapso do mercado imobiliário desencadeou uma crise financeira global, com repercussões significativas em várias economias ao redor do mundo.
A crise imobiliária teve um impacto significativo na economia global. A queda nos preços dos imóveis reduziu a riqueza das famílias e minou a confiança dos consumidores, levando a uma redução nos gastos e investimentos. Isso, por sua vez, contribuiu para a recessão global, com aumento do desemprego e contração econômica em muitos países.
Quanto ao petróleo, a crise financeira de 2008-2009 teve um impacto duplo. Por um lado, a desaceleração econômica global reduziu a demanda por petróleo, já que as empresas reduziram a produção e os consumidores reduziram os gastos. Isso levou a uma queda nos preços do petróleo, já que a oferta excedia a demanda.
Por outro lado, a crise também afetou os investimentos em projetos de exploração e produção de petróleo, especialmente em áreas de alto custo, como as areias betuminosas do Canadá ou o pré-sal no Brasil. A falta de acesso ao crédito e a incerteza econômica levaram muitas empresas a reduzir seus investimentos em novos projetos de petróleo, o que poderia limitar a oferta no futuro e contribuir para aumentos de preços quando a demanda se recuperasse.
Em suma, a crise imobiliária de 2008-2009 teve um impacto profundo na economia global, reduzindo a demanda por petróleo devido à desaceleração econômica e também afetando os investimentos no setor petrolífero.
2014: é o pior tombo de preços desde 2008, causada pelo aumento de produção, em especial nas áreas de xisto dos EUA, uma demanda menor que a esperada na Europa e na Ásia e a recusa dos países da Organização dos Países Exportadores de Petróleo (OPEP) em reduzir seu teto de produção, independentemente do preço no mercado internacional. Com a superprodução, oriunda sobretudo do petróleo de xisto dos EUA, os membros da OPEP, visando não perder mercado, mantiveram suas produções afim de tornar insustentável a produção norte-americana. Alguns países sofrem particularmente com a redução dos preços do petróleo, sobretudo Venezuela, Rússia e Irã, em razão do grande peso das exportações da commodity em suas economias
''')

with tab3:
    st.markdown("<h2 style='text-align: left; color: lightblue;'>2016<h2>", unsafe_allow_html=True)
    st.write('''Em 2016, o mercado global de petróleo enfrentou uma série de desafios que moldaram sua dinâmica, enquanto eventos específicos no Brasil também influenciaram o setor petrolífero do país.
Globalmente, o mercado de petróleo estava sob pressão devido a um excesso de oferta persistente, impulsionado pelo aumento da produção de petróleo de xisto nos Estados Unidos e pela decisão da OPEP (Organização dos Países Exportadores de Petróleo) de manter altos níveis de produção para proteger sua participação de mercado. Essa situação resultou em uma queda significativa nos preços do petróleo bruto.
Além disso, as condições econômicas globais estavam instáveis, com preocupações sobre o crescimento econômico, especialmente na China e na Europa, afetando a demanda por petróleo e contribuindo para o excesso de oferta.
Em setembro de 2016, a OPEP anunciou um acordo preliminar para reduzir a produção de petróleo, buscando estabilizar os preços do petróleo. Essa foi uma virada crucial, pois marcou a primeira vez em vários anos que a OPEP concordou em cortar sua produção.
A eleição presidencial nos Estados Unidos em novembro de 2016 também teve um impacto nos preços do petróleo. A vitória de Donald Trump gerou expectativas de um aumento na demanda por petróleo devido às suas promessas de expansão econômica e investimentos em infraestrutura.
No Brasil, em 2016, a crise econômica e política estava em pleno curso, com o impeachment da presidente Dilma Rousseff em agosto. A crise política gerou incertezas sobre a estabilidade do país, afetando a confiança dos investidores e impactando indiretamente o setor de petróleo.
A Operação Lava Jato, uma investigação de corrupção que envolveu empresas estatais, incluindo a Petrobras, teve um impacto significativo no setor de petróleo do Brasil. As revelações de corrupção e má gestão na Petrobras levaram a uma queda na produção da empresa e afetaram sua capacidade de investimento em novos projetos.
Durante o governo de Michel Temer, foram implementadas reformas no setor de petróleo, incluindo mudanças nas regras de exploração do pré-sal, visando atrair mais investimentos estrangeiros e estimular a produção de petróleo no país.
Portanto, os eventos políticos, econômicos e regulatórios tanto globalmente quanto no Brasil em 2016 desempenharam um papel significativo na dinâmica do mercado de petróleo, influenciando a produção, os preços e os investimentos no setor.
''')

with tab4:
    st.markdown("<h2 style='text-align: left; color: lightblue;'>2020<h2>", unsafe_allow_html=True)
    st.write('''Em 2020, o mercado global de petróleo foi profundamente impactado pela pandemia de COVID-19, resultando em uma crise sem precedentes. Aqui estão alguns dos principais eventos e tendências que moldaram a dinâmica do setor petrolífero durante esse período:
Colapso da demanda: Com o surgimento da pandemia, medidas de bloqueio e restrições de viagens foram implementadas em todo o mundo para conter a propagação do vírus. Isso levou a uma queda abrupta na demanda por petróleo, já que indústrias, transportes e atividades comerciais foram interrompidos. O colapso na demanda foi particularmente pronunciado no segundo trimestre de 2020, quando os preços do petróleo despencaram para níveis historicamente baixos.
Guerra de preços entre Arábia Saudita e Rússia: Em março de 2020, a Arábia Saudita e a Rússia entraram em uma disputa sobre os cortes na produção de petróleo. A Arábia Saudita decidiu aumentar sua produção e oferecer descontos significativos aos compradores, inundando o mercado com petróleo. Isso exacerbou o excesso de oferta e contribuiu para uma queda acentuada nos preços do petróleo.
**Acordo de corte de produção da OPEP+: Em abril de 2020, a OPEP+ (que inclui membros da OPEP e outros produtores liderados pela Rússia) concordou em implementar cortes recordes na produção de petróleo, em uma tentativa de estabilizar os preços e equilibrar o mercado. No entanto, esses cortes foram insuficientes para compensar a queda na demanda causada pela pandemia.
Armazenamento saturado: Com a queda na demanda e o excesso de oferta, os mercados de armazenamento de petróleo ficaram rapidamente saturados. Isso levou a uma situação sem precedentes em que os compradores estavam enfrentando a perspectiva de não ter onde armazenar o petróleo que compraram, resultando em preços negativos no contrato futuro de petróleo WTI em abril de 2020.
Recuperação gradual: Ao longo de 2020, à medida que os países começaram a flexibilizar as restrições e a demanda por petróleo começou a se recuperar, os preços do petróleo também se recuperaram gradualmente. No entanto, a recuperação foi desigual e sujeita a incertezas, especialmente devido a preocupações com novas ondas da pandemia e seu impacto na demanda global.
Em resumo, a pandemia de COVID-19 desencadeou uma crise sem precedentes no mercado global de petróleo em 2020, com uma queda acentuada na demanda, excesso de oferta, preços negativos e uma luta entre os principais produtores para equilibrar o mercado. A recuperação subsequente foi desafiadora e sujeita a incertezas contínuas sobre a evolução da pandemia e seus efeitos na demanda global por petróleo.
''')

with tab5:
    st.markdown("<h2 style='text-align: left; color: lightblue;'>Geopolítica e conflitos<h2>", unsafe_allow_html=True)
    st.write('''**11 de setembro de 2001**
O 11 de setembro de 2001 ficou marcado como um dos dias mais trágicos da história moderna. Dois aviões comerciais sequestrados por terroristas atingiram as duas torres do World Trade Center, na cidade de Nova York, e um terceiro o Pentágono, sede do Departamento de Defesa americano, sediado em Washington, capital dos Estados Unidos. Cerca de três mil pessoas morreram nos ataques, incluindo as vítimas do quarto avião.''')
    st.write('''**Instabilidade e Medo:*** 
             O ataque gerou incerteza e medo nos mercados financeiros. Investidores buscaram ativos considerados mais seguros, como o ouro e o petróleo, levando a um aumento na demanda por petróleo bruto.''')
    st.write('''**Interrupções na Produção e Transporte:** O ataque afetou as operações de produção e transporte de petróleo. As torres do World Trade Center abrigavam importantes empresas do setor, e a destruição dessas instalações teve impacto direto na infraestrutura petrolífera.''')
    st.write('''**Geopolítica e Tensões Regionais:** O 11 de Setembro intensificou as tensões geopolíticas no Oriente Médio. A região é uma das principais produtoras de petróleo, e qualquer ameaça à estabilidade afeta os preços. Além disso, a Guerra ao Terror que se seguiu aos ataques aumentou a preocupação com a segurança das rotas de transporte de petróleo.
Outros conflitos geopolíticos se estenderam com o foco em domínio de território, como a invasão do Iraque em 2003. A invasão liderada pelos EUA no Iraque também foi influenciada pelo acesso às enormes reservas de petróleo do país. Além das alegações sobre armas de destruição em massa, a abertura dessas reservas ao capital estrangeiro motivou a ofensiva americana. O petróleo também era usado por Saddam Hussein para financiar sua influência estratégica na região''')
    st.write('''**Primavera Árabe:**
A Primavera Árabe foi um período de revoltas e protestos que começou em 2010 e se espalhou por vários países do Oriente Médio e do norte da África. Esses eventos tiveram um impacto significativo nos preços do petróleo. 
A Primavera Árabe, um período de revoltas e protestos que começou em 2010, teve um impacto significativo nos preços do petróleo. A redução da produção na Líbia, um importante produtor de petróleo, devido aos conflitos e instabilidade política, resultou em redução da oferta e contribuiu para aumentos significativos nos preços do petróleo. 
Além disso, a incerteza política e os conflitos em outros países da região também afetaram a produção e o transporte de petróleo, levando a oscilações nos preços. Os investidores consideraram a região do Oriente Médio e do norte da África como de alto risco, o que influenciou os mercados de energia e contribuiu para a volatilidade nos preços do petróleo.
''')   