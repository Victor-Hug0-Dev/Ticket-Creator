from summarizer import Summarizer,TransformerSummarizer

body = '''
Em 2016 coloquei como uma das metas do ano
"Aprender a fazer um bom nhoque", mas foi só no final de 2018 que
finalmente fiz um nhoque com cara e sabor de nhoque.
Um prato que eu pensei "Eu pagaria por isso em um restaurante.
Não pagaria muito caro, mas pagaria". E considerando meus talentos gastronômicos,
pra mim isso foi uma baita conquista,
que só foi possível porque eu me empenhei muito mais do que nos anos anteriores.
Em um mês eu fiz mais nhoques (e tentativas de nhoques) do que a soma de todas as tentativas dos dois anos anteriores.
Eu aprendi empiricamente que a repetição constante é um importante hábito para aprendermos a fazer algo que exige técnica,
tal como escrever... Que é uma das minhas metas de 2019 :)
'''

body = ''' Querido diário,
eles dizem que o que eu escrevo é romanticamente dolorido.
Mas por favor não deixe que vejam atrás das minhas pálpebras todos os coágulos de sangue
que guardam as bonitas palavras que se abraçam.
Por favor, você sabe que não vai ser tão bonito quando souberem que a poesia sobe e corta a ponta dos meus dedos e sussurra baixo que devo sangrar para se tornar arte.

*

fui na central das estrelas e me disseram que infelizmente não podia retirar as que você
beijava no meu pescoço.
mas parei para olhar meu corpo nu e percebi que desde que saí da barriga de minha mãe
as estrelas se abrigaram no meu corpo como um presente para lembrar que antes de você,
elas já eram amadas
por mim.

*

nos olhamos e o vento estava forte o bastante para que meu cabelo cobrisse e harmonizasse
todas as nossas lágrimas
nos olhamos e você me disse que dói, eu te abracei e disse que em mim também dói, mas que
o fim estava ali diante de nós.
desde então tenho deixado meu cabelo crescer, e a vitamina está no couro, para onde nossas
lágrimas foram
para onde elas não podem se separar, para onde elas podem dançar juntas.

 '''

bert_model = Summarizer()
bert_summary = ''.join(bert_model(body, min_length=128))
print(bert_summary)

