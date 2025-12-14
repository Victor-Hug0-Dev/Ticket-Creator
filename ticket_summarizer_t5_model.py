from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Verificar se há GPU disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model_checkpoint = "stjiris/t5-portuguese-legal-summarization"
t5_model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
t5_tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

body = '''Em 2016 coloquei como uma das metas do ano
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

t5_prepared_Text = "summarize: "+body

# Mover modelo e tokens para o device correto
t5_model = t5_model.to(device)
tokenized_text = t5_tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

# summarizar
summary_ids = t5_model.generate(tokenized_text,
                                num_beams=4,
                                no_repeat_ngram_size=2,
                                min_length=515,
                                max_length=1028,
                                early_stopping=True)

output = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\n\nSummarized text: \n", output)