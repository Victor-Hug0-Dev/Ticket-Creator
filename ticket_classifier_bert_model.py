from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

label_mapping = {
    0: "Suporte Técnico",
    1: "Devoluções e Trocas",
    2: "Faturamento e Pagamentos", 
    3: "Vendas e pré-vendas",
    4: "Interrupções de serviço e manutenção",
    5: "Suporte ao produto",
    6: "Suporte de TI",
    7: "Atendimento ao Cliente",
    8: "Recursos Humanos",
    9: "Consulta Geral"
}

def predict_fila(input):
    inputs = tokenizer(input, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=1).item()    
 
    label_name = label_mapping.get(prediction, "Desconhecido")
    
    return {
        "prediction": prediction, 
        "fila": label_name
    }

# Carrega o modelo e tokenizer
model_path = "./bert_finetuned_classification"
model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Texto de exemplo
#sample_text = 'meu teclado está com problema poderia mandar alguém do setor para arrumar?'
sample_text = """Você poderia fornecer mais informações sobre serviços de análise de dados
para otimização de investimentos? Sua assistência oportuna é importante para nós.
"""

result = predict_fila(sample_text)
print(result)