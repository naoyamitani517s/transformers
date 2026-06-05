from transformers import AutoTokenizer


token = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenizers(content: str):
    tok = token(content, return_tensors="pt")
    print(f'input_ids: {tok["input_ids"].tolist()}')
    print(f'token_type_ids: {tok["token_type_ids"].tolist()}')
    print(f'attention_mask: {tok["attention_mask"].tolist()}')
    return tok