from transformers import AutoModel


model = AutoModel.from_pretrained("bert-base-uncased")

def tokenizer_embedding(tok:object):
    tok_emb = model(**tok) # tok["input_ids"], tok["token_type_ids"], tok["attention_mask"]を展開
    print(f'tokenizerEmbedding_last_hidden_state: {tok_emb["last_hidden_state"]}')
    print(f'tokenizerEmbedding_pooler_output: {tok_emb["pooler_output"]}')
    return tok_emb