from src.tokenizer import tokenizers
from src.TokenizerEmbedding import tokenizer_embedding
from src.PositionalEncoding import position
from fastapi import FastAPI

app = FastAPI()

@app.post("/")
def main(content:str):
    tok = tokenizers(content)
    tok_emb = tokenizer_embedding(tok)
    # posi = position(tok_emb)
    
    return {"return":tok}


if __name__ == "__main__":
    main()
