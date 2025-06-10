# test_rag.py
from src.rag_qa import RAGQA

rag = RAGQA(
    pdf_path="data/General Notes.pdf",
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
)

print(rag.ask("What’s the HTS Number for donkeys?"))
