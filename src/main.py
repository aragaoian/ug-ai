from model import Chatbot

ignored_docs = [
    "edital_2-2025.md",
    "novas_inscrições.md",
    "renovação_total.md",
    "renovação_parcial.md",
    "faq.md",
]

headers_to_split_on = [
    ("#", "Section"),
    ("##", "Subsection"),
    ("###", "Specific topic"),
]

chatbot = Chatbot(
    chunk_size=0,
    chunk_overlap=0,
    ignored_docs=ignored_docs,
    md_headers_list=headers_to_split_on,
)

query = "O que acontece se eu não seguir as cláusulas da CAFE?"
ans = chatbot.chain_function(query)
print(ans)
