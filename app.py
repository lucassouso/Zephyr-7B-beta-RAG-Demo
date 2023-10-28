from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
import gradio as gr
import ctransformers


local_llm = "zephyr-7b-beta.Q4_K_M.gguf"

config = {
'max_new_tokens': 1024,
'repetition_penalty': 1.1,
'temperature': 0.1,
'top_k': 50,
'top_p': 0.9,
'stream': True,
'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="cuda",
    gpu_layers=100,
    **config
)
print("LLM Initialized...")


prompt_template = """Use as informações dadas na base de conhecimento vinda do PDF, não responda nada que não estiver lá.
Traga sempre o contexto de onde tirou a resposta.

Context: {context}
Question: {question}

Responda sempre em português brasileiro com resposta útil e direta.
Resposta útil:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/bula_dipirona", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})
# query = "O que preciso saber antes de tomar dipirona?"
# semantic_search = retriever.get_relevant_documents(query)
# print(semantic_search)

print("######################################################################")

chain_type_kwargs = {"prompt": prompt}



sample_prompts = ["Quais são os efeitos colaterais da dipirona?", "O que devo saber antes de tomar dipirona?"]

def get_response(input):
  query = input
  chain_type_kwargs = {"prompt": prompt}  
  try:
    qa = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=retriever,
      return_source_documents = True,
      chain_type_kwargs= chain_type_kwargs,
      verbose=True
    )
    response = qa(query)
  except StopIteration:
    response = None
  return response


input = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Faça sua pergunta",
                container=False,
            )

iface = gr.Interface(fn=get_response, 
             inputs=input, 
             outputs="text",
             title="Sir_Bulario Bot",
             description="Conversando com minha bula PDF.",
             examples=sample_prompts,
             allow_flagging='never'
             )


iface.launch(share=True)










            







