import os
import torch
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from .data_utils import extract_text_from_pdf 


def initialize_model(model_name=None, model_path=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # bfloat16 시 연산병목 발생
    )
    if model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
    elif model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto"
        )
    return tokenizer, model


def create_vector_store(
        train_data, 
        embedding_model_name,
        pdf_folder,
):
    print("🔄 벡터 DB 생성 중...")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    train_documents = [
        f"Q: {q}\\nA: {a}" for q, a in zip(train_data["question"], train_data["answer"])
    ]

    # pdf_folder가 주어진 경우, pdf 파일에서 텍스트 추출
    pdf_documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            pdf_documents.append(f"[{pdf_file}] 문서 내용:\n{text}")
            
    documents = train_documents + pdf_documents

    return FAISS.from_texts(documents, embedding)


def create_qa_chain(
    vector_store,
    model,
    tokenizer,
    prompt_template,
    search_model,
    search_k_num=5,
    max_new_tokens=64,
):
    retriever = vector_store.as_retriever(
        search_model=search_model, search_kwargs={"k": search_k_num}
    )
    text_gen_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.1,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
