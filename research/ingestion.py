import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

def process_financial_report(file_path, company, year, quarter):
    print(f"Processing: {company} {year} {quarter}")

    parser = LlamaParse(
        api_key = LLAMA_CLOUD_API_KEY,
        result_type = "markdown",
        #target_pages = "0,1,2",
        extract_images = True, 
        image_output_dir = f"../Docs/extracted_images/{company}",
        language = "en", 
        verbose = True
    )

    documents = parser.load_data(file_path)

    for doc in documents: 
        doc.metadata.update({
            "company": company, 
            "year": year,
            "quarter": quarter
        })

    # Use SentenceSplitter instead of MarkdownElementNodeParser
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(documents)

    return nodes


if __name__ == "__main__":
    data_sources = [
        {
        "path": "/home/yennguyen/Hyena/Docs/20260130_VNM_IR_Newsletter_Q4_2025_68b565f20d.pdf",
        "company": "Vinamilk",
        "year": 2025,
        "quarter": "Q4"
        }
    ]

    all_nodes = []
    for source in data_sources: 
        nodes = process_financial_report(
            source["path"],
            source["company"],
            source["year"],
            source["quarter"]
        )
        all_nodes.extend(nodes)

    print("Building index...")
    index = VectorStoreIndex(all_nodes)
    
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("What are Vinamilk's product types?")
    print(response)
    


   