from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.core import VectorStoreIndex


llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_url = "/home/yennguyen/Hyena/Docs/20260130_VNM_IR_Newsletter_Q4_2025_68b565f20d.pdf"  # also allowed is a file path e.g. /home/downloads/xyz.pdf
pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
documents = pdf_loader.load_data(pdf_url)


index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("what is the revenue of Vinamilk in Q4 2025?")
print(response)

