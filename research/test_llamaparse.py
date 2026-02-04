import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, Settings

load_dotenv(os.path.join(os.getcwd(), '.env')) 
api_key = os.getenv("LLAMA_CLOUD_API_KEY")

parser = LlamaParse(
    api_key=api_key, 
    result_type="markdown",
    verbose=True
)


print("--- Đang bóc tách dữ liệu ---")
documents = parser.load_data("/home/yennguyen/Hyena/Docs/test_1.xlsx")


output_path = "test1_sheet_0.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(documents[0].text)

print(f"--- Đã bóc tách xong! Kết quả lưu tại: {output_path} ---")