# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ragas>=0.2.0",
#     "langchain-openai",
#     "langchain-community<0.3.0"
# ]
# ///

from ragas.evaluation import EvaluationResult

result = EvaluationResult(scores=[{"faithfulness": 1.0}], dataset=None)
print(type(result))
try:
    print(dict(result))
except Exception as e:
    print("dict error:", e)
