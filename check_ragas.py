# /// script
# requires-python = ">=3.12"
# dependencies = ["ragas>=0.4.0"]
# ///
from ragas.metrics import Faithfulness
print("Faithfulness required:", Faithfulness().required_columns)
