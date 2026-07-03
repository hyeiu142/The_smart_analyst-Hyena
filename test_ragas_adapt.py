# /// script
# requires-python = ">=3.12"
# dependencies = ["ragas>=0.4.0", "langchain-openai>=0.3.0", "langchain-community>=0.3.0"]
# ///
import sys
from unittest.mock import MagicMock
_mock_vertexai = MagicMock()
_mock_vertexai.ChatVertexAI = MagicMock
sys.modules.setdefault("langchain_community.chat_models.vertexai", _mock_vertexai)

import inspect
from ragas.metrics import Faithfulness

print("Faithfulness methods:")
f = Faithfulness()
print([m for m in dir(f) if not m.startswith('_')])
