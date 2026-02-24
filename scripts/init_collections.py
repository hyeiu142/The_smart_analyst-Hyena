"""
Chạy 1 lần để khởi tạo 3 Qdrant collections.
Usage:
    cd /home/yennguyen/Hyena
    python scripts/init_collections.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper


def main():
    print("Initializing Qdrant collections...")
    qdrant = QdrantClientWrapper()
    qdrant.ensure_collections()

    print("\nDone! Collections info:")
    for name in [qdrant.TEXT_COLLECTION, qdrant.TABLE_COLLECTION, qdrant.IMAGE_COLLECTION]:
        info = qdrant.client.get_collection(name)
        print(f"  - {name}: {info.points_count} points")


if __name__ == "__main__":
    main()
