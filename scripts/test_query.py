import asyncio
import sys
sys.path.append("/home/yennguyen/Hyena")

from backend.app.core.generation.rag_engine import RAGEngine


async def main():
    engine = RAGEngine()

    questions = [
        "What was the revenue in Q4 2025?",
        "Doanh thu Q4 2025 là bao nhiêu?",
        "How did gross margin change compared to last year?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")

        result = await engine.query(q, top_k=3)

        print(f"A: {result['answer']}")
        print(f"\nSources ({len(result['sources'])} chunks):")
        for i, src in enumerate(result["sources"], 1):
            meta = src["metadata"]
            print(f"  [{i}] Score: {src['score']:.4f} | "
                  f"Page: {meta.get('page')} | "
                  f"Content: {src['content'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())