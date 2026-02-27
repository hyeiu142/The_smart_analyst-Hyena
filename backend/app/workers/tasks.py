import asyncio
from typing import Any, Dict
from backend.app.workers.celery_app import celery_app

def _run_async(coro):
    """Run coroutine from Celery (sync context)"""
    loop = asyncio.new_event_loop()
    try: 
        return loop.run_until_complete(coro)
    finally:        
        loop.close()

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    name="tasks.process_document",
)

def process_document_task(
    self,
    doc_id: str,
    file_path: str,
    metadata: Dict[str, Any],
):
    """
    Background Celery task: 
    1. Call IngestionPipeline.process_document() to process the document and store chunks in Qdrant.
    2. Update status about document store
    """
    from backend.app.core.ingestion.pipeline import IngestionPipeline
    from backend.app.api.v1.documents import update_doc_status

    # Translate Docker path → local path if worker runs outside container
    # Docker: /project/Docs/xxx.pdf → Local: /home/yennguyen/Hyena/Docs/xxx.pdf
    import os
    if not os.path.exists(file_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        local_path = file_path.replace("/project/", project_root + "/")
        if os.path.exists(local_path):
            file_path = local_path

    try:
        update_doc_status(doc_id, "processing")
        print(f"[Celery] Starting ingestion for doc_id={doc_id}")

        pipeline = IngestionPipeline()
        result = _run_async(
            pipeline.process_document(
                file_path=file_path,
                doc_id=doc_id,
                metadata=metadata,
            )
        )

        update_doc_status(doc_id, "completed", result=result)
        print(f"[Celery] Completed doc_id={doc_id} with result: {result}")
        return result
    except Exception as exc:
        error_msg = str(exc)
        print(f"[Celery] Failed doc_id={doc_id}: {error_msg}")
        update_doc_status(doc_id, "failed", error=error_msg)
        raise self.retry(exc=exc)