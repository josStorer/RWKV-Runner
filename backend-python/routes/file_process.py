import os
from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Depends,
    File,
    UploadFile,
)
from pydantic import BaseModel
from typing import Iterator

router = APIRouter()


class FileToTextParams(BaseModel):
    file_name: str
    file_encoding: str = "utf-8"


@router.post("/file-to-text", tags=["File Process"])
async def file_to_text(
    params: FileToTextParams = Depends(), file_data: UploadFile = File(...)
):
    from langchain.schema import Document
    from langchain.document_loaders.blob_loaders import Blob

    # from langchain
    def parse_text(blob: Blob) -> Iterator[Document]:
        yield Document(page_content=blob.as_string(), metadata={"source": blob.source})

    # from langchain
    def parse_pdf(blob: Blob) -> Iterator[Document]:
        import fitz

        with blob.as_bytes_io() as stream:
            doc = fitz.Document(stream=stream)

            yield from [
                Document(
                    page_content=page.get_text(),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.number,
                            "total_pages": len(doc),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc
            ]

    file_parsers = {".txt": parse_text, ".pdf": parse_pdf}

    file_name = file_data.filename or params.file_name
    file_ext = os.path.splitext(file_name)[-1]

    if file_ext not in file_parsers:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file type not supported")

    try:
        pages: Iterator[Document] = file_parsers[file_ext](
            Blob.from_data(
                await file_data.read(),
                encoding=params.file_encoding,
                path=file_name,
            )
        )
        pages = list(pages)
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{e}")

    return {"pages": pages}
