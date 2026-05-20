
#langchain document loading and splitting
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = PROJECT_ROOT / "docs"


def _define_text_splitter():

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size = 512,
        chunk_overlap=100,
    )

def get_default_doc_paths()-> tuple[Path, Path, Path, Path]:
   
    research_workbench = DOC_PATH / "research_workbench_overview.md"
    model_card = DOC_PATH / "model_card.md"
    evaluation_card = DOC_PATH / "evaluation_card.md"
    data_dictionary = DOC_PATH / "data_dictionary.md"

    return research_workbench, model_card, evaluation_card, data_dictionary

def load_markdown_or_text_file(file_path:Path)->Document:

    file_content = file_path.read_text(encoding="utf-8")

    return Document(
        page_content=file_content,
        metadata={
            "source": str(file_path),
            "file_name": file_path.name,
            "document_type": file_path.suffix.replace(".", ""),
        },
    )

def load_pdf_file(pdf_file)->list[Document]:
    pdf_loader = PyMuPDFLoader(str(pdf_file))
    return pdf_loader.load()

def load_document(file_path: Path)->list[Document]:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found at path {file_path}")

    file_extension = file_path.suffix.lower()

    if file_extension in ['.md', '.txt']:
        return [load_markdown_or_text_file(file_path=file_path)]
    
    if file_extension == ".pdf":
        return load_pdf_file(pdf_file=file_path)
    
    raise ValueError (f"The file type {file_extension} is not supported")

def split_documents (documents:list[Document])->list[Document]:

    
    text_splitter = _define_text_splitter()

    chunks = text_splitter.split_documents(documents=documents)

    return chunks

def ingest_documents(file_paths=None)->list[Document]:

    chunks = []

    if file_paths is None:
        file_paths = get_default_doc_paths()

    for file_path in file_paths:
        documents = load_document(file_path=file_path)

        chunk = split_documents(documents=documents)

        chunks.extend(chunk)
    
    return chunks
