from src.vector_store import create_vector_store
from src.config import COLLECTION_NAME

if __name__ == "__main__":
    print(f"Building four-document corpus in {COLLECTION_NAME} ...")
    create_vector_store(batch_size=12)
    print("Completed successfully, baseline and original files have not been changed.")
