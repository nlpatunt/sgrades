import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class LocalFileStorage:
    def __init__(self, base_storage_path: str = os.getenv("FILE_STORAGE_PATH", "./stored_submissions")):
        self.base_path = Path(base_storage_path)
        self.base_path.mkdir(exist_ok=True)

    def generate_file_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def generate_storage_path(
        self, dataset_name: str, submitter_email: str, original_filename: str, timestamp: datetime
    ) -> str:
        email_hash = hashlib.md5(submitter_email.encode()).hexdigest()[:8]
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        storage_dir = self.base_path / str(timestamp.year) / f"{timestamp.month:02d}" / dataset_name / email_hash

        storage_dir.mkdir(parents=True, exist_ok=True)

        clean_filename = "".join(c for c in original_filename if c.isalnum() or c in "._-")
        stored_filename = f"{timestamp_str}_{clean_filename}"

        return str(storage_dir / stored_filename)

    def store_file(self, content: bytes, dataset_name: str, submitter_email: str, original_filename: str) -> Dict[str, Any]:
        timestamp = datetime.utcnow()
        file_hash = self.generate_file_hash(content)
        storage_path = self.generate_storage_path(dataset_name, submitter_email, original_filename, timestamp)

        with open(storage_path, "wb") as f:
            f.write(content)

        return {
            "stored_file_path": storage_path,
            "file_hash": file_hash,
            "file_size": len(content),
            "upload_timestamp": timestamp,
            "original_filename": original_filename,
        }

    def verify_file_integrity(self, file_path: str, expected_hash: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                actual_hash = self.generate_file_hash(f.read())
            return actual_hash == expected_hash
        except Exception:
            return False

    def retrieve_file(self, file_path: str) -> bytes:
        with open(file_path, "rb") as f:
            return f.read()
