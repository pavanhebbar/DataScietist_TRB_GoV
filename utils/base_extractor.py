# utils/base_extractor.py
"""
Abstract base class defining the extraction interface.
Both LocalExtractor and AWSExtractor implement this contract,
making them fully interchangeable in main.py.
"""
from abc import ABC, abstractmethod
import pandas as pd

class BaseExtractor(ABC):

    @abstractmethod
    def extract_distance_log_xlsx(self, source: str) -> pd.DataFrame:
        """
        Extract distance log from an Excel file.
        source: local path (local) or S3 URI (AWS)
        """
        pass

    @abstractmethod
    def extract_distance_log_pdf(self, source: str) -> pd.DataFrame:
        """
        Extract distance log from a scanned multi-page PDF.
        source: local path (local) or S3 URI (AWS)
        """
        pass

    @abstractmethod
    def extract_invoices_docx(self, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract invoice records from a Word doc containing images.
        source: local path (local) or S3 URI (AWS)
        Returns: (invoice_df, confidence_scores_df)
        """
        pass
