#!/usr/bin/env python3
import pandas as pd
import re
import io
from typing import Dict, List, Tuple, Optional

class CSVSecurityValidator:
    def __init__(self):
        # Dangerous SQL patterns
        self.sql_injection_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(--|\#|/\*|\*/)',
            r'(\bOR\b.*=.*\bOR\b)',
            r'(\bAND\b.*=.*\bAND\b)',
            r'(\'.*\bOR\b.*\')',
            r'(\".*\bOR\b.*\")',
            r'(\;.*\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b)',
            r'(\bxp_cmdshell\b)',
            r'(\bsp_executesql\b)',
        ]
        
        # Suspicious content patterns
        self.suspicious_patterns = [
            r'(<script.*?>)',
            r'(javascript:)',
            r'(vbscript:)',
            r'(onload=)',
            r'(onerror=)',
            r'(\beval\b\s*\()',
            r'(\bexec\b\s*\()',
            r'(__import__)',
            r'(\bopen\b\s*\()',
            r'(\bfile\b\s*\()',
        ]
        
        # Valid column name pattern
        self.valid_column_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')
        
    def validate_csv_content(self, csv_content: str, max_size_mb: int = 50) -> Dict:
        """Comprehensive CSV security validation"""
        
        # Check file size
        size_mb = len(csv_content.encode('utf-8')) / (1024 * 1024)
        if size_mb > max_size_mb:
            return {
                "valid": False,
                "error": f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
            }
        
        try:
            # Parse CSV safely
            df = pd.read_csv(io.StringIO(csv_content), nrows=10000)  # Limit rows
            
            # Validate column names
            for col in df.columns:
                if not self.validate_column_name(col):
                    return {
                        "valid": False,
                        "error": f"Invalid column name: '{col}'"
                    }
            
            # Check for suspicious content
            suspicious_content = self.scan_for_suspicious_content(df)
            if suspicious_content:
                return {
                    "valid": False,
                    "error": f"Suspicious content detected: {suspicious_content}"
                }
            
            # Validate data types and content
            content_validation = self.validate_content_safety(df)
            if not content_validation["valid"]:
                return content_validation
            
            return {
                "valid": True,
                "rows": len(df),
                "columns": list(df.columns),
                "info": "CSV validation passed"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"CSV parsing error: {str(e)}"
            }
    
    def validate_column_name(self, column_name: str) -> bool:
        """Validate column names are safe"""
        if len(column_name) > 50:  # Reasonable limit
            return False
        return bool(self.valid_column_pattern.match(str(column_name).strip()))
    
    def scan_for_suspicious_content(self, df: pd.DataFrame) -> Optional[str]:
        """Scan dataframe for SQL injection and malicious patterns"""
        
        # Check column names
        for col in df.columns:
            col_str = str(col).upper()
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, col_str, re.IGNORECASE):
                    return f"SQL pattern in column: {col}"
        
        # Sample first 100 rows for content checking
        sample_df = df.head(100)
        
        for idx, row in sample_df.iterrows():
            for col, value in row.items():
                if pd.isna(value):
                    continue
                    
                value_str = str(value).upper()
                
                # Check for SQL injection patterns
                for pattern in self.sql_injection_patterns:
                    if re.search(pattern, value_str, re.IGNORECASE):
                        return f"SQL injection pattern at row {idx}, column {col}"
                
                # Check for suspicious content
                for pattern in self.suspicious_patterns:
                    if re.search(pattern, value_str, re.IGNORECASE):
                        return f"Suspicious content at row {idx}, column {col}"
        
        return None
    
    def validate_content_safety(self, df: pd.DataFrame) -> Dict:
        """Additional content safety checks"""
        
        # Check for extremely long values that could cause buffer overflows
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                max_length = df[col].astype(str).str.len().max()
                if max_length > 10000:  # 10KB per cell limit
                    return {
                        "valid": False,
                        "error": f"Cell content too long in column {col}: {max_length} chars"
                    }
        
        # Check for binary content (potential malware)
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(50)
                for value in sample_values:
                    value_str = str(value)
                    # Check for binary content indicators
                    if any(ord(char) < 32 or ord(char) > 126 for char in value_str[:100]):
                        if not all(char in '\n\r\t' for char in value_str if ord(char) < 32):
                            return {
                                "valid": False,
                                "error": f"Binary content detected in column {col}"
                            }
        
        return {"valid": True}
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize uploaded filename"""
        # Remove path traversal attempts
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Keep only alphanumeric, dots, hyphens, underscores
        filename = re.sub(r'[^a-zA-Z0-9.\-_]', '', filename)
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        
        # Ensure .csv extension
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        return filename