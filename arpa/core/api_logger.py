"""Comprehensive API call logging system for debugging ARPA extractions.

This module logs every API request and response with detailed analysis of:
- What was requested (schema, prompt, messages)
- What was returned (raw JSON, parsed data)
- Why validation failed (schema mismatches, type errors)
- Comparison between expected vs actual structure

Usage:
    from arpa.core.api_logger import APILogger
    
    logger = APILogger("logs/api_calls.log")
    
    # Log a request
    logger.log_request(call_id="pass1_dataset", schema="DatasetTaskPass", messages=messages)
    
    # Log a response
    logger.log_response(call_id="pass1_dataset", response_text=json_str, parsed_data=data)
    
    # Log validation error
    logger.log_validation_error(call_id="pass1_dataset", schema="DatasetTaskPass", 
                               error=e, actual_data=data)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError


class APILogger:
    """Detailed logger for API calls and schema validation."""
    
    def __init__(self, log_file: str | Path = "api_debug.log"):
        """Initialize API logger.
        
        Args:
            log_file: Path to log file (default: api_debug.log in current dir)
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        self.file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        self.file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(formatter)
        
        # Create logger
        self.logger = logging.getLogger('arpa.api_debug')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)
        
        # Also log to console with less detail
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(console_handler)
        
        # Track call metadata
        self.calls = {}
        
        self._write_header()
    
    def _write_header(self):
        """Write log file header."""
        self.logger.info("=" * 100)
        self.logger.info(f"ARPA API CALL DEBUG LOG")
        self.logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 100)
        self.logger.info("")
    
    def log_request(
        self,
        call_id: str,
        schema_name: str,
        messages: list[dict[str, str]],
        model: str = "unknown",
    ):
        """Log an API request before it's sent.
        
        Args:
            call_id: Unique identifier for this call (e.g., "pass1_dataset")
            schema_name: Name of the Pydantic schema expected
            messages: List of message dicts sent to API
            model: Model name being used
        """
        self.logger.info("╔" + "=" * 98 + "╗")
        self.logger.info(f"║ API REQUEST: {call_id:<84} ║")
        self.logger.info("╠" + "=" * 98 + "╣")
        self.logger.info(f"║ Target Schema: {schema_name:<82} ║")
        self.logger.info(f"║ Model: {model:<90} ║")
        self.logger.info(f"║ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<82} ║")
        self.logger.info("╚" + "=" * 98 + "╝")
        self.logger.info("")
        
        # Log messages
        self.logger.info("📝 MESSAGES SENT TO API:")
        for i, msg in enumerate(messages):
            self.logger.info(f"  [{i+1}] Role: {msg['role']}")
            content = msg['content']
            if len(content) > 1000:
                self.logger.info(f"      Content ({len(content)} chars): {content[:500]}...")
                self.logger.info(f"      ... [truncated] ... {content[-200:]}")
            else:
                self.logger.info(f"      Content: {content}")
        self.logger.info("")
        
        # Store metadata
        self.calls[call_id] = {
            "schema_name": schema_name,
            "model": model,
            "messages": messages,
            "timestamp": datetime.now().isoformat(),
        }
    
    def log_response(
        self,
        call_id: str,
        response_text: str,
        parsed_data: dict[str, Any] | None = None,
        response_time_s: float | None = None,
    ):
        """Log API response after receiving it.
        
        Args:
            call_id: Same ID used in log_request
            response_text: Raw text response from API
            parsed_data: Parsed JSON data (if successfully parsed)
            response_time_s: Response time in seconds
        """
        self.logger.info("╔" + "=" * 98 + "╗")
        self.logger.info(f"║ API RESPONSE: {call_id:<83} ║")
        self.logger.info("╠" + "=" * 98 + "╣")
        if response_time_s:
            self.logger.info(f"║ Response Time: {response_time_s:.2f}s{' ' * 78} ║")
        self.logger.info(f"║ Response Length: {len(response_text)} characters{' ' * 65} ║")
        self.logger.info("╚" + "=" * 98 + "╝")
        self.logger.info("")
        
        # Log raw response
        self.logger.info("📨 RAW API RESPONSE:")
        if len(response_text) > 2000:
            self.logger.info(f"{response_text[:1000]}")
            self.logger.info(f"... [middle truncated, {len(response_text)} total chars] ...")
            self.logger.info(f"{response_text[-500:]}")
        else:
            self.logger.info(response_text)
        self.logger.info("")
        
        # Log parsed data
        if parsed_data:
            self.logger.info("✅ SUCCESSFULLY PARSED TO JSON:")
            self.logger.info(json.dumps(parsed_data, indent=2))
            self.logger.info("")
            
            # Analyze structure
            self._analyze_structure(call_id, parsed_data)
        
        # Update metadata
        if call_id in self.calls:
            self.calls[call_id].update({
                "response_text": response_text,
                "parsed_data": parsed_data,
                "response_time_s": response_time_s,
            })
    
    def log_validation_error(
        self,
        call_id: str,
        schema: type[BaseModel],
        error: Exception,
        actual_data: dict[str, Any],
    ):
        """Log schema validation error with detailed analysis.
        
        Args:
            call_id: Same ID used in log_request
            schema: Pydantic schema class that failed
            error: The validation error
            actual_data: The data that failed validation
        """
        schema_name = schema.__name__
        
        self.logger.error("╔" + "=" * 98 + "╗")
        self.logger.error(f"║ VALIDATION ERROR: {call_id:<80} ║")
        self.logger.error("╠" + "=" * 98 + "╣")
        self.logger.error(f"║ Schema: {schema_name:<89} ║")
        self.logger.error(f"║ Error Type: {type(error).__name__:<85} ║")
        self.logger.error("╚" + "=" * 98 + "╝")
        self.logger.error("")
        
        # Log the error
        self.logger.error("❌ VALIDATION ERROR DETAILS:")
        self.logger.error(str(error))
        self.logger.error("")
        
        # Log data that failed
        self.logger.error("📊 DATA THAT FAILED VALIDATION:")
        self.logger.error(json.dumps(actual_data, indent=2))
        self.logger.error("")
        
        # Detailed analysis for ValidationError
        if isinstance(error, ValidationError):
            self._analyze_validation_errors(call_id, schema, error, actual_data)
        
        # Update metadata
        if call_id in self.calls:
            self.calls[call_id].update({
                "validation_error": str(error),
                "validation_failed": True,
            })
    
    def _analyze_structure(self, call_id: str, data: dict[str, Any]):
        """Analyze the structure of returned data."""
        self.logger.info("🔍 STRUCTURE ANALYSIS:")
        
        def analyze_value(key: str, value: Any, indent: int = 0):
            prefix = "  " * indent
            
            if value is None:
                self.logger.info(f"{prefix}- {key}: null ⚠️ (might cause validation issues)")
            elif isinstance(value, dict):
                # Check if it looks like a ConfidenceField
                if "value" in value and "confidence" in value:
                    self.logger.info(f"{prefix}- {key}: ConfidenceField ✓")
                    self.logger.info(f"{prefix}    value={value.get('value')}, confidence={value.get('confidence')}")
                else:
                    self.logger.info(f"{prefix}- {key}: dict with {len(value)} keys")
                    for k, v in value.items():
                        analyze_value(k, v, indent + 1)
            elif isinstance(value, list):
                if len(value) == 0:
                    self.logger.info(f"{prefix}- {key}: empty list []")
                else:
                    self.logger.info(f"{prefix}- {key}: list with {len(value)} items")
                    if value and isinstance(value[0], dict):
                        analyze_value(f"{key}[0]", value[0], indent + 1)
            elif isinstance(value, str):
                self.logger.info(f"{prefix}- {key}: string = '{value[:50]}...' ⚠️ (plain string, not ConfidenceField)")
            elif isinstance(value, (int, float)):
                self.logger.info(f"{prefix}- {key}: {type(value).__name__} = {value} ⚠️ (plain number, not ConfidenceField)")
            elif isinstance(value, bool):
                self.logger.info(f"{prefix}- {key}: bool = {value}")
            else:
                self.logger.info(f"{prefix}- {key}: {type(value).__name__} = {value}")
        
        for key, value in data.items():
            analyze_value(key, value)
        
        self.logger.info("")
    
    def _analyze_validation_errors(
        self,
        call_id: str,
        schema: type[BaseModel],
        error: ValidationError,
        actual_data: dict[str, Any],
    ):
        """Provide detailed analysis of each validation error."""
        self.logger.error("🔬 DETAILED ERROR ANALYSIS:")
        self.logger.error("")
        
        errors = error.errors()
        
        for i, err in enumerate(errors, 1):
            self.logger.error(f"Error #{i}:")
            self.logger.error(f"  Field Path: {' -> '.join(str(x) for x in err['loc'])}")
            self.logger.error(f"  Error Type: {err['type']}")
            self.logger.error(f"  Error Message: {err['msg']}")
            
            # Get the actual value
            field_path = err['loc']
            actual_value = actual_data
            try:
                for part in field_path:
                    actual_value = actual_value[part]
                self.logger.error(f"  Actual Value: {actual_value}")
                self.logger.error(f"  Actual Type: {type(actual_value).__name__}")
            except (KeyError, TypeError):
                self.logger.error(f"  Actual Value: <not found in data>")
            
            # Explain the problem
            self._explain_error(err, actual_value)
            self.logger.error("")
    
    def _explain_error(self, err: dict, actual_value: Any):
        """Provide human-readable explanation of validation error."""
        error_type = err['type']
        
        if error_type == 'model_type':
            self.logger.error("  💡 EXPLANATION:")
            self.logger.error("     Schema expects a ConfidenceField object with structure:")
            self.logger.error("       {\"value\": <data>, \"confidence\": \"confirmed|inferred|assumed\", \"source\": \"...\"}")
            self.logger.error(f"     But received: {type(actual_value).__name__} = {actual_value}")
            self.logger.error("     FIX: Either update the prompt to request ConfidenceField format,")
            self.logger.error("          or add a validator to wrap plain values into ConfidenceField objects")
        
        elif error_type == 'list_type':
            self.logger.error("  💡 EXPLANATION:")
            self.logger.error("     Schema expects a list (array)")
            self.logger.error(f"     But received: {type(actual_value).__name__} = {actual_value}")
            if actual_value is None:
                self.logger.error("     FIX: Add validator to convert None → [] (empty list)")
        
        elif error_type == 'dict_type':
            self.logger.error("  💡 EXPLANATION:")
            self.logger.error("     Schema expects a dictionary (object)")
            self.logger.error(f"     But received: {type(actual_value).__name__} = {actual_value}")
            if actual_value is None:
                self.logger.error("     FIX: Add validator to convert None → {} (empty dict)")
        
        elif error_type == 'string_type':
            self.logger.error("  💡 EXPLANATION:")
            self.logger.error("     Schema expects a string")
            self.logger.error(f"     But received: {type(actual_value).__name__} = {actual_value}")
            if actual_value is None:
                self.logger.error("     FIX: Make field optional or provide default value")
        
        elif error_type == 'int_type':
            self.logger.error("  💡 EXPLANATION:")
            self.logger.error("     Schema expects an integer")
            self.logger.error(f"     But received: {type(actual_value).__name__} = {actual_value}")
        
        elif error_type == 'float_type':
            self.logger.error("  💡 EXPLANATION:")
            self.logger.error("     Schema expects a float/number")
            self.logger.error(f"     But received: {type(actual_value).__name__} = {actual_value}")
        
        elif error_type == 'enum':
            self.logger.error("  💡 EXPLANATION:")
            self.logger.error("     Schema expects one of a fixed set of values (enum)")
            self.logger.error(f"     But received: {actual_value}")
            self.logger.error(f"     Expected values: {err.get('ctx', {}).get('expected', 'unknown')}")
        
        else:
            self.logger.error(f"  💡 EXPLANATION: Validation error type '{error_type}'")
    
    def write_summary(self):
        """Write summary of all API calls at the end."""
        self.logger.info("")
        self.logger.info("=" * 100)
        self.logger.info("SUMMARY OF ALL API CALLS")
        self.logger.info("=" * 100)
        self.logger.info("")
        
        total_calls = len(self.calls)
        failed_calls = sum(1 for c in self.calls.values() if c.get('validation_failed'))
        success_calls = total_calls - failed_calls
        
        self.logger.info(f"Total API Calls: {total_calls}")
        self.logger.info(f"Successful: {success_calls}")
        self.logger.info(f"Failed Validation: {failed_calls}")
        self.logger.info("")
        
        if failed_calls > 0:
            self.logger.info("Failed Calls:")
            for call_id, metadata in self.calls.items():
                if metadata.get('validation_failed'):
                    self.logger.info(f"  - {call_id} (Schema: {metadata['schema_name']})")
        
        self.logger.info("")
        self.logger.info("=" * 100)
        self.logger.info(f"Log file: {self.log_file.absolute()}")
        self.logger.info("=" * 100)


# Global instance
_global_logger: APILogger | None = None


def get_api_logger(log_file: str | Path = "api_debug.log") -> APILogger:
    """Get or create global API logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = APILogger(log_file)
    return _global_logger


def reset_api_logger():
    """Reset global logger (useful for testing)."""
    global _global_logger
    _global_logger = None
