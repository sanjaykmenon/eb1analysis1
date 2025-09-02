"""
Criterion Insights Analysis Module

This module orchestrates the batch processing of criteria records through LLM analysis
to extract detailed insights about AAO decision patterns, evidence strengths, and
rejection/success factors for each EB-1A criterion type.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .db import (
    get_engine, get_criteria_for_analysis, get_criteria_counts_by_type,
    create_extraction_batch, update_extraction_batch, store_criterion_insights,
    get_processing_progress
)
from .llm import extract_criterion_insights
from .config import settings

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CriterionInsightsProcessor:
    """
    Main processor for extracting insights from criteria rationales using LLM analysis.
    
    Handles batch processing, error logging, progress tracking, and quality validation.
    """
    
    def __init__(self, batch_size: int = 25, error_log_dir: str = "analysis_errors"):
        self.batch_size = batch_size
        self.error_log_dir = Path(error_log_dir)
        self.error_log_dir.mkdir(exist_ok=True)
        self.engine = get_engine()
        
    def get_processing_overview(self) -> Dict[str, Any]:
        """Get overview of criteria available for processing."""
        counts = get_criteria_counts_by_type(self.engine, exclude_processed=True)
        total_unprocessed = sum(counts.values())
        
        progress = get_processing_progress(self.engine)
        
        return {
            "unprocessed_counts": counts,
            "total_unprocessed": total_unprocessed,
            "estimated_batches": {
                criterion: (count + self.batch_size - 1) // self.batch_size 
                for criterion, count in counts.items()
            },
            "processing_progress": progress,
            "batch_size": self.batch_size
        }
    
    def process_criterion_type(self, criterion_type: str, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all unprocessed criteria of a specific type.
        
        Args:
            criterion_type: Type of criterion to process (e.g., 'AWARD', 'MEMBERSHIP')
            max_batches: Maximum number of batches to process (None for all)
        
        Returns:
            Processing results summary
        """
        logger.info(f"Starting processing for criterion type: {criterion_type}")
        
        # Get total count for planning
        total_count = get_criteria_counts_by_type(self.engine, exclude_processed=True).get(criterion_type, 0)
        if total_count == 0:
            logger.info(f"No unprocessed {criterion_type} criteria found")
            return {"status": "no_data", "criterion_type": criterion_type}
        
        # Create batch tracking record
        estimated_batches = (total_count + self.batch_size - 1) // self.batch_size
        actual_batches = min(estimated_batches, max_batches) if max_batches else estimated_batches
        
        batch_id = create_extraction_batch(
            self.engine, 
            criterion_type, 
            actual_batches * self.batch_size
        )
        
        logger.info(f"Created batch {batch_id} for {criterion_type}: {total_count} records, ~{estimated_batches} batches")
        
        # Process in batches
        results = {
            "criterion_type": criterion_type,
            "batch_id": batch_id,
            "total_records": total_count,
            "processed_records": 0,
            "successful_records": 0,
            "error_records": 0,
            "batches_completed": 0,
            "errors": []
        }
        
        try:
            offset = 0
            batch_num = 0
            
            while batch_num < actual_batches:
                batch_num += 1
                logger.info(f"Processing batch {batch_num}/{actual_batches} for {criterion_type}")
                
                # Fetch batch of criteria
                criteria_batch = get_criteria_for_analysis(
                    self.engine,
                    criterion_type=criterion_type,
                    batch_size=self.batch_size,
                    offset=offset,
                    exclude_processed=True
                )
                
                if not criteria_batch:
                    logger.info(f"No more records found for {criterion_type}")
                    break
                
                # Process batch
                batch_results = self._process_batch(criteria_batch, criterion_type, batch_num)
                
                # Update results
                results["processed_records"] += batch_results["processed"]
                results["successful_records"] += batch_results["successful"]
                results["error_records"] += batch_results["errors"]
                results["batches_completed"] += 1
                results["errors"].extend(batch_results["error_details"])
                
                # Update batch tracking
                update_extraction_batch(
                    self.engine,
                    batch_id,
                    processed_count=results["processed_records"],
                    success_count=results["successful_records"],
                    error_count=results["error_records"]
                )
                
                offset += self.batch_size
                
                # Rate limiting - brief pause between batches
                time.sleep(1)
            
            # Mark batch as completed
            error_log_path = None
            if results["errors"]:
                error_log_path = self._save_error_log(criterion_type, batch_id, results["errors"])
            
            update_extraction_batch(
                self.engine,
                batch_id,
                status="completed",
                error_log_path=error_log_path
            )
            
            results["status"] = "completed"
            logger.info(f"Completed {criterion_type}: {results['successful_records']}/{results['processed_records']} successful")
            
        except Exception as e:
            logger.error(f"Fatal error processing {criterion_type}: {e}")
            update_extraction_batch(
                self.engine,
                batch_id,
                status="failed",
                error_log_path=self._save_error_log(criterion_type, batch_id, [{"error": str(e), "type": "fatal"}])
            )
            results["status"] = "failed"
            results["fatal_error"] = str(e)
        
        return results
    
    def _process_batch(self, criteria_batch: List[Dict], criterion_type: str, batch_num: int) -> Dict[str, Any]:
        """Process a single batch of criteria records."""
        
        results = {
            "processed": 0,
            "successful": 0,
            "errors": 0,
            "error_details": []
        }
        
        for i, criterion_record in enumerate(criteria_batch):
            try:
                logger.debug(f"Processing criterion {criterion_record['criterion_id']} ({i+1}/{len(criteria_batch)})")
                
                # Extract insights using LLM
                extraction_result = extract_criterion_insights(criterion_record)
                results["processed"] += 1
                
                if extraction_result["extraction_successful"]:
                    # Store insights in database
                    insights_with_metadata = {
                        **extraction_result["insights"],
                        "quality_metrics": extraction_result["quality_metrics"]
                    }
                    
                    store_criterion_insights(
                        self.engine,
                        criterion_record["criterion_id"],
                        insights_with_metadata
                    )
                    
                    results["successful"] += 1
                    logger.debug(f"Successfully processed criterion {criterion_record['criterion_id']}")
                    
                else:
                    # Log extraction error
                    error_detail = {
                        "criterion_id": criterion_record["criterion_id"],
                        "case_number": criterion_record.get("case_number"),
                        "error": extraction_result["error"],
                        "error_type": extraction_result["error_type"],
                        "rationale_length": len(criterion_record.get("rationale", "")),
                        "batch_num": batch_num,
                        "batch_position": i + 1
                    }
                    
                    results["errors"] += 1
                    results["error_details"].append(error_detail)
                    logger.warning(f"Failed to extract from criterion {criterion_record['criterion_id']}: {extraction_result['error']}")
                
            except Exception as e:
                # Catch any unexpected errors
                error_detail = {
                    "criterion_id": criterion_record.get("criterion_id"),
                    "case_number": criterion_record.get("case_number"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "batch_num": batch_num,
                    "batch_position": i + 1,
                    "unexpected": True
                }
                
                results["errors"] += 1
                results["error_details"].append(error_detail)
                logger.error(f"Unexpected error processing criterion {criterion_record.get('criterion_id')}: {e}")
        
        return results
    
    def _save_error_log(self, criterion_type: str, batch_id: int, errors: List[Dict]) -> str:
        """Save error details to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"errors_{criterion_type}_{batch_id}_{timestamp}.json"
        filepath = self.error_log_dir / filename
        
        error_log = {
            "criterion_type": criterion_type,
            "batch_id": batch_id,
            "timestamp": timestamp,
            "total_errors": len(errors),
            "errors": errors
        }
        
        with open(filepath, 'w') as f:
            json.dump(error_log, f, indent=2, default=str)
        
        logger.info(f"Saved {len(errors)} errors to {filepath}")
        return str(filepath)
    
    def process_all_criteria(self, max_batches_per_type: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all unprocessed criteria across all criterion types.
        
        Args:
            max_batches_per_type: Maximum batches to process per criterion type
        
        Returns:
            Overall processing results
        """
        logger.info("Starting comprehensive criterion analysis across all types")
        
        overview = self.get_processing_overview()
        criterion_types = list(overview["unprocessed_counts"].keys())
        
        if not criterion_types:
            logger.info("No unprocessed criteria found")
            return {"status": "no_data", "message": "All criteria already processed"}
        
        overall_results = {
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "total_criterion_types": len(criterion_types),
            "results_by_type": {},
            "overall_stats": {
                "total_processed": 0,
                "total_successful": 0,
                "total_errors": 0
            }
        }
        
        for i, criterion_type in enumerate(criterion_types):
            logger.info(f"Processing criterion type {i+1}/{len(criterion_types)}: {criterion_type}")
            
            type_results = self.process_criterion_type(criterion_type, max_batches_per_type)
            overall_results["results_by_type"][criterion_type] = type_results
            
            # Update overall stats
            if "processed_records" in type_results:
                overall_results["overall_stats"]["total_processed"] += type_results["processed_records"]
                overall_results["overall_stats"]["total_successful"] += type_results["successful_records"]
                overall_results["overall_stats"]["total_errors"] += type_results["error_records"]
        
        overall_results["status"] = "completed"
        overall_results["end_time"] = datetime.now().isoformat()
        
        logger.info(f"Completed comprehensive analysis: {overall_results['overall_stats']['total_successful']}/{overall_results['overall_stats']['total_processed']} successful")
        
        return overall_results

@dataclass
class RateLimiter:
    """Thread-safe rate limiter for OpenAI API calls."""
    requests_per_minute: int
    requests_per_day: int
    
    def __post_init__(self):
        self.minute_requests = deque()
        self.day_requests = deque()
        self.lock = RLock()
        self.total_delays = 0
        self.total_delay_time = 0.0
    
    def acquire(self) -> float:
        """Acquire permission to make a request. Returns delay time in seconds."""
        with self.lock:
            now = time.time()
            
            # Clean old requests
            minute_ago = now - 60
            day_ago = now - 86400
            
            while self.minute_requests and self.minute_requests[0] < minute_ago:
                self.minute_requests.popleft()
            
            while self.day_requests and self.day_requests[0] < day_ago:
                self.day_requests.popleft()
            
            # Check limits
            delay = 0.0
            
            if len(self.minute_requests) >= self.requests_per_minute:
                # Need to wait until oldest request is > 1 minute old
                delay = max(delay, self.minute_requests[0] + 60 - now)
            
            if len(self.day_requests) >= self.requests_per_day:
                # Need to wait until oldest request is > 1 day old
                delay = max(delay, self.day_requests[0] + 86400 - now)
            
            if delay > 0:
                self.total_delays += 1
                self.total_delay_time += delay
                time.sleep(delay)
                now = time.time()
            
            # Record the request
            self.minute_requests.append(now)
            self.day_requests.append(now)
            
            return delay
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        return {
            "total_delays": self.total_delays,
            "total_delay_time": self.total_delay_time,
            "current_minute_requests": len(self.minute_requests),
            "current_day_requests": len(self.day_requests)
        }

class ParallelCriterionInsightsProcessor:
    """Enhanced processor with parallel processing and advanced rate limiting."""
    
    def __init__(
        self,
        batch_size: int = 25,
        error_log_dir: str = "analysis_errors",
        # Parallelization settings
        max_workers: int = 10,
        criterion_workers: int = 3,
        batch_workers: int = 2,
        # Rate limiting settings
        requests_per_minute: int = 3000,
        requests_per_day: int = 200000,
        # Error handling settings
        max_retries: int = 3,
        retry_delay: float = 1.0,
        # Database settings
        db_pool_size: int = 20,
        dsn: Optional[str] = None,
        # Progress settings
        progress_interval: int = 50,
    ):
        self.batch_size = batch_size
        self.error_log_dir = Path(error_log_dir)
        self.error_log_dir.mkdir(exist_ok=True)
        
        # Parallelization settings
        self.max_workers = max_workers
        self.criterion_workers = criterion_workers
        self.batch_workers = batch_workers
        
        # Rate limiting
        self.rate_limiter = RateLimiter(requests_per_minute, requests_per_day)
        
        # Error handling
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Database settings
        self.db_pool_size = db_pool_size
        self.engine = get_engine(dsn, pool_size=db_pool_size, max_overflow=db_pool_size*2)
        
        # Progress tracking
        self.progress_interval = progress_interval
        self.progress_lock = Lock()
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # Statistics
        self.start_time = None
        self.processing_stats = defaultdict(int)
    
    def get_processing_overview(self) -> Dict[str, Any]:
        """Get processing overview using the existing sequential processor logic."""
        # Delegate to existing implementation but use our engine
        sequential_processor = CriterionInsightsProcessor(
            batch_size=self.batch_size,
            error_log_dir=str(self.error_log_dir)
        )
        sequential_processor.engine = self.engine
        return sequential_processor.get_processing_overview()
    
    def estimate_processing_time(self, total_records: int) -> str:
        """Estimate processing time with parallel processing."""
        # Assume ~2.5 seconds per LLM call with parallel processing
        estimated_seconds = (total_records * 2.5) / self.max_workers
        
        if estimated_seconds < 60:
            return f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"{estimated_seconds/60:.1f} minutes"
        else:
            return f"{estimated_seconds/3600:.1f} hours"
    
    def estimate_cost(self, total_records: int) -> float:
        """Estimate OpenAI API cost."""
        # Rough estimate: $0.003 per criterion analysis
        return total_records * 0.003
    
    def _process_criterion_with_retries(self, criterion_record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single criterion with retry logic and rate limiting."""
        criterion_id = criterion_record['criterion_id']
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                delay = self.rate_limiter.acquire()
                
                # Extract insights using LLM
                extraction_result = extract_criterion_insights(criterion_record)
                
                if extraction_result["extraction_successful"]:
                    # Store insights in database with connection retry
                    self._store_insights_with_retry(criterion_record["criterion_id"], extraction_result)
                    
                    # Update progress
                    with self.progress_lock:
                        self.processed_count += 1
                        self.success_count += 1
                        
                        if self.processed_count % self.progress_interval == 0:
                            logger.info(f"Progress: {self.processed_count} processed, {self.success_count} successful")
                    
                    return {
                        "status": "success",
                        "criterion_id": criterion_id,
                        "attempt": attempt + 1,
                        "rate_limit_delay": delay
                    }
                else:
                    # LLM extraction failed
                    with self.progress_lock:
                        self.processed_count += 1
                        self.error_count += 1
                    
                    return {
                        "status": "llm_failed",
                        "criterion_id": criterion_id,
                        "error": extraction_result.get("error", "Unknown LLM error"),
                        "attempt": attempt + 1
                    }
                    
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed for criterion {criterion_id}: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    # Final failure
                    with self.progress_lock:
                        self.processed_count += 1
                        self.error_count += 1
                    
                    return {
                        "status": "failed",
                        "criterion_id": criterion_id,
                        "error": str(e),
                        "attempts": self.max_retries + 1
                    }
        
        # Should never reach here
        return {"status": "failed", "criterion_id": criterion_id, "error": "Unknown error"}
    
    def _store_insights_with_retry(self, criterion_id: int, extraction_result: Dict[str, Any], max_db_retries: int = 3):
        """Store insights with database retry logic."""
        insights_with_metadata = {
            **extraction_result["insights"],
            "quality_metrics": extraction_result["quality_metrics"]
        }
        
        for attempt in range(max_db_retries):
            try:
                store_criterion_insights(self.engine, criterion_id, insights_with_metadata)
                return
            except Exception as e:
                if attempt < max_db_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)
                    logger.warning(f"Database storage attempt {attempt + 1} failed for criterion {criterion_id}: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise e  # Re-raise on final attempt
    
    def process_criterion_type_parallel(self, criterion_type: str, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """Process a single criterion type using parallel workers."""
        self.start_time = time.time()
        
        logger.info(f"Starting parallel processing for criterion type: {criterion_type}")
        
        # Get total count and setup
        total_count = get_criteria_counts_by_type(self.engine, exclude_processed=True).get(criterion_type, 0)
        
        if total_count == 0:
            return {"status": "no_data", "message": f"No unprocessed {criterion_type} criteria found"}
        
        estimated_batches = (total_count + self.batch_size - 1) // self.batch_size
        actual_batches = min(estimated_batches, max_batches) if max_batches else estimated_batches
        
        # Create batch tracking
        batch_id = create_extraction_batch(self.engine, criterion_type, actual_batches * self.batch_size)
        
        logger.info(f"Processing {criterion_type}: {total_count} records in ~{actual_batches} batches with {self.max_workers} workers")
        
        # Results tracking
        results = {
            "criterion_type": criterion_type,
            "batch_id": batch_id,
            "total_records": total_count,
            "processed_records": 0,
            "successful_records": 0,
            "error_records": 0,
            "batches_completed": 0,
            "errors": [],
            "status": "running"
        }
        
        all_errors = []
        
        try:
            # Process batches in parallel (limited concurrency)
            with ThreadPoolExecutor(max_workers=self.batch_workers) as batch_executor:
                
                # Submit batch processing tasks
                batch_futures = []
                for batch_num in range(1, actual_batches + 1):
                    offset = (batch_num - 1) * self.batch_size
                    future = batch_executor.submit(
                        self._process_batch_parallel,
                        criterion_type, batch_num, offset, actual_batches
                    )
                    batch_futures.append(future)
                
                # Collect results as batches complete
                for future in as_completed(batch_futures):
                    try:
                        batch_result = future.result()
                        
                        # Update overall results
                        results["processed_records"] += batch_result["processed"]
                        results["successful_records"] += batch_result["successful"]
                        results["error_records"] += batch_result["errors"]
                        results["batches_completed"] += 1
                        all_errors.extend(batch_result["error_details"])
                        
                        logger.info(f"Batch {results['batches_completed']}/{actual_batches} completed for {criterion_type}")
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed for {criterion_type}: {e}")
                        results["error_records"] += self.batch_size  # Estimate
                        all_errors.append({
                            "batch_error": str(e),
                            "criterion_type": criterion_type
                        })
            
            # Update batch tracking
            update_extraction_batch(
                self.engine, batch_id,
                processed_count=results["processed_records"],
                success_count=results["successful_records"],
                error_count=results["error_records"],
                status="completed"
            )
            
            results["status"] = "completed"
            results["errors"] = all_errors
            results["processing_time_seconds"] = time.time() - self.start_time
            
            logger.info(f"Completed {criterion_type}: {results['successful_records']}/{results['processed_records']} successful")
            
        except Exception as e:
            # Update batch as failed
            update_extraction_batch(self.engine, batch_id, status="failed")
            results["status"] = "failed"
            results["fatal_error"] = str(e)
            logger.error(f"Fatal error processing {criterion_type}: {e}")
        
        return results
    
    def _process_batch_parallel(self, criterion_type: str, batch_num: int, offset: int, total_batches: int) -> Dict[str, Any]:
        """Process a single batch using parallel workers."""
        logger.info(f"Processing batch {batch_num}/{total_batches} for {criterion_type} (offset={offset})")
        
        # Fetch criteria for this batch
        criteria_batch = get_criteria_for_analysis(
            self.engine,
            criterion_type=criterion_type,
            batch_size=self.batch_size,
            offset=offset,
            exclude_processed=True
        )
        
        if not criteria_batch:
            return {
                "processed": 0,
                "successful": 0,
                "errors": 0,
                "error_details": []
            }
        
        # Process criteria in parallel within the batch
        batch_results = {
            "processed": 0,
            "successful": 0,
            "errors": 0,
            "error_details": []
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all criteria in this batch
            future_to_criterion = {
                executor.submit(self._process_criterion_with_retries, criterion_record): criterion_record
                for criterion_record in criteria_batch
            }
            
            # Collect results
            for future in as_completed(future_to_criterion):
                criterion_record = future_to_criterion[future]
                
                try:
                    result = future.result()
                    batch_results["processed"] += 1
                    
                    if result["status"] == "success":
                        batch_results["successful"] += 1
                    else:
                        batch_results["errors"] += 1
                        batch_results["error_details"].append({
                            "criterion_id": result["criterion_id"],
                            "case_number": criterion_record.get("case_number"),
                            "error": result.get("error", "Unknown error"),
                            "status": result["status"],
                            "attempts": result.get("attempts", 1)
                        })
                
                except Exception as e:
                    batch_results["processed"] += 1
                    batch_results["errors"] += 1
                    batch_results["error_details"].append({
                        "criterion_id": criterion_record["criterion_id"],
                        "case_number": criterion_record.get("case_number"),
                        "error": f"Future exception: {str(e)}",
                        "status": "future_failed"
                    })
        
        return batch_results
    
    def process_all_criteria_parallel(self, max_batches_per_type: Optional[int] = None) -> Dict[str, Any]:
        """Process all criterion types in parallel."""
        self.start_time = time.time()
        
        logger.info("Starting parallel processing of all criterion types")
        
        overview = self.get_processing_overview()
        criterion_types = list(overview["unprocessed_counts"].keys())
        
        if not criterion_types:
            return {"status": "no_data", "message": "All criteria already processed"}
        
        overall_results = {
            "status": "running",
            "start_time": time.time(),
            "total_criterion_types": len(criterion_types),
            "results_by_type": {},
            "overall_stats": {
                "total_processed": 0,
                "total_successful": 0,
                "total_errors": 0
            },
            "rate_limit_stats": {},
            "estimated_cost": 0.0
        }
        
        # Process criterion types in parallel (limited concurrency)
        with ThreadPoolExecutor(max_workers=self.criterion_workers) as criterion_executor:
            
            # Submit criterion type processing tasks
            type_futures = {
                criterion_executor.submit(
                    self.process_criterion_type_parallel,
                    criterion_type,
                    max_batches_per_type
                ): criterion_type
                for criterion_type in criterion_types
            }
            
            # Collect results as criterion types complete
            for future in as_completed(type_futures):
                criterion_type = type_futures[future]
                
                try:
                    type_results = future.result()
                    overall_results["results_by_type"][criterion_type] = type_results
                    
                    # Update overall stats
                    if "processed_records" in type_results:
                        overall_results["overall_stats"]["total_processed"] += type_results["processed_records"]
                        overall_results["overall_stats"]["total_successful"] += type_results["successful_records"]
                        overall_results["overall_stats"]["total_errors"] += type_results["error_records"]
                    
                    logger.info(f"Completed criterion type {criterion_type}")
                    
                except Exception as e:
                    logger.error(f"Failed to process criterion type {criterion_type}: {e}")
                    overall_results["results_by_type"][criterion_type] = {
                        "status": "failed",
                        "error": str(e)
                    }
        
        # Finalize results
        overall_results["status"] = "completed"
        overall_results["end_time"] = time.time()
        overall_results["total_processing_time_seconds"] = time.time() - self.start_time
        overall_results["rate_limit_stats"] = self.rate_limiter.get_stats()
        overall_results["estimated_cost"] = self.estimate_cost(overall_results["overall_stats"]["total_processed"])
        
        # Estimate sequential time for comparison
        sequential_estimate = overall_results["overall_stats"]["total_processed"] * 3.5  # ~3.5s per record sequential
        overall_results["sequential_estimate"] = sequential_estimate
        
        logger.info(f"Completed all criterion types: {overall_results['overall_stats']['total_successful']}/{overall_results['overall_stats']['total_processed']} successful")
        
        return overall_results

def main():
    """Command line interface for criterion insights processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process AAO criteria for insights extraction")
    parser.add_argument("--criterion-type", help="Specific criterion type to process")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for processing")
    parser.add_argument("--max-batches", type=int, help="Maximum batches to process")
    parser.add_argument("--overview", action="store_true", help="Show processing overview only")
    
    args = parser.parse_args()
    
    processor = CriterionInsightsProcessor(batch_size=args.batch_size)
    
    if args.overview:
        overview = processor.get_processing_overview()
        print(json.dumps(overview, indent=2))
        return
    
    if args.criterion_type:
        results = processor.process_criterion_type(args.criterion_type, args.max_batches)
    else:
        results = processor.process_all_criteria(args.max_batches)
    
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()