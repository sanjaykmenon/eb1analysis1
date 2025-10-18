from __future__ import annotations
import os, glob, json
import typer
import importlib.resources as ir
from pathlib import Path
from sqlalchemy import text
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .db import get_engine, run_sql_file
from .pipeline import process_single_pdf

app = typer.Typer(add_completion=False, help="AAO ETL: parse AAO PDFs → Postgres")

# Thread-local storage for database connections
thread_local = threading.local()

def get_thread_engine(dsn: Optional[str] = None):
    """Get a database engine for the current thread."""
    if not hasattr(thread_local, 'engine'):
        thread_local.engine = get_engine(dsn)
    return thread_local.engine

@app.command("init-db")
def init_db(
    sql_path: Optional[str] = typer.Option(None, help="Path to SQL init script"),
    dsn: Optional[str] = typer.Option(None, help="Postgres DSN (overrides DATABASE_URL)"),
):
    """Create tables and enable pgvector."""
    engine = get_engine(dsn)
    if sql_path is None:
        # Load the packaged SQL: src/aao_etl/sql/init_db.sql
        with ir.as_file(ir.files("aao_etl.sql").joinpath("init_db.sql")) as p:
            run_sql_file(engine, str(p))
    else:
        run_sql_file(engine, sql_path)
    typer.echo("✅ Database initialized.")

@app.command()
def process(
    pdf_folder: str,
    dsn: str = typer.Option(None, help="Database connection string"),
    dry_run: bool = typer.Option(False, help="Skip LLM extraction, just regex"),
    resume: bool = typer.Option(False, help="Skip PDFs already in database"),
    max_workers: int = typer.Option(4, help="Number of parallel workers (default: 4)"),
    batch_size: int = typer.Option(100, help="Batch size for progress reporting (default: 100)"),
):
    """Process a folder of AAO decision PDFs with parallel processing."""
    engine = get_engine(dsn)
    
    pdfs = list(Path(pdf_folder).glob("**/*.pdf"))
    if not pdfs:
        typer.echo("No PDFs found.")
        raise typer.Exit(code=0)
    
    # Get already processed PDFs if resuming
    processed_pdfs = set()
    if resume:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT pdf_path FROM decisions WHERE pdf_path IS NOT NULL"))
            processed_pdfs = {row[0] for row in result}
        typer.echo(f"📊 Found {len(processed_pdfs)} already processed PDFs")
    
    # Filter out already processed PDFs
    pdfs_to_process = [str(p) for p in pdfs if str(p) not in processed_pdfs]
    
    if resume and len(pdfs_to_process) < len(pdfs):
        skipped_count = len(pdfs) - len(pdfs_to_process)
        typer.echo(f"🔄 Resuming: skipping {skipped_count} already processed PDFs")
    
    typer.echo(f"🚀 Processing {len(pdfs_to_process)} PDFs with {max_workers} parallel workers...")
    
    # Process PDFs in parallel
    results = []
    failed_pdfs = []
    completed_count = 0
    
    def process_pdf_wrapper(pdf_path: str):
        """Wrapper function for parallel processing."""
        try:
            engine = get_thread_engine(dsn)
            return process_single_pdf(pdf_path=pdf_path, engine=engine, dry_run=dry_run)
        except Exception as e:
            # Fallback error handling if process_single_pdf doesn't catch everything
            return {
                "pdf": os.path.basename(pdf_path),
                "status": "failed",
                "error": f"Worker error: {str(e)}",
                "decision_id": None,
                "case_number": None,
                "outcome": None,
                "criteria_count": 0,
                "evidence_count": 0,
            }
    
    # Use ThreadPoolExecutor for I/O-bound LLM calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(process_pdf_wrapper, pdf_path): pdf_path 
            for pdf_path in pdfs_to_process
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            completed_count += 1
            
            try:
                result = future.result()
                results.append(result)
                
                if result["status"] == "failed":
                    failed_pdfs.append(result)
                    typer.echo(f"❌ [{completed_count}/{len(pdfs_to_process)}] {result['pdf']}: {result['error'][:80]}...")
                else:
                    typer.echo(f"✅ [{completed_count}/{len(pdfs_to_process)}] {result['pdf']} - {result['criteria_count']} criteria, {result['evidence_count']} evidence")
                
                # Progress report every batch_size completions
                if completed_count % batch_size == 0:
                    successful_so_far = len([r for r in results if r["status"] == "success"])
                    failed_so_far = len(failed_pdfs)
                    typer.echo(f"\n📊 Progress: {completed_count}/{len(pdfs_to_process)} completed | ✅ {successful_so_far} successful | ❌ {failed_so_far} failed\n")
                    
            except Exception as e:
                # Handle any unexpected errors from the future
                error_result = {
                    "pdf": os.path.basename(pdf_path),
                    "status": "failed", 
                    "error": f"Future error: {str(e)}",
                    "decision_id": None,
                    "case_number": None,
                    "outcome": None,
                    "criteria_count": 0,
                    "evidence_count": 0,
                }
                results.append(error_result)
                failed_pdfs.append(error_result)
                typer.echo(f"❌ [{completed_count}/{len(pdfs_to_process)}] {os.path.basename(pdf_path)}: Future error: {str(e)}")
    
    # Summary report
    typer.echo("\n" + "="*60)
    typer.echo("📊 FINAL PROCESSING SUMMARY")
    typer.echo("="*60)
    
    successful = [r for r in results if r["status"] == "success"]
    total_criteria = sum(r.get("criteria_count", 0) for r in successful)
    total_evidence = sum(r.get("evidence_count", 0) for r in successful)
    
    typer.echo(f"✅ Successful: {len(successful)}")
    typer.echo(f"❌ Failed: {len(failed_pdfs)}")
    typer.echo(f"📁 Total processed: {len(results)}")
    typer.echo(f"🎯 Total criteria extracted: {total_criteria}")
    typer.echo(f"📋 Total evidence items: {total_evidence}")
    typer.echo(f"⚡ Workers used: {max_workers}")
    
    if failed_pdfs:
        typer.echo(f"\n❌ FAILED PDFs ({len(failed_pdfs)}):")
        for failed in failed_pdfs[:10]:  # Show first 10 failures
            typer.echo(f"  - {failed['pdf']}: {failed['error'][:100]}...")
        if len(failed_pdfs) > 10:
            typer.echo(f"  ... and {len(failed_pdfs) - 10} more (see processing_results.json)")
    
    # Detailed JSON output
    typer.echo(f"\n📋 Saving detailed results to processing_results.json")
    with open("processing_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_found": len(pdfs),
                "already_processed": len(processed_pdfs) if resume else 0,
                "attempted": len(pdfs_to_process),
                "successful": len(successful),
                "failed": len(failed_pdfs),
                "total_criteria": total_criteria,
                "total_evidence": total_evidence,
                "max_workers": max_workers
            },
            "results": results,
            "failed_pdfs": failed_pdfs
        }, f, indent=2, default=str)
    
    if failed_pdfs:
        typer.echo(f"\n⚠️  {len(failed_pdfs)} PDFs failed - see processing_results.json for details")
        raise typer.Exit(code=1)
    else:
        typer.echo("🎉 All PDFs processed successfully!")
        raise typer.Exit(code=0)

@app.command("insights")
def insights_command(
    criterion_type: Optional[str] = typer.Option(None, help="Specific criterion type (AWARD, MEMBERSHIP, etc.)"),
    batch_size: int = typer.Option(25, help="Batch size for LLM processing"),
    max_batches: Optional[int] = typer.Option(None, help="Maximum batches to process"),
    overview_only: bool = typer.Option(False, "--overview", help="Show processing overview only"),
    dsn: Optional[str] = typer.Option(None, help="Database connection string"),
):
    """Extract detailed insights from criteria rationales using LLM analysis."""
    from .criterion_insights import CriterionInsightsProcessor
    
    processor = CriterionInsightsProcessor(batch_size=batch_size, error_log_dir="analysis_errors")
    
    if overview_only:
        typer.echo("📊 Criterion Insights Processing Overview")
        typer.echo("=" * 50)
        
        overview = processor.get_processing_overview()
        
        typer.echo(f"Total unprocessed records: {overview['total_unprocessed']}")
        typer.echo(f"Batch size: {overview['batch_size']}")
        typer.echo()
        
        if overview['unprocessed_counts']:
            typer.echo("Unprocessed counts by criterion type:")
            for criterion, count in overview['unprocessed_counts'].items():
                batches = overview['estimated_batches'][criterion]
                typer.echo(f"  {criterion}: {count} records (~{batches} batches)")
        else:
            typer.echo("✅ All criteria have been processed!")
        
        if overview['processing_progress']:
            typer.echo("\nPrevious processing progress:")
            for progress in overview['processing_progress']:
                typer.echo(f"  {progress['criterion_type']}: {progress['success_records']}/{progress['total_records']} successful")
        
        return
    
    typer.echo("🔍 Starting Criterion Insights Analysis")
    typer.echo("=" * 50)
    
    if criterion_type:
        typer.echo(f"Processing criterion type: {criterion_type}")
        results = processor.process_criterion_type(criterion_type, max_batches)
        
        if results["status"] == "no_data":
            typer.echo(f"ℹ️  No unprocessed {criterion_type} criteria found")
        elif results["status"] == "completed":
            typer.echo(f"✅ Completed {criterion_type}:")
            typer.echo(f"   Processed: {results['processed_records']}")
            typer.echo(f"   Successful: {results['successful_records']}")
            typer.echo(f"   Errors: {results['error_records']}")
            typer.echo(f"   Batches: {results['batches_completed']}")
        elif results["status"] == "failed":
            typer.echo(f"❌ Processing failed for {criterion_type}: {results.get('fatal_error')}")
            raise typer.Exit(code=1)
    else:
        typer.echo("Processing ALL criterion types...")
        results = processor.process_all_criteria(max_batches)
        
        if results["status"] == "no_data":
            typer.echo("ℹ️  No unprocessed criteria found - all have been analyzed!")
        else:
            typer.echo("🎉 Comprehensive analysis completed!")
            typer.echo(f"   Total processed: {results['overall_stats']['total_processed']}")
            typer.echo(f"   Total successful: {results['overall_stats']['total_successful']}")
            typer.echo(f"   Total errors: {results['overall_stats']['total_errors']}")
            
            typer.echo("\nResults by criterion type:")
            for ctype, cresults in results['results_by_type'].items():
                if 'processed_records' in cresults:
                    typer.echo(f"  {ctype}: {cresults['successful_records']}/{cresults['processed_records']} successful")
    
    # Save detailed results
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"insights_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        __import__('json').dump(results, f, indent=2, default=str)
    
    typer.echo(f"\n📋 Detailed results saved to {results_file}")

@app.command("insights-parallel")
def insights_parallel_command(
    criterion_type: Optional[str] = typer.Option(None, help="Specific criterion type (AWARD, MEMBERSHIP, etc.)"),
    batch_size: int = typer.Option(25, help="Records per batch for database operations"),
    max_batches: Optional[int] = typer.Option(None, help="Maximum batches to process per criterion type"),
    overview_only: bool = typer.Option(False, "--overview", help="Show processing overview only"),
    dsn: Optional[str] = typer.Option(None, help="Database connection string"),
    # Parallelization controls
    max_workers: int = typer.Option(10, help="Max concurrent workers for LLM calls (default: 10)"),
    criterion_workers: int = typer.Option(3, help="Max concurrent criterion types (default: 3)"),
    batch_workers: int = typer.Option(2, help="Max concurrent batches per criterion type (default: 2)"),
    # Rate limiting controls
    requests_per_minute: int = typer.Option(3000, help="OpenAI requests per minute limit (default: 3000)"),
    requests_per_day: int = typer.Option(200000, help="OpenAI requests per day limit (default: 200000)"),
    # Error handling
    max_retries: int = typer.Option(3, help="Max retries for failed LLM calls (default: 3)"),
    retry_delay: float = typer.Option(1.0, help="Base delay between retries in seconds (default: 1.0)"),
    # Database concurrency
    db_pool_size: int = typer.Option(20, help="Database connection pool size (default: 20)"),
    # Progress reporting
    progress_interval: int = typer.Option(50, help="Progress report every N records (default: 50)"),
):
    """Extract insights using parallel processing with advanced rate limiting and concurrency controls."""
    from .criterion_insights import ParallelCriterionInsightsProcessor
    
    typer.echo("🚀 Starting Parallel Criterion Insights Analysis")
    typer.echo("=" * 60)
    typer.echo(f"⚙️  Configuration:")
    typer.echo(f"   Max workers (LLM): {max_workers}")
    typer.echo(f"   Criterion workers: {criterion_workers}")
    typer.echo(f"   Batch workers: {batch_workers}")
    typer.echo(f"   Rate limit: {requests_per_minute}/min, {requests_per_day}/day")
    typer.echo(f"   DB pool size: {db_pool_size}")
    typer.echo(f"   Batch size: {batch_size}")
    typer.echo("=" * 60)
    
    processor = ParallelCriterionInsightsProcessor(
        batch_size=batch_size,
        error_log_dir="analysis_errors",
        # Parallelization settings
        max_workers=max_workers,
        criterion_workers=criterion_workers,
        batch_workers=batch_workers,
        # Rate limiting settings
        requests_per_minute=requests_per_minute,
        requests_per_day=requests_per_day,
        # Error handling settings
        max_retries=max_retries,
        retry_delay=retry_delay,
        # Database settings
        db_pool_size=db_pool_size,
        dsn=dsn,
        # Progress settings
        progress_interval=progress_interval,
    )
    
    if overview_only:
        typer.echo("📊 Criterion Insights Processing Overview")
        typer.echo("=" * 50)
        
        overview = processor.get_processing_overview()
        
        typer.echo(f"Total unprocessed records: {overview['total_unprocessed']}")
        typer.echo(f"Batch size: {overview['batch_size']}")
        typer.echo()
        
        if overview['unprocessed_counts']:
            typer.echo("Unprocessed counts by criterion type:")
            total_records = sum(overview['unprocessed_counts'].values())
            estimated_time = processor.estimate_processing_time(total_records)
            
            for criterion, count in overview['unprocessed_counts'].items():
                batches = overview['estimated_batches'][criterion]
                typer.echo(f"  {criterion}: {count} records (~{batches} batches)")
            
            typer.echo(f"\n⏱️  Estimated processing time: {estimated_time}")
            typer.echo(f"💰 Estimated cost: ${processor.estimate_cost(total_records):.2f}")
        else:
            typer.echo("✅ All criteria have been processed!")
        
        if overview['processing_progress']:
            typer.echo("\nPrevious processing progress:")
            for progress in overview['processing_progress']:
                typer.echo(f"  {progress['criterion_type']}: {progress['success_records']}/{progress['total_records']} successful")
        
        return
    
    # Start processing
    if criterion_type:
        typer.echo(f"Processing criterion type: {criterion_type}")
        results = processor.process_criterion_type_parallel(criterion_type, max_batches)
    else:
        typer.echo("Processing ALL criterion types in parallel...")
        results = processor.process_all_criteria_parallel(max_batches)
    
    # Display results
    if results["status"] == "no_data":
        typer.echo("ℹ️  No unprocessed criteria found - all have been analyzed!")
    elif results["status"] == "completed":
        typer.echo("\n🎉 Parallel processing completed!")
        typer.echo("=" * 50)
        
        # Display results for single criterion processing
        if isinstance(results, dict) and 'criterion_type' in results:
            typer.echo(f"✅ Criterion: {results['criterion_type']}")
            typer.echo(f"✅ Total processed: {results['processed_records']}")
            typer.echo(f"✅ Total successful: {results['successful_records']}")
            typer.echo(f"❌ Total errors: {results['error_records']}")
            typer.echo(f"⏱️  Processing time: {results['processing_time_seconds']:.1f} seconds")
            typer.echo(f"📊 Batches completed: {results['batches_completed']}")
            
            if results['errors']:
                typer.echo(f"\n❌ Error details:")
                for error in results['errors']:
                    typer.echo(f"   - {error}")
        
        # Display results for multi-criterion processing
        elif isinstance(results, dict) and 'overall_stats' in results:
            stats = results['overall_stats']
            typer.echo(f"✅ Total processed: {stats['total_processed']}")
            typer.echo(f"✅ Total successful: {stats['total_successful']}")
            typer.echo(f"❌ Total errors: {stats['total_errors']}")
            typer.echo(f"⏱️  Total processing time: {stats['total_time_seconds']:.1f} seconds")
            
            typer.echo(f"\n📊 By criterion type:")
            for criterion_type, criterion_stats in results.get('by_criterion', {}).items():
                typer.echo(f"   {criterion_type}: {criterion_stats['successful']}/{criterion_stats['total']} successful")
        
        else:
            typer.echo("⚠️  Unexpected results format")
            typer.echo(f"Results: {results}")
        
        typer.echo("\n🚀 Ready for more parallel processing!")
    
    elif results["status"] == "failed":
        typer.echo(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
        raise typer.Exit(code=1)
    
    # Save detailed results
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"insights_parallel_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        __import__('json').dump(results, f, indent=2, default=str)
    
    typer.echo(f"\n📋 Detailed results saved to {results_file}")
    
    # Show performance summary
    if results["status"] == "completed" and results['overall_stats']['total_processed'] > 0:
        total_time = results.get('total_processing_time_seconds', 0)
        if total_time > 0:
            records_per_second = results['overall_stats']['total_processed'] / total_time
            typer.echo(f"\n⚡ Performance: {records_per_second:.1f} records/second")
            
            if 'sequential_estimate' in results:
                speedup = results['sequential_estimate'] / total_time
                typer.echo(f"🚀 Speedup vs sequential: {speedup:.1f}x faster")

@app.command("insights-stats")
def insights_stats(
    dsn: Optional[str] = typer.Option(None, help="Database connection string"),
    criterion_type: Optional[str] = typer.Option(None, help="Filter by criterion type"),
    format: str = typer.Option("table", help="Output format: table, json, csv"),
):
    """Show statistics and insights from processed criteria analysis."""
    engine = get_engine(dsn)
    
    # Success rates by criterion and evidence type
    query = """
    SELECT 
        cc.criterion,
        cei.evidence_type,
        COUNT(*) as total_cases,
        AVG(CASE WHEN cc.aao_finding = 'met' THEN 1.0 ELSE 0.0 END) as success_rate,
        AVG(CASE WHEN cei.strength_assessment = 'strong' THEN 1.0 ELSE 0.0 END) as strong_evidence_rate
    FROM claimed_criteria cc
    JOIN criterion_evidence_insights cei ON cc.criterion_id = cei.criterion_id
    WHERE (:criterion_type IS NULL OR cc.criterion = :criterion_type)
    GROUP BY cc.criterion, cei.evidence_type
    HAVING COUNT(*) >= 5
    ORDER BY success_rate DESC, total_cases DESC
    """
    
    params = {"criterion_type": criterion_type}
    
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = result.fetchall()
        columns = result.keys()
    
    if not rows:
        typer.echo("No insights data found. Run 'insights' command first to process criteria.")
        return
    
    if format == "json":
        data = [dict(zip(columns, row)) for row in rows]
        typer.echo(__import__('json').dumps(data, indent=2, default=str))
    elif format == "csv":
        import csv
        import sys
        writer = csv.writer(sys.stdout)
        writer.writerow(columns)
        writer.writerows(rows)
    else:  # table format
        typer.echo("📊 Success Rates by Criterion and Evidence Type")
        typer.echo("=" * 80)
        
        # Header
        header = f"{'Criterion':<18} {'Evidence Type':<25} {'Cases':<8} {'Success %':<10} {'Strong %':<10}"
        typer.echo(header)
        typer.echo("-" * 80)
        
        # Data rows
        for row in rows:
            criterion, evidence_type, total_cases, success_rate, strong_rate = row
            success_pct = f"{success_rate*100:.1f}%" if success_rate else "N/A"
            strong_pct = f"{strong_rate*100:.1f}%" if strong_rate else "N/A"
            
            line = f"{criterion:<18} {evidence_type:<25} {total_cases:<8} {success_pct:<10} {strong_pct:<10}"
            typer.echo(line)
    
    # Most common rejection reasons
    rejection_query = """
    SELECT 
        crp.criterion_type,
        crp.rejection_category,
        COUNT(*) as frequency,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY crp.criterion_type), 1) as percentage
    FROM criterion_rejection_patterns crp
    WHERE (:criterion_type IS NULL OR crp.criterion_type = :criterion_type)
    GROUP BY crp.criterion_type, crp.rejection_category
    ORDER BY crp.criterion_type, frequency DESC
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(rejection_query), params)
        rejection_rows = result.fetchall()
    
    if rejection_rows and format == "table":
        typer.echo(f"\n📉 Most Common Rejection Reasons")
        typer.echo("=" * 60)
        
        current_criterion = None
        for row in rejection_rows[:20]:  # Show top 20
            criterion_type, rejection_category, frequency, percentage = row
            
            if criterion_type != current_criterion:
                if current_criterion is not None:
                    typer.echo()
                typer.echo(f"{criterion_type}:")
                current_criterion = criterion_type
            
            typer.echo(f"  {rejection_category:<30} {frequency:>3} ({percentage:>4.1f}%)")

@app.command("validate-evidence")
def validate_evidence_command(
    decision_id: Optional[int] = typer.Option(None, help="Specific decision ID to validate"),
    batch_size: int = typer.Option(100, help="Batch size for processing multiple decisions"),
    dsn: Optional[str] = typer.Option(None, help="Database connection string"),
):
    """Validate evidence dates against filing dates per instructions."""
    from .db import validate_evidence_calendar
    
    engine = get_engine(dsn)
    
    if decision_id:
        typer.echo(f"🗓️  Validating evidence calendar for decision {decision_id}")
        result = validate_evidence_calendar(engine, decision_id)
        
        if result["status"] == "no_filing_date":
            typer.echo(f"⚠️  No filing date found for decision {decision_id}")
        else:
            typer.echo(f"✅ Validated {result['evidence_checked']} evidence items")
            typer.echo(f"📅 Filing date: {result['filing_date']}")
            
            if result["post_filing_count"] > 0:
                typer.echo(f"🚨 Found {result['post_filing_count']} post-filing evidence items")
                for item in result["auto_tagged"]:
                    typer.echo(f"   - Evidence {item['evidence_id']}: {item['gap_days']} days after filing")
            else:
                typer.echo("✅ All evidence is pre-filing")
    else:
        typer.echo("🗓️  Validating evidence calendar for all decisions with filing dates")
        
        # Get decisions with filing dates
        with engine.connect() as conn:
            decisions = conn.execute(text("""
                SELECT decision_id, case_number, filing_date 
                FROM decisions 
                WHERE filing_date IS NOT NULL
                ORDER BY decision_id
            """)).fetchall()
        
        if not decisions:
            typer.echo("ℹ️  No decisions with filing dates found")
            return
        
        typer.echo(f"📊 Found {len(decisions)} decisions with filing dates")
        
        total_evidence = 0
        total_post_filing = 0
        processed_decisions = 0
        
        for decision_id, case_number, filing_date in decisions:
            result = validate_evidence_calendar(engine, decision_id)
            
            if result["status"] == "completed":
                total_evidence += result["evidence_checked"]
                total_post_filing += result["post_filing_count"]
                processed_decisions += 1
                
                if result["post_filing_count"] > 0:
                    typer.echo(f"🚨 {case_number}: {result['post_filing_count']}/{result['evidence_checked']} post-filing")
            
            # Progress update every batch_size
            if processed_decisions % batch_size == 0:
                typer.echo(f"📈 Progress: {processed_decisions}/{len(decisions)} decisions processed")
        
        typer.echo("\n" + "="*60)
        typer.echo("📊 EVIDENCE CALENDAR VALIDATION SUMMARY")
        typer.echo("="*60)
        typer.echo(f"✅ Decisions processed: {processed_decisions}")
        typer.echo(f"📋 Total evidence items: {total_evidence}")
        typer.echo(f"🚨 Post-filing evidence: {total_post_filing}")
        
        if total_evidence > 0:
            post_filing_rate = (total_post_filing / total_evidence) * 100
            typer.echo(f"📊 Post-filing rate: {post_filing_rate:.1f}%")

@app.command("mcp-server")
def mcp_server_command(
    dsn: Optional[str] = typer.Option(None, help="Postgres DSN (overrides DATABASE_URL)"),
):
    """Start the AAO ETL MCP server for agentic database exploration."""
    from .mcp_server import run_server

    typer.echo("🚀 Starting AAO ETL MCP server (STDIN/STDOUT)...", err=True)
    run_server(dsn)

def main():
    app()
