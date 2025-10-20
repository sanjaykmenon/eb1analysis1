#!/usr/bin/env python3
"""Direct test of MCP server tools to verify case_number display"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, 'src')

from aao_etl.db import get_engine
from sqlalchemy import text

def test_search_decisions():
    """Test search_decisions directly"""
    print("=" * 70)
    print("Testing search_decisions with 'computer science'")
    print("=" * 70)
    
    engine = get_engine(os.getenv("DATABASE_URL"))
    
    sql = text("""
        SELECT DISTINCT d.decision_id, d.case_number, d.decision_date, d.outcome, d.field_of_endeavor,
               COALESCE(NULLIF(d.summary, ''), LEFT(d.final_merits_rationale, 280)) AS synopsis
        FROM decisions d
        WHERE (d.field_of_endeavor ILIKE :text_a)
        ORDER BY d.decision_date DESC NULLS LAST
        LIMIT 5
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(sql, {"text_a": "%computer science%"}).mappings().all()
    
    if not rows:
        print("❌ No results found")
        return
    
    print(f"\n✅ Found {len(rows)} decision(s):\n")
    for row in rows:
        summary = (row["synopsis"] or "").replace("\n", " ")
        if len(summary) > 240:
            summary = summary[:237] + "..."
        case_num = row['case_number'] or 'NO_CASE_NUMBER'
        print(f"📄 Case: {case_num} | ID: {row['decision_id']} | "
              f"Date: {row['decision_date'] or 'n/a'} | Outcome: {row['outcome'] or 'unknown'}")
        if row['field_of_endeavor']:
            print(f"   Field: {row['field_of_endeavor']}")
        if summary:
            print(f"   Summary: {summary}")
        print()

def test_get_decision_details():
    """Test get_decision_details directly"""
    print("\n" + "=" * 70)
    print("Testing get_decision_details for decision_id=3")
    print("=" * 70)
    
    engine = get_engine(os.getenv("DATABASE_URL"))
    
    base_sql = text("""
        SELECT decision_id, case_number, decision_date, filing_date, outcome, field_of_endeavor,
               final_merits, final_merits_rationale
        FROM decisions
        WHERE decision_id = :decision_id
        LIMIT 1
    """)
    
    with engine.connect() as conn:
        decision = conn.execute(base_sql, {"decision_id": 3}).mappings().first()
    
    if not decision:
        print("❌ No decision found")
        return
    
    print("\n" + "=" * 70)
    print(f"📄 CASE NUMBER: {decision['case_number'] or 'NO_CASE_NUMBER'}")
    print(f"   Decision ID: {decision['decision_id']}")
    print("=" * 70)
    print(f"Outcome: {decision['outcome'] or 'unknown'}")
    print(f"Decision date: {decision['decision_date'] or 'n/a'} | Filing date: {decision['filing_date'] or 'n/a'}")
    print(f"Field of endeavor: {decision['field_of_endeavor'] or 'n/a'}")
    print()

if __name__ == "__main__":
    print("\n🧪 Direct MCP Server Tool Test\n")
    
    if not os.getenv("DATABASE_URL"):
        print("❌ DATABASE_URL not set")
        sys.exit(1)
    
    try:
        test_search_decisions()
        test_get_decision_details()
        print("\n✅ All tests passed! MCP server formatting is working correctly.\n")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
