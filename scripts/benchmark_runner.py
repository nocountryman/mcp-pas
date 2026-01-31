#!/usr/bin/env python3
"""
PAS Benchmark Runner
Runs test cases and records results for comparing LLMs and prompt variants.
"""
import asyncio
import os
import sys
from datetime import datetime
from typing import Any
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)

def get_test_cases() -> list[dict]:
    """Fetch all benchmark test cases."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM benchmark_test_cases ORDER BY test_id")
    cases = cur.fetchall()
    conn.close()
    return cases

def compute_jaccard(declared: list[str], expected: list[str]) -> float:
    """Compute Jaccard similarity between two lists."""
    if not declared and not expected:
        return 1.0
    if not declared or not expected:
        return 0.0
    set_declared = set(s.lower() for s in declared)
    set_expected = set(s.lower() for s in expected)
    intersection = len(set_declared & set_expected)
    union = len(set_declared | set_expected)
    return intersection / union if union > 0 else 0.0

def record_result(
    test_case_id: str,
    session_id: str | None,
    model_name: str,
    prompt_variant: str,
    scope_accuracy: float,
    hypothesis_relevance: float,
    critique_coverage: float,
    tree_depth: int,
    declared_scope: list[str],
    notes: str = None
):
    """Record a benchmark result to the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO benchmark_results 
        (test_case_id, session_id, model_name, prompt_variant, 
         scope_accuracy, hypothesis_relevance, critique_coverage, tree_depth,
         declared_scope, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, composite_score
    """, (
        test_case_id, session_id, model_name, prompt_variant,
        scope_accuracy, hypothesis_relevance, critique_coverage, tree_depth,
        declared_scope, notes
    ))
    result = cur.fetchone()
    conn.commit()
    conn.close()
    return result

def get_summary():
    """Get benchmark summary."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM benchmark_summary")
    summary = cur.fetchall()
    conn.close()
    return summary

def print_summary():
    """Print the benchmark summary table."""
    summary = get_summary()
    if not summary:
        print("No benchmark results yet.")
        return
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Variant':<15} {'Tests':<6} {'Scope':<8} {'Rel':<8} {'Crit':<8} {'Score':<8}")
    print("-"*80)
    for row in summary:
        print(f"{row['model_name']:<25} {row['prompt_variant']:<15} {row['tests_run']:<6} "
              f"{row['avg_scope_accuracy'] or 0:.3f}    {row['avg_relevance'] or 0:.3f}    "
              f"{row['avg_critique_coverage'] or 0:.3f}    {row['avg_composite_score'] or 0:.1f}")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PAS Benchmark Runner")
    parser.add_argument("--list", action="store_true", help="List test cases")
    parser.add_argument("--summary", action="store_true", help="Show summary")
    parser.add_argument("--model", type=str, help="Model name to record")
    parser.add_argument("--variant", type=str, default="v1", help="Prompt variant")
    
    args = parser.parse_args()
    
    if args.list:
        cases = get_test_cases()
        print(f"\nBenchmark Test Cases ({len(cases)} total):")
        print("-"*60)
        for case in cases:
            print(f"  {case['test_id']}: [{case['category']}] {case['goal_text'][:50]}...")
        sys.exit(0)
    
    if args.summary:
        print_summary()
        sys.exit(0)
    
    print("Usage: python benchmark_runner.py --list | --summary")
    print("       For interactive testing, use PAS tools directly")
