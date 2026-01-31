#!/usr/bin/env python3
"""
Embedding Model Benchmark for PAS

Compares loading times and semantic quality of different embedding models:
- all-mpnet-base-v2 (current, 768-dim, 110M params)
- all-MiniLM-L6-v2 (fast, 384-dim, 22M params)
- bge-small-en-v1.5 (balanced, 384-dim, 33M params)

Usage:
    python benchmark_embeddings.py

PAS Session: ccc329cd-fc54-423b-a7d5-2004adfaf7ee
Tags: performance, optimization, ml-loading
"""

import time
import statistics
from typing import Dict, List, Tuple, Any
import numpy as np

# Benchmark configuration
MODELS = [
    "sentence-transformers/all-mpnet-base-v2",      # Current (768-dim)
    "sentence-transformers/all-MiniLM-L6-v2",       # Fast (384-dim)
    "BAAI/bge-small-en-v1.5",                       # Balanced (384-dim)
]

# Representative PAS test data
TEST_QUERIES = [
    # Goal-like queries (what PAS uses for hypothesis matching)
    "Fix the bug where user sessions expire too early",
    "Add a new feature to track embedding model performance",
    "Refactor the authentication module for better security",
    
    # Law search queries (what search_relevant_laws uses)
    "cognitive overload reduction",
    "decision fatigue prevention", 
    "confirmation bias mitigation",
    
    # Code search queries (what query_codebase uses)
    "function that handles database connection pooling",
    "error handling in the API layer",
    "async operations and concurrency",
]

# Semantic similarity test pairs (expected to be similar)
SIMILARITY_PAIRS = [
    ("delete the file", "remove the document"),      # Synonyms
    ("user authentication", "login system"),         # Concepts
    ("fix the bug", "resolve the issue"),            # Paraphrase
    ("API endpoint", "REST route"),                  # Domain terms
    ("database query", "SQL operation"),             # Technical synonyms
]

# Dissimilar pairs (expected to NOT be similar) 
DISSIMILAR_PAIRS = [
    ("user authentication", "database connection"),
    ("fix the bug", "add new feature"),
    ("delete file", "create backup"),
]

def benchmark_loading_time(model_name: str, iterations: int = 3) -> Dict[str, float]:
    """Benchmark model loading time over multiple iterations."""
    from sentence_transformers import SentenceTransformer
    import gc
    
    times = []
    for i in range(iterations):
        # Force garbage collection to get clean measurement
        gc.collect()
        
        start = time.perf_counter()
        model = SentenceTransformer(model_name)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        # Clean up for next iteration
        del model
        gc.collect()
        
        print(f"  Iteration {i+1}: {elapsed:.2f}s")
    
    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }

def benchmark_inference_time(model_name: str, queries: List[str], iterations: int = 5) -> Dict[str, float]:
    """Benchmark embedding inference time."""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model.encode(queries)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "per_query": statistics.mean(times) / len(queries),
    }

def benchmark_semantic_quality(model_name: str) -> Dict[str, Any]:
    """Benchmark semantic similarity quality using known pairs."""
    from sentence_transformers import SentenceTransformer, util
    
    model = SentenceTransformer(model_name)
    
    # Test similar pairs - should have high similarity
    similar_scores = []
    for text1, text2 in SIMILARITY_PAIRS:
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        score = float(util.cos_sim(emb1, emb2)[0][0])
        similar_scores.append(score)
    
    # Test dissimilar pairs - should have low similarity
    dissimilar_scores = []
    for text1, text2 in DISSIMILAR_PAIRS:
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        score = float(util.cos_sim(emb1, emb2)[0][0])
        dissimilar_scores.append(score)
    
    # Quality metric: gap between similar and dissimilar
    quality_gap = statistics.mean(similar_scores) - statistics.mean(dissimilar_scores)
    
    return {
        "similar_mean": statistics.mean(similar_scores),
        "similar_min": min(similar_scores),
        "dissimilar_mean": statistics.mean(dissimilar_scores),
        "dissimilar_max": max(dissimilar_scores),
        "quality_gap": quality_gap,
        "similar_scores": similar_scores,
        "dissimilar_scores": dissimilar_scores,
    }

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model metadata."""
    from sentence_transformers import SentenceTransformer
    import gc
    
    model = SentenceTransformer(model_name)
    
    # Get embedding dimension
    sample_embedding = model.encode("test")
    dim = len(sample_embedding)
    
    # Estimate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    
    info = {
        "dimension": dim,
        "params_millions": total_params / 1_000_000,
    }
    
    del model
    gc.collect()
    
    return info

def run_full_benchmark() -> Dict[str, Dict]:
    """Run complete benchmark suite for all models."""
    results = {}
    
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        
        # Model info
        print("\n📊 Model Info...")
        info = get_model_info(model_name)
        print(f"  Dimension: {info['dimension']}")
        print(f"  Parameters: {info['params_millions']:.1f}M")
        
        # Loading time
        print("\n⏱️  Loading Time (3 iterations)...")
        loading = benchmark_loading_time(model_name, iterations=3)
        print(f"  Mean: {loading['mean']:.2f}s ± {loading['std']:.2f}s")
        
        # Inference time
        print("\n⚡ Inference Time (10 queries, 5 iterations)...")
        inference = benchmark_inference_time(model_name, TEST_QUERIES, iterations=5)
        print(f"  Total: {inference['mean']*1000:.1f}ms")
        print(f"  Per query: {inference['per_query']*1000:.2f}ms")
        
        # Semantic quality
        print("\n🎯 Semantic Quality...")
        quality = benchmark_semantic_quality(model_name)
        print(f"  Similar pairs avg: {quality['similar_mean']:.3f}")
        print(f"  Dissimilar pairs avg: {quality['dissimilar_mean']:.3f}")
        print(f"  Quality gap: {quality['quality_gap']:.3f}")
        
        results[model_name] = {
            "info": info,
            "loading": loading,
            "inference": inference,
            "quality": quality,
        }
    
    return results

def print_summary(results: Dict[str, Dict]):
    """Print comparison summary table."""
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    # Determine baseline (mpnet)
    baseline_name = "sentence-transformers/all-mpnet-base-v2"
    baseline = results.get(baseline_name, {})
    baseline_loading = baseline.get("loading", {}).get("mean", 1)
    baseline_inference = baseline.get("inference", {}).get("per_query", 1)
    baseline_quality = baseline.get("quality", {}).get("quality_gap", 1)
    
    print(f"\n{'Model':<45} {'Dim':<6} {'Load (s)':<10} {'Speed':<8} {'Quality':<10} {'Gap':>8}")
    print("-" * 90)
    
    for model_name, data in results.items():
        short_name = model_name.split("/")[-1]
        dim = data["info"]["dimension"]
        load_time = data["loading"]["mean"]
        per_query = data["inference"]["per_query"]
        quality_gap = data["quality"]["quality_gap"]
        
        # Calculate relative metrics
        speed_vs_baseline = baseline_loading / load_time if load_time > 0 else 0
        quality_vs_baseline = quality_gap / baseline_quality if baseline_quality > 0 else 0
        
        speed_str = f"{speed_vs_baseline:.1f}x" if speed_vs_baseline != 1 else "baseline"
        quality_str = f"{quality_vs_baseline:.0%}"
        
        print(f"{short_name:<45} {dim:<6} {load_time:<10.2f} {speed_str:<8} {quality_str:<10} {quality_gap:>8.3f}")
    
    print("\n📌 Decision Criteria:")
    print("  - Speed improvement worth it if quality_gap remains > 0.35")
    print("  - Dimension change (768→384) requires migration")
    print("  - Any model change requires re-embedding existing data")

def main():
    print("🚀 PAS Embedding Model Benchmark")
    print("Session: ccc329cd-fc54-423b-a7d5-2004adfaf7ee")
    print("\nThis benchmark compares loading times and semantic quality")
    print("to help decide if switching from all-mpnet-base-v2 is worthwhile.\n")
    
    try:
        results = run_full_benchmark()
        print_summary(results)
        
        print("\n✅ Benchmark complete!")
        print("\nNext steps:")
        print("1. Review quality gap - is the ~3% drop acceptable?")
        print("2. Review loading time improvement - is 5x worth it?")
        print("3. If switching: run migration to clear existing embeddings")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with: pip install sentence-transformers")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
