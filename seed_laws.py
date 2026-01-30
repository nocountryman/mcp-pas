#!/usr/bin/env python3
"""
Seed script for scientific_laws table.
Inserts foundational software engineering principles with semantic embeddings.
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Software Engineering Laws
# =============================================================================
LAWS = [
    {
        "law_name": "Conway's Law",
        "definition": "Organizations which design systems are constrained to produce designs which are copies of the communication structures of these organizations. Software architecture mirrors organizational structure.",
        "scientific_weight": 0.9,
        "failure_modes": [
            "Team silos creating unnecessary API boundaries",
            "Organizational changes not reflected in architecture",
            "Communication bottlenecks becoming system bottlenecks"
        ]
    },
    {
        "law_name": "Gall's Law",
        "definition": "A complex system that works is invariably found to have evolved from a simple system that worked. A complex system designed from scratch never works and cannot be patched up to make it work.",
        "scientific_weight": 0.85,
        "failure_modes": [
            "Attempting to build complex solution from scratch",
            "Skipping incremental validation steps",
            "Over-engineering initial implementation"
        ]
    },
    {
        "law_name": "Postel's Law (Robustness Principle)",
        "definition": "Be conservative in what you send, be liberal in what you accept. Programs should handle edge cases and malformed input gracefully while producing strict, well-formed output.",
        "scientific_weight": 0.8,
        "failure_modes": [
            "Rejecting valid but unusual input formats",
            "Producing inconsistent or malformed output",
            "Not handling null/empty/whitespace edge cases"
        ]
    },
    {
        "law_name": "Hick's Law",
        "definition": "The time it takes to make a decision increases logarithmically with the number of choices. Reducing options speeds up user decisions in interface design.",
        "scientific_weight": 0.75,
        "failure_modes": [
            "Presenting too many options at once",
            "Hiding critical choices in deep menus",
            "Not providing sensible defaults"
        ]
    },
    {
        "law_name": "CAP Theorem (Brewer's Theorem)",
        "definition": "A distributed data store can provide only two of the following three guarantees: Consistency, Availability, and Partition tolerance. You must choose which two to prioritize.",
        "scientific_weight": 0.95,
        "failure_modes": [
            "Attempting to guarantee all three properties",
            "Not explicitly choosing trade-offs",
            "Ignoring network partition scenarios"
        ]
    },
    {
        "law_name": "Brooks' Law",
        "definition": "Adding manpower to a late software project makes it later. Communication overhead grows quadratically with team size, and new members require ramp-up time.",
        "scientific_weight": 0.85,
        "failure_modes": [
            "Adding developers to accelerate late project",
            "Underestimating onboarding overhead",
            "Ignoring communication complexity growth"
        ]
    },
    {
        "law_name": "Kernighan's Law",
        "definition": "Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are by definition not smart enough to debug it.",
        "scientific_weight": 0.8,
        "failure_modes": [
            "Writing overly clever or terse code",
            "Sacrificing readability for brevity",
            "Complex one-liners instead of clear steps"
        ]
    },
    {
        "law_name": "Pareto Principle (80/20 Rule)",
        "definition": "Roughly 80% of effects come from 20% of causes. In software, 80% of bugs come from 20% of the code, and 80% of users use only 20% of features.",
        "scientific_weight": 0.75,
        "failure_modes": [
            "Equal effort on all features regardless of usage",
            "Not focusing on critical 20% code paths",
            "Over-polishing rarely-used features"
        ]
    },
    {
        "law_name": "Goodhart's Law",
        "definition": "When a measure becomes a target, it ceases to be a good measure. Optimizing for metrics leads to gaming behavior rather than genuine improvement.",
        "scientific_weight": 0.85,
        "failure_modes": [
            "Optimizing for metrics over real outcomes",
            "Creating perverse incentives from targets",
            "Gaming measurements instead of improving"
        ]
    },
    {
        "law_name": "Hyrum's Law",
        "definition": "With a sufficient number of users of an API, all observable behaviors of your system will be depended on by somebody. Any change breaks someone's workflow.",
        "scientific_weight": 0.9,
        "failure_modes": [
            "Implicit API contracts being depended upon",
            "Undocumented behaviors becoming requirements",
            "Breaking changes for edge case users"
        ]
    },
    {
        "law_name": "Linus's Law",
        "definition": "Given enough eyeballs, all bugs are shallow. With sufficient code reviewers, most problems will be caught and fixed quickly.",
        "scientific_weight": 0.7,
        "failure_modes": [
            "Insufficient code review coverage",
            "Reviewers not empowered to block",
            "Review fatigue on large changes"
        ]
    },
    {
        "law_name": "Occam's Razor",
        "definition": "The simplest explanation is usually the correct one. In software, prefer simpler solutions over complex ones when both solve the problem.",
        "scientific_weight": 0.8,
        "failure_modes": [
            "Choosing complex solution when simple exists",
            "Premature optimization or abstraction",
            "Over-architecting for hypothetical scale"
        ]
    },
    {
        "law_name": "Amdahl's Law",
        "definition": "The speedup of a program using multiple processors is limited by the fraction of the program that cannot be parallelized. Sequential bottlenecks limit parallelization gains.",
        "scientific_weight": 0.9,
        "failure_modes": [
            "Parallelizing without addressing sequential bottleneck",
            "Expecting linear speedup from more threads",
            "Ignoring synchronization overhead"
        ]
    },
    {
        "law_name": "Hofstadter's Law",
        "definition": "It always takes longer than you expect, even when you take into account Hofstadter's Law. Software estimation is inherently difficult due to unknown unknowns.",
        "scientific_weight": 0.75,
        "failure_modes": [
            "Underestimating task complexity",
            "Not accounting for unknown unknowns",
            "Optimistic estimates without buffer"
        ]
    },
    {
        "law_name": "Wirth's Law",
        "definition": "Software is getting slower more rapidly than hardware is getting faster. Feature creep and abstraction layers often negate hardware improvements.",
        "scientific_weight": 0.7,
        "failure_modes": [
            "Adding abstraction layers without need",
            "Feature creep negating performance gains",
            "Ignoring efficiency for convenience"
        ]
    },
    # =========================================================================
    # v44b: Psychology Laws for Requirement Extraction
    # Peer-reviewed techniques for understanding user intent
    # =========================================================================
    {
        "law_name": "Hedging Marker Detection",
        "definition": "Modal auxiliaries (might, may, could) and epistemic markers (suggest, seem, probably) indicate uncertain requirements needing clarification. High hedge density = underspecified scope. Source: CoNLL-2010, BioScope Corpus.",
        "scientific_weight": 0.95,
        "law_domain": "psychology",
        "failure_modes": [
            "Treating hedged statements as firm requirements",
            "Missing implicit uncertainty in requests",
            "Not following up on vague language"
        ]
    },
    {
        "law_name": "Illocutionary Force Detection",
        "definition": "User utterances carry illocutionary force (Searle 1969). Directives ('I want', 'should have') signal explicit requirements. Expressives ('I hate', 'frustrated') reveal pain points. Assertives about current state reveal implicit constraints.",
        "scientific_weight": 0.99,
        "law_domain": "psychology",
        "failure_modes": [
            "Treating all statements as equal weight",
            "Missing emotional signals (expressives)",
            "Ignoring state descriptions as constraints"
        ]
    },
    {
        "law_name": "Kano Requirement Categorization",
        "definition": "Requirements affect satisfaction differently. Must-haves cause dissatisfaction only when absent. Delighters create disproportionate satisfaction. Performance scales linearly. Time erodes categories. Source: Kano 1984.",
        "scientific_weight": 0.95,
        "law_domain": "psychology",
        "failure_modes": [
            "Treating all requirements as must-haves",
            "Missing delighter opportunities",
            "Not recognizing satisfaction asymmetry"
        ]
    },
    {
        "law_name": "Gain-Loss Framing Analysis",
        "definition": "Users frame requirements as gains ('save time', 'faster') or losses ('avoid errors', 'prevent crashes'). Loss-framed indicate higher priority due to loss aversion (Kahneman 1979). Detect frame to assess true priority.",
        "scientific_weight": 0.99,
        "law_domain": "psychology",
        "failure_modes": [
            "Treating gain/loss frames equally",
            "Missing urgency signals in loss-framed requests",
            "Not probing deeper on avoidance language"
        ]
    },
    {
        "law_name": "MoSCoW Priority Classification",
        "definition": "Requirements exist in priority tiers. Must/required/critical = non-negotiable. Should = important. Nice-to-have/ideally = optional. Won't = explicit exclusion. Source: DSDM/BABOK.",
        "scientific_weight": 0.90,
        "law_domain": "psychology",
        "failure_modes": [
            "Missing priority signals in language",
            "Not clarifying ambiguous priority",
            "Treating 'nice to have' as must-have"
        ]
    },
    {
        "law_name": "Loss Aversion Principle",
        "definition": "Losses feel ~2x worse than equivalent gains. Users emphasize avoiding negatives over gaining positives. 'Don't want' statements reveal core requirements. Source: Kahneman & Tversky 1979 Prospect Theory.",
        "scientific_weight": 0.99,
        "law_domain": "psychology",
        "failure_modes": [
            "Dismissing negative requirements",
            "Not weighting avoidance language higher",
            "Missing implicit fears in requests"
        ]
    }
]



def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate 768-dimensional embeddings using sentence-transformers.
    Uses all-mpnet-base-v2 which produces 768-dim vectors.
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Loading sentence-transformers model (all-mpnet-base-v2)...")
        model = SentenceTransformer('all-mpnet-base-v2')
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        return embeddings.tolist()
    
    except ImportError:
        print("WARNING: sentence-transformers not installed.")
        print("Using zero vectors as placeholders.")
        print("Install with: pip install sentence-transformers")
        return [[0.0] * 768 for _ in texts]


def seed_laws():
    """Insert scientific laws into the database with embeddings."""
    
    # Connect to database
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set in environment")
    
    conn = psycopg2.connect(database_url)
    register_vector(conn)
    cur = conn.cursor()
    
    # Check if laws already exist
    cur.execute("SELECT COUNT(*) FROM scientific_laws")
    existing_count = cur.fetchone()[0]
    
    if existing_count > 0:
        print(f"Found {existing_count} existing laws.")
        response = input("Do you want to clear and reseed? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborting seed operation.")
            conn.close()
            return
        
        cur.execute("TRUNCATE scientific_laws RESTART IDENTITY CASCADE")
        print("Cleared existing laws.")
    
    # Generate embeddings
    texts = [f"{law['law_name']}: {law['definition']}" for law in LAWS]
    embeddings = get_embeddings(texts)
    
    # Prepare data for insertion
    data = [
        (
            law["law_name"],
            law["definition"],
            law["scientific_weight"],
            law.get("failure_modes", []),  # v14c.1
            law.get("law_domain"),  # v44b: psychology, engineering, etc.
            embedding
        )
        for law, embedding in zip(LAWS, embeddings)
    ]
    
    # Insert laws
    print("Inserting laws into database...")
    execute_values(
        cur,
        """
        INSERT INTO scientific_laws (law_name, definition, scientific_weight, failure_modes, law_domain, embedding)
        VALUES %s
        """,
        data,
        template="(%s, %s, %s, %s, %s, %s::vector)"
    )

    
    conn.commit()
    print(f"✅ Successfully seeded {len(LAWS)} scientific laws.")
    
    # Verify insertion
    cur.execute("SELECT law_name, scientific_weight FROM scientific_laws ORDER BY scientific_weight DESC")
    print("\nSeeded Laws (by weight):")
    print("-" * 50)
    for row in cur.fetchall():
        print(f"  {row[1]:.2f}  {row[0]}")
    
    conn.close()


if __name__ == "__main__":
    seed_laws()
