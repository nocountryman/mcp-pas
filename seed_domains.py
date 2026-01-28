#!/usr/bin/env python3
"""
Seed script for v19 interview domain system.
Inserts predefined domains, dimensions, and question templates.
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# =============================================================================
# Interview Domains with Dimensions and Questions
# =============================================================================
DOMAINS = [
    {
        "domain_name": "ui_design",
        "description": "User interface design including visual aesthetics, layout, and user experience. Covers color schemes, typography, component design, and accessibility.",
        "example_goals": [
            "Design an amazing interface for my app",
            "Create a beautiful dashboard",
            "Build a modern UI for the landing page",
            "Redesign the user profile screen"
        ],
        "dimensions": [
            {
                "name": "colors",
                "description": "Color palette and visual mood",
                "priority": 10,
                "question": {
                    "template": "What color palette best represents your vision?",
                    "type": "complex",
                    "choices": [
                        {
                            "label": "A",
                            "description": "Dark & Moody - Deep blacks, dark grays, accent colors",
                            "pros": ["Modern feel", "Reduces eye strain", "Premium look"],
                            "cons": ["Limited outdoor visibility", "Can feel heavy"]
                        },
                        {
                            "label": "B",
                            "description": "Light & Airy - Whites, soft grays, subtle accents",
                            "pros": ["Clean feel", "Good readability", "Familiar to users"],
                            "cons": ["Can feel generic", "Harder to show hierarchy"]
                        },
                        {
                            "label": "C",
                            "description": "Vibrant & Colorful - Bold colors, gradients, energetic",
                            "pros": ["Memorable", "High engagement", "Brand differentiation"],
                            "cons": ["Can feel overwhelming", "Harder to maintain consistency"]
                        },
                        {
                            "label": "D",
                            "description": "Neutral & Professional - Muted tones, subtle accents",
                            "pros": ["Timeless", "Accessible", "Enterprise appropriate"],
                            "cons": ["Less memorable", "May lack personality"]
                        }
                    ]
                }
            },
            {
                "name": "layout",
                "description": "Page structure and content organization",
                "priority": 20,
                "question": {
                    "template": "How should content be organized on the page?",
                    "type": "complex",
                    "choices": [
                        {
                            "label": "A",
                            "description": "Dashboard Grid - Cards in a flexible grid layout",
                            "pros": ["Shows many items at once", "Scannable", "Responsive"],
                            "cons": ["Can feel busy", "Equal visual weight to all items"]
                        },
                        {
                            "label": "B",
                            "description": "Single Column - Linear, focused content flow",
                            "pros": ["Easy to follow", "Good for reading", "Mobile-first"],
                            "cons": ["Long scroll", "Limited information density"]
                        },
                        {
                            "label": "C",
                            "description": "Sidebar + Main - Navigation on side, content in center",
                            "pros": ["Clear navigation", "Familiar pattern", "Efficient"],
                            "cons": ["Less horizontal space", "Desktop-optimized"]
                        }
                    ]
                }
            },
            {
                "name": "typography",
                "description": "Font choices and text styling",
                "priority": 30,
                "question": {
                    "template": "What typography style fits your brand?",
                    "type": "simple",
                    "choices": [
                        {
                            "label": "A",
                            "description": "Modern Sans-Serif (Inter, Roboto) - Clean, neutral, tech-forward"
                        },
                        {
                            "label": "B",
                            "description": "Classic Serif (Georgia, Playfair) - Traditional, authoritative, editorial"
                        },
                        {
                            "label": "C",
                            "description": "Rounded/Friendly (Nunito, Poppins) - Approachable, casual, playful"
                        }
                    ]
                }
            },
            {
                "name": "interactions",
                "description": "Animations and micro-interactions",
                "priority": 40,
                "question": {
                    "template": "How interactive should the experience feel?",
                    "type": "simple",
                    "choices": [
                        {
                            "label": "A",
                            "description": "Rich Animations - Smooth transitions, hover effects, loading animations"
                        },
                        {
                            "label": "B",
                            "description": "Subtle Feedback - Minimal but clear state changes"
                        },
                        {
                            "label": "C",
                            "description": "Static/Fast - No animations, instant response, accessibility focus"
                        }
                    ]
                }
            },
            {
                "name": "accessibility",
                "description": "Accessibility and inclusive design",
                "priority": 50,
                "is_required": False,
                "question": {
                    "template": "What accessibility level is needed?",
                    "type": "simple",
                    "choices": [
                        {
                            "label": "A",
                            "description": "WCAG AAA - Maximum accessibility, all users supported"
                        },
                        {
                            "label": "B",
                            "description": "WCAG AA - Standard compliance, good for most needs"
                        },
                        {
                            "label": "C",
                            "description": "Best effort - Basic accessibility without strict compliance"
                        }
                    ]
                }
            }
        ]
    },
    {
        "domain_name": "coding_task",
        "description": "Implementation tasks including bug fixes, feature additions, and refactoring. Covers scope, constraints, and success criteria.",
        "example_goals": [
            "Fix the authentication bug",
            "Add caching to the API",
            "Refactor the user service",
            "Implement the new payment flow"
        ],
        "dimensions": [
            {
                "name": "scope",
                "description": "Boundaries of the change",
                "priority": 10,
                "question": {
                    "template": "What is the scope of this change?",
                    "type": "complex",
                    "choices": [
                        {
                            "label": "A",
                            "description": "Minimal - Single file, isolated change",
                            "pros": ["Low risk", "Fast to implement", "Easy to review"],
                            "cons": ["May not fully solve the problem"]
                        },
                        {
                            "label": "B",
                            "description": "Moderate - Multiple files, single module",
                            "pros": ["Balanced approach", "Addresses related concerns"],
                            "cons": ["Requires more testing", "More review needed"]
                        },
                        {
                            "label": "C",
                            "description": "Broad - Cross-module, architectural change",
                            "pros": ["Comprehensive solution", "Addresses root cause"],
                            "cons": ["Higher risk", "Longer timeline", "More stakeholders"]
                        }
                    ]
                }
            },
            {
                "name": "constraints",
                "description": "Technical or business constraints",
                "priority": 20,
                "question": {
                    "template": "Are there constraints I should know about?",
                    "type": "complex",
                    "choices": [
                        {
                            "label": "A",
                            "description": "Time-sensitive - Must ship by specific deadline",
                            "pros": ["Clear priority", "Focus on essentials"],
                            "cons": ["May need to cut corners", "Tech debt risk"]
                        },
                        {
                            "label": "B",
                            "description": "Backward compatible - Must not break existing users",
                            "pros": ["Safe rollout", "No migration needed"],
                            "cons": ["May limit design options", "Complexity"]
                        },
                        {
                            "label": "C",
                            "description": "No major constraints - Quality over speed",
                            "pros": ["Thorough solution", "Future-proof"],
                            "cons": ["Longer timeline"]
                        }
                    ]
                }
            },
            {
                "name": "success_criteria",
                "description": "How to know the task is complete",
                "priority": 30,
                "question": {
                    "template": "How will you know this is done correctly?",
                    "type": "complex",
                    "choices": [
                        {
                            "label": "A",
                            "description": "Tests pass - Unit/integration tests verify behavior",
                            "pros": ["Objective", "Repeatable", "Regression-safe"],
                            "cons": ["Tests may not cover all cases"]
                        },
                        {
                            "label": "B",
                            "description": "Manual verification - Specific scenario works as expected",
                            "pros": ["User-focused", "Catches UX issues"],
                            "cons": ["Not repeatable", "May miss edge cases"]
                        },
                        {
                            "label": "C",
                            "description": "Both tests + manual - Full coverage",
                            "pros": ["Comprehensive", "High confidence"],
                            "cons": ["Takes longer"]
                        }
                    ]
                }
            }
        ]
    },
    {
        "domain_name": "api_design",
        "description": "API and backend design including endpoints, authentication, and data contracts.",
        "example_goals": [
            "Design the REST API for users",
            "Add authentication to the API",
            "Create the data export endpoint",
            "Implement rate limiting"
        ],
        "dimensions": [
            {
                "name": "endpoints",
                "description": "API surface and resource structure",
                "priority": 10,
                "question": {
                    "template": "What resources does this API expose?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Single resource - One entity type (users, products, etc.)"},
                        {"label": "B", "description": "Related resources - Multiple entities with relationships"},
                        {"label": "C", "description": "Aggregate - Dashboard/report style endpoint"}
                    ]
                }
            },
            {
                "name": "auth",
                "description": "Authentication and authorization approach",
                "priority": 20,
                "question": {
                    "template": "What authentication is needed?",
                    "type": "complex",
                    "choices": [
                        {
                            "label": "A",
                            "description": "JWT tokens - Stateless, self-contained",
                            "pros": ["Scalable", "No server storage", "Works cross-domain"],
                            "cons": ["Token size", "Revocation complexity"]
                        },
                        {
                            "label": "B",
                            "description": "Session cookies - Server-side sessions",
                            "pros": ["Easy revocation", "Smaller payload"],
                            "cons": ["Requires session storage", "CSRF risks"]
                        },
                        {
                            "label": "C",
                            "description": "API keys - Simple, machine-to-machine",
                            "pros": ["Simple", "Good for integrations"],
                            "cons": ["Less secure for user auth", "No user identity"]
                        },
                        {
                            "label": "D",
                            "description": "Public - No authentication needed",
                            "pros": ["Simple", "Low friction"],
                            "cons": ["No access control", "Rate limiting critical"]
                        }
                    ]
                }
            },
            {
                "name": "versioning",
                "description": "API versioning strategy",
                "priority": 30,
                "is_required": False,
                "question": {
                    "template": "How should API versions be handled?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "URL versioning - /v1/users, /v2/users"},
                        {"label": "B", "description": "Header versioning - Accept: application/vnd.api+json; version=1"},
                        {"label": "C", "description": "No versioning - Single evolving API"}
                    ]
                }
            }
        ]
    },
    {
        "domain_name": "planning",
        "description": "Project planning and architecture decisions. High-level design and strategic choices.",
        "example_goals": [
            "Plan the new feature roadmap",
            "Design the system architecture",
            "Decide on the tech stack",
            "Create the migration strategy"
        ],
        "dimensions": [
            {
                "name": "timeline",
                "description": "Expected timeline and phases",
                "priority": 10,
                "question": {
                    "template": "What timeline are you working with?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Urgent - Days/week, need quick wins"},
                        {"label": "B", "description": "Standard - Weeks/month, normal planning"},
                        {"label": "C", "description": "Long-term - Months/quarter, strategic planning"}
                    ]
                }
            },
            {
                "name": "stakeholders",
                "description": "Who is involved in decisions",
                "priority": 20,
                "question": {
                    "template": "Who needs to be involved in this decision?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Just me - Technical decision I can make"},
                        {"label": "B", "description": "Team - Need team consensus or review"},
                        {"label": "C", "description": "Cross-functional - Product, design, or leadership input needed"}
                    ]
                }
            }
        ]
    }
]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate 768-dimensional embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Loading sentence-transformers model...")
        model = SentenceTransformer('all-mpnet-base-v2')
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        return embeddings.tolist()
    
    except ImportError:
        print("WARNING: sentence-transformers not installed.")
        print("Using zero vectors as placeholders.")
        return [[0.0] * 768 for _ in texts]


def seed_domains():
    """Insert interview domains, dimensions, and questions."""
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set")
    
    conn = psycopg2.connect(database_url)
    register_vector(conn)
    cur = conn.cursor()
    
    # Check existing domains
    cur.execute("SELECT COUNT(*) FROM interview_domains")
    if cur.fetchone()[0] > 0:
        response = input("Domains exist. Clear and reseed? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborting.")
            conn.close()
            return
        
        cur.execute("TRUNCATE interview_domains CASCADE")
        print("Cleared existing domains.")
    
    # Generate embeddings for domains
    domain_texts = [f"{d['domain_name']}: {d['description']}" for d in DOMAINS]
    domain_embeddings = get_embeddings(domain_texts)
    
    # Insert domains and their dimensions
    for domain, embedding in zip(DOMAINS, domain_embeddings):
        cur.execute(
            """
            INSERT INTO interview_domains (domain_name, description, example_goals, embedding)
            VALUES (%s, %s, %s, %s::vector)
            RETURNING id
            """,
            (domain['domain_name'], domain['description'], domain['example_goals'], embedding)
        )
        domain_id = cur.fetchone()[0]
        print(f"  ✓ Domain: {domain['domain_name']}")
        
        # Insert dimensions
        for dim in domain.get('dimensions', []):
            cur.execute(
                """
                INSERT INTO interview_dimensions (domain_id, dimension_name, description, is_required, priority)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    domain_id,
                    dim['name'],
                    dim['description'],
                    dim.get('is_required', True),
                    dim.get('priority', 50)
                )
            )
            dim_id = cur.fetchone()[0]
            
            # Insert question if present
            if 'question' in dim:
                q = dim['question']
                cur.execute(
                    """
                    INSERT INTO interview_questions (dimension_id, question_template, question_type, choices)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (dim_id, q['template'], q['type'], json.dumps(q['choices']))
                )
            
            print(f"      ✓ Dimension: {dim['name']}")
    
    conn.commit()
    print(f"\n✅ Seeded {len(DOMAINS)} domains with dimensions and questions.")
    conn.close()


if __name__ == "__main__":
    seed_domains()
