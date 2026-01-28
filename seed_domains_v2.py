#!/usr/bin/env python3
"""
Seed script for v19 interview domain system - V2 Research-Grounded Expansion.
Based on: Nielsen Heuristics, Cognitive Load Theory, UX Laws, Film Production Standards, WCAG 2.1 AA.
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
import json

load_dotenv()

# =============================================================================
# RESEARCH-GROUNDED DOMAINS
# =============================================================================

DOMAINS = [
    # =========================================================================
    # DOMAIN 1: UX_FOUNDATIONS (Based on Nielsen's 10 Heuristics)
    # =========================================================================
    {
        "domain_name": "ux_foundations",
        "description": "Core UX principles based on Nielsen's 10 Usability Heuristics. Covers visibility, feedback, user control, consistency, and error handling.",
        "example_goals": [
            "Design an intuitive user experience",
            "Make the app easy to use",
            "Improve usability of the interface",
            "Create a user-friendly design"
        ],
        "dimensions": [
            {
                "name": "visibility_of_status",
                "description": "System feedback and state visibility (Nielsen #1)",
                "priority": 10,
                "question": {
                    "template": "How should users know what's happening in the system?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Always visible - Progress bars, loading states, status badges everywhere", "pros": ["Users never guess", "High confidence"], "cons": ["Visual noise", "Screen clutter"]},
                        {"label": "B", "description": "Contextual - Show status only when relevant (during actions)", "pros": ["Clean interface", "Focus on task"], "cons": ["May miss updates"]},
                        {"label": "C", "description": "On-demand - User can check status but not auto-shown", "pros": ["Minimal UI", "User in control"], "cons": ["Requires user effort"]}
                    ]
                }
            },
            {
                "name": "match_real_world",
                "description": "Using familiar language and concepts (Nielsen #2)",
                "priority": 20,
                "question": {
                    "template": "What mental model should guide the interface?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Domain-native - Use industry terminology (e.g., 'Pre-production', 'Call Sheet')", "pros": ["Familiar to target users", "Professional feel"], "cons": ["Harder for beginners"]},
                        {"label": "B", "description": "Simplified - Plain language, avoid jargon", "pros": ["Accessible to all", "Lower learning curve"], "cons": ["May feel less professional"]},
                        {"label": "C", "description": "Hybrid - Plain by default, domain terms introduced gradually", "pros": ["Best of both", "Teaches users"], "cons": ["More complex to design"]}
                    ]
                }
            },
            {
                "name": "user_control",
                "description": "Undo, escape, and user freedom (Nielsen #3)",
                "priority": 30,
                "question": {
                    "template": "How easy should it be to undo actions?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Full history - Unlimited undo, version history for everything"},
                        {"label": "B", "description": "Recent only - Undo last few actions, snapshots for major changes"},
                        {"label": "C", "description": "Confirm only - Confirmation dialogs for destructive actions, no undo"}
                    ]
                }
            },
            {
                "name": "flexibility_efficiency",
                "description": "Supporting novice and expert users (Nielsen #7)",
                "priority": 40,
                "question": {
                    "template": "How should the interface serve both beginners and experts?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Progressive unlock - Features reveal as user skill develops", "pros": ["Never overwhelming", "Natural learning"], "cons": ["Experts may feel gated"]},
                        {"label": "B", "description": "Mode toggle - 'Simple' vs 'Pro' mode switch", "pros": ["Clear separation", "User choice"], "cons": ["Two UIs to maintain"]},
                        {"label": "C", "description": "All visible, keyboard shortcuts - Everything shown, accelerators for pros", "pros": ["No hidden features"], "cons": ["Busy interface"]},
                        {"label": "D", "description": "Adaptive - AI detects skill level and adjusts", "pros": ["Personalized"], "cons": ["Complex, may misjudge"]}
                    ]
                }
            },
            {
                "name": "error_prevention",
                "description": "Preventing mistakes before they happen (Nielsen #5)",
                "priority": 50,
                "question": {
                    "template": "How proactively should errors be prevented?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Smart defaults + validation - Pre-fill sensible values, validate in real-time", "pros": ["Fewer errors", "Less friction"], "cons": ["May feel restrictive"]},
                        {"label": "B", "description": "Warnings only - Let user do anything, warn on risky actions", "pros": ["Full freedom", "Power users happy"], "cons": ["More user errors"]},
                        {"label": "C", "description": "AI guardrails - AI catches mistakes and suggests fixes", "pros": ["Intelligent help"], "cons": ["AI may be wrong"]}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 2: COGNITIVE_LOAD (Based on Sweller's CLT, Hick's Law, Miller's Law)
    # =========================================================================
    {
        "domain_name": "cognitive_load",
        "description": "Managing mental effort based on Cognitive Load Theory, Hick's Law (decision time), and Miller's Law (7±2 items).",
        "example_goals": [
            "Simplify the complex workflow",
            "Make the interface less overwhelming",
            "Reduce user confusion",
            "Create a clean, focused experience"
        ],
        "dimensions": [
            {
                "name": "information_density",
                "description": "How much information per screen (Miller's Law: 7±2 items)",
                "priority": 10,
                "question": {
                    "template": "How dense should the information be?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Minimal (3-5 items) - One task, one screen", "pros": ["Very focused", "Low cognitive load"], "cons": ["Many clicks", "Slow for experts"]},
                        {"label": "B", "description": "Balanced (5-7 items) - Core info visible, details on demand", "pros": ["Good for most users", "Miller's Law optimized"], "cons": ["Still requires prioritization"]},
                        {"label": "C", "description": "Dense (10+ items) - Data-rich, power user interface", "pros": ["Efficient for experts", "See everything"], "cons": ["Overwhelming for beginners"]}
                    ]
                }
            },
            {
                "name": "choice_count",
                "description": "Number of options per decision point (Hick's Law)",
                "priority": 20,
                "question": {
                    "template": "How many choices should users face at once?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Minimal (2-3) - Binary or simple choices only", "pros": ["Fast decisions", "Clear path"], "cons": ["May feel limiting"]},
                        {"label": "B", "description": "Standard (4-6) - Enough variety, still manageable", "pros": ["Good balance", "Hick's Law optimized"], "cons": ["Requires good labeling"]},
                        {"label": "C", "description": "Extensive (7+) - All options visible", "pros": ["No hidden options"], "cons": ["Decision paralysis risk"]}
                    ]
                }
            },
            {
                "name": "progressive_disclosure",
                "description": "Revealing complexity gradually (Jakob Nielsen, 1980s)",
                "priority": 30,
                "question": {
                    "template": "How should complexity be revealed?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Project-driven - Features unlock when your project needs them", "pros": ["Just-in-time learning", "Never overwhelming"], "cons": ["May delay discovery"]},
                        {"label": "B", "description": "Skill-based - Unlock as user demonstrates capability", "pros": ["Matches user growth"], "cons": ["May frustrate fast learners"]},
                        {"label": "C", "description": "Manual expand - 'Show advanced' toggles everywhere", "pros": ["User chooses", "Predictable"], "cons": ["Extra clicks"]},
                        {"label": "D", "description": "All visible - No hiding, use visual hierarchy instead", "pros": ["No surprises"], "cons": ["Can overwhelm"]}
                    ]
                }
            },
            {
                "name": "chunking_strategy",
                "description": "How information is grouped into digestible pieces",
                "priority": 40,
                "question": {
                    "template": "How should information be grouped?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "By workflow phase - Group by when it's used"},
                        {"label": "B", "description": "By entity type - Group by what it is (people, places, things)"},
                        {"label": "C", "description": "By frequency - Most used first, rare stuff hidden"}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 3: LEARNING_SYSTEM (Based on Gamification Research, Duolingo)
    # =========================================================================
    {
        "domain_name": "learning_system",
        "description": "Teaching users through the interface. Based on gamification research and Self-Determination Theory.",
        "example_goals": [
            "Create an app that teaches filmmaking",
            "Build a learning-focused interface",
            "Design onboarding that sticks",
            "Make users feel competent"
        ],
        "dimensions": [
            {
                "name": "progression_model",
                "description": "How users advance their skills (Duolingo patterns)",
                "priority": 10,
                "question": {
                    "template": "How should users progress through learning?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Linear curriculum - Fixed path, unlock next after completing current", "pros": ["Clear structure", "No confusion"], "cons": ["Rigid, may bore some"]},
                        {"label": "B", "description": "Open world - Learn anything in any order", "pros": ["Freedom", "Self-directed"], "cons": ["May miss fundamentals"]},
                        {"label": "C", "description": "Project-driven - Skills unlock when projects need them", "pros": ["Contextual learning", "Immediately useful"], "cons": ["Learning tied to work"]},
                        {"label": "D", "description": "Assessment - Test what you know, skip mastered topics", "pros": ["Efficient for experienced", "Respects prior knowledge"], "cons": ["More complex to build"]}
                    ]
                }
            },
            {
                "name": "motivation_elements",
                "description": "Gamification mechanics (Points, badges, streaks)",
                "priority": 20,
                "question": {
                    "template": "What motivational elements should be included?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Full gamification - XP, levels, badges, streaks, leaderboards", "pros": ["Highly engaging", "Habit-forming"], "cons": ["Can feel childish", "Extrinsic motivation risk"]},
                        {"label": "B", "description": "Light gamification - Progress bars, achievements, no competition", "pros": ["Motivating", "Not overwhelming"], "cons": ["May not hook everyone"]},
                        {"label": "C", "description": "Professional - Certifications, skill endorsements only", "pros": ["Serious feel", "Career value"], "cons": ["Less fun"]},
                        {"label": "D", "description": "None - Pure productivity, no game elements", "pros": ["Clean, focused"], "cons": ["Less engaging"]}
                    ]
                }
            },
            {
                "name": "teaching_style",
                "description": "How instruction is delivered",
                "priority": 30,
                "question": {
                    "template": "How should learning content appear?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Inline hints - Contextual tooltips during workflow", "pros": ["Non-intrusive", "Just-in-time"], "cons": ["Easy to miss"]},
                        {"label": "B", "description": "Micro-lessons - 30-second tips, bite-sized content", "pros": ["Quick to consume", "Memorable"], "cons": ["Limited depth"]},
                        {"label": "C", "description": "Full tutorials - Step-by-step guides when requested", "pros": ["Comprehensive", "Self-paced"], "cons": ["Time consuming"]},
                        {"label": "D", "description": "Pro examples - 'Spielberg used this shot...'", "pros": ["Inspiring", "Context-rich"], "cons": ["May not apply to all"]}
                    ]
                }
            },
            {
                "name": "feedback_loops",
                "description": "How users know they're improving",
                "priority": 40,
                "question": {
                    "template": "How should users receive feedback on their progress?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Real-time stats - Dashboard showing skill growth, time spent"},
                        {"label": "B", "description": "Milestone celebrations - Congratulations at key achievements"},
                        {"label": "C", "description": "AI review - Automated feedback on their work quality"},
                        {"label": "D", "description": "Peer comparison - See how you compare to others (anonymized)"}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 4: VISUAL_DESIGN (Based on Atomic Design by Brad Frost)
    # =========================================================================
    {
        "domain_name": "visual_design",
        "description": "Visual aesthetics and design system. Based on Atomic Design methodology by Brad Frost.",
        "example_goals": [
            "Create a beautiful interface",
            "Design a cohesive visual language",
            "Build a design system for the app",
            "Make it look professional"
        ],
        "dimensions": [
            {
                "name": "visual_language",
                "description": "Overall aesthetic and mood",
                "priority": 5,
                "question": {
                    "template": "What visual language fits the brand?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Cinematic - Dark UI, film frames, spotlight effects, amber accents", "pros": ["Resonates with filmmakers", "Premium feel"], "cons": ["Limited to creative apps"]},
                        {"label": "B", "description": "Minimal - Clean, whitespace, content-first", "pros": ["Timeless", "Focus on work"], "cons": ["Can feel sterile"]},
                        {"label": "C", "description": "Playful - Rounded corners, illustrations, friendly", "pros": ["Approachable", "Fun to use"], "cons": ["May lack professionalism"]},
                        {"label": "D", "description": "Editorial - Magazine-like, typography-heavy", "pros": ["Sophisticated", "Great for content"], "cons": ["May not fit all apps"]}
                    ]
                }
            },
            {
                "name": "color_palette",
                "description": "Color scheme and mood",
                "priority": 10,
                "question": {
                    "template": "What color palette best represents the vision?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Dark mode - Deep blacks, dark grays, accent colors", "pros": ["Modern", "Reduces eye strain", "Premium"], "cons": ["Limited outdoor visibility"]},
                        {"label": "B", "description": "Light mode - Whites, soft grays, subtle accents", "pros": ["Clean", "Good readability"], "cons": ["Can feel generic"]},
                        {"label": "C", "description": "Vibrant - Bold colors, gradients", "pros": ["Memorable", "High engagement"], "cons": ["Can overwhelm"]},
                        {"label": "D", "description": "Adaptive - Light/dark based on system/user preference", "pros": ["Respects user choice"], "cons": ["Two palettes to maintain"]}
                    ]
                }
            },
            {
                "name": "typography",
                "description": "Font choices and text styling",
                "priority": 20,
                "question": {
                    "template": "What typography style fits the brand?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Modern Sans-Serif (Inter, Roboto) - Clean, neutral, tech-forward"},
                        {"label": "B", "description": "Geometric (Outfit, Poppins) - Contemporary, friendly"},
                        {"label": "C", "description": "Classic Serif (Georgia, Playfair) - Traditional, authoritative"},
                        {"label": "D", "description": "Custom/Display - Unique brand identity"}
                    ]
                }
            },
            {
                "name": "component_style",
                "description": "UI component aesthetics (Atomic Design)",
                "priority": 30,
                "question": {
                    "template": "What component style should be used?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Rounded/Soft - Soft edges, shadows, depth (iOS style)"},
                        {"label": "B", "description": "Sharp/Flat - Crisp edges, flat design (Material style)"},
                        {"label": "C", "description": "Glass/Blur - Translucent, blur effects (glassmorphism)"},
                        {"label": "D", "description": "Outline/Wire - Thin borders, minimal fills"}
                    ]
                }
            },
            {
                "name": "animation_level",
                "description": "Motion and micro-interactions",
                "priority": 40,
                "question": {
                    "template": "How animated should the interface be?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Rich - Smooth transitions, hover effects, loading animations"},
                        {"label": "B", "description": "Subtle - State changes, minimal movement"},
                        {"label": "C", "description": "Minimal - Instant response, accessibility focus"},
                        {"label": "D", "description": "User preference - Respect 'reduce motion' settings"}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 5: ACCESSIBILITY (Based on WCAG 2.1 AA)
    # =========================================================================
    {
        "domain_name": "accessibility",
        "description": "Inclusive design based on WCAG 2.1 AA standards. Covers perceivable, operable, understandable, robust (POUR).",
        "example_goals": [
            "Make the app accessible",
            "Ensure WCAG compliance",
            "Support screen readers",
            "Design for all abilities"
        ],
        "dimensions": [
            {
                "name": "compliance_level",
                "description": "WCAG conformance target",
                "priority": 10,
                "question": {
                    "template": "What accessibility compliance level is needed?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "WCAG AAA - Maximum accessibility", "pros": ["All users supported", "Future-proof"], "cons": ["Highest effort", "May limit design"]},
                        {"label": "B", "description": "WCAG AA - Standard compliance (4.5:1 contrast, keyboard nav)", "pros": ["Industry standard", "Good balance"], "cons": ["Some edge cases missed"]},
                        {"label": "C", "description": "WCAG A - Basic accessibility", "pros": ["Lower effort"], "cons": ["Many users excluded"]},
                        {"label": "D", "description": "Best effort - No strict compliance", "pros": ["Design freedom"], "cons": ["Accessibility gaps"]}
                    ]
                }
            },
            {
                "name": "keyboard_navigation",
                "description": "Full keyboard accessibility (WCAG 2.1.1)",
                "priority": 20,
                "question": {
                    "template": "How important is full keyboard navigation?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Essential - Every feature keyboard accessible"},
                        {"label": "B", "description": "Primary flows - Main actions keyboard accessible"},
                        {"label": "C", "description": "Basic - Tab navigation only"}
                    ]
                }
            },
            {
                "name": "screen_reader_support",
                "description": "Assistive technology compatibility",
                "priority": 30,
                "question": {
                    "template": "How comprehensive should screen reader support be?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Full ARIA - All states announced, live regions"},
                        {"label": "B", "description": "Semantic HTML - Proper elements, labels, alt text"},
                        {"label": "C", "description": "Basic - Key elements labeled"}
                    ]
                }
            },
            {
                "name": "skill_accessibility",
                "description": "Accommodating different skill levels (not just disability)",
                "priority": 40,
                "question": {
                    "template": "How should the app adapt to user skill levels?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Dynamic - Features reveal as user demonstrates capability", "pros": ["Personalized", "Never overwhelming"], "cons": ["May frustrate experts"]},
                        {"label": "B", "description": "User-set - 'I'm a beginner/pro' setting", "pros": ["User control"], "cons": ["User may misjudge"]},
                        {"label": "C", "description": "Same for all - Single interface, good documentation", "pros": ["Predictable", "One design"], "cons": ["Compromise for all"]}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 6: FILM_PRODUCTION (Based on StudioBinder, Movie Magic, Industry Standards)
    # =========================================================================
    {
        "domain_name": "film_production",
        "description": "Film production workflow modules based on industry standards (StudioBinder, Movie Magic, Gorilla).",
        "example_goals": [
            "Build a film production app",
            "Create a pre-production tool",
            "Design a scheduling interface",
            "Make a call sheet generator"
        ],
        "dimensions": [
            {
                "name": "production_phases",
                "description": "Industry-standard phases",
                "priority": 10,
                "question": {
                    "template": "Which production phases should be supported?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Full pipeline - Development, Pre, Production, Post, Distribution", "pros": ["Complete solution"], "cons": ["Large scope"]},
                        {"label": "B", "description": "Production core - Pre, Production, Post only", "pros": ["Focused", "Most needed"], "cons": ["Missing dev/distribution"]},
                        {"label": "C", "description": "Pre-production focus - Script, Breakdown, Budget, Schedule", "pros": ["Deep specialization"], "cons": ["Limited to planning"]}
                    ]
                }
            },
            {
                "name": "script_breakdown",
                "description": "Script analysis automation",
                "priority": 20,
                "question": {
                    "template": "How should script breakdown work?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "AI auto-breakdown - AI tags elements (locations, props, cast)", "pros": ["Fast", "Consistent"], "cons": ["May miss nuance"]},
                        {"label": "B", "description": "Manual tagging - User highlights and categorizes", "pros": ["Full control", "Learning opportunity"], "cons": ["Time consuming"]},
                        {"label": "C", "description": "AI + human review - AI suggests, user confirms", "pros": ["Best of both"], "cons": ["Still requires review"]}
                    ]
                }
            },
            {
                "name": "scheduling_style",
                "description": "Schedule visualization (Stripboard standard)",
                "priority": 30,
                "question": {
                    "template": "How should the shooting schedule be visualized?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Classic stripboard - Industry standard colored strips"},
                        {"label": "B", "description": "Calendar view - Day/week calendar with scenes"},
                        {"label": "C", "description": "Timeline - Gantt-style horizontal timeline"},
                        {"label": "D", "description": "List view - Simple list of shooting days"}
                    ]
                }
            },
            {
                "name": "call_sheet_delivery",
                "description": "How call sheets reach crew",
                "priority": 40,
                "question": {
                    "template": "How should call sheets be delivered to crew?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "SMS + Email - Text message with link, full email attachment", "pros": ["Reaches everyone", "SMS for urgency"], "cons": ["SMS costs"]},
                        {"label": "B", "description": "App notification - Push + in-app view", "pros": ["Rich features", "Tracking"], "cons": ["Requires app install"]},
                        {"label": "C", "description": "Email only - PDF attachment", "pros": ["Simple", "Universal"], "cons": ["May miss inbox"]}
                    ]
                }
            },
            {
                "name": "budget_structure",
                "description": "Film budget organization standard",
                "priority": 50,
                "is_required": False,
                "question": {
                    "template": "How should budgets be organized?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Industry standard - Above-the-line, Below-the-line, Production, Post", "pros": ["Pro-compatible", "Export ready"], "cons": ["Complex for beginners"]},
                        {"label": "B", "description": "Simplified categories - People, Places, Things, Post", "pros": ["Easy to understand"], "cons": ["May not export correctly"]},
                        {"label": "C", "description": "Hidden complexity - Simple input, AI structures properly", "pros": ["Beginner friendly", "Pro export"], "cons": ["Magic may confuse"]}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 7: COLLABORATION (Team workflows)
    # =========================================================================
    {
        "domain_name": "collaboration",
        "description": "Multi-user workflows, permissions, and team communication.",
        "example_goals": [
            "Enable team collaboration",
            "Build multi-user editing",
            "Create permission system",
            "Design team workflows"
        ],
        "dimensions": [
            {
                "name": "collaboration_mode",
                "description": "How team members work together",
                "priority": 10,
                "question": {
                    "template": "How should team members collaborate?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Real-time sync - Figma-style live cursors, instant updates", "pros": ["Immediate feedback", "Feel connected"], "cons": ["Complex tech", "Conflicts possible"]},
                        {"label": "B", "description": "Async comments - Mark up and respond later", "pros": ["Works across time zones", "Thoughtful responses"], "cons": ["Slower feedback"]},
                        {"label": "C", "description": "Turn-based - One person edits, others view", "pros": ["No conflicts", "Clear ownership"], "cons": ["Slower workflows"]}
                    ]
                }
            },
            {
                "name": "permissions_model",
                "description": "Access control granularity",
                "priority": 20,
                "question": {
                    "template": "How granular should permissions be?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Role-based - Film roles (Director, Producer, DP, Editor)", "pros": ["Industry aligned", "Easy to understand"], "cons": ["May not fit all teams"]},
                        {"label": "B", "description": "Simple tiers - Owner, Editor, Viewer", "pros": ["Easy to manage"], "cons": ["May be too broad"]},
                        {"label": "C", "description": "Module-based - Access to specific features (Budget, Schedule)", "pros": ["Granular", "Secure"], "cons": ["Complex to configure"]},
                        {"label": "D", "description": "Custom - Create any permission set", "pros": ["Full flexibility"], "cons": ["Complex, error-prone"]}
                    ]
                }
            },
            {
                "name": "team_communication",
                "description": "How team discussions happen",
                "priority": 30,
                "question": {
                    "template": "How should team communicate within the app?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "In-app chat - Slack-like threaded conversations"},
                        {"label": "B", "description": "Item comments - Discussion tied to specific artifacts"},
                        {"label": "C", "description": "Notifications only - No chat, use external tools"},
                        {"label": "D", "description": "Hybrid - Comments + lightweight in-app messaging"}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 8: MOBILE_UX (On-set and offline usage)
    # =========================================================================
    {
        "domain_name": "mobile_ux",
        "description": "Mobile and on-set experience design. Offline capability, touch optimization.",
        "example_goals": [
            "Make it work on mobile",
            "Design for on-set use",
            "Enable offline editing",
            "Optimize for tablets"
        ],
        "dimensions": [
            {
                "name": "mobile_strategy",
                "description": "Mobile platform approach",
                "priority": 10,
                "question": {
                    "template": "What mobile strategy should be used?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "PWA - Installable web app, offline capable", "pros": ["Shared codebase", "No app store"], "cons": ["Limited native features"]},
                        {"label": "B", "description": "Native apps - Separate iOS/Android apps", "pros": ["Best performance", "Native features"], "cons": ["3 codebases"]},
                        {"label": "C", "description": "Responsive web - Same app adapts to screen", "pros": ["Simplest", "One codebase"], "cons": ["Not installable, no offline"]},
                        {"label": "D", "description": "Hybrid (React Native/Flutter) - Cross-platform native", "pros": ["Native feel, shared code"], "cons": ["Learning curve"]}
                    ]
                }
            },
            {
                "name": "offline_capability",
                "description": "Working without internet",
                "priority": 20,
                "question": {
                    "template": "How should offline mode work?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Queue changes - Edit offline, sync when back", "pros": ["Full functionality", "On-set friendly"], "cons": ["Conflict resolution needed"]},
                        {"label": "B", "description": "Read-only - View downloaded content only", "pros": ["Simple, no conflicts"], "cons": ["Can't edit on set"]},
                        {"label": "C", "description": "No offline - Always requires connection", "pros": ["Simplest"], "cons": ["Unusable in field"]}
                    ]
                }
            },
            {
                "name": "touch_optimization",
                "description": "Touch interface design (Fitts's Law: 44-48dp targets)",
                "priority": 30,
                "question": {
                    "template": "How should touch interaction be optimized?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Touch-first - Large targets (48dp), gesture-heavy"},
                        {"label": "B", "description": "Balanced - Works with mouse and touch"},
                        {"label": "C", "description": "Desktop-first - Precise controls, hover states"}
                    ]
                }
            },
            {
                "name": "input_modes",
                "description": "On-set input methods",
                "priority": 40,
                "question": {
                    "template": "What input methods should be supported?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Full multi-modal - Touch + voice + camera (continuity photos)"},
                        {"label": "B", "description": "Touch + voice - Dictate notes, tap to confirm"},
                        {"label": "C", "description": "Touch only - Standard mobile input"}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 9: TECHSTACK (Framework and platform choices)
    # =========================================================================
    {
        "domain_name": "techstack",
        "description": "Technology choices for frontend, backend, and infrastructure.",
        "example_goals": [
            "Choose the right stack",
            "Plan the architecture",
            "Decide on frameworks",
            "Select the database"
        ],
        "dimensions": [
            {
                "name": "frontend_framework",
                "description": "UI framework choice",
                "priority": 10,
                "question": {
                    "template": "What frontend framework should be used?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "React + Next.js - Large ecosystem, SSR, easy hiring", "pros": ["Most popular", "Great tooling"], "cons": ["Can be complex"]},
                        {"label": "B", "description": "Vue + Nuxt - Simpler learning curve, good for MVPs", "pros": ["Gentle learning curve", "Fast to build"], "cons": ["Smaller ecosystem"]},
                        {"label": "C", "description": "Svelte + SvelteKit - Less boilerplate, great performance", "pros": ["Fast, lightweight"], "cons": ["Smaller community"]},
                        {"label": "D", "description": "Plain HTML/CSS/JS - No framework", "pros": ["Simplest", "No dependencies"], "cons": ["Manual everything"]}
                    ]
                }
            },
            {
                "name": "design_system",
                "description": "Component library choice",
                "priority": 20,
                "question": {
                    "template": "What component library/system should be used?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Shadcn/ui - Radix + Tailwind, copy-paste components", "pros": ["Full control", "Accessible"], "cons": ["Manual updates"]},
                        {"label": "B", "description": "Tailwind + Headless - Utility classes, unstyled components", "pros": ["Flexible", "Performance"], "cons": ["More design work"]},
                        {"label": "C", "description": "Material/Chakra - Opinionated, fast to build", "pros": ["Quick start"], "cons": ["Looks like every other app"]},
                        {"label": "D", "description": "Custom from scratch - Full control", "pros": ["Unique"], "cons": ["Highest effort"]}
                    ]
                }
            },
            {
                "name": "backend_approach",
                "description": "API and backend choice",
                "priority": 30,
                "question": {
                    "template": "What backend approach should be used?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Python FastAPI - Great for AI features", "pros": ["ML-friendly", "Fast", "Modern"], "cons": ["Different from frontend"]},
                        {"label": "B", "description": "Node.js (Express/Fastify) - JavaScript full-stack", "pros": ["Same language", "Large ecosystem"], "cons": ["Less ML-friendly"]},
                        {"label": "C", "description": "Supabase - BaaS for rapid development", "pros": ["Very fast to start", "Realtime built-in"], "cons": ["Vendor lock-in"]},
                        {"label": "D", "description": "Go/Rust - Performance-critical", "pros": ["Blazing fast"], "cons": ["Smaller ecosystem, harder"]}
                    ]
                }
            },
            {
                "name": "database",
                "description": "Data persistence choice",
                "priority": 40,
                "question": {
                    "template": "What database should be used?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "PostgreSQL - Relational, robust, pg_vector for AI"},
                        {"label": "B", "description": "MongoDB - Document store, flexible schema"},
                        {"label": "C", "description": "SQLite - Simple, file-based, embedded"},
                        {"label": "D", "description": "Supabase (managed Postgres) - Hosted with extras"}
                    ]
                }
            }
        ]
    },
    
    # =========================================================================
    # DOMAIN 10: COMPLEXITY_HIDING (Smart defaults, AI assistants)
    # =========================================================================
    {
        "domain_name": "complexity_hiding",
        "description": "Hiding complexity from beginners while supporting pro features. AI-assisted interfaces.",
        "example_goals": [
            "Make complex features simple",
            "Hide the complexity",
            "Use AI to help users",
            "Simplify the workflow"
        ],
        "dimensions": [
            {
                "name": "smart_defaults",
                "description": "Pre-filling sensible values",
                "priority": 10,
                "question": {
                    "template": "How should defaults be determined?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Industry standards - Pre-fill with common values for the domain", "pros": ["Teaches best practices"], "cons": ["May not fit all cases"]},
                        {"label": "B", "description": "User history - Learn from user's past choices", "pros": ["Personalized"], "cons": ["Needs data first"]},
                        {"label": "C", "description": "AI inference - Analyze context, suggest values", "pros": ["Smart", "Contextual"], "cons": ["May be wrong"]},
                        {"label": "D", "description": "Explicit - User always sets values", "pros": ["Full control"], "cons": ["More work"]}
                    ]
                }
            },
            {
                "name": "ai_structuring",
                "description": "AI converting natural input to structured data",
                "priority": 20,
                "question": {
                    "template": "How should AI help structure data?",
                    "type": "complex",
                    "choices": [
                        {"label": "A", "description": "Full magic - 'Bought lens $500' → proper expense category, tax, etc.", "pros": ["Zero friction", "Beginner friendly"], "cons": ["May misinterpret"]},
                        {"label": "B", "description": "AI suggestion - AI suggests structure, user confirms", "pros": ["Transparent", "User validates"], "cons": ["Extra step"]},
                        {"label": "C", "description": "Templates - User picks template, fills fields", "pros": ["Predictable"], "cons": ["More rigid"]},
                        {"label": "D", "description": "Manual - User enters all structured data", "pros": ["Full control"], "cons": ["Friction"]}
                    ]
                }
            },
            {
                "name": "expert_escape",
                "description": "Letting pros bypass simplification",
                "priority": 30,
                "question": {
                    "template": "How should experts access full complexity?",
                    "type": "simple",
                    "choices": [
                        {"label": "A", "description": "Always available - 'Advanced' toggle everywhere"},
                        {"label": "B", "description": "Settings-based - 'Pro mode' in user settings"},
                        {"label": "C", "description": "Contextual - System detects when user needs more"},
                        {"label": "D", "description": "Not available - Everyone uses simple interface"}
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
    existing_count = cur.fetchone()[0]
    if existing_count > 0:
        response = input(f"{existing_count} domains exist. Clear and reseed? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborting.")
            conn.close()
            return
        
        cur.execute("TRUNCATE interview_domains CASCADE")
        conn.commit()
        print("Cleared existing domains.")
    
    # Generate embeddings for domains
    domain_texts = [f"{d['domain_name']}: {d['description']}" for d in DOMAINS]
    domain_embeddings = get_embeddings(domain_texts)
    
    total_dimensions = 0
    total_questions = 0
    
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
            total_dimensions += 1
            
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
                total_questions += 1
            
            print(f"      ✓ Dimension: {dim['name']}")
    
    conn.commit()
    print(f"\n✅ Seeded {len(DOMAINS)} domains, {total_dimensions} dimensions, {total_questions} questions.")
    conn.close()


if __name__ == "__main__":
    seed_domains()
