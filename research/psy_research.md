Key Psychological & Cognitive Models
Self-Determination Theory (SDT) – Emphasizes autonomy, competence, relatedness as core needs driving intrinsic motivation
. Questions can probe whether an experience feels self-directed, skill-enhancing, or socially connected to reveal underlying motivation.
Prospect Theory (Kahneman & Tversky) – Highlights loss aversion and framing effects: people weigh losses heavier than gains and reverse risk preferences when scenarios are framed as losses vs gains
. Designing paired questions with gain/loss wording can expose a user’s risk tolerance or affective bias. For example, framing a design choice in terms of “safety” vs “risk” can dramatically flip user choices
.
Behavioral Economic Heuristics – Anchoring, availability, confirmation bias, etc., affect responses. For instance, presenting a reference value (“Most people choose X”) can anchor answers. Framing language (e.g. “90% fat” vs “10% lean”) influences preferences
. Question order and context can prime availability: asking about a negative scenario first tends to lower subsequent ratings of satisfaction
. Awareness of these effects lets us craft subtle primes or controls.
Personality Models (e.g. Big Five) – Broad traits like Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism (OCEAN) characterize people
. (For example, one study found only Agreeableness weakly correlated with aesthetic preferences
.) Tailoring questions (e.g. “I prefer bold vs. safe designs”) can hint at personality-driven preferences, though traits often predict only some preferences. Established scales (MBTI, HEXACO, etc.) implicitly guide many quizzes.
Values Frameworks (Means-End Chain) – The Means-End Chain theory posits a hierarchy from concrete attributes (A) to consequences (C) to personal values (V)
. UX “laddering” interviews repeatedly ask “Why is that important?” to climb this A→C→V ladder and uncover core values like security, belonging, or enjoyment
. In practice, MCQs can implement this by following a selected answer with “Why?” subquestions or by offering answer choices that map to different values.
Cognitive Load Theory – Working memory is limited
. Long, complex stems or too many choices overload users. Best practice is concise wording and minimal necessary context
. For example, medical-education guidelines recommend succinct stems to avoid extraneous cognitive load
. In UX interviews, this means keeping multiple-choice options few (3–4 choices)
 and parallel in structure, and avoiding “all/none of the above” which increases load
.
Dual-Process/Cognitive Style – People vary in reliance on intuitive (System 1) vs analytic (System 2) thinking. Tricks like Cognitive Reflection Test items (e.g. “10¢ ball problem”) can differentiate snap intuitions from deliberative reasoning
. Including a few counterintuitive MCQs can reveal overconfidence or the tendency to default to an obvious (but wrong) answer
.
Social Biases – Acquiescence and social desirability biases distort answers
. People often agree with statements or pick choices that cast them in a good light. To counteract this, questions should be framed neutrally, and options should include socially undesirable choices to check consistency. For example, parallel “true/false” phrasing or reverse-keyed items can reveal acquiescence bias.
Motivational & Needs Theories – Beyond SDT and Maslow, UX-specific hierarchies exist. For example, an adapted Maslow/UX hierarchy outlines levels from basic functionality up to delight
. Questions might assess whether users focus on fundamental needs (e.g. “Does it work?”) versus higher-level desires (e.g. “Does it feel meaningful?”). Aligning questions with such tiers helps uncover at what level users are operating.
<table> <thead> <tr><th>Model/Theory</th><th>Core Concept</th><th>Relevance to Questions</th></tr> </thead> <tbody> <tr><td><b>Self-Determination (SDT)</b>:contentReference[oaicite:20]{index=20}</td><td>Autonomy, competence, relatedness drive motivation.</td><td>Probe if users feel in control, capable, connected (e.g. “Which feature makes you feel most in charge?”).</td></tr> <tr><td><b>Prospect Theory</b>:contentReference[oaicite:21]{index=21}</td><td>Loss aversion; risk preferences flip by framing.</td><td>Frame choices as gains vs losses or safe vs risky to expose aversion/seeking biases.</td></tr> <tr><td><b>Decision Heuristics</b></td><td>Anchoring, framing, priming, etc.</td><td>Be aware: wording and context bias answers (e.g. “riskier” vs “safer” framing reverses responses:contentReference[oaicite:22]{index=22}). Use controlled framing to test biases.</td></tr> <tr><td><b>Big Five Personality</b>:contentReference[oaicite:23]{index=23}</td><td>OCEAN traits in people.</td><td>Use trait-relevant questions (e.g. preference for novelty vs routine) to gauge personality-driven UX preferences (though correlations are often modest:contentReference[oaicite:24]{index=24}).</td></tr> <tr><td><b>Means-End / Laddering</b>:contentReference[oaicite:25]{index=25}</td><td>Hierarchy: Attributes → Consequences → Core Values.</td><td>Follow “Why?” chains: ask about features, then why they matter, to reveal deeper values like security or belonging:contentReference[oaicite:26]{index=26}:contentReference[oaicite:27]{index=27}.</td></tr> <tr><td><b>Cognitive Load Theory</b>:contentReference[oaicite:28]{index=28}</td><td>Working memory is limited; extraneous info hinders answers.</td><td>Keep MCQs concise and clear. Use few, non-overlapping options. Avoid “all/none” choices:contentReference[oaicite:29]{index=29}. Structure questions to minimize re-reading effort:contentReference[oaicite:30]{index=30}.</td></tr> <tr><td><b>System 1 vs 2 (Dual Process)</b></td><td>Fast intuitive vs slow analytic thought.</td><td>Include some trick questions or time constraints to see if users default to gut reactions. Overconfidence on intuitive errors reveals reliance on System 1:contentReference[oaicite:31]{index=31}.</td></tr> <tr><td><b>Social & Response Biases</b>:contentReference[oaicite:32]{index=32}</td><td>Acquiescence, social desirability, memory limits distort answers.</td><td>Pose both positively- and negatively-phrased items to catch acquiescence. Use concrete behavioral options (e.g. “Do X or Y” rather than “Always/Sometimes”) to reduce false agreement:contentReference[oaicite:33]{index=33}:contentReference[oaicite:34]{index=34}.</td></tr> <tr><td><b>Maslow/UX Needs Hierarchy</b>:contentReference[oaicite:35]{index=35}</td><td>From basic functionality up to meaningful delight.</td><td>Design questions to address different need levels. For example, ask “Which is most important: Reliability, usability, or excitement?” to see if users prioritize basics or higher-level values:contentReference[oaicite:36]{index=36}.</td></tr> </tbody> </table>
Techniques for Designing Psychologically Diagnostic Questions
Attribute Laddering (Means-End interviews) – Start with a concrete preference and repeatedly ask “Why is that important?”. Each “why” reveals a deeper consequence or value
. In MC form, this can be done by chaining questions: e.g. first ask “Which feature matters most (A/B/C)?”, then “Why is that feature important?”. This uncovers hidden values behind surface choices.
Situational/Scenario Questions (SJTs) – Present a realistic scenario with multiple response options
. For UX, craft a scenario (“You have 5 minutes to complete task X on the interface; what do you do?”) and give MC actions reflecting different strategies or emotions. SJTs have been used to infer traits (e.g. conscientiousness) and can reveal implicit decision criteria. The chosen option indicates priorities and problem-solving style.
Trade-off/Forced-Choice Questions – Present two (or more) desirable outcomes or attributes and force a selection. For example: “Would you rather have a faster app that uses more battery, or a slower app that saves battery?” Such trade-offs reveal priorities and decision weights. You can also present partial points or best–worst scaling: e.g. “Rank or pick the most/least important features among [A, B, C].” (Best–Worst MCQs let users indicate extremes for deeper insight.)
Contrasting Frames – Ask closely related questions with different framing to detect inconsistencies. For example, one version might ask “Would you prefer feature A or B?” and another “Would you tolerate feature B or A?”. Changes in choice can signal uncertainty or ambivalence. Likewise, posing one question in positive terms and a near-identical question in negative terms checks for acquiescence or framing effects.
Cognitive Load Checks – To estimate mental effort, include a hidden “nearly identical” question or allow an “I don’t know”/“skip” option. Users under high load may choose the default or ‘skip’ more often. Also, measure response time: unusually long or short answers can indicate confusion or guesswork. Keep wording simple and direct (avoiding complex hypotheticals) to measure true preference, as recommended by cognitive load theory
.
Emotional and Values Elicitation – Use affect-laden wording or ask about feelings. For example: “Which of these statements best describes how you feel when [doing X]?” with emotion-focused options. Kano-style questions fall here: ask users to rate their satisfaction if a feature is present versus absent
, classifying features by emotional impact (basic need vs. delight). Similarly, ask users to agree/disagree with statements reflecting core values (e.g. “I value community over efficiency”) to surface latent values.
Best Practices & Frameworks for MCQ Design
Clarity and Brevity – Ensure stems are succinct and unambiguous
. Extraneous context or jargon increases cognitive load. For UX interviews, write questions as if conversing naturally, but keep them no longer than needed. Avoid double negatives or complex grammar. Use concrete terms (e.g. “twice a week” instead of “often”
).
Parallel, Plausible Options – All answer choices should be balanced and plausible
. Distractors should reflect common alternative viewpoints or misconceptions. Avoid “joke” or obviously wrong options; those are eliminated too easily
. Keep option length and structure parallel to prevent highlighting any choice. For example, don’t mix long and short answers or differ in detail level.
Avoid Cognitive Traps – Steer clear of “all of the above” or “none of the above”, as they often skew responses and add load
. Also avoid absolutes like “always”/“never” unless the intent is to test extremity, since such extreme options are rarely true
. Limit to 3–4 options if possible; more choices increase decision time with diminishing psychometric benefit
.
Subtle Framing – Use question wording strategically. For instance, neutral framing reduces bias, while intentional positive/negative framing can test consistency. As one example, polls found that asking “Who would you trust as a safer leader?” versus “Who would be riskier?” could reverse the majority opinion
. When diagnosing user intent, such framing differences can reveal leanings. Always pilot-test wording variations.
Validity & Alignment – Align each question with a clear purpose or construct (the “objective” of that question)
. Just as academic MCQs map to Bloom’s taxonomy levels
, UX questions should map to specific constructs (e.g. “values”, “usability priorities”). Avoid testing multiple concepts in one stem. Whenever possible, anchor options to observed behaviors or concrete scenarios rather than abstract attitudes to improve content validity
.
Iterative Testing and Feedback – Use cognitive interviewing (think-aloud protocols) to check if respondents interpret questions as intended
. Collect pilot data to compute item statistics (see next section). Use expert review or a “question blueprint” to ensure coverage of all target factors
. After deployment, continually refine questions based on response patterns and interviewer insights.
Reusable Question Patterns & Templates
“Why?” Ladder Pattern – Template: Q1: “Which feature is most important to you (A/B/C)?” Q2: “Why is that important to you?”
. Repeat as needed. This explicit laddering moves from choice to rationale, peeling back layers of user motivation.
Situational MCQ (SJT) – Template: Present a brief scenario, then offer 3–4 actions or reactions. Example: “You’re late to a meeting because a feature didn’t work. Do you: (A) Blame the user, (B) Blame the developer, (C) Blame yourself, (D) Try again.” The choice reflects decision style or emotional response
. Use such situational items to probe problem-solving or interpersonal values.
Attribute Trade-off Question – Template: “If you could have either X or Y, which would you choose?” (Often followed by “Why?”). For example: “Would you rather an interface that is 80% complete with no bugs, or 100% complete with some minor bugs?” The answer reveals what the user prioritizes (completeness vs. perfection). Frame options as realistic design trade-offs.
Best–Worst / MaxDiff – Template: Present a set of items and ask the user to mark the most and least important (or likeable) choices. For example: “From this list of potential features, choose the one you value most and the one you value least.” This forced ranking format yields richer preference data than single-choice and mitigates scale bias. It can be implemented sequentially as multiple MCQs.
Contradictory Pair – Template: Ask essentially the same question twice with opposite framing. E.g., Q1: “I find new features exciting.” (Agree/Disagree options) and Q2: “I prefer familiar features over new ones.” The pattern of agreement/disagreement can spot contradictory self-reports. In MC form, one question might be “Which statement best matches you: ‘I like new features’ vs ‘I like familiar features’,” and later reversed. Inconsistency indicates ambivalence or inattention.
Kano-Style Emotion Question – Template: For a given feature, ask two MCQs: one for satisfaction if it’s present and one if absent. E.g., “How would you feel if the app had [Feature X]?” (A: Delighted; B: Neutral; C: Dissatisfied), and “How would you feel if it did NOT have [Feature X]?” (A: Satisfied; B: Neutral; C: Frustrated). Responses classify the feature (must-be, performance, excitement) by emotional impact
. This helps uncover latent emotional drivers behind needs.
Evaluation Metrics for Question Effectiveness
Information Gain / Entropy – Compute how much a question’s answers reduce uncertainty about the user. A question is more effective if it produces a broad, balanced answer distribution (high Shannon entropy) and clearly splits users into different preference groups. In practice, measure the reduction in entropy of the target classification (if using a decision-tree approach) when using the question.
Response Diversity (Entropy) – Check that answer choices are actually used. If one option attracts ~100% of responses, the question yields little insight. Aim for more even distributions (not necessarily uniform, but meaningful variance). Low diversity suggests rewriting or dropping the question.
Difficulty & Discrimination (Psychometrics) – Adapted from test design, calculate:
Difficulty index: e.g. percent choosing each answer (for knowledge tests, percent correct). For UX questions, analogous to how many pick the “expected” option. Avoid items everyone “gets right” or unanimously chooses, as they provide no discrimination.
Discrimination index: correlation between an item response and an independent measure of insight (e.g. total test score or an external criterion). A high discrimination (point-biserial correlation) means the question differentiates “high-detail” vs “low-detail” respondents
. Use this to identify which questions best distinguish users.
Reliability – If questions target the same construct (e.g. underlying value), check internal consistency (Cronbach’s alpha) or test-retest stability. Duplication or “bait” items can reveal random responding. High reliability indicates stable attitudes; low suggests the question may be ambiguous.
Predictive Validity – Assess whether answers forecast actual behavior or outcomes. For instance, do choices on questions correlate with later survey responses, usability scores, or real task performance? In a study, an SJT on “dependability” correlated moderately to highly with standardized Conscientiousness measures
, demonstrating external validity. Compute correlations (Pearson’s r or AUC if binary outcomes) between question patterns and target variables. Strong alignment shows that questions meaningfully capture the intended trait.
Response Process Checks – Use cognitive interviews or think-aloud during piloting to ensure participants interpret questions correctly
. Monitor how often users choose “Other (please specify)” or write in comments. Frequent write-ins may mean answer options are incomplete or misunderstood.
Consequential Analysis – Consider the impact of each question (Messick’s consequential validity
). Does asking it influence subsequent answers (priming) or choices (e.g. anchoring the conversation)? Pilot different orders to detect such effects. Evaluate whether the question changes users’ self-perception (e.g. over-focusing on a minor issue), which may signal wording problems.
Examples from Research & Practice
Laddering in UX (Means-End Interviews) – UX researchers commonly use the laddering technique to find values. For example, asking a user “Why is the app’s visual style important to you?” repeatedly can reveal values like social status or identity
. This qualitative method has been applied to understand why users choose features by mapping attributes to feelings to values.
Kano Model Surveys – Product teams often use Kano questionnaires (as outlined by Qualtrics) to categorize features as must-haves or delights based on emotional response
. This involves paired functional/dysfunctional MCQs that directly tie to user satisfaction and surprise, illustrating how structured questions can elicit emotional drivers.
Situational Judgment Tests in Assessment – In organizational psychology, SJTs with MCQs are widely used to infer traits and competencies
. The “Dependability SJT” example showed that carefully-crafted scenarios and options correlated with conscientiousness
. Similarly, UX interview questions can mimic this format to gauge attributes like patience, attention to detail, or social values.
Consumer and Marketing Research – Traditional consumer studies use means-end theory and value inventory techniques (e.g. Schwartz values, Rokeach scales). While those are often questionnaires, the underlying idea is the same: ask specific questions (often MC or ranking) that indirectly measure core drivers. For instance, a personal values card sort (choosing most vs least important values) is analogous to a MC priority question. The UX orchestrator can incorporate such classic patterns.
Online Personality/Values Quizzes – Many informal systems (e.g. personality quizzes, career fit tests) use psychologically-informed MCQs without explicit citations. These often use scenario-based or forced-choice formats drawn from known models. The orchestrator can take inspiration from such systems. For example, job-fit assessments present MCQs about workplace preferences that subtly map onto constructs like risk-taking or empathy.
Cognitive Interviewing & Think-Aloud Studies – In usability testing, asking users to think aloud while answering MCQs provides ground truth on how questions function. Researchers have shown that revision of question wording based on think-aloud leads to higher validity
. This iterative testing is a real-world practice: pilot the orchestrator’s questions with test interviews to refine psychological diagnostics.
In summary, combining behavioral science models (motivations, biases, needs) with best practices in MCQ writing (clarity, validation) yields a toolkit for interviewing. Use ladders of “why” to climb into user values
, craft scenario-based choices to reveal decision styles
, and always evaluate each question’s information content and validity
. By embedding these patterns and metrics into the orchestrator, the LLM can be nudged to ask rich, diagnostic questions that uncover the latent drives behind user responses. Sources: Authoritative literature on question design and behavioral models
 and UX/psychology research articles.
Citations

Theory – selfdeterminationtheory.org

https://selfdeterminationtheory.org/theory/

Prospect Theory - The Decision Lab

https://thedecisionlab.com/reference-guide/economics/prospect-theory
Cognitive Bias | Questionnaire Design | Feature | Research Live

https://www.research-live.com/article/features/its-the-way-you-ask-it/id/5025182
Cognitive Bias | Questionnaire Design | Feature | Research Live

https://www.research-live.com/article/features/its-the-way-you-ask-it/id/5025182
Cognitive Bias | Questionnaire Design | Feature | Research Live

https://www.research-live.com/article/features/its-the-way-you-ask-it/id/5025182
Cognitive Bias | Questionnaire Design | Feature | Research Live

https://www.research-live.com/article/features/its-the-way-you-ask-it/id/5025182

Big 5 Personality Traits: The 5-Factor Model of Personality

https://www.simplypsychology.org/big-five-personality.html
Predicting Aesthetic Preferences: Does the Big-Five Matters?

https://thesai.org/Downloads/Volume12No12/Paper_23-Predicting_Aesthetic_Preferences.pdf

Laddering: A Research Interview Technique for Uncovering Core Values :: UXmatters

https://www.uxmatters.com/mt/archives/2009/07/laddering-a-research-interview-technique-for-uncovering-core-values.php

Laddering: A Research Interview Technique for Uncovering Core Values :: UXmatters

https://www.uxmatters.com/mt/archives/2009/07/laddering-a-research-interview-technique-for-uncovering-core-values.php

Laddering: A Research Interview Technique for Uncovering Core Values :: UXmatters

https://www.uxmatters.com/mt/archives/2009/07/laddering-a-research-interview-technique-for-uncovering-core-values.php
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Overconfidence in the Cognitive Reflection Test: Comparing Confidence Resolution for Reasoning vs. General Knowledge - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC10219213/

Everyone wants a faster horse. We aim to please. | by Jake Sharratt | UX Collective

https://uxdesign.cc/everyone-wants-a-faster-horse-we-aim-to-please-a38ee8bc1c98?gi=73bc27c541d8

What is the Hierarchy of Needs? | IxDF

https://www.interaction-design.org/literature/topics/hierarchy-of-needs?srsltid=AfmBOopkvjc_kyKHndeNmxOhHPKQrq_CY-TiVDtNOxsnfe3eLSO0JYRR
Cognitive Bias | Questionnaire Design | Feature | Research Live

https://www.research-live.com/article/features/its-the-way-you-ask-it/id/5025182
Situational Judgment Tests as a method for measuring personality: Development and validity evidence for a test of Dependability - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC6392235/

Kano Analysis: the Kano Model Explained - Qualtrics

https://www.qualtrics.com/articles/strategy-research/kano-analysis/
Cognitive Bias | Questionnaire Design | Feature | Research Live

https://www.research-live.com/article/features/its-the-way-you-ask-it/id/5025182
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Situational Judgment Tests as a method for measuring personality: Development and validity evidence for a test of Dependability - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC6392235/
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
Cognitive Bias | Questionnaire Design | Feature | Research Live

https://www.research-live.com/article/features/its-the-way-you-ask-it/id/5025182
Educator's blueprint: A how‐to guide for developing high‐quality multiple‐choice questions - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC9873868/
All Sources
