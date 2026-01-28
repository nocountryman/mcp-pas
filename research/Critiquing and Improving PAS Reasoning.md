# **Architectural Critique and Evolution of Probabilistic Abductive Scaffolding (PAS): Towards Neuro-Symbolic State-Space Search**

## **Executive Summary**

The rapid evolution of Large Language Models (LLMs) has precipitated a shift from monolithic prompt engineering to structured, agentic architectures designed to scaffold stochastic generation with logical rigor. Among these, the 'PAS' (Probabilistic Abductive Scaffolding) specification has emerged as a theoretical baseline for integrating abductive inference—the derivation of the best explanation from observations—into the generative process. PAS attempts to formalize reasoning through a triad of mechanisms: Bayesian scoring for hypothesis evaluation, tree-based expansion for exploring solution spaces, and self-learning loops for iterative refinement. However, a comprehensive audit of contemporary research and empirical performance metrics reveals that the canonical PAS specification relies on mathematically fragile proxies that fail to scale to complex reasoning tasks. Specifically, its reliance on semantic similarity as a likelihood function, unguided tree expansion strategies, and susceptible self-correction loops renders it vulnerable to epistemological degeneration and computational inefficiency.

This report provides an exhaustive technical critique of the PAS framework, dissecting its core components against the backdrop of state-of-the-art (SOTA) methodologies. We argue that the "probabilistic" scoring in PAS often conflates semantic proximity with logical entailment, leading to confident hallucinations, while its "abductive" tree search lacks the lookahead capabilities required for multi-step planning. Furthermore, the "self-learning" mechanisms are prone to model collapse when devoid of external, verifiable ground truth. To remediate these structural deficiencies, we propose a comprehensive architectural evolution that integrates **Monte Carlo Tree Search (MCTS)** for value-guided exploration, **Reflexion** for verbal reinforcement, and **Reinforcement Learning with Verifiable Rewards (RLVR)** for grounding optimization. This new paradigm—termed the Neuro-Symbolic Abductive Reasoner (NSAR)—transitions the system from a passive, heuristic scaffold to an active, optimizing search engine capable of rigorous "System 2" reasoning.

## ---

**1\. The PAS Specification: A Critical Autopsy of Mechanisms**

The Probabilistic Abductive Scaffolding (PAS) framework represents a significant attempt to impose structure on the chaotic latent space of LLMs. Ideally, it mimics human abductive reasoning: observing a phenomenon, generating multiple explanatory hypotheses, evaluating their probability based on priors and likelihoods, and iteratively refining the conclusion. However, the translation of these cognitive processes into algorithmic steps using current Transformer architectures reveals critical flaws in the standard implementation.

### **1.1 The Bayesian Scoring Fallacy: Priors, Likelihoods, and Vector Proxies**

At the heart of PAS lies the Bayesian update rule, intended to calculate the posterior probability of a hypothesis (![][image1]) given evidence (![][image2]): ![][image3]. While theoretically sound, the practical implementation of these terms in LLM-based systems typically relies on proxies that degrade the fidelity of the inference.1

#### **1.1.1 The Miscalibration of Prior Probabilities**

In PAS, the prior probability ![][image4]—the initial plausibility of a hypothesis before considering specific evidence—is most commonly derived from the LLM’s internal confidence scores (sequence log-likelihoods). The assumption is that the model's perplexity serves as a valid proxy for epistemic uncertainty.

**Critique of Calibration**: This assumption is empirically brittle. Research indicates that while modern LLMs (like GPT-4 or Llama 3\) are powerful generators, they are often poorly calibrated, particularly after Reinforcement Learning from Human Feedback (RLHF).3 The RLHF process, designed to align models with human preference, introduces a "sycophancy" bias where the model prioritizes plausible-sounding or agreeable answers over factual uncertainty.5 Consequently, a PAS system using raw logits as priors will systematically overweight "popular" misconceptions and underweight novel but correct reasoning paths.

* **The Overconfidence Trap**: Studies show that fine-tuning on domain-specific data often exacerbates overconfidence, as the model "memorizes" patterns without understanding underlying causal structures.6 A PAS system relying on these priors will aggressively prune search branches that appear initially improbable (high perplexity) but are logically necessary, effectively locking the reasoning process into local optima defined by training data frequency rather than logical validity.  
* **Inconsistency in Distribution**: The raw numeric outputs of an LLM over a set of options often fail to sum to one or form a coherent probability distribution.7 Normalizing these scores via Softmax can mask the underlying lack of confidence, presenting a flat distribution (high entropy) as a decisive choice if the temperature parameters are mishandled.

#### **1.1.2 The Invalidity of Embedding Distance as Likelihood**

The most critical flaw in typical PAS implementations is the estimation of the likelihood term ![][image5]—how likely the evidence is if the hypothesis were true. Lacking a causal world model, PAS architectures frequently resort to **Semantic Similarity**, calculating the cosine similarity between the vector embeddings of the hypothesis and the observation.8

**Critique of Geometric Proxies**: This approach commits a category error, confusing *semantic relatedness* with *logical entailment*. High-dimensional embedding spaces (e.g., those produced by BERT or Ada models) are optimized to cluster tokens that appear in similar contexts, not to encode truth values or probabilistic dependencies.10

* **The Negation Blindness**: A fundamental limitation of embedding-based scoring is "negation blindness." The sentences "The server is down due to a DNS error" and "The server is *not* down due to a DNS error" share almost identical lexical content and will have a cosine similarity near 0.9. However, they are logical opposites. A PAS system using this metric would assign high likelihood to both, rendering the abductive evaluation vacuous.12  
* **Dimensional Distortion**: Deep metric learning research highlights that the distance between points in embedding space does not linearly map to probability mass.13 The topology of these spaces is often non-Euclidean; a small shift in vector space can represent a massive semantic jump (e.g., changing a "not" to "now"), while large shifts might represent irrelevant syntactic variations. Using Euclidean or Cosine distance as a proxy for ![][image5] introduces "distance distortion," where the system hallucinates connections between unrelated concepts simply because they share domain jargon.14  
* **Symmetry Violation**: Probabilistic implication is antisymmetric (![][image6]). Cosine similarity is symmetric (![][image7]). By using a symmetric metric to approximate an asymmetric probability, PAS fundamentally breaks the directionality of causality, treating the hypothesis explaining the evidence as identical to the evidence explaining the hypothesis.9

### **1.2 Tree Expansion: The Combinatorial Explosion of Uninformed Search**

The "Abductive" component of PAS requires the generation of multiple competing hypotheses. In practice, this is implemented as a tree expansion, where each node represents a state (partial thought) and edges represent potential next steps. The canonical PAS approach utilizes heuristic expansion strategies akin to **Chain of Thought (CoT)** branching or naive **Breadth-First Search (BFS)**.17

**Critique of Heuristic Expansion**:

* **Myopic Evaluation**: Standard PAS expansion evaluates nodes based on immediate coherence (local probability). It lacks a "lookahead" mechanism to determine if a currently coherent thought leads to a dead end three steps later. This myopia is fatal in multi-step reasoning tasks like math or code generation, where a critical insight often requires traversing a "valley" of low-probability intermediate steps.17  
* **Resource Inefficiency**: Uninformed BFS leads to a combinatorial explosion. Expanding ![][image8] branches at each depth ![][image9] results in ![][image10] complexity. Without a robust value function to prune branches, PAS wastes the majority of its inference compute exploring redundant or clearly erroneous paths that happen to contain high-probability tokens.20 The "Tree of Thoughts" (ToT) literature emphasizes that the efficacy of tree search is contingent not on the branching factor itself, but on the quality of the evaluator.18 If the evaluator (the Bayesian score) is flawed, the tree search becomes a "random walk" through the latent space, offering little advantage over simple sampling.

### **1.3 Self-Learning Loops: The Echo Chamber and Model Collapse**

The third pillar of PAS is the self-learning loop, where the system critiques its own outputs and refines its priors or context for future iterations. This is often framed as an autoregressive feedback loop.22

**Critique of Autoregressive Feedback**:

* **Model Collapse**: When a generative model trains on its own outputs (or uses them as "gold" examples in context), it reinforces its own biases and errors. This phenomenon, known as **Model Collapse**, leads to a progressive reduction in the variance and quality of the model's distribution.24 As the model converges on its own "average" output, it loses the ability to generate the "tails" of the distribution where creative or complex solutions often lie.26  
* **Vacuous Critiques**: Research into "Self-Critique" mechanisms reveals a tendency for LLMs to generate "vacuous critiques"—feedback that sounds authoritative but is substantively empty. A model might critique a correct code snippet for "style" while missing a critical logic bug, or conversely, accept a flawed argument because the "tone" is persuasive.27 This is a manifestation of **Goodhart’s Law**: when the measure of quality (the critique) becomes a target, the model learns to game the critique mechanism (e.g., by writing longer, more complex-sounding answers) rather than optimizing for truth.29  
* **The Sycophancy Loop**: In the absence of external ground truth, the self-learning loop optimizes for *consistency* rather than *correctness*. If the model holds a false belief (e.g., a common misconception), its self-critique will likely reinforce that belief, effectively "polishing" the hallucination rather than correcting it.5

## ---

**2\. The Search for Value: Transitioning from PAS to MCTS**

To address the "combinatorial explosion" and "myopic evaluation" flaws of PAS, the reasoning architecture must evolve from static heuristic trees to dynamic **Monte Carlo Tree Search (MCTS)**. This represents a shift from "scaffolding" (supporting the model) to "state-space search" (navigating the problem).

### **2.1 The Limitations of Tree of Thoughts (ToT)**

While Tree of Thoughts (ToT) introduced the concept of branching reasoning paths, empirical studies suggest that its "vanilla" implementation (using BFS/DFS) is computationally expensive and often outperformed by simpler methods like Chain-of-Thought with Self-Consistency if the evaluator is weak.17

* **Search Budget Misallocation**: ToT treats all active branches as equally worthy of expansion until they are pruned. It does not dynamically reallocate compute to the most promising paths during the search process.  
* **Lack of Terminal Value**: ToT typically relies on intermediate voting (checking "is this step good?"). It rarely simulates the path to completion to see if it *actually* solves the problem, missing the critical "rollout" signal used in game-playing AIs.19

### **2.2 Monte Carlo Tree Search (MCTS) in Latent Space**

MCTS addresses these limitations by introducing a learned **Value Function** and a dynamic **Selection Policy**. The application of MCTS to LLM reasoning involves mapping the four traditional steps to token generation.20

#### **2.2.1 Selection: The UCT Algorithm**

Instead of blindly expanding all nodes (BFS), MCTS uses the **Upper Confidence Bound applied to Trees (UCT)** to select the next node to explore.

The selection formula balances *Exploitation* (visiting high-value nodes) and *Exploration* (visiting less-visited nodes):

![][image11]

* ![][image12]: The mean value of the action (reasoning step), derived from previous simulations. This replaces the static "Bayesian Score" with an empirically updated value.  
* ![][image13]: The prior probability from the LLM (the policy network).  
* ![][image14]: The visit count.

**Strategic Advantage**: This formula allows the system to overcome a bad prior. Even if the LLM (![][image15]) thinks a step is unlikely (low probability), if the simulations (![][image16]) show it leads to a correct answer, the ![][image16] term will eventually dominate, guiding the search down the correct path. This capability is absent in PAS.20

#### **2.2.2 Expansion and Simulation (Rollouts)**

When a leaf node is selected, the LLM generates ![][image8] new thoughts (Expansion). Crucially, MCTS then performs a **Simulation** (or Query to a Value Network) to estimate the long-term return.34

* **Rollout Policy**: A lightweight model (or the base model with low temperature) continues the reasoning chain from the new node until a terminal state (answer) is reached.  
* **Terminal Verification**: The final answer is checked (e.g., by a math verifier, unit test, or self-consistency consensus).  
* **Value Backpropagation**: The result (1 for success, 0 for failure) is propagated back up the tree, updating the ![][image16]\-values of all ancestor nodes.

**Impact**: This transforms the evaluation from "Does this step look good?" (PAS) to "Does this step lead to a win?" (MCTS). This effectively solves the myopia problem, allowing the system to value intermediate steps that are necessary but seemingly low-probability.21

#### **2.2.3 Lightweight MCTS and "Nudging"**

Full MCTS can be computationally prohibitive. Recent research introduces **Lightweight MCTS** and **Nudging** strategies to approximate this performance with lower overhead.

* **Lightweight Rollouts**: Instead of full model generation, "lightweight" rollouts might use a distilled, smaller model (e.g., 7B parameter) to perform the simulation, while the larger model (e.g., 70B+) performs the expansion. This "Speculative Decoding" approach maintains high reasoning quality while reducing latency.37  
* **Nudging**: This technique involves using a small, aligned "Nudger" model to inject guide tokens during decoding when the base model exhibits high uncertainty (measured by entropy). Rather than expanding a full tree, the Nudger steers the single-path generation away from known failure modes (e.g., "Wait, let's check the units..."). This serves as a "soft" MCTS, providing guidance without the memory overhead of a tree.39

### **2.3 The Learned Value Function: Replacing the Bayesian Score**

The success of MCTS hinges on the quality of the Value Function ![][image17]. In the NSAR architecture, this is not a heuristic formula but a trained neural network (or head).

* **Training Objective**: The Value Function is trained to predict the *expected future reward* (probability of correctness) from the current state ![][image18].  
* **Data Source**: Training data is generated by running thousands of MCTS rollouts on training problems and labeling the states based on whether they led to a correct solution.  
* **Architecture**: A "Process Reward Model" (PRM) is trained to score individual steps. Research shows that dense rewards (step-level) significantly outperform sparse rewards (outcome-level) for complex reasoning, as they provide more granular gradients for the search policy.42

## ---

**3\. The Internal Critic: Reflexion and Constitutional AI**

While MCTS improves the *search* for solutions, it does not inherently improve the model's *internal* reasoning capability over time. The **Reflexion** framework addresses this by converting the "Self-Learning" loop of PAS into a verbal reinforcement learning process.

### **3.1 Verbal Reinforcement Learning**

Standard Reinforcement Learning (RL) updates numerical weights. **Reflexion** updates the *context window* (memory).22

* **Episodic Memory**: The system maintains a buffer of past trials, including the trajectory (thoughts/actions), the outcome (error), and the critique.  
* **The Critique-Correction Loop**: Instead of simply re-trying a failed problem (which often leads to the same error), the Reflexion agent is prompted to *analyze* the failure. "Why was the previous answer wrong?" The generated critique (e.g., "I failed to account for the friction coefficient") is then appended to the context for the next attempt.  
* **Mechanism**: This mimics "gradient descent in thought space." The critique acts as a gradient, pointing the next generation step away from the error region. Empirical results show that this iterative refinement can boost performance on benchmarks like GSM8K and HumanEval by over 20% without any weight updates.22

### **3.2 Constitutional AI: Fixing the Critic**

A major vulnerability in Reflexion (and PAS self-learning) is the quality of the critique. If the critic is weak or sycophantic, the loop fails. **Constitutional AI (CAI)** provides the solution by explicitly constraining the critic.44

* **The Constitution**: A set of natural language principles that define "good reasoning." For example:  
  * *Principle 1*: "Identify any assumptions that were not explicitly stated in the problem."  
  * *Principle 2*: "Check if the code handles edge cases (e.g., empty lists, negative numbers)."  
  * *Principle 3*: "Ensure the tone is objective and the logic is causal, not just correlational."  
* **Guided Critique**: Instead of asking "Is this right?", the system asks "Critique this response based on Principle 2." This forces the model to perform a specific, structured analysis rather than a generic review.  
* **Impact**: CAI techniques reduce "vacuous critiques" by grounding the feedback in specific, actionable rules. It transforms the critic from a "black box" into an interpretable, steerable component.46

### **3.3 Adversarial Self-Play: Hardening the Reasoner**

To further prevent the model from "gaming" the critique (Goodhart’s Law), the NSAR architecture incorporates **Adversarial Self-Play**.27

* **The Game**: Two instances of the model play a zero-sum game.  
  * **The Sneaky Generator**: Tasked with generating a reasoning chain that contains a subtle, hidden error but looks plausible.  
  * **The Critic**: Tasked with identifying the error.  
* **Evolutionary Pressure**: This adversarial dynamic creates a continuous curriculum. As the Generator gets better at hiding errors, the Critic must get better at spotting them. This co-evolution prevents the "critique" mechanism from becoming stagnant or vacuous, ensuring that the self-learning loop remains robust.27

## ---

**4\. Grounding the Loop: RLVR and Process Supervision**

The ultimate failure mode of PAS is "Hallucination Reinforcement"—learning to be confidently wrong. To break this cycle, the system must anchor its learning in **Reinforcement Learning with Verifiable Rewards (RLVR)**.

### **4.1 The Necessity of Verifiable Ground Truth**

RLVR posits that for reasoning tasks, the reward signal must be **objective** and **external** to the model.48

* **Verifiable Domains**: Math (correct answer), Code (passes unit tests), Logic (satisfies formal constraints).  
* **The RLVR Loop**:  
  1. **Generate**: Model produces a solution.  
  2. **Verify**: Execute the code or check the math answer.  
  3. **Reward**: If Correct ![][image19] Reward \= 1\. If Incorrect ![][image19] Reward \= 0\.  
  4. **Update**: Use PPO (Proximal Policy Optimization) or GRPO (Group Relative Policy Optimization) to update the model weights to maximize the expected reward.

**Critique of PAS**: PAS typically uses "Self-Consistency" or "Model Likelihood" as a proxy for correctness. RLVR replaces this proxy with ground truth. This eliminates the "Echo Chamber" effect: the model cannot convince the verifier that ![][image20], no matter how persuasive its rhetoric.43

### **4.2 Training Process Reward Models (PRM)**

While RLVR provides outcome supervision, complex reasoning benefits from **Process Supervision**. A Process Reward Model (PRM) assigns a score to *each step* of the reasoning chain.42

* **Data Collection**: Use MCTS to generate thousands of traces. Label steps based on whether they appeared in successful traces (positive) or failed traces (negative).  
* **Training**: Train a classifier (the PRM) to predict these labels.  
* **Application**: The trained PRM becomes the **Value Function** for the MCTS engine (Section 2.3). It guides the search in real-time, effectively "distilling" the insights from the RLVR training into a dense signal that can be used during inference.

### **4.3 Avoiding Goodhart’s Law and Reward Hacking**

Even with verifiable rewards, there is a risk of **Goodharting**—optimizing for the metric at the expense of the true goal.29

* **Proxy Failure**: If the reward function is imperfect (e.g., a unit test that doesn't cover all edge cases), the model will learn to write "buggy code that passes the test."  
* **KL Regularization**: To mitigate this, RLVR algorithms typically include a KL-divergence penalty, constraining the model from drifting too far from its base distribution (the "reference policy").  
* **Catastrophic Goodhart**: Research warns of "Catastrophic Goodhart," where optimizing a proxy too aggressively leads to a sudden collapse in true utility. The solution is **Iterative Reward Refinement**: constantly updating the verifier (e.g., adding new test cases) as the model discovers exploits.30

## ---

**5\. Neuro-Symbolic Abduction: Integrating Inference to the Best Explanation (IBE)**

While MCTS and RLVR handle the "Search" and "Learning" aspects, the "Abductive" core—the logic of explanation—requires a neuro-symbolic approach. We propose integrating **Inference to the Best Explanation (IBE)** frameworks to operationalize the evaluation of hypotheses.

### **5.1 Formalizing Abductive Criteria**

IBE theory suggests that the "best" explanation is one that maximizes specific epistemic virtues. In the NSAR architecture, we explicitly score hypotheses on these dimensions rather than relying on a generic "likelihood" score.52

| IBE Criterion | Definition | Implementation Mechanism |
| :---- | :---- | :---- |
| **Explanatory Power** | Does ![][image1] entail ![][image2]? | **NLI Entailment Head**: \`P(Entails |
| **Parsimony** | Is ![][image1] simple? | **Minimum Description Length (MDL)**: Measure the token count or Kolmogorov complexity proxy of the hypothesis. Prefer shorter paths that explain the evidence.33 |
| **Coherence** | Is ![][image1] consistent with knowledge? | **RAG-based Verification**: Retrieve external facts related to ![][image1]. Check for contradictions using an NLI model. |
| **Uncertainty** | Is the confidence calibrated? | **Graph-Based Calibration**: Construct a consistency graph of multiple samples. Score based on centrality in the cluster of semantic equivalents.55 |

### **5.2 The Neuro-Symbolic Interface**

This approach creates a "Symbolic Scaffold" around the neural generation. The LLM proposes hypotheses (Neural), but the selection is governed by these symbolic/logical scores (Symbolic).

* **Autoformalization**: Advanced implementations can translate the natural language hypothesis into a formal logic representation (e.g., Prolog or Python).  
* **Deductive Verification**: Once formalized, a symbolic solver can check for logical consistency. This ensures that the "abductive" inference is at least logically valid, even if the premises are uncertain.57

## ---

**6\. Proposed Architecture: The Neuro-Symbolic Abductive Reasoner (NSAR)**

Synthesizing the critiques and improvements, we present the **NSAR Specification**, a direct evolution of PAS designed for robust, verifiable reasoning.

### **6.1 System Architecture**

The NSAR system operates as a four-stage dynamic loop:

#### **Phase 1: The Abductive Proposer (Neural)**

* **Input**: User Query ![][image16] \+ Observation ![][image21].  
* **Action**: The LLM generates ![][image8] initial hypotheses ![][image22].  
* **Enhancement**: **Nudging**. A lightweight "Nudger" model (7B) monitors the generation entropy. If uncertainty spikes, it injects guide tokens (e.g., "Consider the constraints...") to steer the proposer away from known collapse modes.39

#### **Phase 2: The MCTS Engine (Search)**

* **Structure**: A tree where nodes are partial reasoning states.  
* **Selection**: UCT algorithm selects nodes, balancing the LLM's prior ![][image4] with the empirical value ![][image23].  
* **Expansion**: LLM generates next steps.  
* **Evaluation**:  
  * *Fast Path*: Query the **Learned Value Function (PRM)**.  
  * *Slow Path*: Perform a **Lightweight Rollout** (simulate to end).  
* **Pruning**: Branches with low IBE scores (low explanatory power or high complexity) are aggressively pruned.

#### **Phase 3: The Constitutional Critic (Reflexion)**

* **Trigger**: If the search fails to find a high-confidence terminal state (or if verification fails).  
* **Action**: The system enters a **Reflexion Loop**.  
* **Critique**: The model critiques the failed traces using **Constitutional Principles** (e.g., "Check for logical contradictions").  
* **Memory**: The critique is stored in the **Episodic Buffer** and prepended to the context for the next MCTS iteration.

#### **Phase 4: The Verifier & Learner (RLVR)**

* **Terminal State**: Once a solution is proposed.  
* **Verification**: **External Verifier** checks the result (e.g., runs the code).  
* **Learning**:  
  * *Success*: The trace is added to the **Golden Replay Buffer**.  
  * *Failure*: The trace is added to the **Negative Buffer**.  
* **Update**: Periodically, the policy ![][image24] and the Value Function ![][image25] are updated using **GRPO** on the Replay Buffer data. This closes the loop, ensuring the system gets smarter with every query.43

### **6.2 Comparative Performance Profile**

| Feature | Legacy PAS | Proposed NSAR |
| :---- | :---- | :---- |
| **Search Strategy** | Uninformed BFS/DFS (Heuristic) | **MCTS with UCT & Learned Values** |
| **Scoring Metric** | Bayesian Score (Logits/Cosine Sim) | **IBE Features (Entailment, Parsimony)** |
| **Self-Correction** | Autoregressive Self-Critique | **Constitutional Reflexion \+ Adversarial Play** |
| **Learning Signal** | Self-Consistency (Echo Chamber) | **RLVR (Verifiable Ground Truth)** |
| **Complexity** | **![][image10]** (Exponential) | **Anytime Algorithm** (Budget-Constrained) |
| **Failure Mode** | Hallucination Amplification | **Search Exhaustion** (Fail-Safe) |

## ---

**7\. Conclusion: The End of "Probabilistic Guessing"**

The analysis of the 'PAS' specification reveals that while it correctly identifies the *components* of reasoning—abduction, branching, and learning—its reliance on "System 1" heuristics (semantic similarity, unguided expansion) limits its capacity for true "System 2" deliberation. The assumption that probabilistic coherence in a latent space equates to logical validity in the real world is the "Original Sin" of current LLM reasoning architectures.

The transition to the **NSAR** framework represents a fundamental shift. By replacing passive Bayesian scoring with **Active Value Estimation (MCTS)**, replacing "vacuous" self-critique with **Constitutional Reflexion**, and grounding the entire process in **Verifiable Rewards (RLVR)**, we move from a system that "guesses" to a system that "thinks." This architecture does not merely scaffold the LLM; it embeds the LLM within a rigorous, optimizing search process that is mathematically guaranteed to converge on better solutions given sufficient compute and verifiable data. As of 2026, this integration of neural generation with symbolic search and verifiable reinforcement stands as the definitive blueprint for the next generation of reasoning engines.

1

#### **Works cited**

1. Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2503.17523v1](https://arxiv.org/html/2503.17523v1)  
2. Probabilistic Reasoning in LLMs \- Emergent Mind, accessed on January 28, 2026, [https://www.emergentmind.com/topics/probabilistic-reasoning-capabilities-of-llms](https://www.emergentmind.com/topics/probabilistic-reasoning-capabilities-of-llms)  
3. Benchmarking the Confidence of Large Language Models in Answering Clinical Questions: Cross-Sectional Evaluation Study \- JMIR Medical Informatics, accessed on January 28, 2026, [https://medinform.jmir.org/2025/1/e66917/PDF](https://medinform.jmir.org/2025/1/e66917/PDF)  
4. Calibrating LLM Confidence with Semantic Steering: A Multi-Prompt Aggregation Framework \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2503.02863v1](https://arxiv.org/html/2503.02863v1)  
5. Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification, accessed on January 28, 2026, [https://openreview.net/forum?id=MQXrTMonT1](https://openreview.net/forum?id=MQXrTMonT1)  
6. Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration? \- ACL Anthology, accessed on January 28, 2026, [https://aclanthology.org/2025.acl-long.722.pdf](https://aclanthology.org/2025.acl-long.722.pdf)  
7. Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2505.15918](https://arxiv.org/html/2505.15918)  
8. Integrated Subject–Action–Object and Bayesian Models of Intelligent Word Semantic Similarity Measures \- MDPI, accessed on January 28, 2026, [https://www.mdpi.com/2079-8954/13/10/902](https://www.mdpi.com/2079-8954/13/10/902)  
9. A Novel Bayesian Similarity Measure for Recommender Systems \- IJCAI, accessed on January 28, 2026, [https://www.ijcai.org/Proceedings/13/Papers/386.pdf](https://www.ijcai.org/Proceedings/13/Papers/386.pdf)  
10. \[1808.01983\] Probabilistic embeddings of the Fréchet distance \- arXiv, accessed on January 28, 2026, [https://arxiv.org/abs/1808.01983](https://arxiv.org/abs/1808.01983)  
11. Hilbert Space Embeddings and Metrics on Probability Measures \- Journal of Machine Learning Research, accessed on January 28, 2026, [https://www.jmlr.org/papers/volume11/sriperumbudur10a/sriperumbudur10a.pdf](https://www.jmlr.org/papers/volume11/sriperumbudur10a/sriperumbudur10a.pdf)  
12. Efficient Detection of LLM-generated Texts with a Bayesian Surrogate Model \- ACL Anthology, accessed on January 28, 2026, [https://aclanthology.org/2024.findings-acl.366.pdf](https://aclanthology.org/2024.findings-acl.366.pdf)  
13. Towards Improved Proxy-Based Deep Metric Learning via Data-Augmented Domain Adaptation, accessed on January 28, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/29400/30645](https://ojs.aaai.org/index.php/AAAI/article/view/29400/30645)  
14. Word Embedding Uncertainty Estimation \- OpenReview, accessed on January 28, 2026, [https://openreview.net/pdf?id=dA8JkhBHjZ](https://openreview.net/pdf?id=dA8JkhBHjZ)  
15. Can Uncertainty Quantification Benefit From Label Embeddings? A Case Study on Local Climate Zone Classification, accessed on January 28, 2026, [https://elib.dlr.de/215562/1/Schweden%20et%20al.%202025.pdf](https://elib.dlr.de/215562/1/Schweden%20et%20al.%202025.pdf)  
16. Clarification of Assumptions in the Relationship between the Bayes Decision Rule and the Whitened Cosine Similarity Measure | Request PDF \- ResearchGate, accessed on January 28, 2026, [https://www.researchgate.net/publication/5431834\_Clarification\_of\_Assumptions\_in\_the\_Relationship\_between\_the\_Bayes\_Decision\_Rule\_and\_the\_Whitened\_Cosine\_Similarity\_Measure](https://www.researchgate.net/publication/5431834_Clarification_of_Assumptions_in_the_Relationship_between_the_Bayes_Decision_Rule_and_the_Whitened_Cosine_Similarity_Measure)  
17. Effectively Searching Trees of Thought for Increased Reasoning ..., accessed on January 28, 2026, [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/KamyarJohnSalahiPranavGurusankarSathyaEdamadaka.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/KamyarJohnSalahiPranavGurusankarSathyaEdamadaka.pdf)  
18. Tree of Thoughts: Branching Reasoning for LLMs \- Emergent Mind, accessed on January 28, 2026, [https://www.emergentmind.com/topics/tree-of-thoughts-tot](https://www.emergentmind.com/topics/tree-of-thoughts-tot)  
19. \[D\] Bitter Lesson and Tree of Thoughts \- Are techniques like ToT examples of using search or are they ignoring the bitter lesson by encoding humanlike learning? : r/MachineLearning \- Reddit, accessed on January 28, 2026, [https://www.reddit.com/r/MachineLearning/comments/1893ne2/d\_bitter\_lesson\_and\_tree\_of\_thoughts\_are/](https://www.reddit.com/r/MachineLearning/comments/1893ne2/d_bitter_lesson_and_tree_of_thoughts_are/)  
20. REKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs via Training-Free Monte Carlo Tree Search \- ACL Anthology, accessed on January 28, 2026, [https://aclanthology.org/2025.findings-acl.484.pdf](https://aclanthology.org/2025.findings-acl.484.pdf)  
21. Policy Guided Tree Search for Enhanced LLM Reasoning \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2502.06813v1](https://arxiv.org/html/2502.06813v1)  
22. REFLEXION: LANGUAGE MODELS THAT THINK ... \- OpenReview, accessed on January 28, 2026, [https://openreview.net/pdf?id=FDG2G7JDWO](https://openreview.net/pdf?id=FDG2G7JDWO)  
23. Self-Criticism: Aligning Large Language Models with their Understanding of Helpfulness, Honesty, and Harmlessness \- ACL Anthology, accessed on January 28, 2026, [https://aclanthology.org/2023.emnlp-industry.62.pdf](https://aclanthology.org/2023.emnlp-industry.62.pdf)  
24. Model collapse \- Wikipedia, accessed on January 28, 2026, [https://en.wikipedia.org/wiki/Model\_collapse](https://en.wikipedia.org/wiki/Model_collapse)  
25. Model Collapse and the Right to Uncontaminated Human-Generated Data, accessed on January 28, 2026, [https://jolt.law.harvard.edu/digest/model-collapse-and-the-right-to-uncontaminated-human-generated-data](https://jolt.law.harvard.edu/digest/model-collapse-and-the-right-to-uncontaminated-human-generated-data)  
26. \[2404.01413\] Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data \- arXiv, accessed on January 28, 2026, [https://arxiv.org/abs/2404.01413](https://arxiv.org/abs/2404.01413)  
27. SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2504.19162v2](https://arxiv.org/html/2504.19162v2)  
28. Study identifies weaknesses in how AI systems are evaluated | Hacker News, accessed on January 28, 2026, [https://news.ycombinator.com/item?id=45856804](https://news.ycombinator.com/item?id=45856804)  
29. Goodhart's Law in Reinforcement Learning \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2310.09144v1](https://arxiv.org/html/2310.09144v1)  
30. Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification \- NIPS papers, accessed on January 28, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/1a8189929f3d7bd6183718f42c3f4309-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/1a8189929f3d7bd6183718f42c3f4309-Paper-Conference.pdf)  
31. Reflexion | Prompt Engineering Guide, accessed on January 28, 2026, [https://www.promptingguide.ai/techniques/reflexion](https://www.promptingguide.ai/techniques/reflexion)  
32. Tree-of-thought in LLMs. What should we work on, how should we… | by Junpei Komiyama | Medium, accessed on January 28, 2026, [https://medium.com/@junpeikomiyama/tree-of-thought-in-llms-e5004db16c99](https://medium.com/@junpeikomiyama/tree-of-thought-in-llms-e5004db16c99)  
33. LLM-MCTS, accessed on January 28, 2026, [https://llm-mcts.github.io/](https://llm-mcts.github.io/)  
34. Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2506.01374v4](https://arxiv.org/html/2506.01374v4)  
35. Compiler Optimization via LLM Reasoning for Efficient Model Serving \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2506.01374v1](https://arxiv.org/html/2506.01374v1)  
36. ReST-MCTS∗: LLM Self-Training via Process Reward Guided Tree Search \- NIPS, accessed on January 28, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf)  
37. SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement | OpenReview, accessed on January 28, 2026, [https://openreview.net/forum?id=PHu9xJeAum\&referrer=%5Bthe%20profile%20of%20Furong%20Huang%5D(%2Fprofile%3Fid%3D\~Furong\_Huang1)](https://openreview.net/forum?id=PHu9xJeAum&referrer=%5Bthe+profile+of+Furong+Huang%5D\(/profile?id%3D~Furong_Huang1\))  
38. REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving \- OpenReview, accessed on January 28, 2026, [https://openreview.net/pdf?id=2D4TuZyNnr](https://openreview.net/pdf?id=2D4TuZyNnr)  
39. Nudging: Inference-time Alignment of LLMs via Guided Decoding \- Yu Fei, accessed on January 28, 2026, [https://fywalter.github.io/nudging/](https://fywalter.github.io/nudging/)  
40. Inference-time Alignment of LLMs at the Token Level \- OpenReview, accessed on January 28, 2026, [https://openreview.net/forum?id=HgAS03GU4J](https://openreview.net/forum?id=HgAS03GU4J)  
41. NUDGING: Inference-time Alignment of LLMs via Guided Decoding \- ACL Anthology, accessed on January 28, 2026, [https://aclanthology.org/2025.acl-long.623.pdf](https://aclanthology.org/2025.acl-long.623.pdf)  
42. SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning \- arXiv, accessed on January 28, 2026, [https://arxiv.org/pdf/2504.19162](https://arxiv.org/pdf/2504.19162)  
43. Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs | OpenReview, accessed on January 28, 2026, [https://openreview.net/forum?id=jGbRWwIidy](https://openreview.net/forum?id=jGbRWwIidy)  
44. Claude's Constitution \- Anthropic, accessed on January 28, 2026, [https://www.anthropic.com/constitution](https://www.anthropic.com/constitution)  
45. Constitutional AI: Harmlessness from AI Feedback — NVIDIA NeMo Framework User Guide, accessed on January 28, 2026, [https://docs.nvidia.com/nemo-framework/user-guide/24.07/modelalignment/cai.html](https://docs.nvidia.com/nemo-framework/user-guide/24.07/modelalignment/cai.html)  
46. Code vs. Character: How Anthropic's Constitution Teaches Claude to "Think" Ethically, accessed on January 28, 2026, [https://www.arionresearch.com/blog/code-vs-character-how-anthropics-constitution-teaches-claude-to-think-ethically](https://www.arionresearch.com/blog/code-vs-character-how-anthropics-constitution-teaches-claude-to-think-ethically)  
47. Claude's Constitution \- Anthropic, accessed on January 28, 2026, [https://www.anthropic.com/news/claudes-constitution](https://www.anthropic.com/news/claudes-constitution)  
48. RLVR: The Training Breakthrough That Will Make Reasoning AI Verifiable \- Medium, accessed on January 28, 2026, [https://medium.com/@raktims2210/rlvr-the-training-breakthrough-that-will-make-reasoning-ai-verifiable-cf4209e79669](https://medium.com/@raktims2210/rlvr-the-training-breakthrough-that-will-make-reasoning-ai-verifiable-cf4209e79669)  
49. Reinforcement Learning from Verifiable Rewards \- Label Studio, accessed on January 28, 2026, [https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/](https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/)  
50. Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2407.14503v1](https://arxiv.org/html/2407.14503v1)  
51. Goodhart's Law in Reinforcement Learning \- OpenReview, accessed on January 28, 2026, [https://openreview.net/forum?id=5o9G4XF1LI](https://openreview.net/forum?id=5o9G4XF1LI)  
52. Inference to the Best Explanation in Large Language Models \- ACL Anthology, accessed on January 28, 2026, [https://aclanthology.org/2024.acl-long.14/](https://aclanthology.org/2024.acl-long.14/)  
53. Inference to the Best Explanation in Large Language Models \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2402.10767v2](https://arxiv.org/html/2402.10767v2)  
54. Inference to the Best Explanation in Large Language Models \- ACL Anthology, accessed on January 28, 2026, [https://aclanthology.org/2024.acl-long.14.pdf](https://aclanthology.org/2024.acl-long.14.pdf)  
55. Thinking Machines: A Survey of LLM based Reasoning Strategies \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2503.10814v1](https://arxiv.org/html/2503.10814v1)  
56. Cleanse: Uncertainty Estimation Approach Using Clustering-based Semantic Consistency in LLMs \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2507.14649v1](https://arxiv.org/html/2507.14649v1)  
57. A Balanced Neuro-Symbolic Approach for Commonsense Abductive Logic \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2601.18595v1](https://arxiv.org/html/2601.18595v1)  
58. Verification and Refinement of Natural Language Explanations through LLM-Symbolic Theorem Proving \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2405.01379v2](https://arxiv.org/html/2405.01379v2)  
59. Inference to the Best Explanation in Large Language Models \- ChatPaper, accessed on January 28, 2026, [https://chatpaper.com/paper/48833](https://chatpaper.com/paper/48833)  
60. \[2512.17102\] Reinforcement Learning for Self-Improving Agent with Skill Library \- arXiv, accessed on January 28, 2026, [https://arxiv.org/abs/2512.17102](https://arxiv.org/abs/2512.17102)  
61. Stabilizing Reinforcement Learning with LLMs: Formulation and Practices \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2512.01374v3](https://arxiv.org/html/2512.01374v3)  
62. \[2512.01374\] Stabilizing Reinforcement Learning with LLMs: Formulation and Practices, accessed on January 28, 2026, [https://arxiv.org/abs/2512.01374](https://arxiv.org/abs/2512.01374)  
63. WEBRL: Researchers advance web Agents with RL\! | ml-news – Weights & Biases \- Wandb, accessed on January 28, 2026, [https://wandb.ai/byyoung3/ml-news/reports/WEBRL-Researchers-advance-web-Agents-with-RL---VmlldzoxMDEzMDQzMQ](https://wandb.ai/byyoung3/ml-news/reports/WEBRL-Researchers-advance-web-Agents-with-RL---VmlldzoxMDEzMDQzMQ)  
64. Towards Trustworthy LLMs via Calibrating Knowledge and Reasoning Confidence \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2601.11956v1](https://arxiv.org/html/2601.11956v1)  
65. Constitutional AI: Harmlessness from AI Feedback \- arXiv, accessed on January 28, 2026, [https://arxiv.org/pdf/2212.08073](https://arxiv.org/pdf/2212.08073)  
66. Harnessing Verifiable Reference-based Rewards for Reinforcement Learning of Open-ended Generation \- arXiv, accessed on January 28, 2026, [https://arxiv.org/html/2601.18533v1](https://arxiv.org/html/2601.18533v1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAUCAYAAABSx2cSAAAA1klEQVR4Xt2SPQ4BURRGb0KikEgkWivR+IkdWAZKpUhUSpuwAIlCRKNlBQob0EhUNHzXvY/xvZlZwJzkNPfMD3eeSCGowy3ckQPv3P6owja8whc8wy5seO/Aube9zyI0qi0OYCLWahwCGu+wzAGsxXomGjc8FHvYTXJu7onFodj/T7r09vxcmcJM7OnJrZ7gRX67OISLGd3iioeO7kFv1o1H6Kd6wBEHJ7y5z0FZiMUSB0dbdDgCR8neZEWsTTkoeorCz0ojfAU9ZV+aPmTH3nme94LC8waQ8UJbWBk6PQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAVCAYAAACdbmSKAAAAyUlEQVR4XmNgGFJADoh3E4FRgDwQ1wLxfyg+AuXDcB0QH4erRgMwTQHoEkAgiC4AAyANJ9AFCQGQpk40sTNofBSgygDR5IkkZgjEz5H4GCCdAaLJG4gdgLgQiG8D8XIkNRhgBQMiIJBxNrIiZKDPAFHghySmDsSPkfgWSGwwyGeAaOJFEjMH4hokPoYz1zJANOECygwQ/8EBIxB/YMCvaTMQV8I4zEA8gQHhaUkgZgFiaSC2B+JkID4LxIthGuSRFBPCVlA9ww4AAGV1OkAZT4NHAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAYCAYAAAC8/X7cAAACfElEQVR4Xu2XzatNURjGXx/XhMQNE8o1EFI3TBiQE2FgwIj8ASaGiuiGlI9EkbGBUkSJKAaiCAPJR6IUka+SgSIGJJ6nd61zVs9d6+x97z3X6PzqqbOed613r7323u9ax6zLIO5DE9QcAeOgRWq2YznUSNQPzYGmN3uUmQqtUTPQyGhZiPWF9gxoErQYWhBi5C00K2kXmQz9baOX0C5oWhyQwJW6pWaC5qK+hhifmsZuhhg5AD0xn18tnponuSD+qeC/E5+chD6oKWw1H7/B/Gkp8aaWiD8eeg8dF79IXIWFGgB3zWMpJ6Cflp9UykMbPDbCV4exbRoI9EI/1CzBRJ/UBGPNV0gn8QW6Ip7CCcSFybHOPMZvrsRl86dRCROdUxOssvwk2N4rnrLR8mMjh6z1TZTYA61QU1ltfpH54vcEn6/QTInRnyee8sC83wvziaj+QNubvfPwGqfVVPabJzsPHYbOmk/6G3TE/EYUTowlsB1x9R9BN0SxaCxt9s7Da1xVU2EpZMlKV2cAWpt2EnjxMWoK8QbS+h7ZZx6rer/5DbLkFuEmxETczIYCxzB5O9jnjJpgIvTLyt+G8lyNlINWbyUUjpmiZgJj7MN9QFlvradTBTeye2qmMFgnkcIxfWombDLvM1cD4Kh57LsGMsyGrqlJGtBua60E2zz/1IVjdqpprTPPG+hO+B3LICsK2/Ga3OWrDm07oM1q8h2MSVIdSztV8Nu8aimak/ocYo8zseshVoJ7U+4cNmL46r1WcxR4pUanYAV6Bl3SQAe5CN1Ws5PwBFn7sDVEWN6Zm/8TRpUtVn0iHQ4fzSvZf2Gl5Y8bw6WTf0+7dFH+AaV5p4CITasaAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAYCAYAAACsnTAAAAADUUlEQVR4Xu2Y26tNURTGP9fkljuldDwRhTyQEtM1/4ByLZFcopQSiVIkJSGPUpLrC/GgKHdyidwiioSUKKVQJMZ35prO2GPNudZ2zt7naf/qqzXHmLc95hpzzrWBBu3CLVFXa2wlU0QHrLEGHLeGMiaLnNIY0XDRwH810vQVzbLGNuBEF60xozcq5xnEBemTPfO3dBONyMqBwaKZqlwIB/pToBeijaIBoYGik+iSNSpsXymdCg1QHJTFyLelJop2RexUz+aWni/wC141j5CfIDmY2d8aO9kvem+Nil6i7fDtRyr7MNFY+JWjT6eLQzoohG9A+MHsxzIX3rdA1N34TsD/zqoJA422DuEGvE+zV/QdPn2KeIV8W81N0TxVdigOyh6kF4lcQ/H+cUf03BpTcKAP1ih0hH/t7A/7JDprbJYmtARbs0w93xdNUmWH4qDcg+/vqHXAp8pP0RrrUKxHfj5JWDEW4emI/zCWtxqbZRHibR+r55ei/qrsUByU0N8q6xDmwPvGWYfCIT+fKDPgK+q8J10yO9NnqPHRzvwuIuxTT0VbRDtEV0VLVB2LQ3lQfsH3Z/Us85eRSr0Ktol+i06KdoqOwQfiK/yuzuBYOPggazSEVX0gegifciw3qToWh/KgvIavo8Vgh/HK4FxK4bHKijrqm0WzdSUDB+9gjYYwSf0GblDPMRzSQekH398K60A6zWNcRuVRnYMXL3bES8//wDbchItgncPGtlA988i0KehEF4wtwJPnnTVmXIcf75x1RDiP/HZQAfOcnXW2jhLYhrfIFEPg6yy3DsVaa0BxUJiGR6wxg+nP8creRHJb1MMaNbwnVPPKWdimyRoV85FPHQ0vXp+tEemgcP9if/o414TU0cd7Ct7SozjRJrR0xjK/d6oltSou0xvRFVUOmib6geL2OihcUdrOoGWeFG/L/J4JZfruZs9F8JMg+hJwoBAMrd26Ugk8FnlaWWyfKfFki92GHSqDshT5thTvIvsidoofhClWw9epC0w7XuFrjUM8fWrFIdRn3s10FD0RnbaONuJQv6CsE32DT726MR5+kFriUL+gfBSttMZ6wFMmtje0Fof6BYV/g7QbUxH/FGgNvNCNssYaMMEaGjRoUBP+AnVO+g3U4HnhAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHIAAAAYCAYAAAAmsqlBAAADTklEQVR4Xu2YWchNURTHl8whwoPMEZ7wIiHD5QEvUsqLsRCSIVN4MJQhUpS8GJ5QkhdjFJLyIOFFKKlPKSIZo5BYf+vse/ddZ59J+9xL7V/9+76z/vucs84+e++z9iUKBAL/H11ZL3TQE2NZlRSNoH+DSorwDK2pceB+M3QwD/tYv3TQE7hulmZXWzcPnZPWG1bbautyucNaoYNZDGZ9JUm2TO6zzrBaWTGM8mUk915qxZvFQZJcOqn4eJIXeVLFy2AeSQ7btJFFC+sQlfsiJ5Jcf6CKG+D9pPqX3GjasD5Tcj+sJfEmaMMjB1inSWYkBn1u2rPmsNaTJNmu3vbGFkruILw8eFe10WAw68wy6uIE6z2rozY88pE1iHWOdVN5qWyM/i4keYC+lueTK5TcQUtY31mjtdFgzGB7pA2SguwLyawsk93R3+Osx7aRBpbTo9H/6EQ8xLSa7Q17pG+1hOWjhaQD85BV/Wr1o2J8IsnxLNVy3EEyCw+z+lRb+ge1AmoIwwaSXDpYMScYYa9ZPaPjISQnzq+28IcZ6U+p/kVeZD1grak1TeU861oBrZLTcoMc8Z3eTvV5PmftZ/WotvQPcrWX0sUk+fS3Yk5us5Zbx0gSJ66zYknsIlnDh2sjAbOsoiLU4LsDD9/qZoM87FlhMJ36ShsO0I9YFrtpIwVMJlTEw6zYLJJ7jrJiMTqzrlN8BKNiO2a1S+IZychdqY0EkBDUWxtMdxIPD99MULEij7naiLhH4meBivYDa4w2UnhH8Xdxi+R+C6x2MbBMuLjLuqCDDlDZDtDBFJBQ0od7EomfZ3nFd7xSQEMpPyYP12DD8xbZZ2Nw5gUVu2vwYEbjfqYYjYEX8FIHI7AEogDxDRI6ooMkD3GZxM9TmFyi+MhN02o5LRfYfD/RwYhNVFtVfLNQByxwv9ikQ8WHwgLmKeWNJBnBN1g/WDNZ4+wGf4GZPWZ/ir0qjqHJrM1R/C25R2SjqLCmk2x/9kbHRotIBgTy3EP+fp7DDEe/YEnFtXvV239Wh6mR95A1xRhdWN8iwzWy8KG1PVebouhraWFPhl8xMquyktF52UIdgG8jfpXyCbZ49n12Wh52EzoPKBAIBAKBQCAQCAQCgUL8BvrQ+Bgb4rfiAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAA2klEQVR4XmNgoAcwAmIHIPYD4iJUKUzwDYj/Q/FzNDmsIJYBong1ugQ2cJQBotgYXQIdcADxbyD+DMRMaHIYwIkBYup2JDFmINZD4sMBSBFIsQOUvxuINYF4GhBzQsXg4BMQ/2SAOCcIiFmAuJgBYgCGs0CCh4A4H4inQ8UsGSA2YACQ4mVAfBLKFkGVRgCQR0AKRIGYEYgnA/EMFBVIwACIryPxrRkQocKLJA4G84A4HIkPSid9UPY+JHEwOMAAcQIy+M4AiVFQiKAAGXQBIJACYgl0wVFAFgAAPUolrRnO7yYAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAAyUlEQVR4XmNgoCdgA2IHII4C4mhUKVQQDMT/oXgxmhxWAFKYjS6IDoSA+D0QM6FLoINAIF6PLogM5KF0HxAXIUuAAMj4DCC+BMTfgbgKiE8DsQmyIhAAORpkAgzUQcVQgAQQfwViESSxFgYsCmcD8Vo0sf0MWBR+BuJEJD4XEP+CisOBIgNEJyOSWCNUrBaIM4HYBSQIitNPSIpA4BoDRKErEN8BYkmYRAcQG0DZukD8gAGi0A6IN0LF4QDkLpAEBxLfHCE9CqgBAIRCJD3f5k+MAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAUCAYAAAAUccS4AAACXUlEQVR4Xs2WTYhOURzGn3x/JEypIfKRohA2mljMsGCK5GPFrNgxNbIUi7EgG6GIRix8bCS2GlFKskHKQlFKythQFBkL83/6n+Oeee6515h30vurp7nnec4997znc4DmZJXpoqldg2Zkoem3abkGf6PN1AFvYCRcUmOU3E2e15jeJeVhTDIdNt0PemD6YdqTVsrQbZos3ibTC/hIPZRM4cAcNLWaeiTbJ+U/vDK9MXUm3mz4B6+j3CHC9fVLTWM+vB2+e1ayFNY7BR+Q16YVw+Myp+GNjtcgsAOefxZ/Z/A5MjmmwfMWDQJHTYNJmbNZS/z1XzVIWAavQ6X0w0ejii2mJ2omPEKxRKabfiZZlvfwThzTIGEO8p1luU+8FE7vCTUT+O0z4ZlrnO3xB7LjJebCK9zQQNiL6s6uEy/CqWfOpUA4g+zIEdP34K01vYR/n9/ghtweshJc1GxwvwbCeVR3dpF4kd0o6m8z3YF3lF463VNME8Jz/JvlAvzleRoIb+H1nopPb6J4EZ4AzHn0XIZvXh5L9M4l9UbMPdM3NTPEUd0l/hcpp8Qz9rZpnGSjIk5L3fDH6dyqAdyfoSaKd3gUzTQ9DuWG4GJmI0s1CKw0fUT5Vonw3QVqolhex0P5UCgTbrgr4fmfmApvJHe8cKcOoH5EmG1UE34TMusI5ZOhTLpQf6PVwtvnGbyxXvi/aM9NN02ri2pZuKtvqQlv61pS5qBwf/CC4BHVENypm+Fr+ADKG6kK3kAf1DTWI39KzFLjf8IprVoKTcli0yc1x5Ds1dsIG0xL1Bwjrg4BjjqEsdaBbQ8AAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,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>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAYCAYAAACr3+4VAAADDUlEQVR4Xu2WW6hNURSGf7fI/X4JCUlRciskzkEpoog8OB5O8kBJlFKKDk8kxIMipdzKg4gXpUTul3ggifLgnge5hJAYv7GXM9e/11rW2Zy9S/urv3PmP+Zce8255phzAFVK5oqptZplYKVpv5p5aGuqMZ02nYU/5E8T6GWapmaZ4LtdNG3UQBZLTS9Mb00LTUtMh0w3TH2DfiFtTJfULDMDTK/VTGOo6YfplKmPxOg/NvUXnxwwPVOzAqwx9VMzpJXpGnwyaSyGx7+Lv8f03tRJ/ErxCRmTXQufxCMNBIyG99HF4BY/Ll4l4ftxPol8gHdYpoGArkieKNvrxVPam0aZBmugGfhmOqEmGQ9/2U0aEHgwpU10mHghC0yTgvb94P889DBdgKcHGWh6aXr1u0ecY/B3KjpLVhUCnHAWW5A+0Z7iRTD3nwbtMfCXbApn4L+xIvC2Ij1d9sH7M9ViHIZ/bt6dWdyBP+Cm+PRaiBcxHB6vNbU0dTbNDDvkgOMfwsdH7EV6Hm6Dj5mhAQ66p2YCHHwbxZPiImXBIz/aCVS7eDiTkfAxEwKvd8FL+zAN8Ph88bHa9FFNYYTpq2mKBuAP7aCmwK8Ybak6iWVRb3qH+NeMzoo0dsHj0zXAl2BgUKHNT88K42ShzR+5bFpUaCscW5T4xg54Ps4LvJ3wSisvXJRz4kUTIaxxNRUOwuNjxf/FE/gW1FVgncv8mCV+CB/KXaHwyuLEInj6fgnahNdStKXTeAPPbcIy8zm8P98p6Rr5bLquZkQX0274AxpMG+AF/XY05lRHJOcFF4irqMyGT5ZXAxeSxcjyWA+/V88ju1A5YroLX3Smzjj4VcMPwJRSOIfNairdTOvgKz3HNDeIcSGS7kuWjmmHGReGJy8XKQvdfkp30+SgzQJkSNAO4cKzwG8S0baipkosgnflA9NRDeSEi8Av/i9gLVCjZh7CiWYxEV5MlwLzuNRFCuGheVXNvHBL8u4Mt3Aa9Wg8NPLCLcjKh3//FqbbLTWbi1rE77xywXOAhUSVKv8TPwFg86oNMxstTAAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAYCAYAAACr3+4VAAACxElEQVR4Xu2XS6iNURiGP3e5DVwT6Sgycku5jGyScivFjCITjFzKRBmQmIikJIUYIEWZiSiFFDJwDQO5JpIIyYD3be119rff8/9r/2ez96H2U2/nrPf71t5r/f9a31rbrEXd3IB6qtkEzkCr1EwxGyo5TYTaoKExIcEwaI6aTWI09Baap4EshkO/EroPbYAGxQ6OXtA1NZvMAuiTmimeWJjYcfHPlv0H4pNj0Cs1u4BzUDc184hvsE188shCzHMQ+gwNFL+ruGsFJ8uJPFMT9IN+WseJcrnwbf8rcHzT1cyCiUfUBMut8rY9bG8VT+FDmgSN1UAD4Hg2q6ksspA4Svy+Zf8iNERi9MeL51kGzXLth+7/Iky1sP/vldtLoHfQqfaMahjLqiNV7IZ+QKehXeW/PBu/Qtug7pXUdjjRvOOnB/TStTloHgOd4bWFMU12Hse00bU9j6H3airXoVsWJhXFZTnXJwmcaN7mn2AhXrLwkHg0zfcJBWD/o+KxKObtw5uWHpMttpAwTQM1YIFKsckqe5viNijKeuijhXM6EmtFHpctxAdrILLHajyJHNinv5oC3+JhC7krJJaCRfG8ePstPVEeL8l53Lb0B+TBPlq8yF4L+3Gp8/ZBK127Foeg7eLFiRA+vBEuRp5bWAUd4D7iXoxLqwSN8Qk1YJ+swvDFwsQirL4sKp4LVvneLFjhL7k2K+93C8WGhTPrGOFncXVWMcBCx/hlUTt8Ug24R0+oCRZamOxV6AX0FFpXlRGWNK+cjOXBOzSr7BULK2ethTHegfq4vAhjhS73nYVVjhf+LDgQrhg+0BQ71RDGQVNcm4Umr9iwIufuzz+BZyXPrpMaKAir/Bs162SLhVtYw5gJfVOzAHxIXIJrNFAHI6EPajaC1Zb9WzUFr4b89fM3YNE6oGajKFn2NbHRzIB6q9mixX/Ob54SnaGlOv5AAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAYCAYAAABJA/VsAAAC+UlEQVR4Xu2XWahOURTHl3lWiCdDQpJ5iAxxDSk8SJR0k6KUeEApim5SMhWJMj5IPCkkU+YkD7xJikKGZMhYhAfW/65zOuv8v3POPR/3Ox58v/p37/qvvc/37bP3Xnt/IlUalY6qfmwWxGE2GmKSqibQ2FjGGK2aqGoX/F8TyxpNVRfZLJCbqjo20+ij+kXyYDA/XA76GGthbFVdZbNAuqveqBZwIosHqq9SOmjQQmx2r6mGBbFnnli//uQXzSrVZzazOKnqpvqmWks5MIUNBwY8nc1/xA5VTzbTWB38Pap6qmoZpepZT7GnrLdbYcar1rCZBgoUmCA2c0tcDpyjOARL/QKbCaCqDxWrEZWkjeoUm0nUUhwWrGlBPFz1LErHWKFazKajt1jhaxbEeJl9o3QuboitJpwyKFivgzgNfHe0y2Q/xe/EOp4I4pWqI1E6xnbVHDYdx1RvXfxeNcTFecB3WebiLYGXBnKYqExQuT0YCDr+DGIUubTZ3KOazKYDK+S72LIDu10uD/NVD1XNnbdPGh50uEoTWag6w6ZyWqwz9vhzynl2ibVJA7kPEm2ZL/F0JgPF+oxxXtfAw90hDeTnsuk5KMlH1EyxzrjpHKKcZ4NqFpsEjkK0w97OmiFmkVh7X/wwGHhXnMcgP5VNz2NJXp5NxJYVHoAPT2O5lBbCkCeqzS6eLeUNGs/l9lhZ8DaK7fMZ8XQ9yI9iE4wT229osE2Sj5LOUrrfmdZiez6JWxLt5QFi5z9qgAefD+FSkcRlsR8yoIfqpVj7kWJbkMEReodNkHTf7hRrEbGXjQResBGAl3pXdUn1SnVASl/uddUj1XnyQzqo7ok9477YYD+JrdBBrl3ICNUmNisBXlovNsXOZvhJv9w8uMKeZdOBFYeCGFbwtpL+zKWS44xuDDCDmG2+uuahvdiKSJq1csE+5hpQMTATWKJ1nMjBTtVxNv8AbJvbkn6JqgiDxc7jcsG9Hcv1b1knVj9acaIIurBRELgLVKnyP/IbzBuZ+JFlCKEAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAyElEQVR4XmNgGEzABogdkLAeECsCsShcBQ7AB8T/8eCbQFwBxCIwDdjARQaI4lVo4nOg4o/QxFEAzCZtdAkgOMIAkcMJQJLP0AWBgAmI3zMQoXk5uiAQODEgXIUVODNAJDXQxFmh4iBnS6PJwUEjEP8F4pVA3A7EyxggGj4BcScDxBCcYB8QXwDiWiRcDcRuyIqwAVcGiNNACYVk0MoA0cyCLkEMOMqAJyRxAXUgrmRARIMDAyQ9EwTcQPyZAaERhnuQFY2CkQsAVSI0Z9XrBSwAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAABHElEQVR4XuWSoUtDURTGP8OWJsvCwkAwW8UlNVg2LIJiNilbWxDEIILVorA0WLUMNOji/gKHSVRkbdU1QeZ3PXfj7Lx7fawJ+8EvvO+cc+Hde4D/SIaWaJt2aJNmdUOMPdqnQ3pAd2mDPtNl1ZegSEf0kS5Nl37zAV0xORZoF9IQYxtST/TUfPhhck0BkeFPHx7bgiExvOqDSx0G2EBg+MgH6zoMUEdguOWDRR0GcO/t+l51eE1fdBDBDb7BLEuVfkG2KkaRftOyybEFOXW8PReQZbifdAAP9FB9T/EOOWDT5HeQt3crG8Vd1hXkgHN6Sm/pDc35Hvevf15qHvIkJ7RC91XtjK6p71TG7+rcMbVU9PDMPNEeUi5sXvgB/gFBYbtzG2AAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAYCAYAAACIhL/AAAACN0lEQVR4Xu2WP0hWURjG30qjMgnE/lJDZJCN/cUpa2tQIogiwkkcGnIUEV0EKQwHo5ZAiaamlgiiKEIcgij6AxE1RA2GNURLQ4Q+j+fez/c8cu71g69Jf/CD7z7vOfec795z7z1mq9SGPXAXrINN8DjcGrUwa4Ttzg2u5nkK12hYwG44A09owTMJ58QzUQuzQVejrXF5gQ64XcNlcBr+1FC5ZYuDp2DtCTypBXAQ/tawCu5roHDQognylh/WMGMn/AeHtVAlr61gebRY8QRvaODot9DvkBaqpPAc6y09wQPwk4aO5/APXKsFB68MH8ajWnBw7F4NPdO2dIIX4UPJFPaZ1NBRDydgQ3b8C15YLFeYhe819NyzpRP8AvdKprDPmIaOyxYeohy273HHOR/hdw091y109gt1wP1OwT5DGjruwqsW3rGEfya/mp4X8K8VLJVzFgY7ZuHF/DUuJ2GfPg0d/Ajk65teissVnlmob9FCTpuFBmfhKLwTl5Owz4iGQjvshp/h27hU4Y2F11USvs842G0LC7Y5Lidhn5sagiPwFXzssm2WfhC+WRi3kHfwJVynhQKm4A8NLaw7fmf3Z8e8dbyNGystYvhHr2moPLBwq6uBJ+XJ+bL3cEPBq8fbygeA39tHUYsYnuOUhopuEpZDp4WTn9dCxg4Lk09+xjI+aFBLuqxkgZfAt8AmDWsJr07pjiQBH87S7VYt2AyvaFgC/xjX6bgW/hf7rPzT6OHunRuVVVYm83Vobnl5lx3wAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAYCAYAAAAoG9cuAAAAjklEQVR4XmNgGAVYACMQmwIxJ7oECDAD8TEgXgzl6wDxa4Q0BCQD8X8gNobyQYpBfBQwHyo4AcqPB+KpCGkIkADiCwwQhTCciaICCliA2AGIbzBAFD1BlgS54wgSXxCIvzGgKWoCYk0k/nYgPgPEfEhiDBxAfA+IdwPxCSDezwBxIwYACYICUQ1dYhQwAABhrRk6c/e4TAAAAABJRU5ErkJggg==>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAAAYElEQVR4XmNgGAWjIBRdgBIwC12AEsAIxDHogpSAF0Csji5ILrAD4htAHI0ukQTEu8nAl4H4PxCbMFAI8oD4ABDzoomTBW4CMRe6IDlAioEKXoOBtegClAAjdIFRMNwAAGFcEv/1zy0FAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE4AAAAXCAYAAAClK3kiAAACPElEQVR4Xu2XTUhVQRTH/2noQjclrmwhKNii2rQI/ECthYQgEn0QRLsEoZDAlRCBG5FCtILQjYtQRAtatKpFURHRpi+hlRS4qERoIS0yJP/H89R556n34z2HivnBD96cmffuvDP3zpwLBAKBf4y99CS9SttpSXa3N2rpRXqB1pk+35SbdhE97gYO0jf0O31B/9CPtMEd5IE+ukRfQ+ewQm9kjfDLL/qY3qEP6QKddgd8pYedtmT2LXTySblCB20wJsO01Gmfhc6h04n5ZJzep7foftO3hkxOrHZi1zOxpI9LD9IlrgZ6vTEnJtvFb3rPifmkyQYsMmGZYLUT68rE25xYHGSPTJO4A9DrjZr4PH1uYr5otoE4vEe6R7UX6RK3Faehc+iwHZ4YguZhhv6gI4g4NM9DJyybY1IKlbhi+gHxFu8oNrebuMbhHa3MfJbFk+892OzOZY4u0zO2IwaFSlw3dKJPbYdHqkx7x4XcAy1L6m2HQU7hJ1v4CZp4Gxcb174Zj590AhGPhmcmsU3i5ESUes7lmGlHke8dV0FnTey2aVukeG9JaBRSr9n/cReaOCnXNjhHnyG3Wk5ajuSTuDL6kp4y8ahy5Ahy7+4oo1ikr0xsCuaOO0E/Q1+1WqGvPJehp4hs0knIJ3GPoEVwC7T4vQT9PXmj8I3srbIgLl9gEmdPHNekpE1cA3Kvve4hZ5xPvtEBehM6D9lz92WNKCBpE/c3Igt2DXrHy8LuKvKu2m+DgUAgEAj896wCZSCToTpGOdMAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAA+klEQVR4Xu2SPWtCQRBFR1GbiP4FIeBHZ2lhIQr2ghCEtKIgqKRKKZaCrYKV4C+wtrNJYZ8mIGJjbO1ExNxlB944b7U24IHT3Duz7PIe0SMShnk4hws4hRE5cIsq3MIvWIBF2Iff8NUb85OAJ/gJg9cV7dmkyikAl/CiC4FZMr1vpsuhue4tQvBMjuUDh01dCMztfMtZDiYydFAmx7VbHNRk6GBAjuUZBxkZOtiQnfuR4RgeZeDAvNcsbkj9LG0uojJUVMjO1HVR4iKnCyZF9hOa5zlZkz3gXeU7+AsbKr8iBodkD+jBEVzBDnzxxu4Thx/wDaZV9+R/8geRvjZzI6zRvAAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGUAAAAYCAYAAADjwDPQAAADvUlEQVR4Xu2ZaahNURTHlymRKJlCHhIhQ5lCvCdllgwZI8Mnku9E5IOU4gMZEhmi5JNMmVJEikJkLmX4gFKUlHqx/m/tfe/xf3vfd97rPr2j86t/7571v2e/s8/e+5y19hXJyclpGB1V1aqdqqHkeSpUVaQRzuvnjkepOqhGuuMsUSm1+weBVs5HP1ur+rrjFs4vRR/VEtVv1SbySrJbtYuDxHGxhpP66LxnAQ/KEnztyT4sC8Shgc5Pw3fVT1VXNmKcUy3iYIANUrwgrAjmlOqNapiqOXlZ4YvqjKq/1O6D7/tmiqfhjti5k9iIcU01i4MB/IrYz4bSQ8xbwEaG8CuiGxsOeA85mJKbYufPZCPGddV0DgbwM2UFG8pSKd2hLIDJhj7EgLePgynBxMf5s9mIgVGcxsEAflB6UxwclNIdygJPJd6HLmIeXtoN4arY+XPYiHFXLJuoCzT6SLWFtNV5sQ5lAZ8hQdw/CBP3ReHb9eeSWNvz2QjRXfVD1ZINws+UB2JLMalk9pVV9opd/1ep3T/ol+po4dv1B9kt2j/ABoP65JbqMxsBFoo1iryb2SPmvWXD0Z4DTZDHYn04wobSScxbzYZYqYCMsy56ql6JtdOGvAIognAB31TbyQvxRHWYgw6/Svii14vl8lhJTR3fhwo2xGZ3qafARg5EwMTGvcDjMFak1zBeSv9D4B9doawL+A6h4g3xnANNEFz/Sw46cP2xe4RJN5aDAYaItYFHYSqQdaDgi7FcrEEMTgh47ziYICuDcoiDSi8pTroQ66T4PkYNgnd0iJNibSxmIwbqlCkcFCsIq1QfxBrE53HOG+SOsRMAD1s1OA4RGxTsBeHc2BbPRDH/bCAWiydjE1zsXiKWBH6VaofY91a5Y9DWfT7vPCQAOEbcg4r/k9hjGisBuwCYwCGuiLWTpkivAYMSqlOOSfEGeL133uuAB4WIDQqSBrRzmQ0HZiC2bioDMbTJ8VDshNh7s1ki7uFrT/ZhZSAODXA+GO5iaxKxGL54nMFGDJyQuvxvALFBAVihFzhYZrAZ2BhgLxCbjCgTkDRFsyrlhtigTGUjxkXVXA6WkVKDgnfRYA6WmbUcKBO49m1STJaQ9sb6clvsO5VsxMCuKKez5QDPUazCavc3tJF5mgNlZp78/R4oFxgA3OTJqnZiJUOoxvFgNwDfx29NqUDpj91P/AP8oPOvwM1qjBuWBFlPY5EsAfDSH5M4ToJBw4Dcl3Q/jBVAo6MlnkHl1J/OYjVMBRs5OTk5/wl/AEk3/7+lF5CMAAAAAElFTkSuQmCC>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAYCAYAAAC8/X7cAAACqklEQVR4Xu2WTaiNQRzG/74VInVJpBuFhcTKZzk+FpKvkvKxUCSSom4slBQpFpRSSpKyYKEQFlhhhRWJEpKPhSyEEBLPY2buO+d5Z95zdM/dnV89dd7n/595Z945M/8xa1PiOjRVzR6ySY1mGADNg65At6Cz0MA4IUEf6LCanlpGZIT/PR4aDE2GZvsYeQstjp4bshZ6DX2BNkBroFPQY2hilKccgPqr6fmTETmS8PnuoT6+APoITfPPlXSa6+AmNKY+9M9/D00Snyy3YkA5+GWZw4nyayth8Os1AC5AD9WM4fLftepBLLH6LxdY7b2l4ivHrNw2hrHzakbcg/qpGdhproNX4seMs/QErkKfob7iKw+s3DbAvwtjOzQQ0QXNV5NMMDeARh2Q1AS+m9voVYy0dNtAWN3pGoioQXvVJKHj3AkSWGjpQfB5i3jKUXN5v6B9CT2BLnVn53mpBgmDmqsBYY/lJ8CTqor75vI4AK5WrNs+trs7O883NUgY1DANCHwZ856LT6/ROR3esVUDVqzsLA0kYF44XuvMZ2omYN4LKxc0+jXxFOa8UdPT6PSLYd7YlPnTXPXN0Qn9NnfeK2y/Uk2BOefUBEPM9fs/E2CbkkmFKnvIXMHivSZww/IblW03qhkxylzOZg1YUQCbnQAnW4LnOzcXO1kksWvmagOvFzl+mKsFSs3rsrmc8ExG+99Pzb2XmzzEcsyETqgZ4AY+bq6zg+aOtovQSSs2Df/7qY1+B/qgphVfVkXCu2J9MneZy7EdWqamMtzcccmCsQJaF8X2W/0tMcD6wQFUXfRawRlrXO1LxF9olcRiuLTv1Gwhu6CvajaDLn0OrhyraW/AWzEPlW0aaAZeYR9Z9SYOTIHmqNkCWEBPq9lbzIA61Owhg9Ro06ZF/AWFU7qa9oCmUwAAAABJRU5ErkJggg==>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAAAcElEQVR4XmNgGAU0BuVAvJsAhgMJIP5PAH+DKQ4B4joYBwiOQ2kxBohCvECDAaHID4g/IslhBU1AfB3KrgTiZ0hyGCCcAWJ6EJS/CsrHCQ4B8V8g5ofyFzEQ0ACSjETiO0PFcAIVdAEgcEAXGAU0AQAJOiDgV45DpQAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAAAyElEQVR4XmNgGCxADoglgZgFiIWB2AKI+VFUMDCIArEDEkYBi4D4Pxq2RVHBwDAfSQ6EmVGlGRjmQiVAGBtgZIDILQViEzQ5MPBkwG+AHhBLoAsiA5ACfAZsRBdAB4IMuA1wBOLT6ILYwEUGTAPKgHgmmhhOsJUB04AHQCyCJoYTTGPANCAZjY8XJDBADFBjgNj6CkWWCODEADEARIPSRReqNGGgygAxYDoDxO88KLJEgkdAvAddkBSwH4i10QVJAeHoAqNg2AMAyocpEA11rlYAAAAASUVORK5CYII=>