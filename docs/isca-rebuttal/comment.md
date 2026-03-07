ISCA 2026 Paper #548 Reviews and Comments
===========================================================================
Paper #548 Scalable and Efficient Simulation of LLM Training


Review #548A
===========================================================================

Paper summary
-------------
The paper presents Moye, a high-fidelity simulator for large-scale LLM training that enables accurate performance estimation without requiring full-cluster deployments. Moye uses ex-situ tracing on a single GPU to reconstruct per-device execution graphs, combines a white-box model for collective communication with lightweight profiling, and applies an ML-based predictor to capture slowdown from overlapping computation and communication. Across dense and MoE workloads, Moye achieves around 91 percent accuracy in training step time with up to three times lower error than prior simulators, while running orders of magnitude faster, enabling both efficient enumeration of training strategies and realistic extrapolation to larger clusters.

Reviewer expertise
------------------
2. Some familiarity - my research focus is not/has not been in this area,
   but I have read some papers, etc. on this topic

Review confidence and paper clarity
-----------------------------------
3. I understood the paper but missed some details

Experimental methodology
------------------------
5. Excellent

Novelty
-------
3. Significant, encouraging others to build on results

Related work
------------
4. Comprehensive coverage, well done!

Are there violations of ISCA formatting requirements?
-----------------------------------------------------
2. Yes

Questions to prioritize for rebuttal/revision
---------------------------------------------
1. The paper states: “The key technical idea here is to transform the parallel initialization process at each rank into a sequential one, allowing a single device to act as each rank iteratively to trace the corresponding execution graph and GPU memory footprints.”
Would this sequential emulation of ranks introduce out-of-memory risks on a single GPU? Are there minimum GPU memory requirements for this approach to work reliably, and how do these requirements scale with model size and parallelism configuration?

2. In Figure 2, the NCCL-predictor appears to improve consistently with increasing tensor sizes on A100 GPUs, but this trend is not as evident on H100 GPUs. Could the authors clarify the reason for this discrepancy? Is it possible that the NCCL-predictor has been more extensively tuned for A100-class hardware and is not yet fully adapted to H100? As a follow-up, is the reduced error gap on A100s attributable to the transition from latency-dominated to bandwidth-saturated regimes at larger tensor sizes, where analytical models tend to be more accurate?

3. Regarding extrapolation to future architectures, it is not entirely clear how Moye would be applied to a system with a fundamentally different interconnect topology or bandwidth characteristics. Since the white-box communication model relies on profiling existing hardware and libraries, does Moye primarily target architectures for which offline profiling is already possible? Alternatively, can it provide reasonable estimates for architectures that have not yet been taped out and for which no empirical profiling data exists?

4. Finally, I would like clarification on the extensibility of Moye with respect to novel parallelization or sharding strategies. How difficult is it to add support beyond the strategies currently supported? For example, if a new sharding strategy is implemented in Megatron-LM, can it be integrated into Moye in a plug-and-play manner, or does it require reimplementing substantial logic within the simulator?

Additional comments for authors
-------------------------------
The problem addressed in this paper is well motivated. Accurate simulation of large-scale training systems is critically important for forecasting the performance of future architectures, as it can significantly reduce the cost and risk associated with infrastructure planning and hardware design decisions, while enabling earlier convergence on promising design points. I also appreciate that the evaluation explicitly reports simulator runtime in addition to accuracy metrics, as simulation overhead is an important practical consideration that is often overlooked.

One concern is related to the double-blind review policy. The authors explicitly disclose a commit ID in a public Megatron-LM repository (commit 53a350ed), which makes it relatively straightforward to infer the identity of the authors. This appears to violate the double-blind requirements and should be addressed.

Significance of the work
------------------------
3. Significant new result; or incremental contribution that has the
   potential to be the "final word" on a topic.

Fit for ISCA
------------
3. In scope - tackling key problems in and advancing computer
   architectures.

Overall merit
-------------
5. Accept- The paper may have some flaws but I believe it has the potential
   to be a valuable contribution and the concerns can be addressed through
   reasonable effort by the authors, so I will argue in support of it

Would a revision potentially increase your overall merit score?
---------------------------------------------------------------
2. Yes



Review #548B
===========================================================================

Paper summary
-------------
This paper proposes Moye, a simulator for large-scale LLM training that addresses three key challenges: (1) obtaining training workloads without full-scale cluster deployment through ex-situ tracing, (2) efficiently estimating collective communication performance using white-box modeling with profiled parameters, and (3) accounting for computation slowdown from overlapping communication and computation kernels. Moye demonstrates 8% average error for GPT-175B training on 96 H800 GPUs with under 2-minute simulation time, achieving approximately 3× better accuracy than state-of-the-art simulators.

Reviewer expertise
------------------
3. Knowledgeable - I published/worked in the area but might miss the most
   recent related work

Review confidence and paper clarity
-----------------------------------
3. I understood the paper but missed some details

Experimental methodology
------------------------
4. Good

Novelty
-------
2. Low conceptual novelty [provide references in your comments]

Related work
------------
4. Comprehensive coverage, well done!

Strengths
---------
- This work tackles the critical problem regarding efficient simulation in large-scale LLM training, with clear use cases for both exploring parallelization strategies and planning future deployments.

- The proposed system handles three technical challenges (workload tracing, communication modeling, overlap slowdown), which are inadequately addressed in prior works, and demonstrates highly accurate and fast simulation results across various configurations.

Are there violations of ISCA formatting requirements?
-----------------------------------------------------
1. No

Questions to prioritize for rebuttal/revision
---------------------------------------------
- While this work positions itself relative to SimAI conceptually and mentions SimAI's 2-hour overhead for 128-GPU simulation, there is no direct accuracy comparison at scales where both systems can run (e.g., 8-64 GPUs where SimAI has published results).  Could you add a head-to-head comparison with SimAI showing both accuracy and runtime at 8, 16, 32, and 64 GPU scales? 

- While this paper evaluates on GPT models (13B-485B) and Mixtral (8×1.75B, 16×1.75B), these are now somewhat dated, and all MoE evaluation is limited to the Mixtral architecture. Could you add evaluation results on more recent, larger, and diverse models that would represent truly unseen cases for the simulator? For example, Qwen3-235B,  DeepSeek-V3/R1 could be good candidates.

Additional comments for authors
-------------------------------
I think this work is a well-executed paper addressing an important problem with clear practical value. The ex-situ tracing technique is reasonable and the overall approach achieves a compelling balance between accuracy and efficiency. The systematic comparison with related work and clear articulation of gaps in existing solutions are particular strengths.

However, I feel like the evaluation results could be strengthened significantly by: (1) directly comparing with SimAI at overlapping scales to validate the accuracy-efficiency tradeoff claim, and (2) demonstrating generalization to more recent, diverse, and truly unseen model architectures, which are state-of-the-art MoE models like DeepSeek-V3 that this work itself cites as motivation. These additions would substantially increase confidence in the simulator's robustness and practical utility for production systems dealing with rapidly evolving model architectures.

Significance of the work
------------------------
2. Important but extensively studied. Incremental contribution.

Fit for ISCA
------------
3. In scope - tackling key problems in and advancing computer
   architectures.

Fit for ISCA - explanation
--------------------------
Large-scale distributed training simulation

Overall merit
-------------
5. Accept- The paper may have some flaws but I believe it has the potential
   to be a valuable contribution and the concerns can be addressed through
   reasonable effort by the authors, so I will argue in support of it

Would a revision potentially increase your overall merit score?
---------------------------------------------------------------
2. Yes



Review #548C
===========================================================================

Paper summary
-------------
This paper presents Moye, a simulator for LLM training on GPU clusters. The motivation is that prior approaches either require per-device in-situ tracing at full scale or become extremely slow when simulating communication at fine granularity.

Moye’s key design is converting multi-rank initialization into a sequential workflow where a single device iteratively emulates each rank to extract execution graphs without requiring the full cluster. It models collectives using a white-box NCCL-inspired chunk model calibrated via offline profiling and accounts for compute-communication interference via an XGBoost-based slowdown predictor. The implementation supports common training stacks.

Experimentally, it reports 91.4% average step-time accuracy, and 92% accuracy for 96-GPU GPT-175B in <2 min of simulation time.

Reviewer expertise
------------------
4. Expert - This is a core area for me, I am up-to-date with related work,
   and I have published in this area and/or worked on it in industry

Review confidence and paper clarity
-----------------------------------
5. I understood quite well, and the paper was well written

Experimental methodology
------------------------
4. Good

Novelty
-------
2. Low conceptual novelty [provide references in your comments]

Related work
------------
3. Fairly comprehensive; I would add a bit more [elaborate below]

Strengths
---------
- Important and timely problem.

- ex-situ tracing avoids needing full-scale clusters for extracting workloads.

- white-box comms + learned overlap slowdown appears usable with mainstream frameworks.

Weaknesses
----------
- Validation clarity and scope; it is not always obvious which results are hardware-measured vs model-predicted, and where residual error comes from.

- What is the amortization/setup cost? The approach relies on substantial profiling and training a learned slowdown model.

- It’s unclear how well the approach captures hierarchical and topology-specific parallelism choices and irregular MoE routing/all-to-all patterns.

- Profiling and overlap-model training may be expensive to maintain. Moye depends on building an overlap dataset and multiple profilers/collection tools. Since the paper notes slowdowns vary significantly across GPUs due to proprietary implementations, ow often must the slowdown predictor be retrained?

- MoE training frequently involves routing imbalance and heavy all-to-all-like communication; it would help to show that Moye explicitly models these communication patterns and captures their overlap behavior, rather than relying on approximations that may work mainly for dense models.

Are there violations of ISCA formatting requirements?
-----------------------------------------------------
1. No

Questions to prioritize for rebuttal/revision
---------------------------------------------
- Where does the remaining ~9% modeling error come from?

- Please add a concise table mapping major results to their methodology (e.g., hardware-measured vs predicted vs simulator output) and list what calibration data each component uses.

- What is the end-to-end cost to port Moye to a new cluster (new GPUs, topologies, etc.)?

-  How does Moye account for hierarchical bandwidth tiers and topology-aware behavior (e.g., different intra-node vs inter-node collectives behavior)?

- For MoEs, what communication patterns are explicitly modeled and can you provide an error breakdown or discussion about what dominates the remaining error for MoE workloads?

Additional comments for authors
-------------------------------
- GPU devices executes -> GPU devices execute

- These factors results -> These factors result

- Please position the novelty clearly.

Significance of the work
------------------------
4. Important and emerging. Big opportunity for significant gains and/or the
   potential to become a seminal paper.

Fit for ISCA
------------
3. In scope - tackling key problems in and advancing computer
   architectures.

Fit for ISCA - explanation
--------------------------
LLM training, performance model, communication collectives, GPUs

Overall merit
-------------
3. Weak reject - I have significant concerns, but I am open-minded to what
   revision can achieve (possibly with shepherding) and remain open-minded
   about the outcome, and I am open to revising my score upwards based on
   the revision

Would a revision potentially increase your overall merit score?
---------------------------------------------------------------
2. Yes



Review #548D
===========================================================================

Paper summary
-------------
This paper presents Moye, a simulation framework for large-scale LLM training. Moye employs an ex-situ tracing approach and captures the per-rank execution graph using a single GPU device. To obtain the end-to-end training step time, Moye's timeline composer constructs a global timeline using the per-rank workloads (traces). Collective communication time is estimated using an analytical model, while Moye also models slowdowns caused by resource contention between overlapping kernels using an XGBoost-baesd predictor.

Reviewer expertise
------------------
2. Some familiarity - my research focus is not/has not been in this area,
   but I have read some papers, etc. on this topic

Review confidence and paper clarity
-----------------------------------
5. I understood quite well, and the paper was well written

Experimental methodology
------------------------
4. Good

Novelty
-------
2. Low conceptual novelty [provide references in your comments]

Related work
------------
1. I am not up-to-date with the current literature

Strengths
---------
Moye can be a useful tool for estimating LLM training performance across different configurations.

Weaknesses
----------
Technical contributions are incremental.

It does not support some popular parallelism strategies such as sequence parallelism yet.

Are there violations of ISCA formatting requirements?
-----------------------------------------------------
1. No

Questions to prioritize for rebuttal/revision
---------------------------------------------
- Could you also compare with vTrain (MICRO'24)?
- It might be helpful to show diverse uses beyond the 3D parallelism

Additional comments for authors
-------------------------------
The paper is well-written and easy to read. I think Moye can be useful for certain use cases as shown in Section IX. However, this work appears to be primarily engineering-oriented, with limited technical novelty in its underlying techniques. 

While tracing workloads at each rank using a single device is nice, it is a commonly used strategy in simulation frameworks. Similarly, the ML-based overlapping slowdown prediction is a nice addition, but the use of XGBoost for performance prediction is a widely adopted approach. 

Table I provides a nice comparison of existing simulators for LLM training. Still, there exist other works that have similar goals, such as vTrain (MICRO'24). A quantitative comparison with all recent simulators would better demonstrate Moye's advantages.

I like that Section IX discusses the use case of Moye. Presenting additional
diverse use cases would further strengthen the demonstration of Moye's practical value across different scenarios.

Significance of the work
------------------------
2. Important but extensively studied. Incremental contribution.

Fit for ISCA
------------
3. In scope - tackling key problems in and advancing computer
   architectures.

Fit for ISCA - explanation
--------------------------
A simulation framework for LLM training

Overall merit
-------------
3. Weak reject - I have significant concerns, but I am open-minded to what
   revision can achieve (possibly with shepherding) and remain open-minded
   about the outcome, and I am open to revising my score upwards based on
   the revision

Would a revision potentially increase your overall merit score?
---------------------------------------------------------------
2. Yes



Review #548E
===========================================================================

Paper summary
-------------
This paper studies the GPU cluster performance prediction problem for LLM training. Its contributions include a single-node pre-training tracing approach and methods to scale this tracing from a single node to cluster scale. Evaluations on several models show promising prediction accuracy, outperforming baseline systems that are applicable to these settings.

Reviewer expertise
------------------
3. Knowledgeable - I published/worked in the area but might miss the most
   recent related work

Review confidence and paper clarity
-----------------------------------
4. I understood quite well, but the writing needs polishing

Experimental methodology
------------------------
4. Good

Novelty
-------
2. Low conceptual novelty [provide references in your comments]

Related work
------------
4. Comprehensive coverage, well done!

Strengths
---------
- An important problem to study.
- Considering practical concerns, suitable for large-scale adoption.
- Experiments on GPU clusters are solid, and the writing is clear.

Weaknesses
----------
- The models considered are outdated (about two years behind what people are currently training).
- Some important SOTA baselines are not compared (even though reasoning is provided), but a small-scale comparison and ablation study would still be essential.
- Evaluation could be significantly improved (details in my comments).
- Sequentially running tracing on a single node is a good idea, but could this sequential running eventually become a bottleneck—especially as parallelism schemes become highly complex?

Are there violations of ISCA formatting requirements?
-----------------------------------------------------
1. No

Questions to prioritize for rebuttal/revision
---------------------------------------------
See the weakness section.

Additional comments for authors
-------------------------------
I appreciate the following aspects of this paper: Simulation is positioned for both “enumeration and extrapolation” for large-scale training, with clear motivation around massive GPU clusters, complex parallelism strategies, and overlapping communication and computation.  ￼

Also, it considers practical concerns, suitable for large-scale adoption. The design choices directly target “ex-situ simulation,” “high efficiency,” “generality and ease of use,” and “high accuracy,” and the system supports PyTorch, DeepSpeed, and Megatron-LM, plus dense and MoE workloads and GPU memory–aware ex-situ simulation.  ￼

Finally, experiments on GPU clusters are solid, and the writing is clear. The evaluation uses H800 and A800 clusters, compares with Proteus, FlexFlow, and Calculon, reports end-to-end error numbers (e.g., GPT-70B, GPT-175B) and simulation time cost (up to 8,192 GPUs), and breaks down computation, communication, memory usage, and slowdown prediction with figures and tables that are easy to follow.  ￼

The paper, however, still shows some large limitations: The models considered are outdated (about two years behind what people are currently training). The evaluation includes VGG19, GPT2, GPT models (13B to 485B), and Mixtral, and the paper also uses Llama3-8B / Mixtral-8×7B for overlap statistics. If the goal is “what people are currently training,” the set may look centered on GPT and Mixtral rather than newer “parallelism strategies” and “model innovations” the paper highlights.  ￼

A minor limitation could be that: sequentially running tracing on a single node is a good idea, but could this sequential running eventually become a bottleneck—especially as parallelism schemes become highly complex? The core idea is to “transform the parallel initialization process … into a sequential one” so a single device can “act as each rank iteratively” and output per-rank graphs as JSON. The paper separates “one-time, reusable workload tracing cost” from simulation time, but for large N and more complex parallelism schemes (and new dependencies / matching rules), the sequential tracing stage and the need to preserve more dependency logic could become a practical pressure point.  ￼

Finally, there are some minor weaknesses:
- The model mirrors NCCL internals and requires NCCL modification (e.g., chunk size extraction) plus extensive profiling. Portability across NCCL versions or non-NCCL stacks is unclear and may incur high maintenance cost.
- Excluding Astra-Sim and SimAI weakens positioning against established LLM training simulators. Even limited-scope comparisons (comm-only or small-scale packet-level spot checks) would better validate the claimed fidelity–speed tradeoff.
- Lacks a clean end-to-end ablation separating gains from (i) ex-situ tracing, (ii) CC model vs α–β/NCCL-Predictor, and (iii) slowdown model vs no-slowdown/heuristics.
- Claims MoE support, but limited exploration of routing skew/capacity factor extremes and their effect on step-time tails and straggler behavior.

Significance of the work
------------------------
2. Important but extensively studied. Incremental contribution.

Fit for ISCA
------------
3. In scope - tackling key problems in and advancing computer
   architectures.

Overall merit
-------------
3. Weak reject - I have significant concerns, but I am open-minded to what
   revision can achieve (possibly with shepherding) and remain open-minded
   about the outcome, and I am open to revising my score upwards based on
   the revision

Would a revision potentially increase your overall merit score?
---------------------------------------------------------------
2. Yes



Review #548F
===========================================================================

Paper summary
-------------
This paper proposes Moye, a high-fidelity simulator for large-scale distributed ML training that aims to be accurate without the heavy overheads of full discrete-event simulation. It does this by (1) ex-situ workload tracing: capturing per-device training execution graphs using a single device and then extrapolating to large clusters, (2) lightweight but accurate collective-communication estimation (avoiding expensive network discrete-event simulation), and (3) modeling slowdown from interference when communication and computation overlap on the same GPU. It supports both dense and MoE training workloads and adds GPU-memory–aware ex-situ simulation to better reflect real training behavior under memory constraints.

Reviewer expertise
------------------
4. Expert - This is a core area for me, I am up-to-date with related work,
   and I have published in this area and/or worked on it in industry

Review confidence and paper clarity
-----------------------------------
4. I understood quite well, but the writing needs polishing

Experimental methodology
------------------------
3. Average

Novelty
-------
2. Low conceptual novelty [provide references in your comments]

Related work
------------
4. Comprehensive coverage, well done!

Strengths
---------
This paper has the following strengths:
1. It utilizes an analytic way for training simulation, providing ultra-fast speed.
2. It covers test on differen sizes of models, demonstrating the feasibility.

Weaknesses
----------
I have the following concerns on method and experiments.

1. As an approach relying on analytic modeling, the elaboration on how to model the complex training process is essential. However, the corresponding presentation lacks sufficient details. I would expect more discussion on both computation and communicaiton (intra-node and inter-node) part, using figures/ schemes/equations (since you are buliding analytic model).

2. LLM training is complicated with various possible settings: optimizer, parallelism (included here), quantization, inter-layer scheulding within forward/backward etc. From the current writting I do not see the details of handling those factors.

3. Another missing part is on the comparison with the prior efforts. As Table I shows, there exist a large body of training simulators. The experimental comparison with at least part of them should be provided, revealing the tradeoff between simulation speed and accuracy.

Are there violations of ISCA formatting requirements?
-----------------------------------------------------
1. No

Questions to prioritize for rebuttal/revision
---------------------------------------------
1. Please provide more details on analytic modeling part, including computation. For communication, specific details dealing with scale-up and scale-out should be presented. (You have 8192 GPUs so scale-out is a must)

2. Please provide details (including abalation study) on various training related settings, especially when you have different scheluding chocies and optimization choice (e.g., recompuataion)

3. Experimental comparison with prior simulator is expected, including analysis.

Additional comments for authors
-------------------------------
Does your simulator can also work on transformer-based diffusion model or other genAI models? Your experiments show the suppor for CNN.

Significance of the work
------------------------
2. Important but extensively studied. Incremental contribution.

Fit for ISCA
------------
3. In scope - tackling key problems in and advancing computer
   architectures.

Fit for ISCA - explanation
--------------------------
LLM training simulation is very important for ensuring principle study of AI hardware. This paper fits for ISCA.

Overall merit
-------------
3. Weak reject - I have significant concerns, but I am open-minded to what
   revision can achieve (possibly with shepherding) and remain open-minded
   about the outcome, and I am open to revising my score upwards based on
   the revision

Would a revision potentially increase your overall merit score?
---------------------------------------------------------------
2. Yes