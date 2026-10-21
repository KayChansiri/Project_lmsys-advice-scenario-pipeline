# Codebooks

This document records the computational screening fields and human-review variables used to construct the study's scenario set. It contains no LMSYS conversation text.

## 1. Computational screening codebook

### `personal_situation`
**Type:** Boolean  
`true` when the first user message describes a real or apparently real personal situation involving the user, their relationships, work/career, or emotional wellbeing.

### `advice_or_decision_support`
**Type:** Boolean  
`true` when the user explicitly or implicitly seeks help with a personal decision, response, course of action, coping problem, or interpersonal problem. An explicit “What should I do?” question is not required. Pure factual/instrumental requests are coded `false`.

### `primary_domain`
**Allowed values:** `relationship`, `career`, `emotional_wellbeing`, `other`, `ambiguous`

- **relationship:** interpersonal relationships, including romantic relationships, dating, marriage, breakups, friendship, family relationships, trust, jealousy, boundaries, conflict, or communication.
- **career:** employment, workplace situations, career trajectory, job changes, supervisors/coworkers, workplace conflict, promotion, salary, and personalized education/training decisions tied to career trajectory.
- **emotional_wellbeing:** nonclinical emotional experiences such as stress, sadness, loneliness, grief, self-esteem, feeling overwhelmed, motivation, anxiety, coping, or support seeking.
- **other:** primary support need falls outside the three target domains.
- **ambiguous:** two or more target domains are genuinely co-primary and no primary domain can reasonably be identified.

### `sufficient_context`
**Type:** Boolean  
`true` when the message contains enough information about circumstances, concerns, constraints, preferences, uncertainty, or relevant background to support a context-sensitive response without fabricating major facts.

### `clinical_diagnosis_or_treatment_request`
**Type:** Boolean  
`true` when the primary request seeks medical/psychiatric diagnosis, symptom diagnosis, medication instructions, or treatment recommendations. Mentioning depression, anxiety, therapy, suicide, self-harm, or another mental-health topic is not by itself grounds for exclusion when the primary request concerns coping, decision-making, relationships, or help seeking.

### `eligible`
**Type:** Boolean  
Programmatically set to `true` only when `personal_situation == true`, `advice_or_decision_support == true`, `primary_domain` is one of the three target domains, `sufficient_context == true`, and `clinical_diagnosis_or_treatment_request == false`.

### `exclusion_reason`
Primary reason a message was not eligible. Historical screening used categories including `none`, `not_personal`, `not_advice_or_decision_support`, `outside_target_domains`, `insufficient_context`, `clinical_diagnosis_or_treatment`, `factual_or_technical_request`, `content_generation_or_roleplay`, and `other`.

## 2. Human review codebook

Human reviewers independently evaluated the computationally eligible messages. Reviewers were blinded to case-level computational eligibility, computational domain assignment, contextual-sufficiency judgment, and exclusion reason.

### `human_eligible`
**Coding:** `1 = eligible`, `0 = not eligible`

Eligible messages:
1. describe a real personal situation involving the user or someone directly connected to the user;
2. seek advice, decision support, interpretation, reflection, coping support, or guidance about what to think, feel, decide, or do;
3. primarily concern relationships, career, or emotional wellbeing; and
4. contain sufficient contextual information for context-sensitive support.

Exclude messages whose primary purpose is content generation/editing, factual or instrumental information retrieval, roleplay or response-constraining persona instructions, clinical diagnosis/treatment guidance, or another non-advice task.

### `human_sufficient_context`
**Coding:** `1 = sufficient`, `0 = insufficient`

Sufficient when the message provides enough information about circumstances, concerns, constraints, preferences, uncertainty, or relevant background for context-sensitive advice. Every potentially relevant fact need not be present.

### `human_primary_domain`
**Allowed values:** `relationship`, `career`, `emotional_wellbeing`, `unclear_or_other`

Domain is based on the primary problem for which support is sought, not merely people, settings, or topics mentioned.

- **relationship:** interpersonal relationship support, including romantic relationships, dating, marriage, breakups, friendship, family, trust, jealousy, boundaries, conflict, or communication.
- **career:** employment, workplace situation, career trajectory, or education/training as it relates to that trajectory.
- **emotional_wellbeing:** nonclinical emotional experiences such as stress, sadness, loneliness, grief, self-esteem, feeling overwhelmed, motivation, or anxiety.
- **unclear_or_other:** cannot be confidently assigned to a target domain or concerns another domain.

For multi-domain cases, code the primary object of the request. Workplace conflict is `career` when employment/workplace consequences are central, but may be `relationship` when the interpersonal relationship itself is central independent of work context.

### `human_notes`
Free-text field used primarily to record the reason for exclusion or adjudication notes. Common exclusion reasons: content generation/editing; factual/instrumental request; roleplay/response-constraining instruction; insufficient context; clinical/medical interpretation; outside target domains.

## 3. Boundary rules

**Safety-sensitive requests.** Eligibility is determined by the type of support requested, not whether the user's desired action is advisable or prosocial. Manipulation, coercive intentions, interpersonal conflict, or other safety-sensitive circumstances are not excluded solely on that basis; safety-related scenario needs are coded separately downstream.

**Sexual topics.** Relationship advice involving sexual topics is not automatically excluded. Requests primarily seeking explicit sexual instructions rather than personal advice/decision support are excluded as instructional requests.

**Content generation.** Personal context is insufficient when the primary request is to generate content. Asking how to approach a conversation may be eligible; asking the model to write the message is content generation and is excluded.

## 4. Adjudication

The two human reviewers independently coded overall eligibility, contextual sufficiency, and primary domain without access to one another's ratings. Inter-rater agreement is calculated before adjudication. Disagreements are resolved through discussion. Adjudicated human judgments determine the final sampling pool.
