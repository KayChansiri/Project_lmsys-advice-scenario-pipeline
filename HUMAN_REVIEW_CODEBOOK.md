# Human Review Codebook

## Unit of analysis
One first-user message from LMSYS-Chat-1M.

## human_eligible

### yes
The message:
- describes a real or apparently real personal situation; and
- genuinely seeks advice, decision support, coping support, or help deciding what to do.

### no
Examples:
- factual-information request
- coding/technical task
- writing, rewriting, summarization, translation, or roleplay request
- general discussion without a personal situation
- diagnosis/treatment request
- insufficiently substantive request that is not genuinely advice-seeking

## human_primary_domain

Choose exactly one:

### relationship
Romantic relationships, family, friendship, roommates, trust, boundaries, interpersonal conflict, relationship continuation/breakup, or closely related interpersonal support.

### career
Employment, workplace issues, career direction, job search/change, professional development, salary, promotion, resignation, education-to-career decisions, or related work decisions.

### emotional_wellbeing
Everyday emotional distress/wellbeing, loneliness, grief, stress, burnout, anxiety, low mood, self-esteem, coping, support-seeking, motivation, or emotional boundaries, when the primary support need is not better classified as relationship or career.

## human_sufficient_context

### yes
There is enough contextual information to construct a meaningful scenario without inventing major facts.

### no
The message is too vague or sparse to reconstruct the situation responsibly.

## human_notes
Optional. Use for borderline cases or short adjudication notes.
