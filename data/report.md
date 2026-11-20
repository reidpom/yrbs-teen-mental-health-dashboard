---
title: "MIS 545 Group Project Proposal: Teen Mental Health in the YRBS"
date: "November 13, 2026"
---

# The idea in short

We use a national survey of U.S. high school students to find out which everyday behaviors go along with poor mental health, and how much social media use matters compared with things like sleep and bullying. We build a model on the 2023 survey and check whether it still works on the 2025 survey.

**The three questions we answer**

1. Can a student's answers about daily life (sleep, bullying, activity, substance use, social media) predict whether they report feeling persistently sad or hopeless?
2. How much of the link between social media and sadness is left as we account for sleep, bullying and everything else, and is it stronger for some students (girls or boys, younger or older)?
3. Are there recognizable "types" of students, and are some types at much higher risk?

**Why this fits the assignment:** the syllabus asks for a multi-layered analysis using methods from class. This project uses prediction models, feature selection, dimensionality reduction and clustering on one dataset, and the topic is easy to explain to a non-technical audience.

**One rule for how we talk about results:** each student answers the survey once and is not followed over time, so we can say a behavior is *linked to* or *predicts* poor mental health. We cannot say it *causes* it.

# The data

The data is the CDC's Youth Risk Behavior Survey (YRBS), an anonymous questionnaire given to U.S. high school students every two years. It is public and free to use. [CDC data page](https://www.cdc.gov/yrbs/data/index.html) · [2023 questionnaire](https://www.cdc.gov/yrbs/media/pdf/2023/2023_YRBS_Standard_HS_Questionnaire.pdf)

|                                    | 2023 survey     | 2025 survey    |
|:-----------------------------------|:----------------|:---------------|
| Students                           | 20,103          | 10,710         |
| Questions                          | 107             | 105            |
| Our use                            | Build the model | Test the model |
| Students with both key answers     | 15,075          | 9,686          |
| Felt sad or hopeless (among those) | 41.8%           | 34.8%          |

**What the data looks like**

- Each row is one student. Each column is one survey question.
- Almost every answer is a category, such as yes/no or "5 hours, 6 hours, 7 hours." Very few are true numbers. This shapes which methods we can use.
- 92 questions are worded the same way with the same answer choices in both years. We use only those, so the two years can be compared fairly. Both years are already downloaded and combined into one file.

**What we predict (the outcome)**

"During the past 12 months, did you ever feel so sad or hopeless almost every day for two weeks or more in a row that you stopped doing some usual activities?" (yes/no). About 4 in 10 students said yes in 2023.

**What we predict it with**

- Social media: "How often do you use social media?" (8 choices, from never to more than once an hour)
- Sleep, bullying at school and online, physical activity, sports, diet
- Vaping, alcohol and marijuana use
- Safety and violence, unfair treatment at school, housing stability
- Age, sex, grade, race and ethnicity

The other mental health questions (suicide questions, current mental health, trouble concentrating) are left out of the predictors. Including them would be like giving the model the answer key.

**A first look (2023, before any modeling)**

| Social media use       | Students | Felt sad or hopeless |
|:-----------------------|:---------|:---------------------|
| Does not use           | 1,060    | 31.7%                |
| About once a day       | 901      | 31.7%                |
| Several times a day    | 5,858    | 40.5%                |
| About once an hour     | 1,171    | 41.2%                |
| More than once an hour | 4,769    | 49.3%                |

Heavier users report more sadness. The project tests whether that link holds up once sleep, bullying and the rest are considered. Note that 78% of students use social media several times a day or more, so there are few light users to compare against.

# How the study is set up

We build the model on 2023 students and test it on 2025 students it has never seen. This is a tougher and more honest test than the usual approach of randomly splitting one dataset, because it shows whether the patterns hold for a new group of students two years later.

1. **Learn from 2023.** All model building, tuning and feature selection uses 2023 only.
2. **Lock the model.** No changes after this point.
3. **Test on 2025.** We run the locked model once on 2025 and report those results as our headline numbers.

**Known data issues and how we handle them**

+-----------------------+--------------------------------------------+---------------------------------------------+
| Issue                 | What it means                              | What we do                                  |
+:======================+:===========================================+:============================================+
| Skipped questions     | About 1 in 4 skipped the social media      | Keep only students who answered both the    |
|                       | question in 2023 (1 in 12 in 2025). Most   | outcome and social media, so results        |
|                       | stopped partway, skipping about 42         | describe students who answered most of the  |
|                       | questions, and were slightly less likely   | survey. For other questions, treat "no      |
|                       | to feel sad (38% against 42%)              | answer" as its own category                 |
+-----------------------+--------------------------------------------+---------------------------------------------+
| Mostly heavy users    | 78% use social media several times a day   | Expect a small effect, and frame the        |
|                       | or more                                    | project as social media *and other factors* |
+-----------------------+--------------------------------------------+---------------------------------------------+
| The rate changed      | Sadness fell from 41.8% to 34.8% between   | Report a score that is not thrown off by    |
|                       | the two surveys                            | this (AUROC, explained below) and adjust    |
|                       |                                            | the yes/no cutoff                           |
+-----------------------+--------------------------------------------+---------------------------------------------+
| Self-reported answers | Students may under-report or over-report   | State it as a limitation                    |
+-----------------------+--------------------------------------------+---------------------------------------------+
| Lost questions        | Using only the 92 shared questions drops   | Accept it. A fair two-year comparison is    |
|                       | grades and most childhood-adversity items  | worth more                                  |
+-----------------------+--------------------------------------------+---------------------------------------------+

# Methods

The work has three parts: find patterns among students, predict who is at risk, and work out which behaviors matter most.

## Part 1: Find patterns (no outcome involved yet)

+-----------------------+--------------------------------------------+---------------------------------------------+
| Method                | In plain words                             | What we get from it                         |
+:======================+:===========================================+:============================================+
| MCA (Multiple         | Boils 80+ survey questions down to a       | A short list of themes that describe how    |
| Correspondence        | handful of themes, such as "substance use" | students differ                             |
| Analysis)             | or "healthy routine." It is the version of |                                             |
|                       | PCA (from Week 6) built for category       |                                             |
|                       | answers                                    |                                             |
+-----------------------+--------------------------------------------+---------------------------------------------+
| Clustering (k-means   | Sorts students into groups whose answers   | Student "types," and the share of each type |
| and hierarchical,     | look alike                                 | who feel sad or hopeless                    |
| Week 10)              |                                            |                                             |
+-----------------------+--------------------------------------------+---------------------------------------------+
| UMAP                  | Draws all students as dots on a 2D map,    | One picture for the slides that shows the   |
|                       | placing similar students near each other.  | groups                                      |
|                       | We run it on the MCA themes                |                                             |
+-----------------------+--------------------------------------------+---------------------------------------------+

MCA and UMAP work as a pair. MCA does the summarizing and stays explainable. UMAP turns that summary into a picture. The UMAP map is for illustration only: its layout shifts with settings, so we do not draw conclusions from distances on it.

## Part 2: Predict who is at risk

We try several models, from simple to complex, and keep the simplest one that performs well.

| Model               | In plain words                                              | From class              |
|:--------------------|:------------------------------------------------------------|:------------------------|
| Logistic regression | Adds up weighted evidence for and against. Our baseline     | Week 3                  |
| Decision tree       | A flowchart of yes/no questions                             | Week 2                  |
| Random forest       | Many decision trees that vote                               | Builds on Weeks 2 and 8 |
| Voting ensemble     | Several different models that vote together                 | Week 8                  |
| Neural network      | A flexible model that can pick up complicated patterns      | Week 11                 |

Settings for each model are chosen with cross-validation: the 2023 data is split into 5 parts, and the model is repeatedly trained on 4 and checked on the 5th (Week 5).

## Part 3: Work out what matters most

+--------------------------------+------------------------------------------------------------------------------+
| Method                         | In plain words                                                               |
+:===============================+:=============================================================================+
| RFE (Recursive Feature         | Removes the least useful question, retrains, and repeats until only the      |
| Elimination, Lab 6)            | questions that matter are left                                               |
+--------------------------------+------------------------------------------------------------------------------+
| LASSO (Weeks 4 and 6)          | A model that is penalized for using too many questions, so it shrinks the    |
|                                | unhelpful ones to zero                                                       |
+--------------------------------+------------------------------------------------------------------------------+
| Social media in stages         | Add predictors in four steps (background, then social media, then sleep and  |
|                                | bullying, then everything else) and watch how the social media link changes  |
|                                | at each step. Heavy use may work through less sleep and more online          |
|                                | bullying, so adding everything at once would hide it                         |
+--------------------------------+------------------------------------------------------------------------------+

RFE and LASSO are two routes to the same answer. Questions that both keep are the ones we trust.

**Subgroup checks:** we repeat the staged test separately for girls, boys and each grade. An effect that exists only in one group would be averaged away in the overall number.

**A note on cross-validation:** students in the survey come in clusters (same schools and areas), and the file is sorted so students who skipped many questions sit together. We therefore keep each sampling area together in one fold. A plain unshuffled split gives misleading scores on this data.

## How we score the models

- **AUROC:** if we pick one sad student and one non-sad student at random, how often does the model rank the sad one as higher risk? 0.5 is a coin flip and 1.0 is perfect.
- **Recall:** of the students who really are struggling, what share did the model catch? This matters most, because missing an at-risk student is worse than a false alarm.
- **Precision:** of the students the model flagged, what share really are struggling?

**Fairness check (Week 14 topic):** we compare recall for boys and girls and across racial and ethnic groups, to see whether the model misses some groups more than others.

# Early results

A first pass with one basic model (logistic regression, not yet tuned) shows the plan works and gives a preview of the answer: heavy social media use is linked to more sadness, but sleep, bullying and other behaviors predict it far better.

**Social media in stages** (trained on 2023, tested on 2025)

+------------------------------+----------+-------------+---------------------------+
| What the model knows         | Score on | Score on    | Odds of sadness,          |
|                              | 2023     | 2025 (test) | more-than-hourly users vs |
|                              |          |             | non-users                 |
+:=============================+:=========+:============+:==========================+
| 1\. Background only (age,    | 0.653    | 0.640       |                           |
| sex, grade, race)            |          |             |                           |
+------------------------------+----------+-------------+---------------------------+
| 2\. Add social media         | 0.662    | 0.650       | 1.74 times (1.36 to 2.29) |
+------------------------------+----------+-------------+---------------------------+
| 3\. Add sleep and bullying   | 0.750    | 0.724       | 1.68 times (1.37 to 2.11) |
+------------------------------+----------+-------------+---------------------------+
| 4\. Add everything else      | 0.820    | 0.806       | 1.33 times (1.14 to 1.62) |
+------------------------------+----------+-------------+---------------------------+

Scores are AUROC: 0.5 is a coin flip, 1.0 is perfect. The 2023 score uses cross-validation. Ranges in brackets are 95% margins of error: a range that stays above 1.0 means the link is unlikely to be chance.

**How to read this**

- Social media barely improves prediction (0.640 to 0.650). Sleep and bullying improve it a lot (to 0.724).
- The social media link shrinks as other behaviors are added, from 1.74 times the odds to 1.33 times. It gets smaller but does not vanish in this analysis: the full-model range (1.14 to 1.62) stays above 1.0. The robustness checks further down show that this last step is borderline, so the full-model link should be described as small and uncertain.
- The odds compare students on social media more than once an hour (32% of students) with non-users. Students who use it about once an hour show a smaller gap: 1.35 times the odds, and in 2025 they were no sadder than students who use it several times a day. The link is concentrated in the very heaviest group.
- Removing social media from the full model changes nothing (0.806 either way). Whatever it tells us is already captured by the other answers.
- The model holds up on new students: 0.820 on 2023 and 0.806 on 2025.

**Girls and boys**

|                                             | Girls      | Boys       |
|:--------------------------------------------|:-----------|:-----------|
| Felt sad or hopeless (2025)                 | 45.8%      | 24.0%      |
| Sad, daily-or-less users (2023)             | 46.4%      | 24.9%      |
| Sad, more-than-hourly users (2023)          | 61.0%      | 34.7%      |
| Odds for more-than-hourly users, background only | 1.77 times (1.26 to 2.42) | 1.70 times (1.28 to 2.33) |
| Odds for more-than-hourly users, full model      | 1.23 times (0.91 to 1.64) | 1.40 times (1.19 to 1.71) |
| Full model score on 2025                    | 0.787      | 0.776      |

Girls report sadness about twice as often as boys, but the social media link is about the same size for both. The two ranges overlap almost completely, and a direct comparison of girls against boys finds no reliable difference. It is not concentrated in one sex.

**By grade:** the link looks largest in 10th grade (2.31 times the odds with background only, range 1.82 to 3.28) and sits near 1.5 to 1.6 in the other grades. But the grade ranges are wide and overlap, and direct comparisons of 10th grade against each other grade are not reliably different. We treat it as a lead, not a finding.

## Student profiles (MCA, clustering and UMAP)

Part 1 of the methods has now been run on the 2023 students. Mental health answers and background (age, sex, grade) were left out, so the profiles are built only from behaviors and experiences.

**Step 1: themes (MCA).** MCA boiled 81 questions down to themes. The strongest ones are easy to name: substance use together with sexual activity, sexual activity on its own, healthy eating and exercise, and bullying and violence.

**Step 2: profiles (clustering).** Grouping students on the first 10 themes gives five profiles:

| Profile | Share of students | Sad or hopeless, 2023 | Sad or hopeless, 2025 |
|:-----------------------------------------------|:------|:------|:------|
| 1\. Low risk: not sexually active, almost no substance use | 51% | 27% | 24% |
| 2\. Sexually active, little substance use | 16% | 39% | 34% |
| 3\. Vaping and drinking, not sexually active, more bullying | 17% | 63% | 57% |
| 4\. Sexually active and substance use | 12% | 68% | 56% |
| 5\. Heavy multi-risk: daily vaping, heavy marijuana use, exposure to violence | 5% | 73% | 62% |


**How to read this**

- The profiles never saw the mental health answers, yet sadness ranges from 27% in profile 1 to 73% in profile 5.
- The profiles hold up on new students. The 2025 students were placed into the 2023 profiles, and the same pattern appears: low sadness in profiles 1 and 2, high in 3 to 5.
- The themes lose almost no information. A model given only 20 themes scores 0.801 on 2025, against 0.806 for a model given all the original answers. Knowing only a student's profile scores 0.656.
- Heavy social media use is more common in the higher-risk profiles: 31% of profile 1 use it hourly or more, against 43% to 54% in the others.
- A lead to follow up: inside the low-risk profile, hourly-plus users are sadder than students who use it daily or less (31% against 23% in 2023, 27% against 21% in 2025). Inside the other profiles the gap is small. This has no margins of error yet.
- **A second clustering method agrees.** Hierarchical clustering (Ward's method, the other Week 10 technique) was run on the same 10 themes and cut at five groups.
  - Agreement with k-means: 0.66 on the adjusted Rand scale in 2023 and 0.78 in 2025 (0 = chance, 1 = identical).
  - Sadness ranges from 26% to 70% across its groups, close to k-means' 27% to 73%.
  - Both methods find the same large low-risk group and the same small heavy-risk group. They disagree on the middle: hierarchical clustering merges most of profiles 2 and 4 and sets aside a small extra group of 3%.
  - Hierarchical clustering's tree shows no strong natural break at five groups.
  
  Together these back up the caution below: the broad gradient is real, but the exact borders between middle profiles depend on the method.

**Step 3: the map (UMAP).** Each dot is a student, and students with similar answers sit close together. The 2025 students land in the same regions as the 2023 students.


The same map for 2023 and 2025 side by side:


**Cautions**

- The profiles are regions along a continuum, not sharply separate groups: neighboring profiles blend into each other. Five was chosen because it stays the same when the data is resampled and because two clustering methods (k-means and hierarchical) broadly agree on it. With six or more groups the result becomes unstable.
- The gap between sexually active and other students is partly exaggerated, because eight survey questions repeat that same fact.
- Two cleaning choices mattered. Skipped questions are treated as an average answer, and answers given by fewer than 2% of students are merged with the nearest answer. Without these, the themes mostly picked up who skipped questions and a handful of extreme answers.
- The map is for illustration only. Conclusions come from the table, not from distances on the map. We redrew it with three different settings and the structure stayed the same: each time, about 88% of a student's nearest dots belong to the same profile.

**A second way to group students, tested and set aside.** We also tried grouping students directly on the UMAP layout and then naming each group from its answers. The result was perfectly repeatable, but less useful:

- The groups were defined by a few yes-or-no facts: whether a student has had sex, dates, drives, and whether they skipped the last block of questions. Those are facts about age and life stage, not patterns of risk.
- Two of the five groups (10% of students) existed mainly because those students skipped the last questions.
- The groups separate mental health less well. Sadness ranges from 22% to 54% across them, against 27% to 73% across the five profiles above. In particular, they merge the low-risk profile (27% sad) with the vaping-and-drinking profile (63% sad).

This is why the profiles come from the MCA themes and UMAP is used only for the picture. It also explains the separate islands on the map: they reflect those yes-or-no facts and skipped questions.

**A third version, built from risk questions only.** We then removed the yes-or-no facts: "not applicable" answers (never had sex, did not date, did not drive, never tried) count as neutral, the repeated sex questions are cut to three, and identity questions are left out. Students who skipped more than a quarter of the questions were set aside, leaving 71% of the 2023 students.

- **The islands disappear.** Grouping on the UMAP layout now finds a single cloud holding 96% of students. Risk among students is a continuum, not a set of separate camps.
- **Grouping on the MCA themes still works, and the groups are now about risk:**

| Risk-only group | Share of students | Felt sad |
|:------------------------------------------------|:------|:------|
| Low risk: no vaping, drinking or marijuana | 35% | 24% |
| Eats and drinks a lot of everything | 7% | 39% |
| Inactive, poor diet, no substance use | 29% | 42% |
| Vaping, bullied, exposed to violence | 22% | 70% |
| Heavy multi-risk: daily vaping, heavy marijuana | 7% | 75% |


- **It separates sadness slightly better** than the main profiles on the same students (a score of 0.707 against 0.691 from group membership alone), and it adds something new: among students who use no substances, the inactive, poor-diet group reports far more sadness (42%) than the low-risk group (24%).
- **It is not yet the main result.** It was built on 71% of the 2023 students, but it has now been checked on 2025 with every student placed into the nearest group (see below). We keep both versions side by side: comparing them is one of the project's findings.

**The risk-only groups checked on 2025.** The groups were fixed on 2023. Every student in both years, including those who skipped many questions, was then placed into the nearest group.

| Risk-only group | Share 2023 | Felt sad 2023 | Share 2025 | Felt sad 2025 |
|:------------------------------------------------|:-----|:-----|:-----|:-----|
| Low risk: no vaping, drinking or marijuana | 38% | 23% | 41% | 21% |
| Eats and drinks a lot of everything | 6% | 39% | 8% | 28% |
| Inactive, poor diet, no substance use | 29% | 41% | 30% | 35% |
| Vaping, bullied, exposed to violence | 22% | 69% | 18% | 62% |
| Heavy multi-risk: daily vaping, heavy marijuana | 6% | 73% | 4% | 64% |

- **The groups hold up on new students.** The order is the same in 2025, and the group sizes barely move.
- **They beat the main profiles on 2025 too:** a score of 0.676 against 0.656 from group membership alone, across all students.
- Students who skipped many questions fit into the same groups with similar sadness rates, so leaving them out when building the groups did not distort the result.

## Predicting who is at risk (five models)

Part 2 of the methods has now been run. Each model was tuned on 2023 and then tested once on the 2025 students.

| Model | Score on 2023 | Score on 2025 | Sad students caught | Flags that were right |
|:---------------------|:------|:------|:------|:------|
| Logistic regression | 0.827 | 0.811 | 68% | 63% |
| Decision tree | 0.772 | 0.761 | 66% | 57% |
| Random forest | 0.820 | 0.804 | 68% | 62% |
| Neural network (MLP) | 0.823 | 0.811 | 61% | 66% |
| Voting ensemble | 0.828 | 0.813 | 66% | 64% |

- The simplest model is as good as any. The ensemble beats logistic regression by 0.002, which is not worth the added complexity, so logistic regression is the model we keep.
- Every model holds up on new students: scores drop by only about 0.01 to 0.02 from 2023 to 2025.
- In plain numbers for logistic regression: of 3,366 students in 2025 who felt sad or hopeless, it catches 2,294 and misses 1,072. It also flags 1,379 students who were not sad.

## Can a stronger model do better?

We tried four ways to raise the score. Ranges are 95% margins of error on the 2025 students.

| Model | Score on 2025 | Gain over the current choice |
|:---------------------------------------------|:--------------------|:-----------------------|
| Logistic regression (current choice) | 0.811 (0.800 to 0.821) | |
| Logistic regression with sex-specific answers | 0.810 (0.799 to 0.821) | -0.001 (-0.002 to 0.000) |
| Gradient boosting, answers kept in order | 0.813 (0.803 to 0.822) | +0.002 (-0.002 to +0.006) |
| Gradient boosting plus our themes and groups | 0.812 (0.803 to 0.820) | +0.002 (-0.004 to +0.006) |
| Average of logistic regression and boosting | 0.815 (0.805 to 0.824) | +0.004 (+0.001 to +0.007) |

- **No single model reliably beats the current choice.** The gains from boosting are within the margin of error.
- **Averaging two models gives a real but tiny gain** of 0.004. It is not worth giving up a model we can explain.
- **The limit is the survey, not the model.** Every model we tried, except the single decision tree, lands between 0.80 and 0.82. Predicting a yes-or-no feeling from behavior questions seems to top out there.

## A clearer way to report the score

AUROC is hard to explain to a non-technical audience. These measures say the same thing in plain terms (logistic regression, 2025 students):


- **Ranking:** among the tenth of students the model rates highest, 84% felt sad or hopeless. Among the tenth it rates lowest, 4% did. The overall rate is 35%.
- **Reach:** the highest-rated fifth of students contains 43% of all students who felt sad.
- **A cutoff chosen on purpose:** we set the cutoff on 2023 to catch 80% of sad students. On 2025 it catches 77%, at the cost of flagging 30% of students who were not sad.
- **Calibration:** the model's predicted risk averages 38% on 2025 students, while the actual rate was 35%. It overstates slightly because sadness was more common in 2023, when it was trained. Within each tenth of students, predicted and actual rates are within a few points of each other.

## Which questions matter most (RFE and LASSO)

Part 3 has now been run. Both methods work on whole questions: a question is either kept or dropped.


- **A short list is enough.** The 10 strongest questions score 0.797 on 2025, against 0.805 for all 86.
- **Both methods agree on a core of seven:** emotional abuse by a parent or adult at home, sleep, sex, feeling close to people at school, sexual identity, bullying at school and electronic bullying.
- **Social media is not on either short list.** RFE ranks it 32nd of 86. LASSO keeps it in its larger 52-question model (ranked 24th) but drops it from its 15-question model.
- **Beyond the core the two methods disagree.** Their 15-question lists share only 7 questions, and RFE keeps a few that look like noise (potato eating, sunburn). Only the shared core should be treated as reliable.

## Explaining the models with SHAP

SHAP measures how much each answer pushes a student's predicted risk up or down. It explains the boosting model, which is otherwise a black box, and lets us explain a single student's prediction. Both models were trained on 2023 and explained on 3,000 students from 2025.


- **The two models agree.** Eight of the top ten questions are the same for boosting and logistic regression: emotional abuse by a parent, sex, sexual identity, sleep, feeling close to people at school, bullying at school and online, and being treated unfairly at school because of race.
- **It confirms the RFE and LASSO list** with a third, independent method.
- **Social media ranks 27th of 86** in boosting and 28th in logistic regression.
- **The model uses social media the way the odds suggested.** Students who use it a few times a week or less are pushed toward lower risk, and the push turns upward only for the heaviest users (more than once an hour). Even there it is small: about a tenth of the push from emotional abuse.

**Two students explained**


- **High-risk student (86%, did feel sad):** female, four hours of sleep or less, does not feel close to people at school, bisexual, has vaped and experienced forced sex.
- **Low-risk student (9%, did not feel sad):** never emotionally abused, male, sleeps nine hours, feels close to people at school. Having vaped pushes the risk up, but the other answers outweigh it.
- The two students were picked by a fixed rule (the students nearest the 95th and 10th percentiles of predicted risk whose answers matched the prediction), not chosen by hand. Race is used by the model but left off these charts, so no single student is described by race.

## Fairness check

Logistic regression on the 2025 students, using one cutoff for everyone:

| Group | Students | Felt sad | Sad students caught | False alarms |
|:-----------------------------|:------|:------|:------|:------|
| Girls | 4,777 | 46% | 78% | 35% |
| Boys | 4,881 | 24% | 49% | 13% |
| Asian | 475 | 29% | 63% | 20% |
| Black | 1,968 | 35% | 64% | 23% |
| White | 3,404 | 32% | 67% | 19% |
| Hispanic or Latino | 2,994 | 38% | 72% | 25% |
| Multiple races, non-Hispanic | 692 | 41% | 71% | 23% |

- **The biggest gap is by sex.** With one cutoff the model misses about half of the boys who felt sad, while raising far more false alarms for girls. The model ranks students about equally well within each sex (0.79 to 0.80), so this is a cutoff problem: a real screening tool would need a separate cutoff for each sex.
- Gaps between racial and ethnic groups are smaller (63% to 72% caught). American Indian or Alaska Native and Native Hawaiian or Pacific Islander students were too few in 2025 to check.

**Fixing the gap with separate cutoffs.** We chose one cutoff for girls and one for boys using 2023 data only, then applied them unchanged to the 2025 students:

|                         | Sad girls caught | Sad boys caught | False alarms, girls | False alarms, boys |
|:------------------------|:------|:------|:------|:------|
| One cutoff for everyone | 78% | 49% | 35% | 13% |
| Separate cutoffs        | 68% | 70% | 24% | 26% |

- The gap closes: the model now catches about the same share of sad girls and sad boys, with similar false alarm rates.
- It has a cost. Boys are flagged more often, so fewer of those flags are right (46%, down from 55%), and overall false alarms rise from 22% to 25%. The share of sad students caught overall stays at 68% to 69%.
- The cutoffs were set on 2023 and still worked on 2025, so the fix carries over to new students.

## A correction to earlier numbers

Race and ethnicity are coded differently in the 2023 and 2025 files, and the first runs treated the codes as if they matched. Both years now use the same categories. The effect was small: 2025 scores rose slightly (the full basic model from 0.803 to 0.806, background only from 0.620 to 0.640), the social media odds moved by 0.02 at most, and the student profiles were not affected. The tables above show the corrected scores.

## Two robustness checks

We repeated the staged social media analysis two ways: with a second mental health question, and with the CDC's survey weights (which make the sample represent all U.S. high school students). The table shows the odds for more-than-hourly users against non-users, with 95% ranges.

+----------------------+---------------------+---------------------+---------------------+---------------------+
| What the model knows | Sad or hopeless     | Sad or hopeless,    | Poor mental health  | Poor mental health, |
|                      | (main analysis)     | weighted            | in the past 30 days | weighted            |
+:=====================+:====================+:====================+:====================+:====================+
| Background + social  | 1.74 (1.37 to 2.24) | 1.63 (1.03 to 2.70) | 1.89 (1.45 to 2.51) | 2.51 (1.63 to 3.57) |
| media                |                     |                     |                     |                     |
+----------------------+---------------------+---------------------+---------------------+---------------------+
| Add sleep and        | 1.68 (1.39 to 2.10) | 1.63 (1.17 to 2.48) | 1.84 (1.23 to 2.64) | 2.58 (1.34 to 4.26) |
| bullying             |                     |                     |                     |                     |
+----------------------+---------------------+---------------------+---------------------+---------------------+
| Add everything else  | 1.33 (1.14 to 1.65) | 1.29 (0.98 to 1.78) | 1.21 (0.97 to 1.45) | 1.31 (0.94 to 1.74) |
+----------------------+---------------------+---------------------+---------------------+---------------------+

**How to read this**

- **The main story holds in all four versions.** Heavy social media use is clearly linked to worse mental health on its own and after sleep and bullying are added. Once every other behavior is included, the link shrinks to about 1.2 to 1.3 times the odds.
- **The last step is borderline.** In the main analysis the full-model range stays above 1.0. In the other three it dips just below 1.0, so we cannot rule out that the remaining link is chance. The honest summary is: a small link may remain, but the data cannot confirm it.
- **Prediction works for both outcomes.** The full model scores 0.806 on 2025 for sadness and 0.765 for poor mental health. Social media adds about 0.01 in both.
- **Weights barely change the scores** (0.806 unweighted, 0.804 weighted for sadness), but they widen the ranges, because a weighted sample carries less information.
- Weighted, 40% of 2023 students and 33% of 2025 students felt sad or hopeless. Poor mental health in the past 30 days was 28% and 23%.

## Protective factors

We asked whether the social media link is weaker for students who have something going for them. For each factor, students were split into those who have it and those who do not, and the social media odds (more-than-hourly users against non-users, adjusted for background) were worked out inside each half.

| Factor | Sad with it | Sad without it | Social media odds with it | Social media odds without it |
|:-------------------------------------|:------|:------|:------|:------|
| Sleeps 8+ hours on school nights | 27% | 46% | 1.69 | 1.56 |
| Feels close to people at school | 35% | 55% | 1.70 | 2.03 |
| Active 60+ minutes on 5+ days a week | 35% | 48% | 2.04 | 1.70 |
| Plays on a sports team | 38% | 49% | 1.65 | 1.88 |
| Parents usually know where they are | 42% | 54% | 1.73 | 2.88 |

All figures are for 2023.

**How to read this**

- **Every factor goes with much less sadness.** Students who sleep 8 or more hours report sadness at 27%, against 46% for those who sleep less.
- **Four of the five do not change the social media link.** For sleep, school closeness, activity and sports, the odds are about the same in both halves, and none of the differences is reliable. These factors lower a student's starting level of sadness; they do not cancel out heavy social media use.
- **Even so, they matter more than social media does.** A heavy social media user who sleeps 8 or more hours (33% sad) is better off than a light user who does not (38% sad).
- **One lead: parental monitoring.** Among students whose parents usually know where they are, the social media odds are 1.73. Among those whose parents do not, they are 2.88. The difference is reliable in 2023 and points the same way in 2025 (1.69 against 3.66). But it is one of five comparisons and the second group is small, so we treat it as a lead to report, not a proven result.

## Checks on the work

Before presenting, we tested whether the results can be trusted.

- **The data was read correctly.** In both years, all 97 of the CDC's own yes-or-no summary columns agree with the answers we read in, and no answer falls outside the options the CDC defines.
- **The 2025 file matches CDC's 2025 guide.** The record count matches (10,710 usable questionnaires), every question has the same answer options as the guide, and the survey weights add up to the sample size as the guide says. Our race recode matches CDC's 2025 race codes; Middle Eastern or North African students are grouped with White, the same way 2023 counts them. The 2025 guide prints no answer counts, unlike 2023, so counts could not be compared directly. One CDC inconsistency turned up: for the two parental-abuse questions, the guide defines "yes" as "rarely or more", but CDC's yes/no columns in the file use "most of the time or always". It does not affect us, because we use the full answers, not CDC's yes/no columns.
- **The score is not an accident.** When the outcome is scrambled, the model's score falls to 0.49 to 0.51, which is a coin flip. That rules out information leaking into the model.
- **No single question does all the work.** The strongest question on its own (emotional abuse by a parent) scores 0.705. Social media on its own scores 0.551, ranking 40th of 86.
- **Everything reproduces.** All 20 scripts were rerun from scratch in order. The results matched the reported numbers; the only differences were in the ordering of a few weak questions deep in the LASSO list, which has since been fixed.
- **How stable is the list of questions that matter?** We repeated the selection on 40 resampled versions of the data:

| Question | Picked by RFE (top 10) | Picked by LASSO (short list) |
|:----------------------------------|:------|:------|
| Emotional abuse by a parent | 100% | 100% |
| Sex | 98% | 100% |
| Sleep | 100% | 5% |
| Feeling close to people at school | 80% | 2% |
| Sexual identity | 62% | 100% |
| Electronic bullying | 55% | 100% |
| Bullying at school | 48% | 100% |
| Social media | 2% | 0% |

Emotional abuse and sex are chosen every time by both methods. The other five are chosen consistently by one method and less so by the other. Sleep and school closeness are missed by the short LASSO because it penalizes questions with many answer options; they rank 2nd and 4th in the longer LASSO list. Social media almost never makes either list.

**What still needs doing**

- The social media figures in the staged table come from the first basic model. The tuned models and feature selection are now done (see the two sections above), but the staged social media analysis has not been repeated on the tuned model.
- Margins of error come from re-running the model 300 times on reshuffled 2023 data (a bootstrap that keeps each sampling area together). They should be recomputed for the final tuned model.
- An earlier run showed 2023 scoring lower than 2025 (0.751 vs 0.803). That was a flaw in how the 2023 data was split, explained in the cross-validation note above. It is fixed, and the corrected 2023 score is 0.820.