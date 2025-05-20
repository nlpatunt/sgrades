## 🧠 BESESR: Benchmark for Summative Evaluation of Student Responses

This project leverages multiple open-source datasets to benchmark **automated scoring of long-form and short-form student responses**. Below is a summary of the key datasets currently used in this benchmark.

---

### 🔹 [ASAP-AES](https://www.kaggle.com/competitions/asap-aes/data)
- **HF Link**: [HFLink](https://huggingface.co/datasets/nlpatunt/ASAP-AES)
- **Task**: Essay scoring for students in grades 7–10  
- **Response type**: Long-form essays (multi-paragraph)  
- **Score range**: 0–6 or 0–12 depending on the prompt  
- **Example**:

```json
{
  "essay_id": 21112,
  "essay_set": 8,
  "essay": "Laughter plays a huge part in my life...",
  "rater1_domain1": 17,
  "rater2_domain1": 18,
  "domain1_score": 35,
  "rater1_trait1": 4,
  "rater1_trait2": 4,
  "rater1_trait3": 4,
  "rater1_trait4": 4,
  "rater1_trait5": 3,
  "rater1_trait6": 3,
  "rater2_trait1": 4,
  "rater2_trait2": 4,
  "rater2_trait3": 4,
  "rater2_trait4": 4,
  "rater2_trait5": 4,
  "rater2_trait6": 3,
  "rater3_domain1": null,
  "rater3_trait1": null,
  "rater3_trait2": null,
  "rater3_trait3": null,
  "rater3_trait4": null,
  "rater3_trait5": null,
  "rater3_trait6": null
}
```

---

### 🔹 [ASAP-SAS](https://www.kaggle.com/datasets/asap-sas/data)
- **HF Link**: [HFLink](https://huggingface.co/datasets/nlpatunt/ASAP-SAS)
- **Task**: Short-answer scoring focused on science/math comprehension  
- **Response type**: Short, rubric-based answers (typically 1–3 sentences)  
- **Score range**: 0–3  
- **Example**:

```json
{
  "Id": 1,
  "EssaySet": 1,
  "EssayText": "Some additional information that we would need to replicate the experiment is how much vinegar should be placed in each identical container, how or what tool to use to measure the mass of the four different samples and how much distilled water to use to rinse the four samples after taking them out of the vinegar.",
  "Score1": 1,
  "Score2": 1
}
```

---
### 🔹 [AES](https://www.kaggle.com/datasets/jaytonde/aes-dataset)
- **HF Link**: [HFLink](https://huggingface.co/datasets/nlpatunt/AES)
- **Task**: Essay scoring for educational essays across multiple prompts  
- **Response type**: Full-length student essays  
- **Score range**: 0–6 or 0–12 depending on prompt  
- **Example**:

```json
{
  "essay_id": "000d118",
  "full_text": "Many people have car where they live. The thing that people do more than everything is go to work...",
  "score": 3
}
```
---

### 🔹 [ASAP2](https://www.kaggle.com/datasets/lburleigh/asap-2-0)
- **HF Link**: [HFLink](https://huggingface.co/datasets/nlpatunt/ASAP2)
- **Task**: Essay evaluation based on reading comprehension and evidence-based argumentation  
- **Response type**: Long-form student essays in response to informational article prompts  
- **Score range**: 0–4 or 0–6 depending on rubric  
- **Includes**: Demographic indicators, source article, and prompt text  
- **Example**:

```json
{
  "essay_id": "AAAVUP14319000159574",
  "score": 4,
  "full_text": "The author suggests that studying Venus is worthy enough even though it is very dangerous. The author mentioned that on the planet's surface, temperatures average over 800 degrees Fahrenheit, and the atmospheric pressure is 90 times greater than what we experience on our own planet ...",
  "assignment": "In \"The Challenge of Exploring Venus,\" the author suggests studying Venus is a worthy pursuit despite the dangers it presents. Using details from the article, write an essay evaluating how well the author supports this idea...",
  "prompt_name": "Exploring Venus",
  "economically_disadvantaged": "Economically disadvantaged",
  "student_disability_status": "Identified as having disability",
  "ell_status": "No",
  "race_ethnicity": "Black/African American",
  "gender": "F",
  "source_text_1": "The Challenge of Exploring Venus\nVenus, sometimes called the “Evening Star,” is one of the brightest points of light in the night sky... [full article truncated for brevity]",
  "source_text_2": null,
  "source_text_3": null,
  "source_text_4": null
}
```
---

### 🔹 [Rice Chemistry (RICE_CHEM:HFLink)](https://huggingface.co/datasets/nlpatunt/rice__chem)
- **Task**: Rubric-based scoring of chemistry student responses  
- **Response type**: Short-form but structured explanations (around 150 words)  
- **Score range**: Typically 0–5  
- **Example**:

```json
{
  "sis_id": "a068db7d-6013-4d2d-8ff0-eb6700954f61",
  "question_id": "Q1",
  "Prompt": "When studying the emission sources within the Milky Way, a satellite detected interplanetary clouds... briefly explain 1) why the removal of each additional electron requires more energy than the removal of the previous one, and 2) the relative magnitude of the values observed.",
  "student_response": "Coulomb's Law (V(r)=q₁q₂/r) takes into consideration three things: electron-electron repulsion, nuclear charge (Z_eff), and the distance between the observed electron and the nucleus...",
  "Score": 5.0,
  "correctly cites decreased electron electron repulsion": true,
  "relates decreased electron electron repulsion to decreased potential energy": true,
  "3rd and 4th electrons ionized feel same core charge": true,
  "3rd and 4th electrons ionized from n=3 shell and have same radius": true,
  "5th electron ionized from n=2 shell and feels higher core charge": false,
  "5th electron ionized from n=2 shell and has smaller radius": false,
  "correctly explains relationship of potential energy to ionization energy": false,
  "partially explains relationship between potential energy and ionization energy": true
}

```
---


### 🔹 [SciEntSBank](https://huggingface.co/datasets/nlpatunt/SciEntSBank)
- **Task**: Short-answer grading for science education questions  
- **Response type**: 1–3 sentence explanations to open-ended questions  
- **Variants**:
  - **2-way classification**: `correct` / `incorrect`
  - **3-way classification**: `correct` / `partially_correct` / `incorrect`

---

#### 🔹 Example: 2-way Classification

```json
{
  "question_id": "SE_22c",
  "question_text": "Andi and Scott decided to investigate solar water heaters using collectors of different colors (but the same size) as one variable and covered versus uncovered as a second variable. Look at the graph of their data and answer the questions below. Why would comparing only the red uncovered heater with the blue covered heater not provide useful information?",
  "student_answer": "I think red reflected the heat away.",
  "label": "incorrect"
}
```

---

#### 🔹 Example: 3-way Classification

```json
{
  "question_id": "SE_22c",
  "question_text": "Andi and Scott decided to investigate solar water heaters using collectors of different colors (but the same size) as one variable and covered versus uncovered as a second variable. Look at the graph of their data and answer the questions below. Why would comparing only the red uncovered heater with the blue covered heater not provide useful information?",
  "student_answer": "I think red reflected the heat away.",
  "label": "incorrect"
}
```
> *(Note: This example had the same content for both 2-way and 3-way due to dataset overlap — labels differ in other entries.)*

---

### 🔹 [BEEtlE](https://huggingface.co/datasets/nlpatunt/BEEtlE)
- **Task**: Short-answer grading on electronics and physical science topics  
- **Response type**: Short conceptual explanations  
- **Variants**:
  - **2-way classification**: `correct` / `incorrect`
  - **3-way classification**: `correct` / `partially_correct` / `incorrect`

---

#### 🔹 Example: 2-way Classification

```json
{
  "question_id": "BULB_C_VOLTAGE_EXPLAIN_WHY1",
  "question_text": "Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.",
  "student_answer": "there is no connection",
  "label": "incorrect"
}
```

---

#### 🔹 Example: 3-way Classification

```json
{
  "question_id": "BULB_C_VOLTAGE_EXPLAIN_WHY1",
  "question_text": "Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.",
  "student_answer": "there is no connection",
  "label": "incorrect"
}
```
> *(Note: In this sample, both variants share the same data row; label granularity differs across the dataset.)*
---

### 🔹 [Grade Like a Human – OS](https://huggingface.co/datasets/nlpatunt/grade_like_a_human_dataset_os)
- **Task**: Rubric-based grading of long-form student answers to Operating Systems questions  
- **Response type**: Multi-sentence conceptual explanations and numerical reasoning  
- **Score range**: Varies by question (e.g., 0–19 or 0–40), typically using sub-scores  
- **Includes**: Criteria and sample answers for reference  
- **Adversarial flag**: `score_outlier` is used to mark responses flagged as outliers

---

#### 🔹 Example 1 (Q1: Scheduling Algorithms)

```json
{
  "question_id": 1,
  "question": "Now do the same but with jobs of different lengths: 100, 200, and 300. The commands are (./scheduler.py -p SJF -l 100,200,300) and (./scheduler.py -p FIFO -l 100,200,300)...",
  "answer": "The result with SJF is still the same as with FIFO. If the order is changed, the result may be different...",
  "score_1": 6.5,
  "score_2": 6.5,
  "score_3": 6.0,
  "full_points": 19
}
```

---

#### 🔹 Example 2 (Q6: Paging Translation)

```json
{
  "question_id": 6,
  "question": "Use the simulator paging-multilevel-translate.py to perform translations... For each of the following virtual addresses, write down the physical address it translates to or write down that it is a fault...",
  "answer": "Virtual Address 5a23: 101 1010 0010 0011 ... Fault\nVirtual Address 14ab: 0001 0010 1010 1011 ... Fault\n...",
  "score_1": 23.0,
  "score_2": null,
  "score_3": 28,
  "full_points": 40
}
```
---

### 🔹 [Automatic Short Answer Grading (ASAG)](https://huggingface.co/datasets/nlpatunt/automatic_short_answer_grading)
- **Task**: Automatic scoring of short student answers using reference answers and rubric feedback  
- **Response type**: 1–3 sentence technical or conceptual answers  
- **Score range**: Typically 0–5 (decimal scoring supported)  
- **Includes**: Model-generated comments and human-graded score annotations  

---

#### 🔹 Example

```json
{
  "question": "What is the Euler tour traversal of a tree?",
  "desired_answer": "A walk around the tree, starting with the root, where each node is seen three times: from the left, from below, from the right.",
  "student_answer": "it starts node on the left of the root and then proceeds to visits each node in a left to right order, visits the root, and then proceeds to repeat the previous step on the right side of the tree.",
  "grade": 2,
  "comment": "Grade: 2.5/5\n\nComment: The student answer partially describes the traversal but misses some important details..."
}
```
---

### 🔹 [Utilizing Large Language Models for EFL Essay Grading](https://huggingface.co/datasets/nlpatunt/utilizing_large_language_models_for_EFL_essay_grading)
- **Task**: Multi-trait essay scoring using analytic rubrics  
- **Response type**: Long-form argumentative essays by EFL (English as a Foreign Language) students  
- **Score range**: 1–5 (per trait); includes both human and LLM-assigned scores  
- **Domains**: Grammar, Content, Organization, Style, Mechanics  
- **Includes**: Prompts, full essays, rubrics, and comparisons between human and LLM scores  

---

#### 🔹 Example (Essay on Information Technologies in Learning)

```json
{
  "essay_text": "The Importance of Information Technologies in Learning...",
  "default_prompt": "Grade the student essay below based on the rubric provided.",
  "rubrics": {
    "Grammar": {
      "Human_Mean": 4.0,
      "ChatGPT": 4.3,
      "Bard": 3.6
    },
    "Content": {
      "Human_Mean": 3.73,
      "ChatGPT": 4.9,
      "Bard": 4.0
    },
    "Organization": {
      "Human_Mean": 3.87,
      "ChatGPT": 5.0,
      "Bard": 3.6
    },
    "Style": {
      "Human_Mean": 3.73,
      "ChatGPT": 4.1,
      "Bard": 3.5
    },
    "Mechanics": {
      "Human_Mean": 4.13,
      "ChatGPT": 4.3,
      "Bard": 3.5
    }
  }
}
```
---

### 🔹 [Human-AI-Collaborative-Essay-Scoring (CSEE)](https://huggingface.co/datasets/nlpatunt/Human-AI-Collaborative-Essay-Scoring)
- **Task**: Multi-dimensional scoring of English-as-a-Foreign-Language (EFL) essays written in response to open-ended prompts  
- **Response type**: Formal English emails or essays written by Chinese high school students  
- **Score range**: Each essay is rated across three traits:
  - `content_score` (1–5),
  - `language_score` (1–5),
  - `structure_score` (1–5),  
  - with a derived `overall_score` (e.g., 12.0)
- **Includes**: Full essay text, prompt, and scores

---

#### 🔹 Example

```json
{
  "essay_id": 11039372,
  "prompt_id": 1,
  "prompt": "Suppose you are Li Hua, a senior student at Hongxing High School. Your school is currently soliciting ideas for the senior graduation ceremony... Please write an email to your British friend Jim asking for his advice.",
  "essay": "Dear Jim,\nI'm willing to share some ideas about my school's graduation ceremony and I'm writing to ask for your suggestions...\nYours,\nLi Hua",
  "content_score": 4.5,
  "language_score": 4.5,
  "structure_score": 3.0,
  "overall_score": 12.0
}
```
---
