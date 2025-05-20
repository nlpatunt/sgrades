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
