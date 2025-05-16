## 🧠 BESESR: Benchmark for Summative Evaluation of Student Responses

This project leverages multiple open-source datasets to benchmark **automated scoring of long-form and short-form student responses**. Below is a summary of the key datasets currently used in this benchmark.

---

### 🔹 [ASAP-AES](https://www.kaggle.com/competitions/asap-aes/data)
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
