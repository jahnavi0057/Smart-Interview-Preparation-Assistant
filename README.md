# Smart Interview Preparation Assistant

Smart Interview Preparation Assistant is an AI-based CLI tool that analyses job descriptions and resumes to generate relevant interview questions and evaluate candidate answers using NLP techniques (TF-IDF + Cosine Similarity).

## Features

- **Keyword Extraction** – Identifies important technical skills from both the job description and the candidate's resume using a two-pass strategy (known-skill detection + TF-IDF ranking).
- **Skill Matching** – Finds the overlap between job requirements and candidate skills.
- **Question Generation** – Produces targeted interview questions for matched skills drawn from a built-in question bank.
- **Answer Evaluation** – Compares user answers against ideal answers using TF-IDF vectorisation and cosine similarity, returning a score out of 10.
- **Personalised Feedback** – Provides qualitative feedback for each answer and an overall performance summary.
- **CLI Interface** – Fully interactive terminal-based session with optional built-in sample data.

## Project Structure

```
Smart-Interview-Preparation-Assistant/
├── interview_assistant.py   # Main application (all modules + CLI)
├── sample_data.py           # Sample job description, resume, and question bank
├── requirements.txt         # Python dependencies
└── README.md
```

## Requirements

- Python 3.10+
- scikit-learn
- nltk

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/jahnavi0057/Smart-Interview-Preparation-Assistant.git
cd Smart-Interview-Preparation-Assistant

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

NLTK corpora (`stopwords`, `wordnet`, `punkt`) are downloaded automatically on first run.

## Usage

```bash
python interview_assistant.py
```

You will be prompted to either use built-in sample data or enter your own job description and resume.

## Example Session

```
======================================================================
   SMART INTERVIEW PREPARATION ASSISTANT
   Powered by NLP (TF-IDF + Cosine Similarity)
======================================================================

Would you like to use the built-in sample data? (yes/no)
> yes

[Sample data loaded.]

----------------------------------------------------------------------
Extracting keywords from the job description and resume …

Top keywords from Job Description:
  python, machine learning, scikit-learn, data preprocessing, sql, git,
  natural language processing, deep learning, rest api, aws

Top keywords from Resume:
  python, machine learning, scikit-learn, data preprocessing, sql, git,
  natural language processing, rest api, aws

Matched / Relevant Skills (9):
  python, machine learning, scikit-learn, data preprocessing, sql, git,
  natural language processing, rest api, aws

12 interview question(s) generated.

======================================================================

Question 1 of 12:
  What are Python's key features that make it popular for data science?

Your Answer (press Enter twice when done):
Python is popular because of its simple syntax and large ecosystem of
libraries like NumPy, Pandas, and scikit-learn for data science.

  Score   : 6.72 / 10
  Feedback: Good answer! You covered most of the key points. Consider
            adding more specific examples or technical depth.
----------------------------------------------------------------------
...

======================================================================
  FINAL PERFORMANCE SUMMARY
======================================================================
  Overall Score : 72.40 / 120
  Average Score : 6.03 / 10
======================================================================

  Decent performance. Focus on the weaker areas before your interview.
```

## Module Overview

| Module | Function(s) | Description |
|---|---|---|
| Text Preprocessing | `preprocess_text` | Lowercase, remove punctuation, stop-words, lemmatise |
| Keyword Extraction | `extract_keywords`, `get_matched_skills` | TF-IDF + known-skill detection |
| Question Generation | `generate_questions` | Skill-based question lookup with generic fallback |
| Answer Evaluation | `evaluate_answer` | TF-IDF + cosine similarity, score 0–10 |
| CLI Interface | `run_cli` | Interactive terminal session |

