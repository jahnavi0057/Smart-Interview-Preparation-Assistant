"""
interview_assistant.py
----------------------
Smart Interview Preparation Assistant

A CLI-based tool that:
  1. Accepts a job description and candidate resume/skills as input.
  2. Extracts important keywords from both texts.
  3. Generates relevant interview questions based on the extracted skills.
  4. Accepts user answers for each question.
  5. Evaluates each answer using TF-IDF + cosine similarity.
  6. Provides a score (out of 10) and personalised feedback for every answer.

Dependencies: scikit-learn, nltk
"""

import re
import string
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sample_data import (
    DEFAULT_IDEAL_ANSWER,
    QUESTION_BANK,
    SAMPLE_JOB_DESCRIPTION,
    SAMPLE_RESUME,
)

# ---------------------------------------------------------------------------
# One-time NLTK resource download (safe to call multiple times)
# ---------------------------------------------------------------------------

def download_nltk_resources():
    """Download required NLTK corpora if they are not already present."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# ---------------------------------------------------------------------------
# Module 1 – Text Preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Clean and normalise raw text.

    Steps:
      - Convert to lowercase.
      - Remove punctuation and digits.
      - Tokenise into words.
      - Remove English stop-words.
      - Lemmatise each token.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        A single string of space-separated, processed tokens.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Lowercase and remove punctuation / digits
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenise
    tokens = nltk.word_tokenize(text)

    # Remove stop-words and short tokens, then lemmatise
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Module 2 – Keyword Extraction
# ---------------------------------------------------------------------------

# Non-technical words that should never appear as extracted skills even if
# they score highly in a TF-IDF pass on a single document.
_NON_TECHNICAL_STOPWORDS = {
    "candidate", "ideal", "strong", "experience", "work", "working",
    "skill", "skills", "knowledge", "ability", "team", "include",
    "including", "using", "use", "used", "plus", "required", "require",
    "develop", "development", "also", "well", "good", "great", "need",
    "needed", "looking", "like", "role", "position", "job", "company",
    "year", "years", "agile", "environment", "comfortable", "proficient",
    "familiar", "solid", "understanding", "build", "building", "built",
}


def extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """
    Extract the most important technical keywords from *text*.

    Strategy (two-pass):
      1. Directly scan the text for known technical skills drawn from the
         QUESTION_BANK keys — this guarantees question-relevant terms are
         captured.
      2. Supplement with TF-IDF-ranked unigrams / bigrams from the
         pre-processed text, filtering out common non-technical words.

    Parameters
    ----------
    text : str
        Raw input text (job description or resume).
    top_n : int
        Maximum number of keywords to return.

    Returns
    -------
    list[str]
        Ordered list of extracted keywords (known skills first, then TF-IDF).
    """
    text_lower = text.lower()
    keywords: list[str] = []
    seen: set[str] = set()

    # Pass 1 – find known skill names from QUESTION_BANK
    for skill_key in QUESTION_BANK:
        if skill_key in text_lower and skill_key not in seen:
            keywords.append(skill_key)
            seen.add(skill_key)

    # Pass 2 – TF-IDF on the preprocessed text to discover additional terms
    processed = preprocess_text(text)
    if processed.strip():
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=200,
            stop_words="english",
        )
        tfidf_matrix = vectorizer.fit_transform([processed])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        # Build a flat set of all individual words that appear in known skill keys
        _skill_words: set[str] = set()
        for bank_key in QUESTION_BANK:
            for word in bank_key.split():
                _skill_words.add(word)

        keyword_scores = sorted(
            zip(feature_names, scores), key=lambda x: x[1], reverse=True
        )
        for kw, _ in keyword_scores:
            if len(keywords) >= top_n:
                break
            # Skip generic / non-technical tokens
            if kw in seen:
                continue
            if any(part in _NON_TECHNICAL_STOPWORDS for part in kw.split()):
                continue
            if len(kw) <= 2:
                continue
            # For bigrams, require at least one word to be a known skill word
            parts = kw.split()
            if len(parts) > 1 and not any(p in _skill_words for p in parts):
                continue
            keywords.append(kw)
            seen.add(kw)

    return keywords[:top_n]


def get_matched_skills(job_keywords: list[str], resume_keywords: list[str]) -> list[str]:
    """
    Return skills present in **both** the job description and the resume.

    A skill is considered matched if the resume keyword appears as a
    sub-string of a job keyword or vice versa (handles 1-gram vs 2-gram
    overlaps).  Redundant entries are pruned so that when a shorter skill
    is a subset of a longer one that is already in the list, only the
    longer (more specific) skill is kept.

    Parameters
    ----------
    job_keywords : list[str]
        Keywords extracted from the job description.
    resume_keywords : list[str]
        Keywords extracted from the resume.

    Returns
    -------
    list[str]
        De-duplicated, pruned list of matched skills.
    """
    matched = []
    seen: set[str] = set()
    for jk in job_keywords:
        for rk in resume_keywords:
            if jk in rk or rk in jk:
                skill = jk if len(jk) >= len(rk) else rk
                if skill not in seen:
                    matched.append(skill)
                    seen.add(skill)

    # Remove shorter skills that are sub-strings of a longer matched skill
    # e.g. drop "api" if "rest api" is already present
    pruned = []
    for skill in matched:
        dominated = any(
            skill != other and skill in other
            for other in matched
        )
        if not dominated:
            pruned.append(skill)

    return pruned


# ---------------------------------------------------------------------------
# Module 3 – Question Generation
# ---------------------------------------------------------------------------

def generate_questions(skills: list[str]) -> list[tuple[str, str]]:
    """
    Generate interview questions for the given list of skills.

    Questions are drawn from ``QUESTION_BANK`` (see *sample_data.py*).
    Each skill is matched against bank keys using substring search.  If no
    match is found, a generic question is produced.

    Parameters
    ----------
    skills : list[str]
        Skills extracted from the overlap of job description and resume.

    Returns
    -------
    list[tuple[str, str]]
        List of (question, ideal_answer) pairs.
    """
    questions = []
    seen_questions = set()

    for skill in skills:
        matched = False
        skill_lower = skill.lower()

        for bank_key, qa_pairs in QUESTION_BANK.items():
            # Match if the bank key is a substring of the skill or vice versa
            if bank_key in skill_lower or skill_lower in bank_key:
                for question, ideal_answer in qa_pairs:
                    if question not in seen_questions:
                        questions.append((question, ideal_answer))
                        seen_questions.add(question)
                        matched = True

        if not matched:
            # Generic fallback question for unrecognised skills
            generic_q = f"Can you explain your experience and knowledge with {skill}?"
            if generic_q not in seen_questions:
                questions.append((generic_q, DEFAULT_IDEAL_ANSWER))
                seen_questions.add(generic_q)

    return questions


# ---------------------------------------------------------------------------
# Module 4 – Answer Evaluation
# ---------------------------------------------------------------------------

def evaluate_answer(user_answer: str, ideal_answer: str) -> tuple[float, str]:
    """
    Evaluate a user's answer against an ideal answer using cosine similarity.

    Both answers are processed through the text pre-processor and then
    converted to TF-IDF vectors.  The cosine similarity (0–1) is scaled to
    a score out of 10 and mapped to qualitative feedback.

    Parameters
    ----------
    user_answer : str
        The answer provided by the candidate.
    ideal_answer : str
        The reference (ideal) answer for the question.

    Returns
    -------
    tuple[float, str]
        A ``(score, feedback)`` pair where *score* is in [0, 10].
    """
    if not user_answer.strip():
        return 0.0, "No answer provided. Please attempt the question."

    processed_user = preprocess_text(user_answer)
    processed_ideal = preprocess_text(ideal_answer)

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_user, processed_ideal])
    except ValueError:
        # Happens when both documents are empty after preprocessing
        return 0.0, "Could not process the answer. Please provide a more detailed response."

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score = round(similarity * 10, 2)

    feedback = _generate_feedback(score)
    return score, feedback


def _generate_feedback(score: float) -> str:
    """Map a numeric score to a descriptive feedback string."""
    if score >= 8.5:
        return (
            "Excellent! Your answer is comprehensive and closely matches the ideal response. "
            "You demonstrated a strong understanding of the topic."
        )
    elif score >= 7.0:
        return (
            "Good answer! You covered most of the key points. "
            "Consider adding more specific examples or technical depth."
        )
    elif score >= 5.0:
        return (
            "Fair answer. You touched on some relevant concepts but missed several important points. "
            "Review the topic and try to include more technical details."
        )
    elif score >= 3.0:
        return (
            "Below average. Your answer lacks depth and misses key concepts. "
            "Study the topic more thoroughly and practice explaining it clearly."
        )
    else:
        return (
            "Needs significant improvement. Your answer is largely off-topic or too brief. "
            "Please revisit the fundamentals of this topic before your interview."
        )


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def print_separator(char: str = "=", width: int = 70) -> None:
    """Print a horizontal separator line."""
    print(char * width)


def print_banner() -> None:
    """Print the application banner."""
    print_separator()
    print("   SMART INTERVIEW PREPARATION ASSISTANT")
    print("   Powered by NLP (TF-IDF + Cosine Similarity)")
    print_separator()
    print()


def get_multiline_input(prompt: str) -> str:
    """
    Collect multi-line input from the user.

    The user signals the end of input by entering a blank line.

    Parameters
    ----------
    prompt : str
        Instruction shown before the input area.

    Returns
    -------
    str
        The complete multi-line string entered by the user.
    """
    print(prompt)
    print("(Press Enter twice when done)\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            if lines:
                break
        else:
            lines.append(line)
    return "\n".join(lines)


def run_cli() -> None:
    """
    Entry point for the interactive CLI session.

    Flow:
      1. Optionally use sample data or provide custom job description / resume.
      2. Extract and display keywords.
      3. Identify matched skills.
      4. Generate and present interview questions one by one.
      5. Collect user answers and evaluate them.
      6. Show a final performance summary.
    """
    download_nltk_resources()
    print_banner()

    # ------------------------------------------------------------------
    # Step 1 – Collect job description and resume
    # ------------------------------------------------------------------
    print("Would you like to use the built-in sample data? (yes/no)")
    use_sample = input("> ").strip().lower()

    if use_sample in ("yes", "y", ""):
        job_description = SAMPLE_JOB_DESCRIPTION
        resume = SAMPLE_RESUME
        print("\n[Sample data loaded.]\n")
    else:
        job_description = get_multiline_input(
            "\n--- Enter the Job Description ---"
        )
        resume = get_multiline_input(
            "\n--- Enter your Resume / Skills ---"
        )

    # ------------------------------------------------------------------
    # Step 2 – Keyword extraction
    # ------------------------------------------------------------------
    print_separator("-")
    print("Extracting keywords from the job description and resume …\n")

    job_keywords = extract_keywords(job_description, top_n=15)
    resume_keywords = extract_keywords(resume, top_n=15)

    print("Top keywords from Job Description:")
    print("  " + ", ".join(job_keywords))
    print("\nTop keywords from Resume:")
    print("  " + ", ".join(resume_keywords))

    # ------------------------------------------------------------------
    # Step 3 – Skill matching
    # ------------------------------------------------------------------
    matched_skills = get_matched_skills(job_keywords, resume_keywords)

    if not matched_skills:
        # Fall back to job keywords so the session is still useful
        matched_skills = job_keywords[:8]

    print(f"\nMatched / Relevant Skills ({len(matched_skills)}):")
    print("  " + ", ".join(matched_skills))

    # ------------------------------------------------------------------
    # Step 4 – Question generation
    # ------------------------------------------------------------------
    questions = generate_questions(matched_skills)

    if not questions:
        print("\nNo questions could be generated. Please try again with a "
              "more detailed job description.")
        sys.exit(0)

    total_questions = len(questions)
    print(f"\n{total_questions} interview question(s) generated.\n")
    print_separator()

    # ------------------------------------------------------------------
    # Step 5 – Interview loop
    # ------------------------------------------------------------------
    results = []  # list of (question, user_answer, score, feedback)

    for idx, (question, ideal_answer) in enumerate(questions, start=1):
        print(f"\nQuestion {idx} of {total_questions}:")
        print(f"  {question}\n")
        print("Your Answer (press Enter twice when done):")

        # Collect multi-line answer; a blank line or EOF (Ctrl+D) terminates
        # input so the session never hangs.
        answer_lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                break
            answer_lines.append(line)
        user_answer = "\n".join(answer_lines)

        score, feedback = evaluate_answer(user_answer, ideal_answer)

        print(f"\n  Score   : {score} / 10")
        print(f"  Feedback: {feedback}")
        print_separator("-")

        results.append((question, user_answer, score, feedback))

    # ------------------------------------------------------------------
    # Step 6 – Final summary
    # ------------------------------------------------------------------
    print_separator()
    print("  FINAL PERFORMANCE SUMMARY")
    print_separator()

    total_score = sum(r[2] for r in results)
    average_score = total_score / len(results) if results else 0.0
    max_possible = len(results) * 10

    for i, (question, _, score, feedback) in enumerate(results, start=1):
        print(f"\nQ{i}: {question}")
        print(f"    Score   : {score} / 10")
        print(f"    Feedback: {feedback}")

    print_separator()
    print(f"  Overall Score : {total_score:.2f} / {max_possible}")
    print(f"  Average Score : {average_score:.2f} / 10")
    print_separator()

    # Overall recommendation
    if average_score >= 7.5:
        print("\n  Great performance! You are well-prepared for this interview.")
    elif average_score >= 5.0:
        print("\n  Decent performance. Focus on the weaker areas before your interview.")
    else:
        print("\n  Keep practising! Review the topics above and attempt again.")

    print()
    print_separator()
    print("  Thank you for using the Smart Interview Preparation Assistant!")
    print_separator()


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_cli()
