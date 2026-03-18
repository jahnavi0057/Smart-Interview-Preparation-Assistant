"""
sample_data.py
--------------
Contains sample job descriptions, resumes, interview questions,
and ideal answers used for demonstration and testing.
"""

# ---------------------------------------------------------------------------
# Sample job description
# ---------------------------------------------------------------------------
SAMPLE_JOB_DESCRIPTION = """
We are looking for a Python Developer with strong experience in machine learning
and data science. The ideal candidate should be proficient in Python programming,
scikit-learn, TensorFlow or PyTorch, data preprocessing, feature engineering,
model training, and deployment. Experience with SQL databases, REST APIs, and
version control systems like Git is required. Knowledge of deep learning,
natural language processing (NLP), and cloud platforms such as AWS or GCP
is a plus. The candidate should have strong problem-solving skills and be
comfortable working in an Agile development environment.
"""

# ---------------------------------------------------------------------------
# Sample candidate resume / skills
# ---------------------------------------------------------------------------
SAMPLE_RESUME = """
I am a software engineer with 3 years of experience in Python development.
I have worked on multiple machine learning projects involving data preprocessing,
feature engineering, and model building using scikit-learn and TensorFlow.
I am skilled in SQL, REST API development using Flask, and version control with Git.
I have knowledge of natural language processing, including text classification
and sentiment analysis. I have deployed models on AWS and have experience working
in Agile teams. I also have experience with data visualization using Matplotlib
and Seaborn.
"""

# ---------------------------------------------------------------------------
# Question bank: skill -> list of (question, ideal_answer) pairs
# ---------------------------------------------------------------------------
QUESTION_BANK = {
    "python": [
        (
            "What are Python's key features that make it popular for data science?",
            "Python is popular for data science because of its simple and readable syntax, "
            "a large ecosystem of libraries like NumPy, Pandas, scikit-learn, TensorFlow, "
            "and PyTorch, strong community support, interactive notebooks like Jupyter, "
            "and easy integration with other languages and tools. It supports both "
            "procedural and object-oriented programming paradigms.",
        ),
        (
            "Explain the difference between a list and a tuple in Python.",
            "A list is mutable, meaning its elements can be changed after creation, while "
            "a tuple is immutable and cannot be modified once created. Lists use square "
            "brackets and tuples use parentheses. Tuples are generally faster and use "
            "less memory than lists. Lists are used for collections that may change, "
            "while tuples are used for fixed data.",
        ),
    ],
    "machine learning": [
        (
            "What is the difference between supervised and unsupervised learning?",
            "Supervised learning uses labeled training data where the algorithm learns to "
            "map inputs to known outputs, examples include classification and regression. "
            "Unsupervised learning works with unlabeled data and tries to find hidden "
            "patterns or intrinsic structures, examples include clustering and "
            "dimensionality reduction. In supervised learning the correct answer is "
            "known during training, while in unsupervised learning it is not.",
        ),
        (
            "How do you handle overfitting in a machine learning model?",
            "Overfitting can be handled by using techniques such as cross-validation, "
            "regularization (L1 and L2), dropout for neural networks, pruning for "
            "decision trees, gathering more training data, using simpler models, "
            "early stopping during training, and data augmentation. Feature selection "
            "and dimensionality reduction can also help reduce overfitting.",
        ),
    ],
    "scikit-learn": [
        (
            "How do you build and evaluate a classification model using scikit-learn?",
            "To build a classification model in scikit-learn, you first preprocess the "
            "data, then split it into training and test sets using train_test_split. "
            "You instantiate a classifier such as LogisticRegression or "
            "RandomForestClassifier, fit it on the training data, and predict on the "
            "test data. Evaluation is done using metrics like accuracy_score, "
            "confusion_matrix, and classification_report from sklearn.metrics.",
        ),
    ],
    "data preprocessing": [
        (
            "What steps do you follow during data preprocessing?",
            "Data preprocessing steps include handling missing values by imputation or "
            "removal, removing duplicate records, encoding categorical variables using "
            "one-hot encoding or label encoding, feature scaling using normalization or "
            "standardization, outlier detection and treatment, feature selection, and "
            "splitting data into training and testing sets. These steps ensure that the "
            "data is clean and ready for model training.",
        ),
    ],
    "sql": [
        (
            "What is the difference between INNER JOIN and LEFT JOIN in SQL?",
            "An INNER JOIN returns only the rows where there is a match in both tables. "
            "A LEFT JOIN returns all rows from the left table and the matched rows from "
            "the right table. If there is no match in the right table, NULL values are "
            "returned for the right table columns. INNER JOIN is used when you want "
            "only matching records, while LEFT JOIN is used when you want all records "
            "from the left table regardless of matches.",
        ),
    ],
    "git": [
        (
            "Explain the Git workflow and common commands you use.",
            "A typical Git workflow involves cloning a repository, creating a new branch "
            "for a feature or bug fix, making changes and staging them with git add, "
            "committing with git commit, and pushing to the remote with git push. "
            "Common commands include git clone, git branch, git checkout, git merge, "
            "git pull, git stash, and git log. Pull requests are used for code review "
            "before merging into the main branch.",
        ),
    ],
    "natural language processing": [
        (
            "What are common NLP preprocessing steps?",
            "Common NLP preprocessing steps include tokenization which splits text into "
            "words or sentences, removing stop words, stemming or lemmatization to "
            "reduce words to their base form, lowercasing, removing punctuation and "
            "special characters, and vectorization techniques such as TF-IDF or word "
            "embeddings to convert text into numerical representations for machine "
            "learning models.",
        ),
    ],
    "deep learning": [
        (
            "What is the vanishing gradient problem and how is it addressed?",
            "The vanishing gradient problem occurs in deep neural networks when gradients "
            "become very small during backpropagation, causing early layers to learn "
            "very slowly. It is addressed by using activation functions like ReLU instead "
            "of sigmoid or tanh, batch normalization, careful weight initialization "
            "such as Xavier or He initialization, residual connections as in ResNet, "
            "and gradient clipping.",
        ),
    ],
    "rest api": [
        (
            "What are the key principles of RESTful API design?",
            "RESTful API design principles include using stateless communication where "
            "each request contains all necessary information, using standard HTTP methods "
            "GET POST PUT DELETE for CRUD operations, using meaningful resource-based "
            "URLs, returning appropriate HTTP status codes, supporting JSON or XML "
            "response formats, versioning the API, and implementing proper "
            "authentication and authorization.",
        ),
    ],
    "aws": [
        (
            "What AWS services have you used for deploying machine learning models?",
            "For deploying machine learning models on AWS, commonly used services include "
            "Amazon SageMaker for training and deploying models, AWS Lambda for serverless "
            "inference, Amazon EC2 for custom compute environments, Amazon S3 for storing "
            "datasets and model artifacts, Amazon ECR for containerized model deployment, "
            "and AWS API Gateway for exposing model endpoints as REST APIs.",
        ),
    ],
}

# ---------------------------------------------------------------------------
# Default ideal answer used when a question falls outside the question bank
# ---------------------------------------------------------------------------
DEFAULT_IDEAL_ANSWER = (
    "Please provide a clear, concise, and accurate explanation that covers "
    "the key concepts, practical applications, and relevant examples related "
    "to the topic. Use technical terminology appropriately and demonstrate "
    "your understanding with real-world scenarios."
)
