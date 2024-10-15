# Rachel HR Interview Bot

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Technical Details](#technical-details)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

## 1. Introduction

Rachel HR Interview Bot is a cutting-edge, AI-powered interview preparation assistant designed to revolutionize the HR interview process for students, freshers, and experienced professionals. Developed by Sam Naveenkumar V (URK22AI1043) and Aravindan M (URK22AI1026) from the B.Tech Artificial Intelligence and Data Science program at Karunya Institute of Technology and Sciences, Rachel employs advanced natural language processing (NLP) techniques to provide a personalized and comprehensive interview experience.

This README file provides an in-depth overview of the project, its features, technical details, and instructions for setup and usage.

## 2. Features

Rachel HR Interview Bot offers a wide range of features designed to enhance the interview preparation process:

1. **Resume Analysis**: Utilizes PDF extraction and NLP techniques to analyze resumes and identify domains of specialization.
2. **Personalized Question Generation**: Creates tailored technical HR interview questions based on the candidate's background, projects, and chosen job role.
3. **Interactive Chat Interface**: Provides a user-friendly, WhatsApp-like chat interface for a seamless interview simulation experience.
4. **Answer Evaluation**: Employs advanced algorithms to assess user responses and provide ratings on a scale of 0-10.
5. **Constructive Feedback**: Offers detailed feedback on each answer, highlighting strengths and areas for improvement.
6. **Expected Answer Generation**: Provides model answers to help users understand ideal responses to interview questions.
7. **GPU Acceleration**: Utilizes CUDA for faster processing and improved performance.
8. **Customizable Job Roles**: Supports a wide range of job roles across various engineering and scientific disciplines.
9. **Job Description Integration**: Incorporates specific job descriptions to generate highly relevant interview questions.
10. **Chat History Backup**: Allows users to save and review their interview sessions for future reference.

## 3. Technology Stack

Rachel HR Interview Bot leverages a powerful combination of cutting-edge technologies:

- **Python**: The core programming language used for development.
- **Gradio**: For creating the user-friendly web interface.
- **PyTorch**: Utilized for GPU acceleration and deep learning capabilities.
- **Llama-cpp**: Implements the advanced language model for question generation and answer evaluation.
- **spaCy**: Provides natural language processing capabilities for text analysis.
- **PyTextRank**: Used for keyword extraction and text summarization.
- **scikit-learn**: Implements TF-IDF vectorization and cosine similarity for answer comparison.
- **PyPDF2**: Enables PDF parsing for resume analysis.

## 4. Installation

To set up Rachel HR Interview Bot, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/rachel-hr-bot.git
   cd rachel-hr-bot
   ```

2. Create a virtual environment:

   For Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   For macOS and Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the necessary model files:
   - Llama model: Download from [https://ollama.com/library/llama3.2:3b-instruct-q8_0](https://ollama.com/library/llama3.2:3b-instruct-q8_0)
   - spaCy English model: Run `python -m spacy download en_core_web_sm`

5. Set up CUDA (if using GPU acceleration):
   - Ensure you have CUDA-compatible GPU and drivers installed
   - Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## 5. Usage

To launch Rachel HR Interview Bot:

1. Activate the virtual environment (if not already activated).
2. Run the main script:
   ```
   python rachel_hr_bot.py
   ```
3. Open the provided URL in your web browser to access the Gradio interface.

Using the interface:
1. Upload your resume (PDF format) using the file input.
2. Select your job role from the dropdown menu.
3. Enter the job description in the provided text area.
4. Click "Generate Questions" to start the interview simulation.
5. Interact with Rachel by typing your answers in the chat input.
6. Use the "Skip" button to move to the next question if needed.
7. Click "Generate Answer" to see an expected answer for reference.
8. After completing the interview, click "Provide Feedback" for a comprehensive evaluation.

## 6. Technical Details

### 6.1 Resume Analysis

The resume analysis function uses PyPDF2 to extract text from uploaded PDF files:

```python
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
```

This function reads each page of the PDF and concatenates the extracted text, providing a clean string representation of the resume content.

### 6.2 Domain Analysis

The `analyze_domain` function identifies the candidate's specialization based on keywords in the resume:

```python
def analyze_domain(resume_text):
    for domain in job_roles:
        if domain.lower() in resume_text.lower():
            return domain
    return "General"
```

This simple yet effective approach matches resume content against predefined domains, allowing for accurate specialization detection.

### 6.3 Question Generation

Rachel uses the Llama model to generate tailored interview questions:

```python
def generate_hr_questions(domain, job_role, job_description):
    prompt = f"Generate 5 high-quality Technical HR interview questions for a candidate specializing in {domain} for the role of {job_role} with the following job description:\n{job_description}\nFocus on advanced concepts and industry best practices."
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )
    questions = response['choices'][0]['message']['content'].strip().split('\n')
    return [q.strip() for q in questions if q.strip()]
```

This function crafts a prompt using the candidate's domain, job role, and job description, then uses the Llama model to generate relevant technical questions.

### 6.4 Answer Evaluation

The `provide_feedback` function employs a sophisticated algorithm to evaluate user answers:

```python
def provide_feedback(question, user_answer, expected_answer):
    user_answer_lower = user_answer.lower()
    expected_answer_lower = expected_answer.lower()
    question_lower = question.lower()

    user_keywords = set(extract_keywords_textrank(user_answer_lower))
    expected_keywords = set(extract_keywords_textrank(expected_answer_lower))
    question_keywords = set(extract_keywords_textrank(question_lower))

    relevant_keywords = question_keywords.intersection(expected_keywords)
    user_relevant_keywords = user_keywords.intersection(relevant_keywords)
    keyword_relevance = len(user_relevant_keywords) / len(relevant_keywords) if relevant_keywords else 0

    tfidf_matrix = tfidf_vectorizer.fit_transform([user_answer_lower, expected_answer_lower])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    final_score = (0.6 * keyword_relevance + 0.4 * cosine_sim) * 10
    rating = round(final_score)

    # ... (rating-based feedback generation)

    return rating, suggestions + [feedback_details]
```

This function combines keyword analysis and TF-IDF cosine similarity to provide a comprehensive evaluation of the user's answer, generating both a numerical rating and constructive feedback.

### 6.5 GPU Acceleration

Rachel utilizes CUDA for improved performance:

```python
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
device = torch.device("cuda")
torch.cuda.set_device(0)  # Use the first GPU
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Initialize the Llama model with CUDA support
llm = Llama.from_pretrained(
    repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
    filename="llama-3.2-3b-instruct-q8_0.gguf",
    n_gpu_layers=-1,  # Use all GPU layers
    n_ctx=2048,  # Adjust context size as needed
    device=device
)
```

This setup ensures that the Llama model and other computations take full advantage of GPU acceleration, significantly improving processing speed.

### 6.6 User Interface

Rachel's user interface is built using Gradio, providing a clean and intuitive experience:

```python
with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🎓 KITS - Interview Prep Bot")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Resume Analysis", open=False):
                file_input = gr.File(label="📄 Upload your resume (PDF)", file_types=['pdf'])
                upload_button = gr.Button("📤 Upload and Analyze Resume")
                upload_status = gr.Textbox(label="Status")
                detected_domain = gr.Textbox(label="🎯 Detected Specialization")
                job_role_dropdown = gr.Dropdown(label="🔍 Select Job Role", choices=[])
                job_description_input = gr.Textbox(label="📋 Enter Job Description (max 200 words)", max_lines=10)
            
            generate_button = gr.Button("🔄 Generate Questions", elem_classes=["generate-btn"])
            feedback_button = gr.Button("📝 Provide Feedback", elem_classes=["feedback-btn"])

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="💬 Chat")
            chat_input = gr.Textbox(label="Type your answer", placeholder="Type here or click 'Skip' to proceed")
            with gr.Row():
                chat_button = gr.Button("📨 Send")
                skip_button = gr.Button("🔄 Skip")
                generate_answer_button = gr.Button("💡 Generate Answer")

    # ... (state variables and event handlers)
```

This code structure creates a responsive layout with collapsible sections, stylized buttons, and a central chat interface, enhancing user experience and accessibility.

## 7. Contributing

We welcome contributions to Rachel HR Interview Bot! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please ensure your code adheres to the project's coding standards and include appropriate tests for new features.

## 8. License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 9. Acknowledgments

- Sam Naveenkumar V (URK22AI1043) and Aravindan M (URK22AI1026) for their innovative development of Rachel HR Interview Bot.
- Karunya Institute of Technology and Sciences for supporting this project.
- The open-source community for providing the foundational libraries and models used in this project.

---

Rachel HR Interview Bot represents a significant advancement in AI-assisted interview preparation. By combining cutting-edge NLP techniques, GPU acceleration, and a user-friendly interface, Rachel offers a comprehensive solution for candidates looking to excel in technical HR interviews. We hope this tool proves invaluable in your career journey!
