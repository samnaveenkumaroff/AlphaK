# Rachel HR Interview Bot
<img src="https://github.com/user-attachments/assets/fc5e1b56-9794-440e-b6ef-286ef199a2e8" alt="Rachel HR Interview Bot" width="300" height="300">

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
   - [Installation via Python pip](#installation-via-python-pip-)
   - [Quick Start with Docker](#quick-start-with-docker-)
   - [Other Installation Methods](#other-installation-methods)
   - [Troubleshooting](#troubleshooting)
   - [Keeping Your Docker Installation Up-to-Date](#keeping-your-docker-installation-up-to-date)
5. [Usage](#usage)
6. [Technical Details](#technical-details)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)
    
## 1. Introducing

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

### Technical Architecture

Rachel HR Interview Bot is built on a robust and scalable architecture, leveraging cutting-edge technologies:

```
+-------------------+     +------------------+     +------------------+
|   User Interface  |     |  Core Logic      |     |  AI Engine       |
| (Gradio Frontend) | <-> | (Python Backend) | <-> | (Llama 3.2 Model)|
+-------------------+     +------------------+     +------------------+
         ^                        ^                         ^
         |                        |                         |
         v                        v                         v
+-------------------+     +------------------+     +------------------+
|   PDF Processor   |     | NLP Pipeline     |     |  GPU Accelerator |
|   (PyPDF2)        |     | (spaCy, TextRank)|     |  (CUDA)          |
+-------------------+     +------------------+     +------------------+
```

## 4. Installation

### Installation via Python pip 🐍

Open WebUI can be installed using pip, the Python package installer. Before proceeding, ensure you're using **Python 3.11** to avoid compatibility issues.

1. **Install Open WebUI**:
   Open your terminal and run the following command to install Open WebUI:

   ```bash
   pip install open-webui
   ```

2. **Running Open WebUI**:
   After installation, you can start Open WebUI by executing:

   ```bash
   open-webui serve
   ```

This will start the Open WebUI server, which you can access at [http://localhost:8080](http://localhost:8080)

### Quick Start with Docker 🐳

> [!NOTE]  
> Please note that for certain Docker environments, additional configurations might be needed. If you encounter any connection issues, our detailed guide on [Open WebUI Documentation](https://docs.openwebui.com/) is ready to assist you.

> [!WARNING]
> When using Docker to install Open WebUI, make sure to include the `-v open-webui:/app/backend/data` in your Docker command. This step is crucial as it ensures your database is properly mounted and prevents any loss of data.

> [!TIP]  
> If you wish to utilize Open WebUI with Ollama included or CUDA acceleration, we recommend utilizing our official images tagged with either `:cuda` or `:ollama`. To enable CUDA, you must install the [Nvidia CUDA container toolkit](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/) on your Linux/WSL system.

#### Installation with Default Configuration

- **If Ollama is on your computer**, use this command:


  ```bash
  docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```

- **If Ollama is on a Different Server**, use this command:

  To connect to Ollama on another server, change the `OLLAMA_BASE_URL` to the server's URL:

  ```bash
  docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=https://example.com -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```

- **To run Open WebUI with Nvidia GPU support**, use this command:

  ```bash
  docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda
  ```

#### Installation for OpenAI API Usage Only

- **If you're only using OpenAI API**, use this command:

  ```bash
  docker run -d -p 3000:8080 -e OPENAI_API_KEY=your_secret_key -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```

#### Installing Open WebUI with Bundled Ollama Support

This installation method uses a single container image that bundles Open WebUI with Ollama, allowing for a streamlined setup via a single command. Choose the appropriate command based on your hardware setup:

- **With GPU Support**:
  Utilize GPU resources by running the following command:

  ```bash
  docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama
  ```

- **For CPU Only**:
  If you're not using a GPU, use this command instead:

  ```bash
  docker run -d -p 3000:8080 -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama
  ```

Both commands facilitate a built-in, hassle-free installation of both Open WebUI and Ollama, ensuring that you can get everything up and running swiftly.

After installation, you can access Open WebUI at [http://localhost:3000](http://localhost:3000). Enjoy! 😄

### Other Installation Methods

We offer various installation alternatives, including non-Docker native installation methods, Docker Compose, Kustomize, and Helm. Visit our [Open WebUI Documentation](https://docs.openwebui.com/getting-started/) or join our [Discord community](https://discord.gg/5rJgQTnV4s) for comprehensive guidance.

### Troubleshooting

Encountering connection issues? Our [Open WebUI Documentation](https://docs.openwebui.com/troubleshooting/) has got you covered. For further assistance and to join our vibrant community, visit the [Open WebUI Discord](https://discord.gg/5rJgQTnV4s).

#### Open WebUI: Server Connection Error

If you're experiencing connection issues, it's often due to the WebUI docker container not being able to reach the Ollama server at 127.0.0.1:11434 (host.docker.internal:11434) inside the container. Use the `--network=host` flag in your docker command to resolve this. Note that the port changes from 3000 to 8080, resulting in the link: `http://localhost:8080`.

**Example Docker Command**:

```bash
docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

### Keeping Your Docker Installation Up-to-Date

In case you want to update your local Docker installation to the latest version, you can do it with [Watchtower](https://containrrr.dev/watchtower/):

```bash
docker run --rm --volume /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower --run-once open-webui
```

In the last part of the command, replace `open-webui` with your container name if it is different.

Check our Migration Guide available in our [Open WebUI Documentation](https://docs.openwebui.com/tutorials/migration/).

### Ollama Docker image

#### CPU only

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### Nvidia GPU
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

##### Install with Apt
1.  Configure the repository
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
```
2.  Install the NVIDIA Container Toolkit packages
```bash
sudo apt-get install -y nvidia-container-toolkit
```

##### Install with Yum or Dnf
1.  Configure the repository

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```

2. Install the NVIDIA Container Toolkit packages

```bash
sudo yum install -y nvidia-container-toolkit
```

##### Configure Docker to use Nvidia driver
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

##### Start the container

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### AMD GPU

To run Ollama using Docker with AMD GPUs, use the `rocm` tag and the following command:

```
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

#### Run model locally

Now you can run a model:

```
docker exec -it ollama ollama run nemotron:70b-instruct-fp16
```

#### Try different models

More models can be found on the [Ollama library](https://ollama.com/library).

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
