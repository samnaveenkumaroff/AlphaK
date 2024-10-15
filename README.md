# Rachel HR Interview Bot

## Revolutionizing HR Interview Preparation with Advanced AI

![Rachel HR Interview Bot Logo](![user](https://github.com/user-attachments/assets/62422c8c-1db4-4559-bfc6-3995688538e9)
)
![user](https://github.com/user-attachments/assets/efadacd0-2bd7-48c2-a594-7add03631d90)

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Architecture](#architecture)
7. [AI Model Details](#ai-model-details)
8. [Performance Optimization](#performance-optimization)
9. [Future Enhancements](#future-enhancements)
10. [Contributors](#contributors)
11. [License](#license)

## Introduction

Rachel HR Interview Bot is a cutting-edge AI-powered application designed to revolutionize the HR interview preparation process for students, freshers, and experienced professionals. Developed by Sam Naveenkumar V (URK22AI1043) and Aravindan M (URK22AI1026) from the B.Tech Artificial Intelligence and Data Science program at Karunya Institute of Technology and Sciences, this project leverages state-of-the-art natural language processing techniques to provide a personalized and immersive interview experience.

The bot employs advanced algorithms to analyze resumes, identify domains of specialization, and generate complex technical HR interview questions tailored to each candidate's unique profile. By utilizing the latest open-sourced language models and a user-friendly interface built with OpenWeb UI, Rachel aims to bridge the gap between theoretical knowledge and practical interview scenarios.

## Features

- **Resume Analysis**: Utilizes PyPDF2 for extracting text from PDF resumes and employs NLP techniques to identify key information such as specialization, work experience, internships, projects, and courses completed.

- **Domain Identification**: Automatically detects the candidate's domain of specialization based on resume content, enabling targeted question generation.

- **Dynamic Question Generation**: Leverages the Llama 3.2 language model to create complex, technical HR interview questions tailored to the candidate's background and the specific job role.

- **Interactive Chat Interface**: Provides a WhatsApp-like chat experience for seamless interaction between the candidate and the AI interviewer.

- **Real-time Answer Evaluation**: Employs advanced NLP techniques, including TF-IDF vectorization and cosine similarity, to evaluate candidate responses and provide instant feedback.

- **Comprehensive Feedback System**: Offers detailed feedback on each answer, including relevance, technical accuracy, and suggestions for improvement.

- **Answer Generation**: Capability to generate model answers for reference, helping candidates understand the expected response quality.

- **Job Role and Description Integration**: Allows users to select specific job roles and input job descriptions, further personalizing the interview experience.

- **GPU Acceleration**: Utilizes CUDA for enhanced performance, enabling rapid question generation and response evaluation.

## Technology Stack

- **Frontend**: OpenWeb UI (enhanced for user experience)
- **Backend**: Python 3.8+
- **AI Model**: Llama 3.2 (3B parameters, instruction-tuned)
- **NLP Libraries**: spaCy, PyTextRank
- **Machine Learning**: scikit-learn (TfidfVectorizer, cosine_similarity)
- **PDF Processing**: PyPDF2
- **GPU Acceleration**: PyTorch with CUDA support
- **UI Framework**: Gradio

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU
- Git

### Setting up the environment

#### Windows
```bash
python -m venv rachel_env
rachel_env\Scripts\activate
pip install -r requirements.txt
```

#### macOS and Linux
```bash
python3 -m venv rachel_env
source rachel_env/bin/activate
pip install -r requirements.txt
```

### Cloning the repository
```bash
git clone https://github.com/your-repo/rachel-hr-bot.git
cd rachel-hr-bot
```

### Installing dependencies
```bash
pip install -r requirements.txt
```

### Downloading the Llama 3.2 model
```bash
# Run the following command to download the model
python download_model.py
```

## Usage

1. Activate the virtual environment:
   - Windows: `rachel_env\Scripts\activate`
   - macOS/Linux: `source rachel_env/bin/activate`

2. Run the application:
   ```bash
   python rachel_hr_bot.py
   ```

3. Open your web browser and navigate to `http://localhost:7860` to access the Rachel HR Interview Bot interface.

4. Upload your resume, select a job role, and enter the job description to begin the interview process.

## Architecture

Rachel HR Interview Bot follows a modular architecture designed for scalability and maintainability:

1. **Resume Processing Module**: Handles PDF extraction and initial text analysis.
2. **Domain Identification Module**: Classifies the resume into relevant domains.
3. **Question Generation Engine**: Interfaces with the Llama 3.2 model to create tailored questions.
4. **Answer Evaluation System**: Utilizes NLP techniques to assess candidate responses.
5. **Feedback Generation Module**: Synthesizes evaluation results into actionable feedback.
6. **User Interface Layer**: Manages the OpenWeb UI for user interactions.
7. **GPU Acceleration Layer**: Optimizes performance using CUDA capabilities.

## AI Model Details

Rachel utilizes the Llama 3.2 model, specifically the 3B parameter version optimized for instruction-tuning. Key characteristics include:

- **Model**: hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF
- **Size**: 3 billion parameters
- **Training**: Instruction-tuned for dialogue and task completion
- **Capabilities**: Multilingual support, agentic retrieval, and summarization
- **Performance**: Outperforms many open-source and closed chat models on industry benchmarks
- **Deployment**: Optimized for efficient inference using quantization techniques

## Performance Optimization

To ensure smooth operation and rapid response times, Rachel implements several optimization techniques:

1. **GPU Acceleration**: Utilizes CUDA for parallel processing of NLP tasks.
2. **Model Quantization**: Employs 8-bit quantization (Q8_0) to reduce memory footprint without significant performance loss.
3. **Batched Processing**: Implements batched inputs for efficient utilization of GPU resources.
4. **Caching Mechanisms**: Employs strategic caching of intermediate results to minimize redundant computations.
5. **Asynchronous Operations**: Leverages asynchronous programming patterns to improve UI responsiveness.

## Future Enhancements

The Rachel HR Interview Bot project has a robust roadmap for future development, including:

1. Integration of multi-modal inputs (video, audio) for more comprehensive interview simulations.
2. Implementation of sentiment analysis to gauge candidate confidence and stress levels.
3. Expansion of the model to include industry-specific knowledge bases.
4. Development of a collaborative feature for mock group interviews.
5. Integration with popular Applicant Tracking Systems (ATS) for seamless workflow incorporation.

## Contributors

- **Sam Naveenkumar V** (URK22AI1043)
  - Role: Lead Developer
  - Focus: AI Model Integration, NLP Pipeline Development

- **Aravindan M** (URK22AI1026)
  - Role: UI/UX Designer, Backend Developer
  - Focus: OpenWeb UI Enhancement, Performance Optimization

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

For more information, please contact the development team at rachel.hr.bot@karunyauniversity.edu.in

© 2024 Karunya Institute of Technology and Sciences. All Rights Reserved.
