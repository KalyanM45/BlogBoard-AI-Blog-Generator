# BlogBoard — Autonomous AI Article Generator

<p align="center"> <img src="https://img.shields.io/github/license/KalyanM45/BlogBoard-AI-Blog-Generator?style=ROUND" alt="License" /> <img src="https://img.shields.io/github/stars/KalyanM45/BlogBoard-AI-Blog-Generator?style=ROUND" alt="Stars" /> <img src="https://img.shields.io/github/forks/KalyanM45/BlogBoard-AI-Blog-Generator?style=ROUND" alt="Forks" /> <img src="https://img.shields.io/github/issues/KalyanM45/BlogBoard-AI-Blog-Generator?style=ROUND"alt="Issues" />
<img src="https://img.shields.io/badge/Live%20Demo-Website-0A66C2?style=ROUND&logo=github&logoColor=white" />
</p>

## About The Project

BlogBoard is an end-to-end, fully automated blogging platform. It autonomously schedules, writes, formats, and publishes deep-dive technical articles on Machine Learning and Artificial Intelligence directly to a fast, static frontend website.

Powered by **LangGraph** for stateful workflow execution and **Groq** for blazing-fast LLM inference, it ensures that high-quality, zero-fluff, production-grade articles are generated and deployed automatically via **GitHub Actions**.

## Library Requirements

 - Python 3.12+
 - langgraph>=0.2.20
 - groq>=0.11.0
 - python-dotenv>=1.0.1
 - uv (for dependency management)

## Getting Started

This will help you understand how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Installation Steps

### Installation from GitHub

Follow these steps to install and set up the project directly from the GitHub repository:

1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to install the project.
   - Run the following command to clone the GitHub repository:
     ```bash
     git clone https://github.com/KalyanM45/BlogBoard-AI-Blog-Generator.git
     ```

2. **Create a Virtual Environment** (Recommended)
   - It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```bash
     uv venv
     ```

3. **Activate the Virtual Environment**
   - Activate the virtual environment based on your operating system:
       ```bash
       # On Linux/Mac:
       source .venv/bin/activate
       # On Windows:
       .venv\Scripts\activate
       ```

4. **Install Dependencies**
   - Navigate to the project directory:
     ```bash
     cd BlogBoard-AI-Blog-Generator
     ```
   - Run the following command to install project dependencies:
     ```bash
     uv pip install -r backend/requirements.txt
     ```

5. **Run the Project**
   - Start the backend pipeline by running the appropriate command:
     ```bash
     python backend/run.py
     ```

6. **Access the Project**
   - Serve the frontend locally using Python's built-in HTTP server:
     ```bash
     python -m http.server 8000 --directory frontend
     ```
   - Open a web browser and navigate to `http://localhost:8000`.


## API Key Setup

To use this project, you need an API key from Groq to power the Large Language Model inference. Follow these steps to obtain and set up your API key:

1. **Get API Key:**
   - Visit the Groq Console at [console.groq.com](https://console.groq.com/).
   - Follow the instructions to create an account and obtain your API key.

2. **Set Up API Key:**
   - Create a file named `.env` in the project root.
   - Add your API key to the `.env` file:
     ```dotenv
     GROQ_API_KEY=your_api_key_here
     ```

   **Note:** Keep your API key confidential. Do not share it publicly or expose it in your code.<br>

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

• **Report bugs**: If you encounter any bugs, please let us know. Open up an issue and let us know the problem.

• **Contribute code**: If you are a developer and want to contribute, follow the instructions below to get started!

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

• **Suggestions**: If you don't want to code but have some awesome ideas, open up an issue explaining some updates or improvements you would like to see!

#### Don't forget to give the project a star! Thanks again!

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](LICENSE) file for details.<br>

## Acknowledgements

We'd like to extend our gratitude to all individuals and organizations who have played a role in the development and success of this project. Your support, whether through contributions, inspiration, or encouragement, has been invaluable. Thank you for being a part of our journey.