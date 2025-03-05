# US F1 Visa Interview Simulator

This application is a Streamlit-based Visa Interview Simulator built to help users practice their US F1 Visa interviews. It uses GPT-4o-mini powered by OpenAI, Pinecone for vector-based search, and various other technologies to generate realistic interview questions, assess answers, and provide feedback.

### Live Demo
You can interact with the live demo [here](https://visa-simulator.streamlit.app).

## Features
- **Customizable Interview Setup:** Users can input personal details (e.g., name, age, nationality, university) to tailor the simulation.
- **Dynamic Interview Flow:** Based on the selected interview type (Beginner, Intermediate, Advanced), the simulator adjusts the number of questions.
- **AI-Powered Q&A:** Uses GPT-4o-mini to generate and answer questions based on the input provided.
- **Feedback System:** After completing the interview, users can receive detailed feedback on their performance.
- **Vector-based Search:** A custom Pinecone-backed vector database helps to fetch interview-related data and categorize questions.
- **Chat History and Memory:** All responses are stored, and past conversations are used to improve subsequent interactions.

## Installation

To set up the project locally, follow the steps below:

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/visa-interview-simulator.git
cd visa-interview-simulator
```

### 2. Install dependencies:
Install the required Python packages:
```bash
pip install -r requirements.txt
```

- **Streamlit**: To build the interactive web app.
- **OpenAI API**: For GPT-4o-mini powered responses.
- **Pinecone**: For managing vector-based search and storage.
- **Langchain**: For chaining prompts and managing conversational memory.
- **Sentence Transformers**: For generating embeddings from interview-related data.
- **Datasets**: To load and work with external datasets.
  
### 3. Set up your `.streamlit/secrets.toml` file:
You need to store your OpenAI and Pinecone API keys in a secrets file to authenticate with the APIs.

Create a `.streamlit/secrets.toml` file:
```toml
[general]
OPENAI_API_KEY = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "your-pinecone-environment"
TEMPLATE_OFFICER_ONE = "Your predefined officer template 1"
TEMPLATE_OFFICER_TWO = "Your predefined officer template 2"
TEMPLATE_FEEDBACK = "Your feedback template"
```

### 4. Run the application:
To start the simulator, use the following command:
```bash
streamlit run app.py
```

### 5. Access the application:
Once the app is running, open the link provided by Streamlit (typically `http://localhost:8501`) in your browser.

## How it Works

### 1. **Interview Setup**
   - The user is prompted to enter personal details such as their name, age, nationality, university, and funding method.
   - Based on these details, a personalized interview scenario is created.

### 2. **Interview Flow**
   - The user is asked a series of questions according to the interview level they select (Beginner, Intermediate, or Advanced).
   - The AI generates responses to the user’s input and provides related interview questions, fetched from a Pinecone vector database.

### 3. **Feedback**
   - After completing the interview, the user can request feedback.
   - The feedback is generated based on the conversation and provided to the user in a detailed format.

## Technologies Used
- **Streamlit**: A framework to quickly build and share interactive web applications.
- **OpenAI GPT-4o-mini**: For generating AI responses and simulating the interview.
- **Pinecone**: A vector database for storing and querying data related to interview categories.
- **Langchain**: For managing conversational chains and prompts.
- **Sentence Transformers**: For embedding questions and answers in a vector space.
- **Datasets**: To work with external datasets for question-answer generation.

## Contributing
If you’d like to contribute to this project, feel free to fork the repository and submit pull requests. Please follow standard GitHub workflows.

### Issues
If you encounter any bugs or issues, please feel free to open an issue in the repository.

## License
This project is open-source and licensed under the [MIT License](LICENSE).

## Acknowledgements
- **OpenAI GPT-4o-mini**: For natural language understanding and generation.
- **Pinecone**: For vector search and database management.
- **Langchain**: For building efficient chains and prompts in natural language tasks.
- **Streamlit**: For creating interactive web applications effortlessly.

---

*This project is designed to simulate a US F1 Visa interview and help users practice and prepare for the real interview.* 
