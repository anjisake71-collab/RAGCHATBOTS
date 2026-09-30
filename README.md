# GenAI Structured Data QA Assistant

This project allows users to ask natural language questions based on 3 structured CSVs (transactions, subscribers, and revenue). The assistant uses an LLM to convert the question into a Pandas query, then runs and displays the result.

## 🔧 Setup

1. Clone the repo
2. Place your CSV files in `data/`
3. Install dependencies:

```bash
pip install -r requirements.txt      
      