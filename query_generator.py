from openai import OpenAI

def get_query_from_llm(user_question, prompt_template, api_key):
    """
    Generate SQL query from natural language question using OpenAI
    """
    # Initialize client with provided API key
    client = OpenAI(api_key=api_key)
    
    # Replace placeholder with actual question
    prompt = prompt_template.replace("<<QUESTION>>", user_question)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for better results
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate only valid SQL queries without explanations unless asked."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # For consistent results
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        raise Exception(f"LLM Error: {str(e)}")              