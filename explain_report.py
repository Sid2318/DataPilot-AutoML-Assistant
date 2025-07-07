# explain_report.py
import google.generativeai as genai
import os

# Initialize Gemini client with API key
def get_gemini_client():
    # Try environment variable first
    genai.configure(api_key="your api key")

    # 🔮 Load Gemini model
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    return model

def explain_report(report_text, target_column, problem_type, best_model_name):
    model = get_gemini_client()
    if not model:
        return "⚠️ **API Key Issue**: `GOOGLE_API_KEY` environment variable not found."
    
    try:
        # The f-string will automatically use the function arguments
        prompt = f"""
        You are an expert and friendly AI data scientist. Your goal is to explain a complex AutoML report to a non-technical user in a simple, encouraging, and easy-to-understand way.

        Structure the explanation like a story of what the user did and what you (the AI) found. Use the following structure:

        ---

        ### 🤖 Here's a summary of what you asked me to do:

        *   **Your Goal:** You gave me a dataset and asked me to figure out how to predict the **`{target_column}`**.
        *   **The Problem Type:** I analyzed your data and determined this is a **`{problem_type}`** task.
            *   *(Explain what this means in one simple sentence. For example, for classification: "This means we're trying to predict which category something belongs to." For regression: "This means we're trying to predict a specific number.")*

        ### 🏆 Here's what I found after testing different AI models:

        *   **The Contest:** I trained several different types of AI models to see which one was the best at predicting your target. Think of it like a competition to see which model is the smartest for your specific data.
        *   **The Winner:** The **`{best_model_name}`** model was the clear winner!
        *   **Why it Won:** It was the best because it had the highest performance score. *(Explain the main performance metric in simple terms. For example: "This score means it was the most accurate at making correct predictions on new data it hadn't seen before.")*

        ### 🧠 How the Winning Model Works (in simple terms):

        *   Explain how the winning model makes predictions metaphorically. For example, for a Random Forest: "Imagine it's like asking hundreds of experts a question and then taking the most popular answer. This makes its predictions very robust and reliable."

        ### ✨ My Recommendation:

        *   Based on the results, I strongly recommend using the **`{best_model_name}`** model. It's the most effective and reliable for your goal. You can now use this model to make predictions on new data!

        ---

        Use plenty of emojis, bold text for key terms, and a friendly, encouraging tone throughout.

        formate this in report style with headings, bullet points, and clear sections.

        Here is the report to analyze: {report_text}
        """
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        return f"❌ **Unexpected Error:** {error_msg}"