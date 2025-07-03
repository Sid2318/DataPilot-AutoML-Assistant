import google.generativeai as genai

# 🔑 Set your Gemini API key here
genai.configure(api_key="ur key here")

# 🔮 Load Gemini model
model = genai.GenerativeModel("models/gemini-2.0-flash")

# 💬 Send a prompt
response = model.generate_content("Tell me a joke about robots.")

# 📢 Print the response
print(response.text)
