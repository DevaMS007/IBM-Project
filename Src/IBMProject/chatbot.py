import gradio as gr
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF for PDF reading
import re
import os

# Hugging Face access token
HF_TOKEN = "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"

# Model names
MODEL_IBM_GRANITE = "ibm-granite/granite-3.1-8b-instruct"
MODEL_GPT2 = "gpt2"

# Inference client (shared for both models)
client = InferenceClient(token=HF_TOKEN)

# Session state
session = {
    "content": "",
    "questions": [],
    "answers": [],
    "correct_answers": [],
    "feedbacks": [],
    "current_q_index": 0,
    "quiz_started": False,
    "model_name": MODEL_IBM_GRANITE
}

# ✅ File reading function (handles both path and file object)
def read_file(file_obj):
    try:
        # Case 1: Gradio returns file path (string)
        if isinstance(file_obj, str):
            if file_obj.endswith(".txt"):
                with open(file_obj, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_obj.endswith(".pdf"):
                with open(file_obj, "rb") as f:
                    doc = fitz.open(stream=f.read(), filetype="pdf")
                    return "\n".join(page.get_text() for page in doc)
            else:
                return "Unsupported file format"

        # Case 2: File-like object (older Gradio)
        if file_obj.name.endswith(".txt"):
            return file_obj.read().decode("utf-8")
        elif file_obj.name.endswith(".pdf"):
            pdf_bytes = file_obj.read()
            doc = fitz.open("pdf", pdf_bytes)
            return "\n".join(page.get_text() for page in doc)
        else:
            return "Unsupported file format"
    except Exception as e:
        return f"Error reading file: {e}"

# ✅ Unified model calling
def call_model_api(prompt, model_name, max_tokens=200):
    try:
        if model_name == MODEL_IBM_GRANITE:
            response = client.chat_completion(model=model_name, messages=[{"role": "user", "content": prompt}])
            content = response.choices[0].message.content.strip()
            return content
        else:
            response = client.text_generation(model=model_name, inputs=prompt, max_new_tokens=max_tokens)
            content = response.generated_text.strip()
            return content
    except Exception as e:
        return f"Error calling model: {e}"

# ✅ Generate all quiz questions
def generate_all_questions(content, model_name):
    prompt = f"Here is study content:\n{content[:2000]}\n\nGenerate 3 quiz questions."
    output = call_model_api(prompt, model_name, max_tokens=300)

    if model_name == MODEL_IBM_GRANITE:
        questions = [q.strip() for q in output.split("\n") if q.strip()]
    else:
        questions = re.split(r"\d+\.\s+", output)
        questions = [q.strip() for q in questions if q.strip()]

    if not questions or len(questions) < 3:
        questions = [
            "What is the main idea of the content?",
            "List two important points mentioned.",
            "How can the knowledge be applied in practice?"
        ]

    return questions[:3]

def get_correct_answer(content, question, model_name):
    prompt = f"Based on:\n{content[:1500]}\nWhat is a correct answer to:\n{question}"
    return call_model_api(prompt, model_name, max_tokens=100)

def get_feedback(question, user_answer, correct_answer, model_name):
    prompt = (
        f"Question: {question}\n"
        f"User Answer: {user_answer}\n"
        f"Correct Answer: {correct_answer}\n\n"
        f"Give brief feedback on the user's answer."
    )
    return call_model_api(prompt, model_name, max_tokens=100)

# ✅ Quiz start
def start_quiz(file, selected_model):
    session["model_name"] = selected_model
    session["content"] = read_file(file)
    if session["content"].startswith("Error"):
        return "", "", session["content"]

    session["questions"] = generate_all_questions(session["content"], selected_model)
    session["answers"] = []
    session["correct_answers"] = []
    session["feedbacks"] = []
    session["current_q_index"] = 0
    session["quiz_started"] = True

    first_q = session["questions"][0] if session["questions"] else "No questions generated."
    return first_q, "", "Quiz started! Answer the question below 👇"

# ✅ Handle answer submission
def submit_answer(user_answer):
    idx = session["current_q_index"]
    question = session["questions"][idx]
    model_name = session["model_name"]

    correct = get_correct_answer(session["content"], question, model_name)
    feedback = get_feedback(question, user_answer, correct, model_name)

    session["answers"].append(user_answer)
    session["correct_answers"].append(correct)
    session["feedbacks"].append(feedback)

    session["current_q_index"] += 1

    if session["current_q_index"] < len(session["questions"]):
        next_q = session["questions"][session["current_q_index"]]
        return next_q, "", f"✅ Feedback: {feedback}"
    else:
        summary = "🎉 **Quiz Complete!** 🎉\n\n"
        for i in range(len(session["questions"])):
            summary += f"""
**Q{i+1}:** {session['questions'][i]}
**Your Answer:** {session['answers'][i]}
**Correct Answer:** {session['correct_answers'][i]}
**Feedback:** {session['feedbacks'][i]}

---
"""
        session["quiz_started"] = False
        return "🎓 Quiz Summary", "", summary

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📘 Study Quiz Bot (IBM Granite & GPT2 via Hugging Face)")

    with gr.Row():
        file_input = gr.File(label="Upload TXT or PDF")
        model_selector = gr.Dropdown(
            label="Select Model",
            choices=[MODEL_IBM_GRANITE, MODEL_GPT2],
            value=MODEL_IBM_GRANITE
        )
        start_btn = gr.Button("Start Quiz")

    question_box = gr.Textbox(label="Question", interactive=False, lines=3)
    user_answer = gr.Textbox(label="Your Answer", lines=2, placeholder="Type your answer here...")
    submit_btn = gr.Button("Submit Answer")
    feedback_box = gr.Markdown()

    start_btn.click(start_quiz, inputs=[file_input, model_selector], outputs=[question_box, user_answer, feedback_box])
    submit_btn.click(submit_answer, inputs=[user_answer], outputs=[question_box, user_answer, feedback_box])

demo.launch(share=True)
