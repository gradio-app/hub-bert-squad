import gradio as gr
import sys
from bert import QA
model = QA('bert-large-uncased-whole-word-masking-finetuned-squad')

def qa_func(context, question):
    return model.predict(context, question)["answer"]

gr.Interface(qa_func, 
    [
        gr.inputs.Textbox(lines=7, label="Context"), 
        gr.inputs.Textbox(label="Question"), 
    ], 
    gr.outputs.Textbox(label="Answer"),
    title="Question Answer",
    description="BERT-SQuAD is a question answering model that takes 2 inputs: a paragraph that provides context and a question that should be answered. Takes around 6s to run.").launch()

