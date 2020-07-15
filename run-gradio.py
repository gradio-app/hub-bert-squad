import gradio as gr
import sys
from bert import QA
model = QA('bert-large-uncased-whole-word-masking-finetuned-squad')

def qa_func(context, question):
    return model.predict(context, question)["answer"]

samples = [
    ["Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, depending on the provision.",
     "When did Victoria enact its constitution?"],
    ["Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others. Robotics deals with the design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing. These technologies are used to develop machines that can substitute for humans. Robots can be used in any situation and for any purpose, but today many are used in dangerous environments (including bomb detection and de-activation), manufacturing processes, or where humans cannot survive. Robots can take on any form but some are made to resemble humans in appearance. This is said to help in the acceptance of a robot in certain replicative behaviors usually performed by people. Such robots attempt to replicate walking, lifting, speech, cognition, and basically anything a human can do.",
     "What do robots that resemble humans attempt to do?"],
    ["We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.",
     "What does BERT stand for?"]
]

gr.Interface(qa_func, 
    [
        gr.inputs.Textbox(lines=7, label="Context"), 
        gr.inputs.Textbox(label="Question"), 
    ], 
    gr.outputs.Textbox(label="Answer"),
    title="Ask Me Anything",
    description="This model, BERT-SQuAD, is a question answering model that takes 2 inputs: a paragraph that provides context and a question that should be answered.",
    thumbnail="",
    examples=samples).launch()

