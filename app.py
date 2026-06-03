import gradio as gr
from textblob import TextBlob
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
import csv
import os

flair_classifier = TextClassifier.load('en-sentiment')

# Define the button click event
def analyze_sentiment_with_textblob(text):
    textblob_result = TextBlob(text).sentiment.polarity
    if textblob_result > 0:
        textblob_result = f"Positive Sentiment (Polarity: {textblob_result:.2f})"
    elif textblob_result < 0:
        textblob_result = f"Negative Sentiment (Polarity: {textblob_result:.2f})"   
    else:        
        textblob_result = f"Neutral Sentiment (Polarity: {textblob_result:.2f})"
    return textblob_result

def analyze_sentiment_with_vader(text):
    vader_result = SentimentIntensityAnalyzer().polarity_scores(text)
    if vader_result['compound'] > 0.05:
        vader_result = f"Positive Sentiment (Compound: {vader_result['compound']:.2f})"
    elif vader_result['compound'] < -0.05:
        vader_result = f"Negative Sentiment (Compound: {vader_result['compound']:.2f})"
    else:
        vader_result = f"Neutral Sentiment (Compound: {vader_result['compound']:.2f})"
    return vader_result

def analyze_sentiment_with_flair(text):
    if not text.strip():
        return "Veuillez entrer du texte."
    sentence = Sentence(text)
    flair_classifier.predict(sentence)
    label = sentence.labels[0]
    return f"{label.value} Sentiment (Score: {label.score:.2f})"

FLAG_FILE = "flagged_reviews.csv"

def save_flagged_review(text, prediction, model):
    if not text or not text.strip():
        return "Nothing to flag : the text field is empty."
    file_exists = os.path.isfile(FLAG_FILE)
    with open(FLAG_FILE, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Text", "Model", "Prediction", "Timestamp"])
        writer.writerow([text, model, prediction, datetime.now().isoformat(timespec='seconds')])

    return "Review flagged and saved."

examples=[
    "I love this product! It's amazing.",
    "This is the worst thing I've ever bought.",
    "The weather is okay today.",
    "I'm not sure how I feel about this.",
    "This movie was fantastic! I enjoyed every moment.",
    "I hate this! It's terrible.",
    "The service was average, nothing special."
    ]


# Create the Gradio interface with a big app name, description, and examples
with gr.Blocks(theme=gr.Theme.from_hub("davehornik/Tealy")) as demo:
    # App Name and Description
    gr.Markdown("# Sentiment Analysis Application with different models")
    gr.Markdown("This app analyzes the sentiment of your text using different machine learning models. Enter your text below and click **Analyze** to see the results!")

    # Tabs for Demo and Documentation
    with gr.Tabs():
        # Tab 1: Demo App
        with gr.Tab("TextBlob Model"):
            gr.Markdown("## TextBlob Model")
            gr.Markdown("This is the demo for the TextBlob model.")
            # Input and Output Components
            input_text = gr.Textbox(label="Input Text", placeholder="Enter your text here...", lines=5)
            output_text = gr.Textbox(label="Sentiment Analysis Result", interactive=False)  

            # Buttons
            with gr.Row():
                submit_button = gr.Button("Analyze", variant="primary")
                clear_button = gr.Button("Clear", variant="secondary")
                flag_button = gr.Button("Flag Review", variant="stop")

            gr.Examples(
                examples=examples,  
                inputs=input_text,
                label="Try these examples!"
            )
            
            # Button Actions
            submit_button.click(analyze_sentiment_with_textblob, inputs=input_text, outputs=output_text)
            clear_button.click(fn=lambda: ("", ""), inputs=None, outputs=[input_text, output_text])
            flag_button.click(fn=lambda text, prediction: save_flagged_review(text, prediction, "TextBlob"), inputs=[input_text, output_text], outputs=output_text)

        # Tab 2: Vader Model
        with gr.Tab("Vader Model"):
            gr.Markdown("## Vader Model")
            gr.Markdown("This is the demo for the Vader model.")
            
            input_text_vader = gr.Textbox(label="Input Text", placeholder="Enter your text here...", lines=5)
            output_text_vader = gr.Textbox(label="Sentiment Analysis Result", interactive=False)   
            # Buttons
            with gr.Row():
                submit_button = gr.Button("Analyze", variant="primary")
                clear_button = gr.Button("Clear", variant="secondary")
                flag_button_vader = gr.Button("Flag Review", variant="stop")
                
            gr.Examples(
                examples=examples,  
                inputs=input_text_vader,
                label="Try these examples!"
            )
            
            # Button Actions
            submit_button.click(analyze_sentiment_with_vader, inputs=input_text_vader, outputs=output_text_vader)
            clear_button.click(fn=lambda: ("", ""), inputs=None, outputs=[input_text_vader, output_text_vader])
            flag_button_vader.click(fn=lambda text, prediction: save_flagged_review(text, prediction, "Vader"), inputs=[input_text_vader, output_text_vader], outputs=output_text_vader)

        # Tab 3: Flair Model
        with gr.Tab("Flair Model"):
            gr.Markdown("## Flair Model")
            gr.Markdown("This is the demo for the Flair model.")
            input_text_flair = gr.Textbox(label="Input Text", placeholder="Enter your text here...", lines=5)
            output_text_flair = gr.Textbox(label="Sentiment Analysis Result", interactive=False)
            # Buttons
            with gr.Row():
                submit_button = gr.Button("Analyze", variant="primary")
                clear_button = gr.Button("Clear", variant="secondary")
                flag_button_flair = gr.Button("Flag Review", variant="stop")
            gr.Examples(
                examples=examples,  
                inputs=input_text_flair,
                label="Try these examples!"
            )
            # Button Actions
            submit_button.click(analyze_sentiment_with_flair, inputs=input_text_flair, outputs=output_text_flair)
            clear_button.click(fn=lambda: ("", ""), inputs=None, outputs=[input_text_flair, output_text_flair])
            flag_button_flair.click(fn=lambda text, prediction: save_flagged_review(text, prediction, "Flair"), inputs=[input_text_flair, output_text_flair], outputs=output_text_flair)

        # Tab 4: Documentation
        with gr.Tab("Documentations"): #This is the name for the second tab "Documentation"
            gr.Markdown("## How It Works")
            gr.Markdown("This app uses different machine learning models to analyze the sentiment of the input text. Each model has its own strengths and weaknesses, and you can compare their results by using the different tabs.")
            gr.Markdown("### Features:")
            gr.Markdown("- **Multiple Models:** Compare the sentiment analysis results from TextBlob, Vader, and Flair models.")
            gr.Markdown("- **User-Friendly Interface:** Easily input your text and get instant sentiment analysis results.")
            gr.Markdown("- **Examples:** Try out the provided examples to see how the app works.")
            gr.Markdown("- **Flags:** If you find a review that is misclassified, you can flag it for further analysis. The flagged reviews are saved in a CSV file for review.")

# Launch the interface
demo.launch(debug=True) 