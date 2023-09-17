from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import sent_tokenize, word_tokenize
from tkinter import *
from tkinter import scrolledtext, messagebox

# Load the SentenceTransformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Helper function to extract phrases from a sentence with a given window size
def extract_phrases(sentence, window_size):
    words = word_tokenize(sentence)
    return [' '.join(words[i:i + window_size]) for i in range(len(words) - window_size + 1)]

# Function to get suggestions
def get_suggestions():
    # Input text and standardized phrases from UI
    input_text = input_box.get("1.0", 'end-1c')
    standardized_phrases = [phrase.strip() for phrase in phrases_box.get("1.0", 'end-1c').split('\n') if phrase]

    sentences = sent_tokenize(input_text)
    most_similar_phrases = {}

    # Encode the standardized phrases
    encoded_phrases = model.encode(standardized_phrases)

    # Iterate through sentences and find the most similar phrases
    for i, sentence in enumerate(sentences):
        top_choices = {}
        for window_size in range(1, len(sentence)):
            phrases = extract_phrases(sentence, window_size)
            for phrase in phrases:
                cosine_similarities = cosine_similarity(model.encode([phrase]), encoded_phrases)
                most_similar_index = cosine_similarities[0].argmax()
                current_similarity = cosine_similarities[0, most_similar_index]
                if current_similarity > 0.8:
                    most_similar_phrase = standardized_phrases[most_similar_index]
                    if most_similar_phrase not in top_choices or current_similarity > top_choices[most_similar_phrase][0]:
                        top_choices[most_similar_phrase] = [current_similarity, phrase]
        most_similar_phrases[i] = top_choices


    # Output the suggestions
    output_text = ""
    for i, sentence in enumerate(sentences):
        output_text += f"Sentence {i + 1}: {sentence}\n"
        suggestions_for_sentence = most_similar_phrases[i]
        for replacement, (similarity_score, original_phrase) in suggestions_for_sentence.items():
            output_text += f"Original: '{original_phrase}', Recommended: '{replacement}', Similarity Score: {similarity_score}\n"
        output_text += '\n'
    
    output_window = Toplevel(root)
    output_window.title("Suggestions")
    output_box = scrolledtext.ScrolledText(output_window, wrap=WORD, width=100, height=30)
    output_box.insert(INSERT, output_text)
    output_box.pack(padx=10, pady=10)

# UI setup using tkinter
root = Tk()
root.title("Text Suggestion Tool")

Label(root, text="Enter Text:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
input_box = scrolledtext.ScrolledText(root, wrap=WORD, width=70, height=10)
input_box.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

Label(root, text="Enter Standardized Phrases (one per line):").grid(row=2, column=0, sticky=W, padx=5, pady=5)
phrases_box = scrolledtext.ScrolledText(root, wrap=WORD, width=70, height=10)
phrases_box.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

Button(root, text="Get Suggestions", command=get_suggestions).grid(row=4, column=0, columnspan=2, pady=20)

root.mainloop()