# Text Enhancement with Cosine Similarity

Welcome to my text enhancement journey, where we harness the power of cosine similarity to make text better!

## Getting Started

Let's kick things off by demystifying cosine similarity. At its core, it's a nifty technique that helps us compare phrases by converting them into special vectors.

## Embracing Pre-Trained Models

Instead of building a custom solution from scratch, I've opted for a time-saving approach. We're using the [SentenceTransformer library](https://www.sbert.net/docs/pretrained_models.html), a fantastic tool that effortlessly transforms phrases into vectors. There are lots of available pre-trained models differing in accuracy and speed, I have used most of them, and interestingly the best ones like `all-mpnet-base-v2`, `multi-qa-mpnet-base-cos-v1` showed lower cosine similarity scores compared to `bert-base-nli-mean-tokens` which considered to be giving lower quality embeddings. Nevertheless, everything depends on the threshold that you will give. SOTA sentence transformers give the same results as BERT, but I had to lower the threshold for the former. Thus, I decided to stay on `bert-base-nli-mean-tokens`, as there were not any significant differences. 
## The Challenge of Phrase Extraction

After conquering the conversion hurdle, the next mission is to find meaningful phrases within sentences and compare them with standardized ones. We tried using libraries like NLTK, but they didn't quite cut it. Here is the implementation of the NLTK:

```python
import nltk
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize
# Sample input text
input_text = """
In today's meeting, we discussed a variety of issues affecting our department.
"""
# Tokenize the input text into sentences
sentences = sent_tokenize(input_text)
# Initialize a list to store phrases
phrases = []
# Tokenize each sentence into words and join them to create phrases
for sentence in sentences:
    words = word_tokenize(sentence)
    phrases.extend([' '.join(words[i:i+2]) for i in range(len(words)-1)])
```
The output for the code is here:
```
['In today', "today 's", "'s meeting", 'meeting ,', ', we', 'we discussed', 'discussed a', 'a variety', 'variety of', 'of issues', 'issues affecting', 'affecting our', 'our department', 'department .']
```
It can be seen that NLTK did not catch all crucial phrases from this sentence. 
## The Power of Brute Force

So, I took a different path – a brute force strategy. It may sound intense, but it's surprisingly effective. We slide a window over sentences, extracting all possible phrases. To make our comparisons more efficient, we first calculate vectors for our standardized phrases. After some experimentation, we settled on a similarity score threshold of 0.8, which worked like a charm.

## See the Results
I have implemented a simple UI to put input text and phrases, and after pressing the `get suggestion` button, a new window with the answers will pop up.

Check out the image below. Of course, there's always room for improvement, like optimizing the time it takes to find phrases and creating a custom model for phrase-to-vector conversion.
![Screenshot from 2023-09-17 19-01-02](https://github.com/AbzalAidakhmetov/Text_Improvement_Engine/assets/99760649/d0962140-7ef4-4498-bf6a-8f997802161b)


In conclusion, I have successfully implemented sentence transformers to find similar phrases in the text with the standardized ones. There is a huge amount of implementations for this project like recommendation engine, text summarization, and many others.
