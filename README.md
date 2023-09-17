# Text Enhancement with Cosine Similarity

Welcome to my text enhancement journey, where we harness the power of cosine similarity to make text better!

## Getting Started

Let's kick things off by demystifying cosine similarity. At its core, it's a nifty technique that helps us compare phrases by converting them into special vectors.

## Embracing Pre-Trained Models

Instead of building a custom solution from scratch, I've opted for a time-saving approach. I am using the [SentenceTransformer library](https://www.sbert.net/docs/pretrained_models.html), a fantastic tool that effortlessly transforms phrases into vectors. There are lots of available pre-trained models differing in accuracy and speed, I have used most of them, and interestingly the best ones like `all-mpnet-base-v2`, `multi-qa-mpnet-base-cos-v1` showed lower cosine similarity scores compared to `bert-base-nli-mean-tokens` which considered to be giving lower quality embeddings. Nevertheless, everything depends on the threshold that you will give. SOTA sentence transformers gave almost the same results as BERT, but I had to lower the threshold for the former. Thus, I decided to show both `bert-base-nli-mean-tokens` and `multi-qa-mpnet-base-cos-v1` results.
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

So, I took a different path – a brute force strategy. It may sound intense, but it's surprisingly effective. We slide a window over sentences, extracting all possible phrases. To make our comparisons more efficient, I first calculate vectors for  standardized phrases. After some experimentation, I settled on a similarity score threshold of 0.8 for `bert-base-nli-mean-tokens` and 0.5 for `multi-qa-mpnet-base-cos-v1`, which worked like a charm.

## See the Results
I have implemented a simple UI to put input text and phrases, and after pressing the `Get Suggestions` button, a new window with the answers will pop up.

Check out the image below. Of course, there's always room for improvement, like optimizing the time it takes to find phrases and creating a custom model for phrase-to-vector conversion.
![Screenshot from 2023-09-17 19-01-02](https://github.com/AbzalAidakhmetov/Text_Improvement_Engine/assets/99760649/d0962140-7ef4-4498-bf6a-8f997802161b)

Here is the full answer for the sample text with sample standardized phrases for `bert-base-nli-mean-tokens` with 0.8 threshold:
```
Sentence 1: In today's meeting, we discussed a variety of issues affecting our department.
Original: 'we discussed a', Recommended: 'Conduct an analysis', Similarity Score: 0.8026463389396667

Sentence 2: The weather was unusually sunny, a pleasant backdrop to our serious discussions.
Original: 'pleasant', Recommended: 'Optimal performance', Similarity Score: 0.812814474105835

Sentence 3: We came to the consensus that we need to do better in terms of performance.
Original: 'better in terms of performance .', Recommended: 'Optimal performance', Similarity Score: 0.8266074657440186
Original: 'terms of performance', Recommended: 'Monitor performance metrics', Similarity Score: 0.8337692022323608
Original: 'to the consensus that we need to do', Recommended: 'Ensure compliance', Similarity Score: 0.8177525997161865

Sentence 4: Sally brought doughnuts, which lightened the mood.

Sentence 5: It's important to make good use of what we have at our disposal.
Original: 'make good use', Recommended: 'Optimal performance', Similarity Score: 0.894195556640625
Original: 'use of what we have at our disposal', Recommended: 'Utilize resources', Similarity Score: 0.834140419960022
Original: 'to make good use of what we', Recommended: 'Implement best practices', Similarity Score: 0.8612539768218994
Original: 'what we have at our disposal .', Recommended: 'Streamline operations', Similarity Score: 0.8227188587188721
Original: 'use of what we have at our', Recommended: 'Prioritize tasks', Similarity Score: 0.8033173084259033
Original: 'to make good use of what we have at', Recommended: 'Facilitate collaboration', Similarity Score: 0.8510632514953613

Sentence 6: During the coffee break, we talked about the upcoming company picnic.

Sentence 7: We should aim to be more efficient and look for ways to be more creative in our daily tasks.
Original: 'efficient', Recommended: 'Optimal performance', Similarity Score: 0.890781819820404
Original: 'creative in our', Recommended: 'Foster innovation', Similarity Score: 0.8224096298217773
Original: 'tasks', Recommended: 'Utilize resources', Similarity Score: 0.8649134635925293
Original: 'for ways to be', Recommended: 'Execute strategies', Similarity Score: 0.8356931209564209
Original: 'for ways', Recommended: 'Leverage synergies', Similarity Score: 0.81622314453125
Original: 'be more efficient and', Recommended: 'Enhance productivity', Similarity Score: 0.8472462892532349

Sentence 8: Growth is essential for our future, but equally important is building strong relationships with our team members.
Original: 'Growth', Recommended: 'Drive growth', Similarity Score: 0.9176158905029297
Original: 'strong', Recommended: 'Maintain a high standard', Similarity Score: 0.8209724426269531
Original: 'is essential', Recommended: 'Ensure compliance', Similarity Score: 0.8022600412368774
Original: 'equally important', Recommended: 'Optimal performance', Similarity Score: 0.8187180757522583
Original: 'essential for our', Recommended: 'Implement best practices', Similarity Score: 0.8004742860794067

Sentence 9: As a reminder, the annual staff survey is due next Friday.
Original: 'survey', Recommended: 'Streamline operations', Similarity Score: 0.8061985969543457
Original: 'survey is', Recommended: 'Conduct an analysis', Similarity Score: 0.8064796924591064

Sentence 10: Lastly, we agreed that we must take time to look over our plans carefully and consider all angles before moving forward.
Original: 'Lastly , we', Recommended: 'Streamline operations', Similarity Score: 0.822869062423706
Original: 'our plans', Recommended: 'Execute strategies', Similarity Score: 0.8385440707206726
Original: 'and consider', Recommended: 'Conduct an analysis', Similarity Score: 0.8347948789596558
Original: 'plans carefully', Recommended: 'Implement best practices', Similarity Score: 0.82086181640625
Original: 'carefully and consider', Recommended: 'Ensure compliance', Similarity Score: 0.8193237781524658

Sentence 11: On a side note, David mentioned that his cat is recovering well from surgery.

```
For comparison, here is the `multi-qa-mpnet-base-cos-v1` model with 0.5 threshold:
```
Sentence 1: In today's meeting, we discussed a variety of issues affecting our department.

Sentence 2: The weather was unusually sunny, a pleasant backdrop to our serious discussions.

Sentence 3: We came to the consensus that we need to do better in terms of performance.
Original: 'do better in terms of performance', Recommended: 'Optimal performance', Similarity Score: 0.748181939125061

Sentence 4: Sally brought doughnuts, which lightened the mood.

Sentence 5: It's important to make good use of what we have at our disposal.
Original: 'make good use of what we', Recommended: 'Utilize resources', Similarity Score: 0.5697365999221802

Sentence 6: During the coffee break, we talked about the upcoming company picnic.

Sentence 7: We should aim to be more efficient and look for ways to be more creative in our daily tasks.
Original: 'more efficient', Recommended: 'Optimal performance', Similarity Score: 0.6936391592025757
Original: 'tasks', Recommended: 'Prioritize tasks', Similarity Score: 0.6638244390487671
Original: 'aim to be more efficient and', Recommended: 'Enhance productivity', Similarity Score: 0.6954512000083923
Original: 'look for ways to be more creative in our', Recommended: 'Foster innovation', Similarity Score: 0.5391108989715576

Sentence 8: Growth is essential for our future, but equally important is building strong relationships with our team members.
Original: 'Growth', Recommended: 'Drive growth', Similarity Score: 0.7619696855545044

Sentence 9: As a reminder, the annual staff survey is due next Friday.

Sentence 10: Lastly, we agreed that we must take time to look over our plans carefully and consider all angles before moving forward.

Sentence 11: On a side note, David mentioned that his cat is recovering well from surgery.
```
As can be seen, `bert-base-nli-mean-tokens` gave more results, but the quality was a little bit poor in comparison to `multi-qa-mpnet-base-cos-v1` which gave less number of suggestions with higher quality of phrases.

In conclusion, I have successfully implemented sentence transformers to find similar phrases in the text with standardized ones. There is a huge amount of implementations for this project like recommendation engine, text summarization, and many others.

Thank you for your attention!
