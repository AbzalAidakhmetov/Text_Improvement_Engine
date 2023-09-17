# Text Improvement Engine

Are you ready to embark on a journey to tackle text improvement challenges with the prowess of cosine similarity? Let's dive right in!

## Introduction

In the quest to conquer this intriguing challenge, our first step is to acquaint ourselves with the remarkable concept of cosine similarity. At its core, cosine similarity empowers us to transform phrases into word embeddings or vectors, opening the door to a world of textual analysis possibilities.

## Leveraging Pre-Trained Models

Now, instead of reinventing the wheel, we've opted for a more time-efficient approach by harnessing the capabilities of the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html). This invaluable tool allows us to effortlessly translate phrases into vectors without having to craft our own custom model. In the vast realm of pre-trained sentence transformers available through [Hugging Face](https://huggingface.co/), we've chosen the venerable BERT, specifically the trusty `bert-base-nli-mean-tokens` model.

## The Challenge of Phrase Extraction

With the conversion puzzle solved, we face the formidable task of identifying phrases within sentences that can be compared with standardized phrases. While there are libraries like [NLTK](https://www.nltk.org/) that promise to unearth meaningful word chunks from sentences, our experimentation yielded less-than-stellar results. We believe in not settling for mediocrity!

## The Brilliance of Brute Force

Instead, we've embraced an audacious solution—a brute-force strategy that promises to yield the finest results. The core idea here is to employ a sliding window technique, ranging from a window size of 1 to the length of a sentence, extracting all conceivable phrases. To turbocharge our comparisons, we first calculated vectors for our standardized phrases, which we then employed in subsequent calculations. It took numerous trials, but our perseverance led us to set a threshold of a 0.8 similarity score, delivering stellar outcomes.

## A Glimpse into the Results

The fruits of our labor are beautifully captured in the image below. Undoubtedly, there's room for further refinement, particularly in optimizing the time required for phrase extraction and developing a custom model for phrase-to-embedding conversion.

![Screenshot from 2023-09-17 17-28-51](https://github.com/AbzalAidakhmetov/Text_Improvement_Engine/assets/99760649/4d159ece-6fb4-4278-8df9-2ce7a2015843)


In conclusion, our journey into the realm of text improvement has just begun. Stay tuned for more exciting developments as we continue to enhance our engine's capabilities, forging new frontiers in the world of textual mastery! 🚀
