







from transformers import pipeline

# Explicitly specify the model to avoid warnings
classifier = pipeline(task="sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Perform sentiment analysis
result = classifier("David is not good, but i am not very confident")

print(result)
