from transformers import AutoTokenizer

# Load the tokenizer from the downloaded model path
tokenizer = AutoTokenizer.from_pretrained("/Volumes/Pardhu_2TB/llama/Models/Llama3.2-11B-Vision")

# Test encoding a sentence
text = "Hello, how are you?"
tokens = tokenizer.encode(text, return_tensors="pt")

# Print Token IDs
print("Tokenized Input:", tokens)

decoded_text = tokenizer.decode(tokens[0][1])
print("Decoded Text:", decoded_text)