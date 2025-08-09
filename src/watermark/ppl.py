import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class PerplexityEval:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    def evaluate(self, text):
        """Calculate perplexity for a given text using the model."""

        # Tokenize the text
        encodings = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encodings.input_ids.to(self.device)

        # Calculate perplexity
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()
