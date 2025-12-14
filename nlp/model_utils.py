from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from evaluate import load as load_evaluate
import numpy as np
import nltk

def generate_continuations(model_name, contexts, max_new_tokens=50, device=None):
	"""
	Generate next-sentence/continuation for a list of context strings.

	Args:
		model_name (str): HuggingFace causal LM name (e.g., "gpt2").
		contexts (list of str): List of context sentences/paragraphs.
		max_new_tokens (int): Max tokens to generate per context.
		device (str or None): "cuda" or "cpu". If None, use GPU if available.

	Returns:
		List[str]: Generated continuations, in same order as input contexts.
	"""
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	# Load model & tokenizer
	gen_tokenizer = AutoTokenizer.from_pretrained(model_name)
	gen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
	gen_model.eval()

	# Add a padding token if it doesn't exist (important for some models like GPT2)
	if gen_tokenizer.pad_token is None:
		gen_tokenizer.pad_token = gen_tokenizer.eos_token

	continuations = []

	for context in contexts:
		# Tokenize input
		inputs = gen_tokenizer(context, return_tensors="pt").to(device)

		# Check if input_ids is empty before generation
		if inputs["input_ids"].shape[1] == 0:
			print(f"Warning: Empty input_ids for context: '{context}'. Skipping generation.")
			continuations.append("") # Append an empty string or a placeholder
			continue

		# Generate
		outputs = gen_model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=True,      # allow variation, can set False for deterministic
			temperature=0.8,    # creativity
			top_p=0.9,          # nucleus sampling
			pad_token_id=gen_tokenizer.eos_token_id
		)

		# Extract generated continuation (skip input tokens)
		continuation = gen_tokenizer.decode(
			outputs[0][inputs['input_ids'].shape[1]:],
			skip_special_tokens=True
		)

		continuations.append(continuation.strip())

	return continuations

def lm_score(model, tokenizer, context, continuation, device = "cpu"):
	"""
	Computes log-probability of continuation given context.
	Higher = better continuation.
	"""
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	
	# Ensure model is on the correct device
	model = model.to(device)
	
	# Add a padding token to the tokenizer, using the eos_token as pad_token
	# This is necessary for models like GPT-2 that don't have a default pad token
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	text = context + continuation
	inputs = tokenizer(text, return_tensors="pt")

	# Handle cases where tokenized input is empty
	if inputs["input_ids"].shape[1] == 0:
		return -float('inf') # Return a very low score for empty inputs
	
	# Move all inputs to device
	inputs = {k: v.to(device) for k, v in inputs.items()}
	
	with torch.no_grad():
		# Create labels tensor separately to ensure it's on the correct device
		labels = inputs["input_ids"].clone()
		outputs = model(**inputs, labels=labels)
		# outputs.loss is average negative log-likelihood
		nll = outputs.loss.item()

	# Convert NLL to log-probability (higher is better)
	log_prob = -nll * inputs["input_ids"].shape[1]
	return log_prob

def nsp_score(bert_model, bert_tokenizer, context, continuation, device = "cpu"):
	"""
	Returns probability that continuation follows context.
	"""
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	
	# Ensure model is on the correct device
	bert_model = bert_model.to(device)

	inputs = bert_tokenizer(context, continuation, return_tensors="pt")

	# Handle cases where tokenized input is empty
	if inputs["input_ids"].shape[1] == 0:
		return 0.0 # Return 0 probability for empty inputs

	inputs = {k: v.to(device) for k, v in inputs.items()}
	with torch.no_grad():
		outputs = bert_model(**inputs)
		logits = outputs.logits
		probs = logits.softmax(dim=1)[0]
	# Index 0 = IsNextSentence, 1 = NotNextSentence
	return probs[0].item()

# BLEU requires punkt tokenizer

def evaluate_predictions(predictions, references):
	"""
	Compare model predictions with ground truth using:
	- BERTScore
	- ROUGE (1,2,L)
	- BLEU
	"""
	nltk.download("punkt")
	nltk.download('punkt_tab')
	# Ensure matching lengths
	assert len(predictions) == len(references), "Prediction and reference length mismatch."

	# ============================
	# 1. BERTScore
	# ============================
	bertscore = load_evaluate("bertscore")
	bert_res = bertscore.compute(
		predictions=predictions,
		references=references,
		model_type="bert-base-uncased",
		device="cpu"  # Run evaluation on CPU to avoid device conflicts
	)
	bert_precision = np.mean(bert_res["precision"])
	bert_recall = np.mean(bert_res["recall"])
	bert_f1 = np.mean(bert_res["f1"])

	# ============================
	# 2. ROUGE
	# ============================
	rouge = load_evaluate("rouge")
	rouge_res = rouge.compute(
		predictions=predictions,
		references=references,
	)

	# ============================
	# 3. BLEU (corpus)
	# ============================
	from nltk.translate.bleu_score import corpus_bleu

	# NLTK corpus BLEU expects tokenized inputs ➙ we will use word_tokenize
	tokenized_preds = [nltk.word_tokenize(p) for p in predictions]
	tokenized_refs = [[nltk.word_tokenize(r)] for r in references]

	bleu_score = corpus_bleu(tokenized_refs, tokenized_preds)

	# ============================
	# Final Output
	# ============================
	return {
		"bertscore_precision": bert_precision,
		"bertscore_recall": bert_recall,
		"bertscore_f1": bert_f1,
		"rouge1": rouge_res["rouge1"],
		"rouge2": rouge_res["rouge2"],
		"rougeL": rouge_res["rougeL"],
		"bleu": bleu_score,
	}

def all_model_evaluation(model, tokenizer, bert_model, bert_tokenizer, contexts, candidates, device="cpu"):
    """
    Computes:
    - Average LM, NSP,  Final hybrid ranking scores
    - BERTScore (P, R, F1)
    - ROUGE (1, 2, L)
    - BLEU

    Returns a single dictionary with all metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    total_lm = 0
    total_nsp = 0
    total_final = 0
    count = 0

    predictions = []
    references = []

    for context, continuation in zip(contexts, candidates):
        # Skip evaluation if context or continuation is empty to prevent errors
        if not context or not continuation:
            continue

        lm = lm_score(model, tokenizer, context, continuation, device)
        nsp = nsp_score(bert_model, bert_tokenizer, context, continuation, device)


        # Final hybrid score
        final_score = (0.6 * lm) + (0.2 * (nsp * 10))

        # Accumulate averages
        total_lm += lm
        total_nsp += nsp
        #total_judge += judge
        total_final += final_score
        count += 1

        # Store for evaluation metrics
        predictions.append(continuation)
        references.append(context)

    # Safety check
    if count == 0:
        return {}

    # Compute averages
    avg_lm = total_lm / count
    avg_nsp = total_nsp / count
    #avg_judge = total_judge / count
    avg_final = total_final / count

    # Run BERTScore, ROUGE, BLEU evaluation
    text_eval = evaluate_predictions(predictions, references)

    # Merge results
    return {
        "avg_lm": avg_lm,
        "avg_nsp": avg_nsp,
        "avg_final": avg_final,
        **text_eval,
    }