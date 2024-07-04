import gc
import pdb

import lightning as L
import torch
from ipdb import set_trace
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from modules.data_modules.asthma_classification import AsthmaClassification
from modules.data_modules.cad_classification import CADClassification
from modules.data_modules.cohort_abdominal_classification import (
    CohortAbdominalClassification,
)
from modules.data_modules.cohort_alcohol_abuse_classification import (
    CohortAlcoholAbuseClassification,
)
from modules.data_modules.cohort_drug_abuse_classification import (
    CohortDrugAbuseClassification,
)
from modules.data_modules.cohort_english_classification import (
    CohortEnglishClassification,
)
from modules.data_modules.cohort_make_decisions_classification import (
    CohortMakeDecisionsClassification,
)
from modules.data_modules.concept_problem_extraction import ConceptProblemExtraction
from modules.data_modules.concept_test_extraction import ConceptTestExtraction
from modules.data_modules.concept_treatment_extraction import ConceptTreatmentExtraction
from modules.data_modules.diabetes_mellitus_classification import (
    DiabetesMellitusClassification,
)
from modules.data_modules.drug_extraction import DrugExtraction
from modules.data_modules.medication_extraction import MedicationExtraction
from modules.data_modules.mimic_mortality_prediction import MimicMortalityPrediction
from modules.data_modules.obesity_classification import ObesityClassification
from modules.data_modules.risk_factor_cad_extraction import RiskFactorCADExtraction



class Seq2SeqLMInferenceModule(L.LightningModule):
    def __init__(self, model, max_length=1024, max_new_tokens=256):
        super().__init__()
        self.model = model
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        idxs = batch.pop("idx")
        # send batch keys to device:
        # for key in batch:
        #     batch[key] = batch[key].to(self.device)
        preds = self.model.generate(**batch, max_length=self.max_length)
        return (idxs, batch["input_ids"], preds)


class CausalLMInferenceModule(L.LightningModule):
    def __init__(self, model, max_length=1024, max_new_tokens=256):
        super().__init__()
        self.model = model
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        idxs = batch.pop("idx")
        input_ids = batch["input_ids"]
        input_length = batch["input_ids"].shape[1]
        preds = self.model.generate(
            **batch, max_new_tokens=self.max_new_tokens, output_scores=True
        )
        # also save the logits
        preds = preds[:, input_length:]
        return (
            idxs,
            input_ids,
            preds,
        )


NAME_TO_MODULE = {
    "cohort_drug_abuse_classification": CohortDrugAbuseClassification,
    "cohort_alcohol_abuse_classification": CohortAlcoholAbuseClassification,
    "cohort_english_classification": CohortEnglishClassification,
    "cohort_make_decisions_classification": CohortMakeDecisionsClassification,
    "cohort_abdominal_classification": CohortAbdominalClassification,
    "mimic_mortality_prediction": MimicMortalityPrediction,
    "obesity_classification": ObesityClassification,
    "diabetes_mellitus_classification": DiabetesMellitusClassification,
    "asthma_classification": AsthmaClassification,
    "cad_classification": CADClassification,
    "medication_extraction": MedicationExtraction,
    "drug_extraction": DrugExtraction,
    "risk_factor_cad_extraction": RiskFactorCADExtraction,
    "concept_test_extraction": ConceptTestExtraction,
    "concept_problem_extraction": ConceptProblemExtraction,
    "concept_treatment_extraction": ConceptTreatmentExtraction
}

# must generate more tokens for tasks that require list outputs
MAX_NEW_TOKENS = {
    "text": 64,
    "logit": 32,
    "list": 256,
}


def load_model_and_tokenizer(model_name_or_path: str, device="cuda", eval_type="logit"):

    max_new_tokens = MAX_NEW_TOKENS[eval_type]
    print(
        f"Running model {model_name_or_path} with max_new_tokens {max_new_tokens} with eval_type {eval_type}"
    )
    if (
        "Llama-2-7b-chat-hf" in model_name_or_path
        or "Llama-2-13b-chat" in model_name_or_path 
    ):  # https://www.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 2048,  # llama-2 has a 4k context size
            "max_new_tokens": max_new_tokens,
            "prompt_format": """<s>[INST] <<SYS>>
You are an intelligent clinical language model.
Below is a snippet of a patient's discharge summary, followed by an instruction from a healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide an accurate answer to the instruction, while being concise.
<</SYS>>

CLINICAL NOTE: {text}
INSTRUCTION: {instruction} [/INST]
ANSWER: """,
        }
    elif "medalpaca" in model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 2048,  # 2048 from Llama 1's context-size
            "max_new_tokens": max_new_tokens,
            "prompt_format": """Context: {text}

Instruction: {instruction}

Answer: """,
        }
    elif "mistral" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            attn_implementation="sdpa",
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 2048,  # mistral has a 32k context size
            "max_new_tokens": max_new_tokens,
            "prompt_format": """[INST] CLINICAL NOTE: {text}
INSTRUCTION: {instruction}
ANSWER: [/INST]
""",
        }
    elif "meditron" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 2048,  # llama 2 has a 4k context size
            "max_new_tokens": max_new_tokens,
            "prompt_format": """<|im_start|>system
You are an intelligent clinical language model.
Below is a snippet of a patient's discharge summary, followed by an instruction from a healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide an accurate answer to the instruction, while being concise.<|im_end|>
<|im_start|>user
Discharge summary:
{text}

Instruction:
{instruction}<|im_end|>
<|im_start|>assistant
Answer: """,
        }
    elif "sclepius" in model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
            # load_in_8bit=True,
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 2048,  # Asclepius finetuned from Llama1,
            # whose context is 2048
            # NOTE: Jiuding has it as the Llama 2 version but looks like it is the Llama 1 ver.
            "max_new_tokens": max_new_tokens,
            "prompt_format": """You are an intelligent clinical language model.

[Discharge Summary Start]
{text}
[Discharge Summary End]

[Instruction Start]
{instruction}
[Instruction End]

Above, we provide you with a part of the discharge summary and the instruction that the healthcare professional gave about it. Generate a response to the  healthcare professional’s instruction using the given  discharge summary.
Here are the requirements:
- Your response must be accurate and concise to the instruction.
- If the instruction is not fully answerable within the given discharge summary, explain the reason why it is unanswerable using the given information.
- Do not say that you cannot respond as an AI model.
- Do not ask back nor rephrase the instruction.

Response: """,
        }

    elif "flan" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            load_in_4bit=True,
            device_map="auto",
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "EncoderDecoder",
            "batch_size": 1,
            # TODO: if time permits re-run all with 2048 context
            "max_length": 2048,  # flan-ul2 has a 2048 context size as per the paper
            "max_new_tokens": max_new_tokens,
            "prompt_format": """CLINICAL NOTE: {text} INSTRUCTION: {instruction} ANSWER: """,
        }
    elif "Cammel" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 2048,  # llama-2 has a 4k context size
            "max_new_tokens": max_new_tokens,
            "prompt_format": """<s>[INST] <<SYS>>
You are an intelligent clinical language model.
Below is a snippet of a patient's discharge summary, followed by an instruction from a healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide an accurate answer to the instruction, while being concise.
<</SYS>>

CLINICAL NOTE: {text}
INSTRUCTION: {instruction} [/INST]
ANSWER: """,
        }

    elif "alpaca" in model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 2048,  # 2048 from Llama 1's context-size
            "max_new_tokens": max_new_tokens,
            "prompt_format": """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{text}

### Response:
""",
        }
    elif "llama" in model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 2048,
            "max_new_tokens": max_new_tokens,
            "prompt_format": """### Instruction:
In your capacity as a medical assistant, help the user's with their medical inquiries, and follow their instructions. 
{instruction}

### Input:
{text}

### Output: """,
        }
    elif "t5" in model_name_or_path or "T5" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/work/frink/private_datasets/huggingface_cache/",
        )
        model_config = {
            "model_type": "EncoderDecoder",
            "batch_size": 256,
            "max_length": 512,  # t5 has a max length of 512
            "max_new_tokens": max_new_tokens,
            "prompt_format": """You are an intelligent clinical language model.
Below is a snippet of a patient's discharge summary, followed by an instruction from a healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide an accurate answer to the instruction, while being concise.

[Discharge Summary Start]
{text}
[Discharge Summary End]

[Instruction Start]
{instruction}
[Instruction End]

[Answer]
""",
        }
    else:
        raise NotImplementedError(
            "Model type {} is not supported".format(model_name_or_path)
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    if model_config["model_type"] == "DecoderOnly":
        module = CausalLMInferenceModule(
            model,
            max_length=model_config["max_length"],
            max_new_tokens=model_config["max_new_tokens"],
        )
    elif model_config["model_type"] == "EncoderDecoder":
        module = Seq2SeqLMInferenceModule(
            model,
            max_length=model_config["max_length"],
            max_new_tokens=model_config["max_new_tokens"],
        )
    else:
        raise NotImplementedError("Model type is not supported")

    return module, tokenizer, model_config


def load_data_module(
    dataset_name,
    tokenizer,
    root_path,
    truncation_strategy,
    model_config,
    total_samples=None,
    is_alpaca=False,
):
    assert (
        dataset_name in NAME_TO_MODULE.keys()
    ), f"Dataset {dataset_name} is not supported"
    return NAME_TO_MODULE[dataset_name](
        root_path=root_path,
        tokenizer=tokenizer,
        truncation_strategy=truncation_strategy,
        model_config=model_config,
        total_samples=total_samples,
        is_alpaca=is_alpaca,
        is_medtuned=False,
    )


def compute_per_sample_loss(inputs, logits, pad_token_id):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous().double()
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1))
    loss_per_sample = loss_per_sample * (shift_labels != pad_token_id).float()
    loss_per_sample = loss_per_sample.mean(axis=1)
    return loss_per_sample


def get_num_tokens(tokenizer, inputs_label_etc):
    tokens = tokenizer(inputs_label_etc, return_tensors="pt")
    # set_trace()
    num_label_tokens = len(tokens["input_ids"][0]) - (
        1 if inputs_label_etc[0] != "\n" else 3
    )
    return num_label_tokens, tokens


# pass the logits of the labels for each datsets (yes, no, present etc)
# get the input logits, match these with the passed logit from step 1
# yes , no 1 token
# unmentioned - 2
# TOKEN_MAP = {
# "model_name":
#       ("yes", 1), (unmentioend, 2)} etc


def do_logit_based_evaluation(batch, module, tokenizer):
    input_ids = batch["input_ids"].to(module.device)
    attention_mask = batch["attention_mask"].to(module.device)
    label_cls = batch["label_cls"]
    label_space = batch["label_space"][0]
    pred_texts = []
    gold_texts = []
    classes = []
    losses = []
    label_logits = []
    batch_probas = []
    logits = (
        module.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        .logits.detach()
        .cpu()
    )
    for i, label_in_input in enumerate(label_space):
        if batch["input_text"][0][-1] == "\n":
            label_in_input = f"\n{label_in_input}"
        num_label_tokens, tokens = get_num_tokens(tokenizer, label_in_input)

        # p(token_1 | text) * (token_2 | text + token_1)
        if num_label_tokens > 1:
            text_with_label = [
                batch["input_text"][0] + label_in_input.replace("\n", "")
            ]
            input_with_label = tokenizer(text_with_label, return_tensors="pt")
            logits_with_label = (
                module.model(
                    input_ids=input_with_label["input_ids"].to(module.device),
                    attention_mask=input_with_label["attention_mask"].to(module.device),
                    labels=input_with_label["input_ids"].to(module.device),
                )
                .logits.detach()
                .cpu()
            )
            logits_needed = logits_with_label[:, -(num_label_tokens + 1) : -1, :]
            probabilities = []
            for i in range(num_label_tokens):
                ith_input_id = tokens["input_ids"][0, -(num_label_tokens - i)]
                ith_logits = logits_needed[0, i, :]

                probabilities.append(
                    torch.nn.functional.softmax(ith_logits, dim=-1)[ith_input_id]
                )
            # comput proba as product of all probas
            probabilities = torch.prod(torch.stack(probabilities))

        else:
            tokens = tokens["input_ids"][0, -1:]
            logits_needed = logits[:, -1, :]
            probabilities = torch.nn.functional.softmax(logits_needed, dim=-1)[0][
                tokens
            ][0]

        # # dont append to the list (we dont know the number for each)
        # # for now I am adding it as a tuple but there might be a better way to get the final auroc
        label_logits.append((logits_needed, num_label_tokens))
        # # after we get num_logits, slide and append it to a list
        # loss_per_sample = torch.Tensor(
        #     compute_per_sample_loss(
        #         inputs_label["input_ids"].cpu(), logits, tokenizer.pad_token_id
        #     )
        # )
        # losses.append(loss_per_sample)
        batch_probas.append(probabilities)

        torch.cuda.empty_cache()
        gc.collect()

    batch_probas = torch.stack(batch_probas).unsqueeze(0)
    batch_probas = batch_probas / batch_probas.sum()
    # detach losses and put in cpu
    classes = torch.argmax(batch_probas, dim=1)
    # batch probas into list
    batch_probas = batch_probas.cpu().numpy().tolist()
    # set_trace()
    # batch probast into list
    # set_trace()
    # TODO make this work with batch size > 1
    for i in range(input_ids.shape[0]):
        predicted_label = classes[i]

        predicted_text = label_space[predicted_label]

        pred_texts.append(predicted_text)
        gold_text = label_space[label_cls[i]]

        gold_texts.append(gold_text)
    return (
        classes.cpu().numpy(),
        label_cls.cpu().numpy(),
        pred_texts,
        gold_texts,
        [batch_probas],
    )


def do_logit_based_evaluation_new(batch, module, tokenizer):
    input_ids = batch["input_ids"].to(module.device)
    attention_mask = batch["attention_mask"].to(module.device)
    label_cls = batch["label_cls"]
    label_space = batch["label_space"][0]
    pred_texts = []
    gold_texts = []
    classes = []
    losses = []
    label_logits = []
    batch_probas = []

    idxs, _, pred_ids = model.predict_step(batch_full_text)
    preds_text = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    probabilities = torch.nn.functional.softmax(logits_needed, dim=-1)[0][tokens][0]

    # # dont append to the list (we dont know the number for each)
    # # for now I am adding it as a tuple but there might be a better way to get the final auroc
    label_logits.append((logits_needed, num_label_tokens))
    # # after we get num_logits, slide and append it to a list
    # loss_per_sample = torch.Tensor(
    #     compute_per_sample_loss(
    #         inputs_label["input_ids"].cpu(), logits, tokenizer.pad_token_id
    #     )
    # )
    # losses.append(loss_per_sample)
    batch_probas.append(probabilities)

    torch.cuda.empty_cache()
    gc.collect()

    batch_probas = torch.stack(batch_probas).unsqueeze(0)
    batch_probas = batch_probas / batch_probas.sum()
    # detach losses and put in cpu
    classes = torch.argmax(batch_probas, dim=1)
    # batch probas into list
    batch_probas = batch_probas.cpu().numpy().tolist()
    # set_trace()
    # batch probast into list
    # set_trace()
    # TODO make this work with batch size > 1
    for i in range(input_ids.shape[0]):
        predicted_label = classes[i]

        predicted_text = label_space[predicted_label]

        pred_texts.append(predicted_text)
        gold_text = label_space[label_cls[i]]

        gold_texts.append(gold_text)
    return (
        classes.cpu().numpy(),
        label_cls.cpu().numpy(),
        pred_texts,
        gold_texts,
        [batch_probas],
    )


# """
# """You are an intelligent clinical language model.
# Below is a snippet of a patient's discharge summary, followed by an instruction from a healthcare professional.
# Write a response that appropriately completes the instruction.
# The response should provide an accurate answer to the instruction, while being concise.
