from pathlib import Path

import torch
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

TARGET_MODULES = {"Qwen3-4B": r"(?:.*\.self_attn\.[qkvo]_proj.*|.*\.mlp\.(?:gate|up|down)_proj.*)",
                  "Gemma3-4B": r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
                  "Llama3.2-3B": r"(?:.*\.self_attn\.(?:q|k|v|o)_proj|.*\.mlp\.(?:gate|up|down)_proj)"}
# TARGET_MODULES =


def get_common_setup(model_cache, model_id):
    """Returns model, tokenizer, and peft_config common to both pipelines."""
    full_path = model_cache / model_id
    if model_cache == Path("output"):
        full_path = full_path / "saved_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(full_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(full_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2"
        attn_implementation="sdpa",
    )
    # messages = [
    #     {
    #         "role": "system",
    #         "content": [{"type": "text", "text": "You are a helpful assistant."}]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Describe this image in detail."}
    #         ]
    #     }
    # ]
    # inputs = tokenizer.apply_chat_template(
    #     messages, add_generation_prompt=True, tokenize=True,
    #     return_dict=True, return_tensors="pt"
    # ).to(model.device)
    # input_len = inputs["input_ids"].shape[-1]
    #
    # with torch.inference_mode():
    #     generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    #     generation = generation[0][input_len:]
    #
    # decoded = tokenizer.decode(generation, skip_special_tokens=True)
    # print(decoded)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES[model_id.split("_")[0]],
        # modules_to_save=["embed_tokens", "lm_head"]
        modules_to_save=["lm_head"]
    )
    model = get_peft_model(model, peft_config)
    assert isinstance(model, PeftModel)

    wrapped = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            wrapped.append(name)

    print(f"LoRA-wrapped modules ({len(wrapped)}):")
    for n in wrapped:
        print(n)
    # return model, tokenizer, peft_config

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    def count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Total params:", count_params(model))
    print("Trainable params:", count_trainable(model))
    print("Trainable %:", 100 * count_trainable(model) / count_params(model))
    if model.config.model_type == "gemma3":
        model.config.model_type = "llama"
    return model, tokenizer, peft_config


def get_dataset(dataset_path):
    dataset = load_dataset(dataset_path)
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    # train_dataset = train_dataset.add_column("images", [None] * len(train_dataset))
    # test_dataset = test_dataset.add_column("images", [None] * len(test_dataset))
    return train_dataset, test_dataset