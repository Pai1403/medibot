# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model

# model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# # Enable 4-bit quantization using bitsandbytes
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Load model in 4-bit
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",  # Normal Float 4 (better than fp4)
#     bnb_4bit_use_double_quant=True,  # Further reduce memory usage
# )

# # Load Model & Tokenizer
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # QLoRA Configuration
# lora_config = LoraConfig(
#     r=16,                      # Low-rank dimension (increase for better adaptation)
#     lora_alpha=32,             # Scaling factor
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# # Apply LoRA
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model
# from accelerate import infer_auto_device_map, dispatch_model

# # ✅ Model Name
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# # ✅ Enable 4-bit Quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  
#     bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 (saves memory)
#     bnb_4bit_quant_type="nf4",  # NF4 is better than FP4 for quantization
#     bnb_4bit_use_double_quant=True,  # Further reduces memory usage
# )

# # ✅ Clear GPU Memory Before Loading
# torch.cuda.empty_cache()

# # ✅ Load Model with Proper CPU Offloading
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto"  # Auto distributes between CPU & GPU
# )

# # ✅ Use Accelerate to Automatically Balance CPU & GPU Memory
# device_map = infer_auto_device_map(model, max_memory={0: "7GB", "cpu": "12GB"})
# model = dispatch_model(model, device_map=device_map, offload_dir="offload")

# # ✅ Load Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # ✅ LoRA Configuration
# lora_config = LoraConfig(
#     r=16,  # Low-rank adaptation
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# # ✅ Apply LoRA to Model
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# # ✅ Run Sample Inference (To Verify GPU Usage)
# input_text = "What are the symptoms of pneumonia?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# # Generate Response
# with torch.no_grad():
#     output = model.generate(input_ids, max_length=100)

# # ✅ Print the Model's Response
# print("Model Response:", tokenizer.decode(output[0], skip_special_tokens=True))




# from datasets import load_dataset
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model
# # Load dataset (Assuming JSON format)
# dataset = load_dataset("json", data_files="medical_qa.json")

# # Preprocessing
# def preprocess_function(examples):
#     prompt = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]
#     return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

# # Tokenize dataset
# tokenized_datasets = dataset.map(preprocess_function, batched=True)


# from transformers import TrainingArguments, Trainer

# training_args = TrainingArguments(
#     output_dir="./mistral_qLoRA_medical",
#     per_device_train_batch_size=2,  # Keep small due to VRAM limits
#     gradient_accumulation_steps=4,  # Simulates larger batch size
#     optim="paged_adamw_32bit",
#     learning_rate=2e-4,
#     weight_decay=0.01,
#     num_train_epochs=3,
#     fp16=True,  # Mixed precision for efficiency
#     save_steps=500,
#     logging_steps=50,
#     report_to="none",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
# )

# trainer.train()
###############################################################################################################################################
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model
# from datasets import load_dataset
# from transformers import TrainingArguments, Trainer

# # ✅ Define Model Name
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# # ✅ Function to Clear GPU Memory
# def clear_vram():
#     torch.cuda.empty_cache()

# # ✅ Function to Load Model with 4-bit Quantization on GPU
# def load_model():
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,  
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map={"": 0},  # ✅ Explicitly set ALL layers to GPU
#     ).to("cuda")  # ✅ Move model explicitly to GPU

#     return model

# # ✅ Function to Load Tokenizer
# def load_tokenizer():
#     return AutoTokenizer.from_pretrained(MODEL_NAME)

# # ✅ Function to Apply LoRA for Fine-Tuning
# def apply_lora(model):
#     lora_config = LoraConfig(
#         r=16,  
#         lora_alpha=32,
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=0.1,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
    
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
#     return model

# # ✅ Function to Load and Tokenize Dataset
# from datasets import load_dataset

# def load_and_tokenize_dataset(tokenizer):
#     dataset_path = "C:/Users/Siddharth Pai/Desktop/github projects/medquad.csv"  # Correct path to your CSV file
#     dataset = load_dataset("csv", data_files=dataset_path)
    
#     # Check if pad_token is set, if not, use eos_token or add one
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
#         # Or add a new pad token if eos_token doesn't exist
#         # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     # Tokenize both 'question' and 'answer' columns
#     def tokenize_function(examples):
#         # Ensure that each entry is a string and handle missing values (NaN or None)
#         questions = [str(q) if q is not None else "" for q in examples['question']]
#         answers = [str(a) if a is not None else "" for a in examples['answer']]
        
#         return tokenizer(questions, answers, padding=True, truncation=True)
    
#     tokenized_datasets = dataset.map(tokenize_function, batched=True)
#     return tokenized_datasets




# # ✅ Function to Configure Training Arguments
# def get_training_args():
#     return TrainingArguments(
#         output_dir="./mistral_qLoRA_medical",
#         per_device_train_batch_size=1,  # 🔹 Small batch size to fit in GPU
#         gradient_accumulation_steps=4,  
#         optim="paged_adamw_32bit",
#         learning_rate=2e-4,
#         weight_decay=0.01,
#         num_train_epochs=3,
#         fp16=True,  
#         save_steps=500,
#         logging_steps=50,
#         report_to="none",
#     )

# # ✅ Function to Train the Model
# def train_model(model, tokenized_datasets):
#     training_args = get_training_args()
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_datasets["train"],
#     )

#     trainer.train()

# # ✅ Function to Run Sample Inference
# def generate_response(model, tokenizer, input_text):
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")  # ✅ Move input to GPU

#     with torch.no_grad():
#         output = model.generate(input_ids.to("cuda"), max_length=100)  # ✅ Ensure computation stays on GPU

#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # ✅ Main Execution
# if __name__ == "__main__":
#     clear_vram()  # Free up GPU memory
    
#     model = load_model()  # Load model with GPU-only quantization
#     tokenizer = load_tokenizer()  # Load tokenizer
#     model = apply_lora(model)  # Apply LoRA fine-tuning

#     tokenized_datasets = load_and_tokenize_dataset(tokenizer)  # Load and preprocess dataset
#     train_model(model, tokenized_datasets)  # Train model

#     # ✅ Test the Model After Training
#     test_input = "What are the symptoms of pneumonia?"
#     response = generate_response(model, tokenizer, test_input)
#     print("Model Response:", response)

###############################################################################################################
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model
# from datasets import load_dataset
# from transformers import TrainingArguments, Trainer
# from accelerate import Accelerator

# # ✅ Define Model Name
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# # ✅ Function to Clear GPU Memory
# def clear_vram():
#     torch.cuda.empty_cache()

# # ✅ Function to Load Model with 4-bit Quantization on GPU
# def load_model():
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,  
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map={"": 0},  # ✅ Explicitly set ALL layers to GPU
#     ).to("cuda")  # ✅ Move model explicitly to GPU

#     return model

# # ✅ Function to Load Tokenizer
# def load_tokenizer():
#     return AutoTokenizer.from_pretrained(MODEL_NAME)

# # ✅ Function to Apply LoRA for Fine-Tuning
# def apply_lora(model):
#     lora_config = LoraConfig(
#         r=16,  
#         lora_alpha=32,
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=0.1,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
    
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
#     return model

# # ✅ Function to Load and Tokenize Dataset
# from datasets import DatasetDict

# # def load_and_tokenize_dataset(tokenizer):
# #     dataset_path = "medquad.csv"  # Correct path to your CSV file
# #     dataset = load_dataset("csv", data_files=dataset_path)
    
# #     # Check if pad_token is set, if not, use eos_token or add one
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
# #         # Or add a new pad token if eos_token doesn't exist
# #         # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# #     # Tokenize both 'question' and 'answer' columns
# #     def tokenize_function(examples):
# #         # Ensure that each entry is a string and handle missing values (NaN or None)
# #         questions = [str(q) if q is not None else "" for q in examples['question']]
# #         answers = [str(a) if a is not None else "" for a in examples['answer']]
        
# #         return tokenizer(questions, answers, padding=True, truncation=True, max_length=512)
    
# #     tokenized_datasets = dataset.map(tokenize_function, batched=True)

# #     # Manually split the dataset into train and test (80% train, 20% test)
# #     tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)

# #     return tokenized_datasets
# from transformers import Trainer, TrainingArguments

# # ✅ Function to Load and Tokenize Dataset (with labels)
# def load_and_tokenize_dataset(tokenizer):
#     dataset_path = "C:/Users/Siddharth Pai/Desktop/github projects/medquad.csv"  # Correct path to your CSV file
#     dataset = load_dataset("csv", data_files=dataset_path)
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
    
#     # Tokenize both 'question' and 'answer' columns and include labels
#     def tokenize_function(examples):
#         questions = [str(q) if q is not None else "" for q in examples['question']]
#         answers = [str(a) if a is not None else "" for a in examples['answer']]
        
#         encodings = tokenizer(questions, padding=True, truncation=True, max_length=512)
#         labels = tokenizer(answers, padding=True, truncation=True, max_length=512)
#         encodings['labels'] = labels['input_ids']
#         return encodings
    
#     tokenized_datasets = dataset.map(tokenize_function, batched=True)
#     tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)
#     return tokenized_datasets

# # ✅ Training Function
# def train_model(model, tokenized_datasets):
#     training_args = TrainingArguments(
#         output_dir="./mistral_qLoRA_medical",
#         per_device_train_batch_size=1,  # Small batch size to fit in GPU
#         gradient_accumulation_steps=4,
#         optim="paged_adamw_32bit",
#         learning_rate=2e-4,
#         weight_decay=0.01,
#         num_train_epochs=3,
#         fp16=True,
#         save_steps=500,
#         logging_steps=50,
#         report_to="none",
#         use_cache=False,  # Disable cache to prevent conflict with gradient checkpointing
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_datasets["train"],
#         eval_dataset=tokenized_datasets["test"],  # Use test split for evaluation if available
#     )

#     trainer.train()

# # ✅ Function to Configure Training Arguments
# # def get_training_args():
# #     return TrainingArguments(
# #         output_dir="./mistral_qLoRA_medical",
# #         per_device_train_batch_size=1,  # Small batch size to fit in GPU
# #         gradient_accumulation_steps=4,  # Effective batch size is 1 * 4 = 4
# #         optim="paged_adamw_32bit",
# #         learning_rate=2e-4,
# #         weight_decay=0.01,
# #         num_train_epochs=3,
# #         fp16=True,  # Mixed precision training
# #         save_steps=500,
# #         logging_steps=50,
# #         report_to="none",
# #         gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
# #         no_cuda=False,  # Use CUDA for GPU acceleration
# #     )

# # ✅ Function to Train the Model
# # def train_model(model, tokenized_datasets):
# #     training_args = get_training_args()
    
# #     trainer = Trainer(
# #         model=model,
# #         args=training_args,
# #         train_dataset=tokenized_datasets["train"],
# #         eval_dataset=tokenized_datasets["test"],  # Use a test split if available
# #     )

# #     trainer.train()

# # ✅ Function to Run Sample Inference
# def generate_response(model, tokenizer, input_text):
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")  # Move input to GPU

#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=100)  # Ensure computation stays on GPU

#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # ✅ Main Execution
# if __name__ == "__main__":
#     clear_vram()  # Free up GPU memory
    
#     # Initialize Accelerator
#     accelerator = Accelerator()

#     model = load_model()  # Load model with GPU-only quantization
#     tokenizer = load_tokenizer()  # Load tokenizer
#     model = apply_lora(model)  # Apply LoRA fine-tuning

#     tokenized_datasets = load_and_tokenize_dataset(tokenizer)  # Load and preprocess dataset

#     # Move model to the accelerator's device
#     model = accelerator.prepare(model)

#     train_model(model, tokenized_datasets)  # Train model

#     # ✅ Test the Model After Training
#     test_input = "What are the symptoms of pneumonia?"
#     response = generate_response(model, tokenizer, test_input)
#     print("Model Response:", response)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator

# ✅ Define Model Name
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# ✅ Function to Clear GPU Memory
def clear_vram():
    torch.cuda.empty_cache()

# ✅ Function to Load Model with 4-bit Quantization on GPU
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},  # Explicitly set ALL layers to GPU
    ).to("cuda")  # Move model explicitly to GPU

    return model

# ✅ Function to Load Tokenizer
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Function to Apply LoRA for Fine-Tuning
def apply_lora(model):
    lora_config = LoraConfig(
        r=16,  
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ✅ Function to Load and Tokenize Dataset (with labels)
def load_and_tokenize_dataset(tokenizer, dataset_path="medquad.csv"):
    dataset = load_dataset("csv", data_files=dataset_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
    
    # Tokenize both 'question' and 'answer' columns and include labels
    def tokenize_function(examples):
        questions = [str(q) if q is not None else "" for q in examples['question']]
        answers = [str(a) if a is not None else "" for a in examples['answer']]
        
        # Ensure consistent padding and truncation for both questions and answers
        encodings = tokenizer(questions, padding="max_length", truncation=True, max_length=512)
        labels = tokenizer(answers, padding="max_length", truncation=True, max_length=512)
        encodings['labels'] = labels['input_ids']
        
        return encodings
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)
    return tokenized_datasets


# ✅ Training Function
def train_model(model, tokenized_datasets, accelerator):
    training_args = TrainingArguments(
        output_dir="./mistral_qLoRA_medical",
        per_device_train_batch_size=1,  # Small batch size to fit in GPU
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        fp16=True,
        save_steps=500,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # Use test split for evaluation if available
    )

    trainer.train()

# ✅ Function to Run Sample Inference
def generate_response(model, tokenizer, input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")  # Move input to GPU

    with torch.no_grad():
        output = model.generate(input_ids, max_length=100)  # Ensure computation stays on GPU

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Main Execution
if __name__ == "__main__":
    clear_vram()  # Free up GPU memory
    
    # Initialize Accelerator
    accelerator = Accelerator()

    # Load and prepare model
    model = load_model()  # Load model with GPU-only quantization
    tokenizer = load_tokenizer()  # Load tokenizer
    model = apply_lora(model)  # Apply LoRA fine-tuning

    # Load and tokenize dataset
    tokenized_datasets = load_and_tokenize_dataset(tokenizer)

    # Move model to the accelerator's device
    model = accelerator.prepare(model)

    # Train the model
    train_model(model, tokenized_datasets, accelerator)  

    # Test the Model After Training
    test_input = "What are the symptoms of pneumonia?"
    response = generate_response(model, tokenizer, test_input)
    print("Model Response:", response)
