# Efficient Fine-tuning of LLaMA-2 7B for NER Task: A Perspective from LLM Instruction

## 1. Literature Review
- LLaMA-2 7B: A large language model developed by Facebook AI Research (FAIR) ([Link to the paper](https://arxiv.org/pdf/2307.09288.pdf))
- NER (Named Entity Recognition): A subtask of information extraction that seeks to locate and classify named entities in text into pre-defined categories such as person names, organizations, locations, etc. ([Link to the paper](https://www.aclweb.org/anthology/C16-1311/))
- LLM Instruction: A paradigm of using large language models for various tasks by providing instructions or prompts ([Link to the paper](https://arxiv.org/abs/2203.02155))
- LoRA: Low-Rank Adaptation of Large Language Models: A method for fine-tuning large language models with low-rank adaptation ([Link to the paper](https://arxiv.org/abs/2106.09685))
- QLoRA: Quantized Low-Rank Adaptation of Large Language Models: A method for fine-tuning large language models with quantized low-rank adaptation ([Link to the paper](https://arxiv.org/abs/2203.02155)) 

## 2. Experimental Design
### 2.1 Research Question
How can we efficiently fine-tune LLaMA-2 7B for the NER task by leveraging the LLM instruction paradigm, light-weight training, and low-resource training strategies?

### 2.2 Methodology
1. Prepare a small labeled dataset for NER (e.g., NAVER NER dataset, multiconer dataset(only using Korean data), kmounlp datasets)
2. Design a set of instructions or prompts for the NER task that can be understood by LLaMA-2 7B
- Instruction template : 
    ```bash
    ### Instruction:
    다음 아래 문장에서 개체명을 추출하려 합니다. 개체명은 아래 규칙(spacy를 따름)으로 태깅시켜 주세요.
    - PER: Person
    - ORG: Organization
    - GPE: Geopolitical Entity
    - LOC: Location
    - FAC: Facility
    - QUANTITY: Quantity
    - ORDINAL: Ordinal
    - CARDINAL: Cardinal
    - DATE: Date
    - TIME: Time
    - MONEY: Money
    - PERCENT: Percent
    - PRODUCT: Product
    - EVENT: Event
    - WORK_OF_ART: Work of Art
    - LANGUAGE: Language
    - LAW: Law
    - NORP: Nationalities or religious or political groups
    - MISC: Miscellaneous
    아래 문장들에서 개체명을 추출하세요.

    ### Text:
    ${input_text}

    ### Tags:
    ${output_tags}
    ```
    * input_text: input text for NER task
        - ex) "이순신은 조선의 장군이다."
    * output_tags: output tags for NER task
        - ex) \<PERSON:이순신\> \<ORG:조선\>

3. Fine-tune LLaMA-2 7B using the following strategies:
   - Light-weight training: 
        - Use a smaller batch size, gradient accumulation, and mixed-precision training to reduce memory usage and training time
        - Use 4-bit quantization, QLoRA.
   - Low-resource training: trained only 1xT4 GPU on colab environment
   - Though of lot of try, OOM was raised after few steps runned. I couldn't find the best way to train the model with low-resource training. So, it trained only few steps.
4. Evaluate the fine-tuned model on a held-out test set and compare its performance with spacy model.

### 2.3 Evaluation Metrics
- Accuracy and F1 and True Positive score for each named entity type

## 3. Result
* as the memory shortage and Colab is no longer provide GPU allocation, I couldn't infer the model.
