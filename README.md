# LLM Prompt Recovery Challenge - Kaggle Competition

## Overview

This repository is dedicated to my participation in the "LLM Prompt Recovery Challenge", a Kaggle competition that focuses on uncovering the prompts used by Google's Gemma LLM to transform texts. The primary objective is to develop techniques that can accurately predict the prompts used in various transformations, enhancing our understanding of how LLMs interpret and process inputs.

### Why This Is Important

Delving into the mechanics of prompt engineering allows us to better tailor AI behaviors in applications ranging from automated content generation to AI-driven interactive systems, thereby advancing the field of natural language processing. Understanding the interpretability of LLMs—how they generate outputs based on specific inputs—is crucial in this era of AI transparency. Insights gained from challenges like this can lead to the development of techniques to detect when a text has been altered by an LLM, offering implications for content authenticity verification. Moreover, improving our understanding of prompt dynamics enhances our ability to control and predict LLM behavior, making these systems more reliable and understandable to users.

For more details about the competition, visit the [Kaggle Competition Page](https://www.kaggle.com/competitions/llm-prompt-recovery/overview).

### My Approach

#### 1. Modeling Differences

My primary strategy involved leveraging a large language model (LLM) to explicitly model the differences between an original text and its rewritten version by Gemma. This approach was aimed at capturing subtle nuances in language transformation, providing a deep learning framework capable of understanding complex prompt-driven text alterations.

#### 2. Data Generation for Fine-Tuning

To effectively train the LLM, I employed a Direct Preference Optimization (DPO) approach, which involved generating and scoring a set of preferred and rejected responses. This step was crucial for training the LLM with parameter-efficient fine-tuning, allowing the model to adapt to the specific task of prompt recovery without extensive computational resources.

#### 3. Training and Testing

The final step involved rigorous training and testing phases. During training, the model learned from a dynamically enhanced dataset, iteratively improving its ability to deduce the original prompts from the Gemma-generated texts. Testing was conducted to ensure the model's accuracy and its ability to generalize across unseen data, validating the effectiveness of the DPO method in a controlled environment.


## Repository Structure

- `data_generation/` - Contains scripts and notebooks for synthetic data generation.
- `final_checkpoint_phi_2/` - Stores final performance Lora adapters for the Phi-2 model.
- `final_checkpoint-1000/` - Contains training checkpoint weights.
- `data_generation.ipynb` - Shows how training data was synthetically generated, highlighting scoring based on semantic similarity.
- `dpo_main.ipynb` - Details the training and fine-tuning process using transformer models and quantization.
- `requirements.txt` - Lists dependencies required to run the provided code.

## Skills Developed

This project enhanced my capabilities in several key areas:

- **Synthetic Data Generation**: Crafting datasets that mimic real-world data to train machine learning models.
- **Semantic Analysis**: Utilizing advanced techniques to analyze and match the semantic fidelity of model outputs to human expectations.
- **Fine-Tuning LLMs**: Adjusting and optimizing large language models to specific tasks and datasets to improve their performance and applicability.
- **Machine Learning Pipelines**: Designing and deploying robust NLP pipelines using state-of-the-art tools like Hugging Face’s Transformers.
- **Prompt Engineering**: Crafting prompts that effectively guide LLMs towards generating desired textual outputs.

These skills are crucial for advancing in careers related to AI and machine learning, especially in roles focused on developing and deploying AI solutions that interact with human language.


### Start and Close Dates
- **Start Date**: February 27, 2024
- **Close Date**: April 17, 2024

## Competition Details

### Challenge
The competition challenges participants to deduce the original prompts used by the LLM from a dataset where each entry consists of an original and a rewritten text.

### Evaluation
The quality of predictions is assessed using the "Sharpened Cosine Similarity" which evaluates the closeness of the predicted prompts to the true prompts, emphasizing accuracy and depth in understanding LLM prompt processing.

## Dataset Generation

Participants were provided with minimal initial data, necessitating the generation of a comprehensive dataset from scratch. This involved:

- **Finding or Creating Original Texts**: Sourcing texts suitable for the project or creating new content that could realistically be used as LLM input.
- **Rewriting Using Gemma**: Utilizing the Gemma model to generate rewritten versions of these texts based on specific, unseen prompts.
- **Labeling and Scoring**: Each pair of original and rewritten texts was then labeled with the appropriate prompt and scored for semantic similarity to ensure alignment with the expected transformations.

The process required extensive use of various NLP tools and frameworks to simulate realistic LLM interactions and generate valid training data.
