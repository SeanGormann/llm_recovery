# LLM Prompt Recovery Challenge - Kaggle Competition

## Overview

This repository is dedicated to my participation in the "LLM Prompt Recovery Challenge", a Kaggle competition that focuses on uncovering the prompts used by Google's Gemma LLM to transform texts. The primary objective is to develop techniques that can accurately predict the prompts used in various transformations, enhancing our understanding of how LLMs interpret and process inputs.

### Why This Is Important
Delving into the mechanics of prompt engineering allows us to better tailor AI behaviors in applications ranging from automated content generation to AI-driven interactive systems, thereby advancing the field of natural language processing.

## Repository Structure

- `data_generation/` - Contains scripts and notebooks for synthetic data generation.
- `final_checkpoint_phi_2/` - Stores final performance Lora adapters for the Phi-2 model.
- `final_checkpoint-1000/` - Contains training checkpoint weights.
- `data_generation.ipynb` - Shows how training data was synthetically generated, highlighting scoring based on semantic similarity.
- `dpo_main.ipynb` - Details the training and fine-tuning process using transformer models and quantization.
- `requirements.txt` - Lists dependencies required to run the provided code.

## Skills Developed and Career Benefits

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
