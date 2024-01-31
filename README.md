# Game of Thrones GPT

## Overview
This project leverages the LangChain framework and Google’s Gemini model to train 'GOT_GPT', a Retrieval-Augmented Generation (RAG) based Large Language Model (LLM). The model is trained on the complex narrative of the Game of Thrones series, enabling it to understand and answer intricate queries about the show’s plot.

## Model Training
The model was trained using the LangChain framework and Google’s Gemini model. The training data consisted of the complex narrative of the Game of Thrones series.

## Data Acquisition, Preprocessing & Deployment
Data scraping techniques were implemented to gather 78 text-rich documents for relevant training data. Hugging Face’s Instructor Embedding was used for preprocessing the scraped text. The model was then deployed on Hugging Face Spaces, in the form of a Streamlit Application.

## Usage
The model is deployed on Hugging Face Spaces. You can access it here.
https://huggingface.co/spaces/kabirnawani/GOT_GPT
