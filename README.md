# doc-analysis-using-LLMs

# 👨🏻‍💻 Acumen - Academic Paper Insights

- In the digital age, the volume of academic literature has exponentially increased, making it challenging for researchers to keep up with the latest advancements and discoveries in their fields. Traditional methods of literature review and data extraction are time-consuming and inefficient, often resulting in missed critical insights.
- To address these challenges, this project explores the integration of large language models (LLMs) with Semantic Scholar, an AI-powered research tool designed to help users navigate vast amounts of academic papers.


# Achivements of the project:-

- Automated Document Summarization: Leverage the power of LLMs to automatically summarise academic papers, providing concise and accurate overviews of the content.
- Key Entity Extraction: Identify and extract key entities such as authors, institutions, and specific research terms from the academic papers.
- Trend Identification: Analyse multiple documents to identify prevalent themes, trends, and patterns within the research domain.
- Efficient Data Processing: Implement tools and techniques to handle large volumes of text data efficiently, addressing the input size limitations of LLMs.
- User-Friendly Interface: All above points can be done via a simple and interactive UI that allows users to easily query and retrieve summarised information and trends from a large corpus of academic literature.

# How to run this project?

- Download the zip file of this project and extract it in a folder in your system.
- Then get your unique keys of semantic scholar and huggingface from their official websites.
- Then update these keys in config.py file
- Then execute this command in the terminal to install all required libraries
"
  pip install -r requirements.txt
"
- Now, execute this command in the terminal to interact with this project
"
  streamlit run gui.py
"
