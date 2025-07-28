# Challenge-1b PDF Content Extractor & Analyzer

This project is a sophisticated pipeline designed to automatically extract and analyze relevant sections from a collection of PDF documents based on a specific persona and job description. It uses NLP models to identify, rank, and refine content, providing a targeted summary of the most important information.

The entire application is containerized using Docker, ensuring a consistent and reproducible environment for execution.

---

## Key Features

-   **Persona-Based Section Extraction**: Utilizes a combination of semantic search and keyword matching to identify document sections that are most relevant to a given user persona (e.g., "Travel Planner," "HR professional").
-   **Intelligent Heading Detection**: Employs a multi-faceted approach to find section headings, analyzing font size, bold styling, and common linguistic patterns (e.g., "Chapter 1", "Introduction") to distinguish them from body text.
-   **Hybrid Relevance Scoring**: Ranks the importance of each extracted section using a weighted score derived from:
    -   Semantic similarity to the job description.
    -   Semantic similarity to the persona role.
    -   Persona-specific keyword and pattern matching.
-   **Lightweight & Efficient Models**: Uses a combination of `prajjwal1/bert-tiny` for embeddings and `t5-small` for summarization. This keeps the total model size under 300 MB, making the application much easier to package and distribute.
-   **Containerized & Reproducible**: Packaged in a Docker container, which bundles the code, dependencies, and models into a single, isolated unit. This guarantees that the application runs the same way everywhere, regardless of the host machine's configuration.

---

## How It Works (Technical Overview)

This project is built for efficiency and reliability by leveraging Docker.

### Dockerization Strategy

The included `Dockerfile` builds a self-contained and efficient image using a **multi-stage build process**:

1.  **Automated Model Fetching**: The **key feature** of this setup is that the NLP models are downloaded automatically during the `docker build` process. A dedicated "downloader" stage runs the `download_models.py` script. This makes the setup process much simpler for the end-user.
2.  **Slim Base Image**: It starts from `python:3.9-slim`, which provides the necessary Python runtime without extra bloat.
3.  **Clean Final Image**: Because the model download happens in a separate stage that gets discarded, the final image only contains the necessary code, dependencies, and models, without any of the download scripts or build-time tools.
4.  **Build Caching**: Docker caches the download layer. The models will only be downloaded the very first time you build the image or if the download script changes, making subsequent builds much faster.

### Execution Flow

When the container runs, it executes `process_collections.py`, which performs the following steps:
1.  Initializes the `PDFAnalyzer`, loading the NLP models (`bert-tiny` and `t5-small`) from the `./models` directory inside the container.
2.  Reads the input configuration from `challenge1b_input.json` for each collection.
3.  Processes each specified PDF, extracting and scoring sections.
4.  Deduplicates and ranks the top 5 most relevant sections across all documents.
5.  Generates the final `challenge1b_output.json` file within each collection's directory inside the container.
