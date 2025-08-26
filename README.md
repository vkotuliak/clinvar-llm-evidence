# ClinVar Data Processing and Analysis Pipeline

This project provides a set of scripts to download, process, and analyze data from the ClinVar database, PubMed, and associated publications to prepare it for Large Language Model (LLM) analysis.

## Data Processing Workflow

Follow these steps sequentially to process the data.

### 1. Download ClinVar Database
First, download the ClinVar database from the official NCBI FTP server:
[https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/)

Download the `ClinVarFullRelease_*.xml.gz` file and rename it to `clinvar_dataset.xml.gz`.

### 2. Convert ClinVar XML to CSV
Run the following script to convert the downloaded XML data into a CSV file. This will create `data/clinvar_csv_dataset.csv`.

```bash
python3 src/s01_clinvar_xml_to_csv.py
```

### 3. Fetch PubMed Abstracts
This script extracts unique PubMed references from the CSV file and fetches the abstract for each article.

```bash
python3 src/s03_fetch_pubmed_abstracts.py
```

### 4. Download Linked PDFs
To train the model, you will also need the full text from linked PDF articles.

**a. Generate DOI list:**
First, create a `dois.txt` file containing all the DOIs from the dataset.
```bash
python3 scripts/make_doi_txt.py
```

**b. Download PDFs with PyPaperBot:**
Use `PyPaperBot` to download the PDFs using the generated DOI list. Using `--use-doi-as-filename` is recommended for easier retrieval later.

```bash
python3 -m PyPaperBot --doi-file="dois.txt" --dwn-dir="./papers_by_doi" --use-doi-as-filename
```

### 5. Extract Text from PDFs
Process the downloaded PDFs to extract their text content.

```bash
python3 src/s04_extract_pdfs.py
```

### 6. Prepare Data for Analysis
This script preprocesses and organizes the data from all previous steps (CSV, abstracts, PDF text) to make it ready for modeling.

```bash
python3 src/s05_data_preparation.py
```

### 7. LLM Analysis
Once the data is prepared, use this script to perform the analysis with a Large Language Model.

```bash
python3 src/s06_LLM_analysis.py
```

## Additional Scripts

### Data Visualization
*   `scripts/make_graphics.py`: Generates various plots and graphics from the ClinVar CSV data.

### Data Exploration
*   `src/s02_analyze_citation_sources.py`: Analyzes the citation sources in the dataset. We chose to focus on PubMed as it comprises the majority of citations.
*   `scripts/clinvar_preview.py`: A utility script to inspect a small snippet of the ClinVar XML. It can be used with the provided `data/example.xml`.
*   `scripts/find_variants_with_papers.py`: A script to identify and list genetic variants that have associated scientific papers.

### Helper Scripts & Utilities
*   `scripts/structure_data.py`: A helper script to structure data into the final format required for the LLM input.
*   `scripts/find_duplicates.py`: A utility to find and report duplicate entries within the dataset.
*   `scripts/num_words_in_pdf.py`: Calculates and shows the word count for the extracted text from the downloaded
