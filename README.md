# Climate Policy Text Analysis
A machine learning pipeline to analyze national climate change policies using text classification, embedding models, and health indicator correlations.

This is a repository for the pre-interview task, as assigned for the interview stage in the **Lancet Countdown: Tracking Progress on Health and Climate Change (Working Group on Public and Political Engagement, WG5)** project.

## Repository Structure
    .
    ├── data
    │   ├── CCLW.csv
    │   └── WDI_Data.csv
    ├── images
    ├── output_data
    │   ├── panel_data.csv
    ├── src
    │   ├── climate_policy_indicator.ipynb
    │   ├── preparing_data.ipynb
    │   ├── statistical_analysis.R
    │   ├── extract_content.py
    │   ├── summarizer_script.py
    │   └── run_summarizer.sh
    ├── LICENSE
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## Data
This project uses:

- [Climate Change Laws of the World (CCLW)](https://climate-laws.org/)
- [World Development Indicators (WDI)](https://databank.worldbank.org/source/world-development-indicators)

Please the downloaded data in the `data/` folder in their csv format.

## Eperimental Setup
### Task 1 overview (ClimatePolicyIndicatorDevelopment)
Using the Climate Change Laws of the World database, develop a quantitative indicator that captures a specific aspect of national climate change policy or ambition based on the text of laws or policies. You're encouraged to use advanced text analysis techniques (e.g., machine learning, topic modeling, sentiment analysis, embeddings), though simpler methods are acceptable if well justified. The goal is to extract meaningful policy insights from the textual data.

- <b>Data</b>: [Climate Change Laws of the World](./data/#CCLW.csv)
     - 8974 climate-related laws, rgulations, policies and directives.
     - Spans 196+ countries and territories.
     - Each record includes researcher-written summaries, classifications (e.g. adaptation, mitigation, disaster risk management), metadata like enactment dates, coverage type, and legislative processes. 
- <b>Code</b>: [Jupyter Notebook on classifying the climate policy indicator using text-based analysis](./src/#climate_policy_indicator.ipynb)
     - I preprocess the policy summaries in both English and non-English languages to clean and standardize the text. Sentence embeddings are then generated from the processed text to serve as input features for a simple neural network, which performs multi-label classification to identify the topics or themes of the policies. Additionally, I construct an ambition score as a supplementary feature, which can be incorporated into the embedding to enrich the model's representation.

### Task 2 overview (Statistical/ComputationalAnalysis)
Select one or more credible health indicators (e.g., from WHO, Global Burden of Disease, or World Bank) and justify your choice based on its relevance to climate policy or political engagement. Then, conduct a quantitative analysis—such as a panel data model—to examine the relationship between your climate policy indicator and the selected health outcome(s).

- <b>Data</b>:
    1) [Climate Change Laws of the World](./data/#CCLW.csv)
    2) [World Development Indicators](./data/#WDI_Data.csv)
          - WDI dataset (which is a flagship project of World Bank) which contain environmental, health, economic, demographic and social indicators that spans over time and approximately 217 countries and over 40 country groupings. The dataset I used is a filtered subset of the full dataset, and contain information from 2015 onwards.
- <b>Code</b>:
    - [Data Linkage code](./src/#preparing_data.ipynb)
      
      Using the country codes and years from the CCLW dataset, I apply one-hot encoding to the policy themes and calculate the number of unique themes (or topics) addressed by each country over time. Then I merge this data with corresponding country-year information from the WDI dataset, enabling a unified panel structure for hypothesis testing and quantitative analysis.
    - [Hypothesis testing to find the significance of indicators with respect to themes of the policy laws](./src/#statistical_analysis.R)
      
      I conducted a series of regression-based hypothesis tests to identify which climate policy variables had statistically significant effects on a range of socioeconomic outcomes (indicators in the WDI dataset), while controlling for country and year effects. Additiionally I assessed whether the model assumptions were met for each case. Through the quantitative analysis, I studided which climate policies appeared to influence development outcomes and whether those effects are statistically reliable and in the expected direction.
 
### Additional experimentation
It was observed that the policy summaries in the CCLW dataset often lacked sufficient detail for reliable text analysis and theme identification. To address this, I developed a Python script to efficiently scrape full content from the associated URLs of policy documents and web pages ([web scrapping](./src/#extract_content.py)). These sources are frequently more secure than typical websites, requiring careful handling. To further process the often lengthy documents (sometimes even exceeding 30 pages), I also created a [summarization script](./src/#summarizer_script.py) to distill the content into more manageable and informative summaries. This approach provides a more resource-efficient and content-rich alternative to the original summaries, enabling deeper and more meaningful computational text analysis. To streamline the workflow, I documented a [shell script](./src/#run_summarizer.sh) that automates the execution of both steps.

## Reproducing the Analysis

1) **Clone the repository**:
     ```bash
     git clone https://github.com/yourusername/yourproject.git
     cd yourproject
2) **Create a virtual environment**:
     ```bash
     python -m venv yourpythonenvironment
3) **Activate the environment**:
     ```bash
     source yourpythonenvironment/bin/activate (Linux)
     yourpythonenvironment/Scripts/activate (Windows)
4) **Install dependencies**:
     ```bash
     pip install -r requirements.txt
5) **Download datasets**:
     - Place CCLW in ```data/CCLW.csv```
     - Place WDI data in ```data/WDI_Data.csv``` (filtered to 2015+)
6) **(Optional) Run content extraction and summarization**:
     - To scrape full text from policy URLs and generate summaries:
       ```bash
       bash run_summarizer.sh
   Note: It was **NOT** used in our analysis for this task.
7) **Run the climate policy classification using the kernel environment as yourpythonenvironment**:
     ```bash
     jupyter notebook src/climate_policy_indicator.ipynb
8) **View outputs**:
     - Check visualization in ```images/```
9) **Prepare merged dataset for hypothesis testing**:
     ```bash
     jupyter notebook src/preparing_data.ipynb```

- Resulting dataset stored in ```output_data/panel_dataset.csv```
10) **Run hypothesis testing analysis**:
     - Open and execute
     ```bash
     Rscript src/statistical_analysis.R
11) **View results**:
     - Outputs will be saved in outputs/ or printed in notebooks/scripts
     - Figures generated for review and stored in ```images/```

   


