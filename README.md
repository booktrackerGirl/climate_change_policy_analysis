This is a repository for the pre-interview task, as assigned for the interview stage in the **Lancet Countdown: Tracking Progress on Health and Climate Change (Working Group on Public and Political Engagement, WG5)** project.

## Task 1 overview (ClimatePolicyIndicatorDevelopment)
Using the Climate Change Laws of the World database, develop a quantitative indicator that captures a specific aspect of national climate change policy or ambition based on the text of laws or policies. You're encouraged to use advanced text analysis techniques (e.g., machine learning, topic modeling, sentiment analysis, embeddings), though simpler methods are acceptable if well justified. The goal is to extract meaningful policy insights from the textual data.

- <b>Data</b>: [Climate Change Laws of the World](./data/#CCLW.csv)
     - 8974 climate-related laws, rgulations, policies and directives.
     - Spans 196+ countries and territories.
     - Each record includes researcher-written summaries, classifications (e.g. adaptation, mitigation, disaster risk management), metadata like enactment dates, coverage type, and legislative processes. 
- <b>Code</b>: [Jupyter Notebook on classifying the climate policy indicator using text-based analysis](./src/#climate_policy_indicator.ipynb)
     - Using the text in the summaries, we pre-process the text accordingly on both English and non-English texts. Upon which, I use sentence embeddings to create features as an input to a simple neural network, which classifies the topic or theme of the policies in a multi-label format. I also create an additional feature called ambition score which can be used in the embedding.

## Task 2 overview (Statistical/ComputationalAnalysis)
Select one or more credible health indicators (e.g., from WHO, Global Burden of Disease, or World Bank) and justify your choice based on its relevance to climate policy or political engagement. Then, conduct a quantitative analysis—such as a panel data model—to examine the relationship between your climate policy indicator and the selected health outcome(s).

- <b>Data</b>:
    1) [Climate Change Laws of the World](./data/#CCLW.csv)
    2) [World Development Indicators](./data/#WDI_Data.csv)
          - WDI dataset (which is a flagship project of World Bank) which contain environmental, health, economic, demographic and social indicators that spans over time and approximately 217 countries and over 40 country groupings. The dataset I used is a filtered subset of the full dataset, and contain information from 2015 onwards.
- <b>Code</b>:
    - [Data Linkage code](./src/#preparing_data.ipynb)
      Using the country codes and years from the CCLW dataset, we create one hot encoding of the policy themes and calculate the number of policy themes traversed by each country along the years. Using the same information from the WDI dataset, we join them to prepare them for our hypothesis testing.
    - [Hypothesis testing of the significance of the indicators by policy laws](./src/#statistical_analysis.R)
 
  ## Additional experimentation
