#!/bin/bash

echo 'Starting to extract content through web and file scrapping'
python src/extract_content.py --input_file './data/CCLW' --output_file './output_data/content_data' 

sleep 60 * 60

echo 'Starting the summarizer script to summarize the collected content'
python src/summarizer_script.py