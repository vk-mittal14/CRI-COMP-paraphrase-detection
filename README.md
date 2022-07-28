Install required packages.

`pip install pandas tqdm torch pickle5 transformers==4.20.1 nltk==3.6.5 sentence-transformers==2.2.2`


Unzip the testing data 

`unzip indices_testing.zip`

Run the following commands to reproduce the results. 

```
python make_data_csv.py
python  bert_sim.py
python calculate_sim.py
python filter_sim.py
```
The above steps will produce various files, please use the `testdata_final_submission_80_wordprocess59.csv` for evaluation.



If you face any issues in reproducing the results, please email to vivek.mittal051@gmail.com
