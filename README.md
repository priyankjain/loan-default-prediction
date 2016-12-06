# loan-default-prediction
CS-578 project of loan default prediction
Author: Priyank Jain

##Environment Setup:
To run the experiments as mentioned in the report, please make sure you Python environment setup with following versions and libraries. Anaconda python is good to have, since it has a lot of libraries in-built.

```
Python 3.5.2 |Anaconda custom (64-bit)
numpy==1.11.2
pandas==0.19.0
scikit-learn==0.18
matplotlib==1.5.1
```

###Running The Experiments:
```shell
cd sources/
python analysis.py
python main.py
```

The `analysis.py` script is used to run experiment 1 as per the report. It would generate charts in reports folder and consume data from dataser folder. This script would complete within 2 minutes.

The `main.py` script is used to run experiments 2, 3, and 4 as per the report. It would generate charts and reports in reports folder and consume data from dataset folder. Certain classifier configurations and outputs would be written to console by this script. This script takes around 6 hours in total to run.

