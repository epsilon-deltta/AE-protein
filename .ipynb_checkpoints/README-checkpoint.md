## Usage

```
# start in project base directory
python -m ae.train -m ae0
python -m ae.evaluate -p ./ae/models/ae0_80.pt -th 0.004
python -m ae.predict -p ./ae/models/ae0_80.pt -d  ./data/sample.txt -o ./ae/output.csv
python -m ae.predict -s WNQHNHFDNV -p ./ae/models/ae0_80.pt
# WNQHNHFDNV > WNSFNHFPNV
```
```
python train.py -m ae0
python evaluate.py -p ./models/ae0_80.pt -th 0.004
python predict.py -m ./models/ae0_80.pt -d  ./data/sample.txt -o ./output.csv
python predict.py -s WNQHNHFDNV -m ./models/ae0_80.pt
```
## Experiment results

Label 0: {'mean': 0.0063138764946244616, 'std': 0.00416648373169449, 'q1': 0.0031339654233306646, 'q2': 0.005561222322285175, 'q3': 0.008553454652428627}
Label 1: {'mean': 0.003230442024151987, 'std': 0.0025557172705587706, 'q1': 0.0015390751650556922, 'q2': 0.002339747967198491, 'q3': 0.003943340503610671}

- ae0

Threshold: 0.0025
 {'acc': 0.7479049319961533, 'recall': 0.5456560283687943, 'precision': 0.6031357177853993, 'f1': 0.5729578775890156, 'confusion': array([[4213,  810],
       [1025, 1231]])}

Threshold: 0.0027
 {'acc': 0.7439208682511333, 'recall': 0.5868794326241135, 'precision': 0.5868794326241135, 'f1': 0.5868794326241135, 'confusion': array([[4091,  932],
       [ 932, 1324]])}

Threshold: 0.003
 {'acc': 0.730732243440033, 'recall': 0.6467198581560284, 'precision': 0.5564454614797865, 'f1': 0.5981959819598195, 'confusion': array([[3860, 1163],
       [ 797, 1459]])}

Threshold: 0.0033
 {'acc': 0.7153455144937492, 'recall': 0.6861702127659575, 'precision': 0.5315934065934066, 'f1': 0.5990712074303406, 'confusion': array([[3659, 1364],
       [ 708, 1548]])}

Threshold: 0.0035
 {'acc': 0.7051792828685259, 'recall': 0.7092198581560284, 'precision': 0.517799352750809, 'f1': 0.5985783763561542, 'confusion': array([[3533, 1490],
       [ 656, 1600]])}

Threshold: 0.0037
 {'acc': 0.6954251957686496, 'recall': 0.7322695035460993, 'precision': 0.5059724349157734, 'f1': 0.5984423111755117, 'confusion': array([[3410, 1613],
       [ 604, 1652]])}

Threshold: 0.004
 {'acc': 0.6764665476026926, 'recall': 0.7557624113475178, 'precision': 0.48589341692789967, 'f1': 0.5915004336513443, 'confusion': array([[3219, 1804],
       [ 551, 1705]])}

Threshold: 0.0045
 {'acc': 0.6511883500480835, 'recall': 0.785904255319149, 'precision': 0.4630451815095325, 'f1': 0.5827444535743631, 'confusion': array([[2967, 2056],
       [ 483, 1773]])}
