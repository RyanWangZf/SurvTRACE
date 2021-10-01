## ⭐SurvTRACE: Transformers for Survival Analysis with Competing Events

This repo provides the implementation of **SurvTRACE** for survival analysis. It is easy to use with only the following codes:

```python
from survtrace.dataset import load_data
from survtrace.evaluate_utils import Evaluator
from survtrace.train_utils import Trainer
from survtrace.config import STConfig
from survtrace.model import SurvTraceSingle

# use METABRIC dataset
STConfig['data'] = 'metabric'
df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig)

# initialize model
model = SurvTraceSingle(STConfig)

# execute training
trainer = Trainer(model)
trainer.fit((df_train, df_y_train), (df_val, df_y_val))

# evaluating
evaluator = Evaluator(df, df_train.index)
evaluator.eval(model, (df_test, df_y_test))

print("done!")
```



### 🔥See the demo

Please refer to **experiment_metabric.ipynb** and **experiment_support.ipynb** !



### 🔥How to config the environment

Use our pre-saved conda environment!

```shell
conda env create --name survtrace --file=survtrace.yml
conda activate survtrace
```

or try to install from the requirement.txt

```shell
pip3 install -r requirements.txt
```

