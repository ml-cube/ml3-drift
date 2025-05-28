<p align="center">
  <img src="https://raw.githubusercontent.com/ml-cube/ml3-drift/refs/heads/main/images/logo.png" alt="ml3-drift" height="50%">
  <h3 align="center">
    Easy-to-embed drift detection
  </h3>
</p>


`ml3-drift` is an open source AI library that provides seamless integration of drift detection algorithms into existing Machine Learning and AI frameworks. The purpose is to simplify the implementation process and enable developers to easily incorporate drift detection into their pipelines.

## ✅ Supported Frameworks

These are the frameworks we currently support. We will add much more in the future! Let us know if you are interested in a specific framework!

| Framework | How |  Example   |
| ----------| ------ | ------ |
| <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="scikit-learn" height="20"> <span style="white-space: nowrap;">[`scikit-learn`](https://scikit-learn.org/stable/)</span> | Provides a [scikit-learn compatible](https://scikit-learn.org/stable/developers/develop.html) drift detector that integrates easily into existing scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). |   [Mixed data monitoring](examples/sklearn/mixed_data_monitoring.py)                                            |
| <span style="white-space: nowrap;"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="huggingface" height="20"> [`transformers`](https://github.com/huggingface/transformers)</span> (by [`huggingface`](https://huggingface.co/)) | A minimal wrapper for the [Pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) object that looks like a Pipeline, behaves like a Pipeline but also monitors the output of the wrapped Pipeline.. Works with any [feature extraction](https://huggingface.co/tasks/feature-extraction) pipeline, both images and text. |   [Text data monitoring](examples/huggingface/text_embedding_monitoring.py)                                            |


## 🛠️ Usage

`ml3-drift` components are designed to be easily integrated into your existing code. You should be able to use them with minimal changes to your code.

Here is a simple example with `scikit-learn`:

```python
import logging

import numpy as np
from ml3_drift.sklearn.univariate.ks import KSDriftDetector
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml3_drift.callbacks.base import logger_callback
from functools import partial

logger = logging.getLogger(__name__)

# Define your pipeline as usual, but also add a drift detector.
# The detector accepts a list of functions to be called when a drift is detected.
# The first argument of the function is a dataclass containing some information
# about the drift (check it out in ml3_drift/callbacks/models.py).
drift_detector = KSDriftDetector(
    callbacks=[
        partial(
            logger_callback,
            logger=logger,
            level=logging.CRITICAL,
        )
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", StandardScaler()),
        ("monitoring", drift_detector),
        (
            "model",
            DecisionTreeRegressor(),
        ),
    ]
)

# When fitting the pipeline, the drift detector will
# save the training data as reference data.
# No effect on the model training.
pipeline = pipeline.fit(X_train, y_train)

# When making predictions, the drift detector will
# check if the incoming data is similar to the reference data
# and execute the callback you specified if a drift is detected.
predictions = pipeline.predict(X_test)
```

The example callback we provided will simply log a message when a drift is detected. For instance:

```
Drift detected on feature at index 0 by drift detector KSDriftDetector.
 p-value = 2.2027963703339932e-07
 Threshold = 0.005
```

You can find other examples in the [examples](examples) folder. For more information, please refer to the [documentation](https://ml-cube.github.io/ml3-drift/).


## 📦 Installation

`ml3-drift` is available on [PyPI](https://pypi.org/project/ml3-drift/) and supports Python versions from 3.10 to 3.13, included.

The integration with the different frameworks are managed through extra dependencies. The plain `ml3-drift` package comes without any dependency, which means that you need to specify the framework you want to use when installing the package. Otherwise, if you are just experimenting, you can install the package with all the available extras.

You can use pip:

```bash
pip install ml3-drift[all] # install all the dependencies
pip install ml3-drift[sklearn] # install only sklearn dependency
pip install ml3-drift[huggingface] # install huggingface dependency
```

or [uv](https://docs.astral.sh/uv)

```bash
uv add ml3-drift --all-extras # install all the dependencies
uv add ml3-drift --extra sklearn # install only sklearn dependency
uv add ml3-drift --extra huggingface # install only huggingface dependency
```


## ❓ What is drift detection? Why do we need it?

Machine Learning algorithms rely on the assumption that the data used during training comes from the same distribution as the data seen in production.

However, this assumption rarely holds true in the real world, where conditions are dynamic and constantly evolving. These distributional changes, if not addressed properly, can lead to a decline in model performance. This, in turn, can result in inaccurate predictions or estimations, potentially harming the business.

Drift Detection, often referred to as Monitoring, is the process of continuously tracking the performance of a model and the distribution of the data it is operating on. The objective is to quickly detect any changes in data distribution or behavior, so that corrective actions can be taken in a timely manner.


## 😅 Yet another drift detection library?

Not really. While there are many *great* open source drift detection libraries out there ([`nannyml`](https://github.com/nannyml/nannyml), [`river`](https://github.com/online-ml/river), [`evidently`](https://github.com/evidentlyai/evidently) just to name a few), we observed a lack of standardization in the API and misalignments with common ML interfaces. Our goal is to offer known drift detection algorithms behind a single unified API, tailored for relevant ML and AI frameworks such as [`scikit-learn`](https://scikit-learn.org/stable/) and [`huggingface`](https://github.com/huggingface/transformers). Hopefully, this won't be the [15th competing standard](https://xkcd.com/927/) 😉.

## 🚀 Contributing

We welcome contributions to `ml3-drift`! Since we are at a very early stage, we are looking forward to feedbacks, ideas and bug reports. Feel free to open an [issue](https://github.com/ml-cube/ml3-drift/issues) if you have any questions or suggestions.

### Local Development

These are the steps you need to follow to set up your local development environment.

We use [uv](https://docs.astral.sh/uv) as package manager and [just](https://github.com/casey/just) as command runner. Once you have both installed, you can clone the repository and run the following command to set up your development environment:

```bash
just dev-sync
```

The previous command will install all optional dependencies. If you want to install only one of them, run:

```bash
just dev-sync-extra extra-to-install
# for instance, just dev-sync-extra sklearn
```

Make sure you install the pre-commit hooks by running:

```bash
just install-hooks
```

To format your code, lint it and run tests, you can use the following command:

```bash
just validate
```

Notice that tests are run according to the installed libraries. If you don't have scikit-learn installed, all tests related to it will be skipped.

## 📜 License

This project is licensed under the terms of the Apache License Version 2.0. For more details, please refer to the [LICENSE](LICENSE) file. All contributions to this project will be distributed under the same license.

## 👥 Authors

This project was originally developed at [ML cube](https://www.mlcube.com/home_2/) and has been open-sourced to benefit the ML community, from which we deeply welcome contributions.

While `ml3-drift` provides easy to use and integrated drift detection algorithms, companies requiring enterprise-grade monitoring, advanced analytics and insights capabilities might be interested in trying out our product, the ML cube Platform.

The ML cube Platform ([website](https://www.mlcube.com/platform/), [docs](https://ml-cube.github.io/ml3-platform-docs/)) is a comprehensive end-to-end ModelOps framework that helps you trust your AI models and GenAI applications by providing several functionalities, such as data and model monitoring, drift root cause analysis, performance-safe model retraining and LLM security. It can both be used during the development phase of your models and in production, to ensure that your models are performing as expected and quickly detect and understand any issues that may arise.

If you'd like to learn more about our product or wonder how we can help you with your AI projects, visit our websites or contact us at [info@mlcube.com](mailto:info@mlcube.com).
