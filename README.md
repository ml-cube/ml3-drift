<p align="center">
  <img src="https://raw.githubusercontent.com/ml-cube/ml3-drift/refs/heads/main/images/logo.png" alt="ml3-drift" height="50%">
  <h3 align="center">
    Your Drift Detection Toolbox
  </h3>
</p>

<p align="center">
  <a href="https://github.com/ml-cube/ml3-drift/actions"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/ml-cube/ml3-drift/publish.yaml"></a>
  <a href="https://github.com/ml-cube/ml3-drift/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/ml-cube/ml3-drift.svg?color=blue"></a>
  <a href="https://pypi.org/project/ml3-drift/" target="_blank"><img src="https://img.shields.io/pypi/v/ml3-drift" alt="PyPi"></a>
  <a href="https://pepy.tech/project/ml3-drift" target="_blank"><img src="https://pepy.tech/badge/ml3-drift" alt="PyPi Downloads"></a>
  <!--
  TODO: UNCOMMENT once we have proper documentation<a href="https://ml-cube.github.io/ml3-drift/"><img alt="Documentation" src="https://img.shields.io/website?url=https%3A%2F%2Fml-cube.github.io%2Fml3-drift%2F&up_message=online&down_message=offline
"></a>
!-->

</p>

`ml3-drift` is an open source AI library that provides seamless integration of drift detection algorithms and techniques into Data Science Workflows. It does so by providing 3 main modules:
- 🤖 **Monitoring Algorithms**: a collection of univariate and multivariate drift detection algorithms, both for batch and online settings. Some of them are implemented from scratch, while others are wrappers around existing libraries.
- 🧩 **Framework Integrations**: this components allows the integration of drift detection algorithms into existing Machine Learning and AI frameworks, such as `scikit-learn` and `transformers (huggingface)`. This enables developers to easily add drift detection to their existing pipelines with minimal code changes.
- 📊 **Distribution Analyzers**: a set of tools for analyzing the distribution of data and detecting drifts over time in a given static dataset.

## ✅ Features

### Monitoring Algorithms

Monitoring algorithms are the building blocks of the library. They expose a common interface (`fit` - `detect`) that makes them easy to use and swappable in different contexts.

While the other modules leverage these algorithms, they can also be used in a standalone fashion. This is particularly useful when you want to experiment with different algorithms or use them in a custom way.

This table summarizes the algorithms we currently support, along with their characteristics. We will add more in the future, let us know if you are interested in a specific algorithm!

| Algorithm | Mode | Data Type | Data Dimensionality | Notes |
| --------- | ---- | --------- | ------------------- | ----- |
| [Kolmogorov-Smirnov (KS) Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) | Batch | Continuous | Univariate | Uses scipy [`ks_2samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) method |
| [Chi-Square](https://en.wikipedia.org/wiki/Chi-squared_test) | Batch | Categorical | Univariate | Uses scipy [`chi2_contingency`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html) method |
| [Bonferroni Correction Algorithm](https://en.wikipedia.org/wiki/Bonferroni_correction) | Batch | Continuous / Categorical | Multivariate | Wraps a univariate monitoring algorithm and applies Bonferroni correction to execute multivariate detection. |
| [ADWIN](https://www.researchgate.net/publication/220907178_Learning_from_Time-Changing_Data_with_Adaptive_Windowing) | Online | Continuous | Univariate | Wraps [River library](https://riverml.xyz/dev/api/drift/ADWIN/) class |
| [KSWIN](https://arxiv.org/abs/2007.05432) | Online | Continuous | Univariate | Wraps [River library](https://riverml.xyz/dev/api/drift/KSWIN/) class |
| [PageHinkley](https://www.scirp.org/reference/referencespapers?referenceid=2474051) | Online | Continuous | Univariate | Wraps [River library](https://riverml.xyz/dev/api/drift/PageHinkley/) class |

#### Example

Here is a simple example of how to use the `KSDriftDetector` algorithm in a standalone fashion:

```python
import numpy as np
from ml3_drift.monitoring.algorithms.batch.ks import KSAlgorithm
from ml3_drift.callbacks.base import print_callback

# Create some sample data
np.random.seed(42)
reference_data = np.random.normal(0, 1, 1000)
current_data = np.random.normal(0.5, 1, 1000)

# Initialize the drift detector. It's possible to pass a list of callbacks
# that will be executed when a drift is detected.
drift_detector = KSAlgorithm(callbacks=[print_callback])
# Fit the detector on the reference data
drift_detector.fit(reference_data)
# Detect drift on the current data
drift = drift_detector.detect(current_data)
```

The example callback we provided will simply print a message when a drift is detected. For instance:

```
Drift Detected, drift info: {'test_statistic': 4.2252283893369713e-26, 'statistic_threshold': 0.005}
```

### Framework Integrations

These are the frameworks we currently support. We will add more in the future, let us know if you are interested in a specific framework!

| Framework | How |  Example   |
| ----------| ------ | ------ |
| <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="scikit-learn" height="20"> <span style="white-space: nowrap;">[`scikit-learn`](https://scikit-learn.org/stable/)</span> | Provides a [scikit-learn compatible](https://scikit-learn.org/stable/developers/develop.html) drift detector that integrates easily into existing scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). |   [Mixed data monitoring](examples/sklearn/mixed_data_monitoring.py)                                            |
| <span style="white-space: nowrap;"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="huggingface" height="20"> [`transformers`](https://github.com/huggingface/transformers)</span> (by [`huggingface`](https://huggingface.co/)) | A minimal wrapper for the [Pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) object that looks like a Pipeline, behaves like a Pipeline but also monitors the output of the wrapped Pipeline. Works with any [feature extraction](https://huggingface.co/tasks/feature-extraction) pipeline, both images and text. |   [Text data monitoring](examples/huggingface/text_embedding_monitoring.py)                                            |

#### Example

`ml3-drift` framework components are designed to be easily integrated into your existing code. You should be able to use them with minimal changes to your code.

Here is a simple example with `scikit-learn`:

```python
from ml3_drift.monitoring.algorithms.batch.ks import KSAlgorithm
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml3_drift.callbacks.base import print_callback

from ml3_drift.sklearn.base import SklearnDriftDetector

# Define your pipeline as usual, but also add a drift detector.
# The detector accepts a list of functions to be called when a drift is detected.
# The first argument of the function is a dataclass containing some information
# about the drift (check it out in ml3_drift/callbacks/models.py).
drift_detector = KSAlgorithm(callbacks=[print_callback])

pipeline = Pipeline(
    steps=[
        ("preprocessor", StandardScaler()),
        # Wrap the drift detector in the SklearnDriftDetector
        # to make it compatible with sklearn pipelines.
        ("monitoring", SklearnDriftDetector(drift_detector)),
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

The example callback we provided will simply print a message when a drift is detected. For instance:

```
Drift Detected, drift info: {'test_statistic': 3.35710076793659e-13, 'statistic_threshold': 0.005}
```

You can find other examples in the [examples](https://github.com/ml-cube/ml3-drift/tree/main/examples) folder. For more information, please refer to the [documentation](https://ml-cube.github.io/ml3-drift/).

### Distribution Analyzers

This module provides tools for analyzing the distributions present in a static dataset. This helps understanding the data and highlighting potential issues that might arise when using it to train a model.

We currently provide 2 analyzers: one is batch-based, the other is online-based. They work in a similar way but have a slightly different approach. They both accept a couple of monitoring algorithms, one for continuous features and one for categorical features, but:
- the [Batch analyzer](src/ml3_drift/analysis/analyzer/batch.py) 📦 splits the dataset into subsets of a given size and compares each subset with the others in other to identify macro-batches of data belonging to different distributions. It is also able to identify recurring distributions, i.e. a distribution that appears multiple times (but not in a contiguous way) in the dataset.
- the [Online analyzer](src/ml3_drift/analysis/analyzer/online.py) 🌊 processes the dataset in a sequential way, creating sliding windows of contiguous samples. If a drift is detected, the dataset is "split" and the algorithms reset on the new data. The output is a more granular view of the distribution changes over time. However, it is not able to identify recurring distributions. Notice this class hasn't been tested yet, use it at your own risk :).

#### Example

This is a simple example of how to use the Batch analyzer:

```python
from ml3_drift.analysis.analyzer.batch import BatchDataDriftAnalyzer
from ml3_drift.monitoring.algorithms.batch.bonferroni import (
    BonferroniCorrectionAlgorithm,
)
from ml3_drift.monitoring.algorithms.batch.chi_square import ChiSquareAlgorithm
from ml3_drift.monitoring.algorithms.batch.ks import KSAlgorithm


continuous_algo = BonferroniCorrectionAlgorithm(algorithm=KSAlgorithm(p_value=0.05))
categorical_algo = BonferroniCorrectionAlgorithm(
    algorithm=ChiSquareAlgorithm(p_value=0.05)
)

analyzer = BatchDataDriftAnalyzer(
    continuous_monitoring_algorithm=continuous_algo,
    categorical_monitoring_algorithm=categorical_algo,
    batch_size=50,
)

X = ...
y = ...

# Run analyze
report = analyzer.analyze(
    X,
    y=y,
    continuous_columns=[0, 1],
    categorical_columns=[2, 3],
    y_categorical=False,
)

print(report)
```

The output is a report containing the indexes of the samples partitioning the datasets into different distributions, along with indication of recurring distributions. For instance:

```python
# Indicates that the dataset is composed by 3 different distributions.
# Also, the first and the third distributions are similar.
Report(concepts=[(0, 299), (300, 599), (600, 899)], same_distributions={0: [2]})
```


## 📦 Installation

`ml3-drift` is available on [PyPI](https://pypi.org/project/ml3-drift/) and supports Python versions from 3.10 to 3.13, included.

All integrations with external libraries are managed through extra dependencies. The plain `ml3-drift` package comes without any dependency, which means that you need to specify the framework you want to use when installing the package. Otherwise, if you are just experimenting, you can install the package with all the available extras.

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

Not really. There are many *great* open source drift detection libraries out there ([`nannyml`](https://github.com/nannyml/nannyml), [`river`](https://github.com/online-ml/river), [`evidently`](https://github.com/evidentlyai/evidently) just to name a few), our goal is a bit different. While we also offer some algorithms implementations (even though some of them are just wrappers around these libraries), we mainly focus on the integration of drift detection practices into existing ML and AI workflows. During our experience in the field, we observed a lack of standardization in the API and misalignments with common ML interfaces. We want to fill this gap by providing a library that is easy to use, flexible and that can be easily integrated into existing and working systems.

Hopefully, this won't be the [15th competing standard](https://xkcd.com/927/) 😉.

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

Notice that tests are run according to the installed libraries. For instance, if you don't have scikit-learn installed, all tests which requires it will be skipped.

## 📜 License

This project is licensed under the terms of the Apache License Version 2.0. For more details, please refer to the [LICENSE](LICENSE) file. All contributions to this project will be distributed under the same license.

## 👥 Authors

This project was originally developed at [ML cube](https://www.mlcube.com/home_2/) and has been open-sourced to benefit the ML community, from which we deeply welcome contributions.

While `ml3-drift` provides easy to use and integrated drift detection algorithms, companies requiring enterprise-grade monitoring, advanced analytics and insights capabilities might be interested in trying out our product, the ML cube Platform.

The ML cube Platform ([website](https://www.mlcube.com/platform/), [docs](https://ml-cube.github.io/ml3-platform-docs/)) is a comprehensive end-to-end ModelOps framework that helps you trust your AI models and GenAI applications by providing several functionalities, such as data and model monitoring, drift root cause analysis, performance-safe model retraining and LLM security. It can both be used during the development phase of your models and in production, to ensure that your models are performing as expected and quickly detect and understand any issues that may arise.

If you'd like to learn more about our product or wonder how we can help you with your AI projects, visit our websites or contact us at [info@mlcube.com](mailto:info@mlcube.com).
