#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# new entity label
LABELS = ["S ML alg", "US ML alg", "ML soft"]

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
// accuracy
    (
        "If we have a separate test set we can evaluate performance on this in order to estimate the accuracy of our method.",
        {"entities": [(92, 100, "EVM")]}
    ),
    (
        "The accuracy of an MC approximation increases with sample size.",
        {"entities": [(4, 12, "EVM")]}
    ),
    (
        "In high dimensional problems we might prefer a method that only depends on a subset of the features for reasons of accuracy and interpretability.",
        {"entities": [(115, 123, "EVM")]}
    ),
    (
        "In machine learning we often care more about predictive accuracy than in interpreting the parameters of our models.",
        {"entities": [(56, 64, "EVM")]}
    ),
    (
        "However accuracy is not the only important factor when choosing a method.",
        {"entities": [(8, 16, "EVM")]}
    ),
// cross validation
    (
        "A simple but popular solution to this is to use cross validation (CV). The idea is simple: we split the training data into K folds; then for each fold k ∈ {1 . . .  K} we train on all the folds but the k’th and test on the k’th in a round-robin fashion",
        {"entities": [(48, 64, "EVM")]}
    ),
    (
        "It is common to use K = 5; this is called 5-fold CV. If we set K = N  then we get a method called leave-one out cross validation or LOOCV.",
        {"entities": [(112, 128, "EVM")]}
    ),
    (
        "We can use methods such as cross validation to empirically choose the best method for our particular problem.",
        {"entities": [(27, 43, "EVM")]}
    ),
    (
        "The principle problem with cross validation is that it is slow since we have to fit the model multiple times.",
        {"entities": [(27, 43, "EVM")]}
    ),
    (
        "Use cross validation to choose the strength of the 2 regularizer.",
        {"entities": [(4, 20, "EVM")]}
    ),
    (
        "In supervised learning we can always use cross validation to select between non-probabilistic models of different complexity but this is not the case with unsupervised learning.",
        {"entities": [(41, 57, "EVM")]}
    ),
    (
        "This is likely to be much faster than cross validation especially if we have many hyper-parameters (e.g. as in ARD).",
        {"entities": [(38, 54, "EVM")]}
    ),
// likelihood
    (
        "Use grid-search over a range of K’s using as an objective function cross-validated likelihood.",
        {"entities": [(83, 93, "EVM")]}
    ),
    (
        "That is they can use likelihood models of the form p(x t:t+l |z t = k d t = l) which generate l correlated observations if the duration in state k is for l time steps.",
        {"entities": [(21, 31, "EVM")]}
    ),
    (
        "It makes more sense to try to approximate the smoothed distribution rather than the backwards likelihood term.",
        {"entities": [(94, 104, "EVM")]}
    ),
    (
        "The gradient of the log likelihood can be rewritten as the expected feature vector according to the empirical distribution minus the model’s expectation of the feature vector.",
        {"entities": [(24, 34, "EVM")]}
    ),
    (
        "We call this algorithm stochastic maximum likelihood or SML.",
        {"entities": [(42, 52, "EVM")]}
    ),
    (
        "Nevertheless coordinate descent can be slow. An alternative method is to update all the parameters at once by simply following the gradient of the likelihood.",
        {"entities": [(147, 157, "EVM")]}
    ),
// recall
    (
        "From this table we can compute the true positive rate (TPR) also known as the sensitivity recall or hit rate.",
        {"entities": [(90, 96, "EVM")]}
    ),
    (
        "Precision measures what fraction of our detections are actually positive and recall measures what fraction of the positives we actually detected.",
        {"entities": [(77, 83, "EVM")]}
    ),
    (
        "Alternatively one can quote the precision for a fixed recall level such as the precision of the first K = 10 entities.",
        {"entities": [(54, 60, "EVM")]}
    ),
    (
        "The method had a precision of 66% when the recall was set to 10%; while low this is substantially more than rival variable-selection methods such as lasso and elastic net which were only slightly above chance.",
        {"entities": [(43, 49, "EVM")]}
    ),
// Alexa
    (
        "This week I read through a history of everything I've said to Alexa and it felt a little bit like reading an old diary.",
        {"entities": [(62, 67, "MLA")]}
    ),
    (
        "There are more than 100m Alexa-enabled devices in our homes.",
        {"entities": [(25, 30, "MLA")]}
    ),
    (
        "Amazon does a great job of giving you control over your privacy with Alexa.",
        {"entities": [(69, 74, "MLA")]}
    ),
    (
        "When you speak to Alexa a recording of what you asked Alexa is sent to Amazon.",
        {"entities": [(18, 23, "MLA")]}
    ),
// NLP
    (
        "NLP has been considered a subdiscipline of Artificial Intelligence.",
        {"entities": [(0, 3, "MLA")]}
    ),
    (
        "Natural Language Processing (NLP) is a major area of artificial intelligence research which in its turn serves as a field of application and interaction of a number of other traditional AI areas.",
        {"entities": [(29, 32, "MLA")]}
    ),
    (
        "Natural Language Processing (NLP) is a major area of artificial intelligence research which in its turn serves as a field of application and interaction of a number of other traditional AI areas.",
        {"entities": [(0, 27, "MLA")]}
    ),
    (
        "The ontology is done using NLP technique where semantics relationships defined in WordNet.",
        {"entities": [(27, 30, "MLA")]}
    ),
    (
        "RU-EVAL is a biennial event organized in order to estimate the state of the art in Russian NLP resources methods and toolkits and to compare various methods and principles implemented for Russian.",
        {"entities": [(91, 94, "MLA")]}
    ),
    (
        "However there are barriers that must be addressed by organizers to enable task-takers to isolate specific NLP subtasks for focused research.",
        {"entities": [(106, 109, "MLA")]}
    ),
// autonomous
    (
        "Two branches of the trend towards" agents" that are gaining currency are interface agents software that actively assists a user in operating an interactive interface and autonomous agents software that takes action without user intervention and operates concurrently.",
        {"entities": [(170, 180, "MLA")]}
    ),
    (
        "One category of research in Artificial Life is concerned with modeling and building so-called adaptive autonomous agents.",
        {"entities": [(103, 113, "MLA")]}
    ),
    (
        "This book deals with an important topic in distributed AI: the coordination of autonomous agents' activities.",
        {"entities": [(79, 89, "MLA")]}
    ),
    (
        "Developing autonomous or driver-assistance systems for complex urban traffic poses new algorithmic and system-architecture challenges.",
        {"entities": [(11, 21, "MLA")]}
    ),
    (
        "An autonomous floor-cleaning robot comprises a self-adjusting cleaning head subsystem that includes a dual-stage brush assembly having counter-rotating asymmetric brushes and an adjacent but independent vacuum assembly.",
        {"entities": [(3, 13, "MLA")]}
    ),
    (
        "AAFID was the first architecture that proposed the use of autonomous agents for doing intrusion detection.",
        {"entities": [(58, 68, "MLA")]}
    ),
// bioinformatics
    (
        "This review article aims to provide an overview of the ways in which techniques from artificial intelligence can be usefully employed in bioinformatics both for modelling biological data and for making new discoveries.",
        {"entities": [(137, 151, "MLA")]}
    ),
    (
        "Artificial intelligence (AI) has increasingly gained attention in bioinformatics research and computational molecular biology.",
        {"entities": [(66, 80, "MLA")]}
    ),
    (
        "In this review the theory and main principles of the SVM approach are outlined and successful applications in traditional areas of bioinformatics research.",
        {"entities": [(131, 145, "MLA")]}
    ),
    (
        "Soft computing is make several latent in bioinformatics especially by generating low-cost low precision (approximate) good solutions.",
        {"entities": [(41, 55, "MLA")]}
    ),
    (
        "It has a wide spectrum of applications such as natural language processing search engines medical diagnosis bioinformatics and more.",
        {"entities": [(108, 122, "MLA")]}
    ),
// pattern recognition
    (
        "This paper presents an electromyographic (EMG) pattern recognition method to identify motion commands for the control of a prosthetic arm by evidence accumulation based on artificial intelligence with multiple parameters.",
        {"entities": [(47, 66, "MLA")]}
    ),
    (
        "The techniques may be classified broadly into two categories—the conventional pattern recognition approach and the artificial intelligence (AI) based approach.",
        {"entities": [(78, 97, "MLA")]}
    ),
    (
        "We sought to test the hypothesis that a novel 2-dimensional echocardiographic image analysis system using artificial intelligence-learned pattern recognition can rapidly and reproducibly calculate ejection fraction (EF).",
        {"entities": [(138, 157, "MLA")]}
    ),
    (
        "This paper reports the use of a variety of pattern recognition techniques such as the learning machine and the Fisher discriminant.",
        {"entities": [(43, 62, "MLA")]}
    ),
    (
        "However my focus will not be on these types of pattern-recognition problems.",
        {"entities": [(47,66, "MLA")]}
    ),
// recomendations
    (
        "The avatar searches available information and makes recommendations to the user based on data.",
        {"entities": [(52,67,"MLA")]}
    ),
    (
        "Learning from GPS history data for collaborative recommendation.",
        {"entities": [(49,63,"MLA")]}
    ),
    (
        "A learning method was employed for Web pages recommendations and book recommendations in Mooney.",
        {"entities": [(45,60,"MLA")]}
    ),
// feature selection
    (
        "One common approach to tackling both of these problems is to perform feature selection to remove “irrelevant” features that do not help much with the classification problem.",
        {"entities": [(69, 86, "MLP")]}
    ),
    (
        "We introduced the topic of feature selection in Section 3.5.4 where we discussed methods for finding input variables which had high mutual information with the output.",
        {"entities": [(27, 44, "MLP")]}
    ),
    (
        "Feature selection in this context is equivalent to selecting a subset of the training examples which can help reduce overfitting and computational cost.",
        {"entities": [(0,17,"MLP")]}
    ),
    (
        "Note that the topic of feature selection and sparsity is currently one of the most active areas of machine learning/ statistics.",
        {"entities": [(23, 40, "MLP")]}
    ),
    (
        "To improve computational and statistical performance some feature selection was performed.",
        {"entities": [(58, 75, "MLP")]}
    ),
    (
        "We can create a challenging feature selection problem. In the experiments below we add 5 extra dummy variables.",
        {"entities": [(28, 45, "MLP")]}
    ),
// imputation
    (
        "The goal of imputation is to infer plausible values for the missing entries.",
        {"entities": [(12, 22, "MLP")]}
    ),
    (
        "An interesting example of an imputation-like task is known as image inpainting.",
        {"entities": [(29, 39, "MLP")]}
    ),
    (
        "Nevertheless the method can sometimes give reasonable results if there is not much missing data and it is a useful method for data imputation.",
        {"entities": [(131, 141, "MLP")]}
    ),
    (
        "As an example of this procedure in action let us reconsider the imputation problem from Section 4.3.2.3 which had N = 100 10-dimensional data cases with 50% missing data.",
        {"entities": [(64, 74, "MLP")]}
    ),
    (
        "Another interesting example of an imputation-like task is known as collaborative filtering.",
        {"entities": [(34, 44, "MLP")]}
    ),
// normalization
    (
        "The first term is just the normalization constant required to ensure the distribution sums to 1.",
        {"entities": [(27, 40, "MLP")]}
    ),
    (
        "The normalization constant only exists (and hence the pdf is only well defined) if ν > D − 1.",
        {"entities": [(4, 17, "MLP")]}
    ),
    (
        "So assuming the relevant normalization constants are tractable we have an easy way to compute the marginal likelihood.",
        {"entities": [(25, 38, "MLP")]}
    ),
    (
        "Since all the potentials are locally normalized since they are CPDs there is no need for a global normalization constant so Z = 1.",
        {"entities": [(98, 111, "MLP")]}
    ),
    (
        "Since all the potentials are locally normalized since they are CPDs there is no need for a global normalization constant so Z = 1.",
        {"entities": [(37, 47, "MLP")]}
    ),
    (
        "Hence satsifying normalization and local consistency is enough to define a valid distribution for any tree. Hence μ ∈ M(T ) as well.",
        {"entities": [(17, 30, "MLP")]}
    ),
// pipeline
    (
        "This article reviews the evaluation and optimization of the preprocessing steps for bloodoxygenation-level-dependent (BOLD) functional magnetic resonance imaging (fMRI).",
        {"entities": []}
    ),
    (
        "However there is little consensus on the optimal choice of data preprocessing steps to minimize these effects.",
        {"entities": []}
    ),
    (
        "It has been established thatthe chosen preprocessing steps (or \"pipeline\") may significantly affect fMRI results.",
        {"entities": [(64, 72, "MLP")]}
    ),
    (
        "The automated analysis pipeline comprises data import normalization replica merging quality diagnostics and data export.",
        {"entities": [(23, 31, "MLP")]}
    ),
    (
        "The apparatus is formed as a pipeline having a translation and scaling section",
        {"entities": [(29, 37, "MLP")]}
    ),
    (
        "To address these challenges we developed an automated software pipeline called Rnnotator.",
        {"entities": [(63, 71, "MLP")]}
    ),
// caffe
    (
        "We will use some Python code and a popular open source deep learning framework called Caffe to build the classifier.",
        {"entities": [(86,91,"MLS")]}
    ),
    (
        "Use tail -f model_1_train.log to view Caffe's progress.",
        {"entities": [(38,43,"MLS")]}
    ),
    (
        "The caffe "tools/extra/parse_log.sh" file requires a small change to use on OS X.",
        {"entities": [(4, 9, "MLS")]}
    ),
    (
        "I followed Caffe's tutorial on LeNet MNIST using GPU and it worked great.",
        {"entities": [(11, 16, "MLS")]}
    ),
    (
        "Alternatively Caffe has built in a function called iter_size.",
        {"entities": [(14, 19, "MLS")]}
    ),
    (
        "Caffe estimates the gradient (more accurately) weights are updated and the process continues.",
        {"entities": [(0, 5, "MLS")]}
    ),
    (
        "The feature iter_size is a Caffe function per se but you are correct that it is an option that you set in the solver protobuf file.",
        {"entities": [(27, 32, "MLS")]}
    ),
// keras
    (
        "Define your model using the easy to use interface of Keras.",
        {"entities": [(53, 58, "MLS")]}
    ),
    (
        "You can use the simple intuitive API provided by Keras to create your models.",
        {"entities": [(49, 54, "MLS")]}
    ),
    (
        "The Keras API is modular Pythonic and super easy to use.",
        {"entities": [(4, 9, "MLS")]}
    ),
    (
        "If you’re comfortable writing code using pure Keras go for it and keep doing it.",
        {"entities": [(46, 51, "MLS")]}
    ),
    (
        "Using Keras in deep learning allows for easy and fast prototyping as well as running seamlessly on CPU and GPU.",
        {"entities": [(6, 11, "MLS")]}
    ),
    (
        "The main advantages of Keras are described below.",
        {"entities": [(23, 28, "MLS")]}
    ),
// scikit learn
    (
        "One of the best known is Scikit-Learn a package that provides efficient versions of a large number of common algorithms.",
        {"entities": [(25, 37, "MLS")]}
    ),
    (
        "A benefit of this uniformity is that once you understand the basic use and syntax of Scikit-Learn for one type of model switching to a new model or algorithm is very straightforward.",
        {"entities": [(85, 97, "MLS")]}
    ),
    (
        "The best way to think about data within Scikit-Learn is in terms of tables of data.",
        {"entities": [(40, 52, "MLS")]}
    ),
    (
        "While some Scikit-Learn estimators do handle multiple target values in the form of a two-dimensional [n_samples n_targets] target array we will primarily be working with the common case of a one-dimensional target array.",
        {"entities": [(11, 23, "MLS")]}
    ),
    (
        "Many machine learning tasks can be expressed as sequences of more fundamental algorithms and Scikit-Learn makes use of this wherever possible.",
        {"entities": [(93, 105, "MLS")]}
    ),
// spacy
    (
        "spaCy excels at large-scale information extraction tasks.",
        {"entities": [(0, 5, "MLS")]}
    ),
    (
        "Independent research in 2015 found spaCy to be the fastest in the world.",
        {"entities": [(35, 40, "MLS")]}
    ),
    (
        "With spaCy you can easily construct linguistically sophisticated statistical models for a variety of NLP problems.",
        {"entities": [(5, 10, "MLS")]}
    ),
    (
        "The new pretrain command teaches spaCy's CNN model to predict words based on their context.",
        {"entities": [(33, 38, "MLS")]}
    ),
    (
        "spaCy is an open-source software library for advanced natural language processing.",
        {"entities": [(0, 5, "MLS")]}
    ),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="animal", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    for label in LABELS:
        ner.add_label(label)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "Do you like horses the way Joe does?"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
