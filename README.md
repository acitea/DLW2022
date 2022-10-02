# DLW2022

## Packages needed
nltk, torch, numpy

# Usage
Run the main.py file

## Inspiration
Having been reported that teachers in Singapore are overworked, and having less than ideal work-life balance, we sought to think of how we could lighten their load. Setting questions, test papers and practices for their students are some of the mentally strenuous work that they do, hence we planned on assisting them on that.

## What _(we conceptualised)_ that it would do
It would be able to generate questions based on certain topics that were requested through Natural Language Generation (NLG), and in the process, it would be able to categorise the topics of questions that is fed to it. 

## What we realistically could manage
Currently, it can only barely categorise questions being fed into it into PSLE Maths topics. 

## How we built it
From maths PSLE-level test papers found online, their questions were labelled and fed into a machine-learning NLP model trainer using the bag of words approach. 

## Challenges we ran into
For being relatively inexperienced in python, let alone in AI/ML, jumping straight into learning how to do NLP was an extremely massive mountain to scale up. Nevertheless, it was a fruitful few days digesting loads of information on NLP.  

## What's next for InfiSeries
The roadmap was that our model will be improved to also recognise linguistic syntaxes to aid in its accuracy of categorising the questions fed into it. Next, it would use NLG based on those syntaxes, paired with BERT and WordNets, to finally be able to do what we envisioned it does. 
Furthermore, we have hopes that it won't be restricted to maths questions; it may be able to work for other subjects like english or more, but currently, it is completely out of our level of expertise.
