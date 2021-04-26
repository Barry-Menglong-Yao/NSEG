Graph based Neural Sentence Ordering
=====================================================================

### Installation

The following packages are needed:

- Python == 3.6
- Pytorch == 1.5.0
- torchtext == 0.3
- Stanford POS tagger or Dependency Parser
- Glove (100 dim)

### Dataset Format
*.lower: each line is a document: sentence_0 <eos> sentence_1 <eos> sentence_2

*.eg:
entity1:i-r means entity1 is in the sentence_i and its role is r.

Other datasets are easy to access and process.
We also recommand a high-quality dataset for sentence ordering, ROC story.

### Preprocessing

Use a dependency parser to get POS and syntax

Select the word as entity if the POS is noun

Find the nsubj and dobj to get the roles ( or just use a POS tagger and ignore the roles if you think the dependency parser is time-consuming)

### Training and Evaluation
bash run.sh




### 
nsubj: nominal subject
A nominal subject is a noun phrase which is the syntactic subject of a clause. The governor of this relation
might not always be a verb: when the verb is a copular verb, the root of the clause is the complement of
the copular verb, which can be an adjective or noun.
“Clinton defeated Dole” nsubj(defeated, Clinton)
“The baby is cute” nsubj(cute, baby)