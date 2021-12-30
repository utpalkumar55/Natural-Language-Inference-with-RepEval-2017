# Natural-Language-Inference-with-RepEval-2017
Natural Language Inference with RepEval 2017 on Multi-Genre Natural Language Inference (MultiNLI) corpus. This dataset can be found at https://cims.nyu.edu/~sbowman/multinli/. The dataset folder needs to be included to run this project. Also, [fra.txt](https://github.com/utpalkumar55/Natural-Language-Inference-with-RepEval-2017/files/7794688/fra.txt) file needs to be included in the "encoder decoder" folder. This project uses keras and numpy.

This project works using The RepEval 2017 Shared Task model. This model was meant to evaluate Natural Language Model based on sentence encoders. This is more like natural language inference problem over sentence pairs. This project works on on Stanford Natural Language Inference (SNLI) style corpus dataset. Basically, all the sentences are of three classes (contradiction, entailment and neutral). There are 393k pairs sentence of five genres available for training and 20000 pairs of ten genres available for testing a model. There are two sets of testing data. One is matched where training and testing pairs are collected from same sources. Another is mismatched where training and testing pairs are collected from different sources. Examples from the three classes of sentences in the dataset are given below.

![snli](https://user-images.githubusercontent.com/3108754/147729742-25b80ce0-aad5-444e-8854-2822803b5350.JPG)

Project report file contains the details and the results.
