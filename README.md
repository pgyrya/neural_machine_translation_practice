# Practice Project - Neural Machine Translation
## Abstract
In this practice project I explore application of neural machine translation to the practical problem of text normalization that is an integral part of translating free-form written text into spoken form. Following recent research papers, I contrast two recurrent neural network architectures that are both capable enough to generate adequate translations, agile enough to be train-able using limited resources, and intuitive enough to offer practical ideas for applications in multiple fields. I also overview training approach that helped me navigate through some of the hurdles of network tuning, and avoid settling on ‘trivial’ translations that may be associated with local minima of loss function or areas of flat gradient.

## Acknowledgements
I have leveraged established Python libraries such as Keras and TensorFlow to implement neural translation algorithm discussed in this report. For neural network architecture, I have leveraged code base shared on the following GitHub repositories of [seq2seq](https://github.com/farizrahman4u/seq2seq) and [recurrentshop](https://github.com/farizrahman4u/recurrentshop) shared as open source by Fariz Rahman. 

After leveraging this library, I have also discovered another interesting and illustrative implementation of similar approach at GitHub location of [keras-attention](https://github.com/datalogue/keras-attention) shared by Datalogue.

Finally, the research and development community effort that led to development and implementation of deep learning techniques, and make them easily accessible on the Web has been an invaluable help to my learning project. A few references are included as links in the report.
