# Nascent Transcription Encoder Model Greenfield Implementation

This repository contains a new greenfield implementation of my nascent transcription encoder model and using the lessons learned from the past eight months of development period. 
The key differences in this implementation are that we pull data in directly from PySAM tools and use a significantly cleaned up implementation in PyTorch that uses a BERT-liked masked autoencoder model.

# Specification

- Data Loading
    - Takes input from both a FASTA and a BAM or CRAM file.
    - It uses Pi same tools or an equivalent Python library to efficiently pull regions from the genome based on coordinates.
    - The primary function is to take a set of input regions of known size and return those to the fixed size vectors, say 256 base pairs, where sequence is one hot encoded and associated transcription from the SAM/BAM file is per coordinate showing the read coverage on the positive and negative strand.
    - We also want some sort of test suite that allows us to primarily verify the speed of the data loading.
    - The core use of this implementation is to provide fixed size input regions across the genome where I am already providing the center point of these input regions and using that to provide input for a masked BERT-like model.
- Model
    - The model architecture here is pretty simple. So the key idea here is that our input is some set of regions from multiple cell types and run that through a BERT-style masked autoencoder in order to learn cell type's specific features.
    - Input regions tend to be pretty small on the order of 256 base pairs, although we would like to scale them up in the future.
    - Model construction is as follows. 
        - First, we take our sequence and we encode it using some sort of one-dimensional convolutional embedding to put into our transformer. 
        - Second, encode our transcription on both strands and transform that into an embedding that is added to this new residual stream from our sequence. 
        - Third, encode the one-hot encoded identity of cell type and add that as well to the residual stream. 
        - The idea here is that we want to encode all three of these features together in the residual stream so that we can learn cell type specific features downstream using this autooncoder.
        - If we have our input data, then we can run it through a standard bird-style autoencoder. We want to be pretty flexible about the number of layers and number of heads provided here. 
        - The key aspects that we want to implement are that we want a rope-style transformer for this bird autoencoder. And we would also like to use variable random masking where we are masking somewhere between 15 and 40% of the input regions each time so that we are training a model to be spatially aware, preferably in sort of chunks. 
        - Historically, we've had good use using plain old Adam for the optimizer here, although we have also used shampoo, which works okay. Otherwise, the training is pretty bog standard.
        - For loss, this basically breaks down into three different losses that we balance out using VQGAN-style gradient normalization. For the reads, we use normal root-mean-squared reconstruction loss, and we make sure that our reads are normalized so that they sum to one. For the sequence, we use KL divergence loss, and for the cell type, we should just stick with KL divergence loss so that we could use any other single class classification task as well for that. 
        - This might be obvious, but it is important that we renormalize our outputs effectively before putting them into the loss functions, whether that is with softmax or whatever is appropriate for the data type.
    - Sparse Autoencoder:
        - Once we have the initial BERT style model trained, then what we do is we train a sparse autoencoder to learn interpretable features.
        - Using this sparse autoencoder, we pull the top activating features by +3 sigma from fitting a normal distribution of the activations, and we use those to generate attributions for each of our three input features, so sequence, cell type, as well as reads.
        - Using those attributions, we then can run our sequence characteristics, particularly through TF-Medisco and see if our learn motifs are mechanistically linked with known sets of transcriptional factor regulatory characteristics. 
        - The key idea here is being able to use the attributions for cell type as well since we are going to be learning attributions for each of the one-hot encoded cell types to pull features that are seemingly learned as highly indicative of one cell type or another. 
        - The SAE should be trained using a Top-K approach, tunable K, with a total size of 256 neurons, as this is more backed by recent research than the ReLU version originally pioneered by Anthropic.
