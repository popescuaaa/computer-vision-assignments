"""
    Autoencoders are neural networks that can be used to reduce the data into a low dimensional
    latent space by stacking multiple non-linear transformations(layers). They have a encoder-decoder architecture.
    The encoder maps the input to latent space and decoder reconstructs the input. They are trained using
    back propagation for accurate reconstruction of the input. In the latent space has lower dimensions than the input,
    autoencoders can be used for dimensionality reduction. By intuition, these low dimensional latent variables should
    encode most important features of the input since they are capable of reconstructing it.

    Comparison
    - PCA is essentially a linear transformation but Auto-encoders are capable of modelling complex non linear functions
    - PCA features are totally linearly uncorrelated with each other since features are projections onto the orthogonal
    basis. But autoencoded features might have correlations since they are just trained for accurate reconstruction.
    - PCA is faster and computationally cheaper than autoencoders.
    - A single layered autoencoder with a linear activation function is very similar to PCA.
    - Autoencoder is prone to overfitting due to high number of parameters.
        - (though regularization and careful design can avoid this)

    In this experiment I will be using a simple autoencoder with a linear activation function, which is a good choice for
    this problem, as it is very similar to PCA. The main purpose of this experiment is to see how the performance of the
    autoencoder changes the accuracy of the classifier.


"""
