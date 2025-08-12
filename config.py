# Dictionary storing network parameters.
params = {
    'batch_size': 128,# Batch size.
    'num_epochs': 30,# Number of epochs to train for.
    'd_learning_rate' : 2e-4,# Learning rate for discriminator.
    'g_learning_rate' : 2e-4,# Learning rate for generator.
    #'learning_rate': 2e-4,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 10,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'CIFAR10'}# Dataset to use. Choose from {MNIST, CelebA，CIFAR10}. CASE MUST MATCH EXACTLY!!!!!