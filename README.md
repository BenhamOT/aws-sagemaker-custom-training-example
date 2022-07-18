# aws-sagemaker-custom-training-example

This repo contains code that can be used to train a custom image classification model on AWS
Sagemaker (not using a Sagemaker built-in model). Although the code is specific to an 
image classification model, the structure can be reused to train other types of models. 

Training in AWS sagemaker is split into 3 types. 

> You can use an AWS off the shelf
build-in algorithm for model training. Examples of these can be found here: 
https://github.com/aws/amazon-sagemaker-examples/tree/main/introduction_to_amazon_algorithms.

> At the other end of the spectrum, you can have complete control over the model training
process by creating your own model training images: 
https://docs.aws.amazon.com/sagemaker/latest/dg/studio-byoi.html

> Finally, there is the option to write your own custom code while utilizing common
ML framework (PyTorch, TensorFlow, MXNet) containers managed by AWS.
Some more example can be found here: https://github.com/aws/amazon-sagemaker-examples/tree/main/frameworks

This repo provides an example of the final option listed above, utilizing using custom
TensorFlow training code and an AWS managed TensorFlow training image.

#### An overview of how it works

The training_notebook.ipynb contains code that can be used in a notebook on sagemaker studio
to set up the model training environment (The S3 location of the data, the IAM permissions,
the ML framework being used, the hyperparameter to be used). The code that determines exactly
what happens during the training process is contained within another script. In this repo there
are two example scripts: mobilenet_v3_sagemaker_training.py and xception_sagemaker_training.py. 
These scripts are mostly identical, the only difference is the underlying model being trained.

The only real sagemaker specific part of the training scripts is the arguments that 
need to be passed in. These are contained with the _parse_args function --> 


    def _parse_args():
        parser = argparse.ArgumentParser()

        # Data, model, and output directories
        # model_dir is always passed in from SageMaker. By default, this is a S3 path under the default bucket.
        parser.add_argument('--model_dir', type=str)
        parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
        
        # hyper-parameters
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--fine-tune-learning-rate", type=float, default=0.00001)
        parser.add_argument("--fine-tune-epochs", type=int, default=10)
        parser.add_argument("--fine-tune-layer", type=int, default=100)
        parser.add_argument("--drop-out-percentage", type=float, default=0.1)
        parser.add_argument("--random-rotation", type=float, default=0.2)
        parser.add_argument("--negative-weight", type=float, default=1.0)
        parser.add_argument("--positive-weight", type=float, default=1.0)
        return parser.parse_known_args()


This allows you to pass in information such as the location of the training and
validation data and any hyperparameters you want to include. 