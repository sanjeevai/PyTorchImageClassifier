import json
import argparse
import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image


# define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, help='Name of checkpoint file to use for predicting')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--image_path', type=str, help='Path of image file which will be used for prediction')
parser.add_argument('--category_names', type=str, help='JSON file containing mapping of number to labels')
parser.add_argument('--top_k', type=int, help='Return top k predictions')

args = parser.parse_args()
# load the checkpoint
def load_checkpoint(filepath):
    ckp = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = ckp["classifier"]
    model.load_state_dict(ckp["state_dict"])
    model.class_to_idx = ckp['class_to_idx']
    model.cuda()
    return model

# Take image file as an input and predict the class for it
def predict(image_path, top_k=5):

    # Use command line arguments if specified
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    if args.gpu:
        gpu = args.gpu
    if args.image_path:
        image_path = args.image_path
    if args.category_names:
        category_names = args.category_names
    if args.top_k:
        top_k = args.top_k
    
    # load the checkpoint
    model = load_checkpoint(checkpoint_path)

    # use GPU if available

    if gpu & torch.cuda.is_available():
        model.cuda()

    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    model_input = model_input.cuda()
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(args.top_k)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]

    # label mapping from file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image

print(predict(args.image_path))