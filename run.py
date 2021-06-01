from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms

# OUR
from utils import ImageandPatchs, ImageDataset, generatemask, getGF_fromintegral, calculateprocessingres, rgb2gray,\
    applyGridpatch

# MIDAS
import midas.utils
from midas.models.midas_net import MidasNet
from midas.models.transforms import Resize, NormalizeImage, PrepareForNet

# PIX2PIX : MERGE NET
from pix2pix.options.test_options import TestOptions
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

import time
import os
import torch
import cv2
import numpy as np
import argparse
import warnings
warnings.simplefilter('ignore', np.RankWarning)

# select device
device = torch.device("cuda")
print("device: %s" % device)

# Global variables
pix2pixmodel = None
midasmodel = None
srlnet = None
factor = None
whole_size_threshold = 3000  # R_max from the paper
GPU_threshold = 1600 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted 

# MAIN PART OF OUR METHOD
def run(dataset, option):

    # Load merge network
    opt = TestOptions().parse()
    global pix2pixmodel
    pix2pixmodel = Pix2Pix4DepthModel(opt)
    pix2pixmodel.save_dir = './pix2pix/checkpoints/mergemodel'
    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()

    # Decide which depth estimation network to load
    if option.depthNet == 0:
        midas_model_path = "midas/model.pt"
        global midasmodel
        midasmodel = MidasNet(midas_model_path, non_negative=True)
        midasmodel.to(device)
        midasmodel.eval()
    elif option.depthNet == 1:
        global srlnet
        srlnet = DepthNet.DepthNet()
        srlnet = torch.nn.DataParallel(srlnet, device_ids=[0]).cuda()
        checkpoint = torch.load('structuredrl/model.pth.tar')
        srlnet.load_state_dict(checkpoint['state_dict'])
        srlnet.eval()

    # Generating required directories
    result_dir = option.output_dir
    os.makedirs(result_dir, exist_ok=True)

    if option.savewholeest:
        whole_est_outputpath = option.output_dir + '_wholeimage'
        os.makedirs(whole_est_outputpath, exist_ok=True)

    if option.savepatchs:
        patchped_est_outputpath = option.output_dir + '_patchest'
        os.makedirs(patchped_est_outputpath, exist_ok=True)

    # Generate mask used to smoothly blend the local pathc estimations to the base estimate.
    # It is arbitrarily large to avoid artifacts during rescaling for each crop.
    mask_org = generatemask((3000, 3000))
    mask = mask_org.copy()

    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = 0.2
    if option.R0:
        r_threshold_value = 0
    elif option.R20:
        r_threshold_value = 0.2

    # Go through all images in input directory
    print("start processing")
    for image_ind, images in enumerate(dataset):
        print('processing image', image_ind, ':', images.name)

        # Load image from dataset
        img = images.rgb_image
        input_resolution = img.shape

        scale_threshold = 3  # Allows up-scaling with a scale up to 3

        # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
        # supplementary material.
        whole_image_optimal_size, patch_scale = calculateprocessingres(img, option.net_receptive_field_size,
                                                                       r_threshold_value, scale_threshold,
                                                                       whole_size_threshold)

        print('\t wholeImage being processed in :', whole_image_optimal_size)

        # Generate the base estimate using the double estimation.
        whole_estimate = doubleestimate(img, option.net_receptive_field_size, whole_image_optimal_size,
                                        option.pix2pixsize, option.depthNet)
        if option.R0 or option.R20:
            path = os.path.join(result_dir, images.name)
            midas.utils.write_depth(path, whole_estimate, bits=2)
            continue

        # Output double estimation if required
        if option.savewholeest:
            path = os.path.join(whole_est_outputpath, images.name)
            if option.output_resolution == 1:
                midas.utils.write_depth(path,
                                        cv2.resize(whole_estimate, (input_resolution[1], input_resolution[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2)
            else:
                midas.utils.write_depth(path, whole_estimate, bits=2)

        # Ensures that the image is not up-scaled more than scale_threshold before the local refinement process starts.
        if 2*whole_image_optimal_size > scale_threshold*max(img.shape[:2]):
            whole_image_optimal_size = int(scale_threshold*max(img.shape[:2])/4)

        threshold_temp = min(whole_size_threshold, scale_threshold*max(img.shape[:2]))

        # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
        # small high-density regions of the image.
        global factor
        factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / threshold_temp), 0.2)
        print('Adjust factor is:', 1/factor)

        # Compute the target resolution.
        if img.shape[0] > img.shape[1]:
            a = 2*whole_image_optimal_size
            b = round(2*whole_image_optimal_size*img.shape[1]/img.shape[0])
        else:
            a = round(2*whole_image_optimal_size*img.shape[0]/img.shape[1])
            b = 2*whole_image_optimal_size

        img = cv2.resize(img, (int(b/factor), int(a/factor)), interpolation=cv2.INTER_CUBIC)

        base_size = option.net_receptive_field_size*2

        # Extract selected patches for local refinement
        patchset = generatepatchs(img, base_size)

        print('Target resolution: ', img.shape)

        # Computing a scale in case user prompted to generate the results as the same resolution of the input.
        # Notice that our method output resolution is independent of the input resolution and this parameter will only
        # enable a scaling operation during the local patch merge implementation to generate results with the same resolution
        # as the input.
        if option.output_resolution == 1:
            mergein_scale = input_resolution[0] / img.shape[0]
            print('Dynamicly change merged-in resolution; scale:', mergein_scale)
        else:
            mergein_scale = 1

        imageandpatchs = ImageandPatchs(option.data_dir, images.name, patchset, img, mergein_scale)
        whole_estimate_resized = cv2.resize(whole_estimate, (int(img.shape[1]*mergein_scale),
                                            int(img.shape[0]*mergein_scale)), interpolation=cv2.INTER_CUBIC)
        imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
        imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

        print('\t Resulted depthmap res will be :', whole_estimate_resized.shape[:2])
        print('patchs to process: '+str(len(imageandpatchs)))

        # Enumerate through all patches, generate their estimations and refining the base estimate.
        for patch_ind in range(len(imageandpatchs)):
            
            # Get patch information
            patch = imageandpatchs[patch_ind] # patch object
            patch_rgb = patch['patch_rgb'] # rgb patch
            patch_whole_estimate_base = patch['patch_whole_estimate_base'] # corresponding patch from base
            rect = patch['rect'] # patch size and location
            patch_id = patch['id'] # patch ID
            org_size = patch_whole_estimate_base.shape # the original size from the unscaled input
            print('\t processing patch', patch_ind, '|', rect)

            # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
            # field size of the network for patches to accelerate the process.
            patch_estimation = doubleestimate(patch_rgb, option.net_receptive_field_size, option.patch_netsize,
                                              option.pix2pixsize, option.depthNet)

            # Output patch estimation if required
            if option.savepatchs:
                path = os.path.join(patchped_est_outputpath, imageandpatchs.name + '_{:04}'.format(patch_id))
                midas.utils.write_depth(path, patch_estimation, bits=2)

            
            patch_estimation = cv2.resize(patch_estimation, (option.pix2pixsize, option.pix2pixsize),
                                          interpolation=cv2.INTER_CUBIC)

            patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (option.pix2pixsize, option.pix2pixsize),
                                                   interpolation=cv2.INTER_CUBIC)

            # Merging the patch estimation into the base estimate using our merge network:
            # We feed the patch estimation and the same region from the updated base estimate to the merge network
            # to generate the target estimate for the corresponding region.
            pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

            # Run merging network
            pix2pixmodel.test()
            visuals = pix2pixmodel.get_current_visuals()

            prediction_mapped = visuals['fake_B']
            prediction_mapped = (prediction_mapped+1)/2
            prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

            mapped = prediction_mapped

            # We use a simple linear polynomial to make sure the result of the merge network would match the values of
            # base estimate
            p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
            merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)

            merged = cv2.resize(merged, org_size, interpolation=cv2.INTER_CUBIC)

            # Get patch size and location
            w1 = rect[0]
            h1 = rect[1]
            w2 = w1 + rect[2]
            h2 = h1 + rect[3]

            # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
            # and resize it to our needed size while merging the patches.
            if mask.shape != org_size:
                mask = cv2.resize(mask_org, org_size, interpolation=cv2.INTER_LINEAR)

            tobemergedto = imageandpatchs.estimation_updated_image

            # Update the whole estimation:
            # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
            # blending at the boundaries of the patch region.
            tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
            imageandpatchs.set_updated_estimate(tobemergedto)

        # Output the result
        path = os.path.join(result_dir, imageandpatchs.name)
        if option.output_resolution == 1:
            midas.utils.write_depth(path,
                                    cv2.resize(imageandpatchs.estimation_updated_image,
                                               (input_resolution[1], input_resolution[0]),
                                               interpolation=cv2.INTER_CUBIC), bits=2)
        else:
            midas.utils.write_depth(path, imageandpatchs.estimation_updated_image, bits=2)

    print("finished")


# Generating local patches to perform the local refinement described in section 6 of the main paper.
def generatepatchs(img, base_size):
    
    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) +\
        np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum()/len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size/2))
    stride = int(round(blsize*0.75))

    # Get initial Grid
    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    print("Selecting patchs ...")
    patch_bound_list = adaptiveselection(grad_integral_image, patch_bound_list, gf)

    # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
    # patch
    patchset = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], 'size'), reverse=True)
    return patchset


# Adaptively select patches
def adaptiveselection(integral_grad, patch_bound_list, gf):
    patchlist = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32/factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = getGF_fromintegral(integral_grad, bbox)/(bbox[2]*bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:

                bbox_test[0] = bbox_test[0] - int(search_step/2)
                bbox_test[1] = bbox_test[1] - int(search_step/2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = getGF_fromintegral(integral_grad, bbox_test)/(bbox_test[2]*bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patchlist[str(count)]['rect'] = bbox
            patchlist[str(count)]['size'] = bbox[2]
            count = count + 1
    
    # Return selected patches
    return patchlist


# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pixsize, net_type):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, net_type)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, net_type)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    # Resize back to input resolution
    prediction_end_res = cv2.resize(prediction_mapped, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return prediction_end_res


# Generate a single-input depth estimation
def singleestimate(img, msize, net_type):
    if msize > GPU_threshold:
        print(" \t \t DEBUG| GPU THRESHOLD REACHED", msize, '--->', GPU_threshold)
        msize = GPU_threshold

    if net_type == 0:
        return estimatemidas(img, msize)
    elif net_type == 1:
        return estimatesrl(img, msize)


# Inference on SGRNet
def estimatesrl(img, msize):
    # SGRNet forward pass script adapted from https://github.com/KexianHust/Structure-Guided-Ranking-Loss
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_resized = cv2.resize(img, (msize, msize), interpolation=cv2.INTER_CUBIC).astype('float32')
    tensor_img = img_transform(img_resized)

    # Forward pass
    input_img = torch.autograd.Variable(tensor_img.cuda().unsqueeze(0), volatile=True)
    with torch.no_grad():
        output = srlnet(input_img)

    # Normalization
    depth = output.squeeze().cpu().data.numpy()
    min_d, max_d = depth.min(), depth.max()
    depth_norm = (depth - min_d) / (max_d - min_d)

    depth_norm = cv2.resize(depth_norm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return depth_norm

# Inference on MiDas-v2
def estimatemidas(img, msize):
    # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2

    transform = Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodel.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True, help='input files directory '
                                                                    'Images can be .png .jpg .tiff')
    parser.add_argument('--output_dir', type=str, required=True, help='result dir. result depth will be png.'
                                                                      ' vides are JMPG as avi')
    parser.add_argument('--savepatchs', type=int, default=0, required=False,
                        help='Activate to save the patch estimations')
    parser.add_argument('--savewholeest', type=int, default=0, required=False,
                        help='Activate to save the base estimations')
    parser.add_argument('--output_resolution', type=int, default=1, required=False,
                        help='0 for results in maximum resolution 1 for resize to input size')
    parser.add_argument('--net_receptive_field_size', type=int, required=False)  # Do not set the value here
    parser.add_argument('--pix2pixsize', type=int, default=1024, required=False)  # Do not change it
    parser.add_argument('--depthNet', type=int, default=0, required=False,
                        help='use to select different base depth networks 0:midas 1:strurturedRL')
    parser.add_argument('--R0', action='store_true')
    parser.add_argument('--R20', action='store_true')
    parser.add_argument('--Final', action='store_true')

    # Check for required input
    option_, _ = parser.parse_known_args()
    print(option_)
    if int(option_.R0) + int(option_.R20) + int(option_.Final) == 0:
        assert False, 'Please activate one of the [R0, R20, Final] options using --[R0]'
    elif int(option_.R0) + int(option_.R20) + int(option_.Final) > 1:
        assert False, 'Please activate only ONE of the [R0, R20, Final] options'

    if option_.depthNet == 1:
        from structuredrl.models import DepthNet

    # Setting each networks receptive field and setting the patch estimation resolution to twice the receptive
    # field size to speed up the local refinement as described in the section 6 of the main paper.
    if option_.depthNet == 0:
        option_.net_receptive_field_size = 384
        option_.patch_netsize = 2*option_.net_receptive_field_size
    elif option_.depthNet == 1:
        option_.net_receptive_field_size = 448
        option_.patch_netsize = 2*option_.net_receptive_field_size
    else:
        assert False, 'depthNet can only be 0,1'

    # Create dataset from input images
    dataset_ = ImageDataset(option_.data_dir, 'test')

    # Run pipeline
    run(dataset_, option_)
