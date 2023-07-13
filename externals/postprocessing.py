import cv2
import numpy as np
from scipy import ndimage

def post_process(probability, threshold, min_size, hole_size_threshold):
    # Apply threshold to create binary mask
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    # 1. Label the small holes
    labels, num_labels = ndimage.label(np.logical_not(mask))
    hole_sizes = ndimage.sum(np.logical_not(mask), labels, range(num_labels + 1))
    small_holes = hole_sizes < hole_size_threshold
    small_holes_mask = small_holes[labels]
    mask_filled = mask.astype(bool) | small_holes_mask

    # Connected-component labeling - remove large elements
    num_component, component = cv2.connectedComponents(mask_filled.astype(np.uint8))

    # Calculate the size of each component
    component_sizes = np.bincount(component.ravel())[1:]

    # Create a mask of components that are above the minimum size
    mask_large_components = component_sizes >= min_size
    predictions = np.isin(component, np.nonzero(mask_large_components)[0] + 1).astype(np.float32)
    num = np.sum(mask_large_components)

    # predictions = np.zeros_like(probability, np.float32)
    # num = 0
    # component_sizes = []
    # for c in range(1, num_component):
    #     p = (component == c)
    #     component_sizes.append(p.sum())
    #     if p.sum() > min_size:
    #         predictions[p] = 1
    #         num += 1

    return predictions, num, component_sizes
