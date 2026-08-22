import numpy as np
import numpy.linalg as linalg
import matplotlib
import matplotlib.image as img
from PIL import Image
import threading

def decompose_channels(image_path) -> list:
    imageR = np.asarray(Image.open(image_path)).copy()
    imageG = imageR.copy()
    imageB = imageR.copy()
    images = [imageR, imageG, imageB]
    for i in range(len(images)):
        for row in images[i]:
            for col in row:
                if i == 0:
                    col[1] = 0
                    col[2] = 0
                if i == 1:
                    col[0] = 0
                    col[2] = 0
                if i == 2:
                    col[0] = 0
                    col[1] = 0
    return images

def recompose_channels(R, G, B):
    return np.asarray([[[R[m][n], G[m][n], B[m][n]] for n in range(len(R[m]))] for m in range(len(R))])

def svd_trim(M, rank):
    U, svalues, Vt = linalg.svd(M)
    svalues = svalues[0:rank]
    S = np.zeros((len(M), len(M[0])), dtype=float)
    S[:len(svalues), :len(svalues)] = np.diag(svalues)
    return np.asarray(np.dot(U, np.dot(S, Vt)))

def save_image(M, name):
    print("Saving image "+f"{name}.jpg")
    print("Shape: "+ str(M.shape))
    print("Type: "+ str(type(M)))
    if M.dtype != np.uint8:
        M = M.astype(np.uint8)
    Image.fromarray(M).save(f"{name}.jpg")

def main():
    phases = [5, 10, 50, 200]
    images = decompose_channels("test_image.jpg")

    # Phase 0 - no compression
    RED_0 = images[0][:, :, 0]
    GREEN_0 = images[1][:, :, 1]
    BLUE_0 = images[2][:, :, 2]

    RED_IMG = recompose_channels(RED_0, np.zeros_like(GREEN_0), np.zeros_like(BLUE_0))
    GREEN_IMG = recompose_channels(np.zeros_like(RED_0), GREEN_0, np.zeros_like(BLUE_0))
    BLUE_IMG = recompose_channels(np.zeros_like(RED_0), np.zeros_like(GREEN_0), BLUE_0)
    
    save_image(RED_IMG, "phase0-RED")
    save_image(GREEN_IMG, "phase0-GREEN")
    save_image(BLUE_IMG, "phase0-BLUE")
    save_image(recompose_channels(RED_0, GREEN_0, BLUE_0), "phase0-RECOMBINED")


    # Next phases (actually using the compression)
    for i in range(len(phases)):
        RED = svd_trim(RED_0, phases[i])
        BLUE = svd_trim(BLUE_0, phases[i])
        GREEN = svd_trim(GREEN_0, phases[i])

        RED_IMG = recompose_channels(RED, np.zeros_like(GREEN), np.zeros_like(BLUE))
        GREEN_IMG = recompose_channels(np.zeros_like(RED), GREEN, np.zeros_like(BLUE))
        BLUE_IMG = recompose_channels(np.zeros_like(RED), np.zeros_like(GREEN), BLUE)

        save_image(RED_IMG, f"phase{i+1}-RED")
        save_image(GREEN_IMG, f"phase{i+1}-GREEN")
        save_image(BLUE_IMG, f"phase{i+1}-BLUE")
        save_image(recompose_channels(RED, GREEN, BLUE), f"phase{i+1}-RECOMBINED")
main()
