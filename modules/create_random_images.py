import numpy as np
import cv2
import colorsys
import tqdm


def main():
    for i in tqdm.tqdm(range(1000)):
        random_hue = np.random.random()
        random_color = [round(x * 255) for x in colorsys.hsv_to_rgb(random_hue, 1.0, 1.0)]

        random_image = np.random.randint(0, 2, (10, 10, 1), dtype=np.uint8)*255
        random_image = np.repeat(random_image, 3, axis=-1)
        black_pixels_mask = np.all(random_image == [0, 0, 0], axis=-1)
        random_image[black_pixels_mask] = random_color
        random_image = cv2.resize(random_image, (224, 224), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite("./random_images/{:05d}.png".format(i), random_image)
        # cv2.imshow("Winname", random_image)
        # cv2.waitKey(10)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()