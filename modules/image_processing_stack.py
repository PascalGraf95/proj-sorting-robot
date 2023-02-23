mean_vals = get_mean_patch_value(image)
correction_factors = get_white_balance_parameters(mean_vals)
image = correct_image_white_balance(image, correction_factors)
image = equalize_histograms(image, True, 1.4, (8, 8))
image = get_image_patch(image, (500, 850), 700)
image = cv2.medianBlur(image, 21)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)
image = cv2.bitwise_not(image)
kernel = np.ones((3, 3), np.uint8)
image = cv2.erode(image, kernel, iterations=1))
kernel = np.ones((3, 3), np.uint8)
image = cv2.dilate(image, kernel, iterations=1))
contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = []
for c in contours:
    if cv2.contourArea(c) >= 1000:
        filtered_contours.append(c)
                    