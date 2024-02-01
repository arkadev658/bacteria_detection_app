import streamlit as st
import cv2
import numpy as np



def process_image(image):
    bact = [[60, 100, 10], [1, 1, 253], [80, 18, 10]]
    bact_name = ['bact_x', 'bact_y', 'bact_z']

    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img is None:
        st.warning("Unable to read the uploaded image. Please make sure it is a valid image file.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    brightness_threshold = 128

    gray_blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=100, sigmaSpace=75)
    roi_mask = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    l = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if 80 < area < 5000:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            box_size = int(1.5 * radius)
            x1 = max(0, center[0] - box_size)
            y1 = max(0, center[1] - box_size)
            x2 = min(img.shape[1], center[0] + box_size)
            y2 = min(img.shape[0], center[1] + box_size)

            rect_area = (x2 - x1) * (y2 - y1)
            min_rect_area = 7000
            max_rect_area = 13000
            greater_side = max((x2 - x1), (y2 - y1))
            smaller_side = min((x2 - x1), (y2 - y1))
            if min_rect_area < rect_area < max_rect_area and greater_side / smaller_side < 1.1:
                cv2.circle(img, center, radius, (0, 0, 255), 2)

                rad2 = radius * 0.7
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.circle(mask, center, int(rad2), 255, -1)

                circular_region = cv2.bitwise_and(img, img, mask=mask)
                circular_region = cv2.cvtColor(circular_region,cv2.COLOR_RGB2BGR)

                b, g, r = cv2.split(circular_region)

                i = 50
                j = 0
                name = ['blue', 'green', 'red']
                test_res = []

                for channel in [b, g, r]:
                    median_pixel_value = np.median(channel[channel > 0])
                    if median_pixel_value >= bact[l][j]:
                        test_res.append(True)
                    else:
                        test_res.append(False)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f'{name[j]}={median_pixel_value:.1f}'
                    text_size, _ = cv2.getTextSize(text, font, 0.5, 1)[0]
                    text_position = (center[0], center[1] + i)
                    cv2.putText(img, text, text_position, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    i += 20
                    j += 1

                if False in test_res:
                    cv2.putText(img, f'negative for {bact_name[l]}', (center[0] - 40, center[1] - 40),
                                font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(img, f'positive for {bact_name[l]}', (center[0] - 40, center[1] - 40),
                                font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                l += 1

    return img


def main():
    st.title("Bacteria Detection App")
    st.write("Upload an image, and the app will process it to detect bacteria.")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Process Image"):
            # Process the image
            processed_img = process_image(uploaded_image)

            if processed_img is not None:
                # Display the processed image
                st.image(processed_img, caption="Processed Image", use_column_width=True)


if __name__ == "__main__":
    main()
