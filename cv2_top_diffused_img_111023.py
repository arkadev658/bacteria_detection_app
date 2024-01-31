import cv2
import numpy as np
import os

input_folder = 'D:\\Manakin image algo\\Manakin images\\Top_diffused_images_augmented_2_labelled\\Top_diffused_images_augmented_2'
output_folder = 'D:\\Manakin image algo\\Manakin images\\Top_diffused_images_augmented_2_labelled'

bact=[[60,100,10],[1,1,253],[80,18,10]]
bact_name=['bact_x','bact_y','bact_z']
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load your image
        img = cv2.imread(input_path)
        
        
        
        # Define the coordinates and dimensions of the region to crop (adjust these as needed)
        #x = 1900
        #y = 1100
        #width = 700
        #height = 260

        # Crop the region of interest
        #img = img[y:y+height, x:x+width]


   
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #add weightage to color channel
        
        # Calculate the mean brightness of the image
        mean_brightness = np.mean(gray)
        
        # Define a brightness threshold to differentiate between bright and dark images
        brightness_threshold = 128  # Adjust this threshold as needed
        
        # Check if the image is considered bright or dark
       
        # bright image processing 
        # Apply bilateral blur to reduce noise
        gray_blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=100, sigmaSpace=75)
        
        # Use adaptive thresholding to find regions of interest (ROIs)
        # Adjust the blockSize and C values as needed for your specific image
        roi_mask = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours in the ROI mask
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        l=0
        
        # Iterate through the contours and find circles
        for contour in contours:
            # Find the area of the contour
            area = cv2.contourArea(contour)
        
            # Adjust this threshold as needed to filter out small or large regions
            if 80 < area < 5000:
                # Fit a circle to the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Define the size of the bounding box relative to the circle radius
                box_size = int(1.5 * radius)    #add the circle
        
                # Calculate the coordinates for the bounding box
                x1 = max(0, center[0] - box_size)
                y1 = max(0, center[1] - box_size)
                x2 = min(img.shape[1], center[0] + box_size)
                y2 = min(img.shape[0], center[1] + box_size)
                
                rect_area = (x2 - x1) * (y2 - y1)
                min_rect_area = 7000 
                max_rect_area = 13000
                greater_side=max((x2-x1),(y2-y1))
                smaller_side=min((x2-x1),(y2-y1))
                if min_rect_area < rect_area and max_rect_area > rect_area and greater_side/smaller_side<1.1:
        
                # Draw the bounding box on the original image
                    #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img, center, radius, (0,0,255) , 2)
                # color extraction
                    rad2=radius*0.7
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)

                    # Generate a circular mask
                    cv2.circle(mask, center, int(rad2), 255, -1)
                    
                    # Extract the circular region from the image using the mask
                    circular_region = cv2.bitwise_and(img, img, mask=mask)
                    
                    b,g,r=cv2.split(circular_region)
                    
                    # Calculate the median pixel value within the circular area
                    #median_pixel_value = np.median(circular_region[circular_region > 0])
                    i=50
                    j=0
                    name=['blue','green','red']
                    for channel in [b,g,r]:
                        test_res=[]
                        median_pixel_value = np.median(channel[channel>0])
                        if median_pixel_value>=bact[l][j]:
                            test_res.append(True)
                        else:
                            test_res.append(False)
                    # Annotate the median value at the top of the outer circle
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = f'{name[j]}={median_pixel_value:.1f}'
                        text_size, _ = cv2.getTextSize(text, font, 0.5, 1)[0]
                        text_position = (center[0],center[1]+i)
                        cv2.putText(img, text, text_position, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        i+=20
                        j+=1
                    if False in test_res:
                        cv2.putText(img, f'negative for {bact_name[l]}', (center[0]-40,center[1]-40), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(img, f'positive for {bact_name[l]}', (center[0]-40,center[1]-40), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    l+=1
                        
                             
        
        cv2.imwrite(output_path, img)
        print(f'Copied {filename} to {output_folder}')