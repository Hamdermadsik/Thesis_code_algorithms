function binary = otsu_and_binarize_image(img)
    % Binarize the image using Otsu's thresholding
    % Input:
    %   img - grayscale image to be binarized
    % Output:
    %   binary - binary image (1 for foreground, 0 for background)
    
    % Apply Otsu's thresholding
    thresh = graythresh(img);  % Returns a normalized threshold between 0 and 1
    thresh = thresh * max(img(:));  % Scale threshold based on the image's intensity range

    % Initialize binary image with zeros (same size as input image)
    binary = zeros(size(img));
    thresh = thresh*1.2;
    % Set pixels to 1 where the image intensity is above the threshold
    binary(img > thresh) = 1;
    
    binary = preprocess_image(binary, 1);

    % Optional: plot the original and binary images side-by-side
    % figure;
    % subplot(1, 2, 1); imshow(img, []); title('Original Image');
    % subplot(1, 2, 2); imshow(binary, []); title('Binary Image');
end

function closedImage = preprocess_image(binary_image, radius)
    % Apply morphological closing to fill small gaps in the binary image
    % Input:
    %   binary_image - binary image to be processed
    %   radius - radius of the disk-shaped structuring element (default is 1)
    % Output:
    %   closedImage - binary image after morphological closing

    if nargin < 2
        radius = 1; % Default radius if not provided
    end

    % Create a disk-shaped structuring element
    se = strel('disk', radius);

    % Apply morphological closing
    closedImage = imclose(binary_image, se);
end