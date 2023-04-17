SEG = imread("photo2_ret.jpg");
GT = imread("photo2label.jpg");
% binarize(0~255 to 0~1)
SEG = imbinarize(SEG, 0.3);
GT = imbinarize(GT, 0.1);

% IOU
iou = Cal_IOU(SEG, GT);
% precision recall F1 
precision = Precision(SEG, GT);
recall = Recall(SEG, GT);
f1 = F1(SEG, GT);
% Dice
dr = Dice_Ratio(SEG, GT);


function iou = Cal_IOU(SEG, GT)
    [rows, cols] = size(SEG);
    
    % To count the number of pixels with a pixel value of 1 in the tag GT and the segmentation result SEG
    % Initialize the value
    label_area = 0; % The area of the label image (the total number of pixels)
    seg_area = 0;   % Area of segmentation result
    intersection_area = 0; % Intersection area
    combine_area = 0;      % Joint area of two areas

    % Calculate the area of each part
    for i = 1: rows
        for j = 1: cols
            if GT(i, j)==1 && SEG(i, j)==1
                intersection_area = intersection_area + 1;
                label_area = label_area + 1;
                seg_area = seg_area + 1;
            elseif GT(i, j)==1 && SEG(i, j)~=1
                label_area = label_area + 1;
            elseif GT(i, j)~=1 && SEG(i, j)==1
                seg_area = seg_area + 1;
            end
        end
    end

    combine_area = combine_area + label_area + seg_area - intersection_area;

    iou = double(intersection_area) / double(combine_area);
    fprintf('IOU = %f\n', iou);
end

% precision
function precision = Precision(SEG, GT)
    precision = double(sum(SEG(:) & GT(:))) / double(sum(SEG(:)));
end

% recall
function recall = Recall(SEG, GT)
    recall = double(sum(SEG(:) & GT(:))) / double(sum(GT(:)));
end

% F1 score
function f1 = F1(SEG, GT)
    precision = Precision(SEG, GT);
    recall = Recall(SEG, GT);
    f1 = double(precision * recall * 2) / double(precision + recall);
    fprintf("recall = %f\n", recall);
    fprintf("precision = %f\n", precision);
    fprintf("f1 = %f\n", f1);
end

% Dice_Ratio
function dr = Dice_Ratio(SEG, GT)
    dr = 2*double(sum(uint8(SEG(:) & GT(:)))) / double(sum(uint8(SEG(:))) + sum(uint8(GT(:))));
    fprintf("Dice = %f\n", dr);
end