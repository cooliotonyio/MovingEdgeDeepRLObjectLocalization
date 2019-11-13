from PIL import ImageDraw

def draw_bbox(img, bbox, width = 2, fill = "green"):
    '''
    Takes in PIL.Image, bbox info and draws bounding box
    '''
    draw = ImageDraw.Draw(img)
    p1 = tuple(bbox[:2])
    p2 = (p1[0] + bbox[2], p1[1])
    p3 = (p2[0], p2[1] + bbox[3])
    p4 = (p1[0], p3[1])
    draw.line([p1,p2,p3,p4,p1], width=width, fill=fill)
    return img

def get_area(bbox):
    '''
    Gets area of bbox
    '''
    return bbox[4] * bbox[3]

def get_iou(bbox1, bbox2):
    bb1_x1, bb1_y1, bb2_x1, bb2_y1 = bbox1[0], bbox1[1], bbox2[0], bbox2[1]
    bb1_x2, bb1_y2, bb2_x2, bb2_y2 = bb1_x1 + bbox1[2], bb1_y1 + bbox1[3], bb2_x1 + bbox2[2], bb2_y1 + bbox2[3]

    x_left, x_right = max(bb1_x1, bb2_x1), min(bb1_x2, bb2_x2)
    y_top, y_bottom = max(bb1_y1, bb2_y1), min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = get_area(bbox1)
    bb2_area = get_area(bbox2)
    
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    # TODO: Remove assert statements
    assert iou >= 0.0
    assert iou <= 1.0
    return iou