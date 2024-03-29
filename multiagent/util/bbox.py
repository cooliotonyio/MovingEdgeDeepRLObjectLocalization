from PIL import ImageDraw

def draw_bbox(img, bbox, width = 2, fill = "green"):
    """Takes in PIL.Image, bbox info and draws bounding box"""
    draw = ImageDraw.Draw(img)
    p1, p2, p3, p4 = bbox_to_points(bbox)
    draw.line([p1,p2,p3,p4,p1], width=width, fill=fill)
    return img

def draw_cross(img, bbox, fill="black"):
    draw = ImageDraw.Draw(img)

    p1, p2, p3, p4 = bbox_to_points(bbox)

    xcenter = (p1[0] + p2[0]) * 0.5
    width_size = int((p2[0] - p1[0]) * 0.2)
    ycenter = (p2[1] + p3[1]) * 0.5
    height_size = int((p3[1] - p2[1]) * 0.2)

    draw.line(((xcenter, p1[1]), (xcenter, p3[1])), width=width_size, fill=fill)
    draw.line(((p1[0], ycenter), (p3[0], ycenter)), width=height_size, fill=fill)
    return img
    
def get_area(bbox):
    """Gets area of bbox"""
    return bbox[2] * bbox[3]

def get_iou(bbox1, bbox2):
    """
    Returns the IoU (Intersection over Union) of bbox1 and bbox2
    iou(b1, b2) = area(intersection(b1, b2)) / area(union(b1, b2))
    """
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

    return iou

def bbox_to_points(bbox):
    """Return points rotating clockwise from the top right corner"""
    p1 = tuple(bbox[:2])
    p2 = (p1[0] + bbox[2], p1[1])
    p3 = (p2[0], p2[1] + bbox[3])
    p4 = (p1[0], p3[1])
    return p1, p2, p3, p4

def bbox_to_corners(bbox):
    """Returns the top-left and bottom-right corners of a bbox"""
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


def bbox_center(bbox):
    """Returns the (x,y) of the center of the box"""
    return (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))

def center_distance(bbox1, bbox2):
    """Returns the euclidean distance between the centers of two bbox"""
    return euclidean(bbox_center(bbox1), bbox_center(bbox2))

def euclidean(p1, p2):
    """Returns euclidean distances of two points"""
    return ((p1[0] - p2[0]) ** 2 +  (p1[1] - p2[1]) ** 2) ** 0.5
