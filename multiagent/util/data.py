import tensorflow as tf

def stream_images(image_ids, id_to_filename):
    """Returns a generator of (image_id, tf.Image)"""
    for image_id in image_ids:
        yield image_id, process_image_filename(id_to_filename[image_id])

def find_all_instances(all_instances, image_id, category_id):
    """Find instances from all_instances that are in image_id and are of category_id""" 
    target_bboxs, keys = [], []
    for key, instance in all_instances.items():
        if instance['image_id'] == image_id and instance['category_id'] == category_id:
            target_bboxs.append(instance["bbox"])
            keys.append(key)
    return keys, target_bboxs

def find_image_ids_with_category(instances, category_id):
    """Return image_ids of all images that contain an instance of category_id"""
    image_ids = set()
    for _, instance in instances.items():
        if instance['category_id'] == category_id:
            image_ids.add(instance['image_id'])
    return image_ids

def process_image_filename(filename):
    """Load actual image data from a given filename"""
    img = tf.io.read_file(filename)
    return tf.image.decode_jpeg(img, channels=3)