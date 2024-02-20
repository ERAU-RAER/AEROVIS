import cv2 as cv
import numpy as np 
import os
import random
import string

BACKGROUND_SOURCE_IMG_PATH = os.path.abspath("../assets/ground.png");
BACKGROUND_IMAGE = cv.imread(BACKGROUND_SOURCE_IMG_PATH)

BACKGROUND_ORIGINAL_HEIGHT, BACKGROUND_ORIGINAL_WIDTH = BACKGROUND_IMAGE.shape[:2]
RESULT_IMG_WIDTH = 1920
RESULT_IMG_HEIGHT = 1080 
SHAPE_SIZE = 50
SHAPE_IMG_SIZE = 150

OUTPUT_FOLDER = os.path.abspath("../assets/results")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def get_random_background_segment():
    start_x = random.randint(0, BACKGROUND_ORIGINAL_WIDTH-RESULT_IMG_WIDTH) 
    start_y = random.randint(0, BACKGROUND_ORIGINAL_HEIGHT-RESULT_IMG_HEIGHT) 
    end_x = start_x + RESULT_IMG_WIDTH 
    end_y = start_y + RESULT_IMG_HEIGHT 

    segment = BACKGROUND_IMAGE[start_y:end_y, start_x:end_x]
    center = ((end_x - start_x) / 2, (end_y - start_y) / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, random.randint(0, 360), 1.0)
    rotated_segment = cv.warpAffine(segment, rotation_matrix, (RESULT_IMG_WIDTH, RESULT_IMG_HEIGHT), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    return rotated_segment


def calculate_absolute_center(image, relative_center):
    height, width = image.shape[:2]
    absolute_x = int(width * relative_center[0])
    absolute_y = int(height * relative_center[1])
    return (absolute_x, absolute_y)


def create_transparent_image():
    return np.zeros((SHAPE_IMG_SIZE, SHAPE_IMG_SIZE, 4), dtype=np.uint8)


def draw_circle(image, center, color, character, character_color):
    cv.circle(image, center, SHAPE_SIZE, color, thickness=-1, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)), character, character_color)
    return image


def draw_semicircle(image, center, color, character, character_color):
    cv.ellipse(image, center, (SHAPE_SIZE+10, SHAPE_SIZE+10), 0, 0, 180, color, thickness=-1, lineType=cv.LINE_AA)
    draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+35), character, character_color)
    return image


def draw_quarter_circle(image, center, color, character, character_color):
    cv.ellipse(image, (center[0]-(SHAPE_SIZE//2)-10, center[1]-(SHAPE_SIZE//2)-10), (SHAPE_SIZE*2, SHAPE_SIZE*2), 0, 0, 90, color, thickness=-1, lineType=cv.LINE_AA)
    draw_character(image, (SHAPE_IMG_SIZE//2+5, (SHAPE_IMG_SIZE//2)+15), character, character_color)
    return image


def draw_triangle(image, center, color, character, character_color):
    vertices = np.array([
        [center[0] - SHAPE_SIZE, center[1] + SHAPE_SIZE],
        [center[0], center[1] - SHAPE_SIZE],
        [center[0] + SHAPE_SIZE, center[1] + SHAPE_SIZE]
    ], np.int32)
    cv.fillPoly(image, [vertices], color, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+15), character, character_color)
    return image


def draw_rectangle(image, center, color, character, character_color):
    top_left = (center[0] - SHAPE_SIZE, center[1] - int(SHAPE_SIZE / 2))
    bottom_right = (center[0] + SHAPE_SIZE, center[1] + int(SHAPE_SIZE / 2) + 10)
    cv.rectangle(image, top_left, bottom_right, color, -1, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+10), character, character_color)
    return image


def draw_pentagon(image, center, color, character, character_color):
    pts = np.array([[np.cos(theta) * SHAPE_SIZE + center[0], np.sin(theta) * SHAPE_SIZE + center[1]] 
                    for theta in np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 5])
    cv.fillPoly(image, [np.array(pts, np.int32)], color, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+5), character, character_color)
    return image


def draw_star(image, center, color, character, character_color):
    num_vertices = 5

    outer_radius = SHAPE_SIZE + 15 
    inner_radius = outer_radius * np.cos(2 * np.pi / num_vertices) / np.cos(np.pi / num_vertices)
    outer_angle = 2 * np.pi / num_vertices
    inner_angle = np.pi / num_vertices

    points = []

    for i in range(num_vertices):
        outer_x = int(center[0] + outer_radius * np.cos(outer_angle * i))
        outer_y = int(center[1] - outer_radius * np.sin(outer_angle * i))
        inner_x = int(center[0] + inner_radius * np.cos(outer_angle * i + inner_angle))
        inner_y = int(center[1] - inner_radius * np.sin(outer_angle * i + inner_angle))
        points.append((outer_x, outer_y))
        points.append((inner_x, inner_y))

    star_points = np.array(points, np.int32).reshape((-1, 1, 2))
    cv.fillPoly(image, [star_points], color, lineType=cv.LINE_AA)

    image = draw_character(image, (SHAPE_IMG_SIZE//2-2, (SHAPE_IMG_SIZE//2)+5), character, character_color)

    return image


def draw_cross(image, center, color, character, character_color):
    thickness = (SHAPE_SIZE // 2) + 15 

    start_point_h = (center[0] - SHAPE_SIZE, center[1] - thickness // 2)
    end_point_h = (center[0] + SHAPE_SIZE, center[1] + thickness // 2)
    cv.rectangle(image, start_point_h, end_point_h, color, -1)

    start_point_v = (center[0] - thickness // 2, center[1] - SHAPE_SIZE)
    end_point_v = (center[0] + thickness // 2, center[1] + SHAPE_SIZE)
    cv.rectangle(image, start_point_v, end_point_v, color, -1)

    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)), character, character_color)

    return image


def rotate_shape_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    new_image = np.zeros((height, width, 4), dtype=np.uint8)
    new_image[..., :3] = rotated_image[..., :3]
    new_image[..., 3] = cv.warpAffine(image[..., 3], rotation_matrix, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    
    return new_image


def draw_character(image, pos, character, character_color):
    font = cv.FONT_HERSHEY_DUPLEX
    font_scale = 2 
    thickness = 6 
    cv.putText(image, character, (pos[0]-20, pos[1]+15), font, font_scale, character_color, thickness, cv.LINE_AA)
    return image


def overlay_shape_on_background(background, shape_img, relative_center):
    bg_height, bg_width = background.shape[:2]
    shape_height, shape_width = shape_img.shape[:2]

    center_x = int(relative_center[0] * bg_width)
    center_y = int(relative_center[1] * bg_height)
    top_left_x = center_x - shape_width // 2
    top_left_y = center_y - shape_height // 2
    y1, y2 = top_left_y, top_left_y + shape_height
    x1, x2 = top_left_x, top_left_x + shape_width
    alpha_s = shape_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # blending 
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha_s * shape_img[:, :, c] +
                                       alpha_l * background[y1:y2, x1:x2, c])

    return background


def get_safe_center(bg_width, bg_height, shape_width, shape_height):
    x_margin = shape_width / 2
    y_margin = shape_height / 2

    center_x = random.uniform(x_margin, bg_width - x_margin)
    center_y = random.uniform(y_margin, bg_height - y_margin)

    relative_center_x = center_x / bg_width
    relative_center_y = center_y / bg_height

    return (relative_center_x, relative_center_y)


COLORS = {
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
    "red": (0, 0, 255, 255),
    "blue": (255, 0, 0, 255),
    "green": (0, 255, 0, 255),
    "purple": (128, 0, 128, 255),
    "brown": (19, 69, 139, 255),
    "orange": (0, 165, 255, 255)
}

DRAW_SHAPE_FUNCTIONS = {
    0: draw_circle,
    1: draw_semicircle,
    2: draw_quarter_circle,
    3: draw_triangle,
    4: draw_rectangle,
    5: draw_pentagon,
    6: draw_star,
    7: draw_cross
}

for i in range(100):
    path = os.path.join(OUTPUT_FOLDER, f"{i}.jpg")
    image = get_random_background_segment()

    center = get_safe_center(RESULT_IMG_WIDTH, RESULT_IMG_HEIGHT, SHAPE_IMG_SIZE, SHAPE_IMG_SIZE) 
    shape = random.randint(0, 7)
    shape_color = random.choice(list(COLORS.values())) 
    characters = string.ascii_uppercase + string.digits
    character = random.choice(characters)
    character_color = random.choice(list(COLORS.values()))

    draw_function = DRAW_SHAPE_FUNCTIONS.get(shape, None)
    shape_image = draw_function(create_transparent_image(), (SHAPE_IMG_SIZE//2, SHAPE_IMG_SIZE//2), shape_color, character, character_color) 
    shape_image = rotate_shape_image(shape_image, random.randint(0, 360))
    result = overlay_shape_on_background(image, shape_image, center) 
    cv.imwrite(path, result)
    