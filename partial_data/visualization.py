"""Bounding box visualization utils.

Note: Adapted from https://github.com/tensorflow/models/blob/master/research/object_detection/
    utils/visualization_utils.py
"""
import numpy as np

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_bounding_box_on_image(img,
                               hmin, wmin, hmax, wmax,
                               color='red',
                               thickness=4,
                               font_size=24,
                               font_filepath=None,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Parameters
    ----------
    img: PIL.Image
        The image to be modified.
    hmin: float
        Bounding box min along the image height axis.
    wmin: float
        Bounding box min along the image width axis.
    hmax: float
        Bounding box max along the image height axis.
    wmax: float
        Bounding box max along the image width axis.
    color: str
        Bounding box line color.
    thickness: int
        Bounding box line thickness.
    font_size: int
        Display font size for labels.
    font_filepath: str
        Filepath where file for desired font is located.
    display_str_list: tuple
        List of strings to display in box (each to be shown on its own line).
    use_normalized_coordinates: bool
        If True, treat coordinates as normalized to image dimensions
        (should be between 0-1).  Otherwise treat coordinates as absolute pixel units.
    """
    draw = ImageDraw.Draw(img)
    im_width, im_height = img.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (wmin * im_width, wmax * im_width, hmin * im_height, hmax * im_height)
    else:
        (left, right, top, bottom) = (wmin, wmax, hmin, hmax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                  width=thickness,
                  fill=color)
    try:
        font_filepath = font_filepath or 'arial.ttf'
        font = ImageFont.truetype(font_filepath, font_size)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image(img,
                                 boxes,
                                 color=None,
                                 thickness=4,
                                 font_size=24,
                                 font_filepath=None,
                                 display_str_list_list=()):
    """Draws bounding boxes on image.

    Parameters
    ----------
    img: PIL.Image
        The image to be modified.
    boxes: np.array
        An Nx4 numpy array where each row is a set of bounding box coordinates
        specified as (hmin, wmin, hmax, wmax). The coordinates should be normalized
        between [0, 1].
    color: str
        Bounding box line color.
    thickness: int
        Bounding box line thickness.
    font_size: int
        Display font size for labels.
    font_filepath: str
        Filepath where file for desired font is located.
    display_str_list_list: list(tuple)
        A list of tuples for each bounding box. The reason to pass a list of
        strings for a bounding box is that it might contain multiple labels.

    Raises
    ------
    ValueError: if boxes is not a [N, 4] array
    """
    # Get box colors
    if color is None:
        display_str_list_list_unq = set(display_str_list_list)
        color_map_inds = np.random.permutation(len(STANDARD_COLORS))[:len(display_str_list_list_unq)]
        color_map = {display_str_list: STANDARD_COLORS[ind] for display_str_list, ind in
                     zip(display_str_list_list_unq, color_map_inds)}
        colors = [color_map[display_str_list] for display_str_list in display_str_list_list]
    elif type(color) == dict:
        colors = [color[display_str_list] for display_str_list in display_str_list_list]
    else:
        colors = [color] * len(display_str_list_list)

    # Validate bounding boxes
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')

    # Add boxes to image
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(img, boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
                                   color=colors[i], thickness=thickness, font_size=font_size,
                                   font_filepath=font_filepath, display_str_list=display_str_list)
