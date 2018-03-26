from lxml import etree


def annotations_and_baselines_to_transkribus_xml(annotations, baselines, name, image_size):
    height, width = image_size
    root = etree.Element("PcGts")
    
    page = etree.SubElement(root, "Page")
    page.set("imageFilename", name)
    page.set("imageWidth", str(width))
    page.set("imageHeight", str(height))
    
    line_annotations = _split_lines(annotations)

    text_region = etree.SubElement(page, "TextRegion")

    coords = etree.SubElement(text_region, "Coords")
    coords.set("points", _get_coords(line_annotations))

    for annotation, baseline in zip(line_annotations, baselines):
        text_line = etree.SubElement(text_region, "TextLine")
        
        line_coords = etree.SubElement(text_line, "Coords")
        line_coords.set("points", _get_coords([annotation]))

        line_baseline = etree.SubElement(text_line, "Baseline")
        line_baseline.set("points", _get_baseline(baseline))

        line_text_equiv = etree.SubElement(text_line, "TextEquiv")
        line_unicode = etree.SubElement(line_text_equiv, "Unicode")
        line_unicode.text = _get_text(annotation)
        
    return str(etree.tostring(root, pretty_print=True))


def _get_baseline(baseline):
    x, y, w = baseline
    return str(x) + "," + str(y) + " " + str(x + w) + "," + str(y)


def _get_coords(annotations):
    top = _get_top(annotations[0])
    bottom = _get_bottom(annotations[-1])

    left = None
    right = None

    for a in annotations:
        l = _get_left(a)
        if left is None or l < left:
            left = l

        r = _get_right(a)
        if right is None or r > right:
            right = r    

    return str(left) + "," + str(top) + " " + str(left) + "," + str(bottom) + " " + str(right) + "," + str(bottom) + " " + str(right) + "," + str(top)


def _get_text(annotation):
    text = ""
    for a in annotation:
        text += a[0]

    return text

def _get_top(annotation):
    top = None
    for a in annotation:
        y = a[1][1]
        if top is None or y < top:
            top = y
    return top


def _get_bottom(annotation):
    bottom = None
    for a in annotation:
        y = a[1][1] + a[1][3]
        if bottom is None or y > bottom:
            bottom = y
    return bottom


def _get_left(annotation):
    first_annotation = annotation[0]
    return first_annotation[1][0]


def _get_right(annotation):
    last_annotation = annotation[-1]
    return last_annotation[1][0] + last_annotation[1][2]


def _split_lines(annotations):
    line_annotations = []
    current_line = []
    last_x = 0

    for annotation in annotations:
        character, position = annotation
        x, y, w, h = position

        if x > last_x:
            current_line.append(annotation)

        else:
            line_annotations.append(current_line)
            current_line = [annotation]
        
        last_x = x

    line_annotations.append(current_line)

    return line_annotations

