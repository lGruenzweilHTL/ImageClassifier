from PIL import Image
import matplotlib.pyplot as plt


def binarize_image(img):
    threshold = 175  # The closer to 0, the more black will be discarded
    return img.point(lambda p: 0 if p < threshold else 255)


class PixelGroup:
    def __init__(self, le, r, t, b):
        self.bounds = [le, r, t, b]  # left, right, top, bottom

    def is_inside(self, x, y):
        return self.bounds[0] <= x <= self.bounds[1] and self.bounds[3] <= y <= self.bounds[2]

    def can_append(self, x, y):
        return self.is_inside(x, y) or self.is_adjacent(x, y)

    def is_adjacent(self, x, y):
        dy = 40  # Distance threshold y-Axis
        dx = 1  # Distance threshold x-Axis
        return ((abs(x - self.bounds[0]) <= dx or abs(x - self.bounds[1]) <= dx) and
                (abs(y - self.bounds[2]) <= dy or abs(y - self.bounds[3]) <= dy))

    def append(self, x, y):
        self.bounds[0] = min(self.bounds[0], x)  # left
        self.bounds[1] = max(self.bounds[1], x)  # right
        self.bounds[2] = max(self.bounds[2], y)  # top
        self.bounds[3] = min(self.bounds[3], y)  # bottom


def plot(notes, image, binary_img, groups):
    # Plot the binary image and each cropped note using subplots
    fig, axes = plt.subplots(1, len(notes) + 2, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[1].imshow(binary_img, cmap="gray")
    axes[1].set_title("Binary Image")
    print("Note bounds: (left, right, top, bottom)")
    for i, note in enumerate(notes):
        print(f"Note {i + 1}: {groups[i].bounds}")
        axes[i + 2].imshow(note, cmap="gray")
        axes[i + 2].set_title(f"Note {i + 1}")

    plt.show()


def segment_notes(img_path, render):
    image = Image.open(img_path)
    binary_img = binarize_image(image.convert("L"))
    # Identify pixel groups in binary image
    groups = []
    for x in range(binary_img.width):
        print(f"Processing column {x}")
        for y in range(binary_img.height):
            px = binary_img.getpixel((x, y))
            if px == 0:
                could_append = False
                for g in groups:
                    if g.can_append(x, y):
                        could_append = True
                        g.append(x, y)
                        break

                if not could_append:
                    groups.append(PixelGroup(x, x+1, y+1, y))
                    print(f"New group added. Total groups: {len(groups)}")

    print(f"Found {len(groups)} notes")
    notes = [binary_img.crop((g.bounds[0], g.bounds[3], g.bounds[1], g.bounds[2])) for g in groups]
    if render:
        plot(notes, image, binary_img, groups)
    return notes
