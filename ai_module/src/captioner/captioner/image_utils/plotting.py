import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_captioned_images(images, captions, bold_indices, out_path):

    num_cols = len(images)
    num_rows = len(images[0])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))  # Adjust figsize as needed

    print(axes.shape)

    # Plot each image with its caption
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            ax = axes[i][j]
            img = images[j][i]
            caption = captions[j][i]

            # Load and display the image
            ax.imshow(img)
            ax.axis('off')  # Remove axis

            # Add the caption below the image
            if bold_indices[j] == i:
                ax.set_title(caption, fontsize=24, pad=10, fontweight='bold')  # Adjust fontsize and pad as needed
            else:
                ax.set_title(caption, fontsize=24, pad=10)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)