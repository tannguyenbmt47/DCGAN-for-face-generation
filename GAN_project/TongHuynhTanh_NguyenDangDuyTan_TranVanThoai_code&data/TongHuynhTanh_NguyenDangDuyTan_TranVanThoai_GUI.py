import tkinter as tk
from PIL import ImageTk, Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the GAN generator from .h5 file
generator = tf.keras.models.load_model('best_model.h5')

# Create the GUI window
window = tk.Tk()
window.title("GAN Image Generator")
window.geometry("500x500")
window.resizable(0, 0)

# Function to generate and display the images
def generate_images():
    noise = tf.random.normal([16, 128])
    generated_images = generator.predict(noise)

    # Clear any existing images
    for widget in image_frame.winfo_children():
        widget.destroy()

    # Display the generated images
    for i in range(generated_images.shape[0]):
        image = (generated_images[i, :, :, :] * 0.5 + 0.5)
        image = Image.fromarray((image * 255).astype('uint8'))
        image = ImageTk.PhotoImage(image)

        label = tk.Label(image_frame, image=image)
        label.image = image  # Keep a reference to prevent garbage collection
        label.grid(row=i // 4, column=i % 4)

# Create a button to generate images
generate_button = tk.Button(window, text="Generate Images", command=generate_images)
generate_button.pack(pady=10, expand=True)

# Create a frame to hold the generated images
image_frame = tk.Frame(window)
image_frame.pack()

# Start the GUI event loop

window.mainloop()
