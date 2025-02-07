import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
import numpy as np

from n2n4m.plot import Visualiser
from n2n4m.crism_image import CRISMImage
from n2n4m.summary_parameters import IMPLEMENTED_SUMMARY_PARAMETERS

RATIO_DATA = "/home/rob_platt/pixel_classifier/data/CRISM_ML"


class ChannelViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Channel Viewer GUI")
        self.hover_paused = False

        # Create plot frame
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create image loading frame
        self.prompt_file_selection()

    def prompt_file_selection(self):
        """Open a separate window to prompt file selection on launch."""
        self.file_window = tk.Toplevel(self.root)
        self.file_window.title("Select an Image File")
        self.file_window.geometry("300x100")

        self.file_window_label = tk.Label(self.file_window, text="Please select an image file:")
        self.file_window_label.pack(pady=10)

        file_button = tk.Button(self.file_window, text="Choose File", command=self.load_image)
        file_button.pack(pady=5)

    def update_file_loading_status(self, status_message: str):
        """Update file loading and image processing status in the GUI."""
        self.file_window_label.config(text=status_message)
        self.file_window.update_idletasks()

    def setup_left_plot(self):
        left_frame = tk.Frame(self.plot_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_left = Figure(figsize=(5, 5))
        self.ax_left = self.fig_left.add_subplot(111)
        self.ax_left.set_title("No Image Loaded")

        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=left_frame)
        self.canvas_left.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True
        )

        # Add toolbar for left plot
        toolbar_left = NavigationToolbar2Tk(self.canvas_left, left_frame)
        toolbar_left.update()
        self.canvas_left.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True
        )

        self.canvas_left.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas_left.mpl_connect("button_press_event", self.toggle_hover)

    def setup_right_plot(self):
        right_frame = tk.Frame(self.plot_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig_right = Figure(figsize=(5, 5))
        self.ax_right = self.fig_right.add_subplot(111)
        self.ax_right.set_title("Channel View")

        self.canvas_right = FigureCanvasTkAgg(
            self.fig_right, master=right_frame
        )
        self.canvas_right.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True
        )

        # Add toolbar for right plot
        toolbar_right = NavigationToolbar2Tk(self.canvas_right, right_frame)
        toolbar_right.update()
        self.canvas_right.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True
        )

    def setup_controls(self, num_channels: int = 438):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        file_button = tk.Button(
            control_frame, text="Choose File", command=self.load_image
        )
        file_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.channel_dropdown = ttk.Combobox(
            control_frame, values=list(range(num_channels))
        )
        self.channel_dropdown.set(60)
        self.channel_dropdown.bind(
            "<<ComboboxSelected>>", self.update_left_plot
        )
        self.channel_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

    def update_left_plot(self, event):
        if event == "Initialization":
            selected_channel = 60
        else:
            selected_channel = int(self.channel_dropdown.get())
        self.ax_left.clear()
        self.ax_left.imshow(
            self.image_array[:, :, selected_channel], cmap="viridis"
        )
        self.ax_left.set_title(f"Channel {selected_channel}")
        self.canvas_left.draw()

    def toggle_hover(self, event):
        if event.inaxes == self.ax_left and event.button == 1:  # Left click
            self.hover_paused = not self.hover_paused
            status = "Paused" if self.hover_paused else "Active"
            self.ax_right.set_title(f"Channel View ({status})")
            self.canvas_right.draw()

    def on_hover(self, event):
        if not self.hover_paused and event.inaxes == self.ax_left:
            x, y = int(event.xdata), int(event.ydata)
            # if 0 <= x < 400 and 0 <= y < 400:
            self.ax_right.clear()
            self.ax_right.plot(self.image_array[y, x, :])
            self.ax_right.set_title("Pixel ({}, {}) Channel View".format(x, y))

            self.canvas_right.draw()

    def load_image(self):
        # Open file dialog and get file path
        filepath = filedialog.askopenfilename(
            title="Select CRISM Image",
            filetypes=[("IMG files", "*.img"), ("All files", "*.*")],
        )

        # Only proceed if a file was selected
        if not filepath:
            return

        try:
            self.update_file_loading_status("Loading image...")
            # Create CRISM image object
            image = CRISMImage(filepath)

            self.update_file_loading_status("Processing image...")
            # Process image ratios
            image.ratio_image(RATIO_DATA)

            self.update_file_loading_status("Calculating summary parameters...")
            # Calculate all required summary parameters
            summary_parameters = [
                *IMPLEMENTED_SUMMARY_PARAMETERS.keys()
            ]

            for parameter in summary_parameters:
                image.calculate_summary_parameter(parameter)

            # Create visualizer and get image data
            visualizer = Visualiser(image)
            visualizer.get_image(60)

            # Store the image data
            self.image_array = visualizer.raw_image_copy

            # Initialize left plot
            self.setup_left_plot()

            # Initialize right plot
            self.setup_right_plot()

            # Initialize controls
            num_channels = self.image_array.shape[2]
            self.setup_controls(num_channels)

            # Update the display
            self.update_left_plot("Initialization")

            # Reset right plot
            self.ax_right.clear()
            self.ax_right.set_title("Channel View")
            self.canvas_right.draw()

            self.file_window.destroy()
        except Exception as e:
            self.ax_left.clear()
            self.ax_left.set_title(f"Error loading image: {str(e)}")
            self.canvas_left.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChannelViewer(root)
    root.mainloop()
