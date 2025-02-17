import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)
from matplotlib.figure import Figure
import numpy as np
import onnxruntime

from n2n4m.plot import Visualiser
from n2n4m.crism_image import CRISMImage
from n2n4m.summary_parameters import IMPLEMENTED_SUMMARY_PARAMETERS
from n2n4m.n2n4m_denoise import clip_bands
from n2n4m.preprocessing import impute_bad_values_in_image

from classification_plot import convert_to_coords_filter_regions_by_conf

RATIO_DATA = "/home/rob_platt/pixel_classifier/data/CRISM_ML"


class ChannelViewer:
    def __init__(self, root, filepath: str = None):
        """Initialize the Channel Viewer GUI.
        If image filepath passed, image loading prompt is skipped.
        """
        self.root = root
        self.root.title("Channel Viewer GUI")
        self.hover_paused = False
        self.show_classification = True

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(4, weight=0)

        # Create plot frame
        self.plot_frame = tk.Frame(root)
        self.plot_frame.grid(row=0, column=0, sticky="nesw")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        if filepath:
            self.filepath = filepath
            self.load_image(filepath)
        else:
            self.prompt_file_selection()

    def prompt_file_selection(self):
        """Open a separate window to prompt file selection on launch."""
        self.file_window = tk.Toplevel(self.root)
        self.file_window.title("Select an Image File")
        self.file_window.geometry("300x100")

        self.file_window_label = tk.Label(
            self.file_window, text="Please select an image file."
        )

        file_button = tk.Button(
            self.file_window, text="Choose File", command=self.load_image
        )

        self.file_window_label.place(relx=0.5, rely=0.3, anchor="center")
        file_button.place(relx=0.5, rely=0.5, anchor="center")

    def update_file_loading_status(self, status_message: str):
        """Update file loading and image processing status in the GUI."""
        self.file_window_label.config(text=status_message)
        self.file_window.update_idletasks()

    def setup_left_plot(self):
        left_frame = tk.Frame(self.plot_frame)
        left_frame.grid(column=0, row=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        self.fig_left = Figure(figsize=(8, 8))
        self.ax_left = self.fig_left.add_subplot(111)

        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=left_frame)
        self.canvas_left.get_tk_widget().grid(row=0, column=0, sticky="nesw")
        self.canvas_left.get_tk_widget().columnconfigure(0, weight=1)
        self.canvas_left.get_tk_widget().rowconfigure(0, weight=1)

        self.canvas_left.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas_left.mpl_connect("button_press_event", self.toggle_hover)

    def setup_right_plot(self):
        right_frame = tk.Frame(self.plot_frame)
        right_frame.grid(column=1, row=0, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        self.fig_right = Figure(figsize=(8, 4))
        self.ax_right = self.fig_right.add_subplot(111)
        self.ax_right.set_title("Channel View")

        self.canvas_right = FigureCanvasTkAgg(
            self.fig_right, master=right_frame
        )
        self.canvas_right.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas_right.get_tk_widget().columnconfigure(0, weight=1)
        self.canvas_right.get_tk_widget().rowconfigure(0, weight=1)

    def setup_controls(
        self, summary_params: list = [], num_channels: int = 438
    ):
        """
        Set up control bar for the GUI. Instantiates the following widgets:
        - File selection button to load a new image
        - Dropdown menu for what image to display
        - Optional dropdown menu for channel selection
        - Classification button
        """

        def on_popdown_show(combo: ttk.Combobox):
            """Make combobox dropdown drop up instead of down."""
            try:
                popup = combo.master.tk.eval(
                    f"ttk::combobox::PopdownWindow {combo._w}"
                )
                # get the height of the listbox inside popup window
                popup_height = combo.master.tk.call(
                    "winfo", "reqheight", f"{popup}.f.l"
                )
                # get the height of combobox
                combo_height = combo.winfo_height()
                # set the position of the popup window
                ttk.Style().configure(
                    "Custom.TCombobox",
                    postoffset=(0, -combo_height - popup_height, 0, 0),
                )
            except tk.TclError:
                pass

        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=4, column=0, sticky="nsew")

        # File selection button
        file_button = tk.Button(
            self.control_frame, text="Choose File", command=self.load_image
        )
        file_button.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Dropdown menu for image selection
        summary_params.append("Ratioed Image Channel")
        self.image_selection_dropdown = ttk.Combobox(
            self.control_frame,
            values=summary_params,
            style="Custom.TCombobox",
            width=25,
        )
        self.image_selection_dropdown.bind(
            "<FocusIn>",
            lambda e: on_popdown_show(self.image_selection_dropdown),
        )
        self.image_selection_dropdown.bind(
            "<<ComboboxSelected>>", self.update_left_plot
        )
        self.image_selection_dropdown.set("Ratioed Image Channel")
        self.image_selection_dropdown.grid(
            row=1, column=1, padx=5, sticky="nsew"
        )
        image_selection_label = tk.Label(
            self.control_frame, text="Image Selection:"
        )
        image_selection_label.grid(row=0, column=1, padx=5, sticky="nsew")

        # Dropdown menu for channel selection
        self.channel_dropdown = ttk.Combobox(
            self.control_frame,
            values=list(range(num_channels)),
            style="Custom.TCombobox",
        )
        self.channel_dropdown.bind(
            "<FocusIn>", lambda e: on_popdown_show(self.channel_dropdown)
        )
        self.channel_dropdown.set(60)
        self.channel_dropdown.bind(
            "<<ComboboxSelected>>", self.update_left_plot
        )
        self.channel_dropdown.grid(row=1, column=2, padx=5, sticky="nsew")
        channel_label = tk.Label(
            self.control_frame, text="Image Channel Selection:"
        )
        channel_label.grid(row=0, column=2, padx=5, sticky="nsew")

        # Button for classification
        self.classification_button = tk.Button(
            self.control_frame,
            text="Classify",
            bg="#008000",
            activebackground="#5ce65c",
            command=self.classify,
        )
        self.classification_button.grid(
            row=0, column=3, rowspan=2, padx=5, sticky="nsew"
        )

    def update_left_plot(self, event):
        image_selection = self.image_selection_dropdown.get()
        if image_selection == "Ratioed Image Channel":
            if event == "Initialization":
                selected_channel = 60
            else:
                self.channel_dropdown.state(["!disabled"])
                selected_channel = int(self.channel_dropdown.get())
            self.ax_left.clear()
            self.ax_left.imshow(
                self.image_array[:, :, selected_channel], cmap="viridis"
            )
            self.ax_left.set_title(f"Channel {selected_channel}")
        elif image_selection in IMPLEMENTED_SUMMARY_PARAMETERS:
            self.ax_left.clear()
            self.ax_left.imshow(
                self.visualizer.get_summary_parameter(image_selection),
                cmap="viridis",
            )
            self.ax_left.set_title(f"{image_selection} Summary Parameter")
            self.channel_dropdown.state(["disabled"])
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
            self.ax_right.clear()
            self.ax_right.plot(self.image_array[y, x, :])
            self.ax_right.set_title("Pixel ({}, {}) Channel View".format(x, y))

            self.canvas_right.draw()

    def add_classification_controls(self):
        """Add controls for classification results to control panel."""
        self.toggle_classification_button = tk.Button(
            self.control_frame,
            text="Classification Results (Off)",
            command=self.toggle_classification,
        )
        self.toggle_classification_button.grid(
            row=0, column=4, rowspan=2, padx=5, sticky="nsew"
        )

    def toggle_classification(self):
        """Plot classification results on top of the left plot."""
        self.update_left_plot("Classification_Toggle")
        if self.show_classification:
            self.toggle_classification_button.config(
                text="Classification Results (Off)"
            )
            for mineral, coords in self.pred_coords.items():
                self.ax_left.scatter(
                    coords[0], coords[1], s=0.1, label=mineral
                )
        else:
            self.toggle_classification_button.config(
                text="Classification Results (On)"
            )
        self.canvas_left.draw()
        self.show_classification = not self.show_classification

    def preprocess_image(self, image) -> np.ndarray:
        """Preprocess the image for the model. Applies the following steps:
        - Clip image channels to 248 bands
        - Impute bad values (Ratioed I/F > 10) in the image
        - Scale the image between 0 and 1
        """
        image = image.reshape(-1, 438)  # 438 bands
        image, _ = clip_bands(image)
        image = image[:, :248]  # 248 bands to use for the model
        image, _ = impute_bad_values_in_image(image, threshold=10)
        min_vals = np.min(image, axis=-1, keepdims=True)
        max_vals = np.max(image, axis=-1, keepdims=True)

        image_scaled = (image - min_vals) / ((max_vals - min_vals) + 0.00001)
        return image_scaled

    def batch_array(self, array):
        # Ensure the input array has the correct number of channels (248)
        if array.shape[1] != 248:
            raise ValueError("Input array must have 248 channels.")

        # Calculate the number of full blocks and the remainder
        num_full_blocks = array.shape[0] // 1024
        remainder = array.shape[0] % 1024

        # Split into full blocks
        blocks = [
            array[i * 1024: (i + 1) * 1024] for i in range(num_full_blocks)
        ]

        # Handle the remainder by padding with zeros if necessary
        if remainder > 0:
            padded_block = np.zeros((1024, 248))
            padded_block[:remainder] = array[num_full_blocks * 1024:]
            blocks.append(padded_block)

        return np.array(blocks, dtype="float32"), remainder

    def classify(self):
        """Classify the CRISM cube using the CRISM Classifier model."""
        ort_session = onnxruntime.InferenceSession(
             "/home/rob_platt/CRISM_classifier_application/Notebooks/vae_classifier_1024.onnx", providers=["CPUExecutionProvider"]
        )

        image = np.empty_like(self.image_array)
        image[:] = self.image_array
        image = self.preprocess_image(image)
        batches, remainder = self.batch_array(image)

        pred_probs = []
        for x in batches:
            onnx_input = x[:, np.newaxis, :]
            onnxruntime_input = {"input.1": onnx_input}
            pred_probs.append(ort_session.run(None, onnxruntime_input)[1])

        pred_probs = np.array(pred_probs).reshape(-1, 38)
        # Remove the padding from the last batch if necessary
        if remainder > 0:
            pred_probs = pred_probs[: -(1024 - remainder)]
        pred_probs = pred_probs.reshape(*self.image_array.shape[:2], 38)
        self.pred_cls = np.argmax(pred_probs, axis=-1)
        self.pred_conf = np.max(pred_probs, axis=-1)

        self.pred_coords = convert_to_coords_filter_regions_by_conf(
            self.pred_cls, self.pred_conf
        )
        self.pred_coords = {
            k: v
            for k, v in sorted(
                self.pred_coords.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        }

        self.add_classification_controls()
        self.toggle_classification()
        self.classification_button.config(state="disabled")

    def load_image(self, filepath=None):
        # Open file dialog and get file path
        if not filepath:
            filepath = filedialog.askopenfilename(
                title="Select CRISM Image",
                filetypes=[("IMG files", "*.img"), ("All files", "*.*")],
            )
        try:
            # self.update_file_loading_status("Loading image...")
            # Create CRISM image object
            image = CRISMImage(filepath)

            # self.update_file_loading_status("Processing image...")
            # Process image ratios
            image.ratio_image(RATIO_DATA)

            # self.update_file_loading_status(
            #     "Calculating summary parameters..."
            # )
            # # Calculate all required summary parameters
            summary_parameters = [*IMPLEMENTED_SUMMARY_PARAMETERS.keys()]

            for parameter in summary_parameters:
                image.calculate_summary_parameter(parameter)

            # Create visualizer and get image data
            self.visualizer = Visualiser(image)
            self.visualizer.get_image(60)

            # Store the image data
            self.image_array = self.visualizer.raw_image_copy

            # Initialize left plot
            self.setup_left_plot()

            # Initialize right plot
            self.setup_right_plot()

            # Initialize controls
            num_channels = self.image_array.shape[2]
            self.setup_controls(summary_parameters, num_channels)

            # Update the display
            self.update_left_plot("Initialization")

            # Reset right plot
            self.ax_right.clear()
            self.ax_right.set_title("Channel View")
            self.canvas_right.draw()

            # self.file_window.destroy()

        except Exception as e:
            self.ax_left.clear()
            self.ax_left.set_title(f"Error loading image: {str(e)}")
            self.canvas_left.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChannelViewer(
        root,
        "/home/rob_platt/CRISM_classifier_application/data/FRT00009A16_07_IF166L_TRR3.img",
    )
    root.mainloop()
