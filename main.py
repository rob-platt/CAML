import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
import numpy as np
import onnxruntime
import threading
import os

from n2n4m.plot import Visualiser
from n2n4m.crism_image import CRISMImage
from n2n4m.summary_parameters import IMPLEMENTED_SUMMARY_PARAMETERS
from n2n4m.n2n4m_denoise import clip_bands
from n2n4m.preprocessing import impute_bad_values_in_image
from n2n4m.wavelengths import ALL_WAVELENGTHS

from classification_plot import (
    convert_to_coords_filter_regions_by_conf,
    CLASS_NAMES,
    mineral_colours,
)
from CustomSlider import Slider

class CAMEL:
    def __init__(self, root, filepath: str = None):
        """Initialize the CAMEL (CRISM Analysis using MachinE Learning) GUI.
        If image filepath passed, image loading prompt is skipped.
        """
        self.root = root
        self.root.title("CAMEL")
        self.hover_paused: bool = False
        self.show_classification: bool = True
        self.classification_flag: bool = False
        self.x_pos: int = 0
        self.y_pos: int = 0
        self.filepath: str = None

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(4, weight=0)

        # Create plot frame
        self.plot_frame = tk.Frame(root, bg="white")
        self.plot_frame.grid(row=0, column=0, sticky="nesw")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.rowconfigure(1, weight=3)

        if filepath:
            self.filepath = filepath
            self.load_image_subroutine(filepath)
        else:
            self.prompt_crism_ml_file_selection()

    def prompt_crism_ml_file_selection(self):
        """Open a separate window to prompt CRISM_ML dataset
        directory selection on launch."""
        self.crism_ml_dataset_window = tk.Toplevel(self.root)
        self.crism_ml_dataset_window.title("Select CRISM_ML Dataset Directory")
        self.crism_ml_dataset_window.geometry("300x100")

        self.crism_ml_dataset_label = tk.Label(
            self.crism_ml_dataset_window,
            text="Please select the CRISM_ML dataset directory.",
        )
        self.crism_ml_dataset_label.place(relx=0.5, rely=0.3, anchor="center")

        crism_ml_dataset_button = tk.Button(
            self.crism_ml_dataset_window,
            text="Choose Dataset Directory",
            command=self.crism_ml_dataset_selection,
        )
        crism_ml_dataset_button.place(relx=0.5, rely=0.5, anchor="center")

    def prompt_image_file_selection(self):
        """Open a separate window to prompt image file selection
        after CRISM_ML dataset selection."""
        self.file_window = tk.Toplevel(self.root)
        self.file_window.title("Select an Image File")
        self.file_window.geometry("300x100")

        self.file_window_label = tk.Label(
            self.file_window, text="Please select an image file."
        )

        file_button = tk.Button(
            self.file_window,
            text="Choose File",
            command=self.load_image_subroutine,
        )

        self.file_window_label.place(relx=0.5, rely=0.3, anchor="center")
        file_button.place(relx=0.5, rely=0.5, anchor="center")

    def crism_ml_dataset_selection(self):
        """Open file dialog to select the CRISM_ML dataset."""
        self.crism_ml_dataset = filedialog.askdirectory(
            title="Select CRISM_ML Dataset",
        )
        if self.crism_ml_dataset:
            self.crism_ml_dataset_window.destroy()
            self.prompt_image_file_selection()

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

        # Add Zoom & Pan Toolbar
        toolbar_frame = tk.Frame(left_frame)
        toolbar_frame.grid(
            row=1, column=0, sticky="ew"
        )  # Place toolbar below plot
        toolbar = NavigationToolbar2Tk(self.canvas_left, toolbar_frame)
        toolbar.update()

        self.canvas_left.mpl_connect(
            "motion_notify_event", self.update_right_plot
        )
        self.canvas_left.mpl_connect("button_press_event", self.toggle_hover)

    def setup_right_plot(self):
        right_frame = tk.Frame(self.plot_frame, bg="white")
        right_frame.grid(column=1, row=0, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        self.fig_right = Figure(figsize=(8, 6))
        self.ax_right = self.fig_right.add_subplot(111)
        self.ax_right.set_title("Spectrum Plot")

        self.canvas_right = FigureCanvasTkAgg(
            self.fig_right, master=right_frame
        )
        self.canvas_right.get_tk_widget().grid(row=0, column=0, sticky="ew")
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
        self.control_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        # # File selection button
        # file_button = tk.Button(
        #     self.control_frame, text="Choose File", command=self.load_image
        # )
        # file_button.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

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
        image_selection_label.grid(row=0, column=1, padx=5)  # sticky="nsew")

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
        self.channel_dropdown.grid(row=1, column=2, padx=5)  # sticky="nsew")
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
            command=self.classification_subroutine,
        )
        self.classification_button.grid(
            row=0, column=3, rowspan=2, padx=5, sticky="nsew"
        )

        spectrum_slider_label = tk.Label(
            self.control_frame, text="Spectrum Wavelength Range:"
        )
        spectrum_slider_label.grid(row=0, column=14, padx=5, sticky="nsew")

        # Slider to control x-axis range of spectrum plot
        self.spectrum_range_slider = Slider(
            self.control_frame,
            width=200,
            height=20,
            min_val=0,
            max_val=len(ALL_WAVELENGTHS),
            step_size=1,
            init_lis=[0, len(ALL_WAVELENGTHS)],
            show_value=True,
        )
        self.spectrum_range_slider.setValueChangeCallback(
            self.update_right_plot
        )
        self.spectrum_range_slider.grid(
            row=1, column=14, columnspan=2, padx=5, sticky="e"
        )
        self.control_frame.columnconfigure(13, weight=1)

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
            self.x_pos, self.y_pos = int(event.xdata), int(event.ydata)
            self.canvas_right.draw()

    def update_right_plot(self, event):
        if not self.hover_paused:
            if event.inaxes == self.ax_left:
                x_pos, y_pos = int(event.xdata), int(event.ydata)
                if (
                    (x_pos > 0)
                    and (x_pos < self.image_array.shape[1] - 1)
                    and (y_pos > 0)
                    and (y_pos < self.image_array.shape[0] - 1)
                ):
                    self.x_pos, self.y_pos = x_pos, y_pos

        line_col = "black"

        if self.classification_flag:
            conf = self.pred_conf[self.y_pos, self.x_pos]
            cls = self.pred_cls[self.y_pos, self.x_pos]
            cls_name = CLASS_NAMES[cls]
            # check if cls in self.pred_coords
            if len(self.pred_coords[cls]) > 1:
                # Check if the current pixel is in the coordinates for the
                # class and therefore survived the filtering
                if (
                    self.x_pos in self.pred_coords[cls][0]
                    and self.y_pos in self.pred_coords[cls][1]
                ):
                    try:
                        line_col = mineral_colours[cls_name]
                    except KeyError:
                        line_col = "black"

        if self.hover_paused or event.inaxes == self.ax_left:
            self.ax_right.clear()
            self.min_wavelength_idx = int(
                self.spectrum_range_slider.getValues()[0]
            )
            self.max_wavelength_idx = int(
                self.spectrum_range_slider.getValues()[1]
            )
            self.ax_right.plot(
                ALL_WAVELENGTHS[
                    self.min_wavelength_idx : self.max_wavelength_idx
                ],
                self.ratioed_array[
                    self.y_pos,
                    self.x_pos,
                    self.min_wavelength_idx : self.max_wavelength_idx,
                ],
                color=line_col,
            )
            self.ax_right.set_xlabel("Wavelength (μm)")
            self.ax_right.set_ylabel("Ratioed I/F")
            if self.classification_flag:
                self.ax_right.set_title(
                    f"Pixel ({self.x_pos}, {self.y_pos}) Spectrum Plot\nClass: {cls_name}, Confidence: {conf*100:.2f}%"
                )
            else:
                self.ax_right.set_title(
                    f"Pixel ({self.x_pos}, {self.y_pos}) Spectrum Plot"
                )
            self.canvas_right.draw()

    def display_loading_window(self, message: str):
        """Create a loading window with a message."""
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("Loading...")
        self.loading_window.geometry("400x100")

        loading_label = tk.Label(self.loading_window, text=message)
        loading_label.place(relx=0.5, rely=0.3, anchor="center")
        self.progress_bar = ttk.Progressbar(
            self.loading_window,
            orient="horizontal",
            length=300,
            mode="indeterminate",
        )
        self.progress_bar.place(relx=0.5, rely=0.5, anchor="center")
        self.progress_bar.start()

        self.loading_window.update_idletasks()

    def add_classification_controls(self):
        """Add controls for classification results to control panel."""
        # Add button for toggling display of classification results
        self.toggle_classification_button = tk.Button(
            self.control_frame,
            text="Classification Results (Off)",
            command=self.toggle_classification,
            wraplength=100,
        )
        self.toggle_classification_button.grid(
            row=0, column=4, rowspan=2, padx=5, sticky="nsew"
        )

        filtering_label = tk.Label(
            self.control_frame, text="Filtering Options:"
        )
        filtering_label.grid(row=0, column=5, padx=5, sticky="nesw")

        # Add slider for confidence threshold
        self.confidence_slider = tk.Scale(
            self.control_frame,
            from_=0,
            to=1,
            resolution=0.05,
            orient="horizontal",
            label="Confidence Threshold",
            length=200,
            showvalue=True,
        )
        self.confidence_slider.set(0.0)
        self.confidence_slider.grid(
            row=1, column=5, columnspan=3, padx=5, sticky="nsw"
        )

        # Add slider for connected components threshold
        self.connect_comp_slider = tk.Scale(
            self.control_frame,
            from_=0,
            to=100,
            resolution=5,
            orient="horizontal",
            label="Connected Components",
            length=200,
            showvalue=True,
        )
        self.connect_comp_slider.set(0)
        self.connect_comp_slider.grid(
            row=1, column=8, columnspan=3, padx=5, sticky="nsw"
        )

        # Add button to run filtering
        self.run_filtering_button = tk.Button(
            self.control_frame,
            text="Run Filtering",
            command=self.classification_filter,
        )
        self.run_filtering_button.grid(
            row=0, column=11, rowspan=2, padx=5, sticky="nsw"
        )

        # Add button to save the image
        self.save_button = tk.Button(
            self.control_frame, text="Save Image", command=self.save_file
        )
        self.save_button.grid(
            row=0, column=12, rowspan=2, padx=5, sticky="nsw"
        )

    def classification_filter(self):
        """Filter classification results based on confidence threshold,
        and connected components."""
        conf_threshold = self.confidence_slider.get()
        min_components = self.connect_comp_slider.get()

        self.pred_coords = convert_to_coords_filter_regions_by_conf(
            self.pred_cls,
            self.pred_conf,
            min_confidence=conf_threshold,
            min_area=min_components,
        )
        self.pred_coords = {
            k: v
            for k, v in sorted(
                self.pred_coords.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        }
        self.plot_classification()

    def plot_classification(self):
        # clear left plot
        self.ax_left.clear()
        self.update_left_plot("Plot Classification")
        for mineral, coords in self.pred_coords.items():
            if mineral == 1:
                continue
            if len(coords) > 1:
                # If a custom colour is defined for the mineral, use it
                try:
                    self.scatter = self.ax_left.scatter(
                        coords[0],
                        coords[1],
                        s=0.75,
                        marker="s",
                        label=CLASS_NAMES[mineral],
                        color=mineral_colours[CLASS_NAMES[mineral]],
                    )
                # Otherwise, use the default colours
                except KeyError:
                    self.scatter = self.ax_left.scatter(
                        coords[0],
                        coords[1],
                        s=0.75,
                        label=CLASS_NAMES[mineral],
                        marker="s",
                    )
        # self.ax_left.callbacks.connect("xlim_changed", self.update_marker_size)
        # self.ax_left.callbacks.connect("ylim_changed", self.update_marker_size)
        self.canvas_left.draw()
        self.plot_classification_legend()

    # def update_marker_size(self, event=None):
    #     if event == "xlim_changed" or event == "ylim_changed":
    #         xlim = self.ax_left.get_xlim()
    #         # Compute the new marker size based on pixel scaling
    #         pixel_width = (xlim[1] - xlim[0]) / 10  # Adjust based on the image resolution
    #         new_size = (pixel_width ** 2)  # Scale factor to match original size
    #         print(new_size)
    #         self.scatter.set_sizes([new_size] * len(self.scatter.get_offsets()))  # Update sizes
    #         self.canvas_left.draw_idle()

    def plot_classification_legend(self):
        """Plot legend for class predictions across base of plot frame."""
        legend_frame = tk.Frame(self.plot_frame)
        legend_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        legend_frame.columnconfigure(0, weight=1)
        legend_frame.rowconfigure(0, weight=1)

        legend_fig = Figure(figsize=(16, 0.5))
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")

        legend_ax.legend(
            *self.ax_left.get_legend_handles_labels(),
            loc="center",
            frameon=False,
            ncol=10,
            markerscale=10,
        )

        legend_frame_canvas = FigureCanvasTkAgg(
            legend_fig, master=legend_frame
        )
        legend_frame_canvas.get_tk_widget().grid(
            row=0, column=0, sticky="nsew"
        )
        legend_frame_canvas.get_tk_widget().columnconfigure(0, weight=1)
        legend_frame_canvas.get_tk_widget().rowconfigure(0, weight=1)
        legend_frame_canvas.draw()

    def toggle_classification(self):
        """Plot classification results on top of the left plot."""
        if self.show_classification:
            self.toggle_classification_button.config(
                text="Classification Results (Off)"
            )
            self.plot_classification()
        else:
            self.ax_left.clear()
            self.update_left_plot("Classification Off")
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
            array[i * 1024 : (i + 1) * 1024] for i in range(num_full_blocks)
        ]

        # Handle the remainder by padding with zeros if necessary
        if remainder > 0:
            padded_block = np.zeros((1024, 248))
            padded_block[:remainder] = array[num_full_blocks * 1024 :]
            blocks.append(padded_block)

        return np.array(blocks, dtype="float32"), remainder

    def classify(self):
        """Classify the CRISM cube using the CRISM Classifier model."""

        ort_session = onnxruntime.InferenceSession(
            (
                "vae_classifier_1024.onnx"
            ),
            providers=["CPUExecutionProvider"],
        )

        image = np.empty_like(self.visualizer.image.ratioed_image)
        image[:] = self.visualizer.image.ratioed_image
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

    def classification_subroutine(self):
        """Subroutine to classify the image in a separate thread."""
        self.display_loading_window("Classifying image...")

        thread = threading.Thread(target=self.classify, daemon=True)
        thread.start()

        def check_thread_state(thread):
            """Check the state of the thread"""
            if not thread.is_alive():
                self.add_classification_controls()
                self.classification_filter()
                self.toggle_classification()
                self.classification_button.config(state="disabled")
                self.loading_window.destroy()
                self.classification_flag = True
            else:
                self.root.after(1000, check_thread_state, thread)

        check_thread_state(thread)

    def load_image(self, filepath):
        """Load the CRISM image from the given filepath."""
        image = CRISMImage(filepath)
        image.ratio_image(self.crism_ml_dataset)
        summary_parameters = [*IMPLEMENTED_SUMMARY_PARAMETERS.keys()]

        for parameter in summary_parameters:
            image.calculate_summary_parameter(parameter)

        # Create visualizer and get image data
        self.visualizer = Visualiser(image)
        self.visualizer.get_image(60)
        self.visualizer.get_ratioed_spectrum((100, 100))

        # Store the image data
        self.image_array = self.visualizer.raw_image_copy
        self.ratioed_array = self.visualizer.ratioed_image_copy
        self.summary_parameters = summary_parameters

    def load_image_subroutine(self, filepath=None):
        # Open file dialog and get file path
        if not filepath:
            self.filepath = filedialog.askopenfilename(
                title="Select CRISM Image",
                filetypes=[("IMG files", "*.img"), ("All files", "*.*")],
            )
        if self.filepath:
            self.display_loading_window("Loading image...")

            def check_thread_state(thread):
                """Check the state of the thread"""
                if not thread.is_alive():
                    self.setup_left_plot()
                    self.setup_right_plot()
                    num_channels = self.image_array.shape[2]
                    self.setup_controls(self.summary_parameters, num_channels)

                    # Update the display
                    self.update_left_plot("Initialization")

                    # Reset right plot
                    self.ax_right.clear()
                    self.ax_right.set_title("Channel View")
                    self.canvas_right.draw()

                    self.file_window.destroy()
                    self.loading_window.destroy()
                else:
                    self.root.after(1000, check_thread_state, thread)

            try:
                thread = threading.Thread(
                    target=self.load_image, args=(self.filepath,)
                )
                thread.start()
                check_thread_state(thread)

            except Exception as e:
                self.update_file_loading_status(
                    f"Error loading image: {str(e)}"
                )

    def save_file(self):
        """Save the image and currently displayed classification results to
        file."""
        save_dir = filedialog.askdirectory(title="Select Save Directory")

        if save_dir:
            # Make an image layer of the classification results
            pred_im_filtered = np.full_like(self.pred_cls, 65535.0)
            for mineral, coords in self.pred_coords.items():
                if len(coords) > 0:
                    pred_im_filtered[coords[1], coords[0]] = mineral

            output_image = np.zeros((self.image_array.shape), dtype="float32")
            output_image[:] = self.image_array
            # encode any bad values
            output_image[np.isnan(output_image)] = 65535.0
            output_image[output_image < 0] = 65535.0
            output_image[output_image > 1000] = 65535.0

            output_image[:, :, -2] = pred_im_filtered
            output_image[:, :, -1] = self.pred_conf

            output_image = output_image.astype("float32")

            image_name = self.filepath.split("/")[-1].split(".")[0]
            self.visualizer.image.write_image(
                os.path.join(save_dir, image_name + ".hdr"),
                output_image,
                reverse_bands=True,
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = CAMEL(
        root,
    )
    root.mainloop()
