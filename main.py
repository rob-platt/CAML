import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
import numpy as np
import threading
import json
import os
import warnings

from n2n4m.plot import Visualiser
from n2n4m.crism_image import CRISMImage
from n2n4m.summary_parameters import IMPLEMENTED_SUMMARY_PARAMETERS
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS

from classification_plot import (
    convert_to_coords_filter_regions_by_conf,
    CLASS_NAMES,
    mineral_colours,
)
from CustomSlider import Slider
from Predict import Classifier

CONFIG_PATH = "CAML_config.json"
ICON_PATH = "CAML_icon.png"

# path update if running from PyInstaller
if os.path.exists("_internal"):
    ICON_PATH = os.path.join("_internal", "CAML_icon.png")


class CAML:
    def __init__(self, root):
        """
        Initialize the CAML (CRISM Analysis with Machine Learning) GUI.
        If image filepath passed, image loading prompt is skipped.
        """
        self.root = root  # root tkinter frame
        self.hover_paused: bool = False  # flag if right plot is fixed
        # flag if classification has been run
        self.classification_flag: bool = False
        # flag if class preds should be displayed
        self.show_classification: bool = False
        # flag if spectra reconstruction should be displayed
        self.show_reconstruction: bool = False
        # image array for the CRISM image
        self.image_array: np.ndarray = None
        # ratioed array for the CRISM image
        self.ratioed_array: np.ndarray = None
        # visualizer object for the CRISM image
        self.visualizer: Visualiser = None
        # reconstructed image array
        self.reconstructed_image: np.ndarray = None
        self.x_pos: int = 0  # x position of hovered pixel in left plot
        self.y_pos: int = 0  # y position of hovered pixel in left plot
        self.pred_cls: np.ndarray = None  # predicted class labels
        self.pred_conf: np.ndarray = None  # predicted class confidences
        self.pred_coords: dict = {}  # predicted class coordinates
        self.scatter = None  # scatter plot of classification results
        self.min_wavelength_idx: int = 0  # min wavelength index for spectrum
        self.max_wavelength_idx: int = 438  # max wavelength index for spectrum
        self.filepath: str = None  # path to the CRISM image file
        self.config: dict = {}  # configuration dictionary
        self.crism_ml_dataset: str = None  # path to CRISM_ML dataset
        self.false_colour_composite: np.ndarray = None  # false colour comp

        icon_img = tk.PhotoImage(file=ICON_PATH)
        self.root.iconphoto(True, icon_img)
        self.root.title("CAML")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(4, weight=0)

        # Create plot frame
        self.plot_frame = tk.Frame(root, bg="white")
        self.plot_frame.grid(row=0, column=0, sticky="nesw")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        if self.load_config():
            self.crism_ml_dataset = self.config["crism_ml_dataset_dir"]
            if not self.check_bland_mat_file():
                self.crism_ml_dataset = None
                self.prompt_crism_ml_file_selection()
            else:
                self.prompt_image_file_selection()
        else:
            self.prompt_crism_ml_file_selection()

    def load_config(self):
        """Check if the config file exists, if yes then load it."""
        if not os.path.exists(CONFIG_PATH):
            return False
        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)
        return True

    def write_config(self):
        """Write the current configuration to the config file."""
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.config, f)

    def prompt_crism_ml_file_selection(self):
        """
        Open a separate window to prompt CRISM_ML dataset directory
        selection on launch.
        """
        self.crism_ml_dataset_window = tk.Toplevel(self.root)
        self.crism_ml_dataset_window.title(
            "Select CRISM_ML Dataset File (CRISM_bland_unratioed.mat)"
        )
        self.crism_ml_dataset_window.geometry("300x100")

        self.crism_ml_dataset_window.transient(root)
        self.crism_ml_dataset_window.grab_set()

        self.crism_ml_dataset_window.protocol(
            "WM_DELETE_WINDOW",
            lambda: self.close_window(self.crism_ml_dataset_window),
        )

        self.crism_ml_dataset_label = tk.Label(
            self.crism_ml_dataset_window,
            text="Please select the CRISM_ML dataset file.",
        )
        self.crism_ml_dataset_label.place(relx=0.5, rely=0.3, anchor="center")

        crism_ml_dataset_button = tk.Button(
            self.crism_ml_dataset_window,
            text="Choose Dataset Directory",
            command=self.crism_ml_dataset_selection,
        )
        crism_ml_dataset_button.place(relx=0.5, rely=0.5, anchor="center")

    def prompt_image_file_selection(self):
        """
        Open a separate window to prompt image file selection
        after CRISM_ML dataset selection.
        """
        self.file_window = tk.Toplevel(self.root)
        self.file_window.title("Select an Image File")
        self.file_window.geometry("300x100")

        self.file_window.transient(root)
        self.file_window.grab_set()

        self.file_window.protocol(
            "WM_DELETE_WINDOW", lambda: self.close_window(self.file_window)
        )

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

    def check_bland_mat_file(self):
        """
        Check to see if the selected directory exists and has the training
        data in it. The name must exactly match for the CRISM_ML library to use
        it.
        """
        if not os.path.exists(self.crism_ml_dataset):
            return False
        if not os.path.exists(
            os.path.join(self.crism_ml_dataset, "CRISM_bland_unratioed.mat")
        ):
            return False
        return True

    def crism_ml_dataset_selection(self):
        """Open file dialog to select the CRISM_ML dataset."""
        self.crism_ml_dataset = filedialog.askopenfilename(
            title="Select CRISM_ML Dataset File",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
        )
        if self.crism_ml_dataset:
            self.crism_ml_dataset = os.path.dirname(self.crism_ml_dataset)
            if not self.check_bland_mat_file():
                self.crism_ml_dataset = None
                self.crism_ml_dataset_label.config(
                    text="Invalid file selected. Please try again."
                )
            else:
                self.crism_ml_dataset_window.destroy()
                self.config["crism_ml_dataset_dir"] = self.crism_ml_dataset
                self.write_config()
                self.prompt_image_file_selection()

    def update_file_loading_status(self, status_message: str):
        """Update file loading and image processing status in the GUI."""
        self.file_window_label.config(text=status_message)
        self.file_window.update_idletasks()

    def setup_left_plot(self):
        """
        Set up the image plot for the GUI. Takes the left hand side of the
        plot_frame, row 0 column 0. Within left_frame, plot image canvas and
        matplotlib navigation toolbar. Hover and on-click controls determine
        spectrum plot.
        """
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
        toolbar_frame = tk.Frame(self.plot_frame)
        toolbar_frame.grid(
            row=2, column=0, sticky="w"
        )  # Place toolbar below plot
        toolbar = NavigationToolbar2Tk(self.canvas_left, toolbar_frame)
        toolbar.update()

        self.canvas_left.mpl_connect(
            "motion_notify_event", self.update_right_plot
        )
        self.canvas_left.mpl_connect("button_press_event", self.toggle_hover)

    def setup_right_plot(self):
        """
        Set up the spectrum plot for the GUI. Takes the right hand side of
        the plot_frame, row 0 column 1. Within right_frame, plot spectrum
        canvas.
        """
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
        - Spectrum range slider
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

        # Dropdown menu for image selection
        summary_params.insert(0, "Image Band")
        summary_params.insert(0, "False Colour Composite")
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
        self.image_selection_dropdown.set("False Colour Composite")
        self.image_selection_dropdown.grid(
            row=1, column=1, padx=5, sticky="nsew"
        )
        image_selection_label = tk.Label(
            self.control_frame, text="Image Layer:"
        )
        image_selection_label.grid(row=0, column=1, padx=5)  # sticky="nsew")

        # Dropdown menu for channel selection
        self.channel_dropdown = ttk.Combobox(
            self.control_frame,
            # We make the assumption that any missing channels are at the end
            values=ALL_WAVELENGTHS[:num_channels],
            style="Custom.TCombobox",
        )
        self.channel_dropdown.bind(
            "<FocusIn>", lambda e: on_popdown_show(self.channel_dropdown)
        )
        self.channel_dropdown.set(ALL_WAVELENGTHS[60])
        self.channel_dropdown.bind(
            "<<ComboboxSelected>>", self.update_left_plot
        )
        self.channel_dropdown.grid(row=1, column=2, padx=5)  # sticky="nsew")
        channel_label = tk.Label(
            self.control_frame, text="Spectral Wavelength (μm):"
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
        spectrum_slider_label.grid(row=0, column=17, padx=5, sticky="nsew")

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
            row=1, column=17, columnspan=2, padx=5, sticky="e"
        )
        # Add weight to empty column to ensure spectral slider always on right
        self.control_frame.columnconfigure(15, weight=1)

    def update_left_plot(self, event):
        """
        Update the left plot based on the selected image band or summary
        parameter.
        Special event options:
        - "Initialization": Set the default channel to 1.394890μm
        - "Plot Classification": Plot the classification results on the image
        """
        # If want to update base image and then overlay classification
        # do this to ensure classification plotted on top
        if self.show_classification and event != "PlotClassification":
            self.plot_classification()
            return
        image_selection = self.image_selection_dropdown.get()
        if image_selection == "Image Band":
            self.channel_dropdown.state(["!disabled"])
            selected_channel = self.channel_dropdown.get()
            channel_idx = ALL_WAVELENGTHS.index(float(selected_channel))
            self.ax_left.clear()
            self.ax_left.imshow(
                self.image_array[:, :, channel_idx], cmap="viridis"
            )
            self.ax_left.set_title(f"{selected_channel}μm Band")
        elif image_selection in IMPLEMENTED_SUMMARY_PARAMETERS:
            self.ax_left.clear()
            self.ax_left.imshow(
                self.visualizer.get_summary_parameter(image_selection),
                cmap="viridis",
            )
            self.ax_left.set_title(f"{image_selection} Summary Parameter")
            self.channel_dropdown.state(["disabled"])
        elif image_selection == "False Colour Composite":
            self.ax_left.clear()
            self.ax_left.imshow(self.false_colour_composite)
            self.ax_left.set_title("False Colour Composite")
            self.channel_dropdown.state(["disabled"])
        self.canvas_left.draw()

    def create_false_colour_composite(self):
        """
        Create a false colour composite image from the CRISM image.
        """
        r_channel_idx = ALL_WAVELENGTHS.index(2.529510)
        g_channel_idx = ALL_WAVELENGTHS.index(1.506610)
        b_channel_idx = ALL_WAVELENGTHS.index(1.079960)

        r_channel = self.image_array[:, :, r_channel_idx]
        g_channel = self.image_array[:, :, g_channel_idx]
        b_channel = self.image_array[:, :, b_channel_idx]

        # 99th percentile clip
        r_channel[r_channel > np.nanpercentile(r_channel, 99)] = (
            np.nanpercentile(r_channel, 99)
        )
        g_channel[g_channel > np.nanpercentile(g_channel, 99)] = (
            np.nanpercentile(g_channel, 99)
        )
        b_channel[b_channel > np.nanpercentile(b_channel, 99)] = (
            np.nanpercentile(b_channel, 99)
        )

        # scale the channels between 0 and 255
        r_channel = (r_channel - np.nanmin(r_channel)) / (
            np.nanmax(r_channel) - np.nanmin(r_channel)
        )
        g_channel = (g_channel - np.nanmin(g_channel)) / (
            np.nanmax(g_channel) - np.nanmin(g_channel)
        )
        b_channel = (b_channel - np.nanmin(b_channel)) / (
            np.nanmax(b_channel) - np.nanmin(b_channel)
        )

        # ignore casting runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        r_channel = (r_channel * 255).astype(np.uint8)
        g_channel = (g_channel * 255).astype(np.uint8)
        b_channel = (b_channel * 255).astype(np.uint8)
        warnings.resetwarnings()
        self.false_colour_composite = np.stack(
            [r_channel, g_channel, b_channel], axis=-1
        )

    def toggle_hover(self, event):
        """
        If the left plot is left clicked, freeze the right (spectrum) plot on
        that pixel. Unfreeze when left click again.
        """
        if event.inaxes == self.ax_left and event.button == 1:  # Left click
            self.hover_paused = not self.hover_paused
            status = "Paused" if self.hover_paused else "Active"
            right_plot_title = self.ax_right.get_title()
            self.ax_right.set_title(f"{status}\n{right_plot_title}")
            self.x_pos, self.y_pos = int(event.xdata), int(event.ydata)
            self.canvas_right.draw()

    def update_right_plot(self, event):
        """
        Plot the spectrum of the pixel currently hovered over in the left plot.
        Disabled if hover is paused. If classification has been run, plot the
        line in the colour of the predicted class, and display the class name
        and confidence in the title.
        """

        def get_wavelength_index(wavelength: float):
            """
            Get the index in PLEBANI_WAVELENGTHS for a given wavelength. If
            the wavelength is not in the list, return the closest shorter
            wavelength index. If a shorter wavelength is not available, return
            0.

            Reconstructions use the 248 bands from PLEBANI_WAVELENGTHS, but the
            we are plotting using the 438 bands from ALL_WAVELENGTHS. To ensure
            the plot works wrt the spectral slider, need to use this.
            """
            try:
                return PLEBANI_WAVELENGTHS.index(wavelength)
            except ValueError:
                all_idx = ALL_WAVELENGTHS.index(wavelength)
                for i in range(all_idx, -1, -1):
                    try:
                        return PLEBANI_WAVELENGTHS.index(ALL_WAVELENGTHS[i])
                    except ValueError:
                        pass
                return 0

        # Catch changes to wavelength range slider
        if isinstance(event, list):
            event = "PlotRangeUpdate"

        if not self.hover_paused:
            if event == "Initialization":
                self.x_pos, self.y_pos = 100, 100
            elif (
                event == "PlotRangeUpdate"
                or event == "ReconstructionOn"
                or event == "ReconstructionOff"
            ):
                pass
            elif event.inaxes == self.ax_left:
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

        if (
            self.hover_paused
            or event == "Initialization"
            or event == "PlotRangeUpdate"
            or event == "ReconstructionOn"
            or event == "ReconstructionOff"
            or event.inaxes == self.ax_left
        ):
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
                label="Spectrum",
            )

            # Catch if plotted range is outside of reconstruction band range
            if self.classification_flag and self.min_wavelength_idx > 248:
                self.show_reconstruction = False
            elif self.classification_flag:
                self.show_reconstruction = True

            # Plot the reconstructed spectrum if the flag is set
            if self.show_reconstruction:
                min_recon_idx = get_wavelength_index(
                    ALL_WAVELENGTHS[self.min_wavelength_idx]
                )
                if self.max_wavelength_idx > 248:
                    max_recon_idx = 248
                else:
                    max_recon_idx = get_wavelength_index(
                        ALL_WAVELENGTHS[self.max_wavelength_idx]
                    )
                self.ax_right.plot(
                    PLEBANI_WAVELENGTHS[min_recon_idx:max_recon_idx],
                    self.reconstructed_image[
                        self.y_pos,
                        self.x_pos,
                        min_recon_idx:max_recon_idx,
                    ]
                    - 0.02,  # offset to separate the two lines
                    color=line_col,
                    alpha=0.5,
                    label="Reconstructed Spectrum",
                )
                self.ax_right.legend(loc="lower center")

            self.ax_right.set_xlabel("Wavelength (μm)")
            self.ax_right.set_ylabel("Ratioed I/F")
            if self.classification_flag:
                self.ax_right.set_title(
                    (
                        f"Pixel ({self.x_pos}, {self.y_pos}) Spectrum Plot\n"
                        f"Class: {cls_name}, Confidence: {conf*100:.2f}%"
                    )
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
        """
        Add controls for classification results to control panel.
        Instantiates the following widgets:
        - Button to toggle classification results
        - Slider for confidence threshold
        - Slider for connected components threshold
        - Button to run filtering
        - Button to save the image
        """
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

        # Add toggle for displaying spectra reconstruction
        self.toggle_spectra_button = tk.Button(
            self.control_frame,
            text="Spectra Reconstruction (Off)",
            command=self.toggle_spectra_reconstruction,
            wraplength=100,
        )
        self.toggle_spectra_button.grid(
            row=0, column=16, rowspan=2, padx=5, sticky="ens"
        )

    def classification_filter(self):
        """
        Filter classification results based on confidence threshold,
        and connected components. Then replot the predictions.
        """
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
        """
        Plot classification predictions as a scatter plot on top of the left
        plot. If a custom colour is defined for the mineral, use it. Otherwise,
        use the default colours. Clears left_plot and replots the current base
        image to ensure predictions aren't plotted on top of each other. Calls
        plot_classification_legend to add a legend to the plot frame.
        """
        self.ax_left.clear()
        self.update_left_plot("PlotClassification")
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
        self.canvas_left.draw()
        self.plot_classification_legend()

    def plot_classification_legend(self):
        """Plot legend for class predictions across base of plot frame."""
        # Add an additional row to the plot frame for the legend to go in
        self.plot_frame.rowconfigure(1, weight=1, minsize=120)
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
        self.show_classification = not self.show_classification
        if self.show_classification:
            self.toggle_classification_button.config(
                text="Classification Results (Off)"
            )
            self.plot_classification()
        else:
            self.ax_left.clear()
            self.update_left_plot("ClassificationOff")
            self.toggle_classification_button.config(
                text="Classification Results (On)"
            )
        self.canvas_left.draw()

    def toggle_spectra_reconstruction(self):
        """Toggle the display of the spectra reconstruction."""
        self.show_reconstruction = not self.show_reconstruction
        if self.show_reconstruction:
            self.toggle_spectra_button.config(
                text="Spectra Reconstruction (Off)"
            )
            self.update_right_plot("ReconstructionOn")
        else:
            self.ax_right.clear()
            self.update_right_plot("ReconstructionOff")
            self.toggle_spectra_button.config(
                text="Spectra Reconstruction (On)"
            )
        self.canvas_right.draw()

    def classify(self):
        """Classify the CRISM cube using the CRISM Classifier model."""
        model = Classifier()
        self.reconstructed_image, self.pred_cls, self.pred_conf = (
            model.predict(self.ratioed_array)
        )

    def classification_subroutine(self):
        """
        Subroutine to classify the image in a separate thread.
        Allows the user to continue using the GUI while the classification
        is running, and a loading window is displayed. After classification is
        complete, initialize the classification controls, and plot the
        predictions.
        """
        self.display_loading_window("Classifying image...")

        thread = threading.Thread(target=self.classify, daemon=True)
        thread.start()

        def check_thread_state(thread):
            """Check the state of the thread"""
            if not thread.is_alive():
                self.show_classification = True
                self.show_reconstruction = True
                self.add_classification_controls()
                self.classification_filter()
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
        self.create_false_colour_composite()

    def load_image_subroutine(self, filepath=None):
        """
        Subroutine to load the image from the given filepath.
        If no filepath is given, open a file dialog to select the image.
        After loading the image, set up the left and right plots, and the
        control panel.
        """
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
                    self.update_right_plot("Initialization")

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
        """
        Save the image and currently displayed classification results to
        file.
        """
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

    def close_window(self, window):
        """Close the window and exit the program."""
        window.destroy()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CAML(
        root,
    )
    root.mainloop()
    input("Press Enter to exit...")
