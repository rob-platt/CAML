import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)
from matplotlib.figure import Figure

from n2n4m.plot import Visualiser
from n2n4m.crism_image import CRISMImage
from n2n4m.summary_parameters import IMPLEMENTED_SUMMARY_PARAMETERS

RATIO_DATA = "/home/rob_platt/pixel_classifier/data/CRISM_ML"


class ChannelViewer:
    def __init__(self, root, filepath: str = None):
        """Initialize the Channel Viewer GUI.
        If image filepath passed, image loading prompt is skipped.
        """
        self.root = root
        self.root.title("Channel Viewer GUI")
        self.hover_paused = False
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

        self.fig_right = Figure(figsize=(8, 8))
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

        control_frame = tk.Frame(self.root)
        control_frame.grid(row=4, column=0, sticky="nsew")

        file_button = tk.Button(
            control_frame, text="Choose File", command=self.load_image
        )
        file_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        summary_params.append("Ratioed Image Channel")

        self.image_selection_dropdown = ttk.Combobox(
            control_frame,
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
            row=0, column=1, padx=5, pady=5, sticky="nsew"
        )

        self.channel_dropdown = ttk.Combobox(
            control_frame,
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
        self.channel_dropdown.grid(
            row=0, column=2, padx=5, pady=5, sticky="nsew"
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
        "/home/rob_platt/pixel_classifier/data/FRT00008F68_07_IF165L_TRR3.img",
    )
    root.mainloop()
