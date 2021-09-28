#!/usr/bin/python
# Gui imports----
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter.filedialog import askopenfile
from tkinter import ttk
import matplotlib.pyplot as plt
import StlProjection as STL

matplotlib.use("TkAgg")

# global variables
file_name = ""
graph_index = 0
graph_index_max = 0
x_arr = []
y_arr = []


class FoamSlicer(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="logo.ico")
        tk.Tk.wm_title(self, "Foam Slicer")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_columnconfigure(2, weight=2)

        self.frames = {}

        for F in ([StartPage]):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        import_label_text = tk.StringVar()
        label = tk.Label(self, textvariable=import_label_text, width=25, bd=2, bg="Gray")
        label.grid(column=0, row=0, pady=10,  padx=5, sticky="W")
        import_label_text.set("No file loaded")

        import_text = tk.StringVar()
        import_button = tk.Button(self, textvariable=import_text, command=lambda: import_stl_file(self, import_text,
                                                                                                  import_label_text,
                                                                                                  label))
        import_button.grid(column=1, row=0, sticky="W")
        import_text.set("Import")

        label_speed = tk.Label(self, text="Feed Rate")
        label_angle = tk.Label(self, text="Number of cuts")

        label_speed.grid(column=0, row=1)
        label_angle.grid(column=0, row=2)

        speed_text = tk.StringVar()
        angle_text = tk.StringVar()

        speed_text.set("200")
        angle_text.set("4")

        entry_speed = tk.Entry(self, width=6, textvariable=speed_text)
        entry_angle = tk.Entry(self, width=6, textvariable=angle_text)

        entry_speed.grid(column=1, row=1)
        entry_angle.grid(column=1, row=2)

        # label2 = tk.Label(self, text="Computed cuts")
        # label2.grid(column=2, row=0, padx=5)
        # figure2 = plt.Figure(figsize=(5, 4), dpi=100)
        # ax2 = figure2.add_subplot(111)
        # line2 = FigureCanvasTkAgg(figure2, self)
        # line2.get_tk_widget().grid(column=2, row=0, rowspan=6)
        # # ax2.plot([1, 0.5], [0, 1], color='red')
        # ax2.set_title('STL object outline')

        project_text = tk.StringVar()
        project_text.set("Find outline")
        project_button = tk.Button(self, textvariable=project_text, command=lambda: update_graph(self,
                                                                                                 project_text))
        project_button.grid(column=0, columnspan=2, row=3)

        place_holder_label = tk.Label(self, text=" ", width=70, height=30)
        place_holder_label.grid(row=1, column=2, rowspan=3, columnspan=2)

        right_button = tk.Button(self, text="Right", command=lambda: next_projection(self))
        right_button.grid(column=3, row=4)

        left_button = tk.Button(self, text="Left", command=lambda: previous_projection(self))
        left_button.grid(column=2, row=4)


def import_stl_file(self, button_text_object, label_text_object, label_object):
    file = askopenfile(parent=self, mode='rb', title="Select a file",
                       filetype=[("STL file", "*.stl")])
    if file:
        button_text_object.set("Select new")
        global file_name
        file_name = file.name
        backslash_location = file_name.rfind('/')
        label_text_object.set(file_name[backslash_location+1:]+" selected")
        label_object.config(bg="#B8FA95")


def next_projection(self):
    global graph_index
    graph_index += 1

    if graph_index >= graph_index_max:
        graph_index = graph_index_max-1

    draw_graph(self, graph_index)


def previous_projection(self):
    global graph_index
    graph_index -= 1

    if graph_index < 0:
        graph_index = 0

    draw_graph(self, graph_index)


def update_graph(self, project_text_object):
    project_text_object.set("Processing")

    global x_arr
    global y_arr
    global graph_index
    global graph_index_max

    graph_index = 0
    x_arr, y_arr = STL.project(file_name)
    graph_index_max = len(x_arr)

    draw_graph(self, graph_index)
    project_text_object.set("Find outline")


def draw_graph(self, proj_id):
    figure2 = plt.Figure(figsize=(5, 4), dpi=100)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, self)
    line2.get_tk_widget().grid(column=2, row=1, rowspan=3, columnspan=2)
    ax2.plot(x_arr[proj_id], y_arr[proj_id], color='red')
    ax2.set_title('STL object outline')


if __name__ == '__main__':
    app = FoamSlicer()
    app.mainloop()
